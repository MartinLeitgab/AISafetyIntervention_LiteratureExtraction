"""
KG-Judge: Knowledge Graph Validation and Improvement System
A precise and rigorous auditor for knowledge graphs extracted by LLMs
"""

import asyncio
import hashlib
import json
import os
import shutil
import signal
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, NotRequired, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from fire import Fire
from openai import AsyncOpenAI
from openai.types.batch import Batch
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import ValidationError

from intervention_graph_creation.src.local_graph_extraction.core.node import GraphNode
from intervention_graph_creation.src.local_graph_extraction.core.paper_schema import (
    PaperSchema,
)
from intervention_graph_creation.src.local_graph_extraction.llm_judge.schema import (
    AddNodeFix,
    DeleteFix,
    FixProps,
    GPT_Assessment,
    MergeFix,
    ProposedFixes,
    ValidationReport,
    create_validation_prompt,
)
from intervention_graph_creation.src.local_graph_extraction.llm_judge.utilities import (
    BatchOutput,
    BatchResult,
    CompletionsRequest,
    DataSource,
    EverythingInTheBatchHasAnError,
    JudgeBatch,
    JudgeBatchResult,
    JudgeErrorCode,
    JudgeRequest,
    unknown_judge_request,
    upload_and_create_batch,
)

# Loads the OpenAI API key from .env file OPENAI_API_KEY
load_dotenv()

URL_To_Text_Map = Dict[str, Dict[str, str]]


class ValidationIssue(TypedDict):
    severity: Literal["BLOCKER", "MAJOR", "MINOR", "STYLE"]
    issue: str
    where: str
    suggestion: str


class LocalAssessment(TypedDict):
    local_validation: bool
    issues: List[ValidationIssue]


class OpenAIResponseResponse(TypedDict):
    status_code: int
    request_id: str
    body: Dict[str, Any]
    """Use ChatCompletion.construct on body to get the ChatCompletion object."""


class OpenAIResponse(TypedDict):
    custom_id: str
    id: str
    response: OpenAIResponseResponse


class FinalGraphMetaError(TypedDict):
    version: Literal["1.0"]
    error: str


class FinalGraphMeta(TypedDict):
    version: Literal["1.0"]
    source_hash: str
    validation_timestamp: Optional[str]


class Final_Knowledge_Graph(TypedDict):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    meta: FinalGraphMeta | FinalGraphMetaError


class JudgeReport(TypedDict):
    final_graph: Final_Knowledge_Graph
    decision: Dict[str, Any]
    validation_report: Dict[str, Any]
    proposed_fixes: Dict[str, Any]
    rationale_record: Dict[str, Any]
    url: str
    paper_id: str
    ard_file_source: str
    errors: Optional[List[str]]


class AssessmentSuccess(TypedDict):
    type: Literal["success"]
    gpt_assessment: GPT_Assessment
    custom_id: str


class AssessmentFailure(TypedDict):
    type: Literal["failure"]
    error: str
    raw_response: str
    custom_id: NotRequired[str]
    error_code: JudgeErrorCode


GPT_Assessment_Result = AssessmentSuccess | AssessmentFailure


@dataclass
class JudgeInput:
    original_text: str
    kg_output: PaperSchema
    data_source: DataSource


class KGJudge:
    def __init__(self, output_dir: Path):
        """Initialize KG-Judge with OpenAI API client."""
        self.client = AsyncOpenAI()
        self.total_tokens_used = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.requests_with_errors: Dict[str, List[str]] = {}
        self.temp_dir = Path(tempfile.gettempdir()) / "kg_judge_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.running_batch_ids: List[str] = []
        self.output_dir = output_dir
        self.cancelled = False
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, self._cancel)

    def _cancel(self):
        print("Got your ctrl+c, cancelling..., please wait for cleanup...")
        self.cancelled = True

    async def judge_knowledge_graph_batch(
        self,
        inputs: List[JudgeInput],
        how_many_batches_in_flight_at_once: int,
        batch_size: int,
    ):
        """
        Creates batch files for OpenAI API batch processing.
        Args:
            original_texts: List of source texts
            kg_outputs: List of knowledge graph outputs from LLM
            original_prompts: List of original prompts used for KG extraction
            batch_size: The maximum number of requests per batch file.
        Returns:
            List of JudgeReports for each input
        """

        # order is not guaranteed, so let's create a map
        # of custom_id to batch request
        custom_id_to_request: Dict[str, JudgeRequest] = {}
        all_requests: List[JudgeRequest] = []
        for i, judge_input in enumerate(inputs):
            validation_prompt = create_validation_prompt(
                judge_input.original_text, judge_input.kg_output
            )
            custom_id = f"request-{i}"
            request: CompletionsRequest = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5-nano",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are KG-Judge, a precise and rigorous auditor for knowledge graphs. Always return valid JSON in the exact format requested.",
                        },
                        {"role": "user", "content": validation_prompt},
                    ],
                    # "temperature": 0.1,
                    "temperature": 1.0,
                    # "max_tokens": 4000,
                    "max_completion_tokens": 16_000,
                    "response_format": {"type": "json_object"},
                },
            }
            judge_request = JudgeRequest(
                request=request,
                original_text=judge_input.original_text,
                kg_output=judge_input.kg_output,
                data_source=judge_input.data_source,
                custom_id=custom_id,
            )
            all_requests.append(judge_request)
            custom_id_to_request[custom_id] = judge_request
        batch_files: List[JudgeBatch] = []
        for i, batch_start in enumerate(range(0, len(all_requests), batch_size)):
            batch = all_requests[batch_start : batch_start + batch_size]
            batch_file_path = self.temp_dir / f"batch_requests_{i + 1}.jsonl"

            with open(batch_file_path, "w") as f:
                for req in batch:
                    f.write(json.dumps(req.request) + "\n")
            batch_files.append(
                JudgeBatch(file_path=str(batch_file_path), requests=batch)
            )
        for meta_batch_index in range(
            0, len(batch_files), how_many_batches_in_flight_at_once
        ):
            slice_of_batches = batch_files[
                meta_batch_index : meta_batch_index + how_many_batches_in_flight_at_once
            ]
            if self.cancelled:
                for batch in slice_of_batches:
                    self.save_results(
                        self._error_result(
                            "Cancelled", "Cancelled", "batch_cancelled", batch.requests
                        )
                    )
                continue

            # Immediately upload each batch after it is written
            batch_results = await asyncio.gather(
                *(
                    upload_and_create_batch(self.client, batch)
                    for batch in slice_of_batches
                )
            )
            # monitor and retrieve the resultts for each batch
            # Monitor the status of each submitted batch using the OpenAI API until completion
            # This does it sequentially, but that's fine for now, because the time for this
            # processing is significantly shorter than the time waiting for the batch to complete
            for batch_result in batch_results:
                result_content = await self._poll_for_batch_completion(batch_result)
                if result_content.type == "error":
                    if self.cancelled:
                        break
                    self.save_results(
                        self._error_result(
                            result_content.message,
                            result_content.raw,
                            result_content.error_code,
                            batch_result.batch.requests,
                        ),
                    )
                    continue

                if result_content.error_content is not None:
                    for line in result_content.error_content.splitlines():
                        response_text = line.strip()
                        if not response_text:
                            # Skip empty lines
                            continue
                        self._parse_judge_error(custom_id_to_request, response_text)

                if result_content.content is not None:
                    # Parse the results (each line is a JSON object)
                    lines = result_content.content.splitlines()
                    for line in lines:
                        response_text = line.strip()
                        if not response_text:
                            # Skip empty lines
                            continue
                        gpt_assessment = self._parse_judge_response(response_text)
                        if gpt_assessment["type"] == "failure":
                            self.save_results(
                                self._error_result(
                                    gpt_assessment["error"],
                                    gpt_assessment["raw_response"],
                                    gpt_assessment["error_code"],
                                    [
                                        custom_id_to_request.get(
                                            gpt_assessment.get("custom_id", "unknown"),
                                            unknown_judge_request("unknown"),
                                        )
                                    ],
                                ),
                            )
                            continue
                        judge_request = custom_id_to_request.get(
                            gpt_assessment["custom_id"]
                        )
                        if judge_request is None:
                            # this should not happen
                            self.save_results(
                                self._error_result(
                                    f"Unknown custom_id in GPT response {gpt_assessment['custom_id']}",
                                    "",
                                    "unknown_custom_id",
                                    [
                                        unknown_judge_request(
                                            gpt_assessment["custom_id"]
                                        )
                                    ],
                                ),
                            )
                            continue

                        # Perform local validation checks
                        local_validation = self._perform_local_validation(
                            judge_request.kg_output
                        )
                        combined_report = self._combine_validations(
                            gpt_assessment,
                            local_validation,
                            judge_request.kg_output,
                            judge_request.original_text,
                            judge_request.data_source,
                        )
                        self.save_results([combined_report])
            if self.cancelled:
                print("Cancelling all running batches...")
                cancelled_count = 0
                for batch_result in batch_results:
                    # Attempt to cancel running batches
                    try:
                        await self.client.batches.cancel(batch_result.batch_id)
                        cancelled_count += 1
                    except Exception as e:
                        print(
                            f"Could not cancel batch {batch_result.batch_id}: {str(e)}"
                        )
                print(f"Cancelled {cancelled_count}/{len(batch_results)} batches.")
                for batch_result in batch_results:
                    self.save_results(
                        self._error_result(
                            "Cancelled",
                            "Cancelled",
                            "batch_cancelled",
                            batch_result.batch.requests,
                        )
                    )

    def save_results(self, results: List[JudgeReport]):
        """Save the judge reports to JSON files in the specified output directory."""
        for report in results:
            errors = report.get("errors")
            file_name = f"{report['ard_file_source']}__{report['paper_id']}"
            if errors is not None:
                self.requests_with_errors[file_name] = errors
            output_file = self.output_dir / f"{file_name}.json"
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

    def _parse_judge_error(
        self, custom_id_to_request: Dict[str, JudgeRequest], line: str
    ):
        try:
            line_data: OpenAIResponse = json.loads(line)
            custom_id = line_data["custom_id"]
            self.save_results(
                self._error_result(
                    f"Unknown error in gpt response, custom_id: {custom_id}. Got this error:",
                    line,
                    "unknown_custom_id",
                    [
                        custom_id_to_request.get(
                            custom_id, unknown_judge_request(custom_id)
                        )
                    ],
                ),
            )
        except Exception as e:
            self.save_results(
                self._error_result(
                    f"Could not parse error line from batch error file: {str(e)}. But this is the error that caused this:",
                    line,
                    "parse_or_validate_error",
                    [unknown_judge_request("unknown")],
                ),
            )

    async def _poll_for_batch_completion(
        self, batch_result: JudgeBatchResult
    ) -> BatchResult:
        if batch_result.is_debug:
            with open("extraction_validator/debug_batch_results/1.jsonl", "r") as f:
                return BatchOutput(content=f.read(), type="content")

        while True:
            if self.cancelled:
                return EverythingInTheBatchHasAnError(
                    message="Batch processing was cancelled.",
                    raw="",
                    error_code="batch_cancelled",
                )
            batch = await self.client.batches.retrieve(batch_result.batch_id)
            print(f"Batch {batch_result.batch_id}: status = {batch.status}")
            if batch.status in ["completed", "failed", "expired", "cancelled"]:
                break
            await asyncio.sleep(15)
        # if completed -> retrieve results
        if batch.status != "completed":
            return EverythingInTheBatchHasAnError(
                message=f"Batch did not complete successfully, final status: {batch.status}",
                raw=self._get_open_batch_errors(batch),
                error_code="batch_not_completed",
            )

        output_file_id = batch.output_file_id

        error_content = None
        if batch.error_file_id is not None:
            error_content_result = await self.client.files.content(batch.error_file_id)
            error_content = error_content_result.text

        if output_file_id is None:
            if error_content is not None:
                return BatchOutput(
                    error_content=error_content,
                    content=None,
                )
            return EverythingInTheBatchHasAnError(
                message="Batch completed but no output_file_id found",
                raw=self._get_open_batch_errors(batch),
                error_code="missing_output",
            )

        # downloading results, it will be a .jsonl file
        result_content = await self.client.files.content(output_file_id)
        content = result_content.text

        if os.environ.get("DEBUG_SAVE_BATCH_RESULTS") == "1":
            # Save a copy of the batch results for debugging
            debug_dir = Path("extraction_validator/debug_batch_results")
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = debug_dir / f"batch_{batch_result.batch_id}_{timestamp}.jsonl"
            with open(debug_file, "w") as f:
                f.write(content)

        return BatchOutput(content=content, error_content=error_content)

    def _get_open_batch_errors(self, batch: Batch) -> str:
        """Extract error messages from an OpenAI Batch object."""
        error_messages: List[str] = []
        if batch.errors and batch.errors.data:
            for i, error in enumerate(batch.errors.data):
                if error.message:
                    error_messages.append(f"Error {i}: {error.message}")
        return "; ".join(error_messages)

    def _error_result(
        self,
        error: str,
        raw_response: str,
        error_code: JudgeErrorCode,
        batch_requests: List[JudgeRequest],
    ) -> List[JudgeReport]:
        reports: List[JudgeReport] = []
        for request in batch_requests:
            local_validation = self._perform_local_validation(request.kg_output)
            combined_report = self._combine_validations(
                {
                    "type": "failure",
                    "error": error,
                    "raw_response": raw_response,
                    "error_code": error_code,
                    "custom_id": request.custom_id,
                },
                local_validation,
                request.kg_output,
                request.original_text,
                request.data_source,
            )
            reports.append(combined_report)
        return reports

    def _parse_judge_response(self, response_text: str) -> GPT_Assessment_Result:
        try:
            raw: OpenAIResponse = json.loads(response_text)
        except json.JSONDecodeError as e:
            return {
                "type": "failure",
                "error": "Could not extract JSON from GPT response, error: " + str(e),
                "raw_response": response_text,
                "error_code": "parse_or_validate_error",
            }
        custom_id = raw["custom_id"]
        if raw["response"]["status_code"] != 200:
            return {
                "type": "failure",
                "error": f"OpenAI API error: {raw['response']['status_code']}",
                "raw_response": response_text,
                "error_code": "http_error",
                "custom_id": custom_id,
            }
        body = raw["response"]["body"]
        try:
            chat_completion = ChatCompletion.model_validate(body)

            if chat_completion.usage:
                self.total_tokens_used += chat_completion.usage.total_tokens
                self.total_prompt_tokens += chat_completion.usage.prompt_tokens
                self.total_completion_tokens += chat_completion.usage.completion_tokens

            choice = chat_completion.choices[0]
            if choice.finish_reason != "stop":
                return {
                    "type": "failure",
                    "error": f"GPT did not finish properly, finish_reason: {choice.finish_reason}",
                    "raw_response": response_text,
                    "error_code": choice.finish_reason,
                    "custom_id": custom_id,
                }

            content = choice.message.content

            if content is None:
                return {
                    "type": "failure",
                    "error": "No content in GPT response",
                    "raw_response": response_text,
                    "error_code": "empty_response",
                    "custom_id": custom_id,
                }

            result = GPT_Assessment.model_validate_json(content)
            return {
                "type": "success",
                "gpt_assessment": result,
                "custom_id": custom_id,
            }
        except ValidationError as e:
            return {
                "type": "failure",
                "error": "Could not validate GPT response into GPT_Assessment model: "
                + str(e),
                "raw_response": response_text,
                "error_code": "parse_or_validate_error",
                "custom_id": custom_id,
            }

    def _perform_local_validation(self, kg: PaperSchema) -> LocalAssessment:
        """Perform local validation checks as backup/supplement."""

        issues: List[ValidationIssue] = []
        # Removed basic checks already covered by the pydantic model validation
        # Only put checks here that are not covered by the model validation

        # Node validation
        node_names = set[str]()
        for node in kg.nodes:
            node_names.add(node.name)

        # Edge validation
        for edge_i, edge in enumerate(kg.edges):
            if edge.source_node not in node_names:
                issues.append(
                    {
                        "severity": "BLOCKER",
                        "issue": f"Edge references non-existent source node: {edge.source_node}",
                        "where": f"edges[{edge_i}].source_node",
                        "suggestion": f"Create node '{edge.source_node}' or fix reference",
                    }
                )

            if edge.target_node not in node_names:
                issues.append(
                    {
                        "severity": "BLOCKER",
                        "issue": f"Edge references non-existent target node: {edge.target_node}",
                        "where": f"edges[{edge_i}].target_node",
                        "suggestion": f"Create node '{edge.target_node}' or fix reference",
                    }
                )
        return {"local_validation": True, "issues": issues}

    def _combine_validations(
        self,
        gpt_assessment_result: GPT_Assessment_Result,
        local_validation: LocalAssessment,
        kg: PaperSchema,
        original_text: str,
        data_source: DataSource,
    ) -> JudgeReport:
        """Combine GPT's assessment with local validation results."""

        if gpt_assessment_result["type"] == "failure":
            # Fallback to local validation if GPT fails
            return self._create_fallback_report(
                local_validation,
                gpt_assessment_result["error"],
                gpt_assessment_result["raw_response"],
                gpt_assessment_result["error_code"],
                kg,
                data_source,
            )

        gpt_assessment = gpt_assessment_result["gpt_assessment"]

        # Generate proposed fixes based on issues found
        proposed_fixes = self._generate_proposed_fixes(gpt_assessment.validation_report)

        # Apply fixes to create final graph
        final_graph = self._apply_fixes_to_graph(kg, proposed_fixes, original_text)

        # Create complete report
        return JudgeReport(
            decision=gpt_assessment.decision.model_dump()
            if gpt_assessment.decision
            else {},
            validation_report=gpt_assessment.validation_report.model_dump()
            if gpt_assessment.validation_report
            else {},
            proposed_fixes=(
                gpt_assessment.proposed_fixes.model_dump()
                if gpt_assessment.proposed_fixes
                else {}
            ),
            final_graph=final_graph,
            rationale_record=gpt_assessment.rationale_record.model_dump()
            if gpt_assessment.rationale_record
            else {},
            url=data_source.url,
            paper_id=data_source.paper_id,
            ard_file_source=data_source.ard_file_source,
            errors=None,
        )

    def _generate_proposed_fixes(
        self, validation_report: Optional[ValidationReport]
    ) -> ProposedFixes:
        """Generate proposed fixes based on validation issues."""

        # fixes = ProposedFixes(add_nodes=[], merges=[], deletions=[])

        if validation_report is None:
            return ProposedFixes(add_nodes=[], merges=[], deletions=[])
        add_nodes: List[AddNodeFix] = []
        merges: List[MergeFix] = []
        deletions: List[DeleteFix] = []

        # Process schema issues
        for issue in validation_report.schema_check or []:
            if issue.issue is not None and "missing" in issue.issue.lower():
                if issue.where is not None and "node" in issue.where:
                    add_nodes.append(
                        AddNodeFix(
                            id=f"generated_node_{len(add_nodes)}",
                            type="concept",
                            name="Generated Node",
                            props=FixProps(
                                stable_key=hashlib.md5(
                                    f"gen_node_{len(add_nodes)}".encode()
                                ).hexdigest()[:8]
                            ),
                        )
                    )

        # Process referential issues
        for issue in validation_report.referential_check or []:
            if issue.severity == "BLOCKER":
                for node_id in issue.ids or []:
                    add_nodes.append(
                        AddNodeFix(
                            id=node_id,
                            type="concept",
                            name=node_id,
                            props=FixProps(
                                stable_key=hashlib.md5(node_id.encode()).hexdigest()[:8]
                            ),
                        )
                    )

        # Process duplicates
        for duplicate in validation_report.duplicates or []:
            ids = duplicate.ids or []
            if duplicate.kind == "node" and len(ids) > 1:
                target_id = ids[0]
                absorbed_ids = ids[1:]
                merges.append(
                    MergeFix(
                        target_id=target_id,
                        absorbed_ids=absorbed_ids,
                        retargeted_edge_ids=[],
                    )
                )

        # Process orphans
        for orphan in validation_report.orphans or []:
            if (
                orphan.suggested_fix is not None
                and "delete" not in orphan.suggested_fix.lower()
            ):
                deletions.append(
                    DeleteFix(
                        kind="node",
                        id=orphan.node_id or f"unknown_{uuid.uuid4().hex[:8]}",
                        reason="orphaned_node",
                    )
                )

        return ProposedFixes(add_nodes=add_nodes, merges=merges, deletions=deletions)

    def _apply_fixes_to_graph(
        self, kg: PaperSchema, proposed_fixes: ProposedFixes, original_text: str
    ) -> Final_Knowledge_Graph:
        """Apply fixes to create the final improved knowledge graph."""

        # Start with original graph
        final_nodes = list(kg.nodes)
        final_edges = list(kg.edges)

        existing_names = {node.name for node in final_nodes}

        # Apply add_nodes fixes
        for add_node in proposed_fixes.add_nodes or []:
            candidate_name = add_node.name if add_node.name else add_node.id
            # Ensure unique node names
            if candidate_name in existing_names:
                continue
            existing_names.add(candidate_name)
            final_nodes.append(
                GraphNode(
                    name=add_node.name if add_node.name else add_node.id,
                    aliases=[add_node.name if add_node.name else add_node.id],
                    type=add_node.type if add_node.type else "concept",
                    description="Auto-generated node based on validation",
                    concept_category=(
                        "framework" if add_node.type == "concept" else None
                    ),
                    intervention_lifecycle=None,
                    intervention_maturity=None,
                    intervention_lifecycle_rationale=None,
                    intervention_maturity_rationale=None,
                    node_rationale=None,
                )
            )

        # Apply deletions
        nodes_to_delete = {d.id for d in proposed_fixes.deletions if d.kind == "node"}
        final_nodes = [
            n.model_dump() for n in final_nodes if n.name not in nodes_to_delete
        ]

        final_edges = [
            e.model_dump()
            for e in final_edges
            if e.source_node not in nodes_to_delete
            and e.target_node not in nodes_to_delete
        ]

        return {
            "nodes": final_nodes,
            "edges": final_edges,
            "meta": {
                "version": "1.0",
                "source_hash": hashlib.md5(original_text.encode()).hexdigest(),
                "validation_timestamp": datetime.now().isoformat(),
            },
        }

    def _create_fallback_report(
        self,
        local_validation: LocalAssessment,
        error: str,
        raw_error: str,
        error_code: str,
        kg: PaperSchema,
        data_source: DataSource,
    ) -> JudgeReport:
        """Create a fallback report when GPT API fails."""

        local_issues = local_validation["issues"]
        has_blockers = any(issue["severity"] == "BLOCKER" for issue in local_issues)

        return JudgeReport(
            decision={
                "is_valid_json": True,
                "has_blockers": has_blockers,
                "flag_underperformance": len(local_issues) > 5,
                "summary": f"Local validation completed. Found {len(local_issues)} issues, {sum(1 for i in local_issues if i['severity'] == 'BLOCKER')} blockers.",
            },
            validation_report={
                "schema_check": local_issues,
                "referential_check": [],
                "orphans": [],
                "duplicates": [],
                "rationale_mismatches": [],
                "coverage": {"expected_edges_from_source": []},
            },
            proposed_fixes={
                "add_nodes": [],
                "merges": [],
                "deletions": [],
            },
            final_graph=self._kg_to_dict(kg),
            rationale_record={
                "method": "local_fallback",
                "notes": ["Used local validation due to API failure"],
            },
            url=data_source.url,
            paper_id=data_source.paper_id,
            ard_file_source=data_source.ard_file_source,
            errors=[f"System error: {error}", f"Raw response: {raw_error}", error_code],
        )

    def _kg_to_dict(self, kg: PaperSchema) -> Final_Knowledge_Graph:
        """Convert KnowledgeGraph to dictionary format."""
        return {
            "nodes": [node.model_dump() for node in kg.nodes],
            "edges": [edge.model_dump() for edge in kg.edges],
            "meta": {
                "version": "1.0",
                # "source_hash": hashlib.md5(original_text.encode()).hexdigest(),
                "source_hash": "fallback_hash",
                "validation_timestamp": None,
            },
        }


def get_by_file_url_to_text_map(ard_path: str) -> URL_To_Text_Map:
    jsonl_files = [
        os.path.join(ard_path, f) for f in os.listdir(ard_path) if f.endswith(".jsonl")
    ]
    out: URL_To_Text_Map = {}
    for jsonl_file in jsonl_files:
        stem = Path(jsonl_file).stem
        out_dict: Dict[str, str] = {}
        with open(jsonl_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    url = data["url"]
                    out_dict[url] = data["text"]
                except Exception:
                    continue
        out[stem] = out_dict
    return out


def get_all_json_files(base: Path) -> List[str]:
    """Get all JSON files that are exactly one directory level below base (base/*/*.json)."""
    json_files: List[str] = []
    for sd in sorted(p for p in base.iterdir() if p.is_dir()):
        json_files.extend(str(p) for p in sorted(sd.glob("*.json")) if p.is_file())
    return json_files


def find_url(kg_output: PaperSchema) -> Optional[str]:
    """Find the URL in the KG output metadata, if available."""
    for item in kg_output.meta:
        if item.key == "url":
            if item.value and isinstance(item.value, str):
                return item.value
    return None


async def process_directory(
    ard_dir: str,
    processed_dir: str,
    output_dir: str,
    how_many_batches_in_flight_at_once: int,
    batch_size: int,
):
    base = Path(processed_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Directory not found or not a directory: {base}")

    json_files = get_all_json_files(base)

    by_file_url_to_text_map = get_by_file_url_to_text_map(ard_dir)
    error_files: List[Tuple[str, str]] = []

    inputs: List[JudgeInput] = []

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # debug_test = []
    for json_file in json_files:
        the_split = Path(json_file).stem.split("__")
        if len(the_split) != 2:
            error_files.append((json_file, "Filename does not match expected pattern"))
            continue
        [ard_file_source, paper_id] = the_split
        url_to_text_map = by_file_url_to_text_map.get(ard_file_source)

        if url_to_text_map is None:
            error_files.append(
                (json_file, f"No URL to text mapping for {ard_file_source}")
            )
            continue
        try:
            with open(json_file, "r") as f:
                try:
                    kg_output = PaperSchema.model_validate_json(f.read())
                except Exception as e:
                    error_files.append((json_file, f"Failed to validate json: {e}"))
                    continue

                url = find_url(kg_output)
                if url is None:
                    error_files.append(
                        (json_file, "No URL found in KG output metadata")
                    )
                    continue
                original_text = url_to_text_map.get(url)
                if original_text is None:
                    error_files.append(
                        (json_file, f"No original text found for URL: {url}")
                    )
                    continue
                inputs.append(
                    JudgeInput(
                        original_text=original_text,
                        kg_output=kg_output,
                        data_source=DataSource(
                            url=url, paper_id=paper_id, ard_file_source=ard_file_source
                        ),
                    )
                )
                # debug_test.append({
                #     "text": original_text,
                #     "url": url,
                # })

        except Exception as e:
            error_files.append((json_file, str(e)))
            continue
    # with open("./extraction_validator/test_ard/arxiv.jsonl", "w") as f:
    #     for item in debug_test:
    #         f.write(json.dumps(item) + "\n")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    judge = KGJudge(output_dir=output_dir_path)
    # Run batch validation

    await judge.judge_knowledge_graph_batch(
        inputs,
        how_many_batches_in_flight_at_once=how_many_batches_in_flight_at_once,
        batch_size=batch_size,
    )
    with open(output_dir_path / "errors.json", "w") as f:
        json.dump(judge.requests_with_errors, f, indent=2)
    print(
        "KG-Judge processing complete."
        f" Total tokens used: {judge.total_tokens_used},"
        f" Prompt tokens: {judge.total_prompt_tokens},"
        f" Completion tokens: {judge.total_completion_tokens}."
    )

    with open(output_dir_path / "summary.json", "w") as f:
        json.dump(
            {
                "total_tokens_used": judge.total_tokens_used,
                "total_prompt_tokens": judge.total_prompt_tokens,
                "total_completion_tokens": judge.total_completion_tokens,
                "total_requests_unable_to_judbe_becase_of_techincal_errors": len(
                    judge.requests_with_errors
                ),
            },
            f,
            indent=2,
        )
    if not os.environ.get("DEBUG_SAVE_BATCH_REQUESTS") == "1":
        shutil.rmtree(judge.temp_dir)
    else:
        print(f"Debug mode: temporary files retained at {judge.temp_dir}")


def main(
    processed_dir: str,
    ard_dir: str,
    output_dir: str,
    # TODO look at openAI limits
    how_many_batches_in_flight_at_once: int = 5,
    batch_size: int = 5,
):
    """Main function to run KG-Judge on a directory of processed KG outputs.

    Args:
        processed_dir: Directory containing processed KG output files. These are the inputs to be judgged.
        ard_dir: The directory of the ARD dataset. You can pass the full ARD directory, and it will only judge those listed in processed_dir.
        output_dir: Directory to save the output files.
        how_many_batches_in_flight_at_once: Number of concurrent batches to process.
        batch_size: Number of requests per batch.
    """

    asyncio.run(
        process_directory(
            ard_dir,
            processed_dir,
            output_dir,
            how_many_batches_in_flight_at_once,
            batch_size,
        )
    )


if __name__ == "__main__":
    Fire(main)
