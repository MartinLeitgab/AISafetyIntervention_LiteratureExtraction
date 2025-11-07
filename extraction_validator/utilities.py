"""
Lightweight PreLLM KG/JSON validator

Validates:
- Schema shape and required fields
- ID format and uniqueness
- Referential integrity (edges -> nodes, chains alternate node/edge IDs & all IDs exist)
- Orphan nodes (degree 0 and unused in chains)
- Duplicate nodes by (type, normalized name)
"""

# pyright: strict

from typing import Any, List, Literal, Optional, TypedDict, cast
from openai import AsyncOpenAI
from dataclasses import dataclass
from os import environ
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
)

from extraction_validator.schema import PaperSchema
import uuid

USE_DEBUG_BATCH = environ.get("USE_DEBUG_BATCH", "0") == "1"


class CompletionsRequest(TypedDict):
    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/chat/completions"]
    body: CompletionCreateParamsNonStreaming


@dataclass
class DataSource:
    url: str
    paper_id: str
    ard_file_source: str


@dataclass
class JudgeRequest:
    request: CompletionsRequest
    original_text: str
    kg_output: PaperSchema
    data_source: DataSource
    custom_id: str


def unknown_judge_request(custom_id: str) -> JudgeRequest:
    return JudgeRequest(
        request=CompletionsRequest(
            custom_id=custom_id,
            method="POST",
            url="/v1/chat/completions",
            body=cast(Any, {}),  # empty body for unknown request
        ),
        original_text="A text that I don't know about",
        kg_output=PaperSchema(nodes=[], edges=[]),
        data_source=DataSource(
            url=str(uuid.uuid4()), paper_id=str(uuid.uuid4()), ard_file_source=""
        ),
        custom_id=custom_id,
    )


@dataclass
class JudgeBatch:
    file_path: str
    requests: List[JudgeRequest]


@dataclass
class JudgeBatchResult:
    is_debug: bool
    """Whether this is a debug batch (just for testing purpose, not uploaded to OpenAI)"""
    batch_id: str
    batch: JudgeBatch


@dataclass
class BatchOutput:
    content: Optional[str] = None
    error_content: Optional[str] = None
    type: Literal["content"] = "content"


JudgeErrorCode = Literal[
    "unknown",
    "unknown_custom_id",
    "parse_or_validate_error",
    "batch_not_completed",
    "missing_output",
    "http_error",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
    "empty_response",
    "batch_cancelled",
    "batch_status_check_error",
    "process_completed_batch_error",
    "process_result_content_error",
]


@dataclass
class EverythingInTheBatchHasAnError:
    message: str
    raw: str
    error_code: JudgeErrorCode
    type: Literal["error"] = "error"


BatchResult = BatchOutput | EverythingInTheBatchHasAnError


def ok(value: bool) -> str:
    return "PASS" if value else "FAIL"


async def upload_and_create_batch(
    client: AsyncOpenAI, judge_batch: JudgeBatch
) -> JudgeBatchResult:
    if USE_DEBUG_BATCH:
        return JudgeBatchResult(
            is_debug=True, batch_id="debug-batch-id", batch=judge_batch
        )

    # Upload the file
    with open(judge_batch.file_path, "rb") as f:
        uploaded_file = await client.files.create(file=f, purpose="batch")

    # Create the batch
    batch = await client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    return JudgeBatchResult(is_debug=False, batch_id=batch.id, batch=judge_batch)
