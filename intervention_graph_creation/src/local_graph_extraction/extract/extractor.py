import os
import json
import time
import httpx
import asyncio
import logging
import statistics
from pathlib import Path
from typing import Any, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib


from dotenv import load_dotenv
from openai import OpenAI

from config import load_settings
from intervention_graph_creation.src.prompt.final_primary_prompt import PROMPT_EXTRACT
from intervention_graph_creation.src.local_graph_extraction.extract.utilities import (
    safe_write,
    split_text_and_json,
    filter_dict,
    write_failure,
)

MODEL = "o3"
REASONING_EFFORT = "medium"
SETTINGS = load_settings()
META_KEYS = frozenset(
    ["authors", "date_published", "filename", "source", "source_filetype", "title", "url"]
)

LOGS_DIR = SETTINGS.paths.logs_dir
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / f"extraction_{datetime.now().strftime('%m-%d_%H-%M')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Extractor:
    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        http_client = httpx.Client(
            timeout=httpx.Timeout(300.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=300.0),
            headers={"Connection": "keep-alive", "Keep-Alive": "timeout=300, max=1000"},
            transport=httpx.HTTPTransport(retries=3, verify=True),
        )

        self.client = OpenAI(
            api_key=api_key,
            timeout=300.0,
            max_retries=3,
            http_client=http_client,
        )

        self._papers_status: List[Dict[str, Any]] = []
        self._batch_times: List[float] = []

        self.total_batches = 0
        self.completed_batches = 0

        self._last_log_time = 0.0
        self._log_lock = asyncio.Lock()

    # ---------------------------
    # Failure handling (delegates to write_failure)
    # ---------------------------
    def _handle_failure(
        self,
        *,
        req: Optional[Dict],
        err: Optional[Exception],
        stage: Optional[str] = None,
        raw_line: Optional[str] = None,
        result: Optional[Dict] = None,
        raw_response_status_note: Optional[str] = None,
        meta: Optional[Dict] = None,
        file_type: Optional[str] = None,
    ) -> None:
        """Record a failure using write_failure and update in-memory stats."""
        # Determine paper_id
        paper_id = "unknown"
        if req:
            if req.get("paper_id"):
                paper_id = req["paper_id"]
            else:
                fp = req.get("file_path")
                paper_id = fp.stem if isinstance(fp, Path) else str(fp or "unknown")

        # Determine file_type if not provided
        if not file_type and req:
            ft = req.get("file_type")
            if not ft:
                fp = req.get("file_path")
                if isinstance(fp, Path):
                    suffix = fp.suffix.lower().lstrip(".")
                    ft = suffix if suffix else None
            file_type = ft

        # Build context payload (kept compact; no deep parsing of result)
        context: Dict[str, Any] = {
            "stage": stage,
            "file_type": file_type,
            "meta": meta,
        }

        if req:
            request_obj = req.get("request") or {}
            body = request_obj.get("body") or {}
            context.update({
                "request": {
                    "endpoint": request_obj.get("url"),
                    "custom_id": request_obj.get("custom_id"),
                    "body_keys": list(body.keys()) if isinstance(body, dict) else None,
                },
                "file_path": str(req.get("file_path")) if req.get("file_path") else None,
            })

        if raw_line:
            snippet = raw_line if len(raw_line) <= 4096 else raw_line[:4096] + "...(truncated)"
            context["raw_error_snippet"] = snippet

        if raw_response_status_note:
            context["raw_response_status_note"] = raw_response_status_note

        if result is not None:
            # Keep only top-level fields to avoid huge dumps
            context["result_keys"] = list(result.keys())

        # Delegate to write_failure (moves output → error folder, writes error.txt)
        write_failure(
            output_root=SETTINGS.paths.output_dir,
            error_root=SETTINGS.paths.extraction_error_dir,
            paper_id=paper_id,
            err=err,
            error_type=stage,
            context=context,
            logger=logger,
        )

        self._papers_status.append({"status": "failed"})

    # ---------------------------
    # Output writing
    # ---------------------------
    def write_batch_outputs(
        self, out_dir: Path, stem: str, response_body: Dict, meta: Dict
    ) -> None:
        raw_path = out_dir / f"{stem}_raw_response.txt"
        json_path = out_dir / f"{stem}.json"
        summary_path = out_dir / f"{stem}_summary.txt"

        safe_write(raw_path, json.dumps(response_body, ensure_ascii=False, indent=2))

        try:
            output_text = self.get_output_text_from_response_body(response_body)
            text_part, json_part = split_text_and_json(output_text)
            safe_write(summary_path, text_part or "")

            if json_part:
                parsed = json.loads(json_part)
                if meta:
                    parsed["meta"] = meta
                safe_write(json_path, json.dumps(parsed, ensure_ascii=False, indent=2))
        except Exception as e:
            self._handle_failure(
                req={"paper_id": stem, "request": {}, "file_type": None},
                err=e,
                stage="write_batch_outputs",
                raw_line=None,
                result=None,
                meta=meta,
                file_type=None,
            )

    def get_output_text_from_response_body(self, response_body: Dict) -> str:
        if "output" not in response_body:
            raise ValueError("Response body does not contain 'output'")
        output = response_body["output"]
        message = next((item for item in output if item.get("type") == "message"), None)
        if not message:
            raise ValueError("No message object in output")
        if message.get("status") != "completed":
            raise ValueError(f"Message not completed (status={message.get('status')})")
        text_obj = next((item for item in message.get("content", []) if item.get("type") == "output_text"), None)
        if not text_obj:
            raise ValueError("No output_text found")
        return text_obj["text"]

    # ---------------------------
    # Batch request creation
    # ---------------------------
    def create_batch_requests(
        self, input_dir: Path, first_n: Optional[int] = None
    ) -> List[Dict]:
        batch_requests: List[Dict] = []

        pdf_files = list(input_dir.glob("*.pdf"))
        jsonl_files = list(input_dir.glob("*.jsonl"))

        if first_n:
            pdf_files = pdf_files[:first_n]

        # PDFs
        for idx, pdf_path in enumerate(pdf_files):
            paper_id = pdf_path.stem
            out_dir = SETTINGS.paths.output_dir / paper_id
            err_dir = SETTINGS.paths.extraction_error_dir / paper_id

            if out_dir.exists() or err_dir.exists():
                logger.info("Skipping already processed or failed PDF: %s", pdf_path.name)
                continue

            try:
                with pdf_path.open("rb") as fh:
                    f = self.client.files.create(file=fh, purpose="user_data")
                file_id = f.id
                custom_id = f"pdf_{pdf_path.stem}_{idx}"

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": MODEL,
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_file", "file_id": file_id},
                                    {"type": "input_text", "text": PROMPT_EXTRACT},
                                ],
                            }
                        ],
                        "reasoning": {"effort": REASONING_EFFORT},
                    },
                }
                batch_requests.append(
                    {
                        "request": request,
                        "file_path": pdf_path,
                        "file_type": "pdf",
                        "meta": {"filename": pdf_path.name},
                        "paper_id": paper_id,
                    }
                )
            except Exception as e:
                self._handle_failure(
                    req={"paper_id": paper_id, "file_path": pdf_path, "request": {}, "file_type": "pdf"},
                    err=e,
                    stage="create_batch_requests (pdf)",
                    meta={"filename": pdf_path.name},
                    file_type="pdf",
                )

        # JSONL
        json_items_cap = first_n if first_n else None
        for jsonl_path in jsonl_files:
            if json_items_cap is not None and json_items_cap <= 0:
                break

            try:
                with jsonl_path.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        if json_items_cap is not None and len(batch_requests) >= first_n:
                            break
                        if not line.strip():
                            continue

                        try:
                            paper_json = json.loads(line)
                        except Exception as e:
                            pid = f"{jsonl_path.stem}__badjson_{idx}"
                            self._handle_failure(
                                req={"paper_id": pid, "file_path": jsonl_path, "request": {}, "file_type": "jsonl"},
                                err=e,
                                stage="create_batch_requests (jsonl parse)",
                                meta={"filename": jsonl_path.name},
                                file_type="jsonl",
                            )
                            continue

                        url = paper_json.get("url", f"line_{idx}")
                        md5_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
                        paper_id = f"{jsonl_path.stem}__{md5_hash}"

                        out_dir = SETTINGS.paths.output_dir / paper_id
                        err_dir = SETTINGS.paths.extraction_error_dir / paper_id

                        if out_dir.exists() or err_dir.exists():
                            logger.info("Skipping already processed or failed JSONL: %s", paper_id)
                            continue

                        custom_id = f"jsonl_{paper_id}_{idx}"
                        request = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/responses",
                            "body": {
                                "model": MODEL,
                                "input": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "input_text", "text": PROMPT_EXTRACT},
                                            {
                                                "type": "input_text",
                                                "text": f"\n\nHere is the paper for analysis:\n\n{paper_json['text']}",
                                            },
                                        ],
                                    }
                                ],
                                "reasoning": {"effort": REASONING_EFFORT},
                            },
                        }
                        batch_requests.append(
                            {
                                "request": request,
                                "file_path": jsonl_path,
                                "file_type": "jsonl",
                                "paper_id": paper_id,
                                "meta": filter_dict(paper_json, META_KEYS),
                            }
                        )
            except Exception as e:
                self._handle_failure(
                    req={"paper_id": jsonl_path.stem, "file_path": jsonl_path, "request": {}, "file_type": "jsonl"},
                    err=e,
                    stage="create_batch_requests (jsonl file read)",
                    meta={"filename": jsonl_path.name},
                    file_type="jsonl",
                )

        return batch_requests

    # ---------------------------
    # Batch job lifecycle
    # ---------------------------
    def create_batch_input_file(self, batch_requests: List[Dict], batch_num: int = 1) -> str:
        batch_input_path = SETTINGS.paths.output_dir / f"batch_input_{batch_num}_{int(time.time())}.jsonl"
        with batch_input_path.open("w", encoding="utf-8") as f:
            for batch_req in batch_requests:
                f.write(json.dumps(batch_req["request"], ensure_ascii=False) + "\n")

        with batch_input_path.open("rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")

        try:
            batch_input_path.unlink()
        except Exception:
            pass

        return batch_input_file.id

    def create_batch_job(self, input_file_id: str, description: str) -> str:
        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": description},
        )
        return batch.id

    async def process_dir_batch_async(
        self,
        input_dir: Path,
        first_n: Optional[int] = None,
        batch_size: int = 5,
        description: str = "Paper extraction batch",
    ) -> None:
        SETTINGS.paths.output_dir.mkdir(parents=True, exist_ok=True)
        batch_requests = await self._run_in_thread(self.create_batch_requests, input_dir, first_n)

        if not batch_requests:
            logger.warning("No files to process.")
            return

        batch_chunks = [
            {
                "chunk": batch_requests[i:i + batch_size],
                "batch_num": i // batch_size + 1,
                "description": f"{description} - Batch {i // batch_size + 1}",
            }
            for i in range(0, len(batch_requests), batch_size)
        ]

        self.total_batches = len(batch_chunks)
        logger.info("Processing %d batches concurrently", self.total_batches)

        tasks = [self._process_single_batch_async(info) for info in batch_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch %d failed with exception: %s", batch_chunks[i]["batch_num"], result)

        self.print_summary()

    async def _process_single_batch_async(self, batch_info: Dict[str, Any]) -> Any:
        chunk = batch_info["chunk"]
        batch_num = batch_info["batch_num"]
        desc = batch_info["description"]

        in_progress = self.total_batches - self.completed_batches
        logger.info(
            "Batch %d started (Processed: %d / In progress: %d / Total: %d)",
            batch_num, self.completed_batches, in_progress, self.total_batches
        )

        t0 = time.time()
        input_file_id = await self._run_in_thread(self.create_batch_input_file, chunk, batch_num)
        batch_id = await self._run_in_thread(self.create_batch_job, input_file_id, desc)

        completed_batch = await self._wait_for_batch_completion_async(batch_id, batch_num)

        elapsed = time.time() - t0
        self._batch_times.append(elapsed)

        await self._run_in_thread(self._process_batch_results, completed_batch, chunk, elapsed)

        self.completed_batches += 1
        in_progress = self.total_batches - self.completed_batches
        logger.info(
            "Batch %d completed in %.1fs (Processed: %d / In progress: %d / Total: %d)",
            batch_num, elapsed, self.completed_batches, in_progress, self.total_batches
        )

        return completed_batch

    async def _wait_for_batch_completion_async(
        self, batch_id: str, batch_num: int, check_interval: int = 60
    ) -> Any:
        while True:
            batch = await self._run_in_thread(self.client.batches.retrieve, batch_id)

            if batch.status == "completed":
                return batch

            if batch.status in ["failed", "expired", "cancelled"]:
                raise RuntimeError(f"Batch {batch_num} ended with status {batch.status}")

            now = time.time()

            async with self._log_lock:
                if now - self._last_log_time >= check_interval:
                    in_progress = self.total_batches - self.completed_batches
                    logger.info(
                        "Progress update: Processed: %d / In progress: %d / Total: %d",
                        self.completed_batches, in_progress, self.total_batches
                    )
                    self._last_log_time = now

            await asyncio.sleep(60)

    # ---------------------------
    # Batch result processing
    # ---------------------------
    def _process_batch_results(self, batch: Dict, batch_requests: List[Dict], elapsed: float) -> None:
        if batch.status != "completed":
            return

        custom_id_map = {req["request"]["custom_id"]: req for req in batch_requests}
        processed_ids = set()

        def _get_paper_id(req: Dict) -> str:
            if not req:
                return "unknown"
            if "paper_id" in req:
                return req["paper_id"]
            file_path = req.get("file_path")
            if isinstance(file_path, Path):
                return file_path.stem
            return str(file_path or "unknown")

        if getattr(batch, "output_file_id", None):
            try:
                file_response = self.client.files.content(batch.output_file_id)
            except Exception as e:
                logger.exception("Failed to read output_file_id: %s", e)
                return

            for raw_line in file_response.text.strip().split("\n"):
                if not raw_line:
                    continue
                result = None
                try:
                    result = json.loads(raw_line)
                    cid = result.get("custom_id")
                    processed_ids.add(cid)

                    req = custom_id_map.get(cid)
                    paper_id = _get_paper_id(req)

                    if result.get("error"):
                        self._handle_failure(
                            req=req,
                            err=None,
                            stage="batch_output_file (error object)",
                            raw_line=raw_line,
                            result=result,
                            raw_response_status_note=None,
                            meta=(req or {}).get("meta"),
                            file_type=(req or {}).get("file_type"),
                        )
                        continue

                    resp_body = result.get("response", {}).get("body")
                    if not resp_body:
                        self._handle_failure(
                            req=req,
                            err=Exception("No 'response.body' in output file"),
                            stage="batch_output_file (no response.body)",
                            raw_line=raw_line,
                            result=result,
                            meta=(req or {}).get("meta"),
                            file_type=(req or {}).get("file_type"),
                        )
                        continue

                    try:
                        _ = self.get_output_text_from_response_body(resp_body)
                    except Exception as e:
                        status_note = ""
                        try:
                            out = resp_body.get("output", [])
                            message_obj = next((it for it in out if it.get("type") == "message"), None)
                            status_note = f"Message status: {message_obj.get('status') if message_obj else 'N/A'}"
                        except Exception:
                            pass
                        self._handle_failure(
                            req=req,
                            err=e,
                            stage="batch_output_parsing",
                            raw_line=raw_line,
                            result=result,
                            raw_response_status_note=status_note or None,
                            meta=(req or {}).get("meta"),
                            file_type=(req or {}).get("file_type"),
                        )
                        continue

                    out_dir = SETTINGS.paths.output_dir / paper_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    self.write_batch_outputs(out_dir, paper_id, resp_body, (req or {}).get("meta"))
                    tokens = resp_body.get("usage", {}).get("total_tokens", 0)
                    self._papers_status.append({"status": "success", "elapsed": elapsed, "tokens": tokens})

                except Exception as e:
                    req = custom_id_map.get(result.get("custom_id")) if isinstance(result, dict) else None
                    self._handle_failure(
                        req=req,
                        err=e,
                        stage="batch_output_file (json parse)",
                        raw_line=raw_line,
                        result=result if isinstance(result, dict) else None,
                        meta=(req or {}).get("meta") if req else None,
                        file_type=(req or {}).get("file_type") if req else None,
                    )

        if getattr(batch, "error_file_id", None):
            try:
                file_response = self.client.files.content(batch.error_file_id)
            except Exception as e:
                logger.exception("Failed to read error_file_id: %s", e)
                return

            for raw_line in file_response.text.strip().split("\n"):
                if not raw_line:
                    continue
                result = None
                try:
                    result = json.loads(raw_line)
                    cid = result.get("custom_id")
                    processed_ids.add(cid)

                    req = custom_id_map.get(cid)

                    self._handle_failure(
                        req=req,
                        err=None,
                        stage="batch_error_file",
                        raw_line=raw_line,
                        result=result,
                        meta=(req or {}).get("meta") if req else None,
                        file_type=(req or {}).get("file_type") if req else None,
                    )

                except Exception as e:
                    self._handle_failure(
                        req=None,
                        err=e,
                        stage="batch_error_file (json parse)",
                        raw_line=raw_line,
                        result=None,
                        meta=None,
                        file_type=None,
                    )

        for cid, req in custom_id_map.items():
            if cid not in processed_ids:
                self._handle_failure(
                    req=req,
                    err=Exception("No response in batch output"),
                    stage="batch_missing_custom_id",
                    raw_line=None,
                    result=None,
                    meta=(req or {}).get("meta"),
                    file_type=(req or {}).get("file_type"),
                )

    # ---------------------------
    # Thread helper
    # ---------------------------
    async def _run_in_thread(self, func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, func, *args, **kwargs)

    # ---------------------------
    # Summary
    # ---------------------------
    def print_summary(self) -> None:
        total = len(self._papers_status)
        success = sum(1 for p in self._papers_status if p.get("status") == "success")
        failed = sum(1 for p in self._papers_status if p.get("status") == "failed")

        tokens = [p["tokens"] for p in self._papers_status if p.get("status") == "success" and "tokens" in p]

        logger.info("=== 📊 Extraction Summary ===")
        logger.info("Total papers processed: %d", total)
        logger.info("Successful:             %d", success)
        logger.info("Failed:                 %d", failed)

        if self._batch_times:
            logger.info("\n--- ⏱️ Time per batch (seconds) ---")
            logger.info("Average: %.1fs", statistics.mean(self._batch_times))
            logger.info("Median:  %.1fs", statistics.median(self._batch_times))
            logger.info("Min:     %.1fs", min(self._batch_times))
            logger.info("Max:     %.1fs", max(self._batch_times))

        if tokens:
            logger.info("\n--- 🔤 Token usage per article ---")
            logger.info("Average: %d", int(statistics.mean(tokens)))
            logger.info("Median:  %d", int(statistics.median(tokens)))
            logger.info("Min:     %d", min(tokens))
            logger.info("Max:     %d", max(tokens))
            logger.info("Total:   %d", sum(tokens))


if __name__ == "__main__":
    extractor = Extractor()

    input_dir = SETTINGS.paths.input_dir
    total_articles = 1000
    batch_size = 10

    async def main():
        await extractor.process_dir_batch_async(
            input_dir=input_dir,
            first_n=total_articles,
            batch_size=batch_size,
            description=f"Paper extraction (first {total_articles}, {batch_size} per batch)"
        )

    asyncio.run(main())
