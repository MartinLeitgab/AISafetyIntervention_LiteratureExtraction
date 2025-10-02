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

from dotenv import load_dotenv
from openai import OpenAI

from config import load_settings
from intervention_graph_creation.src.prompt.final_primary_prompt import PROMPT_EXTRACT
from intervention_graph_creation.src.local_graph_extraction.extract.utilities import (
    safe_write,
    split_text_and_json,
    write_failure,
    url_to_id,
    filter_dict,
)

# =========================
# CONFIG & CONSTANTS
# =========================

MODEL = "o3"
REASONING_EFFORT = "medium"
SETTINGS = load_settings()
META_KEYS = frozenset(
    ["authors", "date_published", "filename", "source", "source_filetype", "title", "url"]
)

# =========================
# LOGGING SETUP
# =========================

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
            write_failure(SETTINGS.paths.output_dir, SETTINGS.paths.extraction_error_dir, stem, e)

    def get_output_text_from_response_body(self, response_body: Dict) -> str:
        if "output" not in response_body:
            raise ValueError("Response body does not contain 'output' field")
        output = response_body["output"]
        message = next((item for item in output if item.get("type") == "message"), None)
        if not message or message.get("status") != "completed":
            raise ValueError("Message not completed")
        text_obj = next((item for item in message.get("content", []) if item.get("type") == "output_text"), None)
        if not text_obj:
            raise ValueError("No output_text found")
        return text_obj["text"]

    def create_batch_requests(
        self, input_dir: Path, first_n: Optional[int] = None
    ) -> List[Dict]:
        batch_requests: List[Dict] = []

        pdf_files = list(input_dir.glob("*.pdf"))
        jsonl_files = list(input_dir.glob("*.jsonl"))

        if first_n:
            pdf_files = pdf_files[:first_n]

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
                    }
                )
            except Exception as e:
                write_failure(SETTINGS.paths.output_dir, SETTINGS.paths.extraction_error_dir, paper_id, e)

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
                            write_failure(SETTINGS.paths.output_dir, SETTINGS.paths.extraction_error_dir, pid, e)
                            continue

                        paper_id = jsonl_path.stem + "__" + url_to_id(
                            paper_json.get("url", f"line_{idx}")
                        )
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
                                "meta": filter_dict(paper_json, META_KEYS),
                            }
                        )
            except Exception as e:
                write_failure(SETTINGS.paths.output_dir, SETTINGS.paths.extraction_error_dir, jsonl_path.stem, e)

        return batch_requests

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
        last_log_time = 0.0
        while True:
            batch = await self._run_in_thread(self.client.batches.retrieve, batch_id)
            if batch.status == "completed":
                return batch
            if batch.status in ["failed", "expired", "cancelled"]:
                raise RuntimeError(f"Batch {batch_num} ended with status {batch.status}")

            now = time.time()
            if now - last_log_time >= check_interval:
                in_progress = self.total_batches - self.completed_batches
                logger.info(
                    "Progress update: Processed: %d / In progress: %d / Total: %d",
                    self.completed_batches, in_progress, self.total_batches
                )
                last_log_time = now

            await asyncio.sleep(60)

    def _process_batch_results(self, batch: Dict, batch_requests: List[Dict], elapsed: float) -> None:
        if batch.status != "completed":
            return

        custom_id_map = {req["request"]["custom_id"]: req for req in batch_requests}
        processed_ids = set()

        if getattr(batch, "output_file_id", None):
            file_response = self.client.files.content(batch.output_file_id)
            for line in file_response.text.strip().split("\n"):
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    cid = result.get("custom_id")
                    processed_ids.add(cid)

                    if result.get("error"):
                        error_obj = result["error"]
                        err_msg = f"{error_obj.get('code')}: {error_obj.get('message')}" if isinstance(error_obj, dict) else str(error_obj or "Unknown error")
                        error_json_str = json.dumps(error_obj, indent=2, ensure_ascii=False)
                        req = custom_id_map.get(cid)
                        file_path = req.get("file_path") if req else None
                        paper_id = file_path.stem if isinstance(file_path, Path) else str(file_path or "unknown")
                        diag = f"❌ Processing failed for {paper_id}\n{err_msg}\n\nFull error JSON:\n{error_json_str}\n"
                        err_dir = SETTINGS.paths.extraction_error_dir / paper_id
                        err_dir.mkdir(parents=True, exist_ok=True)
                        (err_dir / "error.txt").write_text(diag, encoding="utf-8")
                        self._papers_status.append({"status": "failed"})
                        continue

                    resp_body = result["response"]["body"]
                    req = custom_id_map.get(cid)
                    if not req:
                        continue
                    file_path = req.get("file_path")
                    paper_id = file_path.stem if isinstance(file_path, Path) else str(file_path or "unknown")
                    out_dir = SETTINGS.paths.output_dir / paper_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    self.write_batch_outputs(out_dir, paper_id, resp_body, req.get("meta"))
                    tokens = resp_body.get("usage", {}).get("total_tokens", 0)
                    self._papers_status.append({"status": "success", "elapsed": elapsed, "tokens": tokens})

                except Exception as e:
                    cid = result.get("custom_id")
                    req = custom_id_map.get(cid)
                    file_path = req.get("file_path") if req else None
                    paper_id = file_path.stem if isinstance(file_path, Path) else str(file_path or "unknown")
                    write_failure(SETTINGS.paths.output_dir, SETTINGS.paths.extraction_error_dir, paper_id, e)
                    self._papers_status.append({"status": "failed"})

        if getattr(batch, "error_file_id", None):
            file_response = self.client.files.content(batch.error_file_id)
            for line in file_response.text.strip().split("\n"):
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    cid = result.get("custom_id")
                    processed_ids.add(cid)
                    error_obj = result.get("error")
                    err_msg = f"{error_obj.get('code')}: {error_obj.get('message')}" if isinstance(error_obj, dict) else str(error_obj or "Unknown error")
                    error_json_str = json.dumps(error_obj, indent=2, ensure_ascii=False)
                    req = custom_id_map.get(cid)
                    file_path = req.get("file_path") if req else None
                    paper_id = file_path.stem if isinstance(file_path, Path) else str(file_path or "unknown")
                    diag = f"❌ Processing failed for {paper_id}\n{err_msg}\n\nFull error JSON:\n{error_json_str}\n"
                    err_dir = SETTINGS.paths.extraction_error_dir / paper_id
                    err_dir.mkdir(parents=True, exist_ok=True)
                    (err_dir / "error.txt").write_text(diag, encoding="utf-8")
                    self._papers_status.append({"status": "failed"})
                except Exception as e:
                    write_failure(SETTINGS.paths.output_dir, SETTINGS.paths.extraction_error_dir, "unknown", e)
                    self._papers_status.append({"status": "failed"})

        for cid, req in custom_id_map.items():
            if cid not in processed_ids:
                file_path = req.get("file_path") if req else None
                paper_id = file_path.stem if isinstance(file_path, Path) else str(file_path or "unknown")
                write_failure(SETTINGS.paths.output_dir, SETTINGS.paths.extraction_error_dir, paper_id, Exception("No response in batch output"))
                self._papers_status.append({"status": "failed"})

    async def _run_in_thread(self, func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, func, *args, **kwargs)

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
            logger.info("Total:   %.1fs (~%.1f min)", sum(self._batch_times), sum(self._batch_times)/60)

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
    total_articles = 100
    batch_size = 10

    async def main():
        await extractor.process_dir_batch_async(
            input_dir=input_dir,
            first_n=total_articles,
            batch_size=batch_size,
            description=f"Paper extraction (first {total_articles}, {batch_size} per batch)"
        )

    asyncio.run(main())
