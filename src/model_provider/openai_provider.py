import asyncio
import json
import os
import tempfile
from typing import List, Dict, Any, Optional, Type
from openai import AsyncOpenAI
from openai.lib._parsing._completions import type_to_response_format_param
from datetime import datetime
from pydantic import BaseModel

# Module-level defaults/constants
OPENAI_BATCH_ENDPOINT = "/v1/chat/completions"
OPENAI_COMPLETION_WINDOW = "24h"
OPENAI_POLL_INTERVAL_SECS = 10

class OpenAIModelProvider:

    def __init__(
        self,
        api_key: str,
        model_name: str = "o3",
        batch_size_limit: int = 3000,
    ):
        """Initialize the OpenAI provider with batch processing configuration.

        Args:
            api_key: OpenAI API key
            model_name: Model to use
            batch_size_limit: Number of requests per batch (recommended: 1K-5K)
        """
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
        self.batch_size_limit = batch_size_limit

        # Directory to store batch manifests for later resumption
        self.manifest_dir: str = os.path.abspath("openai_batch_manifests")
        os.makedirs(self.manifest_dir, exist_ok=True)
        self.last_batch_manifest_path: Optional[str] = None

        print(
            f"Initialized OpenAIModelProvider(model={self.model_name}, batch_size_limit={self.batch_size_limit}, manifest_dir={self.manifest_dir})"
        )

    async def batch_infer_structured(self, prompts: List[str], model_cls: Type[BaseModel]) -> List[Optional[BaseModel]]:
        """Batch inference returning Pydantic instances for each prompt."""
        if not prompts:
            print("No prompts provided for batch_infer_structured; returning empty list.")
            return []

        total_prompts = len(prompts)
        num_chunks = (total_prompts + self.batch_size_limit - 1) // self.batch_size_limit
        print(
            f"Creating {num_chunks} batch job(s) for {total_prompts} prompts (chunk_size_limit={self.batch_size_limit})"
        )

        batch_jobs: List[Dict[str, Any]] = []
        for chunk_start in range(0, len(prompts), self.batch_size_limit):
            chunk = prompts[chunk_start:chunk_start + self.batch_size_limit]
            chunk_end_inclusive = chunk_start + len(chunk) - 1
            try:
                batch_file_path = await self._create_batch_file(
                    chunk, chunk_start, extra_body={"response_format": type_to_response_format_param(model_cls)}
                )
                print(
                    f"Prepared input JSONL for chunk [{chunk_start}..{chunk_end_inclusive}] at {batch_file_path}"
                )
                with open(batch_file_path, "rb") as f:
                    batch_file = await self.openai_client.files.create(file=f, purpose="batch")
                print(
                    f"Uploaded input file (id={getattr(batch_file, 'id', 'unknown')}) for chunk [{chunk_start}..{chunk_end_inclusive}]"
                )
                batch_job = await self.openai_client.batches.create(
                    input_file_id=batch_file.id,
                    endpoint=OPENAI_BATCH_ENDPOINT,
                    completion_window=OPENAI_COMPLETION_WINDOW,
                )
                print(
                    f"Created batch job (id={getattr(batch_job, 'id', 'unknown')}) for chunk [{chunk_start}..{chunk_end_inclusive}]"
                )
                batch_jobs.append({
                    "job_id": batch_job.id,
                    "chunk_size": len(chunk),
                    "chunk_start": chunk_start,
                    "input_file_id": batch_file.id,
                    "input_local_path": batch_file_path,
                    "status": "created",
                })
            except Exception as e:
                print(f"Failed to create structured batch job for chunk [{chunk_start}..{chunk_end_inclusive}]: {e}")
                batch_jobs.append({
                    "error": str(e),
                    "chunk_size": len(chunk),
                    "chunk_start": chunk_start,
                    "status": "creation_failed",
                })

        # Best-effort manifest
        try:
            self.last_batch_manifest_path = self._export_batch_manifest(
                batch_jobs=batch_jobs,
                total_count=len(prompts),
            )
            print(f"Wrote batch manifest to {self.last_batch_manifest_path}")
        except Exception as e:
            print(f"Failed to write batch manifest (structured): {e}")

        created_count = sum(1 for j in batch_jobs if j.get("status") == "created" and "job_id" in j)
        print(f"Waiting on {created_count}/{len(batch_jobs)} created batch job(s)...")

        # Await all jobs and parse with provided model
        async def _await_struct(job: Dict[str, Any]) -> Dict[str, Any]:
            chunk_size = int(job.get("chunk_size", 0))
            chunk_start = int(job.get("chunk_start", 0))
            if job.get("status") == "creation_failed" or "error" in job or "job_id" not in job:
                print(f"Skipping await for chunk starting at {chunk_start} due to creation failure.")
                return {"chunk_start": chunk_start, "chunk_size": chunk_size, "results": [None] * chunk_size}
            try:
                results = await self._wait_for_batch_completion(job["job_id"])
                parsed = self._parse_batch_results_structured(results, chunk_size, model_cls)
                print(f"Parsed {len(parsed)} result(s) for chunk starting at {chunk_start}")
                return {"chunk_start": chunk_start, "chunk_size": chunk_size, "results": parsed}
            except Exception as e:
                print(f"Structured job await failed: {e}")
                return {"chunk_start": chunk_start, "chunk_size": chunk_size, "results": [None] * chunk_size}

        chunks = await asyncio.gather(*(_await_struct(j) for j in batch_jobs), return_exceptions=True)
        consolidated_any = self._consolidate_chunks_generic(chunks, total_count=len(prompts), default_value=None)
        success = sum(1 for c in consolidated_any if isinstance(c, BaseModel))
        print(f"Batch inference complete: {success}/{len(consolidated_any)} successful.")
        return [c if isinstance(c, BaseModel) else None for c in consolidated_any]

    def _export_batch_manifest(self, batch_jobs: List[Dict[str, Any]], total_count: int) -> str:
        manifest: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model": self.model_name,
            "endpoint": OPENAI_BATCH_ENDPOINT,
            "completion_window": OPENAI_COMPLETION_WINDOW,
            "total_count": total_count,
            "jobs": batch_jobs,
        }

        filename = f"openai_batch_manifest_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        manifest_path = os.path.join(self.manifest_dir, filename)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest_path

    async def wait_structured_from_manifest(self, manifest_path: str, model_cls: Type[BaseModel]) -> List[Optional[BaseModel]]:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            print(f"Loaded manifest from {manifest_path}")
        except Exception as e:
            print(f"Failed to read manifest '{manifest_path}': {e}")
            return []

        jobs: List[Dict[str, Any]] = manifest.get("jobs", [])
        total_count: int = int(manifest.get("total_count", 0))
        print(f"Resuming {len(jobs)} job(s) from manifest; total_count={total_count}")

        # Await all jobs and parse with provided model
        async def _await_struct(job: Dict[str, Any]) -> Dict[str, Any]:
            chunk_size = int(job.get("chunk_size", 0))
            chunk_start = int(job.get("chunk_start", 0))
            if job.get("status") == "creation_failed" or "error" in job or "job_id" not in job:
                print(f"Skipping await for chunk starting at {chunk_start} due to creation failure.")
                return {"chunk_start": chunk_start, "chunk_size": chunk_size, "results": [None] * chunk_size}
            try:
                results = await self._wait_for_batch_completion(job["job_id"])
                parsed = self._parse_batch_results_structured(results, chunk_size, model_cls)
                print(f"Parsed {len(parsed)} result(s) for chunk starting at {chunk_start}")
                return {"chunk_start": chunk_start, "chunk_size": chunk_size, "results": parsed}
            except Exception as e:
                print(f"Structured job await failed: {e}")
                return {"chunk_start": chunk_start, "chunk_size": chunk_size, "results": [None] * chunk_size}

        chunks = await asyncio.gather(*(_await_struct(j) for j in jobs), return_exceptions=True)
        consolidated_any = self._consolidate_chunks_generic(chunks, total_count=total_count, default_value=None)
        success = sum(1 for c in consolidated_any if isinstance(c, BaseModel))
        print(f"Manifest wait complete: {success}/{total_count} successful.")
        return [c if isinstance(c, BaseModel) else None for c in consolidated_any]



    async def _create_batch_file(
        self,
        prompts: List[str],
        start_index: int,
        *,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".jsonl", prefix=f"batch_{start_index}_"
        )

        try:
            with os.fdopen(temp_fd, "w") as f:
                f.write("\n".join(
                    json.dumps({
                        "custom_id": str(i),
                        "method": "POST",
                        "url": OPENAI_BATCH_ENDPOINT,
                        "body": {
                            "model": self.model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            **(extra_body or {}),
                        },
                    })
                    for i, prompt in enumerate(prompts)
                ) + "\n")
            print(f"Wrote JSONL for {len(prompts)} prompt(s) at {temp_path}")
            return temp_path

        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    async def _wait_for_batch_completion(self, batch_id: str) -> List[Dict[str, Any]]:
        last_status = None
        last_counts_repr = None
        while True:
            batch_status = await self.openai_client.batches.retrieve(batch_id)

            status = getattr(batch_status, "status", "unknown")
            counts = getattr(batch_status, "request_counts", None)
            # Normalize counts for comparison/logging
            if isinstance(counts, dict):
                total = counts.get("total")
                completed = counts.get("completed")
                failed = counts.get("failed")
                counts_repr = f"completed={completed}/{total}, failed={failed}"
            else:
                counts_repr = None

            if status != last_status or counts_repr != last_counts_repr:
                if counts_repr:
                    print(f"Batch {batch_id} status={status} ({counts_repr})")
                else:
                    print(f"Batch {batch_id} status={status}")
                last_status = status
                last_counts_repr = counts_repr

            if status == "completed":
                if batch_status.output_file_id:
                    output_file = await self.openai_client.files.content(
                        batch_status.output_file_id
                    )
                    lines = [line for line in output_file.text.strip().split("\n") if line.strip()]
                    print(f"Batch {batch_id} completed with {len(lines)} output line(s).")
                    return [json.loads(line) for line in lines]
                else:
                    print(f"Batch {batch_id} completed but no output file available")
                    raise RuntimeError("Batch completed but no output file available")

            elif status == "failed":
                error_file_id = getattr(batch_status, "error_file_id", None)
                if isinstance(error_file_id, str):
                    try:
                        error_file = await self.openai_client.files.content(error_file_id)
                        print(
                            f"Batch failed (id={batch_id}) error_file_id={error_file_id}: {error_file.text}"
                        )
                    except Exception as fetch_err:
                        print(
                            f"Batch failed (id={batch_id}) error_file_id={error_file_id} (failed to fetch details): {fetch_err}"
                        )
                else:
                    print(
                        f"Batch failed (id={batch_id}). No error_file_id provided by API. Likely input JSONL validation failure. Check your JSONL and OpenAI dashboard."
                    )
                return []

            elif status in ["cancelled", "expired"]:
                print(
                    f"Batch {status} (id={batch_id}). input_file_id={getattr(batch_status, 'input_file_id', 'unknown')}"
                )
                return []

            await asyncio.sleep(OPENAI_POLL_INTERVAL_SECS)

    def _parse_batch_results_structured(
        self,
        results: List[Dict[str, Any]],
        expected_count: int,
        model_cls: Type[BaseModel],
    ) -> List[Optional[BaseModel]]:
        ordered: List[Optional[BaseModel]] = [None] * expected_count
        for r in results:
            try:
                idx = int(r.get("custom_id", -1))
                resp: Dict[str, Any] = r.get("response", {})
                if 0 <= idx < expected_count and resp.get("status_code") == 200:
                    body = resp.get("body", {})
                    choices = body.get("choices", [])
                    content = (choices[0].get("message", {}) if choices else {}).get("content") or "{}"
                    # Validate
                    if hasattr(model_cls, "model_validate_json"):
                        ordered[idx] = model_cls.model_validate_json(content)  # type: ignore[attr-defined]
                    else:
                        ordered[idx] = model_cls.parse_raw(content)  # type: ignore[attr-defined]
            except Exception as e:
                print(f"Failed to parse structured batch result: {e}")
        return ordered

    def _consolidate_chunks_generic(self, chunks: List[Any], total_count: int, default_value: Any) -> List[Any]:
        consolidated: List[Any] = [default_value] * total_count
        for entry in chunks:
            if isinstance(entry, Exception) or not isinstance(entry, dict):
                continue
            chunk_start = int(entry.get("chunk_start", 0))
            results: List[Any] = entry.get("results", [])
            for i, content in enumerate(results):
                absolute_index = chunk_start + i
                if 0 <= absolute_index < total_count:
                    consolidated[absolute_index] = content if content is not None else default_value
        return consolidated
