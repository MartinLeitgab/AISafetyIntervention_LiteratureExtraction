import os
import json
import time
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from config import load_settings
from intervention_graph_creation.src.prompt.final_primary_prompt import PROMPT_EXTRACT
from intervention_graph_creation.src.local_graph_extraction.extract.utilities import (
    safe_write,
    split_text_and_json,
    stringify_response,
    extract_output_text,
    write_failure,
    url_to_id,
    filter_dict,
)

MODEL = "o3"
REASONING_EFFORT = "medium"
SETTINGS = load_settings()
META_KEYS = frozenset(['authors', 'date_published', 'filename', 'source', 'source_filetype', 'title', 'url'])


class Extractor:
    """Upload PDF -> call model -> save raw/summary/json."""
    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)

        self._n = 0
        self._sum_sec = 0.0
        self._sum_in = 0
        self._sum_out = 0
        self._sum_tot = 0

    def upload_pdf_get_id(self, pdf_path: Path) -> str:
        with pdf_path.open("rb") as fh:
            f = self.client.files.create(file=fh, purpose="user_data")
        return f.id

    def call_openai_file(self, file_id: str) -> Any:
        messages = [{
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file_id},
                {"type": "input_text", "text": PROMPT_EXTRACT},
            ],
        }]
        return self.client.responses.parse(
            model=MODEL,
            input=messages,
            reasoning={"effort": REASONING_EFFORT},
        )

    def call_openai_text(self, file_text: str) -> Any:
        messages = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": PROMPT_EXTRACT},
                {"type": "input_text", "text": f"\n\nHere is the paper for analysis:\n\n{file_text}"},
            ],
        }]
        return self.client.responses.parse(
            model=MODEL,
            input=messages,
            reasoning={"effort": REASONING_EFFORT},
        )

    def write_outputs(self, out_dir: Path, stem: str, resp: Any, meta: json) -> None:
        raw_path = out_dir / f"{stem}_raw_response.txt"
        json_path = out_dir / f"{stem}.json"
        summary_path = out_dir / f"{stem}_summary.txt"

        safe_write(raw_path, stringify_response(resp))

        output_text = extract_output_text(resp)
        text_part, json_part = split_text_and_json(output_text)

        safe_write(summary_path, text_part or "")

        if json_part is None:
            raise ValueError("No JSON block found in output_text")

        parsed = json.loads(json_part)

        if meta:
            parsed['meta'] = meta

        safe_write(json_path, json.dumps(parsed, ensure_ascii=False, indent=2))

    def _usage_from_resp(self, resp: Any):
        try:
            usage = getattr(resp, "usage", None)
            if isinstance(usage, dict):
                tin = int(usage.get("input_tokens", 0))
                tout = int(usage.get("output_tokens", 0))
                ttot = int(usage.get("total_tokens", tin + tout))
                return tin, tout, ttot
        except Exception:
            pass
        try:
            as_dict = resp.dict() if hasattr(resp, "dict") else {}
            if as_dict:
                u = as_dict.get("usage", {}) or {}
                if u:
                    tin = int(u.get("input_tokens", 0))
                    tout = int(u.get("output_tokens", 0))
                    ttot = int(u.get("total_tokens", tin + tout))
                    return tin, tout, ttot
                out0 = (as_dict.get("output") or [{}])
                if isinstance(out0, list) and out0:
                    u = out0[0].get("usage", {}) or {}
                    tin = int(u.get("input_tokens", 0))
                    tout = int(u.get("output_tokens", 0))
                    ttot = int(u.get("total_tokens", tin + tout))
                    return tin, tout, ttot
        except Exception:
            pass
        return 0, 0, 0

    def _accumulate_and_print(self, label: str, name: str, t0: float, resp: Any) -> None:
        tin, tout, ttot = self._usage_from_resp(resp)
        sec = time.time() - t0
        tqdm.write(
            f"[{label}] {name} | {sec:.2f}s | tokens in/out/total: {tin}/{tout}/{ttot}"
        )
        self._n += 1
        self._sum_sec += sec
        self._sum_in  += tin
        self._sum_out += tout
        self._sum_tot += ttot

    def process_pdf(self, path: Path) -> None:
        out_dir = SETTINGS.paths.output_dir / path.stem
        if out_dir.exists():
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        file_id = self.upload_pdf_get_id(path)
        resp = self.call_openai_file(file_id)
        meta = [{"key": "filename", "value": path.name}]
        self.write_outputs(out_dir, path.stem, resp, meta)
        self._accumulate_and_print("PDF", path.name, t0, resp)

    def process_jsonl(self, path: Path, max_items: Optional[int] = None) -> int:
        processed = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                pbar = tqdm(total=max_items, desc=f"{path.name} (items)", leave=False)
                for idx, line in enumerate(f, start=1):
                    if max_items is not None and processed >= max_items:
                        break
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        paper_json = json.loads(line)
                    except Exception as e:
                        pid = f"{path.stem}__badjson_{idx}"
                        write_failure(SETTINGS.paths.output_dir, pid, e)
                        processed += 1
                        pbar.update(1)
                        continue

                    paper_id = path.stem + "__" + url_to_id(paper_json.get('url', f'line_{idx}'))
                    out_dir = SETTINGS.paths.output_dir / paper_id
                    if out_dir.exists():
                        processed += 1
                        pbar.update(1)
                        continue

                    try:
                        t0 = time.time()
                        resp = self.call_openai_text(paper_json['text'])
                        meta = filter_dict(paper_json, META_KEYS)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        self.write_outputs(out_dir, paper_id, resp, meta)
                        self._accumulate_and_print("JSONL", paper_id, t0, resp)
                    except Exception as e:
                        write_failure(SETTINGS.paths.output_dir, paper_id, e)
                    finally:
                        processed += 1
                        pbar.update(1)
                pbar.close()
        except Exception as e:
            raise RuntimeError(f"Failed to read JSON file '{path}': {e}") from e
        return processed

    def process_dir(self, input_dir: Path, first_n: Optional[int] = None) -> None:
        SETTINGS.paths.output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(input_dir.glob('*.pdf'))
        jsonl_files = list(input_dir.glob('*.jsonl'))

        if first_n:
            pdf_files = pdf_files[:first_n]

        print(f"Found {len(pdf_files)} PDFs and {len(jsonl_files)} JSONLs in {input_dir} to process.")
        print(pdf_files + jsonl_files)

        for file in tqdm(pdf_files, desc="PDFs"):
            try:
                self.process_pdf(file)
            except Exception as e:
                write_failure(SETTINGS.paths.output_dir, file.stem, e)

        json_items_cap = first_n if first_n else None
        if jsonl_files and (json_items_cap is None or json_items_cap > 0):
            for file in tqdm(jsonl_files, desc="JSONL files"):
                try:
                    taken = self.process_jsonl(file, max_items=json_items_cap)
                    if json_items_cap is not None:
                        json_items_cap -= taken
                        if json_items_cap <= 0:
                            break
                except Exception as e:
                    write_failure(SETTINGS.paths.output_dir, file.stem, e)

        if self._n:
            avg_sec = self._sum_sec / self._n
            avg_tok = self._sum_tot / self._n
            print("\n=== Summary ===")
            print(f"Papers processed:     {self._n}")
            print(f"Total time (sec):     {self._sum_sec:.2f}")
            print(f"Avg time/paper (sec): {avg_sec:.2f}")
            print(f"Total tokens:         {self._sum_tot} (in={self._sum_in}, out={self._sum_out})")
            print(f"Avg tokens/paper:     {int(avg_tok)}")
        else:
            print("\nNo papers processed.")


if __name__ == "__main__":
    extractor = Extractor()
    extractor.process_dir(SETTINGS.paths.input_dir, 50)
