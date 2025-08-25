import os
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from intervention_graph_creation.src.prompt.final_primary_prompt import PROMPT_EXTRACT

load_dotenv()

MODEL = "gpt-5-2025-08-07"
REASONING_EFFORT = "minimal"
FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)


def safe_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def split_text_and_json(s: str) -> Tuple[str, Optional[str]]:
    m = FENCE_RE.search(s)
    if m:
        js = m.group(1).strip()
        txt = (s[:m.start()] + s[m.end():]).strip()
        return txt, js
    i, j = s.find('{'), s.rfind('}')
    if i != -1 and j != -1 and i < j:
        candidate = s[i:j + 1].strip()
        try:
            json.loads(candidate)
            txt = (s[:i] + s[j + 1:]).strip()
            return txt, candidate
        except Exception:
            pass
    return s.strip(), None


class Extractor:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=api_key)

    def upload_pdf_get_id(self, pdf_path: Path) -> str:
        with pdf_path.open("rb") as fh:
            f = self.client.files.create(file=fh, purpose="user_data")
        return f.id

    def call_openai(self, messages: List[Dict[str, Any]]) -> Any:
        response = self.client.responses.parse(
            model=MODEL,
            input=messages,
            reasoning={"effort": REASONING_EFFORT}
        )
        return response

    def process_dir(self, input_dir: Path, first_n: Optional[int] = None) -> None:
        self.ensure_dir(OUTPUT_DIR)
        pdf_paths = sorted(input_dir.absolute().glob("*.pdf"))
        if first_n:
            pdf_paths = pdf_paths[:first_n]
        for pdf_path in tqdm(pdf_paths):
            out_dir = OUTPUT_DIR / pdf_path.stem
            if out_dir.exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            raw_path = out_dir / f"{pdf_path.stem}_raw_response.txt"
            json_path = out_dir / f"{pdf_path.stem}.json"
            summary_path = out_dir / f"{pdf_path.stem}_summary.txt"
            try:
                file_id = self.upload_pdf_get_id(pdf_path)
                messages = self.build_user_input(file_id)
                response = self.call_openai(messages)
                try:
                    raw_response = response.model_dump_json() if hasattr(response, "model_dump_json") else str(response)
                except Exception:
                    raw_response = str(response)
                safe_write(raw_path, raw_response)
                try:
                    output_text = getattr(response, "output_text", "")
                    if not isinstance(output_text, str):
                        output_text = str(output_text or "")
                except Exception:
                    output_text = ""
                text_part, json_part = split_text_and_json(output_text or "")
                safe_write(summary_path, text_part or "")
                if json_part:
                    try:
                        parsed = json.loads(json_part)
                        safe_write(json_path, json.dumps(parsed, ensure_ascii=False, indent=2))
                    except Exception as e:
                        err = {"error": f"Invalid JSON extracted: {str(e)}"}
                        safe_write(json_path, json.dumps(err, ensure_ascii=False, indent=2))
                else:
                    err = {"error": "No JSON block found in output_text"}
                    safe_write(json_path, json.dumps(err, ensure_ascii=False, indent=2))
            except Exception as e:
                err_text = f"Processing failed for {pdf_path.name}\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                safe_write(raw_path, err_text)
                safe_write(summary_path, "")
                safe_write(json_path, json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))

    @staticmethod
    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_user_input(file_id: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": PROMPT_EXTRACT},
                ],
            }
        ]


if __name__ == "__main__":
    extractor = Extractor()
    extractor.process_dir(INPUT_DIR, 2)
