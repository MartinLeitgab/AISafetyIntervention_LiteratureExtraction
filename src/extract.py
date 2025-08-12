import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

try:
    from src.prompts import EXTRACTION_PROMPT_TEMPLATE, OutputSchema
except ImportError:
    from prompts import EXTRACTION_PROMPT_TEMPLATE, OutputSchema


load_dotenv()

MODEL = "gpt-5-2025-08-07"
# https://platform.openai.com/docs/guides/reasoning
# REASONING_EFFORT = "high"
REASONING_EFFORT = "minimal"
INPUT_DIR = Path("inputdata_development_paper_set")
OUTPUT_DIR = Path("output")


class Extractor:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=api_key)

    def ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def upload_pdf_get_id(self, pdf_path: Path) -> str:
        with pdf_path.open("rb") as fh:
            f = self.client.files.create(file=fh, purpose="user_data")
        return f.id

    def build_user_input(self, file_id: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": EXTRACTION_PROMPT_TEMPLATE},
                ],
            }
        ]

    def call_openai(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # https://platform.openai.com/docs/guides/structured-outputs
        response = self.client.responses.parse(
            model=MODEL,
            input=messages,
            reasoning={
                "effort": REASONING_EFFORT,
                # "summary": "auto",
            },
            text_format=OutputSchema,
        )
        return response

    def process_dir(self, input_dir: Path, first_n: Optional[int] = None) -> None:
        self.ensure_dir(OUTPUT_DIR)

        pdf_paths = sorted(input_dir.glob("*.pdf"))
        if first_n:
            pdf_paths = pdf_paths[:first_n]
        processed_papers = set([x.stem for x in OUTPUT_DIR.glob("*.json")])
        for pdf_path in tqdm(pdf_paths):
            if pdf_path.stem in processed_papers:
                continue

            file_id = self.upload_pdf_get_id(pdf_path)
            messages = self.build_user_input(file_id)
            response = self.call_openai(messages)

            raw_path = OUTPUT_DIR / f"{pdf_path.stem}_raw_response.json"
            with raw_path.open("w", encoding="utf-8") as f:
                f.write(response.model_dump_json())

            output_parsed = response.output_parsed
            output_parsed_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
            with output_parsed_path.open("w", encoding="utf-8") as f:
                f.write(output_parsed.model_dump_json(indent=2))


if __name__ == "__main__":
    extractor = Extractor()
    extractor.process_dir(INPUT_DIR, 5)
