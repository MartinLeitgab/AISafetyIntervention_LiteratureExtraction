import json
from pathlib import Path
from typing import  List, Tuple
from extraction_validator.judge import get_by_file_url_to_text_map
from fire import Fire # type: ignore

def main(
    ard_dir: str,
    processed_dir: str,
    output_dir: str,
):
    base = Path(processed_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Directory not found or not a directory: {base}")

    json_files = list(base.glob("*.json"))

    json_files = [str(f) for f in json_files if f.is_file() and not f.name.endswith("errors.json") and not f.name.endswith("summary.json")]

    by_file_url_to_text_map = get_by_file_url_to_text_map(ard_dir)
    error_files: List[Tuple[str, str]] = []


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
        judge_str = open(json_file, "r").read()
        url = json.loads(judge_str)["url"]
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
        with open(
            Path(output_dir) / f"{ard_file_source}__{paper_id}.json", "w"
        ) as f:
            f.write(f"""Original Text:
{original_text}
Judge Output:
{judge_str}
""")
if __name__ == "__main__":
    Fire(main)