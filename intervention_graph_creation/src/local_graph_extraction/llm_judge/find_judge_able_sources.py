"""Not all sources have recoverable errors that the judge can fix.
This module identifies sources that are likely to have judge-able errors."""

import json
import shutil
from pathlib import Path
from typing import List

from fire import Fire
from tqdm import tqdm

from intervention_graph_creation.src.local_graph_extraction.core.paper_schema import (
    PaperSchema,
)
from intervention_graph_creation.src.local_graph_extraction.llm_judge.judge import (
    find_url,
    get_all_json_files,
    get_by_file_url_to_text_map,
)


def main(
    processed_dir: str,
    ard_dir: str,
    output_dir: str,
) -> None:
    base = Path(processed_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Directory not found or not a directory: {base}")
    json_files = get_all_json_files(base)

    by_file_url_to_text_map = get_by_file_url_to_text_map(ard_dir)

    source: List[str] = []

    for json_file in tqdm(json_files):
        the_split = Path(json_file).stem.split("__")
        if len(the_split) != 2:
            continue
        [ard_file_source, _paper_id] = the_split
        url_to_text_map = by_file_url_to_text_map.get(ard_file_source)

        if url_to_text_map is None:
            continue
        try:
            with open(json_file, "r") as f:
                kg_output = PaperSchema.model_validate_json(f.read())
                url = find_url(kg_output)
                if url is None:
                    continue
                original_text = url_to_text_map.get(url)
                if original_text is None or original_text.strip() == "":
                    continue
                if len(kg_output.nodes) == 0:
                    continue
                source.append(json_file)

        except Exception as _:
            continue
    print(f"Found {len(source)}/{len(json_files)},")
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "potential_judge_able_sources.json"
    with open(output_file, "w") as f:
        json.dump(source, f)
    for a_source in source:
        a_source_path = Path(a_source)
        folder = a_source_path.parent
        shutil.copytree(folder, output_path / folder.name)


if __name__ == "__main__":
    Fire(main)
