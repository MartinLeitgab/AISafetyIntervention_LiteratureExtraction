"""
Grabs the files that had errors during extraction validation
and puts them in a directory for retrying.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from fire import Fire


def main(last_processed_dir: str, last_output_dir: str, new_folder_for_retry: str):
    errors_path = Path(last_output_dir) / "errors.json"
    last_proccessed_dir_path = Path(last_processed_dir)

    new_folder_for_retry_path = Path(new_folder_for_retry)
    new_folder_for_retry_path.mkdir(parents=True, exist_ok=True)
    if os.path.getsize(errors_path) == 0:
        print(f"No errors found in {errors_path}!")
        return
    try:
        with open(errors_path, "r") as f:
            error_dictionary: Dict[str, Any] = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {errors_path}.")
        return
    files_with_errrors = set(error_dictionary.keys())
    for file in files_with_errrors:
        folder_to_copy = last_proccessed_dir_path / file
        if not folder_to_copy.exists():
            print(f"Warning: Source folder {folder_to_copy} does not exist. Skipping.")
            continue
        destination_folder = new_folder_for_retry_path / file
        shutil.copytree(folder_to_copy, destination_folder)


if __name__ == "__main__":
    Fire(main)
