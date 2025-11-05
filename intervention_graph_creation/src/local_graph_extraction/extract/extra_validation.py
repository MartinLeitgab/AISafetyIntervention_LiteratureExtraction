"""Validates knowledge graph extraction output folders and moves invalid ones to an error directory.

Key validation criteria:
- Folder contains ≥3 files
- Contains a JSON file named {folder_name}.json
- JSON has non-empty nodes and edges arrays
- Folder is not already in error directories (extraction/graph/embeddings errors)

Workflow:
1. Iterates through all subfolders in output_dir
2. Validates each folder against the criteria
3. Moves invalid folders to extraction_error_dir

Purpose: Cleans up the output directory by segregating corrupted or incomplete extraction results.
"""

import json
import shutil
from pathlib import Path

from config import load_settings

SETTINGS = load_settings()

ROOT_DIR = SETTINGS.paths.output_dir
ERROR_DIR = SETTINGS.paths.extraction_error_dir
ERROR_DIR.mkdir(exist_ok=True)

SKIP_DIRS = {
    SETTINGS.paths.extraction_error_dir,
    SETTINGS.paths.graph_error_dir,
    SETTINGS.paths.embeddings_error_dir,
}


def _is_relative_to(a: Path, b: Path) -> bool:
    try:
        return a.resolve().is_relative_to(b.resolve())
    except AttributeError:
        try:
            a.resolve().relative_to(b.resolve())
            return True
        except Exception:
            return False


def _in_skip_dirs(folder: Path) -> bool:
    for skip in SKIP_DIRS:
        if skip and _is_relative_to(folder, skip):
            return True
    return False


def validate_json(json_path: Path) -> bool:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not isinstance(nodes, list) or not nodes:
        return False
    if not isinstance(edges, list) or not edges:
        return False
    return True


def validate_folder(folder: Path) -> bool:
    if _in_skip_dirs(folder):
        return True
    try:
        files = list(folder.iterdir())
    except Exception:
        return False
    if len(files) < 3:
        return False
    json_file = folder / f"{folder.name}.json"
    if not json_file.exists():
        return False
    if not validate_json(json_file):
        return False
    return True


def run_validation():
    for subfolder in ROOT_DIR.iterdir():
        if subfolder.is_dir():
            if not validate_folder(subfolder):
                target = ERROR_DIR / subfolder.name
                try:
                    shutil.move(str(subfolder), str(target))
                except Exception:
                    pass


if __name__ == "__main__":
    run_validation()
