from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from config import load_settings
from intervention_graph_creation.src.data_interfaces.dedupe import dedupe_publications
from intervention_graph_creation.src.data_interfaces.models import Publication

SETTINGS = load_settings()
logger = logging.getLogger(__name__)


def _iter_files(p: str) -> List[str]:
    if os.path.isdir(p):
        return [
            os.path.join(p, f)
            for f in os.listdir(p)
            if f.lower().endswith((".json", ".jsonl", ".ndjson"))
        ]
    return [p]


def _iter_records(file_path: str):
    if file_path.lower().endswith((".jsonl", ".ndjson")):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict):
            for key in ("data", "items", "records", "results"):
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        yield item


def load_publications_from_local_ard(
    path_or_dir: str, strict: bool = False
) -> List[Publication]:
    publications: List[Publication] = []

    for fp in _iter_files(path_or_dir):
        for rec in _iter_records(fp):
            if not isinstance(rec, dict):
                if strict:
                    raise ValueError(
                        f"Invalid record (expected dict) in {fp}: {type(rec)}"
                    )
                logger.warning("Skipping non-dict record in %s: %r", fp, type(rec))
                continue

            title = rec.get("title")
            if title is None:
                title = ""
            authors = rec.get("authors")
            if authors is None:
                authors = []
            date = rec.get("date_published")
            if date is None:
                date = "unknown"
            text = rec.get("text")
            if text is None:
                text = ""

            # Skip publications with empty text
            if not str(text).strip():
                logger.debug("Skipping publication with empty text: %s", title)
                continue

            url = rec.get("url")
            if url is None:
                url = None

            publications.append(
                Publication(
                    title=str(title),
                    authors=authors,
                    date_published=str(date),
                    text=text,
                    abstract=None,
                    url=str(url) if isinstance(url, str) else None,
                )
            )

    return publications


def load_publications_from_hf_ard(
    repo_id: str = "StampyAI/alignment-research-dataset",
    revision: Optional[str] = None,
    local_dir: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    dedupe: bool = False,
) -> List[Publication]:
    from huggingface_hub import snapshot_download  # type: ignore

    if allow_patterns is None:
        allow_patterns = ["*.json", "*.jsonl", "*.ndjson"]

    import os

    target_dir = local_dir or os.path.join(os.getcwd(), "alignment-research-dataset")

    repo_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=allow_patterns,
    )

    json_files: List[str] = []
    for root, _dirs, files in os.walk(repo_path):
        for fn in files:
            if fn.lower().endswith((".json", ".jsonl", ".ndjson")):
                json_files.append(os.path.join(root, fn))

    if not json_files:
        raise RuntimeError(
            f"No JSON/JSONL files found in downloaded dataset at {repo_path}"
        )

    publications: List[Publication] = []
    for path in json_files:
        publications.extend(load_publications_from_local_ard(path, strict=False))
    if dedupe:
        publications = dedupe_publications(publications, prefer="latest_date")
    return publications


def main():
    publications = load_publications_from_hf_ard(
        repo_id="StampyAI/alignment-research-dataset",
        allow_patterns=["arxiv/*"],
        local_dir=str(SETTINGS.paths.input_dir),
        dedupe=True,
    )

    print(f"Downloaded publications: {len(publications)}\n")

    for pub in publications[:3]:
        print("=" * 80)
        print(f"Title: {pub.title}")
        print(f"Authors: {', '.join(pub.authors)}")
        print(f"Date: {pub.date_published}")
        print(f"URL: {pub.url}")
        print(f"Text (first 200 chars): {pub.text[:200]!r}\n")


if __name__ == "__main__":
    main()
