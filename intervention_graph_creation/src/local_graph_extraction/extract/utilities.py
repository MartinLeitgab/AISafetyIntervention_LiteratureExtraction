import json
import traceback
import logging
import shutil
import re
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import urlparse


FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)
logger = logging.getLogger("intervention.extractor")


def filter_dict(d: dict, keys: set) -> list[dict]:
    """
    Return a flat list of {"key": key, "value": value}
    for the specified keys that exist in the original dict.
    """
    return [{"key": k, "value": d[k]} for k in keys if k in d]


def safe_write(path: Path, content: str) -> None:
    """Create parents (if needed) and write UTF-8 text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def split_text_and_json(s: str) -> Tuple[str, Optional[str]]:
    """Extract a JSON object (fenced or inline) and return (remaining_text, json_str|None)."""
    s = s or ""
    m = FENCE_RE.search(s)
    if m:
        return (s[:m.start()] + s[m.end():]).strip(), m.group(1).strip()

    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and i < j:
        candidate = s[i:j + 1].strip()
        # Let json library decide validity; JSONDecodeError will bubble up
        json.loads(candidate)
        return (s[:i] + s[j + 1:]).strip(), candidate

    return s.strip(), None


def url_to_id(url: str) -> str:
    parsed = urlparse(url)

    netloc = parsed.netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]

    raw = netloc + parsed.path
    raw = raw.lower()

    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def write_failure(base_dir: Path, paper_id: str, err: Exception) -> None:
    """
    Save failure information for a paper into the extraction_error directory.
    - Moves any already generated files (raw_response, summary, json) from output/.
    - Always writes error.txt with exception type, message, and traceback.
    - Removes empty paper folder from output/.
    """
    err_dir = base_dir / "extraction_error" / paper_id
    err_dir.mkdir(parents=True, exist_ok=True)

    orig_dir = base_dir / paper_id
    candidate_files = [
        orig_dir / f"{paper_id}_raw_response.txt",
        orig_dir / f"{paper_id}_summary.txt",
        orig_dir / f"{paper_id}.json",
    ]

    for src in candidate_files:
        if src.exists():
            dst = err_dir / src.name
            try:
                shutil.move(str(src), str(dst))
            except Exception as move_err:
                logger.warning("Could not move %s → %s: %s", src, dst, move_err)

    error_file = err_dir / "error.txt"
    diag = (
        f"❌ Processing failed for {paper_id}\n"
        f"{type(err).__name__}: {err}\n\n"
        f"Traceback:\n{traceback.format_exc()}"
    )
    error_file.write_text(diag, encoding="utf-8")

    try:
        if orig_dir.exists() and not any(orig_dir.iterdir()):
            orig_dir.rmdir()
            logger.info("🧹 Removed empty output dir: %s", orig_dir)
    except Exception as cleanup_err:
        logger.warning("Could not clean empty dir %s: %s", orig_dir, cleanup_err)

    logger.error("❌ Failed to process %s | Traceback saved to %s", paper_id, error_file)
