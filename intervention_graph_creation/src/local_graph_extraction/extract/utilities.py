import json
import logging
import re
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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
        return (s[: m.start()] + s[m.end() :]).strip(), m.group(1).strip()

    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and i < j:
        candidate = s[i : j + 1].strip()
        # Let json library decide validity; JSONDecodeError will bubble up
        json.loads(candidate)
        return (s[:i] + s[j + 1 :]).strip(), candidate

    return s.strip(), None


def url_to_id(url: str) -> str:
    parsed = urlparse(url)

    netloc = parsed.netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]

    raw = netloc + parsed.path
    raw = raw.lower()

    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def _utc_now_iso() -> str:
    # Matches your existing Z-suffixed UTC format
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def write_failure(
    output_root: Path,
    error_root: Path,
    paper_id: str,
    *,
    err: Optional[BaseException] = None,
    error_type: Optional[str] = None,
    message: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Move all artifacts for a paper from `output_root/<paper_id>` to
    `error_root/<paper_id>`, delete the original folder, and add `error.txt`.

    Parameters
    ----------
    output_root : Path
        Root of successful artifacts (e.g., SETTINGS.paths.output_dir)
    error_root : Path
        Root of error artifacts (e.g., SETTINGS.paths.extraction_error_dir)
    paper_id : str
        Paper identifier used as the subfolder name under both roots.
    err : Exception | None
        An exception object (if you caught one).
    error_type : str | None
        High-level category/stage (e.g., "create_batch_requests", "parsing", "network").
    message : str | None
        Optional human-readable message if you don't have an exception or want to add detail.
    context : dict | None
        Extra diagnostic fields to dump (request IDs, HTTP status, meta, etc.).
    logger : logging.Logger | None
        Optional logger; if provided we log progress/warnings.

    Returns
    -------
    Path
        The path to the error directory for this paper: error_root / paper_id
    """

    log = logger or logging.getLogger(__name__)

    # Normalize folder names and ensure destination exists
    clean_id = paper_id.split(".")[0]
    src_dir = output_root / clean_id
    dst_dir = error_root / clean_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Helper: move a file or directory into dst_dir, resolving name collisions
    def _move_into_dst(p: Path) -> None:
        target = dst_dir / p.name
        if target.exists():
            # Avoid collisions by suffixing with a timestamp
            stamped = dst_dir / f"{p.stem}_{int(datetime.now().timestamp())}{p.suffix}"
            try:
                shutil.move(str(p), str(stamped))
            except Exception as move_err:
                log.warning("Could not move %s → %s: %s", p, stamped, move_err)
        else:
            try:
                shutil.move(str(p), str(target))
            except Exception as move_err:
                log.warning("Could not move %s → %s: %s", p, target, move_err)

    # 1) Move everything currently in output/<paper_id> to error/<paper_id>
    if src_dir.exists() and src_dir.is_dir():
        for p in src_dir.iterdir():
            _move_into_dst(p)
        # 2) Remove the now-empty source directory
        try:
            shutil.rmtree(src_dir, ignore_errors=True)
        except Exception as rm_err:
            log.warning("Could not delete source dir %s: %s", src_dir, rm_err)
    else:
        log.info("No source dir to move (skipping): %s", src_dir)

    # 3) Write error.txt
    error_file = dst_dir / "error.txt"

    diag: Dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "paper_id": clean_id,
        "error_type": error_type or (type(err).__name__ if err else None),
        "message": message or (str(err) if err else None),
    }

    # Attach context (request IDs, HTTP status, meta, stage, etc.)
    if context:
        try:
            # Make sure it's JSON-serializable; fall back to str() if not
            json.dumps(context)
            diag["context"] = context
        except Exception:
            diag["context"] = {
                "note": "context not JSON-serializable",
                "repr": repr(context),
            }

    # Include traceback if an exception is provided
    tb = traceback.format_exc() if err is not None else None
    if tb and tb.strip() != "NoneType: None\n":
        diag["traceback"] = tb

    try:
        error_file.write_text(
            json.dumps(diag, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as write_err:
        # As a last resort, write a plaintext version
        log.warning(
            "Could not write JSON error.txt, falling back to plaintext: %s", write_err
        )
        text = (
            f"timestamp: {diag['timestamp']}\n"
            f"paper_id: {clean_id}\n"
            f"error_type: {diag.get('error_type')}\n"
            f"message: {diag.get('message')}\n"
            f"context: {repr(diag.get('context'))}\n\n"
            f"traceback:\n{diag.get('traceback') or ''}\n"
        )
        error_file.write_text(text, encoding="utf-8")

    log.error("❌ Failure recorded for %s | Details: %s", clean_id, error_file)
    return dst_dir
