import json
import traceback
import re
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import urlparse


FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)


def extract_output_text(resp: Any) -> str:
    """Return assistant's textual output, tolerating various SDK shapes."""
    value = getattr(resp, "output_text", resp["output_text"] if "output_text" in resp else "")
    return value if isinstance(value, str) else str(value or "")


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


def stringify_response(resp: Any) -> str:
    """Serialize raw SDK response to string for diagnostics."""
    try:
        return resp.model_dump_json() if hasattr(resp, "model_dump_json") else str(resp)
    except Exception:
        return str(resp)
    

def url_to_id(url: str) -> str:
    parsed = urlparse(url)

    netloc = parsed.netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]

    raw = netloc + parsed.path
    raw = raw.lower()

    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def write_failure(out_dir: Path, pdf_name: str, e: Exception) -> None:
    """Uniformly persist failure diagnostics for a single PDF."""
    stem = Path(pdf_name).stem
    out_dir = out_dir / "error" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{stem}_raw_response.txt"
    json_path = out_dir / f"{stem}.json"
    summary_path = out_dir / f"{stem}_summary.txt"

    diag = (
        f"Processing failed for {pdf_name}\n"
        f"{type(e).__name__}: {e}\n\n"
    )
    print(diag)
    diag += f"Traceback:\n{traceback.format_exc()}"
    safe_write(raw_path, diag)
    safe_write(summary_path, "")
    safe_write(json_path, json.dumps(
        {"error": f"{type(e).__name__}: {str(e)}"},
        ensure_ascii=False, indent=2
    ))
