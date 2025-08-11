from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def normalize_title(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\u2010-\u2015\-_:;,.!?]+", "", s)  # Remove special characters for normalization
    return s


def extract_arxiv_id_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(?P<id>\d{4}\.\d{5}(?:v\d+)?)", url)
    return m.group("id") if m else None


def parse_arxiv_id_from_filename(filename: str) -> Optional[str]:
    """Extract an arXiv id (with version if present) from a filename like '2311.07590v4.pdf'."""
    m = re.search(r"(?P<id>\d{4}\.\d{5}(?:v\d+)?)(?:\.pdf)$", filename)
    return m.group("id") if m else None


def split_references(text: str) -> Tuple[str, Optional[str]]:
    """Split a document's text into main content and references/bibliography if detected. Look for a heading like "References", "Bibliography", "Works Cited".

    Returns a tuple (content, references). If no references are detected, references is None.
    """
    if not text:
        return "", None

    s = text.replace("\r\n", "\n").replace("\r", "\n")

    keywords = [
        "References",
        "Bibliography",
        "Works Cited",
        "Literature Cited",
        "References and Notes",
        "Reference list",
        "Reference List"
    ]
    # Include both original and uppercase versions
    all_keywords = keywords + [k.upper() for k in keywords]
    keywords_pattern = "|".join(re.escape(k) for k in all_keywords)
    heading_re = re.compile(
        rf"^\s*(?:\d+\s+)?({keywords_pattern})\s*$",
        re.MULTILINE,
    )
    m = heading_re.search(s)
    if m:
        split_at = m.start()
        content = s[:split_at].rstrip()
        refs = s[split_at:].lstrip()
        return content, refs

    return s, None
