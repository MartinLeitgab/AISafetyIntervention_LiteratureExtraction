from __future__ import annotations

import logging
import re
from typing import Optional

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
