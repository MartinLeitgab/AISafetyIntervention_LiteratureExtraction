from __future__ import annotations

import logging
from typing import Dict, List
from datetime import datetime

from .models import Publication
from .utils import extract_arxiv_id_from_url, normalize_title

logger = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def dedupe_publications(publications: List[Publication], prefer: str = "latest_date") -> List[Publication]:
    """Deduplicate publications by arXiv ID if present, else by (normalized title, authors set).

    prefer:
      - "latest_date": keep the record with the most recent date
      - "longest_text": keep the record with the longest text
    """
    best: Dict[str, Publication] = {}

    def make_key(p: Publication) -> str:
        arx = extract_arxiv_id_from_url(p.url)
        if arx:
            return f"arxiv:{arx.lower()}"
        authors_norm = tuple(sorted(a.strip().lower() for a in p.authors))
        return f"title:{normalize_title(p.title)}|authors:{authors_norm}"

    def better(p_new: Publication, p_old: Publication) -> bool:
        if prefer == "longest_text":
            return len(p_new.text or "") > len(p_old.text or "")
        unknown_new = p_new.date_published == "unknown"
        unknown_old = p_old.date_published == "unknown"

        if unknown_new != unknown_old:
            # Prefer known date over unknown
            return not unknown_new

        if not unknown_new and not unknown_old:
            dt_new = datetime.strptime(p_new.date_published, DATE_FORMAT)
            dt_old = datetime.strptime(p_old.date_published, DATE_FORMAT)
            if dt_new != dt_old:
                return dt_new > dt_old
        return len(p_new.text or "") > len(p_old.text or "")

    for p in publications:
        k = make_key(p)
        if k not in best or better(p, best[k]):
            best[k] = p

    return list(best.values())
