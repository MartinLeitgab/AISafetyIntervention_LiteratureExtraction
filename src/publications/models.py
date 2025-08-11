from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Publication:
    title: str
    authors: List[str]
    date_published: str
    text: str
    abstract: Optional[str] = None
    url: Optional[str] = None
