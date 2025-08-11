from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Publication:
    title: str
    authors: List[str]
    date_published: str
    text: str
    references: Optional[str] = None
    abstract: Optional[str] = None
    url: Optional[str] = None

    def __post_init__(self) -> None:
        # If references not provided, attempt to split them from the text heuristically
        if self.references is None and isinstance(self.text, str) and self.text:
            try:
                from .utils import split_references  # local import to avoid circular deps

                content, refs = split_references(self.text)
                self.text = content
                self.references = refs
            except Exception:
                # On any failure, keep original text and None references
                pass
