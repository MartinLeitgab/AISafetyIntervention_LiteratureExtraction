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
    pdf_file_path: Optional[str] = None

    def __post_init__(self) -> None:
        # If references not provided, attempt to split them from the text heuristically
        if self.references is None and isinstance(self.text, str) and self.text:
            try:
                from .utils import (  # local import to avoid circular deps
                    split_references,
                )

                content, refs = split_references(self.text)
                self.text = content
                self.references = refs
            except Exception:
                # On any failure, keep original text and None references
                pass

        # If PDF file path provided, remove the references from the file
        if self.pdf_file_path and self.references:
            try:
                from .utils import (  # local import to avoid circular deps
                    remove_references_from_pdf,
                )

                remove_references_from_pdf(self.pdf_file_path)
            except Exception as e:
                # On any failure, log the error but don't crash
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to modify PDF {self.pdf_file_path}: {e}"
                )
