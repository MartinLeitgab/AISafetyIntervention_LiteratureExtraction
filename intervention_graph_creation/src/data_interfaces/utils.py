from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def normalize_title(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(
        r"[\u2010-\u2015\-_:;,.!?]+", "", s
    )  # Remove special characters for normalization
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
        "Reference List",
    ]
    keywords_pattern = "|".join(re.escape(k) for k in keywords)
    heading_re = re.compile(
        # All keywords followed by any amount of spaces and then at least one tab or newline
        rf"^.*({keywords_pattern})\s*(?=[\t\n])",
        re.MULTILINE | re.IGNORECASE,
    )
    m = heading_re.search(s)
    if m:
        split_at = m.start()
        content = s[:split_at].rstrip()
        refs = s[split_at:].lstrip()
        return content, refs

    return s, None


def remove_references_from_pdf(pdf_file_path: str) -> None:
    """Remove references section from the PDF file by creating a new PDF with only the content before references."""
    if not pdf_file_path:
        return

    import os
    from pypdf import PdfReader, PdfWriter

    # Read the original PDF
    reader = PdfReader(pdf_file_path)

    # Extract all text to find references section
    full_text = ""
    page_texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_texts.append(page_text)
        full_text += page_text + "\n"

    # Use the same logic as split_references to find where references start
    from .utils import split_references

    content_before_refs, _ = split_references(full_text)

    if not content_before_refs or content_before_refs == full_text:
        # No references found or references are the entire content, don't modify
        return

    # Find which page the references section starts on
    current_text = ""
    split_page = len(page_texts)  # Default to last page if not found

    for i, page_text in enumerate(page_texts):
        if len(current_text + page_text) >= len(content_before_refs):
            # References section starts somewhere on this page or later
            split_page = i + 1  # Keep pages up to and including this one
            break
        current_text += page_text + "\n"

    # Create new PDF with only pages before references
    writer = PdfWriter()

    # Add pages up to where references start
    for i in range(min(split_page, len(reader.pages))):
        writer.add_page(reader.pages[i])

    # Write the modified PDF back to the same path
    backup_path = pdf_file_path + ".backup"
    os.rename(pdf_file_path, backup_path)

    try:
        with open(pdf_file_path, "wb") as output_file:
            writer.write(output_file)
    except Exception:
        # If writing fails, restore the backup
        os.rename(backup_path, pdf_file_path)
        raise
    else:
        # If successful, remove the backup
        os.remove(backup_path)
