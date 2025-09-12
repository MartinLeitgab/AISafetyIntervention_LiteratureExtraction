from __future__ import annotations

import logging
import os
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
from pypdf import PdfReader

from .utils import parse_arxiv_id_from_filename
from .models import Publication

logger = logging.getLogger(__name__)

_ARXIV_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _fetch_arxiv_metadata(
    arxiv_id: str,
) -> Tuple[Optional[str], List[str], Optional[str], Optional[str]]:
    """Fetch (title, authors, published, summary) for a given arXiv id using the arXiv Atom API."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = resp.read()

    root = ET.fromstring(data)
    entry = root.find("atom:entry", _ARXIV_ATOM_NS)
    if entry is None:
        raise ValueError(f"No arXiv entry found for id {arxiv_id}")

    title_el = entry.find("atom:title", _ARXIV_ATOM_NS)
    summary_el = entry.find("atom:summary", _ARXIV_ATOM_NS)
    published_el = entry.find("atom:published", _ARXIV_ATOM_NS)
    authors = [
        a.findtext("atom:name", default="", namespaces=_ARXIV_ATOM_NS) or ""
        for a in entry.findall("atom:author", _ARXIV_ATOM_NS)
    ]
    authors = [a for a in authors if a]

    title = title_el.text.strip() if title_el is not None and title_el.text else None
    summary = (
        summary_el.text.strip() if summary_el is not None and summary_el.text else None
    )
    published = (
        published_el.text.strip()
        if published_el is not None and published_el.text
        else None
    )
    return title, authors, published, summary


def _extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pypdf."""
    reader = PdfReader(file_path)
    pages: List[str] = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages).strip()


def load_publications_from_folder(folder_path: str) -> List[Publication]:
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    publications: List[Publication] = []

    for name in sorted(os.listdir(folder_path)):
        if not name.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(folder_path, name)

        arxiv_id = parse_arxiv_id_from_filename(name)
        if not arxiv_id:
            raise ValueError(f"Filename does not contain a valid arXiv id: {name}")

        title, authors, published, summary = _fetch_arxiv_metadata(arxiv_id)
        if not title:
            raise ValueError(f"Missing title from arXiv metadata for id {arxiv_id}")
        if not isinstance(published, str) or not published:
            raise ValueError(
                f"Missing published date from arXiv metadata for id {arxiv_id}"
            )

        text = _extract_text_from_pdf(pdf_path)

        url = f"https://arxiv.org/abs/{arxiv_id}"

        pub = Publication(
            title=title,
            authors=authors,
            date_published=published,
            text=text,
            abstract=summary,
            url=url,
            pdf_file_path=pdf_path,
        )
        publications.append(pub)

    return publications


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _download_file(url: str, dest_path: str, timeout: int = 30) -> None:
    tmp_path = dest_path + ".part"
    with (
        urllib.request.urlopen(url, timeout=timeout) as resp,
        open(tmp_path, "wb") as out,
    ):
        out.write(resp.read())
    os.replace(tmp_path, dest_path)


def load_publications_from_arxiv_ids(
    arxiv_ids: List[str],
    download_pdf: bool = True,
    pdf_dir: Optional[str] = None,
) -> List[Publication]:
    publications: List[Publication] = []

    if download_pdf:
        pdf_dir = pdf_dir or os.path.join(os.getcwd(), "arxiv_pdfs")
        ensure_dir(pdf_dir)

    for arxiv_id in arxiv_ids:
        t, a, p, s = _fetch_arxiv_metadata(arxiv_id)
        if not t:
            raise ValueError(f"Missing title from arXiv metadata for id {arxiv_id}")
        if not p:
            raise ValueError(
                f"Missing published date from arXiv metadata for id {arxiv_id}"
            )

        text = ""
        pdf_path = None
        if download_pdf:
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            filename = url.rsplit("/", 1)[-1]
            pdf_path = os.path.join(pdf_dir, filename)
            if not os.path.exists(pdf_path):
                _download_file(url, pdf_path)
            text = _extract_text_from_pdf(pdf_path)

        publications.append(
            Publication(
                title=t,
                authors=a or [],
                date_published=p,
                text=text,
                abstract=s,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_file_path=pdf_path,
            )
        )

    return publications
