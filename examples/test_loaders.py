import os
import traceback
import sys
import logging

logging.basicConfig(level=logging.ERROR)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INPUT_PDF_DIR = os.path.join(REPO_ROOT, "inputdata_development_paper_set")

# Ensure 'src' is on sys.path so 'publications' package is importable when running directly
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from publications import (
    load_publications_from_folder,
    load_publications_from_hf_ard,
    load_publications_from_arxiv_ids,
)
from publications.utils import parse_arxiv_id_from_filename


def test_load_from_folder():
    print("=== Test: load_publications_from_folder ===")
    pubs = load_publications_from_folder(INPUT_PDF_DIR)
    print(f"Loaded {len(pubs)} publications from folder")
    for p in pubs[:3]:
        refs_len = len(p.references) if p.references else 0
        print(f"- {p.title} | authors={len(p.authors)} | date={p.date_published} | abstract_len={len(p.abstract)} | text_len={len(p.text)} | refs_len={refs_len}")


def test_load_from_hf():
    print("=== Test: load_publications_from_hf_ard ===")
    try:
        pubs = load_publications_from_hf_ard(
            repo_id="StampyAI/alignment-research-dataset",
            local_dir=os.path.join(REPO_ROOT, "alignment-research-dataset"),
            dedupe=True,
        )
        print(f"Loaded {len(pubs)} publications from HF ARD")
        for p in pubs[:3]:
            refs_len = len(p.references) if p.references else 0
            print(f"- {p.title} | authors={p.authors} | date={p.date_published} | text_len={len(p.text)} | refs_len={refs_len}")
    except Exception:
        print("HF ARD test failed:")
        traceback.print_exc()


def test_load_from_arxiv_ids():
    print("=== Test: load_publications_from_arxiv_ids ===")
    # Derive a couple of arXiv IDs from local PDFs
    names = [n for n in os.listdir(INPUT_PDF_DIR) if n.lower().endswith(".pdf")]
    ids = []
    for n in sorted(names)[:3]:
        arxiv_id = parse_arxiv_id_from_filename(n)
        if arxiv_id:
            ids.append(arxiv_id)
    if not ids:
        raise RuntimeError("No arXiv IDs derived from local PDFs")
    pubs = load_publications_from_arxiv_ids(
        ids, download_pdf=True, pdf_dir=os.path.join(REPO_ROOT, "arxiv_pdfs_test")
    )
    print(f"Loaded {len(pubs)} publications from arXiv IDs: {ids}")
    for p in pubs:
        refs_len = len(p.references) if p.references else 0
        print(f"- {p.title} | authors={len(p.authors)} | date={p.date_published} | abstract_len={len(p.abstract)} | text_len={len(p.text)} | refs_len={refs_len}")


if __name__ == "__main__":
    # Run tests sequentially; let exceptions stop the run to surface issues
    test_load_from_folder()
    test_load_from_hf()
    test_load_from_arxiv_ids()
