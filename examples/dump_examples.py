import os
import sys
from typing import List, Tuple

# Ensure 'src' is on sys.path so 'publications' package is importable when running directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from publications.utils import split_references, parse_arxiv_id_from_filename  # type: ignore


def extract_text_from_pdf(file_path: str) -> str:
    from pypdf import PdfReader  # local import to keep startup fast

    reader = PdfReader(file_path)
    pages: List[str] = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages).strip()


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def dump_for_pdfs(pdf_dir: str, out_dir: str, limit: int = 5) -> List[Tuple[str, str, str]]:
    """
    For up to `limit` PDFs in `pdf_dir`, extract original text, split content/references,
    and write to files under `out_dir`.

    Returns list of tuples of written base paths (original, content, references or "").
    """
    names = [n for n in sorted(os.listdir(pdf_dir)) if n.lower().endswith(".pdf")]
    results: List[Tuple[str, str, str]] = []
    for name in names[:limit]:
        pdf_path = os.path.join(pdf_dir, name)
        arxiv_id = parse_arxiv_id_from_filename(name) or os.path.splitext(name)[0]

        # Extract raw
        raw_text = extract_text_from_pdf(pdf_path)

        # Split
        content, refs = split_references(raw_text)

        # Paths
        base = os.path.join(out_dir, arxiv_id)
        orig_path = base + "_original.txt"
        content_path = base + "_content.txt"
        refs_path = base + "_references.txt"

        # Write individual files
        write_text(orig_path, raw_text)
        write_text(content_path, content)
        if refs:
            write_text(refs_path, refs)
        else:
            # If no refs detected, still create an empty file for clarity
            write_text(refs_path, "")

        results.append((orig_path, content_path, refs_path))

    return results


def main() -> None:
    pdf_dir = os.path.join(REPO_ROOT, "inputdata_development_paper_set")
    out_dir = os.path.join(REPO_ROOT, "examples_output")
    if not os.path.isdir(pdf_dir):
        raise SystemExit(f"PDF directory not found: {pdf_dir}")
    written = dump_for_pdfs(pdf_dir, out_dir)
    print("Wrote example dumps:")
    for ori, con, ref in written:
        print(f"- {ori}")
        print(f"  {con}")
        print(f"  {ref}")


if __name__ == "__main__":
    main()
