import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

from intervention_graph_creation.src.local_graph_extraction.extract.extractor import Extractor
from intervention_graph_creation.src.local_graph_extraction.extract.utilities import (
    url_to_id,
    filter_dict,
)
from config import load_settings

SETTINGS = load_settings()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

META_KEYS = frozenset(["authors", "date_published", "filename", "source", "source_filetype", "title", "url"])

class BatchResumer(Extractor):
    def __init__(self):
        super().__init__()

    def parse_custom_id(self, custom_id: str) -> Tuple[str, str, int]:
        if custom_id.startswith("pdf_"):
            rest = custom_id[len("pdf_"):]
            base, idx = rest.rsplit("_", 1)
            return "pdf", base, int(idx)
        if custom_id.startswith("jsonl_"):
            rest = custom_id[len("jsonl_"):]
            base, idx = rest.rsplit("_", 1)
            return "jsonl", base, int(idx)
        base, idx = custom_id.rsplit("_", 1)
        return "jsonl", base, int(idx)

    def build_meta_index(self, jsonl_paths: List[Path]) -> Dict[str, Dict]:
        index: Dict[str, Dict] = {}
        for jp in jsonl_paths:
            if not jp.exists():
                continue
            with jp.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    base = jp.stem
                    pid = f"{base}__{url_to_id(rec.get('url', f'line_{i}'))}"
                    meta = filter_dict(rec, META_KEYS)
                    index[pid] = meta
        return index

    def collect_custom_ids(self, batch) -> Set[str]:
        cids: Set[str] = set()
        ofid = getattr(batch, "output_file_id", None)
        efid = getattr(batch, "error_file_id", None)
        if ofid:
            try:
                of = self.client.files.content(ofid)
                for line in of.text.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        cid = obj.get("custom_id")
                        if cid:
                            cids.add(cid)
                    except Exception:
                        continue
            except Exception:
                pass
        if efid:
            try:
                ef = self.client.files.content(efid)
                for line in ef.text.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        cid = obj.get("custom_id")
                        if cid:
                            cids.add(cid)
                    except Exception:
                        continue
            except Exception:
                pass
        return cids

    def make_batch_requests_for(self, custom_ids: Set[str], meta_index: Dict[str, Dict]) -> List[Dict]:
        reqs: List[Dict] = []
        for cid in custom_ids:
            try:
                ftype, pid, _ = self.parse_custom_id(cid)
            except Exception:
                ftype, pid = "jsonl", cid
            meta = meta_index.get(pid, {})
            reqs.append({
                "request": {"custom_id": cid},
                "paper_id": pid,
                "meta": meta,
                "file_type": ftype,
                "file_path": None,
            })
        return reqs

    def process_existing_batch(self, batch_id: str, meta_index: Dict[str, Dict]) -> None:
        batch = self.client.batches.retrieve(batch_id)
        if batch.status != "completed":
            logger.warning(f"Batch {batch_id} not completed (status={batch.status})")
            return
        custom_ids = self.collect_custom_ids(batch)
        if not custom_ids:
            logger.warning(f"No custom_ids found for batch {batch_id}")
            return
        batch_requests = self.make_batch_requests_for(custom_ids, meta_index)
        self._process_batch_results(batch, batch_requests, elapsed=0.0)

    def resume_many(self, batch_ids: List[str], jsonl_sources: List[Path]) -> None:
        meta_index = self.build_meta_index(jsonl_sources)
        for bid in batch_ids:
            try:
                self.process_existing_batch(bid, meta_index)
            except Exception as e:
                self._handle_failure(
                    req={"paper_id": "batch_level", "request": {"custom_id": bid}, "file_type": None, "file_path": None},
                    err=e,
                    stage="resume_many.process_existing_batch",
                    meta={"batch_id": bid},
                    file_type=None,
                )
        self.print_summary()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batches-file", type=Path, required=False)
    p.add_argument("--batch-id", action="append", default=[])
    p.add_argument("--sources", type=Path, nargs="+", required=False)
    p.add_argument("--include-default-arxiv", action="store_true", default=False)
    return p.parse_args()

def main():
    args = parse_args()
    batch_ids: List[str] = []
    if args.batches_file and args.batches_file.exists():
        with args.batches_file.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().strip(",").strip().strip('"').strip("'")
                if not s:
                    continue
                if s.startswith("batch_"):
                    batch_ids.append(s)
    if args.batch_id:
        batch_ids.extend([b for b in args.batch_id if b.startswith("batch_")])
    batch_ids = sorted(set(batch_ids))
    if not batch_ids:
        raise SystemExit("No batch ids provided")

    sources: List[Path] = args.sources or []
    if args.include_default_arxiv:
        sources.append(Path("./AISafetyIntervention_LiteratureExtraction/intervention_graph_creation/data/raw/ard_json/arxiv.jsonl"))
    resumer = BatchResumer()
    resumer.resume_many(batch_ids, sources)

if __name__ == "__main__":
    main()
