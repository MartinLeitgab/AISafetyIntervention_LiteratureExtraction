#!/usr/bin/env python
"""Flag corpus documents whose source body is too short or mostly links (review C3).

The human reading (#176) found one unsupported chain, C27, built from a 215-character
forum post whose body is a single link. The extractor cannot follow links, so on such a
document every node it emits comes from the model's own knowledge. This script flags such
documents across the whole corpus so the release can ship the flag as a per-document
attribute and users can filter on it.

Two flags, both computed from the ARD source text only (no LLM, no graph structure):
  short_source     body shorter than SHORT_CHARS characters
  link_dominated   the body contains at least one URL and, with every URL and markdown
                   link target removed, fewer than PROSE_CHARS characters of text remain

The thresholds are stated choices, not fitted values. C27 must be flagged or the script
exits non-zero: that is the one adjudicated case of the failure this flag targets.

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_review_degenerate_source_flag.py

Inputs:  phase2_results/node_attrs_slim.pkl (node id -> url),
         phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl (the 2,772 chains),
         ../intervention_graph_creation/data/raw/ard_json_full/*.jsonl (source text)
Outputs: phase2_results/experiment_review_degenerate_source_flag_report.json
         phase2_results/degenerate_source_flags.csv (one row per corpus document)
"""

import csv
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
REPO = ROOT.parent
NODES = ROOT / "phase2_results/node_attrs_slim.pkl"
DEDUP = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
ARD = REPO / "intervention_graph_creation/data/raw/ard_json_full"
OUT = ROOT / "phase2_results/experiment_review_degenerate_source_flag_report.json"
OUT_CSV = ROOT / "phase2_results/degenerate_source_flags.csv"

SHORT_CHARS = 1000
PROSE_CHARS = 500
C27_URL = (
    "https://forum.effectivealtruism.org/posts/xqBaBjXYy5yHbpXou/"
    "l-importanza-delle-ia-come-possibile-minaccia-per-l-umanita"
)

URL_RE = re.compile(r"https?://\S+")
MD_LINK_RE = re.compile(r"\]\([^)]*\)")


def prose_chars(text: str) -> int:
    t = MD_LINK_RE.sub("]", text)
    t = URL_RE.sub("", t)
    return len(re.sub(r"\s+", " ", t).strip())


def main() -> int:
    for p, how in (
        (NODES, "experiment_review_prep_slim_nodes.py"),
        (DEDUP, "phase1_dedup_paths.py"),
        (ARD, "intervention_graph_creation/data/raw/ard_json_full/download_ard.py"),
    ):
        if not p.exists():
            raise SystemExit(
                f"FATAL: missing input {p}\n  produced by: {how}\n"
                "  This script does not fall back to any other source."
            )

    na = pickle.load(open(NODES, "rb"))
    corpus_urls = {a.get("url") for a in na.values() if a.get("url")}
    print(f"corpus documents (distinct node urls): {len(corpus_urls):,}", flush=True)

    chains_per_url = Counter()
    for line in open(DEDUP, encoding="utf-8"):
        path = json.loads(line)["path"]
        chains_per_url[na[path[0]]["url"]] += 1
    print(
        f"chains: {sum(chains_per_url.values()):,} over {len(chains_per_url):,} documents"
    )

    rows = {}
    for fp in sorted(ARD.glob("*.jsonl")):
        for line in open(fp, encoding="utf-8"):
            d = json.loads(line)
            u = d.get("url")
            if u not in corpus_urls or u in rows:
                continue
            text = d.get("text") or ""
            n_urls = len(URL_RE.findall(text))
            pc = prose_chars(text)
            rows[u] = {
                "url": u,
                "body_chars": len(text),
                "n_links": n_urls,
                "prose_chars_without_links": pc,
                "short_source": len(text) < SHORT_CHARS,
                "link_dominated": n_urls >= 1 and pc < PROSE_CHARS,
                "chains_in_reporting_unit": chains_per_url.get(u, 0),
            }
        print(f"  {fp.name}: matched {len(rows):,}", flush=True)

    unmatched = corpus_urls - set(rows)
    if unmatched:
        raise SystemExit(
            f"FATAL: {len(unmatched)} corpus documents have no ARD source record; "
            f"example: {next(iter(unmatched))}"
        )

    c27 = rows.get(C27_URL)
    if not c27 or not (c27["short_source"] or c27["link_dominated"]):
        raise SystemExit(f"FATAL: validation case C27 is not flagged: {c27}")

    def tally(pred):
        docs = [r for r in rows.values() if pred(r)]
        cy = [r for r in docs if r["chains_in_reporting_unit"] > 0]
        return {
            "documents": len(docs),
            "documents_pct_of_corpus": round(100 * len(docs) / len(rows), 1),
            "chain_yielding_documents": len(cy),
            "chains": sum(r["chains_in_reporting_unit"] for r in cy),
        }

    report = {
        "study": "degenerate-source flag (review C3): short or link-dominated source bodies",
        "thresholds": {"SHORT_CHARS": SHORT_CHARS, "PROSE_CHARS": PROSE_CHARS},
        "corpus_documents": len(rows),
        "chains_total": sum(chains_per_url.values()),
        "short_source": tally(lambda r: r["short_source"]),
        "link_dominated": tally(lambda r: r["link_dominated"]),
        "either": tally(lambda r: r["short_source"] or r["link_dominated"]),
        "validation_case_C27": c27,
        "LIMITS": (
            "A document-level flag from text length and link density. It does not detect a "
            "node whose content comes from a link inside an otherwise long document (C06, "
            "C12, C15, C24 in the human reading); that needs a per-node check."
        ),
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values())).keys()))
        w.writeheader()
        for u in sorted(rows):
            w.writerow(rows[u])
    print(
        json.dumps(
            {k: report[k] for k in ("short_source", "link_dominated", "either")},
            indent=1,
        )
    )
    print(f"C27: {c27}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
