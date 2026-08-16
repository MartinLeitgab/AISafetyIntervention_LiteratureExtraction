#!/usr/bin/env python
"""What does the mechanism layer hold that a topical index over the same corpus does not?

GitHub issue #166. OPEN_ITEMS.md S12, runbook R3b.

The Introduction claims existing resources over this literature index the DOCUMENT, and
that the argument connecting a risk to a proposed remedy is not queryable in them. That
claim is never tested against an artifact that exists. S6/S8 compare the pipeline against
simpler versions of itself; this compares it against something already built.

The comparison artifact
-----------------------
The AI Safety Graph (mouratoglou2024aisafetygraph, Apart Research) publishes its inputs and
its clustering output. Downloaded to phase2_results/external_artifacts/:

    arxiv_papers_for_llm.csv   7,011 arXiv papers from the same ARD, with title, ABSTRACT,
                               url and paper_id. The abstract is their unit: their LLM sees
                               the title and the abstract, never the document.
    categories.json            the released clustering: 554 distinct papers in 160
                               subcategories under 29 top-level categories.

Same corpus, same LLM-based approach, different unit of analysis -- which makes the
comparison about the unit rather than about model quality.

What is measured
----------------
1. overlap        their papers that are in our corpus, and that yield a gate-selected chain
2. resolution     distinct risks, interventions and directed pairs our graph holds inside
                  one of their subcategories, against the one label the subcategory is
3. direction      the directed risk-to-intervention pairs a topic label cannot express
4. the converse   what their artifact supplies and ours cannot: cross-document grouping.
                  Our structural layer asserts no cross-document edge at all, so every
                  within-subcategory relation is theirs alone. Reported in the same table.

Matching is by arXiv id parsed from the URL, with a normalized-title fallback, and every
unmatched row is counted and reported rather than dropped silently.

CLASS B: no LLM call, no API key, no network at run time. Run from graph_analysis/:

    python -u experiment_review_artifact_comparison.py

Output: phase2_results/experiment_review_artifact_comparison_report.json
"""

from __future__ import annotations

import csv
import json
import pickle
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
EXT = ROOT / "phase2_results/external_artifacts"
CATEGORIES = EXT / "categories.json"
PAPERS_CSV = EXT / "arxiv_papers_for_llm.csv"
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
PATHS = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
OUT = ROOT / "phase2_results/experiment_review_artifact_comparison_report.json"

DOWNLOAD = (
    "curl -sSL -o <file> https://raw.githubusercontent.com/ai-safety-graph/"
    "AISafetyGraph/main/generate_md/<file>"
)


def fail(msg: str, artifact: Path, produced_by: str) -> None:
    raise SystemExit(
        f"FATAL: {msg}\n"
        f"  expected artifact: {artifact}\n"
        f"  produced by: {produced_by}\n"
        "  this script does NOT fall back to a partial artifact or to our own clustering "
        "as a stand-in for theirs."
    )


_ARXIV = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z-]+/[0-9]{7})", re.I
)


def arxiv_id(url: str) -> str | None:
    m = _ARXIV.search(url or "")
    return m.group(1).lower() if m else None


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def main() -> None:
    t0 = time.time()
    for path, what in [
        (CATEGORIES, "AI Safety Graph clustering output (categories.json)"),
        (PAPERS_CSV, "AI Safety Graph input corpus (arxiv_papers_for_llm.csv)"),
    ]:
        if not path.exists():
            fail(f"{what} not found", path, DOWNLOAD)
    for path, what, how in [
        (SLIM, "slim node attributes", "experiment_review_prep_slim_nodes.py"),
        (PATHS, "the released 2,772-chain reporting unit", "phase1_dedup_paths.py"),
    ]:
        if not path.exists():
            fail(f"{what} not found", path, how)

    # ---- their artifact -------------------------------------------------------------
    cats = json.loads(CATEGORIES.read_text(encoding="utf-8"))
    subcat_titles: dict[tuple[str, str], list[str]] = {}
    for cat, subs in cats.items():
        for sub, titles in subs.items():
            subcat_titles[(cat, sub)] = list(titles)
    their_titles = {t for ts in subcat_titles.values() for t in ts}

    csv.field_size_limit(10**9)
    with open(PAPERS_CSV, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_title = {r["title"]: r for r in rows}
    by_norm_title = {norm_title(r["title"]): r for r in rows}

    # ---- our artifact ---------------------------------------------------------------
    na = pickle.load(open(SLIM, "rb"))
    url_of = {nid: a.get("url") for nid, a in na.items()}
    corpus_urls = {u for u in url_of.values() if u}
    corpus_by_arxiv = {}
    corpus_by_title = {}
    for u in corpus_urls:
        aid = arxiv_id(u)
        if aid:
            corpus_by_arxiv.setdefault(aid, u)

    chains_by_url = defaultdict(list)
    with open(PATHS, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            p = rec["path"]
            u = url_of.get(p[0])
            if u:
                chains_by_url[u].append(p)

    def node_name(nid):
        return (na.get(nid, {}) or {}).get("name") or str(nid)

    # ---- match ----------------------------------------------------------------------
    matched, unmatched_no_csv, unmatched_no_corpus = {}, [], []
    for title in sorted(their_titles):
        row = by_title.get(title) or by_norm_title.get(norm_title(title))
        if not row:
            unmatched_no_csv.append(title)
            continue
        aid = arxiv_id(row["url"])
        u = corpus_by_arxiv.get(aid) if aid else None
        if not u:
            u = corpus_by_title.get(norm_title(title))
        if not u:
            unmatched_no_corpus.append(title)
            continue
        matched[title] = u

    # Their whole input corpus against ours, not only the clustered subset.
    their_ids = {arxiv_id(r["url"]) for r in rows}
    their_ids.discard(None)
    input_overlap = sorted(their_ids & set(corpus_by_arxiv))

    # ---- per-subcategory resolution -------------------------------------------------
    per_sub = []
    for (cat, sub), titles in sorted(subcat_titles.items()):
        urls = [matched[t] for t in titles if t in matched]
        chain_urls = [u for u in urls if chains_by_url.get(u)]
        risks, intvs, pairs = set(), set(), set()
        for u in chain_urls:
            for p in chains_by_url[u]:
                risks.add(node_name(p[0]))
                intvs.add(node_name(p[-1]))
                pairs.add((node_name(p[0]), node_name(p[-1])))
        per_sub.append(
            {
                "category": cat,
                "subcategory": sub,
                "their_papers": len(titles),
                "matched_to_our_corpus": len(urls),
                "of_which_chain_yielding": len(chain_urls),
                "our_distinct_risks": len(risks),
                "our_distinct_interventions": len(intvs),
                "our_directed_pairs": len(pairs),
                "their_co_membership_pairs": len(urls) * (len(urls) - 1) // 2,
                "our_cross_document_relations": 0,
            }
        )

    covered = [s for s in per_sub if s["of_which_chain_yielding"] > 0]
    biggest = max(per_sub, key=lambda s: s["of_which_chain_yielding"])
    worked = None
    if biggest["of_which_chain_yielding"]:
        titles = subcat_titles[(biggest["category"], biggest["subcategory"])]
        rows_ = []
        for t in titles:
            u = matched.get(t)
            for p in chains_by_url.get(u, [])[:2]:
                rows_.append(
                    {
                        "paper_title": t,
                        "risk": node_name(p[0]),
                        "intervention": node_name(p[-1]),
                        "chain_nodes": len(p),
                    }
                )
        worked = {
            "category": biggest["category"],
            "subcategory": biggest["subcategory"],
            "their_label_is_one_string": biggest["subcategory"],
            "our_rows": rows_,
        }

    report = {
        "study": "comparison against an existing document-level artifact (issue #166, S12)",
        "their_artifact": {
            "name": "AI Safety Graph (Apart Research), mouratoglou2024aisafetygraph",
            "source": "github.com/ai-safety-graph/AISafetyGraph, generate_md/",
            "unit": "one arXiv paper's title and ABSTRACT",
            "input_papers": len(rows),
            "clustered_papers": len(their_titles),
            "categories": len(cats),
            "subcategories": len(subcat_titles),
            "NOTE": (
                "The released clustering covers 554 papers, not the 'more than five "
                "thousand documents' the live site reports; 7,011 papers are in the "
                "input CSV. Both numbers are theirs; we quote what the released files "
                "contain."
            ),
        },
        "our_artifact": {
            "unit": "one full document, extracted to a typed risk-to-intervention chain",
            "corpus_documents": len(corpus_urls),
            "chain_yielding_documents": len(chains_by_url),
            "chains": sum(len(v) for v in chains_by_url.values()),
        },
        "overlap": {
            "their_input_papers_in_our_corpus": len(input_overlap),
            "their_input_papers_in_our_corpus_pct": round(
                100 * len(input_overlap) / max(1, len(their_ids)), 1
            ),
            "their_clustered_papers_matched": len(matched),
            "their_clustered_papers_unmatched_missing_from_their_own_csv": len(
                unmatched_no_csv
            ),
            "their_clustered_papers_unmatched_not_in_our_corpus": len(
                unmatched_no_corpus
            ),
            "matched_papers_yielding_a_chain": sum(
                1 for u in matched.values() if chains_by_url.get(u)
            ),
        },
        "resolution": {
            "subcategories_with_at_least_one_chain_yielding_paper": len(covered),
            "median_our_directed_pairs_per_covered_subcategory": (
                statistics.median([s["our_directed_pairs"] for s in covered])
                if covered
                else 0
            ),
            "total_our_directed_pairs_inside_their_subcategories": sum(
                s["our_directed_pairs"] for s in covered
            ),
            "total_their_labels_for_the_same_material": len(covered),
            "directed_pairs_per_label": round(
                sum(s["our_directed_pairs"] for s in covered) / max(1, len(covered)), 1
            ),
            "per_subcategory": per_sub,
        },
        "yield_cross_check": {
            "chain_yield_on_their_clustered_papers_pct": round(
                100
                * sum(1 for u in matched.values() if chains_by_url.get(u))
                / max(1, len(matched)),
                1,
            ),
            "arxiv_chain_yield_reported_in_the_paper_pct": 40.5,
            "note": (
                "Their clustered set is arXiv-only, so its chain yield should reproduce the "
                "arXiv row of tab:populations. It is an independent check on that number, "
                "computed over a paper list we did not choose."
            ),
        },
        "what_their_artifact_has_and_ours_does_not": {
            "co_membership_pairs_they_assert": sum(
                s["their_co_membership_pairs"] for s in per_sub
            ),
            "cross_document_relations_we_assert": 0,
            "why": (
                "Every structural edge in our graph comes from one paper (sec:m-structural). "
                "Their clustering is a cross-document statement and ours is not; the "
                "similarity layer is the only place we have anything comparable, and it is "
                "threshold-dependent (sec:m-paths)."
            ),
        },
        "worked_example": worked,
        "unmatched_examples": {
            "missing_from_their_own_csv": unmatched_no_csv[:10],
            "not_in_our_corpus": unmatched_no_corpus[:10],
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k != "resolution"}, indent=2)[
            :3000
        ]
    )
    print("\nresolution summary:")
    print(
        json.dumps(
            {k: v for k, v in report["resolution"].items() if k != "per_subcategory"},
            indent=2,
        )
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
