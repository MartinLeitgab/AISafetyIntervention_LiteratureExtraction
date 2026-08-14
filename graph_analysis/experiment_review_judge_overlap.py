#!/usr/bin/env python
"""Does the judged sample cover the analysed chain set? (reviewer item W-1 / Q1)

Both simulated reviews (NeurIPS + workshop, 2026-08-14) raise the same objection: the 100
judged papers are sampled from SUCCESSFULLY EXTRACTED documents, while every headline
number is computed over the 2,772-chain reporting unit, which comes from the 1,868
CHAIN-YIELDING documents. Appendix G asserts the audited sample is "forum-weighted in the
same way the corpus is" -- true of the 11,779-document corpus, and the reviewers claim it
is false of the chain set.

This script answers the question with data rather than argument:

  1. How many of the 100 judged papers are among the 1,868 chain-yielding papers?
  2. What is the source-type composition of that overlap?
  3. How do the three populations (corpus / judged sample / chain set) compare by source
     type, and what is the per-source-type chain yield that drives any difference?

Class B (no LLM, no network). Run from graph_analysis/:
    python -u experiment_review_judge_overlap.py --judge-reports <dir>

Output: graph_analysis/phase2_results/experiment_review_judge_overlap_report.json
"""

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
DEDUP = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
OUT = ROOT / "phase2_results/experiment_review_judge_overlap_report.json"

# Host -> ARD source type. The ARD source types are the nine directory prefixes the judge
# sample is named by. Hosts not listed here fall into "other_web", which is what the ARD
# "blogs" and "special_docs" types mostly are.
HOST_TO_ARD = {
    "arxiv.org": "arxiv",
    "www.arxiv.org": "arxiv",
    "lesswrong.com": "lesswrong",
    "www.lesswrong.com": "lesswrong",
    "alignmentforum.org": "alignmentforum",
    "www.alignmentforum.org": "alignmentforum",
    "forum.effectivealtruism.org": "eaforum",
    "www.effectivealtruism.org": "eaforum",
    "arbital.com": "arbital",
    "www.arbital.com": "arbital",
    "aisafety.info": "aisafety.info",
    "www.aisafety.info": "aisafety.info",
    "youtube.com": "youtube",
    "www.youtube.com": "youtube",
    "youtu.be": "youtube",
    "intelligence.org": "miri",
    "www.intelligence.org": "miri",
    "agentmodels.org": "agentmodels",
    "docs.google.com": "special_docs",
}


def host_of(url):
    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""


def ard_type_of(url, learned):
    h = host_of(url)
    if h in learned:
        return learned[h]
    if h in HOST_TO_ARD:
        return HOST_TO_ARD[h]
    return "other_web"


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--judge-reports",
        required=True,
        help="directory of the 100 Sonnet-4.5 judge report JSONs (each carries a url)",
    )
    a = ap.parse_args()
    jdir = Path(a.judge_reports)
    if not jdir.is_dir():
        raise SystemExit(
            f"FATAL: judge report directory not found: {jdir}\n"
            "  Produce it with:\n"
            "    git archive origin/anthropic_judge_test "
            "extraction_validator/extend_try_1 | tar -x -C <DEST>\n"
            "  This script does NOT fall back to the summary receipt."
        )
    if not SLIM.exists():
        raise SystemExit(
            f"FATAL: {SLIM} not found.\n"
            "  Produce it with: python -u experiment_review_prep_slim_nodes.py"
        )

    # ---- the judged sample -------------------------------------------------------
    judged = {}
    missing_url = []
    for p in sorted(jdir.glob("*.json")):
        if p.name in ("summary.json", "errors.json"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        url = r.get("url")
        stype = p.name.split("__", 1)[0]
        if not url:
            missing_url.append(p.name)
            continue
        judged[p.name] = {
            "url": url,
            "ard_source_type": stype,
            "paper_id": r.get("paper_id"),
            "n_final_nodes": len((r.get("final_graph") or {}).get("nodes") or []),
        }
    if not judged:
        raise SystemExit(f"FATAL: no judge reports with a url field in {jdir}")

    # Learn host -> ARD type from the judged sample itself: it is the only place where a
    # URL and its ARD source type appear together.
    learned = {}
    conflicts = defaultdict(set)
    for row in judged.values():
        h = host_of(row["url"])
        if h:
            conflicts[h].add(row["ard_source_type"])
    for h, types in conflicts.items():
        if len(types) == 1:
            learned[h] = next(iter(types))

    # ---- the corpus and the chain set --------------------------------------------
    slim = pickle.load(open(SLIM, "rb"))
    corpus_urls = {r["url"] for r in slim.values() if r.get("url")}

    chain_urls = set()
    n_chains = 0
    chains_per_url = Counter()
    for line in open(DEDUP, encoding="utf-8"):
        d = json.loads(line)
        urls = {slim[n]["url"] for n in d["path"] if n in slim}
        if len(urls) == 1:
            u = next(iter(urls))
            chain_urls.add(u)
            chains_per_url[u] += 1
        n_chains += 1

    judged_urls = {r["url"] for r in judged.values()}
    overlap_urls = judged_urls & chain_urls
    judged_in_corpus = judged_urls & corpus_urls

    # ---- source-type composition of the three populations ------------------------
    def compose(urls):
        c = Counter(ard_type_of(u, learned) for u in urls)
        tot = sum(c.values())
        return {
            k: {"n": v, "pct": pct(v, tot)}
            for k, v in sorted(c.items(), key=lambda kv: -kv[1])
        }

    corpus_comp = compose(corpus_urls)
    chain_comp = compose(chain_urls)
    judged_comp_hostmapped = compose(judged_urls)
    judged_comp_filename = {
        k: {"n": v, "pct": pct(v, len(judged))}
        for k, v in sorted(
            Counter(r["ard_source_type"] for r in judged.values()).items(),
            key=lambda kv: -kv[1],
        )
    }

    # chains, not papers, by source type -- the reporting unit's own mix
    chain_unit_comp_counter = Counter()
    for u, k in chains_per_url.items():
        chain_unit_comp_counter[ard_type_of(u, learned)] += k
    tot_chain_unit = sum(chain_unit_comp_counter.values())
    chain_unit_comp = {
        k: {"n": v, "pct": pct(v, tot_chain_unit)}
        for k, v in sorted(chain_unit_comp_counter.items(), key=lambda kv: -kv[1])
    }

    # per-source-type yield: share of corpus documents of that type that yield a chain
    corpus_by_type = Counter(ard_type_of(u, learned) for u in corpus_urls)
    chain_by_type = Counter(ard_type_of(u, learned) for u in chain_urls)
    yield_by_type = {
        t: {
            "corpus_docs": corpus_by_type[t],
            "chain_yielding_docs": chain_by_type.get(t, 0),
            "yield_pct": pct(chain_by_type.get(t, 0), corpus_by_type[t]),
        }
        for t in sorted(corpus_by_type, key=lambda t: -corpus_by_type[t])
    }

    overlap_rows = sorted(
        (
            {
                "file": f,
                "ard_source_type": r["ard_source_type"],
                "url": r["url"],
                "n_chains_in_reporting_unit": chains_per_url[r["url"]],
            }
            for f, r in judged.items()
            if r["url"] in chain_urls
        ),
        key=lambda x: (x["ard_source_type"], x["url"]),
    )

    n_j = len(judged)
    n_ov = len(overlap_urls)
    corpus_yield = pct(len(chain_urls), len(corpus_urls))
    expected = round(len(judged) * len(chain_urls) / len(corpus_urls), 1)

    out = {
        "experiment": "judged-sample vs analysed-chain-set population overlap (W-1 / Q1)",
        "question": (
            "How many of the 100 judged papers are among the chain-yielding papers whose "
            "chains every headline number is computed over, and is the judged sample's "
            "source-type mix the chain set's mix or the document corpus's mix?"
        ),
        "inputs": {
            "judge_reports": str(jdir),
            "chain_set": str(DEDUP),
            "node_attrs": str(SLIM),
        },
        "headline": {
            "n_judged_papers": n_j,
            "n_judged_with_url": n_j,
            "n_judged_urls_present_in_corpus": len(judged_in_corpus),
            "n_corpus_documents": len(corpus_urls),
            "n_chain_yielding_documents": len(chain_urls),
            "corpus_chain_yield_pct": corpus_yield,
            "n_judged_papers_that_yield_a_chain": n_ov,
            "pct_of_judged_sample_that_yields_a_chain": pct(n_ov, n_j),
            "expected_overlap_if_judged_sample_were_uniform_over_corpus": expected,
            "n_chains_in_reporting_unit_covered_by_the_judged_sample": sum(
                chains_per_url[u] for u in overlap_urls
            ),
            "pct_of_2772_chains_covered": pct(
                sum(chains_per_url[u] for u in overlap_urls), n_chains
            ),
        },
        "source_type_composition": {
            "corpus_documents_hostmapped": corpus_comp,
            "chain_yielding_documents_hostmapped": chain_comp,
            "chains_reporting_unit_hostmapped": chain_unit_comp,
            "judged_sample_by_filename_prefix": judged_comp_filename,
            "judged_sample_hostmapped": judged_comp_hostmapped,
        },
        "per_source_type_chain_yield": yield_by_type,
        "overlap_papers": overlap_rows,
        "host_to_ard_type_mapping": {
            "learned_from_judged_sample": learned,
            "static_fallback": HOST_TO_ARD,
            "NOTE": (
                "ARD source types are recoverable for the judged sample from the file "
                "name. For the corpus and the chain set only URLs exist, so types are "
                "assigned by host. Hosts seen in the judged sample carry their observed "
                "type; the rest use the static table; anything unmatched is other_web, "
                "which is where the ARD 'blogs' and most 'special_docs' records land. "
                "Compare like with like: judged_sample_hostmapped against the two "
                "hostmapped population rows, never against the filename-prefix row."
            ),
        },
        "reports_without_a_url_field": missing_url,
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")

    h = out["headline"]
    print(json.dumps(h, indent=1))
    print("\nchain yield by source type:")
    for t, r in yield_by_type.items():
        print(
            f"  {t:16s} corpus={r['corpus_docs']:6d} "
            f"chain-yielding={r['chain_yielding_docs']:5d} ({r['yield_pct']}%)"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
