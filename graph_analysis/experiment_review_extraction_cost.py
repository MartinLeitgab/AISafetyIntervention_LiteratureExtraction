#!/usr/bin/env python
"""What did the extraction cost, per document and in total?

Reviewer question (NeurIPS W5/Q4, workshop V3/Q3): the paper asserts that extraction cost
is linear in corpus size and carried "Scalable" in its title, with no token count, no
wall-clock and no dollar figure anywhere. The batch-API run logs did not survive.

This reconstructs the token bill from artifacts that did survive, without spending a
single inference token.

What is exact
-------------
INPUT tokens. The full ARD source text is on disk (data/raw/ard_json_full/*.jsonl, one
record per document with a `text` field). Every document in the released graph is matched
to its ARD record by URL, and input tokens are counted as

    tokens(PROMPT_EXTRACT) + tokens(document text)

with the same tokenizer family the extractor billed against (o200k_base). Documents in
the graph with no matching ARD record are reported separately and excluded, never
silently imputed.

What is calibrated
------------------
VISIBLE OUTPUT tokens. One complete `_raw_response.txt` survives in
data/processed/2311.07590/. It gives tokens-per-emitted-element for a real response
(nodes + edges), and the released graph gives node and edge counts for every document.
Output is therefore estimated as

    tokens_per_element * (n_nodes + n_edges)

The calibration rests on ONE document. The report states that, and states the per-element
figure so a reader can rescale it if a second response is ever recovered.

What is NOT recoverable, and is not guessed
-------------------------------------------
REASONING tokens. o3 bills reasoning tokens that never appear in the response body, and
nothing on disk records them. The report gives the bill at several assumed
reasoning-to-visible-output ratios and labels every one an assumption. No single dollar
figure is presented as measured.

Prices are inputs to this script, not findings: they are printed with the result so a
stale rate is visible rather than buried.

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_review_extraction_cost.py

Output: graph_analysis/phase2_results/experiment_review_extraction_cost_report.json
"""

from __future__ import annotations

import json
import pickle
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import tiktoken

ROOT = Path(__file__).parent
REPO = ROOT.parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
EDGES = STEP1 / "graph_edge_data.pkl"
ARD = REPO / "intervention_graph_creation/data/raw/ard_json_full"
PROMPT_PY = REPO / "intervention_graph_creation/src/prompt/final_primary_prompt.py"
CALIB = REPO / "intervention_graph_creation/data/processed/2311.07590"
OUT = ROOT / "phase2_results/experiment_review_extraction_cost_report.json"

ENCODING = "o200k_base"

# Rates are ASSUMPTIONS, printed with the result. Every figure below is at the BATCH
# rate, which is half the synchronous rate on both providers; the extraction ran through
# a batch API, so batch is the correct basis and no further discount applies. Reasoning
# tokens bill as output tokens on both providers.
#
# NAME: (synchronous input, synchronous output) USD per million tokens.
SYNC_RATES = {
    "o3 (as run)": (2.00, 8.00),
    "Claude Opus 5": (5.00, 25.00),
    "Claude Sonnet 5": (3.00, 15.00),
    "Claude Haiku 4.5": (1.00, 5.00),
}
BATCH_DISCOUNT = 0.5
RATE_NOTE = (
    "Every figure is at BATCH rates, half the synchronous rate, because the run used a "
    "batch API; no further discount applies. o3 assumed at USD 2/8 per M synchronous. "
    "Anthropic rates from the model catalog cached 2026-06-24: Opus 5 USD 5/25, Sonnet 5 "
    "USD 3/15 (introductory USD 2/10 through 2026-08-31, not used here), Haiku 4.5 "
    "USD 1/5. Verify against current pricing before quoting a dollar figure."
)
TOKENIZER_CAVEAT = (
    "Token counts are measured with o200k_base, the tokenizer the run billed against. "
    "Anthropic models tokenize differently, so the non-o3 rows reprice THIS token volume "
    "at another vendor's rates rather than predicting that vendor's own token count. "
    "Read them as the order of the cost, not as a quote; a count_tokens call on a sample "
    "would pin the difference."
)
# Reasoning-token multipliers on visible output, reported as a band, never as one number.
REASONING_RATIOS = [0.0, 1.0, 2.0, 4.0]


def fail(msg: str, artifact: Path, produced_by: str) -> None:
    raise SystemExit(
        f"FATAL: {msg}\n"
        f"  expected artifact: {artifact}\n"
        f"  produced by: {produced_by}\n"
        "  this script does NOT estimate around a missing input: a cost figure built on "
        "an imputed corpus is worse than no cost figure."
    )


def main():
    t0 = time.time()
    for path, what, how in [
        (SLIM, "slim node attributes", "experiment_review_prep_slim_nodes.py"),
        (EDGES, "edge checkpoint", "phase2_step1_loadandparse.py"),
        (ARD, "ARD source corpus", "download_ard.py in that directory"),
        (PROMPT_PY, "extraction prompt module", "the extraction pipeline source"),
        (CALIB, "surviving raw response", "the original extraction run"),
    ]:
        if not path.exists():
            fail(f"{what} not found", path, how)

    enc = tiktoken.get_encoding(ENCODING)

    # ---- the prompt -----------------------------------------------------------------
    src = PROMPT_PY.read_text(encoding="utf-8")
    marker = "PROMPT_EXTRACT"
    if marker not in src:
        fail("PROMPT_EXTRACT not found in the prompt module", PROMPT_PY, "the pipeline")
    prompt_text = src.split(marker, 1)[1]
    prompt_tokens = len(enc.encode(prompt_text))

    # ---- corpus documents and their emitted element counts --------------------------
    na = pickle.load(open(SLIM, "rb"))
    nodes_per_doc = Counter()
    for a in na.values():
        if a.get("url"):
            nodes_per_doc[a["url"]] += 1
    edges = pickle.load(open(EDGES, "rb"))
    node_url = {n: a.get("url") for n, a in na.items()}
    edges_per_doc = Counter()
    for e in edges:
        if e.get("type") != "EDGE":
            continue
        u = node_url.get(e["source"])
        if u:
            edges_per_doc[u] += 1
    del edges
    corpus_urls = set(nodes_per_doc)
    print(f"corpus documents in the graph: {len(corpus_urls)}", flush=True)

    # ---- ARD source text ------------------------------------------------------------
    text_tokens = {}
    scanned = 0
    for fp in sorted(ARD.glob("*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                scanned += 1
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                u = d.get("url")
                if u not in corpus_urls or u in text_tokens:
                    continue
                t = d.get("text") or ""
                text_tokens[u] = len(enc.encode(t, disallowed_special=()))
        print(f"  {fp.name}: matched {len(text_tokens)} so far", flush=True)

    matched = sorted(text_tokens)
    unmatched = len(corpus_urls) - len(matched)
    if not matched:
        fail("no corpus document matched an ARD record by URL", ARD, "download_ard.py")

    # ---- output calibration on the one surviving response ---------------------------
    stem = CALIB.name
    raw = (CALIB / f"{stem}_raw_response.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    parsed = json.loads((CALIB / f"{stem}.json").read_text(encoding="utf-8"))
    calib_out = len(enc.encode(raw, disallowed_special=()))
    calib_elements = len(parsed.get("nodes", [])) + len(parsed.get("edges", []))
    tok_per_element = calib_out / calib_elements

    # ---- per-document bill ----------------------------------------------------------
    per_doc_in, per_doc_out = [], []
    for u in matched:
        per_doc_in.append(prompt_tokens + text_tokens[u])
        per_doc_out.append(
            tok_per_element * (nodes_per_doc[u] + edges_per_doc.get(u, 0))
        )

    def stats(xs):
        xs = sorted(xs)
        return {
            "mean": round(statistics.mean(xs), 1),
            "median": round(statistics.median(xs), 1),
            "p90": round(xs[int(0.90 * len(xs))], 1),
            "p99": round(xs[int(0.99 * len(xs))], 1),
            "max": round(xs[-1], 1),
            "total": round(sum(xs), 1),
        }

    in_s, out_s = stats(per_doc_in), stats(per_doc_out)
    scale = len(corpus_urls) / len(matched)  # extrapolate to the unmatched tail

    bill = {}
    for name, (sync_in, sync_out) in SYNC_RATES.items():
        rate_in = sync_in * BATCH_DISCOUNT
        rate_out = sync_out * BATCH_DISCOUNT
        rows = {}
        for r in REASONING_RATIOS:
            billed_out = out_s["total"] * (1 + r)
            usd = in_s["total"] / 1e6 * rate_in + billed_out / 1e6 * rate_out
            rows[f"reasoning_x{r:g}_visible_output"] = {
                "billed_output_tokens": round(billed_out),
                "usd_over_matched_documents": round(usd, 2),
                "usd_per_1000_documents": round(usd / len(matched) * 1000, 2),
            }
        bill[name] = {
            "batch_rate_usd_per_M_input": rate_in,
            "batch_rate_usd_per_M_output": rate_out,
            "synchronous_rate_usd_per_M": [sync_in, sync_out],
            "by_reasoning_ratio": rows,
            "ASSUMPTION": "reasoning tokens are not recoverable; each row assumes a ratio",
        }

    report = {
        "experiment": "extraction token bill reconstructed from surviving artifacts (W5/Q4)",
        "SCOPE_NOTE": (
            "Input tokens are exact for every document matched to its ARD record. Visible "
            "output tokens are calibrated on ONE surviving raw response. Reasoning tokens "
            "are unrecoverable and are reported as a band of assumptions, never as a "
            "measurement. No dollar figure here is a measured cost."
        ),
        "tokenizer": ENCODING,
        "prompt": {
            "source": str(PROMPT_PY.relative_to(REPO)),
            "tokens": prompt_tokens,
            "note": "sent once per document, ahead of the document text",
        },
        "coverage": {
            "corpus_documents_in_graph": len(corpus_urls),
            "matched_to_an_ARD_record_by_url": len(matched),
            "unmatched": unmatched,
            "match_rate_pct": round(100 * len(matched) / len(corpus_urls), 1),
            "ard_records_scanned": scanned,
            "NOTE": "unmatched documents are excluded from the totals, not imputed; the "
            "extrapolated_to_full_corpus block scales the matched total by the match rate",
        },
        "input_tokens_per_document_EXACT": in_s,
        "output_calibration": {
            "document": stem,
            "visible_output_tokens": calib_out,
            "emitted_elements_nodes_plus_edges": calib_elements,
            "tokens_per_element": round(tok_per_element, 1),
            "CAVEAT": "n = 1. This is the only complete response that survives. Rescale "
            "the output rows if another is recovered.",
        },
        "visible_output_tokens_per_document_CALIBRATED": out_s,
        "elements_per_document": {
            "mean_nodes": round(
                sum(nodes_per_doc[u] for u in matched) / len(matched), 2
            ),
            "mean_edges": round(
                sum(edges_per_doc.get(u, 0) for u in matched) / len(matched), 2
            ),
        },
        "extrapolated_to_full_corpus": {
            "input_tokens": round(in_s["total"] * scale),
            "visible_output_tokens": round(out_s["total"] * scale),
            "scale_factor": round(scale, 4),
        },
        "pricing_ASSUMED_not_measured": {
            "rate_note": RATE_NOTE,
            "tokenizer_caveat": TOKENIZER_CAVEAT,
            "batch_discount_applied": BATCH_DISCOUNT,
            "bill_by_model": bill,
        },
        "linearity": (
            "One call per document with no tool use, no retrieval loop and no multi-turn "
            "control flow, so call count is exactly the document count and the input bill "
            "is the sum of document lengths. The distribution is what matters for a "
            "projection: the mean document is several times the median, so a corpus with "
            "a different length profile scales by total tokens and not by document count."
        ),
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\nprompt: {prompt_tokens} tokens")
    print(
        f"matched {len(matched)}/{len(corpus_urls)} documents ({report['coverage']['match_rate_pct']}%)"
    )
    print(
        f"input  per doc: mean {in_s['mean']:.0f}  median {in_s['median']:.0f}  p90 {in_s['p90']:.0f}"
    )
    print(
        f"output per doc: mean {out_s['mean']:.0f}  (calibrated at {tok_per_element:.1f} tok/element)"
    )
    print(f"input total  : {in_s['total'] / 1e6:.2f}M tokens")
    print(f"output total : {out_s['total'] / 1e6:.2f}M tokens (visible only)")
    print("\nUSD per 1,000 documents at BATCH rates, by reasoning ratio:")
    print(
        f"  {'model':<18}" + "".join(f"{'x' + f'{r:g}':>10}" for r in REASONING_RATIOS)
    )
    for name, v in bill.items():
        cells = "".join(
            f"{v['by_reasoning_ratio'][f'reasoning_x{r:g}_visible_output']['usd_per_1000_documents']:>10.2f}"
            for r in REASONING_RATIOS
        )
        print(f"  {name:<18}" + cells)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
