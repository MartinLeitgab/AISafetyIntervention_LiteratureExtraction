#!/usr/bin/env python
"""
Judge validation -- FULL receipt for Paper A (Workshop items 2 and 3).

Consolidates every judge artifact Mike produced into ONE validated summary so the
paper never re-derives from raw data.

Inputs (all local, no network, no LLM calls):

  1. --judge-reports  <dir>   100 Sonnet-4.5 judge reports on successful extractions.
                              Repo/branch source:
                                git archive origin/anthropic_judge_test \
                                  extraction_validator/extend_try_1 | tar -x -C <DEST>
  2. --mike-archive   <dir>   Final-archive-from-Mike/ containing:
                                test_extend_all_evaluation_opus_4_5/      (Opus 4.5 rubric)
                                test_extend_all_evaluation_gemini_pro_3/  (Gemini 3 Pro rubric)
                                extend_try_with_extration_and_judge_and_original_text/
                                                                          (3rd grader *_evaluation.json)
  3. --recovery       <dir>   Mike2/judge_recovery_bundle/data/

Output: graph_analysis/phase2_results/experiment_judge_full_report.json

Fails fast: every input directory must exist. No partial-mode fallback.
"""

import argparse
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "phase2_results" / "experiment_judge_full_report.json"

# The judge's schema_check flags missing inline *_rationale fields as BLOCKER. The
# pipeline stores rationale as separate :Rationale nodes (HAS_RATIONALE), so these are a
# judge/extractor schema-version mismatch, NOT an extraction defect. Split before quoting.
RATIONALE_TOKENS = (
    "node_rationale",
    "edge_rationale",
    "edge_confidence_rationale",
    "intervention_lifecycle_rationale",
    "intervention_maturity_rationale",
)

# Ordinal bands used for Fleiss' kappa on the 0-100 rubric scores. Documented choice:
# kappa needs categories, the rubric is continuous, so we bin. Bands chosen a priori as
# quartile-ish quality tiers, NOT tuned to maximise agreement.
BANDS = [(0, 60, "poor"), (60, 75, "fair"), (75, 85, "good"), (85, 101, "excellent")]


def die(msg):
    raise SystemExit(f"FATAL: {msg}")


def need_dir(p: Path, what: str) -> Path:
    if not p.is_dir():
        die(
            f"{what} not found: {p}\n"
            f"  This script does NOT fall back to cached or partial data.\n"
            f"  See the module docstring for how to produce each input."
        )
    return p


def band(score):
    for lo, hi, name in BANDS:
        if lo <= score < hi:
            return name
    return None


def is_rationale_mismatch(text):
    t = (text or "").lower()
    return any(tok in t for tok in RATIONALE_TOKENS)


def source_type_from_paper(paper: str) -> str:
    return paper.split("__", 1)[0]


# ----------------------------------------------------------------------------------
# Part 1 -- judge audit over the 100 successful extractions
# ----------------------------------------------------------------------------------
def judge_audit(src: Path):
    reports = {}
    for p in sorted(src.glob("*.json")):
        if p.name in ("summary.json", "errors.json"):
            continue
        reports[p.name] = json.loads(p.read_text(encoding="utf-8"))
    if not reports:
        die(f"no judge reports in {src}")

    n = len(reports)
    flags = Counter()
    sev_all, sev_rat, sev_sub = Counter(), Counter(), Counter()
    ref_sev = Counter()
    ref_kind = Counter()
    totals = Counter()
    papers_with = Counter()
    by_src = defaultdict(lambda: Counter())
    by_src_n = Counter()

    for fname, r in reports.items():
        stype = source_type_from_paper(fname)
        by_src_n[stype] += 1
        dec = r.get("decision") or {}
        for f in (
            "is_valid_json",
            "has_blockers",
            "flag_underperformance",
            "valid_and_mergeable_after_fixes",
        ):
            if dec.get(f) is True:
                flags[f] += 1

        vr = r.get("validation_report") or {}
        schema = vr.get("schema_check") or []
        refer = vr.get("referential_check") or []
        cov = (vr.get("coverage") or {}).get("expected_edges_from_source") or []

        n_rat = 0
        for it in schema:
            sev = str(it.get("severity", "UNSPECIFIED")).upper()
            sev_all[sev] += 1
            if is_rationale_mismatch(it.get("issue", "")):
                n_rat += 1
                sev_rat[sev] += 1
            else:
                sev_sub[sev] += 1
        for it in refer:
            sev = str(it.get("severity", "UNSPECIFIED")).upper()
            ref_sev[sev] += 1
            if sev in ("BLOCKER", "MAJOR"):
                ref_kind[str(it.get("issue", ""))[:80]] += 1

        pf = r.get("proposed_fixes") or {}
        fg = r.get("final_graph") or {}
        row = {
            "schema_substantive": len(schema) - n_rat,
            "schema_rationale_mismatch": n_rat,
            "referential": len(refer),
            "orphans": len(vr.get("orphans") or []),
            "duplicates": len(vr.get("duplicates") or []),
            "rationale_mismatches": len(vr.get("rationale_mismatches") or []),
            "coverage_flagged_edges": len(cov),
            "add_nodes": len(pf.get("add_nodes") or []),
            "merges": len(pf.get("merges") or []),
            "node_deletions": len(pf.get("deletions") or []),
            "edge_deletions": len(pf.get("edge_deletions") or []),
            "field_changes": len(pf.get("change_node_fields") or []),
            "final_nodes": len(fg.get("nodes") or []),
            "final_edges": len(fg.get("edges") or []),
        }
        for k, v in row.items():
            totals[k] += v
            by_src[stype][k] += v
            if v > 0:
                papers_with[k] += 1

    return {
        "n_papers": n,
        "judge_model": "claude-sonnet-4-5 (Anthropic batch API)",
        "extraction_model": "o3",
        "source_type_counts": dict(by_src_n),
        "decision_flags": {
            k: {"n": v, "pct": round(100 * v / n, 1)} for k, v in flags.items()
        },
        "totals": dict(totals),
        "papers_with_at_least_one": {
            k: {"n": v, "pct": round(100 * v / n, 1)} for k, v in papers_with.items()
        },
        "schema_severity_ALL": dict(sev_all),
        "schema_severity_rationale_mismatch": dict(sev_rat),
        "schema_severity_substantive": dict(sev_sub),
        "referential_severity": dict(ref_sev),
        "referential_top_issues": dict(ref_kind.most_common(10)),
        "per_paper_means": {k: round(v / n, 2) for k, v in totals.items()},
        "CAVEAT_schema": (
            "Blocker-severity schema flags are dominated by the judge expecting inline "
            "*_rationale fields the pipeline stores as separate :Rationale nodes. Quote "
            "schema_severity_substantive, never schema_severity_ALL, as an error rate. "
            "decision_flags.has_blockers and .is_valid_json inherit the same artifact."
        ),
        "CAVEAT_coverage": (
            "proposed_fixes has NO add_edges key in any report. The judge flags "
            "expected-but-absent edges under coverage but cannot propose them, and they "
            "were never adjudicated -- an upper-bound opinion signal, not an omission rate."
        ),
    }


# ----------------------------------------------------------------------------------
# Part 2 -- three meta-graders: per-paper pre/post rubric scores
# ----------------------------------------------------------------------------------
def _find_scored(obj, want: str):
    """Recursively find a numeric value under a key naming want ('pre'|'post') judge score.

    Mike ran the rubric across several prompt iterations, so these directories hold
    HETEROGENEOUS json shapes (>=5 distinct schemas per directory). Hand-coding one shape
    silently drops the rest, so we search the tree instead and report the misses.
    """
    hits = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                if (
                    want in kl
                    and "judge" in kl
                    and "score" in kl
                    and isinstance(v, (int, float))
                    and not isinstance(v, bool)
                ):
                    hits.append(float(v))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return hits[0] if hits else None


def _find_paper_name(x, fallback: str) -> str:
    for k in ("file", "source_file", "file_name", "file_evaluated"):
        v = x.get(k) if isinstance(x, dict) else None
        if isinstance(v, str) and v:
            return v
    ev = x.get("evaluation") if isinstance(x, dict) else None
    if isinstance(ev, dict):
        for k in ("source_file", "file", "file_name"):
            if isinstance(ev.get(k), str) and ev[k]:
                return ev[k]
    name = fallback
    for pre in ("evaluation_",):
        if name.startswith(pre):
            name = name[len(pre) :]
    return name.replace("_evaluation.json", ".json")


def _count_list_anywhere(obj, key: str) -> int:
    total = 0

    def walk(o):
        nonlocal total
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k).lower() == key and isinstance(v, list):
                    total += len(v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return total


def load_grader(d: Path, pattern: str):
    """Return (rows, diagnostics). Never silently drops a file -- misses are counted."""
    rows, no_pair, unparsed = {}, [], 0
    shapes = Counter()
    for p in sorted(d.glob(pattern)):
        if p.name in ("summary.json", "errors.json"):
            continue
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            unparsed += 1
            continue
        shapes[tuple(sorted(x.keys()))[:6]] += 1
        pre = _find_scored(x, "pre")
        post = _find_scored(x, "post")
        paper = _find_paper_name(x, p.name)
        if pre is None or post is None:
            no_pair.append(p.name)
            continue
        rows[paper] = {
            "pre": pre,
            "post": post,
            "source_type": (x.get("source_type") or source_type_from_paper(paper)),
            "missed_concepts": _count_list_anywhere(x, "missed_concepts"),
            "fabricated_content": _count_list_anywhere(x, "fabricated_content"),
            "category_errors": _count_list_anywhere(x, "category_errors"),
        }
    diag = {
        "files_seen": len(list(d.glob(pattern))),
        "rows_with_pre_and_post": len(rows),
        "files_without_a_pre_post_pair": len(no_pair),
        "files_unparseable": unparsed,
        "n_distinct_json_shapes": len(shapes),
        "SHAPE_NOTE": (
            "Directory holds multiple rubric-prompt iterations with different JSON "
            "schemas. Files without a pre/post pair used an earlier single-score rubric "
            "and CANNOT contribute to a pre/post comparison."
        ),
    }
    return rows, diag


def grader_stats(rows):
    pre = [r["pre"] for r in rows.values()]
    post = [r["post"] for r in rows.values()]
    delta = [b - a for a, b in zip(pre, post)]
    return {
        "n": len(rows),
        "pre_mean": round(st.mean(pre), 2),
        "pre_std": round(st.pstdev(pre), 2),
        "post_mean": round(st.mean(post), 2),
        "post_std": round(st.pstdev(post), 2),
        "delta_mean": round(st.mean(delta), 2),
        "delta_median": round(st.median(delta), 2),
        "pct_improved": round(100 * sum(1 for d in delta if d > 0) / len(delta), 1),
        "pct_unchanged": round(100 * sum(1 for d in delta if d == 0) / len(delta), 1),
        "pct_worse": round(100 * sum(1 for d in delta if d < 0) / len(delta), 1),
    }


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else None


def fleiss_kappa(rating_rows, categories):
    """rating_rows: list of dicts {category: count}; equal n raters per row."""
    n_items = len(rating_rows)
    if not n_items:
        return None
    n_raters = sum(rating_rows[0].values())
    if n_raters < 2:
        return None
    p_j = {}
    for c in categories:
        p_j[c] = sum(row.get(c, 0) for row in rating_rows) / (n_items * n_raters)
    P_i = []
    for row in rating_rows:
        s = sum(row.get(c, 0) ** 2 for c in categories) - n_raters
        P_i.append(s / (n_raters * (n_raters - 1)))
    P_bar = sum(P_i) / n_items
    P_e = sum(v**2 for v in p_j.values())
    if P_e >= 1.0:
        return None
    return round((P_bar - P_e) / (1 - P_e), 3)


def grader_agreement(graders):
    names = list(graders)
    common = set.intersection(*[set(g) for g in graders.values()])
    common = sorted(common)
    out = {"n_common_papers": len(common), "graders": names}
    if len(common) < 5:
        out["note"] = "too few common papers for agreement stats"
        return out

    pair = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa = [graders[a][p]["pre"] for p in common]
            pb = [graders[b][p]["pre"] for p in common]
            qa = [graders[a][p]["post"] for p in common]
            qb = [graders[b][p]["post"] for p in common]
            pair[f"{a} vs {b}"] = {
                "spearman_pre": spearman(pa, pb),
                "spearman_post": spearman(qa, qb),
                "mean_abs_diff_pre": round(
                    sum(abs(x - y) for x, y in zip(pa, pb)) / len(common), 2
                ),
                "mean_abs_diff_post": round(
                    sum(abs(x - y) for x, y in zip(qa, qb)) / len(common), 2
                ),
            }

    cats = [b[2] for b in BANDS]
    rows_pre, rows_post = [], []
    for p in common:
        rp, rq = Counter(), Counter()
        for g in names:
            rp[band(graders[g][p]["pre"])] += 1
            rq[band(graders[g][p]["post"])] += 1
        rows_pre.append(rp)
        rows_post.append(rq)

    dir_rows = []
    for p in common:
        c = Counter()
        for g in names:
            c[
                "improved"
                if graders[g][p]["post"] > graders[g][p]["pre"]
                else "not_improved"
            ] += 1
        dir_rows.append(c)

    unan_improved = sum(1 for c in dir_rows if c["improved"] == len(names))
    return {
        **out,
        "pairwise": pair,
        "fleiss_kappa_pre_bands": fleiss_kappa(rows_pre, cats),
        "fleiss_kappa_post_bands": fleiss_kappa(rows_post, cats),
        "fleiss_kappa_improvement_direction": fleiss_kappa(
            dir_rows, ["improved", "not_improved"]
        ),
        "improvement_direction_note": (
            "kappa is UNDEFINED (null) when every grader says 'improved' on every common "
            "paper: with no variance the chance-agreement term P_e = 1. Unanimity is the "
            "reportable fact -- see pct_papers_all_graders_say_improved -- not a kappa."
            if unan_improved == len(common)
            else "kappa computed over both categories."
        ),
        "bands_used": [{"lo": b[0], "hi": b[1], "label": b[2]} for b in BANDS],
        "pct_papers_all_graders_say_improved": round(
            100 * unan_improved / len(common), 1
        ),
        "KAPPA_NOTE": (
            "Rubric scores are continuous 0-100; Fleiss' kappa requires categories, so "
            "scores are binned into the a-priori bands above (not tuned for agreement). "
            "The improvement-direction kappa is binning-free and is the more robust of "
            "the three."
        ),
    }


def opus_error_taxonomy(opus_rows):
    n = len(opus_rows)
    tot = Counter()
    papers_with = Counter()
    by_src = defaultdict(Counter)
    by_src_n = Counter()
    for paper, r in opus_rows.items():
        stype = r["source_type"]
        by_src_n[stype] += 1
        for k in ("missed_concepts", "fabricated_content", "category_errors"):
            tot[k] += r[k]
            by_src[stype][k] += r[k]
            if r[k] > 0:
                papers_with[k] += 1
    return {
        "grader": "claude-opus-4-5 structured extraction_assessment fields",
        "n_papers": n,
        "mapping_to_workshop_taxonomy": {
            "missed_concepts": "missing nodes/content",
            "fabricated_content": "hallucinated content",
            "category_errors": "wrong concept_category / type assignment",
        },
        "totals": dict(tot),
        "papers_with_at_least_one": {
            k: {"n": v, "pct": round(100 * v / n, 1)} for k, v in papers_with.items()
        },
        "per_paper_mean": {k: round(v / n, 2) for k, v in tot.items()},
        "by_source_type": {
            s: {**dict(by_src[s]), "n_papers": by_src_n[s]} for s in sorted(by_src_n)
        },
        "SCOPE_NOTE": (
            "Auto-derived from one meta-grader's structured findings. This is NOT the "
            "manual 50-instance error taxonomy specified for Workshop item 3 -- it is "
            "un-adjudicated LLM output and no human confirmed any instance."
        ),
    }


# ----------------------------------------------------------------------------------
# Part 3 -- failed-extraction recovery
# ----------------------------------------------------------------------------------
def recovery_stats(root: Path):
    cand = need_dir(root / "extraction_error_recoverable_info", "recovery candidates")
    attempts = need_dir(root / "recovered_errors", "recovery attempts")
    recovered = need_dir(root / "recovered_errors_graph", "recovered graphs")

    cand_names = {p.name + ".json" for p in cand.iterdir() if p.is_dir()}
    attempt_files = [
        p
        for p in sorted(attempts.glob("*.json"))
        if p.name not in ("summary.json", "errors.json")
    ]
    rec_files = sorted(recovered.glob("*.json"))
    attempt_names = {p.name for p in attempt_files}
    rec_names = {p.name for p in rec_files}

    # ---- population A: the extraction_error judge-able candidates -------------------
    by_src_attempt = Counter()
    succeeded, sizes_a = [], []
    for p in attempt_files:
        by_src_attempt[source_type_from_paper(p.name)] += 1
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        fg = x.get("final_graph") or {}
        n_nodes = len(fg.get("nodes") or [])
        if n_nodes:
            succeeded.append(p.name)
            sizes_a.append((n_nodes, len(fg.get("edges") or [])))
    by_src_ok = Counter(source_type_from_paper(n) for n in succeeded)

    # ---- population B: recovered_errors_graph --------------------------------------
    nodes, edges = [], []
    by_src_rec = Counter()
    for p in rec_files:
        by_src_rec[source_type_from_paper(p.name)] += 1
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        fg = x.get("final_graph") or {}
        nodes.append(len(fg.get("nodes") or []))
        edges.append(len(fg.get("edges") or []))

    overlap = rec_names & attempt_names
    return {
        "PROVENANCE_FINDING": (
            "The two output directories are DISJOINT: |recovered_errors_graph n "
            "recovered_errors| = "
            f"{len(overlap)} of {len(rec_names)}. recovered_errors_graph also contains "
            "source types absent (agisf) or near-absent (arxiv: 14 files vs 2 candidates) "
            "from the candidate set, so it is NOT the success subset of these attempts -- "
            "it comes from a different failure population, most likely the graph_error set "
            "(91 dirs) the bundle README says it excluded. THEREFORE the widely-quoted "
            "'~60 recovered / ~400 processable ~= 15%' figure divides two different "
            "populations and is NOT supported by this bundle."
        ),
        "population_A_extraction_error_candidates": {
            "n_judgeable_candidates": len(cand_names),
            "n_judge_attempts": len(attempt_files),
            "attempts_match_candidates": len(attempt_names & cand_names)
            == len(attempt_names),
            "n_attempts_producing_nonempty_graph": len(succeeded),
            "recovery_rate_pct": round(100 * len(succeeded) / len(attempt_files), 1)
            if attempt_files
            else None,
            "recovered_graph_size": {
                "mean_nodes": round(st.mean([a for a, _ in sizes_a]), 2)
                if sizes_a
                else None,
                "mean_edges": round(st.mean([b for _, b in sizes_a]), 2)
                if sizes_a
                else None,
            },
            "by_source_type": {
                s: {
                    "attempts": by_src_attempt[s],
                    "recovered": by_src_ok.get(s, 0),
                    "pct": round(100 * by_src_ok.get(s, 0) / by_src_attempt[s], 1),
                }
                for s in sorted(by_src_attempt)
            },
        },
        "population_B_recovered_errors_graph": {
            "n_files": len(rec_files),
            "n_also_in_population_A": len(overlap),
            "denominator": "UNKNOWN -- input population not included in this bundle",
            "recovery_rate_pct": None,
            "graph_size": {
                "mean_nodes": round(st.mean(nodes), 2) if nodes else None,
                "mean_edges": round(st.mean(edges), 2) if edges else None,
                "median_nodes": round(st.median(nodes), 2) if nodes else None,
                "median_edges": round(st.median(edges), 2) if edges else None,
            },
            "by_source_type": dict(by_src_rec),
        },
        "SCOPE_NOTE": (
            "Population A's denominator is the judge-able candidate set (extraction failed "
            "BUT a URL, non-empty source text and >=1 extracted node exist). The larger "
            "unfiltered failure set (processed_ard/extraction_error, 1667 dirs) is "
            "dominated by ARD records with no source text at all, not recoverable in "
            "principle, and excluded from this denominator."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-reports", required=True)
    ap.add_argument("--mike-archive", required=True)
    ap.add_argument("--recovery", required=True)
    a = ap.parse_args()

    jr = need_dir(Path(a.judge_reports), "judge reports dir")
    ma = need_dir(Path(a.mike_archive), "Mike archive dir")
    rc = need_dir(Path(a.recovery), "recovery bundle data dir")

    opus, opus_diag = load_grader(
        need_dir(ma / "test_extend_all_evaluation_opus_4_5", "opus rubric dir"),
        "*.json",
    )
    gemini, gemini_diag = load_grader(
        need_dir(ma / "test_extend_all_evaluation_gemini_pro_3", "gemini rubric dir"),
        "*.json",
    )
    third, third_diag = load_grader(
        need_dir(
            ma / "extend_try_with_extration_and_judge_and_original_text",
            "third-grader dir",
        ),
        "*_evaluation.json",
    )
    for name, g in (("opus", opus), ("gemini", gemini), ("third", third)):
        if not g:
            die(f"grader '{name}' produced zero parsed rows")

    graders = {
        "claude-opus-4-5": opus,
        "gemini-3-pro": gemini,
        "third_grader_gpt-5.1": third,
    }
    grader_diags = {
        "claude-opus-4-5": opus_diag,
        "gemini-3-pro": gemini_diag,
        "third_grader_gpt-5.1": third_diag,
    }

    out = {
        "experiment": "Judge validation -- full receipt (Workshop items 2 + 3)",
        "inputs": {
            "judge_reports": str(jr),
            "mike_archive": str(ma),
            "recovery_bundle": str(rc),
        },
        "item2_judge_audit": judge_audit(jr),
        "item2_meta_graders": {
            name: {**grader_stats(rows), "coverage_diagnostics": grader_diags[name]}
            for name, rows in graders.items()
        },
        "item2_meta_grader_CAVEAT": (
            "The three graders scored DIFFERENT, unequal subsets of the 100 papers, "
            "because each directory mixes several rubric-prompt iterations and only some "
            "iterations emit a pre/post pair. The aggregate means in "
            "extraction_validator/results.md are therefore NOT computed on a common "
            "sample and must not be compared head-to-head as if they were."
        ),
        "item2_grader_agreement": grader_agreement(graders),
        "item3_error_taxonomy_auto": opus_error_taxonomy(opus),
        "item3_recovery": recovery_stats(rc),
        "STILL_OPEN": [
            "Manual 50-instance error taxonomy with human adjudication (item 3 as specced).",
            "Human-anchor spot-check of the judge (no human ground truth anywhere in this receipt).",
            "Confirm the third grader's exact model id (results.md says GPT5.1, STATUS.md says GPT-5.2).",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    json.dump(out, sys.stdout, indent=1)
    print(f"\n\nwrote {OUT}")


if __name__ == "__main__":
    main()
