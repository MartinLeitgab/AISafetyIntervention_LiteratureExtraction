#!/usr/bin/env python
"""
Workshop Item 2 -- quantitative summary of the 100-paper judge run.

Input:  the 100 Anthropic Sonnet 4.5 judge reports produced by Mike (sub-project B,
        2025-11-26 onward), living on branch `anthropic_judge_test` at
        extraction_validator/extend_try_1/.

        Recreate the input directory with:
            git archive origin/anthropic_judge_test extraction_validator/extend_try_1 \
                | tar -x -C <DEST>

        Then run:
            python experiment_judge_item2_summary.py <DEST>/extraction_validator/extend_try_1

Output: graph_analysis/phase2_results/experiment_judge_item2_report.json

This script does NOT compute inter-grader kappa (that needs the per-file rubric JSONs
from the three meta-graders, which are on Mike's local disk -- see
extraction_validator/STATUS.md §4, ticket #147). It computes everything that the
100 judge reports alone can support.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPORT_OUT = (
    Path(__file__).parent / "phase2_results" / "experiment_judge_item2_report.json"
)


def load_reports(src: Path):
    if not src.is_dir():
        raise SystemExit(
            f"FATAL: judge report directory not found: {src}\n"
            f"  Produce it with:\n"
            f"    git archive origin/anthropic_judge_test "
            f"extraction_validator/extend_try_1 | tar -x -C <DEST>\n"
            f"  This script does NOT fall back to any cached or partial copy."
        )
    reports = {}
    for p in sorted(src.glob("*.json")):
        if p.name in ("summary.json", "errors.json"):
            continue
        with p.open(encoding="utf-8") as fh:
            reports[p.name] = json.load(fh)
    if not reports:
        raise SystemExit(f"FATAL: no judge reports found in {src}")
    return reports


def source_type(filename: str) -> str:
    return filename.split("__", 1)[0]


# The judge's schema_check flags missing *_rationale fields as BLOCKER. The extraction
# pipeline stores rationale as separate :Rationale nodes linked by HAS_RATIONALE rather
# than as inline fields, so these are a judge/extractor schema-version mismatch, NOT an
# extraction-quality defect. They must be separated before any error rate is quoted.
RATIONALE_MISMATCH_TOKENS = (
    "node_rationale",
    "edge_rationale",
    "edge_confidence_rationale",
    "intervention_lifecycle_rationale",
    "intervention_maturity_rationale",
)


def is_rationale_field_mismatch(issue_text: str) -> bool:
    t = (issue_text or "").lower()
    return any(tok in t for tok in RATIONALE_MISMATCH_TOKENS)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <path-to-extend_try_1>")
    reports = load_reports(Path(sys.argv[1]))

    n = len(reports)
    decision_flags = Counter()
    sev_schema = Counter()
    sev_schema_rationale = Counter()
    sev_schema_substantive = Counter()
    sev_referential = Counter()
    finding_counts = Counter()
    fix_counts = Counter()
    per_paper = {}
    by_src_raw = defaultdict(list)

    # every paper contributes one row; totals are sums over papers
    for fname, r in reports.items():
        dec = r.get("decision") or {}
        for flag in (
            "is_valid_json",
            "has_blockers",
            "flag_underperformance",
            "valid_and_mergeable_after_fixes",
        ):
            if dec.get(flag) is True:
                decision_flags[flag] += 1

        vr = r.get("validation_report") or {}
        schema = vr.get("schema_check") or []
        refer = vr.get("referential_check") or []
        orphans = vr.get("orphans") or []
        dups = vr.get("duplicates") or []
        rat = vr.get("rationale_mismatches") or []
        cov = (vr.get("coverage") or {}).get("expected_edges_from_source") or []

        n_schema_rationale = 0
        for item in schema:
            sev = str(item.get("severity", "UNSPECIFIED")).upper()
            sev_schema[sev] += 1
            if is_rationale_field_mismatch(item.get("issue", "")):
                n_schema_rationale += 1
                sev_schema_rationale[sev] += 1
            else:
                sev_schema_substantive[sev] += 1
        for item in refer:
            sev_referential[str(item.get("severity", "UNSPECIFIED")).upper()] += 1

        pf = r.get("proposed_fixes") or {}
        add_nodes = pf.get("add_nodes") or []
        merges = pf.get("merges") or []
        deletions = pf.get("deletions") or []
        edge_deletions = pf.get("edge_deletions") or []
        field_changes = pf.get("change_node_fields") or []

        fg = r.get("final_graph") or {}
        n_nodes = len(fg.get("nodes") or [])
        n_edges = len(fg.get("edges") or [])

        row = {
            "source_type": source_type(fname),
            "schema_issues": len(schema),
            "schema_issues_rationale_mismatch": n_schema_rationale,
            "schema_issues_substantive": len(schema) - n_schema_rationale,
            "referential_issues": len(refer),
            "orphans": len(orphans),
            "duplicates": len(dups),
            "rationale_mismatches": len(rat),
            "missing_edges_flagged": len(cov),
            "add_nodes": len(add_nodes),
            "merges": len(merges),
            "node_deletions": len(deletions),
            "edge_deletions": len(edge_deletions),
            "field_changes": len(field_changes),
            "final_nodes": n_nodes,
            "final_edges": n_edges,
            "has_blockers": bool(dec.get("has_blockers")),
        }
        per_paper[fname] = row
        by_src_raw[row["source_type"]].append(row)

        for k in (
            "schema_issues",
            "schema_issues_rationale_mismatch",
            "schema_issues_substantive",
            "referential_issues",
            "orphans",
            "duplicates",
            "rationale_mismatches",
            "missing_edges_flagged",
        ):
            finding_counts[k] += row[k]
        for k in (
            "add_nodes",
            "merges",
            "node_deletions",
            "edge_deletions",
            "field_changes",
        ):
            fix_counts[k] += row[k]

    def pct(x):
        return round(100.0 * x / n, 1)

    # papers with >=1 finding of each kind
    papers_with = {
        k: sum(1 for r in per_paper.values() if r[k] > 0)
        for k in (
            "schema_issues",
            "schema_issues_rationale_mismatch",
            "schema_issues_substantive",
            "referential_issues",
            "orphans",
            "duplicates",
            "rationale_mismatches",
            "missing_edges_flagged",
            "add_nodes",
            "merges",
            "node_deletions",
            "edge_deletions",
            "field_changes",
        )
    }

    by_source_type = {}
    for st, rows in sorted(by_src_raw.items(), key=lambda kv: -len(kv[1])):
        k = len(rows)
        by_source_type[st] = {
            "n_papers": k,
            "pct_with_blockers": round(
                100.0 * sum(1 for r in rows if r["has_blockers"]) / k, 1
            ),
            "mean_missing_edges_flagged": round(
                sum(r["missing_edges_flagged"] for r in rows) / k, 2
            ),
            "mean_add_nodes": round(sum(r["add_nodes"] for r in rows) / k, 2),
            "mean_final_nodes": round(sum(r["final_nodes"] for r in rows) / k, 1),
            "mean_final_edges": round(sum(r["final_edges"] for r in rows) / k, 1),
        }

    out = {
        "experiment": "Workshop Item 2 -- 100-paper judge run quantitative summary",
        "source": "branch anthropic_judge_test :: extraction_validator/extend_try_1/",
        "judge_model": "claude-sonnet-4-5 (Anthropic batch API)",
        "extraction_model": "o3",
        "n_papers": n,
        "decision_flags": {
            k: {"n": v, "pct": pct(v)} for k, v in sorted(decision_flags.items())
        },
        "findings_total": dict(finding_counts),
        "findings_papers_with_at_least_one": {
            k: {"n": v, "pct": pct(v)} for k, v in sorted(papers_with.items())
        },
        "schema_check_severity_ALL": dict(sev_schema),
        "schema_check_severity_rationale_field_mismatch": dict(sev_schema_rationale),
        "schema_check_severity_substantive": dict(sev_schema_substantive),
        "SCHEMA_CAVEAT": (
            "The judge grades against a schema carrying inline *_rationale fields. The "
            "extraction pipeline stores rationale as separate :Rationale nodes linked by "
            "HAS_RATIONALE, so those flags are a judge/extractor schema-version mismatch, "
            "not an extraction defect. Quote only the 'substantive' split as an error rate."
        ),
        "NOTE_no_add_edges_field": (
            "proposed_fixes has no add_edges key in any of the 100 reports (keys: add_nodes, "
            "merges, deletions, edge_deletions, change_node_fields). The judge can flag "
            "expected-but-missing edges under validation_report.coverage but has no schema "
            "slot to propose them -- a judge design gap, not an extraction property."
        ),
        "referential_check_severity": dict(sev_referential),
        "proposed_fix_totals": dict(fix_counts),
        "per_paper_means": {
            k: round(sum(r[k] for r in per_paper.values()) / n, 2)
            for k in (
                "final_nodes",
                "final_edges",
                "missing_edges_flagged",
                "add_nodes",
                "rationale_mismatches",
            )
        },
        "by_source_type": by_source_type,
        "NOT_COMPUTED_HERE": [
            "inter-meta-grader Fleiss kappa (needs per-file rubric JSONs, Mike's disk)",
            "manual 50-instance error taxonomy (ticket #147 item 3)",
            "pre/post rubric scores per paper (only aggregate mean+/-std in results.md)",
        ],
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_OUT.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print(f"\nwrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
