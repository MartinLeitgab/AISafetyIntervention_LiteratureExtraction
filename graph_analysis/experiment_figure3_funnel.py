#!/usr/bin/env python
"""Figure 3: the data-reduction funnel (reviewer item C-W2 / W-9).

Both simulated reviews name this as the single highest-value addition to the paper: four
differently-named reduction operations are described only in prose -- quality cuts,
containment de-duplication of paths, the node merge (measured but NOT applied to this
substrate), and similarity thresholding -- and readers conflate them.

The figure draws two lanes, documents and chains, with every arrow labelled by its
operation and marked APPLIED or MEASURED ONLY.

Plots straight from committed receipts. Class B: no LLM calls, no network, fails fast on a
missing receipt. White background, Okabe-Ito palette (same as Figures 1 and 2).

Run from graph_analysis/:  python -u experiment_figure3_funnel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
RES = HERE / "phase2_results"
AUDIT = RES / "experiment_paper_claim_audit.json"
INTAKE = RES / "experiment_review_intake_recency_report.json"
GATES = RES / "experiment_review_gate_sensitivity_report.json"
OUT_PNG = HERE / "plots" / "figure3_funnel.png"
OUT_PDF = HERE / "plots" / "figure3_funnel.pdf"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREY = "#999999"
VERM = "#D55E00"
INK = "#111111"
MUTED = "#555555"


def need(path: Path, produced_by: str) -> dict:
    if not path.exists():
        sys.stderr.write(
            f"\nFATAL: missing receipt {path}\n  produced by: {produced_by}\n"
            "  this script does NOT re-derive it and does NOT draw a partial figure.\n\n"
        )
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def box(ax, x, y, w, h, label, value, color, fill="white", lw=1.4):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=lw,
            edgecolor=color,
            facecolor=fill,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.62,
        value,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + h * 0.245,
        label,
        ha="center",
        va="center",
        fontsize=8.2,
        color=MUTED,
        zorder=4,
        linespacing=1.25,
    )


def arrow(ax, x0, y0, x1, y1, text, tag, color=BLUE, dy=0.055):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.3,
            color=color,
            zorder=2,
        )
    )
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(
        xm,
        ym + dy,
        text,
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=INK,
        linespacing=1.2,
    )
    ax.text(
        xm,
        ym - dy * 0.62,
        tag,
        ha="center",
        va="top",
        fontsize=7.0,
        color=color,
        style="italic",
    )


def main() -> None:
    audit = need(AUDIT, "graph_analysis/experiment_paper_claim_audit.py")
    intake = need(INTAKE, "graph_analysis/experiment_review_intake_and_recency.py")
    gates = need(GATES, "graph_analysis/experiment_review_gate_sensitivity.py")

    ded = audit["path_set_comparison"]["deduped"]
    raw = audit["path_set_comparison"]["raw"]
    fn = intake["intake_funnel"]
    n_records = fn["ard_records_taken_in_RECONSTRUCTED"]
    n_docs = fn["documents_with_a_parseable_extraction_REDERIVED"]
    n_chain_docs = fn["documents_yielding_a_quality_cut_chain_REDERIVED"]
    n_nodes = 200525
    n_raw = raw["n_paths"]
    n_ded = ded["n_paths"]
    merge_removed = 2385
    cont_lost_pct = gates["containment_losslessness"]["pct_of_raw_chain_set_nodes_lost"]

    plt.rcParams.update(
        {
            "font.size": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(13.0, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w, h = 0.135, 0.155
    y_doc, y_chain = 0.72, 0.34
    xs = [0.015, 0.275, 0.535, 0.850]  # documents lane columns

    # ---- documents lane ---------------------------------------------------------
    ax.text(
        0.015,
        y_doc + h + 0.055,
        "DOCUMENTS",
        fontsize=8.5,
        fontweight="bold",
        color=MUTED,
    )
    box(
        ax,
        xs[0],
        y_doc,
        w,
        h,
        "ARD records taken in\n(reconstructed)",
        f"{n_records:,}",
        GREY,
    )
    box(
        ax,
        xs[1],
        y_doc,
        w,
        h,
        "parseable extractions\n= the corpus",
        f"{n_docs:,}",
        BLUE,
    )
    box(
        ax,
        xs[2],
        y_doc,
        w,
        h,
        "typed nodes extracted\n19,096 risk / 36,959 intv.",
        f"{n_nodes:,}",
        BLUE,
    )
    box(
        ax,
        xs[3],
        y_doc,
        w,
        h,
        "documents yielding\na chain (15.9%)",
        f"{n_chain_docs:,}",
        ORANGE,
        lw=1.8,
    )
    arrow(
        ax,
        xs[0] + w + 0.012,
        y_doc + h / 2,
        xs[1] - 0.012,
        y_doc + h / 2,
        "one o3 call\nper document",
        "APPLIED",
        dy=0.02,
    )
    arrow(
        ax,
        xs[1] + w + 0.012,
        y_doc + h / 2,
        xs[2] - 0.012,
        y_doc + h / 2,
        "schema-constrained\nparse + embed",
        "APPLIED",
        dy=0.02,
    )

    # ---- chains lane ------------------------------------------------------------
    ax.text(
        0.015,
        y_chain + h + 0.055,
        "CHAINS",
        fontsize=8.5,
        fontweight="bold",
        color=MUTED,
    )
    box(
        ax,
        xs[1],
        y_chain,
        w,
        h,
        "paths enumerated\nhop-wise DFS, EDGE only",
        f"{n_raw:,}",
        BLUE,
    )
    box(
        ax,
        xs[2],
        y_chain,
        w,
        h,
        "reporting unit\nevery number below",
        f"{n_ded:,}",
        ORANGE,
        lw=1.8,
    )
    ax.add_patch(
        FancyArrowPatch(
            (xs[1] + w + 0.012, y_chain + h / 2),
            (xs[2] - 0.012, y_chain + h / 2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.3,
            color=BLUE,
            zorder=2,
        )
    )
    x_mid = (xs[1] + w + xs[2]) / 2
    ax.text(
        x_mid,
        y_chain + h / 2 + 0.018,
        "APPLIED",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=BLUE,
        style="italic",
    )
    ax.text(
        x_mid,
        y_chain - 0.02,
        f"containment de-duplication (70%):\nwithin-paper sub-paths dropped,\n{cont_lost_pct}% of chain-set nodes lost",
        ha="center",
        va="top",
        fontsize=7.8,
        color=INK,
        linespacing=1.25,
    )
    # corpus/nodes -> path enumeration
    ax.add_patch(
        FancyArrowPatch(
            (xs[2] + w * 0.30, y_doc - 0.005),
            (xs[1] + w * 0.72, y_chain + h + 0.005),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.3,
            color=BLUE,
            connectionstyle="arc3,rad=0.18",
            zorder=2,
        )
    )
    ax.text(
        0.472,
        (y_doc + y_chain + h) / 2 + 0.012,
        "quality cuts: edge confidence >= 3, intervention\nmaturity >= 3, exactly one risk node, at the root",
        ha="left",
        va="center",
        fontsize=7.8,
        color=INK,
        linespacing=1.25,
    )
    ax.text(
        0.472,
        (y_doc + y_chain + h) / 2 - 0.052,
        "APPLIED   (both gates are LLM-assigned; see the sensitivity table)",
        ha="left",
        va="center",
        fontsize=7.0,
        color=BLUE,
        style="italic",
    )
    # reporting unit -> chain-yielding documents
    ax.add_patch(
        FancyArrowPatch(
            (xs[2] + w + 0.008, y_chain + h * 0.55),
            (xs[3] + w * 0.35, y_doc - 0.008),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.3,
            color=ORANGE,
            connectionstyle="arc3,rad=-0.20",
            zorder=2,
        )
    )
    ax.text(
        0.775,
        0.60,
        "grouped by\nsource document",
        ha="left",
        va="center",
        fontsize=7.8,
        color=INK,
        linespacing=1.25,
    )

    # ---- measured-only operations ------------------------------------------------
    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.015),
            0.72,
            0.175,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.3,
            edgecolor=VERM,
            facecolor="white",
            linestyle=(0, (4, 2)),
            zorder=3,
        )
    )
    ax.text(
        0.375,
        0.152,
        "MEASURED, NOT APPLIED TO THE RELEASED SUBSTRATE",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=VERM,
        zorder=4,
    )
    ax.text(
        0.375,
        0.077,
        f"node merge (cosine >= 0.88 and Jaccard >= 0.05) would remove {merge_removed:,} nodes and manufacture a centrality super-hub;\n"
        "similarity edges (cosine >= 0.80) join documents but enter no chain reported here.\n"
        "Both are costed in the use-case section. The node inventory above is the un-merged count.",
        ha="center",
        va="center",
        fontsize=7.2,
        color=INK,
        linespacing=1.4,
        zorder=4,
    )

    fig.tight_layout(pad=0.4)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
