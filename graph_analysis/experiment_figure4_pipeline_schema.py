#!/usr/bin/env python
"""Figure 4: the two-stage pipeline and what one extraction contains (C-W3 / W-9).

Both simulated reviews note that the paper's only figure is a dataset descriptive, so a
reader has to assemble the node/edge schema, the seven-stage chain and the
extract-then-judge flow entirely from prose. This is the figure a resource paper is
expected to carry.

Panel A  the pipeline: one schema-constrained o3 call per document, then a judge from a
         different provider re-reading the source against the extraction, then three
         meta-graders scoring before and after the judge's proposed repairs.
Panel B  what one extraction contains: the canonical chain in logical-chain order with
         the per-node and per-edge attributes the schema requires.

Palette matches Figure 2 (risk / five-step body ramp in chain order / intervention), so
the two figures read as one system. Class B: no LLM calls, no network, no receipts needed
-- this figure draws the design, not a measurement.

Run from graph_analysis/:  python -u experiment_figure4_pipeline_schema.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "plots" / "figure4_pipeline_schema.png"
OUT_PDF = HERE / "plots" / "figure4_pipeline_schema.pdf"

C_RISK = "#eb6834"
C_INTV = "#1baf7a"
BODY_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK = "#111111"
MUTED = "#555555"
BLUE = "#0072B2"
GREY = "#999999"

STAGES = [
    ("risk", "a named AI\nsafety risk", C_RISK, "white"),
    ("pa", "problem\nanalysis", BODY_RAMP[0], INK),
    ("ti", "theoretical\ninsight", BODY_RAMP[1], "white"),
    ("dr", "design\nrationale", BODY_RAMP[2], "white"),
    ("im", "implementation\nmechanism", BODY_RAMP[3], "white"),
    ("va", "validation\nevidence", BODY_RAMP[4], "white"),
    ("intv", "the proposed\nintervention", C_INTV, "white"),
]


def rbox(ax, x, y, w, h, edge, face="white", lw=1.4, ls="solid", z=3):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.02",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            linestyle=ls,
            zorder=z,
        )
    )


def arrow(ax, x0, y0, x1, y1, color=BLUE, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.3,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.0, 3.9), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    for ax in (axA, axB):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ------------------------------------------------------------------ panel A
    axA.text(
        0.0,
        0.955,
        "A  The two-stage pipeline",
        fontsize=10,
        fontweight="bold",
        ha="left",
    )

    def stage_box(y, h, edge, title, lines, tag=None):
        rbox(axA, 0.10, y, 0.80, h, edge)
        axA.text(
            0.135,
            y + h - 0.055,
            title,
            fontsize=9,
            fontweight="bold",
            color=INK,
            va="top",
        )
        axA.text(
            0.135,
            y + h - 0.135,
            lines,
            fontsize=7.8,
            color=MUTED,
            va="top",
            linespacing=1.35,
        )
        if tag:
            axA.text(
                0.875,
                y + h - 0.055,
                tag,
                fontsize=7.4,
                color=edge,
                ha="right",
                va="top",
                style="italic",
            )

    stage_box(
        0.70,
        0.19,
        GREY,
        "Document",
        "one ARD record, full text, not truncated to its abstract",
    )
    stage_box(
        0.40,
        0.235,
        BLUE,
        "Stage 1  Extraction",
        "o3, reasoning effort medium, one call per document via the batch API\n"
        "no tool use, no retrieval loop, no multi-turn control flow\n"
        "emits the typed graph of panel B; cost is linear in corpus size",
        "extractor",
    )
    stage_box(
        0.115,
        0.235,
        C_INTV,
        "Stage 2  Verification",
        "claude-sonnet-4-5, a different provider, sees source text AND extraction\n"
        "returns schema, referential, orphan, duplicate and coverage findings\n"
        "plus proposed repairs; three meta-graders then score pre and post",
        "judge",
    )
    arrow(axA, 0.50, 0.70, 0.50, 0.638)
    arrow(axA, 0.50, 0.40, 0.50, 0.353)
    axA.text(
        0.515,
        0.668,
        "full text + extraction prompt",
        fontsize=7.4,
        color=INK,
        va="center",
    )
    axA.text(
        0.515,
        0.376,
        "source text + extracted graph",
        fontsize=7.4,
        color=INK,
        va="center",
    )
    axA.text(
        0.10,
        0.055,
        "The verification stage proposes repairs; no repaired graph was rebuilt or re-scored.",
        fontsize=7.4,
        color=MUTED,
        va="center",
    )

    # ------------------------------------------------------------------ panel B
    axB.text(
        0.0,
        0.955,
        "B  What one extraction contains",
        fontsize=10,
        fontweight="bold",
        ha="left",
    )
    n = len(STAGES)
    bw, gap = 0.108, 0.026
    x0 = 0.5 - (n * bw + (n - 1) * gap) / 2
    y, bh = 0.545, 0.235
    for i, (short, label, col, txt) in enumerate(STAGES):
        x = x0 + i * (bw + gap)
        rbox(axB, x, y, bw, bh, col, face=col, lw=0.8)
        axB.text(
            x + bw / 2,
            y + bh * 0.68,
            short,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=txt,
            zorder=4,
        )
        axB.text(
            x + bw / 2,
            y + bh * 0.30,
            label,
            ha="center",
            va="center",
            fontsize=6.6,
            color=txt,
            linespacing=1.2,
            zorder=4,
        )
        if i < n - 1:
            arrow(
                axB, x + bw + 0.003, y + bh / 2, x + bw + gap - 0.003, y + bh / 2, MUTED
            )
    axB.text(
        0.5,
        y + bh + 0.055,
        "the canonical chain, in logical-chain order",
        ha="center",
        fontsize=8,
        color=MUTED,
    )

    rbox(axB, 0.02, 0.245, 0.455, 0.235, GREY, lw=1.1)
    axB.text(
        0.045, 0.435, "every node carries", fontsize=8, fontweight="bold", color=INK
    )
    axB.text(
        0.045,
        0.275,
        "name, aliases, description, source URL\n"
        "concept_category (one of the seven above)\n"
        "interventions also carry maturity 1-4\nand lifecycle 1-6",
        fontsize=7.4,
        color=MUTED,
        va="bottom",
        linespacing=1.4,
    )
    rbox(axB, 0.525, 0.245, 0.455, 0.235, GREY, lw=1.1)
    axB.text(
        0.55, 0.435, "every edge carries", fontsize=8, fontweight="bold", color=INK
    )
    axB.text(
        0.55,
        0.275,
        "a relation type and a description of\nthe claimed relationship\n"
        "edge_confidence 1-5: how explicitly the\npaper asserts the link",
        fontsize=7.4,
        color=MUTED,
        va="bottom",
        linespacing=1.4,
    )
    axB.text(
        0.5,
        0.135,
        "Every edge comes from one paper. The schema permits deviation from the canonical chain:\n"
        "omitted stages and several nodes for one stage are measured, never corrected.",
        ha="center",
        va="center",
        fontsize=7.6,
        color=INK,
        linespacing=1.4,
    )

    fig.tight_layout(pad=0.6)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
