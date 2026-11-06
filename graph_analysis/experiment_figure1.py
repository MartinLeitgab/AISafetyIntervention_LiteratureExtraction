#!/usr/bin/env python
"""The corpus figure for the paper: what the extraction produced, in two panels.

Panel A  the share of chains carrying all five intermediate stages, under the nine gate
         settings of tab:gates (corpus) and under the three arms of the 30-document
         schema ablation, with each row's chain yield beside it. It carries two body
         claims in one view: the gates move yield but not stage completeness, and stage
         completeness comes from the prompt. REVISED 2026-11-06 (v5): this replaced a
         chain-length histogram, a secondary statistic that carried no body claim.
Panel B  the joint distribution of the two attributes that gate the chain set --
         intervention maturity against the best edge confidence reaching the
         intervention -- with the gate corner outlined. This is the quality cut made
         visible: the corner is 11.4% of extracted interventions.

Plots straight from the committed receipt JSONs, so the figure is reproducible without
the two large PKL checkpoints.

REVISED 2026-08-15. The previous version had three panels. Panel B was a single stacked
bar of intervention maturity, an attribute the paper says four times is unvalidated, and
panel C was a two-bar rendering of one ratio (1,868 of 11,779). Both spent page-one space
on quantities a sentence carries. The maturity marginal is now one axis of the joint, and
the yield ratio lives in the text.

NOTE 2026-08-11: an earlier version of the third panel plotted a scope composition
(capability-gap vs human-harm share) derived from the Paper-B routing pass. It was
withdrawn because the routing assignment file is not committed and cannot be re-derived
by a reader at reasonable cost.

Class B: no LLM calls, no network.  Fails fast on a missing receipt.
White background, colour-blind-safe palette, axis text at paper body size.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RES = HERE / "phase2_results"
GATES = RES / "experiment_review_gate_sensitivity_report.json"
ABLATION = RES / "experiment_review_schema_ablation_report.json"
CORNER = RES / "experiment_review_gate_corner_report.json"
OUT_PNG = HERE / "plots" / "figure1_dataset.png"
OUT_PDF = HERE / "plots" / "figure1_dataset.pdf"

# Okabe-Ito: safe under every common form of colour vision deficiency. Validated with
# the dataviz validator (light surface): CVD dE 29.2 worst adjacent, normal dE 36.2,
# both PASS. The orange contrast WARN is relieved by the direct label on that bar.
BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#333333"


def need(path: Path, produced_by: str) -> dict:
    if not path.exists():
        sys.stderr.write(
            f"\nFATAL: missing receipt {path}\n"
            f"  produced by: {produced_by}\n"
            f"  this script does NOT re-derive it from raw data and does NOT\n"
            f"  substitute a cached or partial figure.\n\n"
        )
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    gates = need(GATES, "graph_analysis/experiment_review_gate_sensitivity.py")
    ablation = need(ABLATION, "graph_analysis/experiment_review_schema_ablation.py")
    corner = need(CORNER, "graph_analysis/experiment_review_gate_corner.py")

    grid9 = gates["gate_grid"]
    as_run = grid9["conf>=3, maturity>=3"]
    others = [v for k, v in grid9.items() if k != "conf>=3, maturity>=3"]
    if len(others) != 8:
        raise SystemExit(f"FATAL: expected 9 gate settings, found {len(others) + 1}")
    head = ablation["results"]["headline"]
    arms = [
        ("released extraction", head["A_released_on_the_EF_sample"]),
        ("sentence-shuffled source", head["F"]),
        ("prompt without stage names", head["E"]),
    ]

    joint = corner["joint_maturity_by_best_incident_confidence"]
    placed = corner["n_with_at_least_one_structural_edge"]
    corner_n = corner["gate_corner_maturity_ge3_and_best_conf_ge3"]["n"]
    corner_pct = corner["gate_corner_maturity_ge3_and_best_conf_ge3"]["pct_of_placed"]

    # figsize is the printed size (figure* at \textwidth = 7.0 in), so a font size here is
    # the size on the page; nothing below 7 pt.
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # ---------------------------------------------------------------- panel A
    # Rows top to bottom. Row 0 is the corpus under all nine gate settings; rows 1-3 are
    # the 30-document ablation sample, whose released row is the paired baseline.
    y_gate, y_arms = 3.0, [1.9, 1.0, 0.1]
    ax_a.axhline(2.45, color="#cccccc", lw=0.6)
    ax_a.scatter(
        [v["all5_pct"] for v in others],
        [y_gate] * len(others),
        s=26,
        color=BLUE,
        alpha=0.75,
        zorder=3,
        label="other eight gate settings",
    )
    ax_a.scatter(
        [as_run["all5_pct"]],
        [y_gate],
        s=46,
        color=ORANGE,
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
        label="setting as run",
    )
    lo = min([v["corpus_yield_pct"] for v in others] + [as_run["corpus_yield_pct"]])
    hi = max([v["corpus_yield_pct"] for v in others] + [as_run["corpus_yield_pct"]])
    ax_a.text(
        3,
        y_gate + 0.42,
        f"corpus, nine gate settings: yield {lo}% to {hi}% of documents",
        fontsize=7.2,
        color=INK,
        va="center",
    )
    for (label, arm), y in zip(arms, y_arms):
        ax_a.scatter([arm["pct_chains_all_five"]], [y], s=34, color=BLUE, zorder=3)
        ax_a.text(
            3,
            y + 0.36,
            f"{label}: chain yield {arm['pct_of_attempted_yielding_a_chain']:.0f}%",
            fontsize=7.2,
            color=INK,
            va="center",
        )
    ax_a.text(
        101,
        1.0,
        "30-document\nablation sample",
        fontsize=7,
        color="#666666",
        rotation=270,
        va="center",
        ha="left",
    )
    ax_a.set_xlim(-3, 100)
    ax_a.set_ylim(-0.35, 3.7)
    ax_a.set_yticks([])
    ax_a.spines["left"].set_visible(False)
    ax_a.set_xlabel("chains carrying all five intermediate stages (%)")
    ax_a.set_title(
        "A  Chains carrying all five stages",
        loc="left",
        fontweight="bold",
    )
    ax_a.grid(axis="x", color="#e6e6e6", lw=0.6)
    ax_a.set_axisbelow(True)
    # Direct label instead of a legend box: one highlighted point, named where it sits.
    ax_a.annotate(
        "as run",
        xy=(as_run["all5_pct"], y_gate),
        xytext=(as_run["all5_pct"] - 14, y_gate - 0.3),
        fontsize=7,
        color="#9a6200",
        va="center",
        arrowprops=dict(arrowstyle="-", color="#9a6200", lw=0.7),
    )

    # ---------------------------------------------------------------- panel B
    # Sequential, one hue, light -> dark. Counts are printed in every cell, so identity
    # never rests on colour alone.
    mats = [1, 2, 3, 4]
    confs = [1, 2, 3, 4, 5]
    grid = [[joint[str(m)][str(c)] for c in confs] for m in mats]
    vmax = max(max(row) for row in grid)
    im = ax_b.imshow(
        grid, cmap="Blues", origin="lower", aspect="auto", vmin=0, vmax=vmax
    )
    for i, m in enumerate(mats):
        for j, c in enumerate(confs):
            v = grid[i][j]
            ax_b.text(
                j,
                i,
                f"{v:,}" if v else "0",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if v > 0.55 * vmax else "#222222",
            )
    # Row totals replace a colorbar: every cell is labeled, so the bar added nothing,
    # and the totals are the maturity distribution sec:r-corpus quotes.
    total = sum(sum(row) for row in grid)
    for i, m in enumerate(mats):
        rt = sum(grid[i])
        ax_b.text(
            4.62,
            i,
            f"{rt:,} ({100 * rt / total:.1f}%)",
            ha="left",
            va="center",
            fontsize=7.2,
            color=INK,
        )
    ax_b.text(
        4.62, -0.88, "row total", ha="left", va="center", fontsize=7, color="#666666"
    )
    ax_b.add_patch(
        Rectangle(
            (1.5, 1.5),
            3.0,
            2.0,
            fill=False,
            edgecolor=ORANGE,
            lw=2.0,
            zorder=5,
        )
    )
    ax_b.set_xticks(range(len(confs)), [str(c) for c in confs])
    ax_b.set_yticks(range(len(mats)), ["1", "2", "3", "4"])
    ax_b.set_xlabel("best edge confidence reaching the intervention")
    ax_b.set_ylabel("intervention maturity")
    ax_b.set_title("B  What the two gates admit", loc="left", fontweight="bold")
    ax_b.annotate(
        f"gate corner: {corner_n:,} of {placed:,} ({corner_pct:.1f}%)",
        xy=(4.5, 3.75),
        fontsize=7.4,
        color="#9a6200",
        fontweight="bold",
        ha="right",
        va="center",
    )
    ax_b.set_ylim(-0.5, 4.2)
    ax_b.set_xlim(-0.5, 4.5)
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(True)
        ax_b.spines[sp].set_color("#cccccc")
    del im

    fig.tight_layout(pad=0.5, w_pad=1.2)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
