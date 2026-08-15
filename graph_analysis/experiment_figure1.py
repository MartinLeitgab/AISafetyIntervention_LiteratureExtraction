#!/usr/bin/env python
"""The corpus figure for the paper: what the extraction produced, in two panels.

Panel A  chain-length distribution of the 2,772 collapsed chains on a log count axis,
         with the canonical seven-node chain marked. Log scale because the distribution
         spans three orders of magnitude and a linear axis hides everything but the mode.
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
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
RES = HERE / "phase2_results"
AUDIT = RES / "experiment_paper_claim_audit.json"
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
    audit = need(AUDIT, "graph_analysis/experiment_paper_claim_audit.py")
    corner = need(CORNER, "graph_analysis/experiment_review_gate_corner.py")

    dedup = audit["path_set_comparison"]["deduped"]
    hist = {int(k): v for k, v in dedup["len_hist"].items()}

    joint = corner["joint_maturity_by_best_incident_confidence"]
    placed = corner["n_with_at_least_one_structural_edge"]
    corner_n = corner["gate_corner_maturity_ge3_and_best_conf_ge3"]["n"]
    corner_pct = corner["gate_corner_maturity_ge3_and_best_conf_ge3"]["pct_of_placed"]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.titlesize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    # ---------------------------------------------------------------- panel A
    lengths = sorted(hist)
    counts = [hist[k] for k in lengths]
    colours = [ORANGE if k == 7 else BLUE for k in lengths]
    ax_a.bar(lengths, counts, color=colours, width=0.78)
    ax_a.set_yscale("log")
    ax_a.set_ylim(1, max(counts) * 3.2)
    ax_a.set_xlabel("chain length (nodes)")
    ax_a.set_ylabel("chains (log scale)")
    ax_a.set_xticks(lengths)
    ax_a.set_title("A  Chain length spans 4 to 16 nodes", loc="left", fontweight="bold")
    ax_a.annotate(
        f"canonical 7-node chain\n{hist[7]:,} chains ({dedup['pct_len_eq7']}%)",
        xy=(7, hist[7]),
        xytext=(9.3, hist[7] * 0.34),
        fontsize=8.5,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.9),
    )
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_a.grid(axis="y", color="#e6e6e6", lw=0.6, which="major")
    ax_a.set_axisbelow(True)

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
                fontsize=8.0,
                color="white" if v > 0.55 * vmax else "#222222",
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
    ax_b.set_title("B  What the two quality gates admit", loc="left", fontweight="bold")
    ax_b.annotate(
        f"gate corner\n{corner_n:,} of {placed:,} ({corner_pct}%)",
        xy=(4.5, 3.72),
        fontsize=8.5,
        color=ORANGE,
        fontweight="bold",
        ha="right",
        va="center",
    )
    ax_b.set_ylim(-0.5, 4.25)
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(True)
        ax_b.spines[sp].set_color("#cccccc")
    cb = fig.colorbar(im, ax=ax_b, fraction=0.045, pad=0.03)
    cb.set_label("interventions", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    fig.tight_layout(pad=0.9)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
