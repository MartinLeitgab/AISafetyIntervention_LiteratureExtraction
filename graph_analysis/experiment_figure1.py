#!/usr/bin/env python
"""Figure 1 for the paper: what the extraction produced, in three panels.

Panel A  chain-length distribution of the 2,772 de-duplicated chains, with the
         canonical seven-node chain marked -- the evidence that the extraction
         follows each paper's argument instead of filling a fixed template.
Panel B  LLM-assessed intervention maturity over all 36,959 extracted
         interventions -- the field-state descriptive no metadata directory
         can produce.
Panel C  corpus yield -- the honest counterweight to A and B: only one document
         in six yields a complete high-confidence chain at all.

Plots straight from the committed receipt JSONs, so the figure is reproducible
without the two large PKL checkpoints.

NOTE 2026-08-11: an earlier version of panel C plotted a scope composition
(capability-gap vs human-harm share) derived from the Paper-B routing pass. It
was withdrawn because the routing assignment file is not committed and cannot be
re-derived by a reader at reasonable cost. Panel C now uses only quantities that
the tracked path files and the claim-audit receipt support.

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
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
RES = HERE / "phase2_results"
AUDIT = RES / "experiment_paper_claim_audit.json"
STRENGTH = RES / "experiment_dataset_strength_report.json"
OUT_PNG = HERE / "plots" / "figure1_dataset.png"
OUT_PDF = HERE / "plots" / "figure1_dataset.pdf"

# Okabe-Ito: safe under every common form of colour vision deficiency.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREY = "#999999"
GREEN = "#009E73"
VERM = "#D55E00"


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
    strength = need(
        STRENGTH, "graph_analysis/experiment_dataset_strength_descriptives.py"
    )

    dedup = audit["path_set_comparison"]["deduped"]
    hist = {int(k): v for k, v in dedup["len_hist"].items()}
    n_chains = dedup["n_paths"]

    mat = strength["interventions"]["maturity"]
    n_intv = strength["interventions"]["n"]

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

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.0, 3.3))

    # ---------------------------------------------------------------- panel A
    lengths = sorted(hist)
    counts = [hist[k] for k in lengths]
    colours = [ORANGE if k == 7 else BLUE for k in lengths]
    ax_a.bar(lengths, counts, color=colours, width=0.78)
    ax_a.set_xlabel("chain length (nodes)")
    ax_a.set_ylabel("chains")
    ax_a.set_xticks(lengths)
    ax_a.set_title(
        "A  Chains are not a filled-in template", loc="left", fontweight="bold"
    )
    ax_a.annotate(
        f"canonical 7-node chain\n{hist[7]:,} chains ({dedup['pct_len_eq7']}%)",
        xy=(7, hist[7]),
        xytext=(9.6, hist[7] * 0.82),
        fontsize=8.5,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9),
    )
    ax_a.annotate(
        f"{dedup['pct_len_gt7']}% longer\n(extra nodes per stage)",
        xy=(10.5, hist[7] * 0.20),
        fontsize=8.5,
        color="#333333",
    )
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))

    # ---------------------------------------------------------------- panel B
    order = ["1", "2", "3", "4"]
    labels = ["conceptual", "early-stage", "mature", "deployed"]
    pcts = [mat[k]["pct"] for k in order]
    ns = [mat[k]["n"] for k in order]
    bar_colours = [GREY, BLUE, GREEN, VERM]
    left = 0.0
    for pct, n, lab, col in zip(pcts, ns, labels, bar_colours):
        ax_b.barh([0], [pct], left=left, color=col, height=0.5)
        if pct > 6:
            ax_b.text(
                left + pct / 2,
                0,
                f"{lab}\n{pct}%",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if col != GREY else "#111111",
            )
        left += pct
    ax_b.annotate(
        f"deployed: {mat['4']['pct']}% ({mat['4']['n']:,})",
        xy=(99.0, 0.28),
        xytext=(70, 0.62),
        fontsize=8.5,
        color=VERM,
        arrowprops=dict(arrowstyle="->", color=VERM, lw=0.9),
    )
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(-0.55, 0.85)
    ax_b.set_yticks([])
    ax_b.set_xlabel(f"share of {n_intv:,} extracted interventions (%)")
    ax_b.set_title(
        "B  Almost nothing proposed is deployed", loc="left", fontweight="bold"
    )
    ax_b.spines["left"].set_visible(False)

    # ---------------------------------------------------------------- panel C
    # Corpus yield funnel. Every quantity here comes from the tracked path files
    # or the claim-audit receipt.
    n_docs = 11779
    n_yield = dedup["n_source_papers"]
    stages = [
        (f"documents in corpus\n{n_docs:,}", n_docs, GREY),
        (
            f"yield a complete chain\n{n_yield:,} ({100.0 * n_yield / n_docs:.1f}%)",
            n_yield,
            BLUE,
        ),
    ]
    ypos = [1, 0]
    for (lab, val, col), y in zip(stages, ypos):
        ax_c.barh([y], [val], color=col, height=0.55)
        ax_c.text(
            val + n_docs * 0.02,
            y,
            lab,
            ha="left",
            va="center",
            fontsize=8.5,
            color="#111111",
        )
    ax_c.set_xlim(0, n_docs * 1.62)
    ax_c.set_ylim(-0.8, 1.6)
    ax_c.set_yticks([])
    ax_c.set_xlabel("documents")
    ax_c.set_title(
        "C  One document in six yields a chain", loc="left", fontweight="bold"
    )
    ax_c.annotate(
        f"those {n_yield:,} documents contribute\nthe {n_chains:,} chains in panel A",
        xy=(n_docs * 0.02, -0.55),
        fontsize=8.5,
        color="#333333",
    )
    ax_c.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_c.spines["left"].set_visible(False)

    fig.tight_layout(pad=0.9)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
