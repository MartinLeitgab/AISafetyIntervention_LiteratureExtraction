#!/usr/bin/env python
"""Figure 1 of the paper: the two-stage pipeline and one extracted chain as stored.

Panel A  the pipeline: one schema-prompted o3 call per document, then a judge from a
         different provider re-reading the source against the extraction (the audit).
Panel B  one real chain from the released reporting unit -- the retrieval example of
         sec:r-retrieval (Shavit 2023) -- in logical-chain order, with each node's stored
         name and each edge's stored relation type and confidence.

REVISED 2026-11-06 (v5). The previous version printed at 3.6-4.0 pt and drew panel B as
seven placeholder boxes ("a named AI safety risk"). The figure is now drawn at its printed
size (figure* at \\textwidth = 7.0 in), so a font size below is the size on the page and
nothing is under 7 pt; panel B shows a stored chain, which also replaced the body's
retrieval table. Stage 2 is labeled "Audit", the paper's term; the meta-graders are gone
from the figure because the paper no longer reports them.

Two steps, so the render never needs the 3.2 GB checkpoints:
    python -u experiment_figure4_pipeline_schema.py --rebuild-receipt   # reads the PKLs
    python -u experiment_figure4_pipeline_schema.py                     # plots from receipt

Palette matches the hub figure (risk / five-step body ramp in chain order /
intervention). Class B: no LLM calls, no network.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
NODES = HERE / "phase2_results" / "node_attrs_slim.pkl"
EDGES = (
    HERE
    / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_edge_data.pkl"
)
RECEIPT = HERE / "phase2_results" / "experiment_figure4_chain_receipt.json"
OUT_PNG = HERE / "plots" / "figure4_pipeline_schema.png"
OUT_PDF = HERE / "plots" / "figure4_pipeline_schema.pdf"

# The retrieval example of sec:r-retrieval, arxiv.org/abs/2303.11341, in chain order.
CHAIN_IDS = [1067, 1086, 1087, 1088, 1089, 1090, 1091]
SOURCE_LABEL = "Shavit, 2023"

C_RISK = "#eb6834"
C_INTV = "#1baf7a"
BODY_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK = "#111111"
MUTED = "#444444"
BLUE = "#0072B2"
GREY = "#999999"

CHIPS = [
    ("risk", C_RISK, "white"),
    ("pa", BODY_RAMP[0], INK),
    ("ti", BODY_RAMP[1], "white"),
    ("dr", BODY_RAMP[2], "white"),
    ("im", BODY_RAMP[3], "white"),
    ("va", BODY_RAMP[4], "white"),
    ("intv", C_INTV, "white"),
]
LIFECYCLE = {
    1: "model design",
    2: "pre-training",
    3: "fine-tuning or RL",
    4: "pre-deployment testing",
    5: "deployment",
    6: "other",
}


def rebuild_receipt() -> None:
    for p, how in (
        (NODES, "experiment_review_prep_slim_nodes.py"),
        (EDGES, "phase2_step1_loadandparse.py"),
    ):
        if not p.exists():
            raise SystemExit(f"FATAL: missing {p}\n  produced by: {how}")
    na = pickle.load(open(NODES, "rb"))
    edges = pickle.load(open(EDGES, "rb"))
    pairs = {frozenset(p) for p in zip(CHAIN_IDS, CHAIN_IDS[1:])}
    found = {}
    for e in edges:
        if e["type"] != "EDGE":
            continue
        key = frozenset((e["source"], e["target"]))
        if key in pairs:
            found[key] = {
                "source": e["source"],
                "target": e["target"],
                "relation": e["subtype"],
                "confidence": e["confidence"],
            }
    hops = []
    for a, b in zip(CHAIN_IDS, CHAIN_IDS[1:]):
        k = frozenset((a, b))
        if k not in found:
            raise SystemExit(f"FATAL: no structural edge between {a} and {b}")
        hops.append(found[k])
    nodes = [
        {
            "id": i,
            "name": na[i]["name"],
            "concept_category": na[i].get("concept_category"),
            "type": na[i].get("type"),
            "intervention_maturity": na[i].get("intervention_maturity"),
            "intervention_lifecycle": na[i].get("intervention_lifecycle"),
            "url": na[i].get("url"),
        }
        for i in CHAIN_IDS
    ]
    if len({n["url"] for n in nodes}) != 1:
        raise SystemExit("FATAL: the chain spans more than one source document")
    RECEIPT.write_text(
        json.dumps({"chain": nodes, "hops": hops}, indent=1), encoding="utf-8"
    )
    print(f"wrote {RECEIPT}")


def rbox(ax, x, y, w, h, edge, face="white", lw=1.2, z=3):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.015",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            zorder=z,
        )
    )


def arrow(ax, x0, y0, x1, y1, color=BLUE):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=color,
            zorder=2,
        )
    )


def plot() -> None:
    if not RECEIPT.exists():
        raise SystemExit(
            f"FATAL: missing {RECEIPT}\n"
            "  produced by: python -u experiment_figure4_pipeline_schema.py --rebuild-receipt"
        )
    rec = json.loads(RECEIPT.read_text(encoding="utf-8"))
    chain, hops = rec["chain"], rec["hops"]

    plt.rcParams.update(
        {
            "font.size": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(7.0, 2.75))
    axA = fig.add_axes([0.0, 0.0, 0.33, 1.0])
    axB = fig.add_axes([0.345, 0.0, 0.655, 1.0])
    for ax in (axA, axB):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ------------------------------------------------------------------ panel A
    axA.text(0.02, 0.955, "A  The two stages", fontsize=8.5, fontweight="bold")

    def stage(y, h, edge, title, body, tag=None):
        rbox(axA, 0.04, y, 0.92, h, edge)
        axA.text(0.075, y + h - 0.035, title, fontsize=8, fontweight="bold", va="top")
        axA.text(
            0.075,
            y + h - 0.095,
            body,
            fontsize=7,
            color=MUTED,
            va="top",
            linespacing=1.25,
        )
        if tag:
            axA.text(
                0.93,
                y + h - 0.045,
                tag,
                fontsize=7.2,
                color=edge,
                ha="right",
                va="top",
                style="italic",
            )

    stage(0.76, 0.15, GREY, "Document", "one ARD record, full text")
    stage(
        0.42,
        0.255,
        BLUE,
        "Stage 1  Extraction",
        "o3, one call per document\nno retrieval: links are not followed\nemits the typed graph of panel B",
        "extractor",
    )
    stage(
        0.06,
        0.255,
        C_INTV,
        "Stage 2  Audit",
        "claude-sonnet-4-5, another provider\nreads the source and the extraction\nreturns findings and proposed repairs",
        "judge",
    )
    arrow(axA, 0.5, 0.76, 0.5, 0.68)
    arrow(axA, 0.5, 0.42, 0.5, 0.32)
    axA.text(0.53, 0.72, "text + prompt", fontsize=7, va="center")
    axA.text(0.53, 0.37, "text + extracted graph", fontsize=7, va="center")

    # ------------------------------------------------------------------ panel B
    axB.text(
        0.0,
        0.955,
        f"B  One extracted chain, as stored ({SOURCE_LABEL})",
        fontsize=8.5,
        fontweight="bold",
    )
    top, pitch, chip_h, chip_w = 0.845, 0.112, 0.068, 0.075
    for i, (node, (abbr, col, txt)) in enumerate(zip(chain, CHIPS)):
        y = top - i * pitch
        rbox(axB, 0.0, y - chip_h / 2, chip_w, chip_h, col, face=col, lw=0.6)
        axB.text(
            chip_w / 2,
            y,
            abbr,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=txt,
            zorder=4,
        )
        axB.text(chip_w + 0.018, y, node["name"], va="center", fontsize=7.0, color=INK)
        if i < len(hops):
            h = hops[i]
            axB.text(
                chip_w / 2 + 0.012,
                y - pitch / 2,
                f"{h['relation']}, confidence {h['confidence']}",
                va="center",
                ha="left",
                fontsize=7,
                color=MUTED,
                style="italic",
            )
            axB.plot(
                [chip_w / 2, chip_w / 2],
                [y - chip_h / 2, y - pitch + chip_h / 2],
                color=GREY,
                lw=0.8,
                zorder=1,
            )
    intv = chain[-1]
    axB.text(
        0.0,
        0.015,
        f"Every node stores its source URL. The intervention also stores\n"
        f"maturity {intv['intervention_maturity']} and lifecycle "
        f"{intv['intervention_lifecycle']} ({LIFECYCLE[intv['intervention_lifecycle']]}).",
        fontsize=7,
        color=MUTED,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PNG}\nwrote {OUT_PDF}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild-receipt", action="store_true")
    args = ap.parse_args()
    if args.rebuild_receipt:
        rebuild_receipt()
    plot()
    return 0


if __name__ == "__main__":
    sys.exit(main())
