#!/usr/bin/env python3
"""Do intervention node NAMES absorb the prompt's lifecycle label? Class B: no LLM, no network.

WHY. Two #176 chains judged so far carry a qualifier in the intervention node's name that
does not appear in the source document, and in both the qualifier is the prompt's own
lifecycle label for the value the extractor stored:

    C01  "perform training story sensitivity analysis PRE-DEPLOYMENT"   lifecycle 4
         app:prompt lifecycle 4 = "pre-deployment testing"
    C04  "... intrinsic reward during RL FINE-TUNING ..."               lifecycle 3
         app:prompt lifecycle 3 = "fine-tuning/RL"

If that generalises, a share of the human study's non-zero inference levels are a naming
convention rather than a claim about the source -- a fixable prompt property, not a limit of
the approach. Two chains cannot establish it, so this measures it.

THE CONTRAST IS THE POINT, and a raw rate would be worthless without it. Names in this
corpus mention deployment and training for ordinary reasons, so the measurement is:

    own-label rate   = P(name contains the marker of the lifecycle the node WAS assigned)
    other-label rate = P(name contains the marker of a lifecycle it was NOT assigned)

Absorption means the first materially exceeds the second. If they match, names simply use
this vocabulary and there is nothing here.

Stage 2, on the 20 packet chains only, asks the question that makes it a defect rather than
a curiosity: when a name carries its own lifecycle marker, does that marker appear in the
SOURCE? The packet files carry the full source text, so this needs no new fetch.

    cd graph_analysis
    python -u experiment_review_lifecycle_name_absorption.py
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ATTRS = HERE / "phase2_results" / "node_attrs_slim.pkl"
PACKET = HERE / "phase2_results" / "human_review_packet"
OUT = (
    HERE / "phase2_results" / "experiment_review_lifecycle_name_absorption_report.json"
)

# Markers are the DISTINCTIVE wording of each lifecycle label in app:prompt step 2.
# "RL" and bare "training" are deliberately excluded from lifecycle 3: they are the
# ambient vocabulary of this literature and would manufacture the finding.
MARKERS = {
    1: [r"model design"],
    2: [r"pre-?train"],
    3: [r"fine-?tun"],
    4: [r"pre-?deploy"],
    # "deploy" only where it is NOT the tail of "pre-deployment": the lifecycle-4 label
    # contains the lifecycle-5 marker as a substring, so a naive pattern fires on every
    # pre-deployment name and inflates the control. Handled in hits(), not by regex.
    5: [r"deploy"],
    6: [],
}
PRE_DEPLOY = re.compile(r"pre-?deploy\w*", re.I)
LABEL = {
    1: "model design",
    2: "pre-training",
    3: "fine-tuning/RL",
    4: "pre-deployment testing",
    5: "deployment",
    6: "other",
}
RX = {k: [re.compile(p, re.I) for p in v] for k, v in MARKERS.items()}


def die(msg: str) -> None:
    raise SystemExit(
        f"FATAL: {msg}\n\n"
        f"  Needs {ATTRS} (rebuildable from the released dump) and {PACKET}.\n"
        f"  This script reads only; it judges nothing and calls no model."
    )


def hits(text: str, lc: int) -> bool:
    """Does `text` carry lifecycle `lc`'s distinctive wording?

    Lifecycle 5's marker ("deploy") is a substring of lifecycle 4's label
    ("pre-deployment testing"), so for 5 the pre-deployment occurrences are removed
    before searching. Without this every pre-deployment name counts as a lifecycle-5
    hit and the control rate is inflated.
    """
    if lc == 5:
        text = PRE_DEPLOY.sub(" ", text)
    return any(r.search(text) for r in RX.get(lc, []))


def main() -> int:
    if not ATTRS.is_file():
        die(f"missing node attributes: {ATTRS}")
    with ATTRS.open("rb") as fh:
        attrs = pickle.load(fh)

    own = Counter()
    own_hit = Counter()
    other_hit = Counter()
    other_tot = Counter()
    examples = defaultdict(list)

    for nid, a in attrs.items():
        if (a.get("type") or "").lower() != "intervention":
            continue
        lc = a.get("intervention_lifecycle")
        name = a.get("name") or ""
        if lc is None or not name:
            continue
        lc = int(lc)
        if not MARKERS.get(lc):
            continue  # lifecycle 6 has no distinctive wording to absorb
        own[lc] += 1
        if hits(name, lc):
            own_hit[lc] += 1
            if len(examples[lc]) < 4:
                examples[lc].append({"node": nid, "name": name})
        # control: the same name against every OTHER lifecycle's marker
        for k in MARKERS:
            if k == lc or not MARKERS[k]:
                continue
            other_tot[lc] += 1
            if hits(name, k):
                other_hit[lc] += 1

    corpus = {}
    for lc in sorted(own):
        o = 100 * own_hit[lc] / own[lc]
        c = 100 * other_hit[lc] / other_tot[lc] if other_tot[lc] else 0.0
        corpus[LABEL[lc]] = {
            "lifecycle": lc,
            "n_interventions": own[lc],
            "own_label_in_name_pct": round(o, 1),
            "other_label_in_name_pct": round(c, 1),
            "lift": round(o - c, 1),
            "examples": examples[lc],
        }
        print(
            f"  lifecycle {lc} ({LABEL[lc]:<22}) n={own[lc]:>6}  "
            f"own {o:5.1f}%   other {c:5.1f}%   lift {o - c:+6.1f} pp"
        )

    tot_own = sum(own.values())
    tot_hit = sum(own_hit.values())
    tot_oth = sum(other_hit.values())
    tot_ot = sum(other_tot.values())
    pooled = {
        "n_interventions": tot_own,
        "own_label_in_name_pct": round(100 * tot_hit / tot_own, 1) if tot_own else None,
        "other_label_in_name_pct": round(100 * tot_oth / tot_ot, 1) if tot_ot else None,
    }
    pooled["lift"] = round(
        pooled["own_label_in_name_pct"] - pooled["other_label_in_name_pct"], 1
    )
    print(
        f"\n  POOLED  n={tot_own}  own {pooled['own_label_in_name_pct']}%  "
        f"other {pooled['other_label_in_name_pct']}%  lift {pooled['lift']:+} pp"
    )

    # ---- stage 2: is the absorbed marker actually in the source? -------------
    print("\nStage 2 -- packet chains: is the absorbed marker in the SOURCE text?")
    manifest = json.loads((PACKET / "manifest.json").read_text(encoding="utf-8"))
    dropped = {"C08", "C10", "C11", "C14", "C17", "C19", "C21", "C22", "C25", "C26"}
    per_chain = []
    for m in manifest:
        pid = m["packet_id"]
        if pid in dropped:
            continue
        f = PACKET / "chains" / f"{pid}.md"
        if not f.is_file():
            die(f"missing packet chain file: {f}")
        text = f.read_text(encoding="utf-8")
        marker = text.find("## Full source text")
        if marker < 0:
            die(f"{f} has no '## Full source text' section; packet format changed")
        source = text[marker:]
        intervention = None
        for nid in m["nodes"]:
            a = attrs.get(nid) or attrs.get(str(nid))
            if a and (a.get("type") or "").lower() == "intervention":
                intervention = (nid, a)
        if not intervention:
            continue
        nid, a = intervention
        lc = a.get("intervention_lifecycle")
        if lc is None or not MARKERS.get(int(lc)):
            continue
        lc = int(lc)
        if not hits(a["name"], lc):
            continue
        in_source = hits(source, lc)
        # A marker cannot be "absent" in any interesting sense from a source that carries
        # almost no text. C27's source is 215 characters -- a bare Google Translate link --
        # so its absence is a property of the packet, not of the extraction's naming.
        n_src = len(source.strip())
        degenerate = n_src < 10_000
        per_chain.append(
            {
                "packet_id": pid,
                "node": nid,
                "name": a["name"],
                "lifecycle": lc,
                "lifecycle_label": LABEL[lc],
                "marker_in_source": in_source,
                "source_chars": n_src,
                "source_too_short_to_count": degenerate,
            }
        )
        flag = " [SOURCE UNDER 10k -- excluded from the headline]" if degenerate else ""
        print(
            f"  {pid}  lc={lc} ({LABEL[lc]:<22}) src={n_src:>7,}  marker in source: "
            f"{'YES' if in_source else 'NO -- name-only'}{flag}"
        )

    n_abs = len(per_chain)
    solid = [r for r in per_chain if not r["source_too_short_to_count"]]
    n_bad = sum(1 for r in solid if not r["marker_in_source"])
    print(
        f"\n  {n_abs} of the 20 packet chains carry their lifecycle label in the "
        f"intervention name."
    )
    print(
        f"  Of the {len(solid)} whose source is long enough to test, {n_bad} never state "
        f"that marker in the source."
    )

    report = {
        "study": "does the intervention node name absorb the prompt's lifecycle label",
        "prompted_by": (
            "#176 chains C01 and C04, whose unsupported name qualifiers "
            "('pre-deployment', 'RL fine-tuning') are both the stored lifecycle's label"
        ),
        "method": (
            "own-label rate against an other-label control over every intervention node "
            "in the released graph carrying a lifecycle with distinctive wording; then, on "
            "the 20 in-scope packet chains, whether the absorbed marker appears in the source"
        ),
        "markers_excluded_on_purpose": (
            "'RL' and bare 'training' for lifecycle 3 -- ambient vocabulary in this "
            "literature, and including them would manufacture the result"
        ),
        "corpus": corpus,
        "pooled": pooled,
        "packet_stage_2": {
            "n_in_scope": 20,
            "n_with_absorbed_label": n_abs,
            "n_testable_source_at_least_10k_chars": len(solid),
            "n_whose_marker_is_absent_from_the_source": n_bad,
            "excluded_short_sources": [
                {"packet_id": r["packet_id"], "source_chars": r["source_chars"]}
                for r in per_chain
                if r["source_too_short_to_count"]
            ],
            "per_chain": per_chain,
        },
        "LIMITS": (
            "Lexical, not semantic: a name can carry the idea of its lifecycle without the "
            "marker string, and a marker in a name can be the document's own word. Stage 2 "
            "checks the source for the MARKER, not for the claim. It bounds how often the "
            "qualifier could have come from the rubric; it does not prove that it did."
        ),
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
