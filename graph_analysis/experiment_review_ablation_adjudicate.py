#!/usr/bin/env python3
"""Decide, semantically, whether the recall judge surfaced the intervention we deleted.

The ablation arm of experiment_review_chain_recall.py deletes one INTERVENTION from the
pair list a judge is shown, then asks whether the judge surfaces it as uncaptured. That is
the right test. Scoring it was not: the first two attempts matched the deleted name against
the judge's uncaptured names with token overlap, and the answer is a pure function of the
threshold --

    jaccard >= 0.40 -> 0/20        jaccard >= 0.15 -> 8/20
    jaccard >= 0.25 -> 0/20        jaccard >= 0.10 -> 10/20

-- because both strings are long generated names of the same thing in different words.
"Design AI models with cryptographic shutdown backdoors activated by secret trigger" and
"Insert cryptographic backdoors (off-switches) that are hard to detect" score 0.12 and are
obviously the same intervention. This project has now learned that lesson three times: the
46.5% node re-identification figure, #179's five-point rubric, and here. String similarity
over generated names is not an instrument.

So ask a model the question directly, once per ablated document: is the deleted intervention
among the ones the recall judge flagged as uncaptured? One short call, no source text, no
enumeration -- just a name against a list of names. This is a much easier task than the one
it adjudicates, which is why it is allowed to be the referee.

    accept   the judge surfaced the deleted intervention (however worded) -> DETECTED
    reject   nothing in the list is that intervention                     -> MISSED

🔴 What this licenses and what it does not. It measures the recall judge's SENSITIVITY, and
that number governs whether the 127-document recall run's headline may be published at all.
It does not make the headline correct; it only says how much of what is missing the
instrument can see. A second model adjudicating a first model is still no human.

Class A: metered, but trivially so -- 20 short calls, well under USD 0.10.

    cd graph_analysis
    python -u experiment_review_ablation_adjudicate.py --run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ABL_DEFAULT = HERE / "phase2_results" / "chain_recall_ablation_raw"
OUT = HERE / "phase2_results" / "experiment_review_ablation_adjudicate_report.json"
KEY_ENV = Path.home() / "0_project_work" / "ExistentialRiskBenchmark" / ".env"
KEY_VAR = "ANTHROPIC_API_KEY"
MODEL = "claude-sonnet-4-5-20250929"

PROMPT = """An automated auditor was shown a document and a list of risk-to-intervention \
pairs extracted from it, and asked which arguments the extraction had missed. Before we \
showed it that list, we secretly DELETED one intervention from it.

The deleted intervention is:
    {deleted}

The auditor flagged these interventions as missing from the extraction:
{candidates}

Question: is the deleted intervention among the ones the auditor flagged?

Judge MEANING, not wording. The two descriptions were written independently and will not \
match textually. "Design AI models with cryptographic shutdown backdoors activated by a \
secret trigger" and "Insert cryptographic backdoors (off-switches) that are hard to detect" \
are the SAME intervention. A more general or more specific description of the same \
technique still counts as the same intervention. A different technique addressing the same \
problem does NOT.

Return ONLY JSON:
{{"found": true|false, "which": <1-based index or null>, "why": "one sentence"}}
"""


def die(msg):
    raise SystemExit(f"FATAL: {msg}")


def read_key():
    for line in KEY_ENV.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(f"{KEY_VAR}="):
            return line.split("=", 1)[1].strip().strip("\"'")
    die(f"{KEY_VAR} not in {KEY_ENV}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", required=True)
    ap.add_argument(
        "--dir",
        default=str(ABL_DEFAULT),
        help="raw directory of the ablation run to adjudicate",
    )
    a = ap.parse_args()
    ABL = Path(a.dir)
    out_path = (
        OUT
        if ABL == ABL_DEFAULT
        else OUT.with_name(
            OUT.stem
            + "_"
            + ABL.name.replace("chain_recall_", "").replace("_raw", "")
            + ".json"
        )
    )

    sfp, rfp = ABL / "sample.json", ABL / "results.jsonl"
    for p in (sfp, rfp):
        if not p.is_file():
            die(
                f"missing {p}\n"
                "  produce it with:\n"
                "    python -u experiment_review_chain_recall.py --submit --ablation-only\n"
                "    python -u experiment_review_chain_recall.py --collect <batch_id>"
            )
    sample = {s["custom_id"]: s for s in json.loads(sfp.read_text(encoding="utf-8"))}
    results = [
        json.loads(x) for x in rfp.read_text(encoding="utf-8").splitlines() if x.strip()
    ]

    import anthropic

    client = anthropic.Anthropic(api_key=read_key())
    rows, detected, no_candidates = [], 0, 0
    for r in results:
        s = sample.get(r["custom_id"])
        if not s or not s.get("ablated_intervention"):
            continue
        unc = [
            a.get("intervention")
            for a in (r.get("verdict") or {}).get("arguments", [])
            if (a.get("status") or "").startswith("uncaptured")
            and a.get("intervention")
        ]
        if not unc:
            # The judge flagged nothing at all. Unambiguously a miss; no call needed, and
            # spending one would invite the referee to hallucinate a match against an empty
            # list.
            no_candidates += 1
            rows.append(
                {
                    "custom_id": r["custom_id"],
                    "found": False,
                    "reason": "no uncaptured items",
                }
            )
            continue
        msg = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(
                        deleted=s["ablated_intervention"],
                        candidates="\n".join(
                            f"  {i + 1}. {c}" for i, c in enumerate(unc)
                        ),
                    ),
                }
            ],
        )
        t = msg.content[0].text
        try:
            v = json.loads(t[t.index("{") : t.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            rows.append({"custom_id": r["custom_id"], "found": None, "raw": t[:200]})
            continue
        detected += bool(v.get("found"))
        rows.append(
            {
                "custom_id": r["custom_id"],
                "deleted": s["ablated_intervention"],
                "found": bool(v.get("found")),
                "which": v.get("which"),
                "why": (v.get("why") or "")[:200],
                "n_candidates": len(unc),
            }
        )

    n = sum(1 for x in rows if x.get("found") is not None)
    sens = round(detected / n, 3) if n else None
    report = {
        "study": "semantic adjudication of the chain-recall ablation arm",
        "why": (
            "Token-overlap scoring of this arm returned anything from 0/20 to 10/20 "
            "depending on the threshold, because both names are long generated "
            "descriptions of the same thing. The rate was a property of the threshold, not "
            "of the judge."
        ),
        "model": MODEL,
        "n_adjudicated": n,
        "detected": detected,
        "sensitivity": sens,
        "documents_where_the_judge_flagged_nothing": no_candidates,
        "detail": rows,
        "WHAT_THIS_GOVERNS": (
            "The 127-document recall run reports a 10.6% material miss rate on the audited "
            "100. That number is publishable only in proportion to this sensitivity: at s, "
            "the corrected rate is roughly 10.6/s, and if s is low the honest report is the "
            "sensitivity alone. Quote the two together or neither."
        ),
        "LIMITS": (
            "A second model refereeing a first, both from the same family. The referee's "
            "task is far easier than the one it judges -- one name against a short list, no "
            "source text, no enumeration -- which is the only reason it is allowed to "
            "referee at all. Still not a human."
        ),
    }
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"adjudicated {n}, detected {detected}, sensitivity {sens}")
    print(f"  judge flagged nothing at all in {no_candidates} documents")
    # Print what was actually written. This said {OUT} while writing to {out_path}, which
    # made a gated adjudication look as though it had clobbered the ungated one. It had not.
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
