#!/usr/bin/env python3
"""Prove the code that runs AFTER the money is spent, before it is spent.

`experiment_review_chain_precision.py` submits two metered batches and then parses and
tallies what comes back. The submit path was validated by a live 3-call dry run. The
COLLECT path was not, and that is the half that runs once the spend is already committed --
a bug there does not waste tokens (Anthropic keeps batch results retrievable, so a fixed
script can re-fetch the same results for free) but it does waste a day and it looks exactly
like a burn.

So: fixtures are the real dry-run responses on disk, plus three synthetic failure cases the
live run did not produce. No network, no API key, runs in under a second.

    python -u tests/test_chain_precision_collect.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "graph_analysis" / "experiment_review_chain_precision.py"
DRYRUN_DIR = ROOT / "graph_analysis" / "phase2_results" / "chain_precision_raw"

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def load_module():
    spec = importlib.util.spec_from_file_location("chain_precision", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_receipt_survives(m, rows: list[dict]) -> bool:
    """Call write_receipt for real, against a throwaway path, and see if it raises.

    Redirects the module's RECEIPT constant so the committed receipt is never touched, and
    swallows stdout because write_receipt prints its own summary.
    """
    import contextlib
    import io
    import tempfile

    original = m.RECEIPT
    try:
        with tempfile.TemporaryDirectory() as d:
            m.RECEIPT = Path(d) / "throwaway_receipt.json"
            with contextlib.redirect_stdout(io.StringIO()):
                m.write_receipt(rows, "msgbatch_test_only")
            return m.RECEIPT.is_file()
    except Exception as exc:  # noqa: BLE001 - the point of the test is to report, not raise
        print(f"        raised {type(exc).__name__}: {exc}")
        return False
    finally:
        m.RECEIPT = original


def main() -> int:
    m = load_module()

    print("\n--- parse_verdict against the real dry-run responses ---")
    real_bodies = []
    for fp in sorted(DRYRUN_DIR.glob("dryrun_*.json")):
        body = json.loads(fp.read_text(encoding="utf-8"))["response"]
        real_bodies.append((fp.name, body))
    check(
        "dry-run fixtures present", len(real_bodies) == 3, f"{len(real_bodies)} found"
    )
    for name, body in real_bodies:
        v = m.parse_verdict(body)
        check(f"{name} parses", v is not None)
        if v is None:
            continue
        # The model wraps the object in a ```json fence and adds prose after it. Both must
        # survive, and every field the tally reads must be present.
        for field in (
            "risk_framing",
            "intervention",
            "intermediate_stages",
            "chain_is_a_fair_summary_of_an_argument_the_document_makes",
            "reason_code",
            "confidence",
        ):
            check(f"{name} carries {field}", field in v)
        check(
            f"{name} fair-summary field is a bool",
            isinstance(
                v["chain_is_a_fair_summary_of_an_argument_the_document_makes"], bool
            ),
        )

    print("\n--- parse_verdict on cases the live run did not produce ---")
    check("empty response -> None", m.parse_verdict("") is None)
    check("prose with no object -> None", m.parse_verdict("I cannot do that.") is None)
    check(
        "truncated object -> None, not an exception",
        m.parse_verdict('```json\n{"risk_framing": {"verdict": "supp') is None,
    )
    check(
        "fenced object with trailing prose -> parses",
        (m.parse_verdict('```json\n{"a": 1}\n```\nAnd here is why.') or {}).get("a")
        == 1,
    )

    print("\n--- tally arithmetic on a hand-built set ---")

    def row(arm: str, fair: bool, code: str, risk: str = "supported", quote: str = "q"):
        return {
            "arm": arm,
            "verdict": {
                "risk_framing": {"verdict": risk, "quote": quote},
                "intervention": {"verdict": "unsupported", "quote": ""},
                "intermediate_stages": {"verdict": "partial", "note": ""},
                "chain_is_a_fair_summary_of_an_argument_the_document_makes": fair,
                "reason_code": code,
                "confidence": 4,
            },
        }

    rows = [
        row("real", True, "faithful"),
        row("real", False, "intervention_not_proposed"),
        row("real", False, "risk_framing_invented", risk="unsupported", quote=""),
        {"arm": "real", "parse_error": True},  # must be excluded from the denominator
        row("null_mismatched_pair", False, "chain_belongs_to_a_different_document"),
        row("gate_rejected", False, "intervention_not_proposed"),
        row("gate_rejected", True, "faithful"),
    ]
    t = m.tally(rows, "real")
    check("parse errors excluded from n", t["n_parsed"] == 3, f"n={t['n_parsed']}")
    check("fair counted", t["judged_fair_summary"] == 1)
    check("not-fair counted", t["judged_not_fair"] == 2)
    check(
        "not-fair pct", t["judged_not_fair_pct"] == 66.7, str(t["judged_not_fair_pct"])
    )
    check("unsupported risk framing counted", t["risk_framing_unsupported"] == 1)
    check(
        "supported-with-quote counted",
        t["supported_risk_verdicts_carrying_a_quote"] == 2,
    )
    check(
        "reason codes tallied",
        t["reason_codes"].get("intervention_not_proposed") == 1
        and t["reason_codes"].get("faithful") == 1,
    )
    empty = m.tally(rows, "no_such_arm")
    check(
        "empty arm returns None rates rather than dividing by zero",
        empty["n_parsed"] == 0 and empty["judged_not_fair_pct"] is None,
    )
    tn = m.tally(rows, "null_mismatched_pair")
    check("null arm flagged 100%", tn["judged_not_fair_pct"] == 100.0)
    tr = m.tally(rows, "gate_rejected")
    check(
        "gate-rejected arm tallies", tr["n_parsed"] == 2 and tr["judged_not_fair"] == 1
    )

    print("\n--- one arm outstanding: the case that actually crashed ---")
    # 2026-08-17: arm B ended five minutes before arm A, so the first collect ran with a
    # non-empty gate-rejected arm and an EMPTY real arm. gate_delta was None, the summary
    # print subscripted it, and collect exited 1. The tally test below passed at the time
    # and did not catch it, because the bug was in the print path, not the arithmetic.
    b_only = [r for r in rows if r.get("arm") == "gate_rejected"]
    ta, tb = m.tally(b_only, "real"), m.tally(b_only, "gate_rejected")
    check(
        "arm A empty while arm B has data", ta["n_parsed"] == 0 and tb["n_parsed"] == 2
    )
    check(
        "no difference is computable from one arm",
        ta["judged_not_fair_pct"] is None,
        "so any consumer of it must guard, print paths included",
    )
    check(
        "write_receipt survives a one-arm receipt",
        _write_receipt_survives(m, b_only),
        "regression for the crash above",
    )

    print("\n--- the difference the study actually reports ---")
    delta = round(tr["judged_not_fair_pct"] - t["judged_not_fair_pct"], 1)
    check(
        "arm difference computes in percentage points",
        delta == round(50.0 - 66.7, 1),
        f"{delta} pp",
    )
    check(
        "a negative difference is representable, not clamped",
        delta < 0,
        "arm B cleaner than arm A here, which the receipt must be able to say",
    )

    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'ALL CHECKS PASS'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
