#!/usr/bin/env python3
"""Wait for the two chain-precision batches, with a 30-second heartbeat.

Complies with the logging rule: a progress line lands at least every 30 s whether or not
anything changed, so silence always means the watcher died and never means the work is
merely slow. The remote is queried on a slower cadence (POLL_EVERY) because the batch API
gives no queue-position signal and hammering it buys nothing; the heartbeat repeats the last
known counts in between, with elapsed time so the wait is legible.

Collects each arm the moment it ends, so a slow arm never blocks a finished one.

    cd graph_analysis
    python -u watch_chain_precision_batches.py            # until both end
    python -u watch_chain_precision_batches.py --hours 6  # give up after 6 h
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "experiment_review_chain_precision.py"
RAW = HERE / "phase2_results" / "chain_precision_raw"

HEARTBEAT = 30.0  # seconds; the rule's ceiling
POLL_EVERY = 300.0  # seconds between actual API queries

ARMS = {
    "A": {"id_file": RAW / "batch_id.txt", "flag": "--collect", "n": 210},
    "B": {
        "id_file": RAW / "batch_id_contrast.txt",
        "flag": "--collect-contrast",
        "n": 100,
    },
}


def load_module():
    spec = importlib.util.spec_from_file_location("chain_precision", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()

    m = load_module()
    import anthropic

    client = anthropic.Anthropic(api_key=m.read_key())

    state = {}
    for k, a in ARMS.items():
        if not a["id_file"].is_file():
            log(f"arm {k}: no batch id at {a['id_file']}, skipping")
            continue
        state[k] = {
            "id": a["id_file"].read_text().strip(),
            "status": "unknown",
            "counts": "unknown",
            "done": False,
        }
        log(f"arm {k}: watching {state[k]['id']} ({a['n']} requests)")
    if not state:
        log("nothing to watch")
        return 1

    t0 = time.monotonic()
    deadline = t0 + args.hours * 3600
    next_poll = 0.0

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_poll:
            next_poll = now + POLL_EVERY
            for k, st in state.items():
                if st["done"]:
                    continue
                try:
                    b = client.messages.batches.retrieve(st["id"])
                    st["status"] = b.processing_status
                    c = b.request_counts
                    st["counts"] = (
                        f"ok={c.succeeded} run={c.processing} err={c.errored} "
                        f"exp={c.expired} cancel={c.canceled}"
                    )
                    if b.processing_status == "ended":
                        log(f"arm {k}: ENDED, collecting ...")
                        r = subprocess.run(
                            [sys.executable, "-u", str(SCRIPT), ARMS[k]["flag"]],
                            capture_output=True,
                            text=True,
                            cwd=str(HERE),
                        )
                        print(r.stdout, flush=True)
                        if r.returncode != 0:
                            log(f"arm {k}: COLLECT FAILED rc={r.returncode}")
                            print(r.stderr, flush=True)
                            log(
                                f"arm {k}: results stay retrievable -- fix and re-run "
                                f"{SCRIPT.name} {ARMS[k]['flag']}; no tokens are re-spent"
                            )
                        st["done"] = True
                except Exception as exc:  # noqa: BLE001 - logged, loop continues
                    log(
                        f"arm {k}: query failed ({type(exc).__name__}: {exc}); retrying"
                    )

        if all(st["done"] for st in state.values()):
            log(f"all arms collected after {(time.monotonic() - t0) / 60:.1f} min")
            return 0

        mins = (time.monotonic() - t0) / 60
        parts = [
            f"{k}={'collected' if st['done'] else st['status']}"
            f"{'' if st['done'] else ' [' + st['counts'] + ']'}"
            for k, st in state.items()
        ]
        log(f"waiting {mins:6.1f} min | " + " | ".join(parts))
        time.sleep(HEARTBEAT)

    log(f"gave up after {args.hours} h; batches remain retrievable by id")
    return 1


if __name__ == "__main__":
    sys.exit(main())
