#!/usr/bin/env python3
"""Heartbeat a running Anthropic batch until it ends.

Prints a line every 30 seconds whether or not anything changed, because silence for longer
than that is indistinguishable from a hang. The remote is polled on a slower cadence than
the heartbeat -- there is no point asking the API every 30s about a job that takes an hour.

    cd graph_analysis
    python -u watch_batch.py <batch_id>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HEARTBEAT_S = 30
POLL_S = 120

KEY_ENV = Path.home() / "0_project_work" / "ExistentialRiskBenchmark" / ".env"
KEY_VAR = "ANTHROPIC_API_KEY"


def read_key() -> str:
    for line in KEY_ENV.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(f"{KEY_VAR}="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit(f"FATAL: {KEY_VAR} not in {KEY_ENV}")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -u watch_batch.py <batch_id>")
    batch_id = sys.argv[1]

    import anthropic

    client = anthropic.Anthropic(api_key=read_key())
    t0 = time.monotonic()
    last_poll = -POLL_S
    status, counts = "unknown", {}

    while True:
        now = time.monotonic()
        if now - last_poll >= POLL_S:
            b = client.messages.batches.retrieve(batch_id)
            status = b.processing_status
            rc = b.request_counts
            counts = {
                "succeeded": rc.succeeded,
                "processing": rc.processing,
                "errored": rc.errored,
                "canceled": rc.canceled,
                "expired": rc.expired,
            }
            last_poll = now
        el = int(now - t0)
        print(
            f"[{el // 60:02d}m{el % 60:02d}s] batch {batch_id[:22]} status={status} "
            + " ".join(f"{k}={v}" for k, v in counts.items()),
            flush=True,
        )
        if status == "ended":
            print(f"ENDED after {el // 60}m{el % 60}s", flush=True)
            return 0
        time.sleep(HEARTBEAT_S)


if __name__ == "__main__":
    sys.exit(main())
