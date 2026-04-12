#!/usr/bin/env python3
"""Print current Curiosity capability status.

Reads ~/curiosity/metrics/capability_scores.jsonl and optionally queries
Redis for queue depths, then prints a human-readable summary table.
"""

import json
import os
import sys
from pathlib import Path

METRICS_FILE = Path.home() / "curiosity" / "metrics" / "capability_scores.jsonl"

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

QUEUES = [
    "ASSESS_QUEUE",
    "FORMULATE_QUEUE",
    "SOLVE_QUEUE",
    "VERIFY_QUEUE",
    "MEMORIZE_QUEUE",
]

DIRECTION_SYMBOL = {"↑": "↑", "↓": "↓", "→": "→"}


def load_summary() -> dict[str, dict]:
    """Load latest capability record per domain from the metrics file."""
    if not METRICS_FILE.exists():
        return {}

    latest: dict[str, dict] = {}
    with METRICS_FILE.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                latest[rec["domain"]] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return latest


def direction(delta: float) -> str:
    if delta > 0.01:
        return "↑"
    elif delta < -0.01:
        return "↓"
    return "→"


def get_queue_depths() -> dict[str, int | str]:
    """Query Redis for queue depths. Returns dict of queue → depth (or error str)."""
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
                        socket_connect_timeout=2)
        depths = {}
        for q in QUEUES:
            try:
                info = r.xinfo_stream(q)
                depths[q] = info.get("length", 0)
            except Exception:
                depths[q] = "n/a"
        return depths
    except Exception as exc:
        return {q: f"err: {exc}" for q in QUEUES}


def main() -> None:
    summary = load_summary()

    print("════════════════════════════════════════════════════════════════")
    print("  Curiosity Capability Status")
    print("════════════════════════════════════════════════════════════════")

    if not summary:
        print("  (no capability records found — system may not have run yet)")
        print(f"  Expected metrics file: {METRICS_FILE}")
    else:
        header = f"  {'Domain':<16} {'Current':>8} {'Baseline':>9} {'Delta':>8}  Dir  Cycle"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for domain, rec in sorted(summary.items()):
            pass_rate = rec.get("pass_rate", 0.0)
            baseline  = rec.get("baseline_pass_rate", pass_rate)
            delta     = rec.get("delta", 0.0)
            cycle     = rec.get("cycle", "?")
            dir_sym   = direction(delta)
            print(
                f"  {domain:<16} {pass_rate:>7.1%}  {baseline:>7.1%}  "
                f"{delta:>+7.1%}   {dir_sym}   {cycle}"
            )

    print()
    print("  Redis Queue Depths:")
    print("  " + "─" * 40)
    depths = get_queue_depths()
    for q, depth in depths.items():
        print(f"  {q:<24} {depth}")

    print("════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
