"""
Capability tracker — measures and records improvement over time.
Writes to ~/curiosity/metrics/capability_scores.jsonl (append-only).
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

METRICS_FILE = Path.home() / "curiosity" / "metrics" / "capability_scores.jsonl"


@dataclass
class CapabilitySnapshot:
    timestamp: float
    domain: str
    pass_rate: float
    sample_size: int
    baseline_pass_rate: float  # pass rate at first measurement
    delta: float               # current - baseline (positive = improvement)
    cycle: int


class CapabilityTracker:
    """Tracks capability scores over time and detects compounding improvement."""

    def __init__(self, metrics_file: Path | None = None):
        self._file = metrics_file or METRICS_FILE
        self._file.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def record(self, domain: str, pass_rate: float, sample_size: int, cycle: int) -> None:
        """Append a capability snapshot to the metrics file."""
        baseline = self._get_baseline(domain)
        if baseline is None:
            baseline = pass_rate  # first measurement becomes the baseline

        delta = round(pass_rate - baseline, 6)

        snapshot = CapabilitySnapshot(
            timestamp=time.time(),
            domain=domain,
            pass_rate=round(pass_rate, 6),
            sample_size=sample_size,
            baseline_pass_rate=round(baseline, 6),
            delta=delta,
            cycle=cycle,
        )

        with self._file.open("a") as fh:
            fh.write(json.dumps(asdict(snapshot)) + "\n")

    def get_trend(self, domain: str, last_n: int = 10) -> dict:
        """Return trend stats for a domain: {current, baseline, delta, direction}."""
        records = self._load_domain(domain)
        if not records:
            return {"current": None, "baseline": None, "delta": None, "direction": "→"}

        recent = records[-last_n:]
        current = recent[-1]["pass_rate"]
        baseline = records[0]["baseline_pass_rate"]
        delta = round(current - baseline, 6)

        if delta > 0.01:
            direction = "↑"
        elif delta < -0.01:
            direction = "↓"
        else:
            direction = "→"

        return {
            "current": current,
            "baseline": baseline,
            "delta": delta,
            "direction": direction,
        }

    def get_summary(self) -> dict:
        """Return current capability summary across all domains."""
        all_records = self._load_all()
        # Group by domain, keep only the latest entry per domain
        latest: dict[str, dict] = {}
        for rec in all_records:
            latest[rec["domain"]] = rec

        summary = {}
        for domain, rec in sorted(latest.items()):
            trend = self.get_trend(domain)
            summary[domain] = {
                "pass_rate": rec["pass_rate"],
                "baseline": rec["baseline_pass_rate"],
                "delta": rec["delta"],
                "direction": trend["direction"],
                "cycle": rec["cycle"],
            }
        return summary

    def is_compounding(self, domain: str) -> bool:
        """Return True if pass_rate is trending up over last 5 measurements."""
        records = self._load_domain(domain)
        if len(records) < 2:
            return False

        recent = records[-5:]
        if len(recent) < 2:
            return False

        # Check that the linear trend is upward
        rates = [r["pass_rate"] for r in recent]
        n = len(rates)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(rates) / n
        numerator = sum((xs[i] - mean_x) * (rates[i] - mean_y) for i in range(n))
        denominator = sum((xs[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return False

        slope = numerator / denominator
        return slope > 0.0

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_all(self) -> list[dict]:
        """Load all records from the metrics file."""
        if not self._file.exists():
            return []
        records = []
        with self._file.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _load_domain(self, domain: str) -> list[dict]:
        """Load all records for a specific domain, ordered by timestamp."""
        all_records = self._load_all()
        domain_records = [r for r in all_records if r.get("domain") == domain]
        return sorted(domain_records, key=lambda r: r.get("timestamp", 0))

    def _get_baseline(self, domain: str) -> float | None:
        """Return the baseline pass_rate for a domain (first recorded value)."""
        records = self._load_domain(domain)
        if not records:
            return None
        return records[0]["baseline_pass_rate"]
