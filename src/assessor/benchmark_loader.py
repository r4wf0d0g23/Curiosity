"""
Curiosity — BenchmarkLoader
Loads benchmark task files from disk and manages baseline pass-rate storage.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default benchmark directories (relative to ~/curiosity)
GAP_BENCHMARK_DIR = Path.home() / "curiosity" / "benchmarks" / "gap"
BASELINE_FILE = Path.home() / "curiosity" / "benchmarks" / "baselines.json"


class BenchmarkLoader:
    """Loads gap benchmarks and manages baseline pass-rate storage."""

    def __init__(
        self,
        gap_dir: Optional[Path] = None,
        baseline_file: Optional[Path] = None,
    ):
        self.gap_dir = Path(gap_dir) if gap_dir else GAP_BENCHMARK_DIR
        self.baseline_file = Path(baseline_file) if baseline_file else BASELINE_FILE

    # ── Gap Benchmarks ─────────────────────────────────────────────────────────

    def load_gap_benchmarks(self) -> list[dict]:
        """Load all .json files from the gap benchmark directory.

        Returns a flat list of benchmark task dicts.  Each dict must contain
        at least: id, domain, prompt, expected, check.
        """
        benchmarks: list[dict] = []

        if not self.gap_dir.exists():
            logger.warning("Gap benchmark dir not found: %s", self.gap_dir)
            return benchmarks

        for json_file in sorted(self.gap_dir.glob("*.json")):
            try:
                with json_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)

                # Support both a single dict and a list of dicts in one file
                if isinstance(data, list):
                    for item in data:
                        item.setdefault("_source_file", json_file.name)
                        benchmarks.append(item)
                elif isinstance(data, dict):
                    data.setdefault("_source_file", json_file.name)
                    benchmarks.append(data)
                else:
                    logger.warning("Unexpected JSON structure in %s — skipping", json_file)

            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Failed to load benchmark file %s: %s", json_file, exc)

        logger.info("Loaded %d benchmarks from %s", len(benchmarks), self.gap_dir)
        return benchmarks

    # ── Curiosity Domains ──────────────────────────────────────────────────────

    def load_curiosity_domains(self) -> list[str]:
        """Return a list of domains NOT already covered by the gap benchmarks.

        These are used for curiosity-mode novel probe generation.  The
        candidate pool is a curated set; domains already present in the gap
        suite are filtered out.
        """
        # Curated candidate domains the system can explore
        all_known_domains: list[str] = [
            "math",
            "code",
            "logic",
            "science",
            "history",
            "language",
            "reasoning",
            "statistics",
            "algorithms",
            "data_structures",
            "security",
            "ethics",
            "philosophy",
            "biology",
            "physics",
            "chemistry",
            "economics",
            "geography",
        ]

        covered: set[str] = set()
        benchmarks = self.load_gap_benchmarks()
        for b in benchmarks:
            domain = b.get("domain", "").strip().lower()
            if domain:
                covered.add(domain)

        unexplored = [d for d in all_known_domains if d not in covered]
        logger.debug(
            "Covered domains: %s | Unexplored: %s",
            sorted(covered),
            unexplored,
        )
        return unexplored

    # ── Baselines ──────────────────────────────────────────────────────────────

    def save_baseline(self, results: dict) -> None:
        """Persist current pass-rate results as the regression baseline.

        Args:
            results: Mapping of domain → {pass_rate, ...} as returned by
                     ProbeRunner.run_suite().
        """
        try:
            self.baseline_file.parent.mkdir(parents=True, exist_ok=True)

            # Flatten to {domain: pass_rate} for compact storage
            baseline: dict[str, float] = {}
            for domain, stats in results.items():
                if isinstance(stats, dict):
                    baseline[domain] = float(stats.get("pass_rate", 0.0))
                else:
                    baseline[domain] = float(stats)

            with self.baseline_file.open("w", encoding="utf-8") as fh:
                json.dump(baseline, fh, indent=2)

            logger.info("Baseline saved to %s: %s", self.baseline_file, baseline)

        except OSError as exc:
            logger.error("Failed to save baseline: %s", exc)

    def load_baseline(self) -> dict:
        """Load stored baseline pass rates.

        Returns a dict of {domain: pass_rate} or an empty dict if no baseline
        has been saved yet.
        """
        if not self.baseline_file.exists():
            logger.debug("No baseline file at %s", self.baseline_file)
            return {}

        try:
            with self.baseline_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            logger.info("Loaded baseline from %s: %s", self.baseline_file, data)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load baseline: %s", exc)
            return {}
