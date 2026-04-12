"""
Curiosity — Assessor Daemon (Daemon 1)

Runs in an infinite loop:
  1. Gap Mode  — probes the Server with gap benchmarks, detects failures and
                 regressions, and emits ProblemPackets to ASSESS_QUEUE.
  2. Curiosity Mode — after each gap scan, generates novel probes in unexplored
                 domains and emits them with source="curiosity".

Environment overrides:
  REDIS_HOST              (default: localhost)
  REDIS_PORT              (default: 6379)
  SERVER_URL              (default: http://localhost:8001/v1/chat/completions)
  SERVER_MODEL            (default: curiosity-server)
  GAP_SCAN_INTERVAL       (default: 300 seconds)
  FAILURE_THRESHOLD       (default: 0.10 — flag if >10% failure rate)
  CURIOSITY_NOVEL_COUNT   (default: 3 novel probes per cycle)
  BENCHMARK_GAP_DIR       (overrides default ~/curiosity/benchmarks/gap)
"""

import json
import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path

import redis

# Allow running as `python3 assessor.py` from the src/ tree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.types import ProblemPacket
from assessor.benchmark_loader import BenchmarkLoader
from assessor.probe_runner import ProbeRunner

# ── Logging ────────────────────────────────────────────────────────────────────

LOG_DIR = Path.home() / "curiosity" / "logs"
LOG_FILE = LOG_DIR / "assessor.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("assessor")

# ── Configuration (from env or defaults) ───────────────────────────────────────

REDIS_HOST: str = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.environ.get("REDIS_PORT", "6379"))
SERVER_URL: str = os.environ.get(
    "SERVER_URL", "http://localhost:8001/v1/chat/completions"
)
SERVER_MODEL: str = os.environ.get("SERVER_MODEL", "curiosity-server")
GAP_SCAN_INTERVAL: int = int(os.environ.get("GAP_SCAN_INTERVAL", "300"))
FAILURE_THRESHOLD: float = float(os.environ.get("FAILURE_THRESHOLD", "0.10"))
CURIOSITY_NOVEL_COUNT: int = int(os.environ.get("CURIOSITY_NOVEL_COUNT", "3"))

BENCHMARK_GAP_DIR: Path | None = (
    Path(os.environ["BENCHMARK_GAP_DIR"])
    if "BENCHMARK_GAP_DIR" in os.environ
    else None
)

ASSESS_QUEUE = "ASSESS_QUEUE"

# ── Helpers ────────────────────────────────────────────────────────────────────


def _redis_client() -> redis.Redis:
    """Return a Redis client; retries until available."""
    while True:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            logger.info("Connected to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
            return r
        except redis.RedisError as exc:
            logger.warning("Redis not available (%s) — retrying in 5s…", exc)
            time.sleep(5)


def _priority_score(failure_rate: float, frequency: int, novelty: float) -> float:
    """Compute a priority score: severity × frequency_weight × novelty."""
    severity = failure_rate  # 0–1
    freq_weight = min(1.0, frequency / 10.0)  # normalise; cap at 1.0
    return round(severity * 0.5 + freq_weight * 0.3 + novelty * 0.2, 4)


def _emit_packet(
    r: redis.Redis,
    domain: str,
    description: str,
    failure_rate: float,
    frequency: int,
    novelty_score: float,
    source: str = "gap",
) -> None:
    """Build a ProblemPacket and push it onto ASSESS_QUEUE."""
    priority = _priority_score(failure_rate, frequency, novelty_score)

    packet = ProblemPacket(
        domain=domain,
        description=description,
        failure_rate=round(failure_rate, 4),
        frequency=frequency,
        novelty_score=round(novelty_score, 4),
        priority_score=priority,
        source=source,  # type: ignore[arg-type]
    )

    payload = {
        "id": packet.id,
        "domain": packet.domain,
        "description": packet.description,
        "failure_rate": packet.failure_rate,
        "frequency": packet.frequency,
        "novelty_score": packet.novelty_score,
        "priority_score": packet.priority_score,
        "success_criterion": "",
        "scope": "swim",
        "source": source,
    }

    r.xadd(ASSESS_QUEUE, {"data": json.dumps(payload)})

    logger.info(
        "[emit] ASSESS_QUEUE ← domain=%s source=%s failure_rate=%.2f priority=%.4f id=%s",
        domain,
        source,
        failure_rate,
        priority,
        packet.id,
    )


# ── Gap Mode ───────────────────────────────────────────────────────────────────


def run_gap_mode(
    r: redis.Redis,
    loader: BenchmarkLoader,
    runner: ProbeRunner,
    baseline: dict,
) -> dict:
    """Run a full gap-mode benchmark scan.

    Returns the suite results dict (domain → stats) for this cycle.
    """
    logger.info("=== GAP MODE: starting scan ===")
    benchmarks = loader.load_gap_benchmarks()

    if not benchmarks:
        logger.warning("No gap benchmarks found — skipping gap scan")
        return {}

    suite = runner.run_suite(benchmarks)

    for domain, stats in suite.items():
        failure_rate: float = stats["failure_rate"]
        pass_rate: float = stats["pass_rate"]
        failures: list[dict] = stats["failures"]
        total: int = stats["total"]

        logger.info(
            "[gap] domain=%s | pass_rate=%.1f%% | failure_rate=%.1f%% | total=%d",
            domain,
            pass_rate * 100,
            failure_rate * 100,
            total,
        )

        # ── Failure-rate threshold check ──────────────────────────────────────
        if failure_rate > FAILURE_THRESHOLD:
            description = (
                f"Domain '{domain}' has {failure_rate*100:.1f}% failure rate "
                f"({len(failures)}/{total} benchmarks failing)."
            )
            _emit_packet(
                r,
                domain=domain,
                description=description,
                failure_rate=failure_rate,
                frequency=len(failures),
                novelty_score=0.0,
                source="gap",
            )

        # ── Regression check ──────────────────────────────────────────────────
        if domain in baseline:
            prev_pass_rate: float = baseline[domain]
            drop = prev_pass_rate - pass_rate
            if drop > FAILURE_THRESHOLD:
                description = (
                    f"REGRESSION in domain '{domain}': pass rate dropped from "
                    f"{prev_pass_rate*100:.1f}% to {pass_rate*100:.1f}% "
                    f"(Δ = {drop*100:.1f}%)."
                )
                logger.warning("[regression] %s", description)
                _emit_packet(
                    r,
                    domain=domain,
                    description=description,
                    failure_rate=failure_rate,
                    frequency=len(failures),
                    novelty_score=0.0,
                    source="gap",
                )

    logger.info("=== GAP MODE: scan complete ===")
    return suite


# ── Curiosity Mode ─────────────────────────────────────────────────────────────


def run_curiosity_mode(
    r: redis.Redis,
    loader: BenchmarkLoader,
    runner: ProbeRunner,
    novel_count: int = CURIOSITY_NOVEL_COUNT,
) -> None:
    """Generate and probe novel questions in unexplored domains."""
    logger.info("=== CURIOSITY MODE: generating novel probes ===")

    unexplored = loader.load_curiosity_domains()
    if not unexplored:
        logger.info("[curiosity] All known domains are covered — nothing to explore")
        return

    # Pick up to novel_count domains at random (avoid repeating the same ones)
    chosen_domains = random.sample(unexplored, min(novel_count, len(unexplored)))

    json_format_hint = (
        '{"question": "<the question>", "answer": "<correct answer>", '
        '"difficulty": "medium"}'
    )

    for domain in chosen_domains:
        generation_prompt = (
            f"Generate a challenging test question in domain: {domain}.\n"
            f"Format your response as valid JSON exactly like this:\n"
            f"{json_format_hint}\n"
            "No extra text outside the JSON."
        )

        try:
            raw_response, latency = runner._query_server(generation_prompt)
            logger.debug(
                "[curiosity] Generated probe for domain=%s in %.2fs", domain, latency
            )

            # Parse the generated question
            try:
                # Strip markdown code fences if present
                clean = raw_response.strip()
                if clean.startswith("```"):
                    lines = clean.splitlines()
                    clean = "\n".join(
                        l for l in lines if not l.startswith("```")
                    ).strip()

                generated = json.loads(clean)
                question = generated.get("question", "").strip()
                answer = generated.get("answer", "").strip()

                if not question or not answer:
                    raise ValueError("Missing question or answer in generated JSON")

            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "[curiosity] Failed to parse generated probe for domain=%s: %s",
                    domain,
                    exc,
                )
                # Fall back: treat entire response as the question, no check
                question = raw_response[:300]
                answer = ""

            # Build an ad-hoc benchmark and run it
            ad_hoc_benchmark = {
                "id": f"curiosity_{domain}_{uuid.uuid4().hex[:8]}",
                "domain": domain,
                "prompt": question,
                "expected": answer,
                "check": "llm_judge" if answer else "contains",
            }

            probe_result = runner.run_probe(ad_hoc_benchmark)

            # Curiosity probes always get emitted regardless of pass/fail
            # (they represent unexplored territory worth investigating)
            novelty_score = 0.8  # high novelty — unexplored domain
            failure_rate = 0.0 if probe_result["score"] else 1.0
            description = (
                f"Curiosity probe in unexplored domain '{domain}': '{question[:120]}'"
            )

            _emit_packet(
                r,
                domain=domain,
                description=description,
                failure_rate=failure_rate,
                frequency=1,
                novelty_score=novelty_score,
                source="curiosity",
            )

        except Exception as exc:
            logger.error(
                "[curiosity] Probe failed for domain=%s: %s", domain, exc
            )

    logger.info("=== CURIOSITY MODE: done ===")


# ── Main daemon loop ───────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Assessor daemon starting (v1) …")
    logger.info(
        "Config: REDIS=%s:%d | SERVER=%s | GAP_INTERVAL=%ds | THRESHOLD=%.0f%%",
        REDIS_HOST,
        REDIS_PORT,
        SERVER_URL,
        GAP_SCAN_INTERVAL,
        FAILURE_THRESHOLD * 100,
    )

    r = _redis_client()

    loader = BenchmarkLoader(gap_dir=BENCHMARK_GAP_DIR)
    runner = ProbeRunner(server_url=SERVER_URL, model=SERVER_MODEL)

    baseline: dict = loader.load_baseline()
    baseline_saved: bool = bool(baseline)

    cycle: int = 0

    while True:
        cycle += 1
        logger.info("── Assessor cycle #%d ──", cycle)
        cycle_start = time.monotonic()

        try:
            # 1. Gap Mode scan
            suite = run_gap_mode(r, loader, runner, baseline)

            # 2. Save baseline after first successful full scan
            if suite and not baseline_saved:
                loader.save_baseline(suite)
                baseline = loader.load_baseline()
                baseline_saved = True
                logger.info("Baseline stored after first successful scan")

            # Refresh baseline from disk every cycle (may have been updated externally)
            baseline = loader.load_baseline()

            # 3. Curiosity Mode probes
            run_curiosity_mode(r, loader, runner)

        except Exception as exc:
            # Never exit on exception — log and continue
            logger.exception("Unhandled exception in assessor cycle #%d: %s", cycle, exc)

        elapsed = time.monotonic() - cycle_start
        sleep_for = max(0, GAP_SCAN_INTERVAL - elapsed)

        logger.info(
            "Cycle #%d complete in %.1fs — sleeping %.1fs until next scan",
            cycle,
            elapsed,
            sleep_for,
        )
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
