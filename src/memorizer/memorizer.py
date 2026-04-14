"""
Curiosity — Daemon 5: MEMORIZER
Reads VerificationResults from MEMORIZE_QUEUE, stores MemoryPaths to
ChromaDB + disk, then notifies ASSESS_QUEUE for loop closure.
"""

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import redis

from src.shared.types import (
    MemoryPath,
    ProblemPacket,
    SolutionPlan,
    VerificationResult,
)
from src.memorizer.memory_store import MemoryStore

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(os.path.expanduser("~/curiosity/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "memorizer.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("memorizer")

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
MEMORIZE_QUEUE = "MEMORIZE_QUEUE"
ASSESS_QUEUE = "ASSESS_QUEUE"

# How many ms to block waiting for new Redis stream messages
BLOCK_MS = 5_000

# Run improvement loop every N memories stored
IMPROVEMENT_LOOP_INTERVAL = int(os.environ.get("IMPROVEMENT_LOOP_INTERVAL", 20))


def _redis_connect() -> redis.Redis:
    """Return a connected Redis client, retrying until successful."""
    while True:
        try:
            r = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_timeout=10
            )
            r.ping()
            logger.info("Redis connected at %s:%d", REDIS_HOST, REDIS_PORT)
            return r
        except Exception as exc:
            logger.warning("Redis unavailable (%s), retrying in 5s…", exc)
            time.sleep(5)


def _deserialize_verification_result(data: dict) -> VerificationResult:
    """Reconstruct a VerificationResult from the Redis message payload."""
    direct = data.get("result")
    if isinstance(direct, dict):
        return VerificationResult(
            id=direct.get("id", ""),
            solution_id=direct.get("solution_id", ""),
            problem_id=direct.get("problem_id", ""),
            timestamp=direct.get("timestamp", datetime.utcnow().isoformat()),
            outcome=direct.get("outcome", "fail"),
            criterion_score=float(direct.get("criterion_score", 0.0)),
            regression_detected=bool(direct.get("regression_detected", False)),
            regression_details=direct.get("regression_details", ""),
            checkpoint_id=direct.get("checkpoint_id", ""),
            rolled_back=bool(direct.get("rolled_back", False)),
            failure_mode=direct.get("failure_mode", ""),
        )
    payload_str = data.get("payload") or data.get("verification_result") or ""
    if payload_str:
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            payload = {}
    else:
        # Fields may be stored flat in the message dict
        payload = data

    return VerificationResult(
        id=payload.get("id", ""),
        solution_id=payload.get("solution_id", ""),
        problem_id=payload.get("problem_id", ""),
        timestamp=payload.get("timestamp", datetime.utcnow().isoformat()),
        outcome=payload.get("outcome", "fail"),
        criterion_score=float(payload.get("criterion_score", 0.0)),
        regression_detected=bool(payload.get("regression_detected", False)),
        regression_details=payload.get("regression_details", ""),
        checkpoint_id=payload.get("checkpoint_id", ""),
        rolled_back=bool(payload.get("rolled_back", False)),
        failure_mode=payload.get("failure_mode", ""),
    )


def _deserialize_solution_plan(data: dict) -> SolutionPlan:
    direct = data.get("solution")
    if isinstance(direct, dict):
        return SolutionPlan(
            id=direct.get("id",""), problem_id=direct.get("problem_id",""),
            timestamp=direct.get("timestamp",""), approach=direct.get("approach","prompt_patch"),
            description=direct.get("description",""), modification_spec=direct.get("modification_spec",{}),
            expected_outcome=direct.get("expected_outcome",""), checkpoint_id=direct.get("checkpoint_id",""),
            from_memory=bool(direct.get("from_memory",False)), memory_solution_id=direct.get("memory_solution_id"),
        )
    payload_str = data.get("solution_plan") or ""
    if payload_str:
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = data

    return SolutionPlan(
        id=payload.get("id", ""),
        problem_id=payload.get("problem_id", ""),
        timestamp=payload.get("timestamp", datetime.utcnow().isoformat()),
        approach=payload.get("approach", "weight_edit"),
        description=payload.get("description", ""),
        modification_spec=payload.get("modification_spec", {}),
        expected_outcome=payload.get("expected_outcome", ""),
        checkpoint_id=payload.get("checkpoint_id", ""),
        from_memory=bool(payload.get("from_memory", False)),
        memory_solution_id=payload.get("memory_solution_id"),
    )


def _deserialize_problem_packet(data: dict) -> ProblemPacket:
    direct = data.get("problem")
    if isinstance(direct, dict):
        return ProblemPacket(
            id=direct.get("id",""), timestamp=direct.get("timestamp",""),
            domain=direct.get("domain","unknown"), description=direct.get("description",""),
            failure_rate=float(direct.get("failure_rate",0)), frequency=int(direct.get("frequency",0)),
            novelty_score=float(direct.get("novelty_score",0)), priority_score=float(direct.get("priority_score",0)),
            success_criterion=direct.get("success_criterion",""), criterion_type=direct.get("criterion_type",""),
            scope=direct.get("scope","swim"), source=direct.get("source","gap"),
        )
    payload_str = data.get("problem_packet") or ""
    if payload_str:
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = data

    return ProblemPacket(
        id=payload.get("id", ""),
        timestamp=payload.get("timestamp", datetime.utcnow().isoformat()),
        domain=payload.get("domain", "unknown"),
        description=payload.get("description", ""),
        failure_rate=float(payload.get("failure_rate", 0.0)),
        frequency=int(payload.get("frequency", 0)),
        novelty_score=float(payload.get("novelty_score", 0.0)),
        priority_score=float(payload.get("priority_score", 0.0)),
        success_criterion=payload.get("success_criterion", ""),
        criterion_type=payload.get("criterion_type", ""),
        scope=payload.get("scope", "swim"),
        source=payload.get("source", "gap"),
    )


def _build_memory_path(msg_data: dict) -> MemoryPath:
    """Construct a MemoryPath from a raw Redis stream message dict."""
    # The stream entry has a 'data' key containing a JSON string — parse it first
    raw = msg_data.get("data", "")
    if raw:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = msg_data
    else:
        parsed = msg_data
    problem = _deserialize_problem_packet(parsed)
    solution = _deserialize_solution_plan(parsed)
    result = _deserialize_verification_result(parsed)
    return MemoryPath(problem=problem, solution=solution, result=result)


def process_message(msg_data: dict, store: MemoryStore, r: redis.Redis) -> str:
    """Store one memory path and publish loop-closure event. Returns memory_id."""
    path = _build_memory_path(msg_data)
    memory_id = store.store_path(path)

    # Loop closure — notify Assessor that new memory is available
    # Backpressure: skip requeue if ASSESS_QUEUE is already overloaded
    ASSESS_BACKPRESSURE_LIMIT = 400
    assess_depth = r.xlen(ASSESS_QUEUE)
    if assess_depth >= ASSESS_BACKPRESSURE_LIMIT:
        logger.warning(
            "Loop closure SKIPPED (backpressure): ASSESS depth=%d >= %d (memory_id=%s domain=%s)",
            assess_depth, ASSESS_BACKPRESSURE_LIMIT, memory_id, path.problem.domain,
        )
    else:
        r.xadd(
            ASSESS_QUEUE,
            {
                "event": "memory_added",
                "memory_id": memory_id,
                "domain": path.problem.domain,
                "outcome": path.result.outcome,
                "criterion_score": str(path.result.criterion_score),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    logger.info(
        "Loop closed: memory_id=%s → %s written to %s",
        memory_id,
        path.result.outcome,
        ASSESS_QUEUE,
    )
    return memory_id


def run():
    """Main daemon loop — never exits on exception."""
    logger.info("=" * 60)
    logger.info("MEMORIZER daemon starting")
    logger.info("MEMORIZE_QUEUE → ChromaDB + disk → ASSESS_QUEUE")
    logger.info("=" * 60)

    store = MemoryStore()
    r = _redis_connect()

    # Track stream position; start from the current tail on fresh start
    last_id = "0"
    stored_count = 0

    while True:
        try:
            # Block waiting for new messages
            streams = r.xread({MEMORIZE_QUEUE: last_id}, block=BLOCK_MS, count=10)

            if not streams:
                # Timeout — still alive
                logger.debug("No new messages in %dms, still watching…", BLOCK_MS)
                continue

            for _stream_name, messages in streams:
                for msg_id, msg_data in messages:
                    last_id = msg_id  # advance cursor
                    try:
                        memory_id = process_message(msg_data, store, r)
                        stored_count += 1
                        logger.info(
                            "[%d] Stored memory_id=%s (msg_id=%s)",
                            stored_count,
                            memory_id,
                            msg_id,
                        )

                        # Periodically run improvement analysis
                        if stored_count % IMPROVEMENT_LOOP_INTERVAL == 0:
                            try:
                                store.run_improvement_loop()
                            except Exception as imp_exc:
                                logger.error(
                                    "Improvement loop error (non-fatal): %s", imp_exc
                                )

                    except Exception as msg_exc:
                        logger.error(
                            "Failed to process msg_id=%s: %s", msg_id, msg_exc, exc_info=True
                        )
                        # Continue — never drop the daemon over a bad message

        except redis.RedisError as redis_exc:
            logger.error("Redis error: %s — reconnecting in 5s…", redis_exc)
            time.sleep(5)
            r = _redis_connect()
            last_id = "0"

        except Exception as exc:
            logger.error("Unexpected error: %s — continuing in 2s…", exc, exc_info=True)
            time.sleep(2)


if __name__ == "__main__":
    run()
