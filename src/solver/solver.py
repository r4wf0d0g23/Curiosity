"""
Curiosity — Daemon 3: SOLVER
Memory-first solution retrieval + novel generation.

Flow:
  1. Read ProblemPacket from FORMULATE_QUEUE
  2. Query ChromaDB for similar past PASS solutions
  3a. similarity > 0.75 → adapt prior solution (from_memory=True)
  3b. no match         → generate novel solution via vLLM (port 8001)
  4. Write SolutionPlan to SOLVE_QUEUE
  5. If Verifier returns retry (retry=1 in SOLVE_QUEUE):
       - increment attempt count
       - try next memory hit or regenerate (avoiding known failures)
       - after 5 attempts → emit max_attempts_reached plan

Never exits on exception.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import redis

from src.shared.types import ProblemPacket, SolutionPlan
from src.solver.memory_retriever import MemoryRetriever, SIMILARITY_THRESHOLD
from src.solver.solution_generator import SolutionGenerator, _should_propose_finetune

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(os.path.expanduser("~/curiosity/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "solver.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("solver")

# ── Redis config ──────────────────────────────────────────────────────────────
REDIS_HOST      = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT      = int(os.environ.get("REDIS_PORT", 6379))
FORMULATE_QUEUE = "FORMULATE_QUEUE"
SOLVE_QUEUE     = "SOLVE_QUEUE"
BLOCK_MS        = 2_000   # ms to block on xread before timeout
TRAINING_LOCK_KEY = "CURIOSITY_TRAINING_LOCK"  # set by trainer while GPU is busy
CONSUMER_GROUP  = "solvers"

# ── Solver config ─────────────────────────────────────────────────────────────
MAX_ATTEMPTS         = 5
MEMORY_TOP_K         = 5


# ── Per-problem attempt state ─────────────────────────────────────────────────
class AttemptState:
    """Track retry context for a single problem_id."""

    def __init__(self, problem: ProblemPacket) -> None:
        self.problem        = problem
        self.attempt_count  = 0
        self.tried_memory_ids: list[str] = []    # memory_ids we already adapted
        self.tried_approaches: list[str] = []    # approach strings we tried
        self.memory_hits: list[dict] = []        # cached from first query


# ── Helpers ───────────────────────────────────────────────────────────────────

def _redis_connect() -> redis.Redis:
    """Return a live Redis client, retrying until successful."""
    while True:
        try:
            r = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_timeout=10,
            )
            r.ping()
            logger.info("Redis connected at %s:%d", REDIS_HOST, REDIS_PORT)
            return r
        except Exception as exc:
            logger.warning("Redis unavailable (%s), retrying in 5s…", exc)
            time.sleep(5)


def _deserialize_problem_packet(raw: dict) -> Optional[ProblemPacket]:
    """Try to parse a ProblemPacket from a raw Redis stream message dict."""
    # Messages from Formulator: 'data' key holds JSON
    # Handle retry messages where problem is already a dict
    problem_direct = raw.get("problem")
    if isinstance(problem_direct, dict) and problem_direct.get("domain"):
        payload = problem_direct
    else:
        payload_str = raw.get("data") or ""
        if payload_str:
            try:
                payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
            except (json.JSONDecodeError, TypeError):
                payload = {}
        else:
            payload = raw
        if isinstance(payload.get("problem"), dict):
            payload = payload["problem"]

    # Require at least an id or description to be a valid problem
    if not (payload.get("id") or payload.get("description")):
        return None

    try:
        return ProblemPacket(
            id=payload.get("id", str(uuid.uuid4())),
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
    except Exception as exc:
        logger.error("Failed to deserialize ProblemPacket: %s — raw=%s", exc, payload)
        return None


def _is_retry_message(raw: dict) -> bool:
    """Return True if this SOLVE_QUEUE message is a retry from the Verifier."""
    return raw.get("retry") == "1" or raw.get("from") == "verifier"


def _extract_retry_context(raw: dict) -> tuple[Optional[ProblemPacket], Optional[str]]:
    """
    Parse retry message from Verifier.
    Returns (problem, failure_mode).
    """
    payload_str = raw.get("data", "")
    try:
        payload = json.loads(payload_str) if payload_str else raw
    except json.JSONDecodeError:
        payload = raw

    problem = _deserialize_problem_packet(payload)
    failure_mode = str(payload.get("failure_mode", "unknown"))
    return problem, failure_mode


def _max_attempts_plan(problem: ProblemPacket) -> SolutionPlan:
    """Emit a terminal plan when all retry attempts are exhausted."""
    return SolutionPlan(
        id=str(uuid.uuid4()),
        problem_id=problem.id,
        timestamp=datetime.utcnow().isoformat(),
        approach="prompt_patch",
        description=f"max_attempts_reached: {MAX_ATTEMPTS} attempts exhausted for problem_id={problem.id}",
        modification_spec={
            "status": "max_attempts_reached",
            "max_attempts": MAX_ATTEMPTS,
            "domain": problem.domain,
        },
        expected_outcome="No further attempts will be made for this problem.",
        from_memory=False,
        memory_solution_id=None,
    )


# ── Core solver logic ─────────────────────────────────────────────────────────

class Solver:
    """
    Memory-first solution engine.

    Step 1: Query ChromaDB for past PASS solutions similar to this problem.
    Step 2a: If similarity > 0.75 → adapt best untried memory hit.
    Step 2b: Otherwise → generate novel solution via vLLM (avoiding known failures).
    """

    def __init__(self) -> None:
        self.retriever  = MemoryRetriever()
        self.generator  = SolutionGenerator()
        self._states: dict[str, AttemptState] = {}   # problem_id → AttemptState

    def _get_state(self, problem: ProblemPacket) -> AttemptState:
        if problem.id not in self._states:
            self._states[problem.id] = AttemptState(problem)
        return self._states[problem.id]

    def _purge_state(self, problem_id: str) -> None:
        self._states.pop(problem_id, None)

    def solve(self, problem: ProblemPacket) -> SolutionPlan:
        """
        Produce a SolutionPlan for the given problem.
        Increments attempt count; returns max_attempts_reached after MAX_ATTEMPTS.
        """
        state = self._get_state(problem)
        state.attempt_count += 1

        logger.info(
            "solve: problem_id=%s domain=%s attempt=%d/%d",
            problem.id,
            problem.domain,
            state.attempt_count,
            MAX_ATTEMPTS,
        )

        if state.attempt_count > MAX_ATTEMPTS:
            logger.warning(
                "solve: max_attempts_reached for problem_id=%s", problem.id
            )
            self._purge_state(problem.id)
            return _max_attempts_plan(problem)

        # ── Step 1: Memory-first query (cached after first attempt) ───────────
        if not state.memory_hits:
            logger.info(
                "solve: querying ChromaDB for similar past solutions "
                "(problem_id=%s domain=%s)",
                problem.id,
                problem.domain,
            )
            state.memory_hits = self.retriever.query_similar(problem, top_k=MEMORY_TOP_K)
            logger.info(
                "solve: ChromaDB returned %d pass hits for problem_id=%s",
                len(state.memory_hits),
                problem.id,
            )

        # ── Step 2a: Find best untried memory hit above threshold ─────────────
        best_hit: Optional[dict] = None
        for hit in state.memory_hits:
            similarity  = hit["similarity"]
            memory_id   = hit["memory_id"]

            if memory_id in state.tried_memory_ids:
                logger.debug(
                    "solve: skipping already-tried memory_id=%s", memory_id
                )
                continue

            if similarity >= SIMILARITY_THRESHOLD:
                best_hit = hit
                logger.info(
                    "solve: good memory match memory_id=%s similarity=%.4f "
                    "(threshold=%.2f) — adapting",
                    memory_id,
                    similarity,
                    SIMILARITY_THRESHOLD,
                )
                break
            else:
                logger.info(
                    "solve: memory_id=%s similarity=%.4f below threshold %.2f "
                    "— will generate novel solution",
                    memory_id,
                    similarity,
                    SIMILARITY_THRESHOLD,
                )
                break  # sorted by similarity; no point checking further

        if best_hit is not None:
            # ── Adapt memory solution ─────────────────────────────────────────
            plan = self.retriever.adapt_solution(best_hit, problem)
            state.tried_memory_ids.append(best_hit["memory_id"])
            state.tried_approaches.append(plan.approach)
            logger.info(
                "solve: adapted memory plan_id=%s from_memory=True attempt=%d",
                plan.id,
                state.attempt_count,
            )
            # Override: escalate to lora_finetune if memory-adapted prompt_patch is insufficient
            if plan.approach == "prompt_patch":
                _failures = self.retriever.get_failures(problem.domain)
                if _should_propose_finetune(problem, _failures):
                    logger.info(
                        "solve: overriding memory-adapted prompt_patch → lora_finetune "
                        "(problem_id=%s failure_rate=%.2f)",
                        problem.id, problem.failure_rate,
                    )
                    plan = self.generator._build_finetune_plan(problem)
                    state.tried_approaches.append("lora_finetune")
            return plan

        # ── Step 2b: Generate novel solution via vLLM ─────────────────────────
        logger.info(
            "solve: no qualifying memory hit — querying failures and generating "
            "novel solution for problem_id=%s attempt=%d",
            problem.id,
            state.attempt_count,
        )
        # Back off vLLM if trainer holds the training lock
        _backoff_iters = 0
        try:
            _lock_r = _redis_connect()
            while _lock_r.exists(TRAINING_LOCK_KEY):
                _backoff_iters += 1
                logger.info("Training lock active — backing off vLLM (iter=%d), sleeping 20s", _backoff_iters)
                import time as _time; _time.sleep(20)
        except Exception:
            pass  # non-fatal: if redis check fails, proceed
        failures = self.retriever.get_failures(problem.domain)
        plan = self.generator.generate(problem, failures)
        state.tried_approaches.append(plan.approach)
        logger.info(
            "solve: generated novel plan_id=%s approach=%s attempt=%d",
            plan.id,
            plan.approach,
            state.attempt_count,
        )
        return plan


# ── Daemon loop ───────────────────────────────────────────────────────────────

def run() -> None:
    """Main daemon loop — never exits on exception."""
    logger.info("=" * 60)
    logger.info("SOLVER daemon starting")
    logger.info("FORMULATE_QUEUE → [memory/generate] → SOLVE_QUEUE")
    logger.info("=" * 60)

    solver = Solver()
    r = _redis_connect()

    # Consumer group so multiple solver instances share the queue
    consumer_name = f"solver-{os.getpid()}"
    for queue in [FORMULATE_QUEUE, SOLVE_QUEUE]:
        try:
            r.xgroup_create(queue, CONSUMER_GROUP, id="0", mkstream=True)
            logger.info("Created consumer group '%s' on %s", CONSUMER_GROUP, queue)
        except Exception:
            pass  # group already exists

    while True:
        try:
            # ── Read from both queues via consumer group ──────────────────────
            streams = r.xreadgroup(
                CONSUMER_GROUP,
                consumer_name,
                {FORMULATE_QUEUE: ">", SOLVE_QUEUE: ">"},
                block=BLOCK_MS,
                count=2,
            )

            if not streams:
                logger.debug("No new messages in %dms, still watching…", BLOCK_MS)
                continue

            for stream_name, messages in streams:
                for msg_id, msg_data in messages:

                    try:
                        _dispatch(solver, r, stream_name, msg_id, msg_data)
                    except Exception as msg_exc:
                        logger.error(
                            "Unhandled error processing msg_id=%s stream=%s: %s",
                            msg_id,
                            stream_name,
                            msg_exc,
                            exc_info=True,
                        )
                    finally:
                        try:
                            r.xack(stream_name, CONSUMER_GROUP, msg_id)
                        except Exception:
                            pass
                        # Never drop the daemon

        except redis.RedisError as redis_exc:
            logger.error("Redis error: %s — reconnecting in 5s…", redis_exc)
            time.sleep(5)
            r = _redis_connect()
            cursor_formulate = "0"
            cursor_solve     = "0"

        except Exception as exc:
            logger.error(
                "Unexpected error in main loop: %s — continuing in 2s…",
                exc,
                exc_info=True,
            )
            time.sleep(2)


def _dispatch(
    solver: Solver,
    r: redis.Redis,
    stream_name: str,
    msg_id: str,
    msg_data: dict,
) -> None:
    """Route a single stream message to the appropriate handler."""

    if stream_name == FORMULATE_QUEUE:
        # ── New problem from Formulator ───────────────────────────────────────
        problem = _deserialize_problem_packet(msg_data)
        if problem is None:
            logger.warning(
                "Skipping unparseable FORMULATE_QUEUE message msg_id=%s", msg_id
            )
            return

        logger.info(
            "[NEW] problem_id=%s domain=%s priority=%.3f",
            problem.id,
            problem.domain,
            problem.priority_score,
        )
        plan = solver.solve(problem)
        _publish_plan(r, plan, msg_id, problem)

    elif stream_name == SOLVE_QUEUE and _is_retry_message(msg_data):
        # ── Retry from Verifier ───────────────────────────────────────────────
        problem, failure_mode = _extract_retry_context(msg_data)
        if problem is None:
            logger.warning(
                "Skipping unparseable retry message msg_id=%s", msg_id
            )
            return

        logger.info(
            "[RETRY] problem_id=%s domain=%s failure_mode=%s",
            problem.id,
            problem.domain,
            failure_mode,
        )
        plan = solver.solve(problem)
        _publish_plan(r, plan, msg_id, problem)

    else:
        # Non-retry SOLVE_QUEUE message (our own writes or Verifier passes) — skip
        logger.debug(
            "Ignoring non-retry SOLVE_QUEUE message msg_id=%s", msg_id
        )


def _publish_plan(r: redis.Redis, plan: SolutionPlan, source_msg_id: str, problem: ProblemPacket = None) -> None:
    """Serialize a SolutionPlan (+ problem context) and push it to SOLVE_QUEUE."""
    payload = json.dumps({
        "plan": dataclasses.asdict(plan),
        "problem": dataclasses.asdict(problem) if problem else {},
    }, default=str)
    stream_id = r.xadd(SOLVE_QUEUE, {"data": payload})
    logger.info(
        "Published plan_id=%s approach=%s from_memory=%s problem_id=%s "
        "→ SOLVE_QUEUE stream_id=%s",
        plan.id,
        plan.approach,
        plan.from_memory,
        plan.problem_id,
        stream_id,
    )


if __name__ == "__main__":
    run()
