"""
Curiosity — Daemon 2: FORMULATOR
Reads raw assessment signals from ASSESS_QUEUE, enriches them into fully-formed
ProblemPackets, and publishes sorted results to FORMULATE_QUEUE.

Pipeline per batch
------------------
1. Read up to 10 messages from ASSESS_QUEUE (blocking xread).
2. For each signal:
   a. Deserialise into a working dict.
   b. Generate an automatable success criterion (CriterionGenerator).
   c. Skip if automatable=False — log reason, do not forward.
   d. Classify scope: swim vs bridge (ScopeClassifier, rolling 100-item history).
   e. Compute priority score (PriorityRanker).
   f. Assemble a complete ProblemPacket.
3. Sort batch by priority_score descending.
4. Write each packet to FORMULATE_QUEUE.
5. Repeat forever — never exits.
"""

import dataclasses
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import redis

from src.shared.types import ProblemPacket
from src.formulator.criterion_generator import CriterionGenerator
from src.formulator.scope_classifier import ScopeClassifier
from src.formulator.priority_ranker import PriorityRanker

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(os.path.expanduser("~/curiosity/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "formulator.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("formulator")

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_HOST    = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT    = int(os.environ.get("REDIS_PORT", 6379))
ASSESS_QUEUE   = "ASSESS_QUEUE"
FORMULATE_QUEUE = "FORMULATE_QUEUE"

# How long (ms) to block waiting for new Redis stream messages
BLOCK_MS = 1_000

# Rolling history size for scope classification
HISTORY_MAX = 100

# Batch size
BATCH_SIZE = 10


# ── Redis helpers ─────────────────────────────────────────────────────────────

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
            logger.warning("Redis unavailable (%s) — retrying in 5 s…", exc)
            time.sleep(5)


# ── Deserialisation ───────────────────────────────────────────────────────────

def _parse_signal(msg_data: dict) -> dict:
    """
    Normalise a raw Redis message dict into a flat problem dict.

    The upstream Assessor may write the payload:
      • as a JSON string under the key 'data', or
      • flat in the message dict itself.
    """
    raw = msg_data.get("data", "")
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = msg_data
    else:
        payload = msg_data

    return {
        "id":           payload.get("id", ""),
        "timestamp":    payload.get("timestamp", datetime.utcnow().isoformat()),
        "domain":       payload.get("domain", "unknown"),
        "description":  payload.get("description", ""),
        "failure_rate": float(payload.get("failure_rate", 0.0)),
        "frequency":    int(payload.get("frequency", 1)),
        "novelty_score":float(payload.get("novelty_score", 0.5)),
        "source":       payload.get("source", "gap"),
    }


# ── Core formulation ──────────────────────────────────────────────────────────

def formulate(
    signal: dict,
    criterion_gen: CriterionGenerator,
    scope_cls: ScopeClassifier,
    ranker: PriorityRanker,
    history: deque,
) -> ProblemPacket | None:
    """
    Convert a raw signal dict into a ProblemPacket.

    Returns None if the problem cannot be given an automatable criterion
    (caller must log and skip such signals).
    """
    domain       = signal["domain"]
    description  = signal["description"]
    failure_rate = signal["failure_rate"]
    frequency    = signal["frequency"]
    novelty      = signal["novelty_score"]

    # ── 1. Success criterion ──────────────────────────────────────────────────
    criterion = criterion_gen.generate(signal)

    if not criterion.get("automatable", False):
        logger.warning(
            "SKIPPED domain=%s — non-automatable criterion: %s",
            domain, criterion.get("description", "no description"),
        )
        return None

    # ── 2. Scope classification ───────────────────────────────────────────────
    scope = scope_cls.classify(signal, list(history))

    # ── 3. Priority score ─────────────────────────────────────────────────────
    # severity is proxied by failure_rate (higher failure = more severe)
    priority = ranker.score(
        severity=failure_rate,
        frequency=frequency,
        novelty=novelty,
    )

    # ── 4. Assemble ProblemPacket ─────────────────────────────────────────────
    # Preserve the upstream id when the Assessor provided one; otherwise the
    # dataclass default_factory generates a fresh UUID.
    extra_kwargs = {"id": signal["id"]} if signal["id"] else {}

    packet = ProblemPacket(
        **extra_kwargs,
        timestamp      = signal["timestamp"],
        domain         = domain,
        description    = description,
        failure_rate   = failure_rate,
        frequency      = frequency,
        novelty_score  = novelty,
        priority_score = priority,
        success_criterion = criterion["description"],
        criterion_type = criterion["criterion"],
        scope          = scope,
        source         = signal.get("source", "gap"),
    )

    logger.info(
        "FORMULATED domain=%s scope=%s criterion=%s priority=%.4f packet_id=%s",
        domain, scope, criterion["criterion"], priority, packet.id,
    )
    return packet


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    logger.info("=" * 60)
    logger.info("FORMULATOR daemon starting")
    logger.info("ASSESS_QUEUE → [formulate] → FORMULATE_QUEUE")
    logger.info("=" * 60)

    criterion_gen = CriterionGenerator()
    scope_cls     = ScopeClassifier()
    ranker        = PriorityRanker()
    history: deque = deque(maxlen=HISTORY_MAX)

    r = _redis_connect()
    last_id = "$"  # start from current tail; use '0' to replay all

    processed_total = 0
    skipped_total   = 0

    while True:
        try:
            streams = r.xread({ASSESS_QUEUE: last_id}, block=BLOCK_MS, count=BATCH_SIZE)

            if not streams:
                logger.debug("No new messages in %d ms — still watching…", BLOCK_MS)
                continue

            batch_packets: list[tuple[str, ProblemPacket]] = []  # (msg_id, packet)

            for _stream_name, messages in streams:
                for msg_id, msg_data in messages:
                    last_id = msg_id  # always advance cursor

                    try:
                        signal = _parse_signal(msg_data)

                        packet = formulate(
                            signal, criterion_gen, scope_cls, ranker, history
                        )

                        if packet is None:
                            skipped_total += 1
                            logger.info(
                                "SKIP msg_id=%s domain=%s (non-automatable) "
                                "[skipped_total=%d]",
                                msg_id, signal.get("domain", "?"), skipped_total,
                            )
                            continue

                        batch_packets.append((msg_id, packet))
                        # Update rolling history after successful formulation
                        history.append(signal)

                    except Exception as msg_exc:
                        logger.error(
                            "Error processing msg_id=%s: %s",
                            msg_id, msg_exc, exc_info=True,
                        )
                        # Never crash the daemon over a single bad message

            # ── Sort batch: highest priority first ────────────────────────
            batch_packets.sort(key=lambda t: t[1].priority_score, reverse=True)

            # ── Publish to FORMULATE_QUEUE ─────────────────────────────────
            for msg_id, packet in batch_packets:
                try:
                    r.xadd(
                        FORMULATE_QUEUE,
                        {"data": json.dumps(dataclasses.asdict(packet))},
                    )
                    processed_total += 1
                    logger.info(
                        "[%d] Published packet_id=%s domain=%s priority=%.4f "
                        "scope=%s criterion=%s (from msg_id=%s)",
                        processed_total,
                        packet.id,
                        packet.domain,
                        packet.priority_score,
                        packet.scope,
                        packet.criterion_type,
                        msg_id,
                    )
                except Exception as pub_exc:
                    logger.error(
                        "Failed to publish packet_id=%s: %s",
                        packet.id, pub_exc, exc_info=True,
                    )

        except redis.RedisError as redis_exc:
            logger.error("Redis error: %s — reconnecting in 5 s…", redis_exc)
            time.sleep(5)
            r = _redis_connect()
            last_id = "$"

        except Exception as exc:
            logger.error("Unexpected error: %s — continuing in 2 s…", exc, exc_info=True)
            time.sleep(2)


if __name__ == "__main__":
    run()
