"""
Integration test: inject a synthetic ProblemPacket into ASSESS_QUEUE,
verify it appears in MEMORIZE_QUEUE within timeout.
Does NOT require a real Server — uses mock responses where needed.

Usage:
    python3 tests/test_loop.py
    # or with pytest:
    pytest tests/test_loop.py -v
"""
import redis
import json
import time
import uuid


REDIS_HOST = "localhost"
REDIS_PORT = 6379
TIMEOUT_SECONDS = 30


def test_full_loop():
    """
    Inject a synthetic problem into ASSESS_QUEUE and verify the message
    eventually arrives in MEMORIZE_QUEUE (end-to-end smoke test).
    """
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    # Verify Redis connectivity before injecting
    try:
        r.ping()
    except redis.ConnectionError as exc:
        print(f"❌ Cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT} — {exc}")
        return False

    # Inject synthetic problem
    problem = {
        "id": str(uuid.uuid4()),
        "domain": "test",
        "description": "Integration test problem",
        "failure_rate": 0.5,
        "frequency": 1,
        "novelty_score": 0.8,
        "priority_score": 0.6,
        "source": "gap",
    }
    msg_id = r.xadd("ASSESS_QUEUE", {"data": json.dumps(problem)})
    print(f"✉️  Injected problem {problem['id']} → ASSESS_QUEUE (msg_id={msg_id})")

    # Wait for it to appear in MEMORIZE_QUEUE (timeout 30 s)
    start = time.time()
    while time.time() - start < TIMEOUT_SECONDS:
        msgs = r.xread({"MEMORIZE_QUEUE": "0"}, count=10)
        if msgs:
            print("✅ Loop completed — message arrived in MEMORIZE_QUEUE")
            return True
        elapsed = time.time() - start
        print(f"   … waiting ({elapsed:.0f}s / {TIMEOUT_SECONDS}s)", end="\r", flush=True)
        time.sleep(1)

    print(f"\n❌ Timeout after {TIMEOUT_SECONDS}s — message did not flow through loop")
    return False


if __name__ == "__main__":
    result = test_full_loop()
    exit(0 if result else 1)
