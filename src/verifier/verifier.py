"""
Curiosity — Verifier Daemon (Daemon 4)

Flow per solution plan:
  1. Read SolutionPlan from Redis stream SOLVE_QUEUE
  2. Create checkpoint of current Server state
  3. Apply modification to Server
  4. Run success criterion from Problem Packet
  5. Run regression suite
  6. PASS → commit; write VerificationResult to MEMORIZE_QUEUE
  7. FAIL → rollback; write failure VerificationResult to MEMORIZE_QUEUE;
             requeue problem to SOLVE_QUEUE with failure_mode

Never exits. All exceptions caught, logged, and the loop continues.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import redis
import requests

# ---------------------------------------------------------------------------
# Logging setup — runs before any import from sibling modules
# ---------------------------------------------------------------------------

LOG_DIR = Path(os.environ.get("CURIOSITY_LOG_DIR", Path.home() / "curiosity" / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "verifier.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("curiosity.verifier")

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

# Allow running from repo root without installing the package
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.shared.types import ProblemPacket, SolutionPlan, VerificationResult  # noqa: E402
from src.verifier.checkpoint import (  # noqa: E402
    CheckpointManager,
    CheckpointRecord,
    create_checkpoint,
    restore_checkpoint,
)

# ---------------------------------------------------------------------------
# Config (all override-able via env)
# ---------------------------------------------------------------------------

REDIS_HOST          = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT          = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB            = int(os.environ.get("REDIS_DB", 0))

SOLVE_QUEUE         = os.environ.get("SOLVE_QUEUE", "SOLVE_QUEUE")
MEMORIZE_QUEUE      = os.environ.get("MEMORIZE_QUEUE", "MEMORIZE_QUEUE")

VLLM_BASE_URL       = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")
SYSTEM_PROMPT_PATH  = Path(os.environ.get(
    "CURIOSITY_SYSTEM_PROMPT_PATH",
    Path.home() / "curiosity_code" / "config" / "system_prompt.txt"
))
BENCHMARK_DIR       = Path(os.environ.get(
    "CURIOSITY_BENCHMARK_DIR",
    Path.home() / "curiosity" / "benchmarks" / "regression"
))

REDIS_BLOCK_MS      = int(os.environ.get("REDIS_BLOCK_MS", 2000))
REDIS_RETRY_BACKOFF = float(os.environ.get("REDIS_RETRY_BACKOFF", 5.0))  # seconds
CRITERION_TIMEOUT   = int(os.environ.get("CRITERION_TIMEOUT", 120))       # seconds
REGRESSION_TIMEOUT  = int(os.environ.get("REGRESSION_TIMEOUT", 300))      # seconds


# ---------------------------------------------------------------------------
# Modification applicators
# ---------------------------------------------------------------------------

def _apply_prompt_patch(spec: Dict[str, Any]) -> None:
    """
    Modify the system prompt file.
    spec keys:
      - 'content'   : full replacement text  (mutually exclusive with patch_mode)
      - 'patch_mode': 'append' | 'prepend' | 'replace' (default 'replace')
      - 'insert'    : text to insert  (used with patch_mode)
    """
    content    = spec.get("content")
    patch_mode = spec.get("patch_mode", "replace")
    insert     = spec.get("insert", "")

    SYSTEM_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8") if SYSTEM_PROMPT_PATH.exists() else ""

    if content is not None:
        new_text = content
    elif patch_mode == "append":
        new_text = existing + "\n" + insert
    elif patch_mode == "prepend":
        new_text = insert + "\n" + existing
    else:
        new_text = insert  # 'replace' with insert field

    SYSTEM_PROMPT_PATH.write_text(new_text, encoding="utf-8")
    logger.info("[apply] prompt_patch written to %s", SYSTEM_PROMPT_PATH)


def _apply_lora_finetune(spec: Dict[str, Any]) -> None:
    """
    Placeholder — will load a LoRA adapter into the vLLM instance.
    spec keys (future): 'adapter_path', 'adapter_name', 'rank', 'alpha'
    """
    logger.info("[apply] lora_finetune placeholder — spec: %s", spec)
    # TODO: call vLLM dynamic LoRA load endpoint when available
    # e.g. requests.post(f"{VLLM_BASE_URL}/v1/load_lora_adapter", json=spec)
    raise NotImplementedError(
        "lora_finetune is a v0.1 placeholder — real implementation pending vLLM LoRA API."
    )


def _apply_weight_edit(spec: Dict[str, Any]) -> None:
    """
    Placeholder — will apply ROME/MEMIT weight edits to the model.
    spec keys (future): 'method', 'subject', 'relation', 'target', 'layer_range'
    """
    logger.info("[apply] weight_edit placeholder — spec: %s", spec)
    # TODO: integrate ROME/MEMIT editing library
    raise NotImplementedError(
        "weight_edit is a v0.1 placeholder — real implementation pending ROME/MEMIT integration."
    )


def _apply_composite(spec: Dict[str, Any]) -> None:
    """Apply multiple modification sub-specs in order."""
    steps: List[Dict[str, Any]] = spec.get("steps", [])
    for i, step in enumerate(steps):
        approach = step.get("approach", "")
        logger.info("[apply] composite step %d/%d: %s", i + 1, len(steps), approach)
        _APPLIERS[approach](step.get("spec", {}))


_APPLIERS = {
    "prompt_patch":  _apply_prompt_patch,
    "lora_finetune": _apply_lora_finetune,
    "weight_edit":   _apply_weight_edit,
    "composite":     _apply_composite,
}


def apply_modification(plan: SolutionPlan) -> None:
    """Dispatch to the correct applicator based on plan.approach."""
    approach = plan.approach
    if approach not in _APPLIERS:
        raise ValueError(f"Unknown modification approach: {approach!r}")
    _APPLIERS[approach](plan.modification_spec)


# ---------------------------------------------------------------------------
# Success criterion evaluators
# ---------------------------------------------------------------------------

def _run_benchmark(criterion: str, timeout: int) -> Tuple[bool, float, str]:
    """
    Run a benchmark-style criterion.
    criterion format: 'benchmark:<script_path>:<pass_threshold>'
    Returns (passed, score, detail).
    """
    parts = criterion.split(":", 2)
    if len(parts) < 2:
        return False, 0.0, f"Malformed benchmark criterion: {criterion}"
    script = parts[1] if len(parts) > 1 else ""
    threshold = float(parts[2]) if len(parts) > 2 else 0.8

    script_path = Path(script)
    if not script_path.exists():
        return False, 0.0, f"Benchmark script not found: {script_path}"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=timeout,
        )
        # Expect the script to print a JSON line: {"score": 0.92}
        score = 0.0
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    score = float(data.get("score", 0.0))
                    break
                except Exception:
                    pass
        passed = result.returncode == 0 and score >= threshold
        detail = result.stdout[-500:] + result.stderr[-200:]
        return passed, score, detail
    except subprocess.TimeoutExpired:
        return False, 0.0, f"Benchmark timed out after {timeout}s"
    except Exception as exc:
        return False, 0.0, f"Benchmark error: {exc}"


def _run_unit_test(criterion: str, timeout: int) -> Tuple[bool, float, str]:
    """
    Run pytest on a path embedded in the criterion string.
    criterion format: 'unit_test:<test_path>'
    Returns (passed, score, detail).
    """
    parts = criterion.split(":", 1)
    test_path = parts[1] if len(parts) > 1 else criterion
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "--tb=short", "-q"],
            capture_output=True, text=True, timeout=timeout,
        )
        passed = result.returncode == 0
        # Parse "X passed, Y failed" for a score
        score = 1.0 if passed else 0.0
        for line in result.stdout.splitlines():
            if "passed" in line or "failed" in line:
                try:
                    nums = [int(t) for t in line.split() if t.isdigit()]
                    if nums:
                        total = sum(nums)
                        n_pass = nums[0] if "passed" in line else 0
                        score = n_pass / total if total else 0.0
                except Exception:
                    pass
        detail = result.stdout[-500:] + result.stderr[-200:]
        return passed, score, detail
    except subprocess.TimeoutExpired:
        return False, 0.0, f"unit_test timed out after {timeout}s"
    except Exception as exc:
        return False, 0.0, f"unit_test error: {exc}"


def _run_llm_judge(criterion: str, plan: SolutionPlan, timeout: int) -> Tuple[bool, float, str]:
    """
    Ask the vLLM server to evaluate whether the modification is successful.
    criterion is the natural-language test definition.
    """
    prompt = (
        f"You are evaluating whether an AI server modification was successful.\n"
        f"Modification: {plan.description}\n"
        f"Expected outcome: {plan.expected_outcome}\n"
        f"Success criterion: {criterion}\n\n"
        f"Reply with ONLY a JSON object: {{\"pass\": true/false, \"score\": 0.0-1.0, \"reason\": \"...\"}}"
    )
    try:
        resp = requests.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 200,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Find JSON in response
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            passed = bool(data.get("pass", False))
            score  = float(data.get("score", 1.0 if passed else 0.0))
            reason = str(data.get("reason", ""))
            return passed, score, reason
        return False, 0.0, f"Could not parse LLM judge response: {text[:200]}"
    except requests.Timeout:
        return False, 0.0, f"LLM judge timed out after {timeout}s"
    except Exception as exc:
        return False, 0.0, f"LLM judge error: {exc}"


def evaluate_criterion(
    problem: ProblemPacket,
    plan: SolutionPlan,
    timeout: int = CRITERION_TIMEOUT,
) -> Tuple[bool, float, str]:
    """
    Evaluate the success criterion defined in the problem packet.
    Returns (passed, score 0-1, detail).
    """
    criterion      = problem.success_criterion
    criterion_type = problem.criterion_type

    if criterion_type == "benchmark" or criterion.startswith("benchmark:"):
        return _run_benchmark(criterion, timeout)
    elif criterion_type == "unit_test" or criterion.startswith("unit_test:"):
        return _run_unit_test(criterion, timeout)
    elif criterion_type == "llm_judge":
        return _run_llm_judge(criterion, plan, timeout)
    else:
        # Fallback: treat as llm_judge
        logger.warning("[criterion] Unknown criterion_type %r — falling back to llm_judge", criterion_type)
        return _run_llm_judge(criterion, plan, timeout)


# ---------------------------------------------------------------------------
# Regression suite
# ---------------------------------------------------------------------------

def run_regression_suite(timeout: int = REGRESSION_TIMEOUT) -> Tuple[bool, str]:
    """
    Execute all regression scripts found under BENCHMARK_DIR.
    Returns (all_passed, details).
    """
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    scripts = sorted(BENCHMARK_DIR.glob("*.py")) + sorted(BENCHMARK_DIR.glob("test_*.sh"))

    if not scripts:
        logger.info("[regression] No regression scripts found in %s — skipping", BENCHMARK_DIR)
        return True, "No regression scripts (skipped)"

    failures: List[str] = []
    for script in scripts:
        cmd = [sys.executable, str(script)] if script.suffix == ".py" else ["bash", str(script)]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                failures.append(
                    f"{script.name}: exit={result.returncode}\n"
                    f"  stdout: {result.stdout[-200:]}\n"
                    f"  stderr: {result.stderr[-200:]}"
                )
                logger.warning("[regression] FAIL %s (exit %d)", script.name, result.returncode)
            else:
                logger.info("[regression] PASS %s", script.name)
        except subprocess.TimeoutExpired:
            failures.append(f"{script.name}: TIMEOUT after {timeout}s")
            logger.warning("[regression] TIMEOUT %s", script.name)
        except Exception as exc:
            failures.append(f"{script.name}: ERROR {exc}")
            logger.error("[regression] ERROR %s: %s", script.name, exc)

    all_passed = len(failures) == 0
    details    = "\n".join(failures) if failures else f"All {len(scripts)} regression(s) passed"
    return all_passed, details


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def _make_redis_client() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


def _parse_stream_entry(entry_data: Dict[str, str]) -> Dict[str, Any]:
    raw = entry_data.get("data", "{}")
    return json.loads(raw)


def _deserialize_problem(d: Dict[str, Any]) -> ProblemPacket:
    valid = {f.name for f in dataclasses.fields(ProblemPacket)}
    return ProblemPacket(**{k: v for k, v in d.items() if k in valid})


def _deserialize_plan(d: Dict[str, Any]) -> SolutionPlan:
    valid = {f.name for f in dataclasses.fields(SolutionPlan)}
    return SolutionPlan(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# Verifier core logic (one iteration)
# ---------------------------------------------------------------------------

def _process_one(
    plan: SolutionPlan,
    problem: ProblemPacket,
    r: redis.Redis,
) -> VerificationResult:
    """
    Full verify-or-rollback cycle for one SolutionPlan.
    Returns the VerificationResult (always; never raises).
    """
    result = VerificationResult(
        solution_id=plan.id,
        problem_id=plan.problem_id or problem.id,
    )

    # Step 1 — Checkpoint BEFORE any modification
    try:
        logger.info("[verifier] Creating checkpoint for plan %s (approach=%s)", plan.id, plan.approach)
        cp: CheckpointRecord = create_checkpoint(
            label=f"pre_{plan.approach}_{plan.id[:8]}",
            note=f"Auto-checkpoint before applying plan {plan.id}",
        )
        plan.checkpoint_id = cp.id
        result.checkpoint_id = cp.id
        logger.info("[verifier] Checkpoint created: %s", cp.id)
    except Exception as exc:
        logger.error("[verifier] CHECKPOINT FAILED — aborting plan %s: %s", plan.id, exc)
        result.outcome = "fail"
        result.failure_mode = f"checkpoint_failed: {exc}"
        return result

    # Step 2 — Apply modification
    try:
        logger.info("[verifier] Applying %s modification for plan %s", plan.approach, plan.id)
        apply_modification(plan)
        logger.info("[verifier] Modification applied successfully")
    except NotImplementedError as exc:
        logger.warning("[verifier] Modification not implemented: %s", exc)
        result.outcome = "fail"
        result.failure_mode = f"not_implemented: {exc}"
        # No need to rollback since nothing was actually changed
        return result
    except Exception as exc:
        logger.error("[verifier] Modification FAILED for plan %s: %s\n%s", plan.id, exc, traceback.format_exc())
        result.outcome = "fail"
        result.failure_mode = f"apply_error: {exc}"
        # Rollback since partial writes may have occurred
        _safe_rollback(cp.id, result)
        return result

    # Step 3 — Evaluate success criterion
    try:
        logger.info("[verifier] Evaluating criterion for problem %s (type=%s)", problem.id, problem.criterion_type)
        passed, score, detail = evaluate_criterion(problem, plan)
        result.criterion_score = score
        logger.info("[verifier] Criterion result: passed=%s score=%.3f", passed, score)
    except Exception as exc:
        logger.error("[verifier] Criterion evaluation error: %s", exc)
        passed = False
        score  = 0.0
        detail = f"criterion_eval_error: {exc}"
        result.criterion_score = 0.0

    # Step 4 — Run regression suite
    regression_passed, regression_detail = True, ""
    try:
        logger.info("[verifier] Running regression suite")
        regression_passed, regression_detail = run_regression_suite()
        result.regression_detected = not regression_passed
        result.regression_details  = regression_detail
        logger.info("[verifier] Regression result: passed=%s", regression_passed)
    except Exception as exc:
        logger.error("[verifier] Regression suite error: %s", exc)
        regression_passed = False
        result.regression_detected = True
        result.regression_details  = f"regression_error: {exc}"

    overall_pass = passed and regression_passed

    if overall_pass:
        # Step 5a — PASS: commit (nothing to do for v0.1 — modification already applied)
        result.outcome        = "pass"
        result.failure_mode   = ""
        result.rolled_back    = False
        logger.info("[verifier] ✅ PASS plan=%s score=%.3f", plan.id, score)
    else:
        # Step 5b — FAIL: rollback
        result.outcome = "fail"
        if not passed:
            result.failure_mode = f"criterion_failed: score={score:.3f}; {detail[:200]}"
        if not regression_passed:
            result.failure_mode += ("; " if result.failure_mode else "") + f"regression: {regression_detail[:200]}"
        logger.warning("[verifier] ❌ FAIL plan=%s — rolling back", plan.id)
        _safe_rollback(cp.id, result)

    return result


def _safe_rollback(checkpoint_id: str, result: VerificationResult) -> None:
    """Roll back to checkpoint. Updates result.rolled_back in-place."""
    try:
        restore_checkpoint(checkpoint_id)
        result.rolled_back = True
        logger.info("[verifier] Rolled back to checkpoint %s ✅", checkpoint_id)
    except Exception as exc:
        result.rolled_back = False
        logger.critical("[verifier] ROLLBACK FAILED for checkpoint %s: %s", checkpoint_id, exc)


# ---------------------------------------------------------------------------
# Verifier Daemon class
# ---------------------------------------------------------------------------

class VerifierDaemon:
    """
    Infinite-loop daemon that reads SolutionPlans from SOLVE_QUEUE,
    verifies them with checkpoint/rollback, and writes results to MEMORIZE_QUEUE.
    """

    def __init__(self) -> None:
        self.r: Optional[redis.Redis] = None
        self._stream_cursor: Dict[str, str] = {SOLVE_QUEUE: "$"}

    # ------------------------------------------------------------------
    # Redis connection management
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish Redis connection with retry backoff."""
        while True:
            try:
                client = _make_redis_client()
                client.ping()
                self.r = client
                logger.info("[verifier] Connected to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
                return
            except redis.RedisError as exc:
                logger.error("[verifier] Redis connect failed: %s — retrying in %.1fs", exc, REDIS_RETRY_BACKOFF)
                time.sleep(REDIS_RETRY_BACKOFF)

    def _ensure_connected(self) -> redis.Redis:
        if self.r is None:
            self._connect()
        return self.r  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Read / write helpers
    # ------------------------------------------------------------------

    def _read_next_plan(self) -> Optional[Tuple[str, SolutionPlan, ProblemPacket]]:
        """
        Block on SOLVE_QUEUE for up to REDIS_BLOCK_MS ms.
        Returns (entry_id, plan, problem) or None if timeout.
        """
        r = self._ensure_connected()
        try:
            results = r.xread(self._stream_cursor, block=REDIS_BLOCK_MS, count=1)
        except redis.RedisError as exc:
            logger.error("[verifier] Redis read error: %s", exc)
            self.r = None
            return None

        if not results:
            return None

        stream_name, entries = results[0]
        entry_id, entry_data = entries[0]

        # Advance cursor so we don't re-process
        self._stream_cursor[SOLVE_QUEUE] = entry_id

        try:
            payload = _parse_stream_entry(entry_data)
            plan    = _deserialize_plan(payload.get("plan", payload))
            problem = _deserialize_problem(payload.get("problem", {}))
            logger.info("[verifier] Received plan id=%s approach=%s", plan.id, plan.approach)
            return entry_id, plan, problem
        except Exception as exc:
            logger.error("[verifier] Failed to deserialize entry %s: %s\nraw=%s", entry_id, exc, entry_data)
            return None

    def _publish_result(self, result: VerificationResult) -> None:
        """Write VerificationResult to MEMORIZE_QUEUE."""
        r = self._ensure_connected()
        try:
            r.xadd(MEMORIZE_QUEUE, {"data": json.dumps(asdict(result), default=str)})
            logger.info("[verifier] Result published to %s (outcome=%s)", MEMORIZE_QUEUE, result.outcome)
        except redis.RedisError as exc:
            logger.error("[verifier] Failed to publish result: %s", exc)
            self.r = None

    def _requeue_problem(self, problem: ProblemPacket, plan: SolutionPlan, result: VerificationResult) -> None:
        """Send failed problem back to SOLVE_QUEUE with failure context."""
        r = self._ensure_connected()
        retry_payload = {
            "problem": asdict(problem),
            "plan":    asdict(plan),
            "retry":   "1",
            "failure_mode": result.failure_mode,
        }
        try:
            r.xadd(SOLVE_QUEUE, {
                "data":   json.dumps(retry_payload, default=str),
                "retry":  "1",
                "from":   "verifier",
            })
            logger.info("[verifier] Requeued problem %s to %s (failure_mode=%s)",
                        problem.id, SOLVE_QUEUE, result.failure_mode[:80])
        except redis.RedisError as exc:
            logger.error("[verifier] Failed to requeue problem: %s", exc)
            self.r = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("=" * 70)
        logger.info("[verifier] Daemon starting (pid=%d)", os.getpid())
        logger.info("[verifier] SOLVE_QUEUE=%s  MEMORIZE_QUEUE=%s", SOLVE_QUEUE, MEMORIZE_QUEUE)
        logger.info("[verifier] Checkpoint dir: %s", Path.home() / "curiosity" / "checkpoints")
        logger.info("[verifier] Benchmark dir:  %s", BENCHMARK_DIR)
        logger.info("=" * 70)

        while True:
            try:
                item = self._read_next_plan()
                if item is None:
                    continue

                entry_id, plan, problem = item

                result = _process_one(plan, problem, self._ensure_connected())

                self._publish_result(result)

                if result.outcome == "fail":
                    # Don't requeue placeholder errors — they'll loop forever
                    if "not_implemented" not in result.failure_mode:
                        self._requeue_problem(problem, plan, result)
                    else:
                        logger.info("[verifier] Skipping requeue for not_implemented plan")

            except Exception as exc:
                # Outer catch-all: log and continue — daemon NEVER exits
                logger.critical(
                    "[verifier] Unhandled exception in main loop: %s\n%s",
                    exc, traceback.format_exc(),
                )
                time.sleep(1)  # brief pause to avoid tight crash loop


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    daemon = VerifierDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
