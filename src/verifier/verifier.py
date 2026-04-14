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

import re

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

from src.shared.types import ProblemPacket, SolutionPlan, TrainingJob, VerificationResult  # noqa: E402
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
TRAIN_QUEUE         = os.environ.get("TRAIN_QUEUE", "TRAIN_QUEUE")

# Redis keys used by the trainer daemon
ACTIVE_ADAPTERS_KEY = "CURIOSITY_ACTIVE_ADAPTERS"
TRAINING_LOCK_KEY   = "CURIOSITY_TRAINING_LOCK"

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
VERIFIER_CONSUMER_GROUP = "verifiers"
REDIS_RETRY_BACKOFF = float(os.environ.get("REDIS_RETRY_BACKOFF", 5.0))  # seconds
CRITERION_TIMEOUT   = int(os.environ.get("CRITERION_TIMEOUT", 3600))   # 1 hour default       # seconds
REGRESSION_TIMEOUT  = int(os.environ.get("REGRESSION_TIMEOUT", 1800))  # 30 min default      # seconds


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

    if not new_text.strip():
        logger.warning("[apply] prompt_patch: refusing empty write — spec=%s", spec)
        return
    SYSTEM_PROMPT_PATH.write_text(new_text, encoding="utf-8")
    logger.info("[apply] prompt_patch written to %s", SYSTEM_PROMPT_PATH)


def _apply_lora_finetune(spec: Dict[str, Any]) -> None:
    """
    Handle LoRA fine-tuning plans. Two modes:

    1. Initial dispatch: spec has no 'adapter_path' — the plan is requesting
       training. We build a TrainingJob and push it to TRAIN_QUEUE. The
       verifier then skips further verification (training is async).

    2. Post-training verification: spec has 'adapter_path' + 'adapter_name' —
       the trainer has completed and we verify the loaded adapter by running
       benchmarks with the LoRA request.
    """
    if spec.get("adapter_path"):
        # Post-training: adapter already loaded by trainer daemon, nothing to apply
        logger.info(
            "[apply] lora_finetune: adapter already loaded — name=%s path=%s",
            spec.get("adapter_name"), spec.get("adapter_path"),
        )
        return

    # Initial dispatch: push training job to TRAIN_QUEUE
    logger.info("[apply] lora_finetune: dispatching training job to TRAIN_QUEUE")
    raise _LoraDispatchSignal(spec)


def _apply_weight_edit(spec: Dict[str, Any]) -> None:
    """
    Weight edit via targeted micro-finetune.
    Converts ROME-style edit triples into a mini TrainingJob and dispatches
    to TRAIN_QUEUE. The trainer generates Q&A pairs focused on the specific
    facts and applies QLoRA (rank=8, 1–3 epochs) to teach them.

    spec keys:
      method : str  — "ROME" or similar (informational only)
      edits  : list of {"subject": ..., "relation": ..., "target": ...}
    """
    edits = spec.get("edits", [])
    if not edits:
        # Fallback: spec may use flat keys
        subject  = spec.get("subject", "")
        relation = spec.get("relation", "")
        target   = spec.get("target", "")
        if subject and target:
            edits = [{"subject": subject, "relation": relation, "target": target}]
    if not edits:
        raise ValueError("weight_edit spec missing 'edits' list — cannot dispatch")
    logger.info("[apply] weight_edit: dispatching micro-finetune for %d edit(s) to TRAIN_QUEUE", len(edits))
    raise _WeightEditDispatchSignal(spec, edits)


class _WeightEditDispatchSignal(Exception):
    """Raised by _apply_weight_edit to signal dispatch of a micro-finetune to TRAIN_QUEUE."""
    def __init__(self, spec, edits):
        super().__init__("weight_edit_dispatch")
        self.spec  = spec
        self.edits = edits


class _LoraDispatchSignal(Exception):
    """Raised by _apply_lora_finetune to signal the verifier to dispatch to TRAIN_QUEUE."""
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        super().__init__("lora_dispatch")


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

def _run_benchmark(
    problem: "ProblemPacket",
    plan: "SolutionPlan",
    timeout: int,
    lora_request: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, str]:
    """
    Run a benchmark-style criterion using a JSON benchmark file.
    Parses benchmark_id and threshold from problem.success_criterion.
    Optionally passes a lora_request to vLLM for adapter-based inference.
    Returns (passed, score, detail).
    """
    m = re.search(r"Pass rate on (\S+) suite must reach >= ([\d.]+)", problem.success_criterion)
    if m:
        benchmark_id = m.group(1)
        threshold = float(m.group(2))
    else:
        benchmark_id = f"{problem.domain}_basic"
        threshold = 0.85

    benchmark_path = Path.home() / "curiosity" / "benchmarks" / "gap" / f"{benchmark_id}.json"
    if not benchmark_path.exists():
        return False, 0.0, f"Benchmark not found: {benchmark_id}"

    try:
        items = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if not items:
            return False, 0.0, "Empty benchmark file"

        passed_count = 0
        details: List[str] = []
        for item in items:
            prompt   = item.get("prompt", "")
            expected = item.get("expected", "")

            try:
                request_body: Dict[str, Any] = {
                    "model": "nemotron3-super",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 512,
                }
                if lora_request:
                    request_body["lora_request"] = lora_request

                resp = requests.post(
                    f"{VLLM_BASE_URL}/v1/chat/completions",
                    json=request_body,
                    timeout=timeout,
                )
                resp.raise_for_status()
                answer = resp.json()["choices"][0]["message"]["content"]
                if expected.lower() in answer.lower():
                    passed_count += 1
                else:
                    details.append(f"FAIL: expected {expected!r} not in response")
            except Exception as exc:
                details.append(f"ERROR: {exc}")

        score = passed_count / len(items)
        detail_str = f"{passed_count}/{len(items)} passed"
        if details:
            detail_str += "; " + "; ".join(details[:5])
        return score >= threshold, score, detail_str
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
                "model": "nemotron3-super",
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
        return _run_benchmark(problem, plan, timeout)
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
    except _LoraDispatchSignal as sig:
        # LoRA fine-tuning needs async training — dispatch to TRAIN_QUEUE
        logger.info("[verifier] LoRA dispatch: pushing training job to TRAIN_QUEUE")
        result.outcome = "fail"
        result.failure_mode = "lora_dispatched_to_trainer"
        result._lora_dispatch_spec = sig.spec  # type: ignore[attr-defined]
        return result
    except _WeightEditDispatchSignal as sig:
        # Weight edit — dispatch as micro-finetune to TRAIN_QUEUE
        logger.info("[verifier] Weight-edit dispatch: converting %d edit(s) to micro-finetune", len(sig.edits))
        self._dispatch_weight_edit_job(problem, plan, sig.edits)
        result.outcome = "fail"
        result.failure_mode = "weight_edit_dispatched_to_trainer"
        return result
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
    # For lora_finetune plans returning from training, run A/B comparison
    lora_request = None
    if plan.approach == "lora_finetune" and plan.modification_spec.get("adapter_name"):
        adapter_name = plan.modification_spec["adapter_name"]
        adapter_path = plan.modification_spec.get("adapter_path", "")
        lora_request = {
            "lora_name": adapter_name,
            "lora_int_id": 1,
            "lora_path": adapter_path,
        }
        logger.info("[verifier] Running A/B benchmark: adapter=%s", adapter_name)

    try:
        logger.info("[verifier] Evaluating criterion for problem %s (type=%s)", problem.id, problem.criterion_type)
        if lora_request:
            # Run WITH adapter
            passed_with, score_with, detail_with = _run_benchmark(problem, plan, CRITERION_TIMEOUT, lora_request=lora_request)
            # Run WITHOUT adapter (baseline)
            passed_base, score_base, detail_base = _run_benchmark(problem, plan, CRITERION_TIMEOUT, lora_request=None)
            improvement = score_with - score_base
            logger.info(
                "[verifier] A/B result: with_adapter=%.3f baseline=%.3f improvement=%.3f",
                score_with, score_base, improvement,
            )
            # Pass if adapter improves by >5% over baseline
            passed = improvement > 0.05
            score = score_with
            detail = (
                f"adapter={score_with:.3f} baseline={score_base:.3f} "
                f"improvement={improvement:+.3f}; {detail_with}"
            )
        else:
            # Pause if trainer holds vLLM lock
            _train_lock_iters = 0
            while r.exists(TRAINING_LOCK_KEY):
                _train_lock_iters += 1
                logger.info("[verifier] Training lock active — pausing benchmark (iter=%d)", _train_lock_iters)
                import time as _tv; _tv.sleep(20)
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
        # Step 5a — PASS: commit
        result.outcome        = "pass"
        result.failure_mode   = ""
        result.rolled_back    = False
        logger.info("[verifier] ✅ PASS plan=%s score=%.3f", plan.id, score)

        # For LoRA adapters: keep it loaded permanently (register in active list)
        if lora_request and plan.modification_spec.get("adapter_name"):
            _register_active_adapter(
                r,
                plan.modification_spec["adapter_name"],
                plan.modification_spec.get("adapter_path", ""),
                plan.modification_spec.get("training_domain", ""),
            )
    else:
        # Step 5b — FAIL: rollback
        result.outcome = "fail"
        if not passed:
            result.failure_mode = f"criterion_failed: score={score:.3f}; {detail[:200]}"
        if not regression_passed:
            result.failure_mode += ("; " if result.failure_mode else "") + f"regression: {regression_detail[:200]}"
        logger.warning("[verifier] ❌ FAIL plan=%s — rolling back", plan.id)
        _safe_rollback(cp.id, result)

        # For LoRA adapters: unload and clean up on failure
        if lora_request and plan.modification_spec.get("adapter_name"):
            _unload_failed_adapter(
                r,
                plan.modification_spec["adapter_name"],
                plan.modification_spec.get("adapter_path", ""),
            )

    return result


def _register_active_adapter(
    r: redis.Redis, adapter_name: str, adapter_path: str, domain: str,
) -> None:
    """Register a verified LoRA adapter as permanently active."""
    try:
        import json as _json
        info = _json.dumps({
            "name": adapter_name,
            "path": adapter_path,
            "domain": domain,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        })
        r.hset(ACTIVE_ADAPTERS_KEY, adapter_name, info)
        logger.info("[verifier] Registered active adapter: %s", adapter_name)
    except Exception as exc:
        logger.error("[verifier] Failed to register adapter %s: %s", adapter_name, exc)


def _unload_failed_adapter(r: redis.Redis, adapter_name: str, adapter_path: str) -> None:
    """Unload a failed LoRA adapter from vLLM and clean up files."""
    # Unload from vLLM
    try:
        resp = requests.post(
            f"{VLLM_BASE_URL}/v1/unload_lora_adapter",
            json={"lora_name": adapter_name},
            timeout=30,
        )
        if resp.status_code == 200:
            logger.info("[verifier] Unloaded failed adapter: %s", adapter_name)
        else:
            logger.warning("[verifier] Unload adapter returned %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.error("[verifier] Failed to unload adapter %s: %s", adapter_name, exc)

    # Remove from active adapters list
    try:
        r.hdel(ACTIVE_ADAPTERS_KEY, adapter_name)
    except Exception:
        pass

    # Delete adapter files
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if adapter_dir.exists() and adapter_dir.is_dir():
            import shutil
            try:
                shutil.rmtree(adapter_dir)
                logger.info("[verifier] Deleted failed adapter files: %s", adapter_path)
            except Exception as exc:
                logger.error("[verifier] Failed to delete adapter files %s: %s", adapter_path, exc)


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
        self._stream_cursor: Dict[str, str] = {SOLVE_QUEUE: "0"}

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
            consumer_name = f"verifier-{os.getpid()}"
            try:
                r.xgroup_create(SOLVE_QUEUE, VERIFIER_CONSUMER_GROUP, id="0", mkstream=True)
            except Exception:
                pass
            results = r.xreadgroup(
                VERIFIER_CONSUMER_GROUP, consumer_name,
                {SOLVE_QUEUE: ">"},
                block=REDIS_BLOCK_MS, count=1
            )
        except redis.RedisError as exc:
            logger.error("[verifier] Redis read error: %s", exc)
            self.r = None
            return None

        if not results:
            return None

        stream_name, entries = results[0]
        entry_id, entry_data = entries[0]


        try:
            payload = _parse_stream_entry(entry_data)
            plan    = _deserialize_plan(payload.get("plan", payload))
            problem = _deserialize_problem(payload.get("problem", {}))
            # Preserve attempt_count across requeues (not a SolutionPlan field)
            plan.attempt_count = int(payload.get("attempt_count", "0"))
            logger.info("[verifier] Received plan id=%s approach=%s attempt=%d",
                        plan.id, plan.approach, plan.attempt_count)
            return entry_id, plan, problem
        except Exception as exc:
            logger.error("[verifier] Failed to deserialize entry %s: %s\nraw=%s", entry_id, exc, entry_data)
            return None

    def _publish_result(self, result: VerificationResult, problem: ProblemPacket = None, plan: SolutionPlan = None) -> None:
        """Write full context (result + problem + solution) to MEMORIZE_QUEUE."""
        r = self._ensure_connected()
        try:
            payload = {
                "result": asdict(result),
                "problem": asdict(problem) if problem else {},
                "solution": asdict(plan) if plan else {},
            }
            r.xadd(MEMORIZE_QUEUE, {"data": json.dumps(payload, default=str)})
            logger.info("[verifier] Result published to %s (outcome=%s)", MEMORIZE_QUEUE, result.outcome)
        except redis.RedisError as exc:
            logger.error("[verifier] Failed to publish result: %s", exc)
            self.r = None

    def _dispatch_to_trainer(self, plan: SolutionPlan, problem: ProblemPacket) -> None:
        """Build a TrainingJob from the plan spec and push to TRAIN_QUEUE."""
        r = self._ensure_connected()
        spec = plan.modification_spec

        job = TrainingJob(
            problem_id=problem.id,
            solution_id=plan.id,
            training_domain=spec.get("training_domain", problem.domain),
            n_pairs=int(spec.get("n_pairs", 100)),
            lora_rank=int(spec.get("lora_rank", 16)),
            lora_alpha=int(spec.get("lora_alpha", 32)),
            epochs=int(spec.get("epochs", 2)),
            learning_rate=float(spec.get("learning_rate", 2e-4)),
            target_modules=spec.get("target_modules", ["q_proj", "v_proj"]),
            batch_size=int(spec.get("batch_size", 4)),
            description=spec.get("description", problem.description),
            success_criterion=spec.get("success_criterion", problem.success_criterion),
        )

        payload = {
            "job": asdict(job),
            "problem": asdict(problem),
            "plan": asdict(plan),
        }
        try:
            r.xadd(TRAIN_QUEUE, {"data": json.dumps(payload, default=str)})
            logger.info(
                "[verifier] Dispatched training job %s to %s (domain=%s, n_pairs=%d, rank=%d)",
                job.id, TRAIN_QUEUE, job.training_domain, job.n_pairs, job.lora_rank,
            )
        except redis.RedisError as exc:
            logger.error("[verifier] Failed to dispatch training job: %s", exc)
            self.r = None

    def _dispatch_weight_edit_job(
        self, problem: ProblemPacket, plan: SolutionPlan, edits: list
    ) -> None:
        """Convert ROME edit triples to a micro-finetune TrainingJob and push to TRAIN_QUEUE."""
        r = self._ensure_connected()

        # Build a description that tells the data generator exactly what facts to teach
        fact_lines = []
        for e in edits:
            subj = e.get("subject", "?")
            rel  = e.get("relation", "property")
            tgt  = e.get("target", "?")
            fact_lines.append(f"  - {subj}'s {rel} is: {tgt}")
        facts_block = "\n".join(fact_lines)
        description = (
            f"Weight-edit micro-finetune for {problem.domain}. "
            f"Teach ONLY these specific facts — every Q&A pair must directly reinforce one of them:\n"
            f"{facts_block}\n"
            f"Do NOT generate general domain questions. Target ONLY the listed facts."
        )

        # Small job: 20 pairs per edit, low rank, enough epochs to commit the facts
        n_pairs = max(20, len(edits) * 20)

        job = TrainingJob(
            problem_id=problem.id,
            solution_id=plan.id,
            training_domain=problem.domain,
            n_pairs=n_pairs,
            lora_rank=8,
            lora_alpha=16,
            epochs=3,
            learning_rate=5e-5,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            batch_size=4,
            description=description,
            success_criterion=problem.success_criterion,
        )

        payload = {
            "job":     asdict(job),
            "problem": asdict(problem),
            "plan":    asdict(plan),
        }
        try:
            r.xadd(TRAIN_QUEUE, {"data": json.dumps(payload, default=str)})
            logger.info(
                "[verifier] Dispatched weight-edit job %s to %s "
                "(domain=%s, edits=%d, n_pairs=%d, rank=8)",
                job.id, TRAIN_QUEUE, job.training_domain, len(edits), job.n_pairs,
            )
        except Exception as exc:
            logger.error("[verifier] Failed to dispatch weight-edit job: %s", exc)
            self.r = None

    def _requeue_problem(self, problem: ProblemPacket, plan: SolutionPlan, result: VerificationResult) -> None:
        """Send failed problem back to SOLVE_QUEUE with failure context."""
        current_attempt = getattr(plan, "attempt_count", 0)
        if current_attempt >= 5:
            logger.warning("[verifier] Max retries reached for problem %s — discarding", problem.id)
            return
        r = self._ensure_connected()
        retry_payload = {
            "problem": asdict(problem),
            "plan":    asdict(plan),
            "retry":   "1",
            "failure_mode": result.failure_mode,
            "attempt_count": str(current_attempt + 1),
        }
        try:
            r.xadd(SOLVE_QUEUE, {
                "data":   json.dumps(retry_payload, default=str),
                "retry":  "1",
                "from":   "verifier",
            })
            logger.info("[verifier] Requeued problem %s to %s (attempt=%d, failure_mode=%s)",
                        problem.id, SOLVE_QUEUE, current_attempt + 1, result.failure_mode[:80])
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

        # Claim any messages pending from dead consumers (e.g. after restart)
        # so we don't lose work that was delivered but never acknowledged.
        try:
            r = self._ensure_connected()
            consumer_name = f"verifier-{os.getpid()}"
            claimed = r.xautoclaim(
                SOLVE_QUEUE, VERIFIER_CONSUMER_GROUP, consumer_name,
                min_idle_time=5000, start_id="0-0", count=500
            )
            # xautoclaim returns (next_id, entries, deleted_ids)
            reclaimed = claimed[1] if isinstance(claimed, (list, tuple)) and len(claimed) > 1 else []
            if reclaimed:
                logger.info(
                    "[verifier] Reclaimed %d pending messages from dead consumers",
                    len(reclaimed)
                )
        except Exception as _claim_exc:
            logger.warning("[verifier] Autoclaim on startup failed (non-fatal): %s", _claim_exc)

        while True:
            try:
                item = self._read_next_plan()
                if item is None:
                    continue

                entry_id, plan, problem = item

                result = _process_one(plan, problem, self._ensure_connected())

                self._publish_result(result, problem, plan)

                # Increment throughput counter (read by dashboard as VERIFY depth)
                try:
                    self._ensure_connected().incr("VERIFY_COUNT")
                except Exception as _incr_exc:
                    logger.warning("[verifier] Failed to increment VERIFY_COUNT: %s", _incr_exc)

                if result.outcome == "fail":
                    if result.failure_mode == "lora_dispatched_to_trainer":
                        # Dispatch to TRAIN_QUEUE for async training
                        self._dispatch_to_trainer(plan, problem)
                    elif "not_implemented" not in result.failure_mode:
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
