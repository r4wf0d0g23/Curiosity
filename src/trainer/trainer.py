"""
Curiosity — Trainer Daemon (Daemon 6)

Reads training jobs from TRAIN_QUEUE (Redis stream), performs QLoRA
fine-tuning on the base Nemotron model, and hot-loads the resulting
adapter into vLLM.

Lifecycle per job:
  1. Read TrainingJob from TRAIN_QUEUE
  2. Generate domain-specific Q&A training pairs via vLLM
  3. Stop vLLM inference server (free GPU memory)
  4. Run QLoRA training with peft + bitsandbytes
  5. Restart vLLM with --enable-lora
  6. Load adapter via vLLM LoRA API
  7. Push verification request to VERIFY_QUEUE

Resource constraints:
  - DGX GB10: 120GB unified GPU memory
  - Base model FP4: ~60-80GB
  - QLoRA training: ~20-40GB additional
  - Strategy: stop vLLM → train → restart vLLM with LoRA support

Never exits. All exceptions caught, logged, and the loop continues.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import signal
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
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path(os.environ.get("CURIOSITY_LOG_DIR", Path.home() / "curiosity" / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "trainer.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("curiosity.trainer")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.shared.types import ProblemPacket, SolutionPlan, TrainingJob, VerificationResult  # noqa: E402

# ---------------------------------------------------------------------------
# Config (all override-able via env)
# ---------------------------------------------------------------------------

REDIS_HOST          = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT          = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB            = int(os.environ.get("REDIS_DB", 0))

TRAIN_QUEUE         = os.environ.get("TRAIN_QUEUE", "TRAIN_QUEUE")
VERIFY_QUEUE        = os.environ.get("VERIFY_QUEUE", "VERIFY_QUEUE")
SOLVE_QUEUE         = os.environ.get("SOLVE_QUEUE", "SOLVE_QUEUE")

VLLM_BASE_URL       = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")
VLLM_MODEL          = os.environ.get("VLLM_MODEL", "nemotron3-super")
BASE_MODEL_PATH     = os.environ.get(
    "BASE_MODEL_PATH",
    "/home/rawdata/.cache/huggingface/hub/models--nvidia--Nemotron-3-Super-120B-A12B-NVFP4",
)
ADAPTER_OUTPUT_DIR  = Path(os.environ.get(
    "ADAPTER_OUTPUT_DIR",
    Path.home() / "curiosity" / "adapters",
))
VLLM_START_CMD      = os.environ.get("VLLM_START_CMD", (
    "python3 -m vllm.entrypoints.openai.api_server "
    "--model nvidia/Nemotron-3-Super-120B-A12B-NVFP4 "
    "--served-model-name nemotron3-super "
    "--host 0.0.0.0 --port 8001 "
    "--enable-chunked-prefill --max-num-seqs 4 "

))

REDIS_BLOCK_MS      = int(os.environ.get("REDIS_BLOCK_MS", 5000))
TRAINER_CONSUMER_GROUP = "trainers"
REDIS_RETRY_BACKOFF = float(os.environ.get("REDIS_RETRY_BACKOFF", 5.0))

# Training lock key — other daemons check this to defer work during training
TRAINING_LOCK_KEY   = "CURIOSITY_TRAINING_LOCK"
TRAINING_LOCK_TTL   = int(os.environ.get("TRAINING_LOCK_TTL", 7200))  # 2h max

# Active adapters tracking
ACTIVE_ADAPTERS_KEY = "CURIOSITY_ACTIVE_ADAPTERS"


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def _make_redis_client() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


def _parse_stream_entry(entry_data: Dict[str, str]) -> Dict[str, Any]:
    raw = entry_data.get("data", "{}")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# vLLM lifecycle management
# ---------------------------------------------------------------------------



TRAINING_PROGRESS_KEY = "CURIOSITY_TRAINING_PROGRESS"


def _publish_progress(r_conn, **fields) -> None:
    """Write training progress fields to Redis hash for dashboard."""
    try:
        if r_conn is None:
            return
        fields["ts"] = int(time.time())
        r_conn.hset(TRAINING_PROGRESS_KEY, mapping={k: str(v) for k, v in fields.items()})
        r_conn.expire(TRAINING_PROGRESS_KEY, 7200)  # auto-expire with the lock
    except Exception:
        pass  # non-fatal


def _is_docker_vllm_running() -> bool:
    import subprocess as _sp
    try:
        r = _sp.run(["docker","inspect","--format","{{.State.Running}}","vllm_nemotron"],
                    capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "true"
    except Exception:
        return False

VLLM_DOCKER_RUN_CMD = (
    "docker run -d --name vllm_nemotron --gpus all --ipc host "
    "-v /home/rawdata/.cache:/root/.cache "
    "-v /home/rawdata/curiosity/adapters:/adapters "
    "-p 8001:8001 vllm-node:lora-patched "
    "python3 -m vllm.entrypoints.openai.api_server "
    "--model /root/.cache/huggingface/hub/models--nvidia--Nemotron-3-Super-120B-A12B-NVFP4 "
    "--served-model-name nemotron3-super "
    "--host 0.0.0.0 --port 8001 "
    "--dtype auto --kv-cache-dtype fp8 --quantization fp4 "
    "--gpu-memory-utilization 0.72 --max-model-len 32768 --max-num-seqs 4 "
    "--enable-chunked-prefill --trust-remote-code "
    "--enable-lora --max-lora-rank 64"
)

def _find_vllm_pid() -> Optional[int]:
    """Find the PID of the running vLLM server process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Take the first PID (parent process)
            return int(result.stdout.strip().splitlines()[0])
    except Exception as exc:
        logger.warning("Could not find vLLM PID: %s", exc)
    return None


def _wait_for_vllm(timeout: int = 300) -> bool:
    """Wait for vLLM to become healthy (respond to /v1/models)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=5)
            if resp.status_code == 200:
                logger.info("vLLM is healthy")
                return True
        except Exception:
            pass
        time.sleep(5)
    logger.error("vLLM did not become healthy within %ds", timeout)
    return False


def _stop_vllm() -> Optional[int]:
    """
    Gracefully stop vLLM. Returns the old PID if found, None otherwise.
    Sends SIGTERM, waits up to 30s, then SIGKILL if needed.
    """
    # Docker container takes priority
    if _is_docker_vllm_running():
        logger.info("Stopping Docker vLLM container")
        try:
            import subprocess as _sp2
            _sp2.run(["docker","stop","vllm_nemotron"], timeout=60, check=True)
            _sp2.run(["docker","rm","vllm_nemotron"], timeout=30, check=True)
            logger.info("Docker vLLM stopped and removed")
            return 1
        except Exception as exc:
            logger.error("docker stop failed: %s", exc)
            return None

    pid = _find_vllm_pid()
    if pid is None:
        logger.warning("No vLLM process or container found to stop")
        return None

    logger.info("Stopping vLLM (pid=%d) with SIGTERM", pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        logger.info("vLLM process already gone")
        return pid

    # Wait for graceful shutdown
    for _ in range(60):  # 30 seconds at 0.5s intervals
        try:
            os.kill(pid, 0)  # Check if still alive
            time.sleep(0.5)
        except ProcessLookupError:
            logger.info("vLLM stopped cleanly (pid=%d)", pid)
            return pid

    # Force kill
    logger.warning("vLLM did not stop gracefully — sending SIGKILL")
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)
    except ProcessLookupError:
        pass
    return pid


def _start_vllm() -> bool:
    """Start vLLM with LoRA support enabled. Returns True if healthy."""
    logger.info("Starting Docker vLLM with LoRA support")
    import subprocess as _sp3
    try:
        r = _sp3.run(VLLM_DOCKER_RUN_CMD.split(), capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            logger.error("docker run failed: %s", r.stderr)
            return False
        logger.info("Container started: %s", r.stdout.strip()[:12])
    except Exception as exc:
        logger.error("Failed to start Docker vLLM: %s", exc)
        return False

    return _wait_for_vllm(timeout=300)


def _load_lora_adapter(adapter_name: str, adapter_path: str) -> bool:
    """Load a LoRA adapter into running vLLM via its API."""
    try:
        resp = requests.post(
            f"{VLLM_BASE_URL}/v1/load_lora_adapter",
            json={"lora_name": adapter_name, "lora_path": adapter_path},
            timeout=60,
        )
        if resp.status_code == 200:
            logger.info("LoRA adapter loaded: %s → %s", adapter_name, adapter_path)
            return True
        else:
            logger.error("Failed to load LoRA adapter: %d %s", resp.status_code, resp.text)
            return False
    except Exception as exc:
        logger.error("Error loading LoRA adapter: %s", exc)
        return False


def _unload_lora_adapter(adapter_name: str) -> bool:
    """Unload a LoRA adapter from running vLLM."""
    try:
        resp = requests.post(
            f"{VLLM_BASE_URL}/v1/unload_lora_adapter",
            json={"lora_name": adapter_name},
            timeout=30,
        )
        if resp.status_code == 200:
            logger.info("LoRA adapter unloaded: %s", adapter_name)
            return True
        else:
            logger.error("Failed to unload LoRA adapter: %d %s", resp.status_code, resp.text)
            return False
    except Exception as exc:
        logger.error("Error unloading LoRA adapter: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def _generate_training_pairs(
    domain: str,
    description: str,
    criterion: str,
    n_pairs: int,
) -> List[Dict[str, str]]:
    """
    Use the vLLM server to self-generate high-quality Q&A pairs
    for the target domain. Generates in batches to avoid token limits.
    """
    pairs: List[Dict[str, str]] = []
    batch_size = 10  # generate 10 pairs per LLM call

    generation_prompt = f"""\
You are generating high-quality training data for an AI model.
Domain: {domain}
Description: {description}
Quality criterion: {criterion}

Generate exactly {batch_size} diverse question-answer pairs that would help
a model improve in this domain. Each pair should test a different aspect.

Output ONLY a JSON array of objects with "question" and "answer" keys:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Requirements:
- Answers must be factually correct and detailed
- Questions should vary in difficulty (easy, medium, hard)
- Cover different sub-topics within the domain
- Answers should demonstrate expert-level reasoning"""

    for batch_idx in range(0, n_pairs, batch_size):
        remaining = min(batch_size, n_pairs - len(pairs))
        if remaining <= 0:
            break

        try:
            resp = requests.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json={
                    "model": VLLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are an expert training data generator. Output only valid JSON."},
                        {"role": "user", "content": generation_prompt},
                    ],
                    "temperature": 0.8,  # diverse but not random
                    "max_tokens": 4096,
                },
                timeout=360,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # Parse JSON array from response
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                batch_pairs = json.loads(content[start:end])
                for pair in batch_pairs:
                    if "question" in pair and "answer" in pair:
                        pairs.append({
                            "question": str(pair["question"]),
                            "answer": str(pair["answer"]),
                        })
                logger.info(
                    "Generated %d pairs (batch %d, total %d/%d)",
                    len(batch_pairs), batch_idx // batch_size, len(pairs), n_pairs,
                )
                # Publish live progress (progress_r injected via closure if available)
                if hasattr(_generate_training_pairs, "_progress_r"):
                    _publish_progress(
                        _generate_training_pairs._progress_r,
                        step="generating_pairs",
                        pairs_done=len(pairs),
                        pairs_total=n_pairs,
                        batch=batch_idx // batch_size,
                    )
            else:
                logger.warning("Could not parse JSON array from batch %d", batch_idx // batch_size)
        except Exception as exc:
            logger.error("Training data generation error (batch %d): %s", batch_idx // batch_size, exc)
            continue

    logger.info("Training data generation complete: %d/%d pairs", len(pairs), n_pairs)
    return pairs


# ---------------------------------------------------------------------------
# QLoRA Training
# ---------------------------------------------------------------------------

def _run_qlora_training(
    job: TrainingJob,
    training_pairs: List[Dict[str, str]],
) -> str:
    """
    Run QLoRA fine-tuning using transformers + peft + bitsandbytes.
    Returns the path to the saved adapter directory.

    This function loads the base model in 4-bit quantization, applies
    a LoRA adapter, trains on the generated pairs, and saves the adapter.
    """
    # Lazy imports — these are heavy and only needed during training
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from transformers import BitsAndBytesConfig
    from torch.utils.data import Dataset

    class QAPairDataset(Dataset):
        """Simple dataset wrapping Q&A pairs for causal LM training."""

        def __init__(self, pairs: List[Dict[str, str]], tokenizer, max_length: int = 512):
            self.encodings = []
            for pair in pairs:
                text = (
                    f"### Question:\n{pair['question']}\n\n"
                    f"### Answer:\n{pair['answer']}"
                )
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                # For causal LM, labels = input_ids
                encoded["labels"] = encoded["input_ids"].clone()
                self.encodings.append({k: v.squeeze(0) for k, v in encoded.items()})

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            return self.encodings[idx]

    adapter_dir = ADAPTER_OUTPUT_DIR / job.id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting QLoRA training: model=%s rank=%d alpha=%d epochs=%d pairs=%d",
        BASE_MODEL_PATH, job.lora_rank, job.lora_alpha, job.epochs, len(training_pairs),
    )

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit
    logger.info("Loading base model in 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=job.lora_rank,
        lora_alpha=job.lora_alpha,
        target_modules=job.target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info("Applying LoRA adapter: rank=%d, modules=%s", job.lora_rank, job.target_modules)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    dataset = QAPairDataset(training_pairs, tokenizer)
    logger.info("Training dataset: %d samples", len(dataset))

    # Training arguments
    output_dir = str(adapter_dir / "checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=job.epochs,
        per_device_train_batch_size=job.batch_size,
        gradient_accumulation_steps=max(1, 16 // job.batch_size),
        learning_rate=job.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(
        "Training complete: loss=%.4f, runtime=%.1fs",
        train_result.training_loss,
        train_result.metrics.get("train_runtime", 0),
    )

    # Save adapter (not full model)
    logger.info("Saving adapter to %s", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Clean up GPU memory
    del model
    del trainer
    del dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Adapter saved and GPU memory freed")
    return str(adapter_dir)


# ---------------------------------------------------------------------------
# TrainerDaemon
# ---------------------------------------------------------------------------

class TrainerDaemon:
    """
    Infinite-loop daemon that reads training jobs from TRAIN_QUEUE,
    runs QLoRA fine-tuning, and dispatches verification requests.
    """

    def __init__(self) -> None:
        self.r: Optional[redis.Redis] = None

    # ------------------------------------------------------------------
    # Redis connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        while True:
            try:
                client = _make_redis_client()
                client.ping()
                self.r = client
                logger.info("Connected to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
                return
            except redis.RedisError as exc:
                logger.error("Redis connect failed: %s — retrying in %.1fs", exc, REDIS_RETRY_BACKOFF)
                time.sleep(REDIS_RETRY_BACKOFF)

    def _ensure_connected(self) -> redis.Redis:
        if self.r is None:
            self._connect()
        return self.r  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Training lock — other daemons check this to defer vLLM calls
    # ------------------------------------------------------------------

    def _acquire_training_lock(self) -> bool:
        r = self._ensure_connected()
        try:
            acquired = r.set(TRAINING_LOCK_KEY, "1", ex=TRAINING_LOCK_TTL, nx=True)
            if acquired:
                logger.info("Training lock acquired (TTL=%ds)", TRAINING_LOCK_TTL)
            return bool(acquired)
        except redis.RedisError as exc:
            logger.error("Failed to acquire training lock: %s", exc)
            return False

    def _release_training_lock(self) -> None:
        r = self._ensure_connected()
        try:
            r.delete(TRAINING_LOCK_KEY)
            logger.info("Training lock released")
        except redis.RedisError as exc:
            logger.error("Failed to release training lock: %s", exc)

    # ------------------------------------------------------------------
    # Read from TRAIN_QUEUE
    # ------------------------------------------------------------------

    def _read_next_job(self) -> Optional[Tuple[str, TrainingJob, ProblemPacket, SolutionPlan]]:
        r = self._ensure_connected()
        try:
            consumer_name = f"trainer-{os.getpid()}"
            try:
                r.xgroup_create(TRAIN_QUEUE, TRAINER_CONSUMER_GROUP, id="0", mkstream=True)
            except Exception:
                pass

            # Drain PEL first (reclaimed messages from dead consumers)
            results = r.xreadgroup(
                TRAINER_CONSUMER_GROUP, consumer_name,
                {TRAIN_QUEUE: "0-0"},
                count=1,
            )
            if not results or not results[0][1]:
                # PEL empty — fetch new messages
                results = r.xreadgroup(
                    TRAINER_CONSUMER_GROUP, consumer_name,
                    {TRAIN_QUEUE: ">"},
                    block=REDIS_BLOCK_MS, count=1,
                )
        except redis.RedisError as exc:
            logger.error("Redis read error: %s", exc)
            self.r = None
            return None

        if not results:
            return None

        _stream_name, entries = results[0]
        entry_id, entry_data = entries[0]

        try:
            payload = _parse_stream_entry(entry_data)
            job_data = payload.get("job", payload)
            problem_data = payload.get("problem", {})
            plan_data = payload.get("plan", {})

            # Build TrainingJob from payload
            valid_job_fields = {f.name for f in dataclasses.fields(TrainingJob)}
            job = TrainingJob(**{k: v for k, v in job_data.items() if k in valid_job_fields})

            valid_problem_fields = {f.name for f in dataclasses.fields(ProblemPacket)}
            problem = ProblemPacket(**{k: v for k, v in problem_data.items() if k in valid_problem_fields})

            valid_plan_fields = {f.name for f in dataclasses.fields(SolutionPlan)}
            plan = SolutionPlan(**{k: v for k, v in plan_data.items() if k in valid_plan_fields})

            logger.info(
                "Received training job: id=%s domain=%s n_pairs=%d rank=%d",
                job.id, job.training_domain, job.n_pairs, job.lora_rank,
            )
            return entry_id, job, problem, plan

        except Exception as exc:
            logger.error("Failed to deserialize training job %s: %s", entry_id, exc)
            return None

    # ------------------------------------------------------------------
    # Publish to VERIFY_QUEUE
    # ------------------------------------------------------------------

    def _publish_to_verify(
        self, job: TrainingJob, problem: ProblemPacket, plan: SolutionPlan,
    ) -> None:
        r = self._ensure_connected()
        # Update the plan with adapter info so verifier knows how to test
        plan.modification_spec["adapter_name"] = job.adapter_name
        plan.modification_spec["adapter_path"] = job.adapter_path
        plan.modification_spec["adapter_id"] = job.id

        payload = {
            "plan": asdict(plan),
            "problem": asdict(problem),
            "training_job": asdict(job),
        }
        try:
            r.xadd(SOLVE_QUEUE, {"data": json.dumps(payload, default=str)})
            logger.info("Verification request published for adapter %s", job.adapter_name)
        except redis.RedisError as exc:
            logger.error("Failed to publish verification request: %s", exc)
            self.r = None

    # ------------------------------------------------------------------
    # Track active adapters
    # ------------------------------------------------------------------

    def _register_adapter(self, job: TrainingJob) -> None:
        r = self._ensure_connected()
        try:
            adapter_info = json.dumps({
                "name": job.adapter_name,
                "path": job.adapter_path,
                "domain": job.training_domain,
                "job_id": job.id,
                "loaded_at": datetime.utcnow().isoformat(),
            })
            r.hset(ACTIVE_ADAPTERS_KEY, job.adapter_name, adapter_info)
        except redis.RedisError as exc:
            logger.error("Failed to register adapter: %s", exc)

    # ------------------------------------------------------------------
    # Process one training job
    # ------------------------------------------------------------------

    def _process_job(
        self, job: TrainingJob, problem: ProblemPacket, plan: SolutionPlan,
    ) -> TrainingJob:
        """Execute the full training pipeline for one job."""
        start_time = time.time()
        job.adapter_name = f"lora_{job.training_domain}_{job.id[:8]}"

        # Step 0: Acquire training lock FIRST so solvers/verifiers back off vLLM
        logger.info("Step 0: Acquiring training lock before pair generation")
        if not self._acquire_training_lock():
            logger.error("Could not acquire training lock — aborting")
            job.status = "failed"
            job.error = "could_not_acquire_training_lock_pre"
            return job

        # Step 1: Generate training data (lock held — solvers/verifiers yield)
        logger.info("Step 1/5: Generating %d training pairs for domain '%s'", job.n_pairs, job.training_domain)
        # Wait for any in-flight vLLM requests to drain before starting pair gen
        _drain_start = time.time()
        while time.time() - _drain_start < 60:
            try:
                _mresp = requests.get(f"{VLLM_BASE_URL}/metrics", timeout=3)
                _running = sum(float(l.split()[1]) for l in _mresp.text.splitlines()
                              if l.startswith("vllm:num_requests_running") and not l.startswith("#"))
                if _running == 0:
                    break
                logger.info("[trainer] Waiting for vLLM drain: %d requests in-flight", int(_running))
            except Exception:
                break
            time.sleep(3)
        logger.info("[trainer] vLLM queue drained — starting pair generation")
        job.status = "generating_data"
        try:
            # Inject redis conn for live progress publishing
            _generate_training_pairs._progress_r = self._ensure_connected()
            _publish_progress(
                self._ensure_connected(),
                step="generating_pairs",
                pairs_done=0,
                pairs_total=job.n_pairs,
                domain=job.training_domain,
                job_id=job.id,
                batch=0,
                errors=0,
            )
            training_pairs = _generate_training_pairs(
                domain=job.training_domain,
                description=job.description or problem.description,
                criterion=job.success_criterion or problem.success_criterion,
                n_pairs=job.n_pairs,
            )
            if len(training_pairs) < 10:
                raise ValueError(f"Only generated {len(training_pairs)} pairs — need at least 10")
        except Exception as exc:
            job.status = "failed"
            job.error = f"data_generation_failed: {exc}"
            job.duration_sec = time.time() - start_time
            logger.error("Training data generation failed: %s", exc)
            return job

        # Save training data for reproducibility
        data_path = ADAPTER_OUTPUT_DIR / job.id / "training_data.json"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps(training_pairs, indent=2))
        logger.info("Training data saved to %s", data_path)

        # Step 2: Acquire training lock and stop vLLM
        logger.info("Step 2/5: Stopping vLLM (lock already held)")
        _publish_progress(self._ensure_connected(), step="stopping_vllm", domain=job.training_domain, job_id=job.id, pairs_done=len(training_pairs), pairs_total=job.n_pairs)

        try:
            old_pid = _stop_vllm()
            # Give GPU memory time to be fully released
            time.sleep(5)

            # Step 3: Run QLoRA training
            logger.info("Step 3/5: Running QLoRA training")
            job.status = "training"
            _publish_progress(self._ensure_connected(), step="qlora_training", domain=job.training_domain, job_id=job.id, pairs_done=len(training_pairs), pairs_total=job.n_pairs)
            adapter_path = _run_qlora_training(job, training_pairs)
            job.adapter_path = adapter_path

            # Step 4: Restart vLLM with LoRA support
            logger.info("Step 4/5: Restarting vLLM with LoRA support")
            job.status = "loading"
            _publish_progress(self._ensure_connected(), step="restarting_vllm", domain=job.training_domain, job_id=job.id)
            if not _start_vllm():
                raise RuntimeError("vLLM failed to restart after training")

            # Step 5: Load the adapter
            logger.info("Step 5/5: Loading LoRA adapter into vLLM")
            _publish_progress(self._ensure_connected(), step="loading_adapter", domain=job.training_domain, job_id=job.id)
            if not _load_lora_adapter(job.adapter_name, adapter_path):
                raise RuntimeError(f"Failed to load adapter {job.adapter_name}")

            # Also reload any previously active adapters
            self._reload_active_adapters()

            job.status = "done"
            job.duration_sec = time.time() - start_time
            logger.info(
                "Training job complete: id=%s adapter=%s duration=%.1fs",
                job.id, job.adapter_name, job.duration_sec,
            )

        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.duration_sec = time.time() - start_time
            logger.error("Training pipeline failed: %s\n%s", exc, traceback.format_exc())

            # Try to restart vLLM even on failure
            if not _wait_for_vllm(timeout=10):
                logger.warning("vLLM not running after failure — attempting restart")
                _start_vllm()

        finally:
            self._release_training_lock()

        return job

    def _reload_active_adapters(self) -> None:
        """Reload all previously active adapters after vLLM restart."""
        r = self._ensure_connected()
        try:
            adapters = r.hgetall(ACTIVE_ADAPTERS_KEY)
            for name, info_str in adapters.items():
                try:
                    info = json.loads(info_str)
                    path = info.get("path", "")
                    if path and Path(path).exists():
                        _load_lora_adapter(name, path)
                    else:
                        logger.warning("Adapter %s path no longer exists: %s — removing", name, path)
                        r.hdel(ACTIVE_ADAPTERS_KEY, name)
                except Exception as exc:
                    logger.error("Failed to reload adapter %s: %s", name, exc)
        except redis.RedisError as exc:
            logger.error("Failed to read active adapters: %s", exc)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        # Singleton guard — atomic SETNX
        _sr = self._ensure_connected()
        _singleton_key = "CURIOSITY_TRAINER_PID"
        if _sr:
            # Atomically claim the singleton slot
            claimed = _sr.set(_singleton_key, str(os.getpid()), nx=True, ex=7200)
            if not claimed:
                existing = _sr.get(_singleton_key)
                try:
                    existing_pid = int(existing) if existing else 0
                    os.kill(existing_pid, 0)  # will raise if dead
                    logger.warning("[singleton] Trainer %d already running — exiting", existing_pid)
                    return
                except (OSError, ProcessLookupError, ValueError):
                    # Dead process — forcibly take over
                    _sr.set(_singleton_key, str(os.getpid()), ex=7200)
        logger.info("=" * 70)
        logger.info("Trainer daemon starting (pid=%d)", os.getpid())
        logger.info("TRAIN_QUEUE=%s  VERIFY_QUEUE=%s", TRAIN_QUEUE, VERIFY_QUEUE)
        logger.info("Base model: %s", BASE_MODEL_PATH)
        logger.info("Adapter output: %s", ADAPTER_OUTPUT_DIR)
        logger.info("=" * 70)

        # Startup: reclaim any pending jobs from dead consumers
        try:
            _r = self._ensure_connected()
            _cn = f"trainer-{os.getpid()}"
            claimed = _r.xautoclaim(TRAIN_QUEUE, TRAINER_CONSUMER_GROUP, _cn, min_idle_time=0, start_id="0-0", count=50)
            _rc = claimed[1] if isinstance(claimed, (list, tuple)) and len(claimed) > 1 else []
            if _rc:
                logger.info("[trainer] Reclaimed %d pending jobs on startup", len(_rc))
        except Exception as _e:
            logger.warning("[trainer] Startup autoclaim failed: %s", _e)

        while True:
            try:
                item = self._read_next_job()
                if item is None:
                    continue

                entry_id, job, problem, plan = item

                # Increment training counter
                try:
                    self._ensure_connected().incr("TRAIN_COUNT")
                except Exception:
                    pass

                # Execute training pipeline
                job = self._process_job(job, problem, plan)

                # ACK only on terminal outcomes (done or unrecoverable failure)
                # Lock failures are NOT terminal — job stays in PEL for retry
                _is_terminal = (
                    job.status == "done"
                    or (job.status == "failed" and job.error
                        and "could_not_acquire_training_lock" not in job.error)
                )
                if _is_terminal:
                    try:
                        r = self._ensure_connected()
                        r.xack(TRAIN_QUEUE, TRAINER_CONSUMER_GROUP, entry_id)
                        logger.info("[trainer] ACK entry %s", entry_id)
                    except Exception as _ack_exc:
                        logger.warning("[trainer] XACK failed: %s", _ack_exc)
                else:
                    logger.info("[trainer] Not ACKing %s (lock retry) — stays in PEL", entry_id)

                if job.status == "done":
                    # Register adapter and send to verifier
                    self._register_adapter(job)
                    self._publish_to_verify(job, problem, plan)
                else:
                    # Log failure — don't requeue training failures (expensive)
                    logger.error(
                        "Training job %s failed: %s (duration=%.1fs)",
                        job.id, job.error, job.duration_sec,
                    )
                    # Write failure to MEMORIZE_QUEUE for learning
                    self._publish_failure(job, problem, plan)

            except Exception as exc:
                logger.critical(
                    "Unhandled exception in trainer main loop: %s\n%s",
                    exc, traceback.format_exc(),
                )
                time.sleep(5)

    def _publish_failure(
        self, job: TrainingJob, problem: ProblemPacket, plan: SolutionPlan,
    ) -> None:
        """Write training failure to MEMORIZE_QUEUE so the system can learn from it."""
        r = self._ensure_connected()
        result = VerificationResult(
            solution_id=plan.id,
            problem_id=problem.id,
            outcome="fail",
            criterion_score=0.0,
            failure_mode=f"training_failed: {job.error}",
        )
        payload = {
            "result": asdict(result),
            "problem": asdict(problem),
            "solution": asdict(plan),
            "training_job": asdict(job),
        }
        try:
            memorize_queue = os.environ.get("MEMORIZE_QUEUE", "MEMORIZE_QUEUE")
            r.xadd(memorize_queue, {"data": json.dumps(payload, default=str)})
        except redis.RedisError as exc:
            logger.error("Failed to publish training failure: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    daemon = TrainerDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
