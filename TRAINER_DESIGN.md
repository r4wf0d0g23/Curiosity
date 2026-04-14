# Curiosity Trainer — Architecture Design

## Overview

The Trainer daemon (Daemon 6) adds **real weight modification** to Curiosity's
self-improvement pipeline. Instead of only modifying the system prompt
(`prompt_patch`), the system can now perform QLoRA fine-tuning on the base model
to fix systematic capability gaps.

## Architecture

### Data Flow

```
Assessor → Formulator → Solver → Verifier → Trainer → Verifier (A/B) → Memorizer
                                     │                      │
                                     │  TRAIN_QUEUE         │  SOLVE_QUEUE
                                     └──────────────────────┘
```

1. **Solver** detects systematic failures (failure_rate > 0.3, multiple
   prompt_patch failures) and proposes a `lora_finetune` approach
2. **Verifier** receives the plan, recognizes it needs training, and dispatches
   a `TrainingJob` to `TRAIN_QUEUE`
3. **Trainer** picks up the job, generates training data, runs QLoRA, and
   loads the adapter into vLLM
4. **Verifier** receives the trained plan back (via `SOLVE_QUEUE`) and runs
   A/B benchmarks: with-adapter vs baseline. Pass requires >5% improvement.
5. On PASS: adapter stays loaded permanently. On FAIL: adapter is unloaded
   and files are deleted.

### Approach Selection Heuristic

The `SolutionGenerator` uses this logic to decide when to escalate from
prompt_patch to lora_finetune:

```python
def _should_propose_finetune(problem, failures):
    if problem.failure_rate < 0.3:
        return False
    prompt_patch_failures = sum(1 for f in failures if f["approach"] == "prompt_patch")
    return prompt_patch_failures >= 2
```

This means the system:
- Always tries prompt_patch first (cheap, fast, reversible)
- Escalates to fine-tuning only after 2+ prompt_patch failures on a domain
  with >30% failure rate
- The model can also independently choose lora_finetune via its JSON response

## Training Pipeline

### Step 1: Training Data Generation

While vLLM is still running, the trainer uses the model to self-generate
high-quality Q&A pairs for the failing domain. This is bootstrapped learning:
the model generates its own curriculum.

- Batch size: 10 pairs per LLM call
- Temperature: 0.8 (diverse but grounded)
- Validation: only pairs with both "question" and "answer" keys are kept
- Minimum: 10 pairs required, otherwise job fails

### Step 2: vLLM Stop

Training and inference cannot share GPU memory on the DGX GB10 (120GB unified):
- Base model FP4: ~60-80GB
- QLoRA training overhead: ~20-40GB additional

The trainer acquires a Redis-based `TRAINING_LOCK` (TTL: 2h), then gracefully
stops vLLM via SIGTERM (30s grace → SIGKILL).

### Step 3: QLoRA Training

Using HuggingFace `transformers` + `peft` + `bitsandbytes`:

```
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=bfloat16)
LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
           task_type=CAUSAL_LM)
```

- Optimizer: `paged_adamw_8bit` (memory-efficient)
- Gradient accumulation: auto-scaled to effective batch size 16
- LR schedule: cosine with 5% warmup
- bf16 training throughout

### Step 4: vLLM Restart with LoRA

vLLM is restarted with `--enable-lora --max-lora-rank 64`. The trainer waits
up to 5 minutes for the health check to pass, then loads:
- The new adapter
- All previously active adapters (from `CURIOSITY_ACTIVE_ADAPTERS` Redis hash)

### Step 5: Verification

The trained adapter is pushed back through the verify pipeline. The verifier
runs an A/B comparison:
- WITH adapter: `lora_request: {"lora_name": "...", "lora_int_id": 1, "lora_path": "..."}`
- WITHOUT adapter: standard inference

Pass threshold: adapter must improve benchmark score by >5% over baseline.

## Resource Coordination

### Training Lock

A Redis key `CURIOSITY_TRAINING_LOCK` with NX semantics and 2h TTL prevents:
- Multiple concurrent training jobs
- Other daemons from making vLLM calls during training

The lock is always released in a `finally` block, even on training failure.

### Adapter Lifecycle

| State | Location | vLLM | Files |
|-------|----------|------|-------|
| Training | TRAIN_QUEUE → Trainer | Stopped | Being written |
| Pending verification | SOLVE_QUEUE → Verifier | Running + loaded | `~/curiosity/adapters/{job_id}/` |
| Verified (PASS) | ACTIVE_ADAPTERS hash | Permanently loaded | Kept |
| Verified (FAIL) | Unloaded | Unloaded | Deleted |

### Failure Recovery

- If training fails: vLLM is restarted (if not already running), lock released
- If vLLM fails to restart: logged as critical, training lock released
- If adapter load fails: job marked failed, sent to MEMORIZE_QUEUE for learning
- Training failures are NOT requeued (too expensive) — they go to memorizer
  so the system can learn what training configurations work

## Configuration

All trainer config lives in `config/curiosity.yaml` under the `trainer:` key:

```yaml
trainer:
  base_model_path: nvidia/Nemotron-3-Super-120B-A12B-NVFP4
  adapter_output_dir: ~/curiosity/adapters
  vllm_base_url: http://localhost:8001
  training_lock_ttl: 7200
  default_lora_rank: 16
  default_n_pairs: 100
  default_epochs: 2
  default_target_modules: ["q_proj", "v_proj"]
  finetune_threshold:
    min_failure_rate: 0.3
    min_prompt_patch_failures: 2
```

## Dashboard Integration

The dashboard now shows:
- `TRAIN` queue depth alongside other pipeline queues
- `TRAIN_COUNT` cumulative counter
- `training_active` boolean (lock held)
- `active_adapters` list with name, domain, path, and load timestamp
- Trainer daemon status in the daemon health panel

## Future Work

- **weight_edit approach**: ROME/MEMIT integration for surgical factual corrections
- **Concurrent training**: Use model sharding to train while serving (requires >120GB)
- **Adapter merging**: Merge multiple successful LoRA adapters into one
- **Curriculum learning**: Use memorizer outcomes to improve training data generation
- **Adapter versioning**: Track adapter lineage and allow rollback to prior versions
