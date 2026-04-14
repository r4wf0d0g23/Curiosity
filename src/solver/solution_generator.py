"""
Curiosity — Solver: Solution Generator
Generates novel SolutionPlans via vLLM Server (port 8001) when Memory
has no high-confidence match.

Approach selection:
  - prompt_patch: modify system prompt (low cost, fast iteration)
  - lora_finetune: QLoRA fine-tuning for systematic domain failures
  - weight_edit: ROME/MEMIT surgical edits for specific factual errors
  - composite: multi-step approach combining the above
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

import re

import requests

from src.shared.types import ProblemPacket, SolutionPlan

logger = logging.getLogger("solver.solution_generator")

# ── vLLM Server ───────────────────────────────────────────────────────────────
VLLM_BASE_URL     = "http://localhost:8001"
VLLM_MODEL        = "nemotron3-super"
CHAT_COMPLETIONS  = f"{VLLM_BASE_URL}/v1/chat/completions"
REQUEST_TIMEOUT   = None  # no cap — deep reasoning must never be snipped  # seconds

# Valid approach literals from SolutionPlan
VALID_APPROACHES  = {"weight_edit", "lora_finetune", "prompt_patch", "composite"}


SYSTEM_PROMPT = """\
You are a solution architect for an AI recursive self-improvement system.
Your role is to propose concrete, targeted modifications to improve the model's capabilities.

Approach selection guide:
- prompt_patch: Quick system prompt modifications. Best for framing/instruction issues.
- lora_finetune: QLoRA fine-tuning on generated Q&A pairs. Best for systematic domain
  failures (failure_rate > 0.3) where multiple prompt_patch attempts have failed.
  Requires stopping vLLM, training, and restarting — use only when prompt_patch is insufficient.
- weight_edit: Surgical ROME/MEMIT weight edits. Best for specific factual errors where
  the model consistently gives wrong answers to known questions.

Choose the approach that best matches the problem characteristics.
Always output valid JSON and nothing else."""


def _build_user_prompt(
    problem: ProblemPacket, failures: list[dict]
) -> str:
    failures_text = "None known yet."
    if failures:
        failure_lines = []
        for f in failures[:10]:  # cap at 10 to avoid token overflow
            failure_lines.append(
                f"  - approach={f.get('approach', '?')}  "
                f"failure_mode={f.get('failure_mode', '?')}"
            )
        failures_text = "\n".join(failure_lines)

    return f"""\
Problem:
  description: {problem.description}
  domain: {problem.domain}
  criterion: {problem.success_criterion}
  criterion_type: {problem.criterion_type}
  failure_rate: {problem.failure_rate:.2f}
  novelty_score: {problem.novelty_score:.2f}
  priority_score: {problem.priority_score:.2f}

Known failed approaches for this domain:
{failures_text}

Propose a solution. Output ONLY a JSON object (no markdown, no explanation).

For prompt_patch:
{{
  "approach": "prompt_patch",
  "description": "<what change to make and why>",
  "modification_spec": {{
    "target_domain": "{problem.domain}",
    "prompt_patch": "<system-prompt addition or replacement>",
    "patch_mode": "append"
  }},
  "expected_outcome": "<how this will improve the criterion>"
}}

For lora_finetune (use when failure_rate > 0.3 and prompt_patch has failed):
{{
  "approach": "lora_finetune",
  "description": "<what domain to fine-tune and why>",
  "modification_spec": {{
    "training_domain": "{problem.domain}",
    "n_pairs": 100,
    "lora_rank": 16,
    "epochs": 2,
    "target_modules": ["q_proj", "v_proj"]
  }},
  "expected_outcome": "<how fine-tuning will improve the criterion>"
}}

For weight_edit (use for specific factual errors):
{{
  "approach": "weight_edit",
  "description": "<what factual correction to make>",
  "modification_spec": {{
    "method": "ROME",
    "edits": [
      {{"subject": "<entity>", "relation": "<relation>", "target": "<correct answer>"}}
    ]
  }},
  "expected_outcome": "<how this edit will fix the factual error>"
}}

Do not reuse any of the known failed approaches."""


def _should_propose_finetune(problem: ProblemPacket, failures: list[dict]) -> bool:
    """
    Heuristic: propose lora_finetune when the domain has systematic failures
    that prompt_patch has not been able to fix.
    """
    if problem.failure_rate < 0.05:
        return False
    prompt_patch_failures = sum(
        1 for f in failures if f.get("approach") == "prompt_patch"
    )
    return prompt_patch_failures >= 2


class SolutionGenerator:
    """
    Generate novel SolutionPlans via the vLLM Server (port 8001).
    Called only when MemoryRetriever finds no high-confidence past solution.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        problem: ProblemPacket,
        failures: list[dict],
    ) -> SolutionPlan:
        """
        Call vLLM Server with a structured prompt describing the problem and
        known failures, parse the JSON response, and return a SolutionPlan.

        When systematic failures are detected (failure_rate > 0.3, multiple
        prompt_patch failures), may force a lora_finetune plan instead of
        relying on the model's choice.

        Falls back to a default prompt_patch plan on any error.
        """
        user_prompt = _build_user_prompt(problem, failures)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        logger.info(
            "generate: calling vLLM for problem_id=%s domain=%s failures=%d",
            problem.id,
            problem.domain,
            len(failures),
        )

        try:
            raw = self._call_server(messages)
            plan = self._parse_response(raw, problem)
        except Exception as exc:
            logger.error(
                "generate failed (problem_id=%s): %s — using fallback plan",
                problem.id,
                exc,
            )
            plan = self._fallback_plan(problem, str(exc))

        # Override: if heuristic says finetune and model chose prompt_patch, escalate
        if plan.approach == "prompt_patch" and _should_propose_finetune(problem, failures):
            logger.info(
                "generate: overriding prompt_patch → lora_finetune for problem_id=%s "
                "(failure_rate=%.2f, %d prompt_patch failures)",
                problem.id, problem.failure_rate,
                sum(1 for f in failures if f.get("approach") == "prompt_patch"),
            )
            plan = self._build_finetune_plan(problem)

        logger.info(
            "generate: plan_id=%s approach=%s problem_id=%s",
            plan.id,
            plan.approach,
            problem.id,
        )
        return plan

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call_server(self, messages: list) -> str:
        """POST to vLLM chat completions endpoint; return raw content string."""
        payload = {
            "model": VLLM_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        logger.debug("vLLM POST → %s  model=%s", CHAT_COMPLETIONS, VLLM_MODEL)
        resp = requests.post(CHAT_COMPLETIONS, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        content: str = data["choices"][0]["message"]["content"]
        logger.debug("vLLM raw response length=%d", len(content))
        if not content or not content.strip():
            raise ValueError("vLLM returned empty content — likely bare <think> with no response")

        return content

    def _parse_response(self, raw: str, problem: ProblemPacket) -> SolutionPlan:
        """
        Parse the model's JSON response into a SolutionPlan.
        Strips <think> blocks, markdown fences, then JSON-decodes.
        Handles multiple JSON objects by extracting only the first one.
        """
        # Strip thinking tokens (e.g. vLLM <think>...</think> blocks)
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Strip markdown code fences
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()

        # Find first '{' and match to its closing '}' via brace counter
        # (handles extra data / multiple JSON objects in the response)
        brace_start = text.find("{")
        if brace_start != -1:
            depth = 0
            brace_end = -1
            for i, ch in enumerate(text[brace_start:], start=brace_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        brace_end = i
                        break
            if brace_end != -1:
                text = text[brace_start : brace_end + 1]

        parsed: dict[str, Any] = json.loads(text)

        approach = parsed.get("approach", "prompt_patch")
        if approach not in VALID_APPROACHES:
            logger.warning(
                "Unknown approach '%s' from model; defaulting to prompt_patch", approach
            )
            approach = "prompt_patch"

        plan = SolutionPlan(
            id=str(uuid.uuid4()),
            problem_id=problem.id,
            timestamp=datetime.utcnow().isoformat(),
            approach=approach,            # type: ignore[arg-type]
            description=str(parsed.get("description", "")),
            modification_spec=parsed.get("modification_spec", {}),
            expected_outcome=str(parsed.get("expected_outcome", "")),
            from_memory=False,
            memory_solution_id=None,
        )
        return plan

    def _build_finetune_plan(self, problem: ProblemPacket) -> SolutionPlan:
        """Build a lora_finetune SolutionPlan for systematic domain failures."""
        # Scale training pairs with failure severity
        n_pairs = min(200, max(50, int(problem.failure_rate * 300)))
        lora_rank = 16 if problem.failure_rate < 0.6 else 32
        epochs = 2 if n_pairs <= 100 else 1

        return SolutionPlan(
            id=str(uuid.uuid4()),
            problem_id=problem.id,
            timestamp=datetime.utcnow().isoformat(),
            approach="lora_finetune",
            description=(
                f"QLoRA fine-tuning for domain '{problem.domain}' — "
                f"failure_rate={problem.failure_rate:.2f}, prompt_patch insufficient"
            ),
            modification_spec={
                "training_domain": problem.domain,
                "n_pairs": n_pairs,
                "lora_rank": lora_rank,
                "lora_alpha": lora_rank * 2,
                "epochs": epochs,
                "target_modules": ["q_proj", "v_proj"],
                "learning_rate": 2e-4,
                "batch_size": 4,
                "description": problem.description,
                "success_criterion": problem.success_criterion,
            },
            expected_outcome=(
                f"Systematic improvement on '{problem.domain}' tasks via "
                f"QLoRA weight adaptation with {n_pairs} domain-specific training pairs."
            ),
            from_memory=False,
            memory_solution_id=None,
        )

    def _fallback_plan(self, problem: ProblemPacket, error: str) -> SolutionPlan:
        """Return a minimal prompt_patch plan when generation fails."""
        return SolutionPlan(
            id=str(uuid.uuid4()),
            problem_id=problem.id,
            timestamp=datetime.utcnow().isoformat(),
            approach="prompt_patch",
            description=(
                f"Fallback prompt_patch for domain '{problem.domain}' "
                f"after generation error: {error[:200]}"
            ),
            modification_spec={
                "target_domain": problem.domain,
                "prompt_patch": (
                    f"When handling {problem.domain} tasks, pay extra attention to: "
                    f"{problem.success_criterion}."
                ),
                "patch_mode": "append",
                "is_fallback": True,
            },
            expected_outcome=(
                f"Marginal improvement on '{problem.domain}' tasks via "
                f"system-prompt reinforcement."
            ),
            from_memory=False,
            memory_solution_id=None,
        )
