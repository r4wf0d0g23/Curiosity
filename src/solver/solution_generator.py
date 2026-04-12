"""
Curiosity — Solver: Solution Generator
Generates novel SolutionPlans via vLLM Server (port 8001) when Memory
has no high-confidence match.

v1 focus: prompt_patch approach (modify system prompt to improve domain).
lora_finetune and weight_edit may be proposed by the model but are handled
as stubs by the Verifier.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

import requests

from src.shared.types import ProblemPacket, SolutionPlan

logger = logging.getLogger("solver.solution_generator")

# ── vLLM Server ───────────────────────────────────────────────────────────────
VLLM_BASE_URL     = "http://localhost:8001"
VLLM_MODEL        = "curiosity-server"
CHAT_COMPLETIONS  = f"{VLLM_BASE_URL}/v1/chat/completions"
REQUEST_TIMEOUT   = 120  # seconds

# Valid approach literals from SolutionPlan
VALID_APPROACHES  = {"weight_edit", "lora_finetune", "prompt_patch", "composite"}


SYSTEM_PROMPT = """\
You are a solution architect for an AI recursive self-improvement system.
Your role is to propose concrete, targeted modifications to improve the model's capabilities.
For v1, prefer the prompt_patch approach (modify the system prompt) unless lora_finetune
or weight_edit are more appropriate for the given domain and criterion.
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

Propose a solution. Output ONLY a JSON object (no markdown, no explanation):
{{
  "approach": "prompt_patch",
  "description": "<concise description of what change to make and why>",
  "modification_spec": {{
    "target_domain": "{problem.domain}",
    "prompt_patch": "<the full text of the system-prompt addition or replacement>",
    "patch_mode": "append"
  }},
  "expected_outcome": "<how this will improve the criterion>"
}}

If lora_finetune or weight_edit are more appropriate, use those instead of prompt_patch
and adjust modification_spec accordingly. Do not reuse any of the known failed approaches."""


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
        return content

    def _parse_response(self, raw: str, problem: ProblemPacket) -> SolutionPlan:
        """
        Parse the model's JSON response into a SolutionPlan.
        Strips markdown fences if present, then JSON-decodes.
        """
        # Strip optional ```json ... ``` fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first and last fence lines
            inner_lines = []
            fence_open = False
            for line in lines:
                if line.strip().startswith("```") and not fence_open:
                    fence_open = True
                    continue
                if line.strip() == "```" and fence_open:
                    break
                if fence_open:
                    inner_lines.append(line)
            text = "\n".join(inner_lines).strip()

        # Find first '{' to handle any leading prose
        brace_start = text.find("{")
        brace_end   = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
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
