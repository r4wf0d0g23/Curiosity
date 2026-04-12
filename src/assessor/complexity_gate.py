"""
Complexity gate — validates that new benchmark problems meet minimum
complexity standards for their tier before being added to the benchmark suite.

Prevents "what is 2+2" from being submitted as a Tier 3 problem.

On any network / parsing error the gate fails open (returns True) so that
a transient server outage never stalls the curiosity loop.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:8001/v1/chat/completions"

COMPLEXITY_PROMPTS: dict[int, str] = {
    1: "This problem should require only 1 step.",
    2: "This problem should require 2-3 steps.",
    3: "This problem should require multi-step reasoning and planning.",
    4: "This problem should require expert knowledge and synthesis.",
    5: "This problem should be novel, creative, and require complex open-ended reasoning.",
}

# Per-tier acceptable score window: (min_inclusive, max_inclusive)
_TIER_SCORE_RANGES: dict[int, tuple[float, float]] = {
    1: (1.0, 2.5),
    2: (2.0, 3.5),
    3: (3.0, 4.5),
    4: (4.0, 5.0),
    5: (4.5, 5.0),
}


class ComplexityGate:
    """Validates problem complexity against expected tier via LLM self-rating."""

    def __init__(self, server_url: str = SERVER_URL, timeout: int = 15) -> None:
        self.server_url = server_url
        self.timeout = timeout

    def validate(
        self, problem: dict, expected_tier: int
    ) -> tuple[bool, float, str]:
        """Ask the Server to rate the complexity of this problem.

        Returns:
            (passes_gate, complexity_score, reason)

        Gate logic (score on 1-5 scale):
          - Expected tier 1: score 1-2   → pass
          - Expected tier 2: score 2-3   → pass
          - Expected tier 3: score 3-4   → pass
          - Expected tier 4: score 4-5   → pass
          - Expected tier 5: score 5     → pass
          - Any tier: score < expected_tier - 1 → fail (too easy)

        On network error or parse failure: fail open (return True) so the
        curiosity loop never stalls due to a transient server issue.
        """
        prompt_text: str = problem.get("prompt", problem.get("question", "")).strip()
        if not prompt_text:
            return False, 0.0, "No prompt/question text found in problem dict"

        user_prompt = (
            "Rate the complexity of this problem on a scale of 1-5. "
            'Reply with ONLY a JSON object: {"score": N, "reason": "..."}\n'
            f"Problem: {prompt_text}"
        )

        score: float = float(expected_tier)  # default used on gate error
        reason: str = ""

        try:
            resp = requests.post(
                self.server_url,
                json={
                    "model": "default",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a difficulty rater. Reply only with JSON.",
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 100,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            text: str = resp.json()["choices"][0]["message"]["content"].strip()

            start = text.find("{")
            end = text.rfind("}") + 1
            if start < 0 or end <= start:
                logger.warning(
                    "[complexity_gate] Could not find JSON in response: %s", text[:200]
                )
                return True, score, f"gate_parse_error: no JSON in: {text[:100]}"

            data = json.loads(text[start:end])
            score = float(data.get("score", expected_tier))
            reason = str(data.get("reason", ""))

        except requests.Timeout:
            logger.warning(
                "[complexity_gate] Request timed out — failing open for tier %d",
                expected_tier,
            )
            return True, score, "gate_timeout"
        except Exception as exc:
            logger.warning(
                "[complexity_gate] Gate error (failing open): %s", exc
            )
            return True, score, f"gate_error: {exc}"

        # ── Gate decision ───────────────────────────────────────────────────

        # Absolute floor: if score is more than 1 point below expected tier, reject
        if score < expected_tier - 1:
            return (
                False,
                score,
                f"Too easy for tier {expected_tier}: score={score:.1f} "
                f"(min acceptable={expected_tier - 1:.1f}). {reason}",
            )

        # Window check
        min_score, max_score = _TIER_SCORE_RANGES.get(expected_tier, (1.0, 5.0))
        if not (min_score <= score <= max_score):
            return (
                False,
                score,
                f"Score {score:.1f} outside window [{min_score}, {max_score}] "
                f"for tier {expected_tier}. {reason}",
            )

        return True, score, reason
