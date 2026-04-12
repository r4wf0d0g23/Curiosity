"""
Curriculum manager — tracks difficulty tiers per domain and auto-escalates.

Tiers:
  L1: Single-step, direct answer (e.g. "what is 2+2")
  L2: 2-3 step reasoning required
  L3: Multi-step, requires planning or domain knowledge
  L4: Expert-level, requires synthesis or complex chains
  L5: Novel/creative, requires open-ended reasoning

Escalation rule: pass_rate >= 0.92 for 3 consecutive cycles → unlock next tier
De-escalation rule: pass_rate < 0.50 for 2 consecutive cycles → drop back one tier
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CURRICULUM_STATE_FILE = Path.home() / "curiosity" / "benchmarks" / "curriculum_state.json"

TIER_DESCRIPTIONS = {
    1: "Single-step, direct factual answer",
    2: "2-3 reasoning steps, some domain knowledge",
    3: "Multi-step reasoning, requires planning",
    4: "Expert-level, requires synthesis across concepts",
    5: "Novel/creative, open-ended complex reasoning",
}

_TIER_PROMPT_TEMPLATES: dict[int, str] = {
    1: (
        "Generate a TIER 1 (easy) problem: single-step, direct factual answer. "
        "The problem must be answerable in exactly one reasoning step."
    ),
    2: (
        "Generate a TIER 2 (medium) problem requiring 2-3 reasoning steps and some domain knowledge. "
        "Do NOT generate single-step or trivially easy problems."
    ),
    3: (
        "Generate a TIER 3 (hard) problem: multi-step reasoning requiring planning. "
        "Do NOT generate single-step or trivially easy problems. "
        "The problem must require at least 3 distinct reasoning steps."
    ),
    4: (
        "Generate a TIER 4 (expert) problem requiring expert-level knowledge and synthesis across concepts. "
        "The problem must demand deep domain expertise and integration of multiple non-trivial ideas."
    ),
    5: (
        "Generate a TIER 5 (novel/creative) problem: novel, open-ended, requiring complex reasoning. "
        "The problem should demand innovative thinking and have no single obvious answer path."
    ),
}


class CurriculumManager:
    """Tracks per-domain difficulty tiers and auto-escalates / de-escalates."""

    def __init__(self) -> None:
        self.state: dict = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Load curriculum state from disk.

        Returns: {domain: {current_tier, consecutive_passes, consecutive_fails}}
        Auto-creates an empty state file on first run.
        """
        if CURRICULUM_STATE_FILE.exists():
            try:
                with CURRICULUM_STATE_FILE.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                logger.debug("[curriculum] Loaded state from %s", CURRICULUM_STATE_FILE)
                return data
            except Exception as exc:
                logger.warning(
                    "[curriculum] Failed to load state from %s: %s — starting fresh",
                    CURRICULUM_STATE_FILE,
                    exc,
                )
        return {}

    def _save(self) -> None:
        """Persist curriculum state to disk."""
        try:
            CURRICULUM_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with CURRICULUM_STATE_FILE.open("w", encoding="utf-8") as fh:
                json.dump(self.state, fh, indent=2)
        except Exception as exc:
            logger.error("[curriculum] Failed to save state: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _domain_state(self, domain: str) -> dict:
        """Return (and lazily initialise) per-domain state dict."""
        if domain not in self.state:
            self.state[domain] = {
                "current_tier": 1,
                "consecutive_passes": 0,
                "consecutive_fails": 0,
            }
        return self.state[domain]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tier(self, domain: str) -> int:
        """Return current difficulty tier for domain (default 1)."""
        return int(self._domain_state(domain)["current_tier"])

    def record_result(self, domain: str, pass_rate: float) -> None:
        """Update consecutive pass/fail counters and escalate/de-escalate tiers.

        Escalation:  pass_rate >= 0.92 for 3 consecutive cycles → tier += 1 (max 5)
        De-escalation: pass_rate < 0.50 for 2 consecutive cycles → tier -= 1 (min 1)
        """
        ds = self._domain_state(domain)
        old_tier = ds["current_tier"]

        if pass_rate >= 0.92:
            ds["consecutive_passes"] = ds.get("consecutive_passes", 0) + 1
            ds["consecutive_fails"] = 0
            if ds["consecutive_passes"] >= 3 and ds["current_tier"] < 5:
                ds["current_tier"] += 1
                ds["consecutive_passes"] = 0
                logger.info(
                    "[curriculum] Domain %r ESCALATED: tier %d → %d "
                    "(pass_rate=%.2f, 3 consecutive high-pass cycles)",
                    domain,
                    old_tier,
                    ds["current_tier"],
                    pass_rate,
                )
        elif pass_rate < 0.50:
            ds["consecutive_fails"] = ds.get("consecutive_fails", 0) + 1
            ds["consecutive_passes"] = 0
            if ds["consecutive_fails"] >= 2 and ds["current_tier"] > 1:
                ds["current_tier"] -= 1
                ds["consecutive_fails"] = 0
                logger.info(
                    "[curriculum] Domain %r DE-ESCALATED: tier %d → %d "
                    "(pass_rate=%.2f, 2 consecutive low-pass cycles)",
                    domain,
                    old_tier,
                    ds["current_tier"],
                    pass_rate,
                )
        else:
            # Neutral result — reset both streak counters
            ds["consecutive_passes"] = 0
            ds["consecutive_fails"] = 0

        self._save()

    def get_tier_prompt_instructions(self, domain: str) -> str:
        """Return tier-appropriate generation instructions for curiosity probes.

        Used by the Assessor's curiosity probe generator to inject difficulty
        constraints into the generation prompt.
        """
        tier = self.get_tier(domain)
        return _TIER_PROMPT_TEMPLATES.get(tier, _TIER_PROMPT_TEMPLATES[1])
