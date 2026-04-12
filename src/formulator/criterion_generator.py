"""
Curiosity — Criterion Generator
Produces an automatable success criterion for a given problem description.

Criterion types
---------------
benchmark  — domain maps to a known gap-benchmark suite; target = reduce
             failure rate by 10 percentage points (absolute).
unit_test  — domain is 'code'; the solution must pass a generated test suite.
llm_judge  — fall-through; the Server (vLLM) evaluates the solution.
             If confidence < 0.5 → automatable: False (deprioritize).
"""

import logging
import os

logger = logging.getLogger("formulator.criterion_generator")

# ── Known benchmark suites per domain ─────────────────────────────────────────
# Maps lower-cased domain names to their canonical benchmark suite IDs.
DOMAIN_BENCHMARKS: dict[str, str] = {
    "math":       "math_basic",
    "mathematics":"math_basic",
    "algebra":    "math_algebra",
    "geometry":   "math_geometry",
    "reasoning":  "reasoning_arc",
    "logic":      "reasoning_arc",
    "reading":    "reading_comprehension",
    "science":    "mmlu_science",
    "biology":    "mmlu_biology",
    "chemistry":  "mmlu_chemistry",
    "physics":    "mmlu_physics",
    "history":    "mmlu_history",
    "language":   "mmlu_language",
    "security":   "cybersecurity_ctf",
    "coding":     "humaneval",
    # 'code' is handled separately via unit_test — kept here as fallback
}

# Domains that always use unit_test regardless of benchmark existence
CODE_DOMAINS: set[str] = {"code", "coding", "programming", "software"}

# vLLM endpoint for llm_judge confidence estimation
VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
VLLM_PORT = int(os.environ.get("VLLM_PORT", 8001))
VLLM_MODEL = os.environ.get("VLLM_MODEL", "curiosity-server")

# Minimum confidence for an llm_judge to be considered automatable
LLM_JUDGE_MIN_CONFIDENCE = 0.5


class CriterionGenerator:
    """Generate an automatable success criterion from a problem description."""

    def generate(self, problem: dict) -> dict:
        """
        Args:
            problem: dict with at least:
                - 'domain'       (str)
                - 'description'  (str)
                - 'failure_rate' (float, 0.0–1.0)

        Returns:
            {
              'criterion':    'benchmark' | 'unit_test' | 'llm_judge',
              'description':  str,
              'target_score': float,
              'benchmark_id': str,   # populated for benchmark type, else ''
              'automatable':  bool,
            }
        """
        domain      = problem.get("domain", "").strip().lower()
        description = problem.get("description", "")
        failure_rate = float(problem.get("failure_rate", 0.0))

        # ── 1. Code domain → unit_test ─────────────────────────────────────
        if domain in CODE_DOMAINS:
            return self._unit_test_criterion(description)

        # ── 2. Domain maps to a benchmark suite → benchmark ────────────────
        benchmark_id = DOMAIN_BENCHMARKS.get(domain, "")
        if benchmark_id:
            return self._benchmark_criterion(
                description, benchmark_id, failure_rate
            )

        # ── 3. Fall-through → llm_judge ────────────────────────────────────
        return self._llm_judge_criterion(domain, description)

    # ── Builders ──────────────────────────────────────────────────────────────

    def _benchmark_criterion(
        self, description: str, benchmark_id: str, failure_rate: float
    ) -> dict:
        """Reduce failure rate by 10 pp (absolute floor = 0)."""
        current_pass_rate = 1.0 - failure_rate
        target_score = round(min(current_pass_rate + 0.10, 1.0), 4)
        desc = (
            f"Pass rate on {benchmark_id} suite must reach >= {target_score:.2f} "
            f"(current failure_rate={failure_rate:.2f}, target reduces failures by 10 pp)."
        )
        logger.debug(
            "benchmark criterion: suite=%s target=%.4f", benchmark_id, target_score
        )
        return {
            "criterion":    "benchmark",
            "description":  desc,
            "target_score": target_score,
            "benchmark_id": benchmark_id,
            "automatable":  True,
        }

    def _unit_test_criterion(self, description: str) -> dict:
        """Code domain: auto-generate a test suite."""
        desc = (
            "Auto-generated unit-test suite must pass at >= 0.90 pass-rate. "
            f"Tests cover: {description[:200]}"
        )
        logger.debug("unit_test criterion generated")
        return {
            "criterion":    "unit_test",
            "description":  desc,
            "target_score": 0.90,
            "benchmark_id": "",
            "automatable":  True,
        }

    def _llm_judge_criterion(self, domain: str, description: str) -> dict:
        """Ask the vLLM Server to judge; estimate confidence heuristically."""
        confidence = self._estimate_llm_judge_confidence(domain, description)
        automatable = confidence >= LLM_JUDGE_MIN_CONFIDENCE

        desc = (
            f"LLM-judge evaluation of solution quality for domain='{domain}'. "
            f"Target: judge score >= 0.75. Confidence in criterion: {confidence:.2f}. "
            f"{'AUTOMATABLE' if automatable else 'NON-AUTOMATABLE — deprioritized'}."
        )
        logger.debug(
            "llm_judge criterion: domain=%s confidence=%.2f automatable=%s",
            domain, confidence, automatable,
        )
        return {
            "criterion":    "llm_judge",
            "description":  desc,
            "target_score": 0.75,
            "benchmark_id": "",
            "automatable":  automatable,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _estimate_llm_judge_confidence(self, domain: str, description: str) -> float:
        """
        Heuristic confidence that an LLM judge can reliably evaluate this domain.

        Strategy (no network call to keep the hot path fast):
        - Well-known domains with rich public benchmarks → high confidence.
        - Very short / vague descriptions → lower confidence.
        - Unknown / niche domains → moderate-to-low confidence.

        Returns a float in [0.0, 1.0].
        """
        HIGH_CONFIDENCE_DOMAINS = {
            "language", "writing", "summarization", "translation",
            "question_answering", "qa", "sentiment", "ethics",
            "philosophy", "medicine", "law",
        }
        MEDIUM_CONFIDENCE_DOMAINS = {
            "reasoning", "logic", "common_sense", "trivia",
            "history", "geography",
        }

        base: float
        if domain in HIGH_CONFIDENCE_DOMAINS:
            base = 0.80
        elif domain in MEDIUM_CONFIDENCE_DOMAINS:
            base = 0.65
        else:
            base = 0.45  # niche / unknown → below threshold → non-automatable

        # Penalise vague descriptions
        word_count = len(description.split())
        if word_count < 5:
            base -= 0.20
        elif word_count < 15:
            base -= 0.05

        return round(max(0.0, min(1.0, base)), 4)
