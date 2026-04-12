"""
Curiosity — ProbeRunner
Executes benchmark probes against the vLLM Server and scores responses.
"""

import json
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Default server endpoint
DEFAULT_SERVER_URL = "http://localhost:8001/v1/chat/completions"
DEFAULT_MODEL = "curiosity-server"
DEFAULT_TIMEOUT = 60  # seconds per request


class ProbeRunner:
    """Runs benchmark probes against the vLLM Server and scores responses."""

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        llm_judge_url: Optional[str] = None,
    ):
        self.server_url = server_url
        self.model = model
        self.timeout = timeout
        # For llm_judge checks we re-use the same server by default
        self.llm_judge_url = llm_judge_url or server_url

    # ── Core query ─────────────────────────────────────────────────────────────

    def _query_server(self, prompt: str, url: Optional[str] = None) -> tuple[str, float]:
        """Send a prompt to the vLLM server.

        Returns:
            (response_text, latency_seconds)
        Raises:
            requests.RequestException on network / HTTP errors.
        """
        target_url = url or self.server_url
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        t0 = time.monotonic()
        resp = requests.post(target_url, json=payload, timeout=self.timeout)
        latency = time.monotonic() - t0

        resp.raise_for_status()
        data = resp.json()

        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected response shape: {data}") from exc

        return text, latency

    # ── Response scoring ───────────────────────────────────────────────────────

    def check_response(self, response: str, expected: str, check_type: str) -> bool:
        """Score a model response against the expected answer.

        Args:
            response:   Raw text returned by the model.
            expected:   Ground-truth / target string.
            check_type: One of "exact", "contains", "llm_judge".

        Returns:
            True if the response is considered correct.
        """
        if check_type == "exact":
            return response.strip().lower() == expected.strip().lower()

        elif check_type == "contains":
            return expected.strip().lower() in response.strip().lower()

        elif check_type == "llm_judge":
            return self._llm_judge(response, expected)

        else:
            logger.warning("Unknown check_type '%s' — defaulting to contains", check_type)
            return expected.strip().lower() in response.strip().lower()

    def _llm_judge(self, response: str, expected: str) -> bool:
        """Use the Server itself to judge whether response satisfies expected.

        Returns True if the judge says the answer is correct/acceptable.
        """
        judge_prompt = (
            "You are an impartial judge evaluating an AI's answer.\n\n"
            f"Expected (ideal answer or criterion): {expected}\n\n"
            f"Model's response: {response}\n\n"
            "Is the model's response correct or acceptable given the expected answer? "
            "Reply with a single word: YES or NO."
        )
        try:
            verdict, _ = self._query_server(judge_prompt, url=self.llm_judge_url)
            return verdict.strip().upper().startswith("YES")
        except Exception as exc:
            logger.error("LLM judge failed: %s — treating as incorrect", exc)
            return False

    # ── Single probe ───────────────────────────────────────────────────────────

    def run_probe(self, benchmark: dict) -> dict:
        """Run a single benchmark task against the Server.

        Args:
            benchmark: A benchmark dict with keys: id, domain, prompt,
                       expected, check.

        Returns:
            {
                "benchmark_id": str,
                "domain": str,
                "score": bool,   # True = pass
                "response": str,
                "latency": float,
                "error": str | None,
            }
        """
        result: dict = {
            "benchmark_id": benchmark.get("id", "unknown"),
            "domain": benchmark.get("domain", "unknown"),
            "score": False,
            "response": "",
            "latency": 0.0,
            "error": None,
        }

        try:
            response_text, latency = self._query_server(benchmark["prompt"])
            result["response"] = response_text
            result["latency"] = latency

            passed = self.check_response(
                response_text,
                benchmark.get("expected", ""),
                benchmark.get("check", "contains"),
            )
            result["score"] = passed

            logger.debug(
                "[probe] %s | domain=%s | pass=%s | latency=%.2fs",
                result["benchmark_id"],
                result["domain"],
                passed,
                latency,
            )

        except Exception as exc:
            logger.error(
                "[probe] %s FAILED with exception: %s",
                result["benchmark_id"],
                exc,
            )
            result["error"] = str(exc)
            result["score"] = False

        return result

    # ── Full suite ─────────────────────────────────────────────────────────────

    def run_suite(self, benchmarks: list[dict]) -> dict:
        """Run an entire benchmark suite and aggregate results by domain.

        Args:
            benchmarks: List of benchmark task dicts.

        Returns:
            {
                domain: {
                    "pass_rate": float,       # 0.0 – 1.0
                    "failure_rate": float,
                    "pass_count": int,
                    "total": int,
                    "avg_latency": float,
                    "failures": [             # failing benchmark result dicts
                        { benchmark_id, response, error, ... }
                    ],
                }
            }
        """
        # Group benchmarks by domain first (preserves order within domain)
        domain_map: dict[str, list[dict]] = {}
        for b in benchmarks:
            domain = b.get("domain", "unknown")
            domain_map.setdefault(domain, []).append(b)

        suite_results: dict[str, dict] = {}

        for domain, tasks in domain_map.items():
            pass_count = 0
            failures: list[dict] = []
            latencies: list[float] = []

            for task in tasks:
                probe_result = self.run_probe(task)
                latencies.append(probe_result["latency"])

                if probe_result["score"]:
                    pass_count += 1
                else:
                    failures.append(probe_result)

            total = len(tasks)
            pass_rate = pass_count / total if total > 0 else 0.0
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

            suite_results[domain] = {
                "pass_rate": pass_rate,
                "failure_rate": 1.0 - pass_rate,
                "pass_count": pass_count,
                "total": total,
                "avg_latency": avg_latency,
                "failures": failures,
            }

            logger.info(
                "[suite] domain=%s | pass=%d/%d (%.1f%%) | avg_latency=%.2fs",
                domain,
                pass_count,
                total,
                pass_rate * 100,
                avg_latency,
            )

        return suite_results
