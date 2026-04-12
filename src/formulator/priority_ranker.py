"""
Curiosity — Priority Ranker
Computes a normalised priority score for a ProblemPacket.
"""


class PriorityRanker:
    """Ranks problems by a weighted combination of severity, frequency, and novelty."""

    def score(self, severity: float, frequency: int, novelty: float) -> float:
        """
        Compute priority score in [0.0, 1.0].

        Formula:
            priority = (severity * 0.5)
                     + (min(frequency, 10) / 10 * 0.3)
                     + (novelty * 0.2)

        Args:
            severity:  Proxy for failure_rate — how bad the problem is (0.0–1.0).
            frequency: How often this problem type appears (capped at 10 for scoring).
            novelty:   Distance from already-explored territory (0.0–1.0).

        Returns:
            Weighted priority score in [0.0, 1.0].
        """
        severity = max(0.0, min(1.0, float(severity)))
        frequency = max(0, int(frequency))
        novelty = max(0.0, min(1.0, float(novelty)))

        priority = (
            (severity * 0.5)
            + (min(frequency, 10) / 10.0 * 0.3)
            + (novelty * 0.2)
        )
        return round(min(max(priority, 0.0), 1.0), 6)
