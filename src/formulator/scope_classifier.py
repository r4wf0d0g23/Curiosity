"""
Curiosity — Scope Classifier
Decides whether a problem warrants a one-off fix (swim) or a generalizable
recurring solution (bridge).
"""


class ScopeClassifier:
    """Classifies a problem's scope based on how often the same domain has
    appeared in recent problem history."""

    # Number of same-domain occurrences required to trigger 'bridge'
    BRIDGE_THRESHOLD = 3

    def classify(self, problem: dict, history: list) -> str:
        """Return 'swim' or 'bridge'.

        Args:
            problem: Raw problem dict (must contain a 'domain' key).
            history: Rolling list of past problem dicts (up to 100 entries,
                     most-recent last).  Each entry must contain 'domain'.

        Returns:
            'bridge' if the same domain has appeared >= BRIDGE_THRESHOLD times
            in *history* (not counting the current problem itself).
            'swim' otherwise.
        """
        domain = problem.get("domain", "").strip().lower()
        if not domain:
            return "swim"

        occurrences = sum(
            1 for h in history
            if h.get("domain", "").strip().lower() == domain
        )

        if occurrences >= self.BRIDGE_THRESHOLD:
            return "bridge"
        return "swim"
