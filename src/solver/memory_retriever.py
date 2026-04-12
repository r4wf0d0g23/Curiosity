"""
Curiosity — Solver: Memory Retriever
Queries ChromaDB for similar past solutions and failed approaches.
Embeds problem descriptions via nemotron-embed (port 8004).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from typing import Optional
import uuid

import chromadb
import requests

from shared.types import ProblemPacket, SolutionPlan

logger = logging.getLogger("solver.memory_retriever")

# ── Embedding service ─────────────────────────────────────────────────────────
EMBED_URL   = "http://localhost:8004/v1/embeddings"
EMBED_MODEL = "nemotron-embed"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_HOST     = "localhost"
CHROMA_PORT     = 8000
COLLECTION_NAME = "curiosity_memory"

SIMILARITY_THRESHOLD = 0.75   # cosine distance < (1 - threshold) is a "good match"


def _get_embedding(text: str) -> list[float]:
    """Embed text via nemotron-embed running on port 8004."""
    resp = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _problem_to_embed_text(problem: ProblemPacket) -> str:
    return (
        f"domain: {problem.domain}\n"
        f"description: {problem.description}\n"
        f"success_criterion: {problem.success_criterion}"
    )


class MemoryRetriever:
    """
    Query ChromaDB for similar past problems and retrieve SolutionPlans.
    Uses nemotron-embed (port 8004) for query embedding.
    Only returns solutions with outcome='pass'.
    """

    def __init__(self) -> None:
        self._client: Optional[chromadb.HttpClient] = None
        self._collection = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_collection(self):
        """Lazily connect to ChromaDB; reconnect on failure."""
        if self._collection is None:
            self._client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _reset_connection(self) -> None:
        self._client = None
        self._collection = None

    # ── Public API ────────────────────────────────────────────────────────────

    def query_similar(
        self, problem: ProblemPacket, top_k: int = 5
    ) -> list[dict]:
        """
        Embed the problem description via nemotron-embed (port 8004),
        query ChromaDB collection 'curiosity_memory', and return top_k
        results with similarity scores.

        Filter: only return results with outcome='pass'.

        Returns a list of dicts with keys:
          - memory_id   : str
          - similarity  : float  (0.0–1.0; higher = more similar)
          - metadata    : dict   (ChromaDB metadata)
          - document    : str    (original embed text)
        """
        embed_text = _problem_to_embed_text(problem)
        try:
            embedding = _get_embedding(embed_text)
        except Exception as exc:
            logger.error("Failed to embed problem (query_similar): %s", exc)
            return []

        # Fetch more than top_k so we can filter to 'pass' outcomes
        fetch_k = max(top_k * 3, 15)
        try:
            collection = self._get_collection()
            results = collection.query(
                query_embeddings=[embedding],
                n_results=fetch_k,
                where={"outcome": "pass"},
                include=["metadatas", "distances", "documents"],
            )
        except Exception as exc:
            logger.error("ChromaDB query_similar failed: %s", exc)
            self._reset_connection()
            return []

        hits: list[dict] = []
        if not results or not results.get("ids"):
            return hits

        ids       = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        for memory_id, dist, meta, doc in zip(ids, distances, metadatas, documents):
            # ChromaDB cosine space returns distance in [0, 2]; normalise to similarity
            # distance = 1 − cosine_similarity  →  similarity = 1 − distance
            similarity = max(0.0, 1.0 - dist)

            hits.append(
                {
                    "memory_id": memory_id,
                    "similarity": similarity,
                    "metadata": meta,
                    "document": doc,
                }
            )
            logger.debug(
                "query_similar hit memory_id=%s similarity=%.4f domain=%s approach=%s",
                memory_id,
                similarity,
                meta.get("domain"),
                meta.get("approach"),
            )
            if len(hits) >= top_k:
                break

        logger.info(
            "query_similar problem_id=%s domain=%s → %d pass results (top similarity=%.4f)",
            problem.id,
            problem.domain,
            len(hits),
            hits[0]["similarity"] if hits else 0.0,
        )
        return hits

    def get_failures(self, domain: str) -> list[dict]:
        """
        Get known failed approaches for this domain — used to avoid
        repeating dead ends.

        Returns a list of dicts with keys:
          - memory_id     : str
          - approach      : str
          - failure_mode  : str
          - description   : str  (from metadata document)
        """
        try:
            collection = self._get_collection()
            # Query with a neutral embedding-less approach: use where filter only.
            # ChromaDB doesn't support pure metadata filter without an embedding,
            # so we use a dummy empty query but rely on the where clause.
            results = collection.get(
                where={"$and": [{"outcome": "fail"}, {"domain": domain}]},
                include=["metadatas", "documents"],
            )
        except Exception as exc:
            logger.error("ChromaDB get_failures failed for domain=%s: %s", domain, exc)
            self._reset_connection()
            return []

        failures: list[dict] = []
        if not results or not results.get("ids"):
            return failures

        for memory_id, meta, doc in zip(
            results["ids"], results["metadatas"], results["documents"]
        ):
            failures.append(
                {
                    "memory_id":    memory_id,
                    "approach":     meta.get("approach", "unknown"),
                    "failure_mode": meta.get("failure_mode", ""),
                    "description":  doc,
                }
            )

        logger.info(
            "get_failures domain=%s → %d known failures", domain, len(failures)
        )
        return failures

    def adapt_solution(
        self, prior: dict, current: ProblemPacket
    ) -> SolutionPlan:
        """
        Adapt a prior solution (ChromaDB hit dict) to the current problem.

        Strategy:
        - Preserve approach type and modification_spec structure from prior.
        - Update description and expected_outcome to reference current problem.
        - Tag plan with from_memory=True and the source memory_id.
        """
        meta = prior.get("metadata", {})
        approach  = meta.get("approach", "prompt_patch")
        memory_id = prior.get("memory_id", "")

        # For prompt_patch: keep spec but update target domain/description
        modification_spec: dict = {
            "adapted_from_memory_id": memory_id,
            "target_domain": current.domain,
            "prompt_patch": (
                f"You are an expert assistant for {current.domain} tasks. "
                f"Focus on: {current.success_criterion}. "
                f"Apply the approach that previously worked: {meta.get('approach', 'prompt_patch')}."
            ),
        }

        plan = SolutionPlan(
            id=str(uuid.uuid4()),
            problem_id=current.id,
            timestamp=datetime.utcnow().isoformat(),
            approach=approach,         # type: ignore[arg-type]
            description=(
                f"Adapted from memory {memory_id}: "
                f"Apply {approach} for domain '{current.domain}'. "
                f"Criterion: {current.success_criterion}"
            ),
            modification_spec=modification_spec,
            expected_outcome=(
                f"Improve performance on '{current.domain}' tasks; "
                f"pass criterion: {current.success_criterion}"
            ),
            from_memory=True,
            memory_solution_id=meta.get("solution_id"),
        )
        logger.info(
            "adapt_solution memory_id=%s → plan_id=%s approach=%s",
            memory_id,
            plan.id,
            approach,
        )
        return plan
