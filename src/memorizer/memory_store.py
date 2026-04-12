"""
Curiosity — Memory Store
Storage layer for MemoryPath objects: ChromaDB (vector) + JSON (disk).
"""

import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
import requests

from src.shared.types import MemoryPath, ProblemPacket, SolutionPlan, VerificationResult

logger = logging.getLogger("memorizer.memory_store")

# ── Paths ────────────────────────────────────────────────────────────────────
MEMORY_BASE = Path(os.path.expanduser("~/curiosity/memory"))
PROBLEMS_DIR = MEMORY_BASE / "problems"
SOLUTIONS_DIR = MEMORY_BASE / "solutions"
OUTCOMES_DIR = MEMORY_BASE / "outcomes"
PATHS_DIR = MEMORY_BASE / "paths"

# ── Embedding ────────────────────────────────────────────────────────────────
EMBED_URL = "http://localhost:8004/v1/embeddings"
EMBED_MODEL = "nemotron-embed"

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "curiosity_memory"


def _get_embedding(text: str) -> list[float]:
    """Fetch embedding from nemotron-embed running on port 8004."""
    resp = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _path_to_dict(path: MemoryPath) -> dict:
    """Recursively convert a MemoryPath dataclass to a plain dict."""
    return asdict(path)


def _dict_to_path(d: dict) -> MemoryPath:
    """Reconstruct a MemoryPath from a plain dict."""
    problem = ProblemPacket(**d["problem"])
    solution = SolutionPlan(**d["solution"])
    result = VerificationResult(**d["result"])
    return MemoryPath(
        id=d["id"],
        timestamp=d["timestamp"],
        problem=problem,
        solution=solution,
        result=result,
        improved_by=d.get("improved_by"),
        improvement_of=d.get("improvement_of"),
    )


class MemoryStore:
    """Append-only store for complete problem→solution→outcome paths."""

    def __init__(self):
        self._ensure_dirs()
        self._chroma: Optional[chromadb.HttpClient] = None
        self._collection = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _ensure_dirs(self):
        for d in [PROBLEMS_DIR, SOLUTIONS_DIR, OUTCOMES_DIR, PATHS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def _get_collection(self):
        """Lazily connect to ChromaDB; reconnect on failure."""
        if self._collection is None:
            self._chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            self._collection = self._chroma.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── Core API ──────────────────────────────────────────────────────────────

    def store_path(self, path: MemoryPath) -> str:
        """Store a complete problem→solution→outcome path. Returns memory_id."""
        memory_id = path.id or str(uuid.uuid4())
        path.id = memory_id

        path_dict = _path_to_dict(path)
        outcome = path.result.outcome  # "pass" or "fail"

        # 1. Write JSON to disk (append-only — never overwrite an existing file)
        self._write_json(path_dict, memory_id, outcome)

        # 2. Store vector in ChromaDB
        self._store_chroma(path, path_dict, memory_id, outcome)

        logger.info(
            "Stored memory_id=%s domain=%s outcome=%s approach=%s",
            memory_id,
            path.problem.domain,
            outcome,
            path.solution.approach,
        )
        return memory_id

    def _write_json(self, path_dict: dict, memory_id: str, outcome: str):
        """Write JSON record to disk under paths/, outcomes/, problems/, solutions/."""
        # Primary record
        self._safe_write(PATHS_DIR / f"{memory_id}.json", path_dict)

        # Indexes (human-readable; redundant but append-only)
        domain = path_dict["problem"].get("domain", "unknown")
        approach = path_dict["solution"].get("approach", "unknown")
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

        index_key = f"{domain}__{approach}__{ts}__{memory_id}"
        outcomes_sub = OUTCOMES_DIR / outcome
        outcomes_sub.mkdir(parents=True, exist_ok=True)
        self._safe_write(outcomes_sub / f"{index_key}.json", path_dict)

    def _safe_write(self, filepath: Path, data: dict):
        """Write JSON only if file does not yet exist (append-only guarantee)."""
        if filepath.exists():
            logger.warning("File already exists, skipping write: %s", filepath)
            return
        filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _store_chroma(self, path: MemoryPath, path_dict: dict, memory_id: str, outcome: str):
        """Embed problem description and upsert into ChromaDB."""
        collection = self._get_collection()

        embed_text = (
            f"domain: {path.problem.domain}\n"
            f"description: {path.problem.description}\n"
            f"success_criterion: {path.problem.success_criterion}\n"
            f"approach: {path.solution.approach}\n"
            f"solution_description: {path.solution.description}"
        )

        embedding = _get_embedding(embed_text)

        metadata = {
            "domain": path.problem.domain,
            "outcome": outcome,
            "approach": path.solution.approach,
            "problem_id": path.problem.id,
            "solution_id": path.solution.id,
            "result_id": path.result.id,
            "criterion_score": path.result.criterion_score,
            "timestamp": path.timestamp,
            "failure_mode": path.result.failure_mode,
        }

        collection.upsert(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[embed_text],
            metadatas=[metadata],
        )
        logger.debug("ChromaDB upsert ok memory_id=%s", memory_id)

    # ── Query API ─────────────────────────────────────────────────────────────

    def query_similar(self, problem: ProblemPacket, top_k: int = 5) -> list[MemoryPath]:
        """Find similar past problems using embedding similarity."""
        collection = self._get_collection()

        embed_text = (
            f"domain: {problem.domain}\n"
            f"description: {problem.description}\n"
            f"success_criterion: {problem.success_criterion}"
        )
        embedding = _get_embedding(embed_text)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
        )

        paths = []
        if results and results.get("ids"):
            for memory_id in results["ids"][0]:
                path = self._load_path(memory_id)
                if path:
                    paths.append(path)
        return paths

    def get_successful_solutions(self, domain: str) -> list[MemoryPath]:
        """Get all passing solutions for a domain."""
        return self._load_from_outcomes_dir("pass", domain)

    def get_failures(self, domain: str) -> list[MemoryPath]:
        """Get all failures for a domain — used by Solver to avoid dead ends."""
        return self._load_from_outcomes_dir("fail", domain)

    def _load_from_outcomes_dir(self, outcome: str, domain: str) -> list[MemoryPath]:
        outcome_dir = OUTCOMES_DIR / outcome
        if not outcome_dir.exists():
            return []
        paths = []
        for f in sorted(outcome_dir.glob(f"{domain}__*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                paths.append(_dict_to_path(data))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", f, exc)
        return paths

    def _load_path(self, memory_id: str) -> Optional[MemoryPath]:
        filepath = PATHS_DIR / f"{memory_id}.json"
        if not filepath.exists():
            logger.warning("memory_id not found on disk: %s", memory_id)
            return None
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            return _dict_to_path(data)
        except Exception as exc:
            logger.error("Failed to deserialize %s: %s", filepath, exc)
            return None

    # ── Improvement Loop ──────────────────────────────────────────────────────

    def run_improvement_loop(self):
        """
        Scan stored solutions and look for improvement opportunities.

        Current strategy:
        - For each domain, compare failed and successful approaches.
        - Log domains where failures outnumber successes (signal for Solver).
        - Placeholder for future: propose composite solutions that combine
          high-scoring partial successes.
        """
        logger.info("Running improvement loop scan…")
        outcomes_dir = OUTCOMES_DIR
        if not outcomes_dir.exists():
            return

        domains: set[str] = set()
        for outcome in ("pass", "fail"):
            d = outcomes_dir / outcome
            if d.exists():
                for f in d.glob("*.json"):
                    domain = f.name.split("__")[0]
                    domains.add(domain)

        for domain in sorted(domains):
            successes = self.get_successful_solutions(domain)
            failures = self.get_failures(domain)
            logger.info(
                "Domain '%s': %d successes, %d failures",
                domain,
                len(successes),
                len(failures),
            )
            if failures and not successes:
                logger.warning(
                    "Domain '%s' has only failures — Solver should diversify approach.",
                    domain,
                )
            elif successes:
                best = max(successes, key=lambda p: p.result.criterion_score)
                logger.info(
                    "Domain '%s' best score=%.3f approach=%s",
                    domain,
                    best.result.criterion_score,
                    best.solution.approach,
                )
        logger.info("Improvement loop scan complete.")
