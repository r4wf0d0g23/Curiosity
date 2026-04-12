"""
Curiosity — Memorizer package
Daemon 5: stores MemoryPaths to ChromaDB + disk and closes the improvement loop.
"""

from src.memorizer.memory_store import MemoryStore
from src.memorizer.memorizer import run

__all__ = ["MemoryStore", "run"]
