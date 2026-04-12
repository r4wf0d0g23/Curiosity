"""
Curiosity — Verifier Daemon
Daemon 4: Verifies solution plans with checkpoint/rollback safety.
"""

from .checkpoint import CheckpointManager, create_checkpoint, restore_checkpoint, list_checkpoints
from .verifier import VerifierDaemon

__all__ = [
    "CheckpointManager",
    "create_checkpoint",
    "restore_checkpoint",
    "list_checkpoints",
    "VerifierDaemon",
]
