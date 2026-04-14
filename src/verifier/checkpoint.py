"""
Curiosity — Checkpoint Manager
Manages server-state snapshots before modifications so rollback is always possible.

v0.1: Checkpoints capture the current base-model identity and any active LoRA
adapter configuration (JSON). When real weight editing lands, plug the new
capture/restore logic into _capture_weight_state / _restore_weight_state.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("curiosity.verifier.checkpoint")

# ---------------------------------------------------------------------------
# Config (override via env)
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path(os.environ.get("CURIOSITY_CHECKPOINT_DIR", Path.home() / "curiosity" / "checkpoints"))
VLLM_BASE_URL  = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")
SYSTEM_PROMPT_PATH = Path(os.environ.get(
    "CURIOSITY_SYSTEM_PROMPT_PATH",
    Path.home() / "curiosity_code" / "config" / "system_prompt.txt"
))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CheckpointRecord:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Server identity captured at checkpoint time
    base_model: str = ""                  # model name/path reported by vLLM
    lora_adapter: Optional[Dict[str, Any]] = None   # active LoRA config JSON, if any
    system_prompt_hash: str = ""          # sha256 of system prompt file
    system_prompt_snapshot: str = ""      # full text of system prompt at checkpoint

    # Extension point: real weight snapshot path (future ROME/MEMIT)
    weight_snapshot_path: Optional[str] = None

    # Human-readable note
    note: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _checkpoint_dir() -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR


def _record_path(checkpoint_id: str) -> Path:
    return _checkpoint_dir() / f"{checkpoint_id}.json"


def _snapshot_dir(checkpoint_id: str) -> Path:
    d = _checkpoint_dir() / checkpoint_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash_file(path: Path) -> str:
    import hashlib
    if not path.exists():
        return ""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _get_vllm_model() -> str:
    """Ask vLLM /v1/models for the active model name."""
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if models:
            return models[0].get("id", "unknown")
    except Exception as exc:
        logger.warning("Could not query vLLM model list: %s", exc)
    return "unknown"


def _get_lora_config() -> Optional[Dict[str, Any]]:
    """
    Attempt to read the active LoRA adapter config from vLLM.
    v0.1: vLLM doesn't expose a direct LoRA status endpoint, so we check a
    known config file path on disk. Swap this out for a real API call once
    vLLM's dynamic LoRA endpoint is in use.
    """
    lora_cfg_path_str = os.environ.get("CURIOSITY_LORA_CONFIG", "").strip()
    if not lora_cfg_path_str:
        return None
    lora_cfg_path = Path(lora_cfg_path_str)
    if lora_cfg_path.exists():
        try:
            return json.loads(lora_cfg_path.read_text())
        except Exception as exc:
            logger.warning("Could not read LoRA config: %s", exc)
    return None


def _capture_weight_state(snapshot_dir: Path) -> Optional[str]:
    """
    Extension point for real weight snapshots (ROME/MEMIT).
    Returns path to snapshot artifact, or None if not applicable.
    v0.1: no-op.
    """
    # TODO: implement weight delta capture when real weight editing lands
    return None


def _restore_weight_state(weight_snapshot_path: str) -> None:
    """
    Extension point for restoring real weight snapshots.
    v0.1: no-op.
    """
    # TODO: implement weight restoration when real weight editing lands
    pass


def _save_system_prompt_snapshot(checkpoint_id: str, text: str) -> Path:
    snap = _snapshot_dir(checkpoint_id) / "system_prompt.txt"
    snap.write_text(text, encoding="utf-8")
    return snap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_checkpoint(label: str = "", note: str = "") -> CheckpointRecord:
    """
    Capture current server state and persist to disk.
    Always call this BEFORE applying any modification.
    Returns the CheckpointRecord (use .id for later restore).
    """
    record = CheckpointRecord(label=label or f"auto_{int(time.time())}", note=note)

    # 1. Capture base model
    record.base_model = _get_vllm_model()
    logger.info("[checkpoint] base model: %s", record.base_model)

    # 2. Capture LoRA config
    record.lora_adapter = _get_lora_config()
    if record.lora_adapter:
        logger.info("[checkpoint] LoRA adapter captured")
    else:
        logger.debug("[checkpoint] No LoRA adapter active")

    # 3. Capture system prompt
    if SYSTEM_PROMPT_PATH.exists():
        prompt_text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        record.system_prompt_hash = _hash_file(SYSTEM_PROMPT_PATH)
        record.system_prompt_snapshot = prompt_text
        _save_system_prompt_snapshot(record.id, prompt_text)
        logger.info("[checkpoint] system prompt captured (sha256=%s)", record.system_prompt_hash[:16])
    else:
        logger.debug("[checkpoint] system prompt file not found: %s", SYSTEM_PROMPT_PATH)

    # 4. Extension: real weight snapshot
    snap_dir = _snapshot_dir(record.id)
    record.weight_snapshot_path = _capture_weight_state(snap_dir)

    # 5. Persist record
    _record_path(record.id).write_text(
        json.dumps(asdict(record), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("[checkpoint] Created checkpoint id=%s label=%s", record.id, record.label)
    return record


def restore_checkpoint(checkpoint_id: str) -> CheckpointRecord:
    """
    Roll back server state to a previously saved checkpoint.
    Raises FileNotFoundError if checkpoint does not exist.
    Raises RuntimeError on partial/failed restore.
    """
    rec_path = _record_path(checkpoint_id)
    if not rec_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

    data = json.loads(rec_path.read_text(encoding="utf-8"))
    record = CheckpointRecord(**data)
    logger.info("[checkpoint] Restoring checkpoint id=%s label=%s", record.id, record.label)

    errors: List[str] = []

    # 1. Restore system prompt
    if record.system_prompt_snapshot:
        try:
            SYSTEM_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            SYSTEM_PROMPT_PATH.write_text(record.system_prompt_snapshot, encoding="utf-8")
            logger.info("[checkpoint] System prompt restored")
        except Exception as exc:
            msg = f"Failed to restore system prompt: {exc}"
            logger.error("[checkpoint] %s", msg)
            errors.append(msg)

    # 2. Restore LoRA adapter config (if any)
    if record.lora_adapter:
        lora_cfg_path_str = os.environ.get("CURIOSITY_LORA_CONFIG", "").strip()
        lora_cfg_path = Path(lora_cfg_path_str) if lora_cfg_path_str and Path(lora_cfg_path_str).name else None
        if lora_cfg_path:
            try:
                lora_cfg_path.write_text(json.dumps(record.lora_adapter, indent=2), encoding="utf-8")
                logger.info("[checkpoint] LoRA adapter config restored")
            except Exception as exc:
                msg = f"Failed to restore LoRA config: {exc}"
                logger.error("[checkpoint] %s", msg)
                errors.append(msg)

    # 3. Restore weight snapshot (extension point)
    if record.weight_snapshot_path:
        try:
            _restore_weight_state(record.weight_snapshot_path)
            logger.info("[checkpoint] Weight state restored")
        except Exception as exc:
            msg = f"Failed to restore weight state: {exc}"
            logger.error("[checkpoint] %s", msg)
            errors.append(msg)

    if errors:
        raise RuntimeError(f"Checkpoint restore had {len(errors)} error(s): {'; '.join(errors)}")

    logger.info("[checkpoint] Restore complete for checkpoint id=%s", checkpoint_id)
    return record


def list_checkpoints() -> List[CheckpointRecord]:
    """
    Return all saved checkpoints, sorted newest-first.
    """
    cp_dir = _checkpoint_dir()
    records: List[CheckpointRecord] = []
    for p in sorted(cp_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            records.append(CheckpointRecord(**data))
        except Exception as exc:
            logger.warning("[checkpoint] Could not load %s: %s", p, exc)
    return records


def delete_checkpoint(checkpoint_id: str) -> None:
    """Remove a checkpoint record and its snapshot directory."""
    rec_path = _record_path(checkpoint_id)
    if rec_path.exists():
        rec_path.unlink()
    snap_dir = _checkpoint_dir() / checkpoint_id
    if snap_dir.exists():
        shutil.rmtree(snap_dir, ignore_errors=True)
    logger.info("[checkpoint] Deleted checkpoint id=%s", checkpoint_id)


# ---------------------------------------------------------------------------
# Convenience module-level wrappers (match __init__ exports)
# ---------------------------------------------------------------------------

_default_manager: Optional["CheckpointManager"] = None


class CheckpointManager:
    """Thin stateless wrapper around module-level functions, for DI convenience."""

    def create(self, label: str = "", note: str = "") -> CheckpointRecord:
        return create_checkpoint(label=label, note=note)

    def restore(self, checkpoint_id: str) -> CheckpointRecord:
        return restore_checkpoint(checkpoint_id)

    def list(self) -> List[CheckpointRecord]:
        return list_checkpoints()

    def delete(self, checkpoint_id: str) -> None:
        delete_checkpoint(checkpoint_id)
