"""Checkpoint I/O used by the trainer (writer) and self-play (reader).

Checkpoints are the only synchronization point between the two processes:
the trainer writes ``step_<n>.pt`` and updates ``latest.pt``; workers poll
``latest.pt`` and reload weights without any IPC.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

LATEST_FILENAME = "latest.pt"
META_FILENAME = "latest.json"


@dataclass
class CheckpointMeta:
    step: int
    timestamp: float
    notes: str = ""


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    timestamp: float,
    notes: str = "",
) -> Path:
    """Atomically write ``step_<n>.pt`` and update ``latest.pt``.

    Atomic means: write to a temp file first, then rename, so a reader that
    happens to observe the latest file mid-write never loads a half-written
    blob.
    """

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }

    numbered = checkpoint_dir / f"step_{step:07d}.pt"
    latest = checkpoint_dir / LATEST_FILENAME
    tmp = checkpoint_dir / f".{LATEST_FILENAME}.tmp"

    torch.save(payload, numbered)
    torch.save(payload, tmp)
    os.replace(tmp, latest)

    meta = CheckpointMeta(step=step, timestamp=timestamp, notes=notes)
    meta_path = checkpoint_dir / META_FILENAME
    meta_tmp = checkpoint_dir / f".{META_FILENAME}.tmp"
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f)
    os.replace(meta_tmp, meta_path)

    return numbered


def load_latest_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> Optional[int]:
    """Load latest weights into ``model`` (and optimizer) if available.

    Returns the step number loaded, or ``None`` if no checkpoint exists.
    """

    latest = Path(checkpoint_dir) / LATEST_FILENAME
    if not latest.is_file():
        return None
    payload = torch.load(latest, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload.get("step", 0))


def read_latest_meta(checkpoint_dir: Path) -> Optional[CheckpointMeta]:
    meta_path = Path(checkpoint_dir) / META_FILENAME
    if not meta_path.is_file():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CheckpointMeta(**data)


def resolve_device(device: str) -> torch.device:
    """Resolve the ``"auto"`` placeholder to the best device on this box."""

    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
