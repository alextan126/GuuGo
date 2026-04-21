"""Trainer: sample from the replay buffer and update the network.

Trainer is decoupled from self-play through two on-disk artifacts:

    checkpoint_dir/latest.pt   -- what self-play workers consume
    replay_dir/game_*.pkl      -- what self-play workers produce

A single :meth:`Trainer.step` pulls one batch, computes the AlphaZero
loss, steps the optimizer, and periodically writes a new checkpoint.
``Trainer.run_forever`` loops that until interrupted; ``train_iteration``
is a convenient building block for single-process pipelines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim

from .checkpoints import load_latest_checkpoint, resolve_device, save_checkpoint
from .config import AlphaZeroConfig
from .model import LossStats, PolicyValueNet, compute_loss
from .replay_buffer import ReplayBuffer


@dataclass
class TrainerMetrics:
    step: int = 0
    loss_total: float = 0.0
    loss_policy: float = 0.0
    loss_value: float = 0.0
    loss_history: List[float] = field(default_factory=list)

    def record(self, stats: LossStats) -> None:
        self.step += 1
        self.loss_total = stats.total
        self.loss_policy = stats.policy
        self.loss_value = stats.value
        self.loss_history.append(stats.total)


class Trainer:
    def __init__(
        self,
        config: AlphaZeroConfig,
        replay_buffer: Optional[ReplayBuffer] = None,
        model: Optional[PolicyValueNet] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device if device is not None else resolve_device(config.device)
        self.model = model if model is not None else PolicyValueNet(config).to(self.device)
        # Use explicit ``is None`` rather than ``or`` because an empty
        # ``ReplayBuffer`` is falsy via ``__len__``, which would silently
        # replace the caller's buffer with a fresh one.
        self.replay_buffer = (
            replay_buffer if replay_buffer is not None else ReplayBuffer(config.replay_capacity)
        )

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )

        self.metrics = TrainerMetrics()
        self._loaded = False

    # ---------------- checkpoints ----------------

    def resume_from_checkpoint(self) -> int:
        """Load latest checkpoint into model + optimizer if present.

        Returns the step number loaded, or 0 if none.
        """

        step = load_latest_checkpoint(
            self.config.checkpoint_dir,
            self.model,
            optimizer=self.optimizer,
            map_location=str(self.device),
        )
        if step is not None:
            self.metrics.step = step
            self._loaded = True
            return step
        return 0

    def save_checkpoint(self, notes: str = "") -> Path:
        self.config.ensure_dirs()
        return save_checkpoint(
            self.config.checkpoint_dir,
            self.model,
            self.optimizer,
            step=self.metrics.step,
            timestamp=time.time(),
            notes=notes,
        )

    # ---------------- training ----------------

    def step(self) -> LossStats:
        """Run a single gradient step on a sampled batch."""

        states, policies, values = self.replay_buffer.sample(self.config.batch_size)
        state_t = torch.from_numpy(states).to(self.device)
        policy_t = torch.from_numpy(policies).to(self.device)
        value_t = torch.from_numpy(values).to(self.device)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits, value_pred = self.model(state_t)
        loss, stats = compute_loss(
            logits,
            value_pred,
            policy_t,
            value_t,
            value_weight=self.config.value_loss_weight,
        )
        loss.backward()
        self.optimizer.step()

        self.metrics.record(stats)
        if self.metrics.step % self.config.checkpoint_every_steps == 0:
            self.save_checkpoint()
        return stats

    def train_iteration(
        self,
        num_steps: int,
        min_buffer: Optional[int] = None,
    ) -> List[LossStats]:
        """Train for ``num_steps`` batches if the buffer has enough data."""

        min_buffer = (
            min_buffer if min_buffer is not None else self.config.min_replay_to_train
        )
        if not self.replay_buffer.is_ready(min_buffer):
            return []

        stats: List[LossStats] = []
        for _ in range(num_steps):
            stats.append(self.step())
        return stats

    # ---------------- standalone loop ----------------

    def run_forever(
        self,
        steps_per_cycle: int = 100,
        ingest_each_cycle: bool = True,
    ) -> None:
        """Long-running trainer loop used by ``scripts/train.py``.

        Keeps polling ``replay_dir`` for new self-play game files, adds
        them to the buffer, and trains. Saves a checkpoint on schedule so
        self-play workers pick up new weights.
        """

        self.resume_from_checkpoint()
        cycle = 0
        while True:
            if ingest_each_cycle:
                added = self.replay_buffer.ingest_game_files(self.config.replay_dir)
                if added:
                    print(
                        f"[trainer] cycle {cycle}: ingested {added} examples; "
                        f"buffer size={len(self.replay_buffer)}"
                    )
            if not self.replay_buffer.is_ready(self.config.min_replay_to_train):
                print(
                    f"[trainer] waiting for data: have {len(self.replay_buffer)} / "
                    f"{self.config.min_replay_to_train}"
                )
                time.sleep(2.0)
                cycle += 1
                continue
            stats = self.train_iteration(steps_per_cycle)
            if stats:
                last = stats[-1]
                print(
                    f"[trainer] step={self.metrics.step} "
                    f"loss={last.total:.4f} policy={last.policy:.4f} "
                    f"value={last.value:.4f} buffer={len(self.replay_buffer)}"
                )
            cycle += 1
