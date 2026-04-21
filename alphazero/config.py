"""Central configuration for the AlphaZero MVP pipeline.

All modules read knobs from :class:`AlphaZeroConfig` rather than hardcoding
constants, so experiments only touch one place and a future DGX/Blackwell
deployment can swap in a different config (larger batch, more simulations,
mixed precision, etc.) without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AlphaZeroConfig:
    # ---------- game ----------
    board_size: int = 9
    # 81 board points + 1 pass action. Pass is part of the action space
    # even though the current engine treats it as resignation, because the
    # policy head still needs a slot for it; the network will simply learn
    # a tiny probability for pass outside of lost positions.
    input_channels: int = 3

    # ---------- model ----------
    num_res_blocks: int = 3
    num_channels: int = 96

    # ---------- MCTS ----------
    num_simulations: int = 100
    c_puct: float = 1.5
    # Dirichlet noise mixed into the root prior for exploration.
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    # Sample moves according to visit distribution for the first
    # ``temperature_moves`` plies, then play argmax.
    temperature_moves: int = 15

    # ---------- self-play ----------
    # Hard cap to keep games finite even if MCTS keeps finding playable
    # moves; on reaching this the game is ended via Chinese area scoring.
    max_moves: int = 200
    # How often a self-play worker checks for a new checkpoint.
    checkpoint_poll_seconds: float = 5.0

    # ---------- replay buffer ----------
    replay_capacity: int = 20000
    min_replay_to_train: int = 256
    augment_with_symmetries: bool = True

    # ---------- trainer ----------
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    value_loss_weight: float = 1.0
    checkpoint_every_steps: int = 100
    log_every_steps: int = 20

    # ---------- paths ----------
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    replay_dir: Path = field(default_factory=lambda: Path("replay"))

    # ---------- device ----------
    # Leave as "auto" to pick cuda > mps > cpu at runtime.
    device: str = "auto"

    @property
    def num_board_points(self) -> int:
        return self.board_size * self.board_size

    @property
    def action_size(self) -> int:
        return self.num_board_points + 1  # + pass

    def ensure_dirs(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
