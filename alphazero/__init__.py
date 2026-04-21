"""AlphaZero-style training stack for 9x9 Go.

This package is independent from the pygame GUI: it uses only the rules
engine in :mod:`go_game` plus PyTorch. The three public components mirror
the architecture described in ``Architecture.md``:

- :mod:`alphazero.self_play` generates training examples from MCTS-guided
  self-play.
- :mod:`alphazero.replay_buffer` stores and samples those examples.
- :mod:`alphazero.trainer` consumes batches and publishes new weights.

They communicate through files on disk (checkpoints + replay dump) so the
processes can be started independently and scaled horizontally later.
"""

from .config import AlphaZeroConfig
from .encoding import (
    ACTION_PASS,
    action_to_point,
    encode_engine,
    point_to_action,
    policy_size,
)

__all__ = [
    "AlphaZeroConfig",
    "ACTION_PASS",
    "action_to_point",
    "encode_engine",
    "point_to_action",
    "policy_size",
]
