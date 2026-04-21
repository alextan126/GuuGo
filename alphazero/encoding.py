"""State encoding, move indexing, and symmetry utilities.

A 9x9 board has 81 intersections plus a pass move, giving an
82-dimensional action space. We keep the mapping trivial so it is easy to
reason about during debugging:

    action = row * 9 + col       for board points
    action = 81                  for pass (ACTION_PASS)

The state encoder produces 3 planes of shape (9, 9): current player's
stones, opponent's stones, and a constant turn-indicator plane. This is
deliberately minimal for the MVP; adding history or liberty planes later
is a matter of extending :func:`encode_engine` without touching MCTS or
the trainer.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from go_game.engine import GameEngine
from go_game.types import Color, Point

BOARD_SIZE = 9
NUM_BOARD_POINTS = BOARD_SIZE * BOARD_SIZE
ACTION_PASS = NUM_BOARD_POINTS  # 81
ACTION_SIZE = NUM_BOARD_POINTS + 1  # 82

# Number of D4 symmetries for a square board.
NUM_SYMMETRIES = 8


def policy_size() -> int:
    return ACTION_SIZE


# ---------------- action <-> point ----------------


def point_to_action(point: Point) -> int:
    r, c = point
    return r * BOARD_SIZE + c


def action_to_point(action: int) -> Optional[Point]:
    """Return ``None`` for the pass action, else a ``(row, col)`` tuple."""

    if action == ACTION_PASS:
        return None
    return (action // BOARD_SIZE, action % BOARD_SIZE)


# ---------------- state encoder ----------------


def encode_engine(engine: GameEngine) -> np.ndarray:
    """Encode the current engine state from the side-to-move perspective.

    Shape: ``(C, 9, 9)`` with ``C = 3``. Always float32 so PyTorch can
    consume it without casts.
    """

    size = engine.size
    assert size == BOARD_SIZE, "encoder currently assumes 9x9"
    current = engine.current_player
    opponent = current.opponent()
    grid = engine.board_grid

    current_plane = np.zeros((size, size), dtype=np.float32)
    opponent_plane = np.zeros((size, size), dtype=np.float32)
    for r in range(size):
        row = grid[r]
        for c in range(size):
            v = row[c]
            if v is current:
                current_plane[r, c] = 1.0
            elif v is opponent:
                opponent_plane[r, c] = 1.0

    turn_plane = np.full(
        (size, size),
        1.0 if current is Color.BLACK else 0.0,
        dtype=np.float32,
    )

    return np.stack([current_plane, opponent_plane, turn_plane], axis=0)


def legal_action_mask(engine: GameEngine) -> np.ndarray:
    """Return a float32 mask of shape ``(ACTION_SIZE,)``.

    ``1.0`` for legal actions, ``0.0`` otherwise. The pass action is
    always marked legal, because the engine always accepts a pass (even
    though it concedes the game).
    """

    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for point in engine.iter_legal_points():
        mask[point_to_action(point)] = 1.0
    mask[ACTION_PASS] = 1.0
    return mask


# ---------------- symmetries ----------------


def _transform_plane(plane: np.ndarray, sym: int) -> np.ndarray:
    """Apply the ``sym``-th D4 transform to a single (9, 9) plane."""

    if sym < 0 or sym >= NUM_SYMMETRIES:
        raise ValueError(f"invalid symmetry index {sym}")
    arr = np.rot90(plane, k=sym % 4)
    if sym >= 4:
        arr = np.fliplr(arr)
    return np.ascontiguousarray(arr)


def transform_state(state: np.ndarray, sym: int) -> np.ndarray:
    """Apply a symmetry to a stacked state of shape ``(C, 9, 9)``."""

    return np.stack([_transform_plane(state[c], sym) for c in range(state.shape[0])], axis=0)


def _build_action_permutation(sym: int) -> np.ndarray:
    """Precompute ``perm[a]`` = image of action ``a`` under symmetry ``sym``."""

    perm = np.empty(ACTION_SIZE, dtype=np.int64)
    grid = np.arange(NUM_BOARD_POINTS, dtype=np.int64).reshape(BOARD_SIZE, BOARD_SIZE)
    transformed = _transform_plane(grid.astype(np.float32), sym).astype(np.int64)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            src_action = int(transformed[r, c])
            dst_action = r * BOARD_SIZE + c
            perm[src_action] = dst_action
    perm[ACTION_PASS] = ACTION_PASS
    return perm


_ACTION_PERMS: List[np.ndarray] = [_build_action_permutation(s) for s in range(NUM_SYMMETRIES)]


def transform_policy(policy: np.ndarray, sym: int) -> np.ndarray:
    """Apply a symmetry to a length-82 policy vector."""

    perm = _ACTION_PERMS[sym]
    out = np.empty_like(policy)
    out[perm] = policy
    return out


def augment(state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return the 8 D4-augmented views of a single training example.

    The value target ``z`` is symmetry-invariant, so the caller keeps the
    same ``z`` for every augmented copy.
    """

    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for sym in range(NUM_SYMMETRIES):
        out.append((transform_state(state, sym), transform_policy(policy, sym)))
    return out
