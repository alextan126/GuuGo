"""Bounded FIFO replay buffer for AlphaZero training.

The buffer lives in RAM while the trainer runs. It is fed in two ways:

* Directly, via :meth:`add_examples` (useful for single-process loops).
* From disk, via :meth:`ingest_game_files` which scans
  ``config.replay_dir`` for new ``game_*.pkl`` files written by the
  self-play workers and appends their contents.

Sampling is uniform random. Symmetry augmentation is already applied at
example-generation time in :mod:`alphazero.self_play` (so the buffer
doesn't have to reason about it), but :func:`augment_inline` is provided
for callers that want to augment at sample time instead.
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Set, Tuple

import numpy as np

from .self_play import Example


class ReplayBuffer:
    """FIFO buffer of fixed capacity with uniform random sampling."""

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self._buffer: Deque[Example] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)
        self._ingested_files: Set[str] = set()

    # ---------------- state ----------------

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def size(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self._buffer) >= min_size

    # ---------------- writes ----------------

    def add_examples(self, examples: Iterable[Example]) -> int:
        """Append examples, evicting oldest when over capacity."""

        count = 0
        for ex in examples:
            self._buffer.append(ex)
            count += 1
        return count

    def ingest_game_files(self, replay_dir: Path) -> int:
        """Load new ``game_*.pkl`` files from ``replay_dir``.

        Returns the number of examples added. Files already ingested are
        skipped (tracked by filename). Corrupt / partial files are left
        in place so the author can inspect them.
        """

        replay_dir = Path(replay_dir)
        if not replay_dir.is_dir():
            return 0
        added = 0
        for path in sorted(replay_dir.glob("game_*.pkl")):
            name = path.name
            if name in self._ingested_files:
                continue
            try:
                with open(path, "rb") as f:
                    examples = pickle.load(f)
            except (EOFError, pickle.UnpicklingError, OSError):
                continue
            self._ingested_files.add(name)
            added += self.add_examples(examples)
        return added

    # ---------------- sampling ----------------

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of examples as stacked numpy arrays.

        Returns ``(states, policies, values)`` with shapes
        ``(B, C, 9, 9)``, ``(B, action_size)``, ``(B,)`` respectively.
        """

        if len(self._buffer) == 0:
            raise RuntimeError("cannot sample from empty replay buffer")
        batch_size = min(batch_size, len(self._buffer))
        idx = self._rng.integers(0, len(self._buffer), size=batch_size)

        states = np.empty(
            (batch_size,) + self._buffer[0].state.shape, dtype=np.float32
        )
        policies = np.empty(
            (batch_size, self._buffer[0].policy.shape[0]), dtype=np.float32
        )
        values = np.empty((batch_size,), dtype=np.float32)

        for out_i, buf_i in enumerate(idx):
            ex = self._buffer[int(buf_i)]
            states[out_i] = ex.state
            policies[out_i] = ex.policy
            values[out_i] = ex.value

        return states, policies, values
