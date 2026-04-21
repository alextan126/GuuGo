"""Tests for the replay buffer: capacity, sampling shapes, file ingestion."""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero.encoding import ACTION_SIZE  # noqa: E402
from alphazero.replay_buffer import ReplayBuffer  # noqa: E402
from alphazero.self_play import Example  # noqa: E402


def _make_example(seed: int) -> Example:
    rng = np.random.default_rng(seed)
    return Example(
        state=rng.random((3, 9, 9), dtype=np.float32),
        policy=rng.dirichlet(np.ones(ACTION_SIZE)).astype(np.float32),
        value=float(rng.uniform(-1, 1)),
    )


def test_bounded_capacity_evicts_oldest():
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.add_examples([_make_example(i)])
    assert len(buf) == 5


def test_sample_shapes_and_dtypes():
    buf = ReplayBuffer(capacity=100)
    buf.add_examples(_make_example(i) for i in range(32))
    states, policies, values = buf.sample(16)
    assert states.shape == (16, 3, 9, 9)
    assert policies.shape == (16, ACTION_SIZE)
    assert values.shape == (16,)
    assert states.dtype == np.float32
    assert policies.dtype == np.float32
    assert values.dtype == np.float32


def test_ingest_game_files(tmp_path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    # Write two pickled game files.
    for i in range(2):
        with open(replay_dir / f"game_{i:07d}_abc{i}.pkl", "wb") as f:
            pickle.dump([_make_example(i), _make_example(i + 100)], f)

    buf = ReplayBuffer(capacity=100)
    added = buf.ingest_game_files(replay_dir)
    assert added == 4
    assert len(buf) == 4

    # Re-ingest should be a no-op.
    added_again = buf.ingest_game_files(replay_dir)
    assert added_again == 0
    assert len(buf) == 4


def test_sample_from_empty_raises():
    buf = ReplayBuffer(capacity=10)
    with pytest.raises(RuntimeError):
        buf.sample(4)


def test_is_ready_threshold():
    buf = ReplayBuffer(capacity=50)
    assert not buf.is_ready(1)
    buf.add_examples([_make_example(0)])
    assert buf.is_ready(1)
    assert not buf.is_ready(2)
