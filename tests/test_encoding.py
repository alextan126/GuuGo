"""Tests for the AlphaZero state encoder and symmetry helpers."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero.encoding import (  # noqa: E402
    ACTION_PASS,
    ACTION_SIZE,
    BOARD_SIZE,
    action_to_point,
    augment,
    encode_engine,
    legal_action_mask,
    point_to_action,
    transform_policy,
    transform_state,
)
from go_game import GameEngine  # noqa: E402


def test_action_point_roundtrip():
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            a = point_to_action((r, c))
            assert 0 <= a < ACTION_PASS
            assert action_to_point(a) == (r, c)
    assert action_to_point(ACTION_PASS) is None


def test_encode_engine_shape_and_channels():
    engine = GameEngine()
    state = encode_engine(engine)
    assert state.shape == (3, 9, 9)
    assert state.dtype == np.float32
    # Empty board -> no current / opponent stones, black to move.
    assert state[0].sum() == 0
    assert state[1].sum() == 0
    assert state[2].sum() == 9 * 9  # turn plane = 1 for black


def test_encode_engine_swaps_planes_for_side_to_move():
    engine = GameEngine()
    engine.play((4, 4))  # black
    state = encode_engine(engine)
    # Now white to move: current-player plane should have 0 stones,
    # opponent plane should have the black stone.
    assert state[0].sum() == 0
    assert state[1, 4, 4] == 1.0
    assert state[1].sum() == 1
    assert state[2].sum() == 0  # turn plane = 0 for white


def test_legal_action_mask_on_empty_board():
    mask = legal_action_mask(GameEngine())
    assert mask.shape == (ACTION_SIZE,)
    # All 81 points plus pass are legal on an empty board.
    assert mask.sum() == ACTION_SIZE


def test_symmetry_identity_preserves_state_and_policy():
    engine = GameEngine()
    engine.play((2, 3))
    state = encode_engine(engine)
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[point_to_action((4, 4))] = 1.0
    s0 = transform_state(state, 0)
    p0 = transform_policy(policy, 0)
    assert np.array_equal(s0, state)
    assert np.array_equal(p0, policy)


def test_symmetry_pass_action_invariant():
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[ACTION_PASS] = 1.0
    for sym in range(8):
        out = transform_policy(policy, sym)
        assert out[ACTION_PASS] == pytest.approx(1.0)
        assert out[:ACTION_PASS].sum() == pytest.approx(0.0)


def test_augment_yields_eight_variants():
    engine = GameEngine()
    engine.play((0, 0))
    state = encode_engine(engine)
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[point_to_action((1, 2))] = 1.0
    outs = augment(state, policy)
    assert len(outs) == 8
    for s, p in outs:
        assert s.shape == state.shape
        assert p.shape == policy.shape
        # Each augmented policy is still a valid one-hot.
        assert p.sum() == pytest.approx(1.0)


def test_symmetry_consistency_between_state_and_policy():
    """Encoding a stone at (r,c) and transforming should equal encoding
    the rotated board directly, for every symmetry."""

    from alphazero.encoding import _transform_plane

    engine = GameEngine()
    engine.play((1, 2))
    state = encode_engine(engine)
    # Build a one-hot policy pointing at the stone location.
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[point_to_action((1, 2))] = 1.0
    for sym in range(8):
        s = transform_state(state, sym)
        p = transform_policy(policy, sym)
        # The stone location in the transformed state must match the
        # index where the transformed policy is 1.
        opp_plane = s[1]  # opponent = black after black moves and white-to-move
        rs, cs = np.where(opp_plane > 0.5)
        assert len(rs) == 1
        idx = int(rs[0]) * BOARD_SIZE + int(cs[0])
        assert p[idx] == pytest.approx(1.0)
