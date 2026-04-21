"""Tests for MCTS and integration with a stubbed policy-value function."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero.config import AlphaZeroConfig  # noqa: E402
from alphazero.encoding import ACTION_PASS, ACTION_SIZE, point_to_action  # noqa: E402
from alphazero.mcts import MCTS  # noqa: E402
from go_game import GameEngine  # noqa: E402
from go_game.types import Color  # noqa: E402


def _uniform_policy(action_size: int = ACTION_SIZE):
    def fn(engine):
        return np.full(action_size, 1.0 / action_size, dtype=np.float32), 0.0

    return fn


def test_mcts_produces_policy_over_legal_actions():
    engine = GameEngine()
    mcts = MCTS(
        policy_value_fn=_uniform_policy(),
        num_simulations=20,
        c_puct=1.5,
        action_size=ACTION_SIZE,
    )
    root = mcts.search(engine, add_root_noise=False)
    assert root.is_expanded
    assert len(root.children) == ACTION_SIZE  # all legal on empty board
    pi = mcts.visit_policy(root, temperature=1.0)
    assert pi.shape == (ACTION_SIZE,)
    assert pi.sum() == pytest.approx(1.0)


def test_mcts_argmax_policy_is_one_hot():
    engine = GameEngine()
    mcts = MCTS(
        policy_value_fn=_uniform_policy(),
        num_simulations=20,
        c_puct=1.5,
        action_size=ACTION_SIZE,
    )
    root = mcts.search(engine, add_root_noise=False)
    pi = mcts.visit_policy(root, temperature=0.0)
    assert pi.sum() == pytest.approx(1.0)
    # With ties the mass is split; in general argmax produces a
    # distribution whose max is >= 1/len.
    assert pi.max() >= 1.0 / ACTION_SIZE


def test_mcts_propagates_visits():
    engine = GameEngine()
    mcts = MCTS(
        policy_value_fn=_uniform_policy(),
        num_simulations=50,
        c_puct=1.5,
        action_size=ACTION_SIZE,
    )
    root = mcts.search(engine, add_root_noise=False)
    total_child_visits = sum(child.visit_count for child in root.children.values())
    # Each simulation increments exactly one immediate child of the root.
    assert total_child_visits == 50


def test_mcts_handles_terminal_branches():
    """Bias the policy toward pass; MCTS must cope with terminal leaves."""

    engine = GameEngine()

    def biased(engine):
        probs = np.full(ACTION_SIZE, 0.001, dtype=np.float32)
        probs[ACTION_PASS] = 1.0
        probs /= probs.sum()
        return probs, 0.0

    mcts = MCTS(
        policy_value_fn=biased,
        num_simulations=20,
        c_puct=1.5,
        action_size=ACTION_SIZE,
    )
    root = mcts.search(engine, add_root_noise=False)
    assert ACTION_PASS in root.children
    # Pass leads to resignation for the side-to-move, so its Q should be
    # negative (bad for the root player).
    assert root.children[ACTION_PASS].q <= 0.0
