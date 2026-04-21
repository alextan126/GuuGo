"""Smoke tests for the PyTorch model and trainer batch plumbing.

These tests skip if ``torch`` is not importable so the pure-engine test
suite can still run on a machine without the training deps installed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from alphazero.config import AlphaZeroConfig  # noqa: E402
from alphazero.encoding import ACTION_SIZE  # noqa: E402
from alphazero.model import PolicyValueNet, compute_loss  # noqa: E402
from alphazero.replay_buffer import ReplayBuffer  # noqa: E402
from alphazero.self_play import Example  # noqa: E402
from alphazero.trainer import Trainer  # noqa: E402


def _tiny_config(tmp_path) -> AlphaZeroConfig:
    cfg = AlphaZeroConfig(
        num_res_blocks=1,
        num_channels=16,
        batch_size=8,
        checkpoint_every_steps=10**9,  # never in these tests
        device="cpu",
    )
    cfg.checkpoint_dir = tmp_path / "ckpt"
    cfg.replay_dir = tmp_path / "replay"
    cfg.ensure_dirs()
    return cfg


def _random_example() -> Example:
    rng = np.random.default_rng()
    pi = rng.dirichlet(np.ones(ACTION_SIZE)).astype(np.float32)
    return Example(
        state=rng.random((3, 9, 9), dtype=np.float32),
        policy=pi,
        value=float(rng.uniform(-1, 1)),
    )


def test_forward_shapes():
    cfg = AlphaZeroConfig(num_res_blocks=1, num_channels=16)
    model = PolicyValueNet(cfg)
    x = torch.randn(4, 3, 9, 9)
    logits, value = model(x)
    assert logits.shape == (4, ACTION_SIZE)
    assert value.shape == (4,)


def test_compute_loss_gradients_flow():
    cfg = AlphaZeroConfig(num_res_blocks=1, num_channels=16)
    model = PolicyValueNet(cfg)
    x = torch.randn(4, 3, 9, 9)
    policy_target = torch.softmax(torch.randn(4, ACTION_SIZE), dim=-1)
    value_target = torch.tanh(torch.randn(4))
    logits, value = model(x)
    loss, stats = compute_loss(logits, value, policy_target, value_target)
    loss.backward()
    # At least one parameter must have non-zero gradient.
    has_grad = any(
        p.grad is not None and torch.any(p.grad != 0).item() for p in model.parameters()
    )
    assert has_grad
    assert stats.total >= 0


def test_trainer_step_updates_weights_and_saves(tmp_path):
    cfg = _tiny_config(tmp_path)
    cfg.min_replay_to_train = 4
    buf = ReplayBuffer(capacity=64)
    for _ in range(16):
        buf.add_examples([_random_example()])

    trainer = Trainer(cfg, replay_buffer=buf)
    before = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
    stats = trainer.step()
    after = trainer.model.state_dict()

    # At least one tensor changed.
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert stats.total >= 0

    path = trainer.save_checkpoint(notes="unit-test")
    assert path.is_file()
    # Latest pointer is written too.
    assert (cfg.checkpoint_dir / "latest.pt").is_file()
