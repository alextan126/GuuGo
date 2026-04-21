"""PyTorch policy-value network used by both self-play and the trainer.

Architecture (deliberately small for an MVP that can still train on CPU):

    Input (C, 9, 9)                  C = 3 feature planes
    Conv3x3 -> BN -> ReLU            in_channels -> num_channels
    [ResBlock] x num_res_blocks
        conv3x3 -> BN -> ReLU
        conv3x3 -> BN
        + skip
        ReLU
    Two heads on the shared trunk:
        Policy head: Conv1x1 (2 ch) -> FC -> 82 logits
        Value head:  Conv1x1 (1 ch) -> FC (num_channels) -> FC(1) -> tanh

The loss is the standard AlphaZero combination:
    loss = -sum(pi * log_softmax(policy_logits)) + w_v * MSE(value, z)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AlphaZeroConfig


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class PolicyValueNet(nn.Module):
    """Shared trunk with policy and value heads for a 9x9 Go board."""

    def __init__(self, config: AlphaZeroConfig) -> None:
        super().__init__()
        self.config = config
        c = config.num_channels
        size = config.board_size
        action_size = config.action_size

        self.stem = nn.Sequential(
            nn.Conv2d(config.input_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[_ResBlock(c) for _ in range(config.num_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(c, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * size * size, action_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(size * size, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.stem(x))
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return policy_logits, value


@dataclass
class LossStats:
    total: float
    policy: float
    value: float


def compute_loss(
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    value_weight: float = 1.0,
) -> Tuple[torch.Tensor, LossStats]:
    """AlphaZero loss: cross-entropy on policy + MSE on value.

    ``policy_target`` is a distribution over actions (sums to 1).
    ``value_target`` is a scalar in ``[-1, 1]`` per example.
    """

    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value_pred, value_target)
    total = policy_loss + value_weight * value_loss
    stats = LossStats(
        total=float(total.detach().item()),
        policy=float(policy_loss.detach().item()),
        value=float(value_loss.detach().item()),
    )
    return total, stats
