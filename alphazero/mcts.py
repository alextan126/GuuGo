"""Monte Carlo Tree Search with PUCT selection.

This is a classic AlphaZero-style MCTS. Every node stores

    N -- visit count
    W -- total action-value from the perspective of the player to move
         at the *parent* (i.e. the player who chose this action)
    Q = W / N
    P -- prior probability coming from the network
    children -- dict of action_index -> Node

One simulation walks from the root following the PUCT-best child until it
reaches either a terminal state or an unexpanded leaf, where the leaf is
expanded by one network inference call. The returned value is then
propagated back up, flipping sign every step because the player alternates.

The search is intentionally single-threaded and Python-object based. For a
9x9 board with ~100 simulations per move this is fast enough for MVP
training, and it is much easier to reason about than a batched MCTS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np

from go_game.engine import GameEngine
from go_game.types import Color

from .encoding import ACTION_PASS, action_to_point, legal_action_mask


class PolicyValueFn(Protocol):
    """Callable contract used by MCTS for network inference.

    Returns ``(prior, value)`` where ``prior`` is a length-``action_size``
    probability distribution and ``value`` is a scalar in ``[-1, 1]``
    representing the expected result from the side-to-move perspective.
    """

    def __call__(self, engine: GameEngine) -> Tuple[np.ndarray, float]:
        ...


@dataclass
class Node:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value: float = 0.0

    @property
    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _puct_score(parent_visits: int, child: Node, c_puct: float) -> float:
    """PUCT: Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))."""

    u = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
    # For an unvisited child Q is 0; small FPU (first-play urgency) would
    # go here, but zero is fine for an MVP.
    return child.q + u


class MCTS:
    """Stateless MCTS runner: takes a root engine and returns a policy.

    ``MCTS`` objects don't hold persistent state across moves. Each call
    to :meth:`search` builds a fresh tree rooted at the given engine.
    This is simpler and matches what single-game self-play needs; reusing
    the subtree between moves is a future optimization.
    """

    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        num_simulations: int,
        c_puct: float,
        action_size: int,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.policy_value_fn = policy_value_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.action_size = action_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = rng if rng is not None else np.random.default_rng()

    # ---------------- public ----------------

    def search(self, root_engine: GameEngine, add_root_noise: bool = True) -> Node:
        """Run ``num_simulations`` simulations and return the root node."""

        root = Node()
        self._expand(root, root_engine, add_root_noise=add_root_noise)

        for _ in range(self.num_simulations):
            self._simulate(root, root_engine)

        return root

    def visit_policy(
        self, root: Node, temperature: float = 1.0
    ) -> np.ndarray:
        """Return a policy distribution ``pi`` derived from root visit counts."""

        pi = np.zeros(self.action_size, dtype=np.float32)
        for action, child in root.children.items():
            pi[action] = child.visit_count
        total = pi.sum()
        if total <= 0:
            # Fallback: uniform over expanded children (shouldn't happen
            # after a successful search, but makes us robust).
            if root.children:
                for action in root.children:
                    pi[action] = 1.0
                pi /= pi.sum()
            return pi

        if temperature <= 1e-3:
            # Argmax with ties broken uniformly.
            best = np.flatnonzero(pi == pi.max())
            out = np.zeros_like(pi)
            out[best] = 1.0 / len(best)
            return out

        pi = pi ** (1.0 / temperature)
        pi /= pi.sum()
        return pi

    # ---------------- internals ----------------

    def _simulate(self, root: Node, root_engine: GameEngine) -> None:
        engine = root_engine.clone()
        node = root
        path: List[Tuple[Node, int]] = []  # (parent, action_taken)

        # ``side_to_move`` is our MCTS-internal view of whose turn it is,
        # which always flips after each action. We track it explicitly
        # instead of reading ``engine.current_player`` because the rules
        # engine freezes ``current_player`` once the game ends (for pass =
        # resign), which would confuse the sign of backup otherwise.
        side_to_move = engine.current_player

        # --- Selection ---
        while node.is_expanded and not node.is_terminal:
            action, child = self._select_child(node)
            path.append((node, action))
            self._apply_action(engine, action)
            side_to_move = side_to_move.opponent()
            node = child
            if engine.is_over:
                node.is_terminal = True
                break

        # --- Expansion / evaluation ---
        if node.is_terminal:
            leaf_value = engine.terminal_value(side_to_move)
        else:
            leaf_value = self._expand(node, engine, add_root_noise=False)

        # --- Backup ---
        # ``leaf_value`` is from the perspective of the side to move at
        # the leaf. Walking back up, each step is the parent's perspective,
        # which is the opponent's view, so we flip sign at every step.
        value = leaf_value
        for parent, action in reversed(path):
            value = -value
            child = parent.children[action]
            child.visit_count += 1
            child.value_sum += value
        # Root visit count tracks total sims; not used for selection but
        # useful for diagnostics.
        root.visit_count += 1

    def _select_child(self, node: Node) -> Tuple[int, Node]:
        best_score = -float("inf")
        best_action = -1
        best_child: Optional[Node] = None
        parent_visits = max(1, node.visit_count)
        for action, child in node.children.items():
            score = _puct_score(parent_visits, child, self.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        assert best_child is not None
        return best_action, best_child

    def _expand(
        self,
        node: Node,
        engine: GameEngine,
        add_root_noise: bool,
    ) -> float:
        """Populate children of ``node`` using the policy network.

        Returns the leaf value from the perspective of the side to move.
        """

        if engine.is_over:
            node.is_expanded = True
            node.is_terminal = True
            return engine.terminal_value(engine.current_player)

        prior, value = self.policy_value_fn(engine)
        legal_mask = legal_action_mask(engine)
        masked = prior * legal_mask
        total = masked.sum()
        if total <= 0:
            masked = legal_mask / max(1.0, legal_mask.sum())
        else:
            masked = masked / total

        if add_root_noise and self.dirichlet_alpha is not None and self.dirichlet_epsilon > 0:
            masked = self._mix_dirichlet_noise(masked, legal_mask)

        for action in range(self.action_size):
            if legal_mask[action] <= 0:
                continue
            node.children[action] = Node(prior=float(masked[action]))

        node.is_expanded = True
        return float(value)

    def _mix_dirichlet_noise(
        self, prior: np.ndarray, legal_mask: np.ndarray
    ) -> np.ndarray:
        """Mix Dirichlet noise into legal-action priors for root exploration."""

        legal_indices = np.flatnonzero(legal_mask > 0)
        if legal_indices.size == 0:
            return prior
        noise = self.rng.dirichlet([self.dirichlet_alpha] * legal_indices.size)
        out = prior.copy()
        out[legal_indices] = (
            (1 - self.dirichlet_epsilon) * prior[legal_indices]
            + self.dirichlet_epsilon * noise
        )
        # Re-normalize across legal actions.
        total = out[legal_indices].sum()
        if total > 0:
            out[legal_indices] /= total
        return out

    @staticmethod
    def _apply_action(engine: GameEngine, action: int) -> None:
        point = action_to_point(action)
        if point is None:
            engine.pass_turn()
        else:
            result = engine.play(point)
            if not result.legal:
                raise RuntimeError(
                    f"MCTS tried an illegal action {action}: {result.reason}"
                )
