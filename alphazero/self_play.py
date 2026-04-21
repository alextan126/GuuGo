"""Self-play worker: generate (state, pi, z) examples via MCTS.

A worker loads the latest checkpoint, plays one complete game against
itself using MCTS on every move, then emits one example per move with the
training targets:

    pi  --  the MCTS root visit distribution at that move
    z   --  +1 if the player-to-move at that position eventually won
            the game, -1 if they lost, 0 for a draw.

The examples are optionally 8x-augmented by D4 symmetries before being
written to disk as a pickle under ``config.replay_dir``. The trainer
polls that directory for new games.
"""

from __future__ import annotations

import pickle
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from go_game.engine import GameEngine
from go_game.types import Color

from .checkpoints import load_latest_checkpoint, read_latest_meta, resolve_device
from .config import AlphaZeroConfig
from .encoding import ACTION_PASS, augment, encode_engine, policy_size
from .mcts import MCTS, PolicyValueFn
from .model import PolicyValueNet


@dataclass
class Example:
    """A single training example stored in the replay buffer."""

    state: np.ndarray         # shape (C, 9, 9) float32
    policy: np.ndarray        # shape (action_size,) float32, sums to 1
    value: float              # z in [-1, 1]


def _player_sign(color: Color) -> int:
    return 1 if color is Color.BLACK else -1


def make_torch_policy_value_fn(
    model: PolicyValueNet,
    device: torch.device,
) -> PolicyValueFn:
    """Wrap a PyTorch model as an MCTS-compatible ``PolicyValueFn``.

    The returned callable accepts a :class:`GameEngine`, encodes it, and
    returns ``(prior_probs, value)`` as numpy arrays / floats. Model must
    be in eval mode; caller is responsible for :func:`torch.no_grad`.
    """

    def fn(engine: GameEngine) -> Tuple[np.ndarray, float]:
        state = encode_engine(engine)
        tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = model(tensor)
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy(), float(value.item())

    return fn


class SelfPlayWorker:
    """Generates self-play games using the latest network checkpoint."""

    def __init__(
        self,
        config: AlphaZeroConfig,
        model: Optional[PolicyValueNet] = None,
        device: Optional[torch.device] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config
        self.device = device if device is not None else resolve_device(config.device)
        self.model = model if model is not None else PolicyValueNet(config).to(self.device)
        self.model.eval()
        self.rng = rng if rng is not None else np.random.default_rng()
        self._loaded_step: Optional[int] = None
        self._pv_fn: PolicyValueFn = make_torch_policy_value_fn(self.model, self.device)

    # ---------------- checkpoint sync ----------------

    def maybe_reload_weights(self) -> Optional[int]:
        """Reload ``latest.pt`` if it has advanced since the last load.

        Returns the step number if weights were reloaded, else ``None``.
        """

        meta = read_latest_meta(self.config.checkpoint_dir)
        if meta is None:
            return None
        if self._loaded_step is not None and meta.step <= self._loaded_step:
            return None
        step = load_latest_checkpoint(
            self.config.checkpoint_dir,
            self.model,
            optimizer=None,
            map_location=str(self.device),
        )
        if step is None:
            return None
        self._loaded_step = step
        self.model.eval()
        return step

    # ---------------- single game ----------------

    def generate_game(self) -> List[Example]:
        """Play a single self-play game and return augmented examples."""

        engine = GameEngine(self.config.board_size)
        trajectory: List[Tuple[np.ndarray, np.ndarray, Color]] = []

        mcts = MCTS(
            policy_value_fn=self._pv_fn,
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            action_size=self.config.action_size,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            rng=self.rng,
        )

        move_idx = 0
        while not engine.is_over and move_idx < self.config.max_moves:
            state = encode_engine(engine)
            root = mcts.search(engine, add_root_noise=True)

            temperature = 1.0 if move_idx < self.config.temperature_moves else 0.0
            pi = mcts.visit_policy(root, temperature=temperature)

            trajectory.append((state, pi.astype(np.float32), engine.current_player))

            action = self._sample_action(pi, temperature=temperature)
            self._apply_action_to_engine(engine, action)
            move_idx += 1

        # Reach a definite winner if the cap ended the game still-open.
        if not engine.is_over:
            engine.finish_by_score()
        assert engine.result is not None

        winner = engine.result.winner
        examples: List[Example] = []
        for state, pi, to_move in trajectory:
            if winner is None:
                z = 0.0
            else:
                z = 1.0 if winner is to_move else -1.0
            if self.config.augment_with_symmetries:
                for aug_state, aug_pi in augment(state, pi):
                    examples.append(Example(aug_state, aug_pi, z))
            else:
                examples.append(Example(state, pi, z))
        return examples

    # ---------------- I/O ----------------

    def write_game_file(self, examples: List[Example], step_hint: Optional[int]) -> Path:
        """Persist a completed game's examples to ``replay_dir``."""

        self.config.ensure_dirs()
        game_id = uuid.uuid4().hex[:10]
        step_tag = step_hint if step_hint is not None else 0
        path = self.config.replay_dir / f"game_{step_tag:07d}_{game_id}.pkl"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(path)
        return path

    def run_forever(self, max_games: Optional[int] = None) -> None:
        """Long-running worker loop used by ``scripts/self_play.py``.

        ``max_games`` caps total games (useful for tests / smoke runs);
        ``None`` means run forever.
        """

        games_played = 0
        while True:
            reloaded = self.maybe_reload_weights()
            if reloaded is not None:
                print(f"[self-play] reloaded checkpoint step={reloaded}")
            examples = self.generate_game()
            self.write_game_file(examples, step_hint=self._loaded_step)
            games_played += 1
            print(f"[self-play] game {games_played}: {len(examples)} examples written")
            if max_games is not None and games_played >= max_games:
                return
            time.sleep(0)  # yield

    # ---------------- helpers ----------------

    def _sample_action(self, pi: np.ndarray, temperature: float) -> int:
        if temperature <= 1e-3:
            # argmax with deterministic tie-break
            return int(np.argmax(pi))
        # pi already has temperature applied by mcts.visit_policy
        probs = pi / pi.sum()
        return int(self.rng.choice(len(probs), p=probs))

    def _apply_action_to_engine(self, engine: GameEngine, action: int) -> None:
        if action == ACTION_PASS:
            engine.pass_turn()
            return
        from .encoding import action_to_point
        point = action_to_point(action)
        assert point is not None
        result = engine.play(point)
        if not result.legal:
            raise RuntimeError(
                f"self-play sampled an illegal action {action}: {result.reason}"
            )


def summarize_examples(examples: List[Example]) -> str:
    if not examples:
        return "no examples"
    return (
        f"n={len(examples)} "
        f"state={examples[0].state.shape} "
        f"policy_len={examples[0].policy.shape[0]} "
        f"z_mean={np.mean([e.value for e in examples]):+.3f}"
    )
