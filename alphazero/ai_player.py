"""Bridge between a trained checkpoint and the pygame PvE mode.

The trainer knows how to write ``latest.pt`` into a checkpoint directory.
The PvE GUI needs to turn that file back into "given this board, what
move should I play?". :class:`AIPlayer` is the thin thing that connects
the two.

Design notes
------------
- The pass action (81) is masked out before picking a move. The engine
  treats a pass as immediate resignation, so a mid-trained policy that
  happens to assign it non-trivial probability would just throw games;
  the GUI always wants a real board move if one is legal.
- MCTS is run without Dirichlet noise and the argmax of the visit
  distribution is played. That is the "strong-play" setting used for
  evaluation, not the "exploratory" setting used during self-play.
- The engine passed in is cloned inside :meth:`choose_move`, so callers
  (the GUI, specifically) can keep using the live engine without worrying
  about the MCTS mutating it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from go_game.engine import GameEngine
from go_game.types import Point

from .checkpoints import load_latest_checkpoint, read_latest_meta, resolve_device
from .config import AlphaZeroConfig
from .encoding import ACTION_PASS, action_to_point
from .mcts import MCTS
from .model import PolicyValueNet
from .self_play import make_torch_policy_value_fn


@dataclass
class AIPlayer:
    """Loaded policy-value network plus its MCTS, ready to answer moves."""

    config: AlphaZeroConfig
    model: PolicyValueNet
    device: torch.device
    mcts: MCTS
    loaded_step: int

    # -------------------- construction --------------------

    @staticmethod
    def load_from_checkpoint(
        checkpoint_dir: Path,
        num_simulations: int,
        device: str = "auto",
    ) -> "AIPlayer":
        """Load ``latest.pt`` from ``checkpoint_dir`` into a fresh AIPlayer.

        Raises :class:`FileNotFoundError` if the directory does not
        contain a ``latest.pt`` the checkpoint helpers can read. The GUI
        menu catches that and stays on the menu screen with an error.
        """

        checkpoint_dir = Path(checkpoint_dir)
        config = AlphaZeroConfig(
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        config.num_simulations = num_simulations

        resolved_device = resolve_device(device)
        model = PolicyValueNet(config).to(resolved_device)
        model.eval()

        step = load_latest_checkpoint(
            checkpoint_dir,
            model,
            optimizer=None,
            map_location=str(resolved_device),
        )
        if step is None:
            raise FileNotFoundError(
                f"No checkpoint (latest.pt) found in {checkpoint_dir}"
            )

        pv_fn = make_torch_policy_value_fn(model, resolved_device)
        mcts = MCTS(
            policy_value_fn=pv_fn,
            num_simulations=num_simulations,
            c_puct=config.c_puct,
            action_size=config.action_size,
            # No Dirichlet noise at evaluation time: we want the
            # strongest move, not an exploratory one.
            dirichlet_alpha=None,
            dirichlet_epsilon=0.0,
        )

        return AIPlayer(
            config=config,
            model=model,
            device=resolved_device,
            mcts=mcts,
            loaded_step=step,
        )

    # -------------------- convenience --------------------

    def describe(self) -> str:
        """Short human-readable summary, handy for the GUI title bar."""

        meta = read_latest_meta(self.config.checkpoint_dir)
        step_tag = meta.step if meta is not None else self.loaded_step
        return (
            f"step={step_tag} sims={self.mcts.num_simulations} "
            f"device={self.device}"
        )

    # -------------------- inference --------------------

    def choose_move(self, engine: GameEngine) -> Optional[Point]:
        """Pick the best legal *board* move for the side to play.

        Returns ``(row, col)`` on success, or ``None`` if the position is
        already terminal (the GUI should never ask in that case, but we
        guard anyway).

        The pass action is excluded deliberately; if the policy really
        wants to pass, MCTS will still have burnt visits on real moves
        (there is always at least one legal empty point on 9x9 early on),
        and argmax over the remaining slots gives us the best non-pass
        option.
        """

        if engine.is_over:
            return None

        root = self.mcts.search(engine.clone(), add_root_noise=False)
        pi = self.mcts.visit_policy(root, temperature=0.0)

        # Mask pass: zero its visit weight and renormalize. If the only
        # non-zero entry was pass (MCTS spent all visits there), fall
        # back to any legal board point to avoid conceding.
        pi = pi.copy()
        pi[ACTION_PASS] = 0.0
        total = pi.sum()
        if total > 0:
            action = int(np.argmax(pi))
        else:
            action = _pick_any_legal_board_action(engine)
            if action is None:
                return None

        return action_to_point(action)


def _pick_any_legal_board_action(engine: GameEngine) -> Optional[int]:
    """Last-ditch fallback when MCTS produces no non-pass preference."""

    for point in engine.iter_legal_points():
        r, c = point
        return r * engine.size + c
    return None
