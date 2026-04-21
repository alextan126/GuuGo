"""CLI entry point: run a self-play worker forever.

    python scripts/self_play.py --checkpoint-dir checkpoints --replay-dir replay
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero.config import AlphaZeroConfig
from alphazero.self_play import SelfPlayWorker


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GuuGo self-play worker")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--replay-dir", type=Path, default=Path("replay"))
    p.add_argument("--num-simulations", type=int, default=None)
    p.add_argument("--max-games", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = AlphaZeroConfig(
        checkpoint_dir=args.checkpoint_dir,
        replay_dir=args.replay_dir,
        device=args.device,
    )
    if args.num_simulations is not None:
        config.num_simulations = args.num_simulations
    config.ensure_dirs()

    worker = SelfPlayWorker(config)
    worker.maybe_reload_weights()
    worker.run_forever(max_games=args.max_games)


if __name__ == "__main__":
    main()
