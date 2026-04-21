"""CLI entry point: run the trainer loop forever.

    python scripts/train.py --checkpoint-dir checkpoints --replay-dir replay
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero.config import AlphaZeroConfig
from alphazero.trainer import Trainer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GuuGo AlphaZero trainer")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--replay-dir", type=Path, default=Path("replay"))
    p.add_argument("--steps-per-cycle", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = AlphaZeroConfig(
        checkpoint_dir=args.checkpoint_dir,
        replay_dir=args.replay_dir,
        device=args.device,
    )
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    config.ensure_dirs()

    trainer = Trainer(config)
    trainer.run_forever(steps_per_cycle=args.steps_per_cycle)


if __name__ == "__main__":
    main()
