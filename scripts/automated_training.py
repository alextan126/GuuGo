r"""Single-box automated training loop with parallel self-play.

Self-play is embarrassingly parallel (different games are independent);
gradient descent on a single GPU is not trivially multi-process. So
this script spawns ``--num-workers`` dedicated self-play processes and
keeps the trainer single-process in the foreground. Each cycle is
a self-play burst (all workers play in parallel) followed by a train
burst (the trainer consumes the growing replay buffer). The freshly
trained weights are copied into a shared-memory inference model so
the next self-play burst already uses the latest network -- this is
the alternating "latest-weights" pattern the user asked for.

Architecture on a 20-core / 200 GB box::

    main process                 N worker processes
    ------------                 ------------------
    trainer.model   <----\      worker_model (shared memory, CPU)
    replay buffer        |             |
    training loop        |   forward pass only
                         |             |
           sync every    |    task_queue: "play K games"
           cycle  -------/    result_queue: pickled examples

The network is persisted on a wall-clock schedule (default: every 1
hour) so you get a predictable on-disk cadence regardless of cycle
length. Ctrl-C saves a final checkpoint and joins the workers before
exit.

Typical usage::

    python scripts/automated_training.py \\
        --num-workers 18 \\
        --games-per-worker 1 \\
        --train-steps-per-cycle 200 \\
        --save-interval-seconds 3600

Notes on device choice:
- Workers always run inference on CPU. With a small 9x9 network and
  dozens of CPU cores, CPU-batched inference per worker is fine and
  avoids fighting over one GPU.
- The trainer honors ``--device`` (``auto``/``cuda``/``cpu``/``mps``).
  If the trainer runs on GPU, each cycle we copy its weights down to
  the CPU shared-memory model that the workers see.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero.config import AlphaZeroConfig
from alphazero.model import PolicyValueNet
from alphazero.replay_buffer import ReplayBuffer
from alphazero.self_play import SelfPlayWorker
from alphazero.trainer import Trainer

DEFAULT_SAVE_INTERVAL_SECONDS = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Worker entry point (module-level; must be picklable for ``spawn``)
# ---------------------------------------------------------------------------


def _worker_entry(
    worker_id: int,
    config: AlphaZeroConfig,
    shared_model: PolicyValueNet,
    task_queue: "mp.Queue",
    result_queue: "mp.Queue",
    seed: int,
    write_game_files: bool,
) -> None:
    """Persistent self-play worker.

    Each worker:
    1. Receives a handle to the shared-memory ``PolicyValueNet`` that
       the trainer keeps refreshing. No file I/O is needed to pick up
       new weights -- the storage is physically shared.
    2. Pulls tasks off ``task_queue`` (``None`` = shutdown sentinel).
    3. Plays the requested number of games with :class:`SelfPlayWorker`,
       optionally persists each as a ``.pkl`` (crash-recovery / distributed
       ingestion), and ships examples back through ``result_queue``.
    """

    # Keep each worker single-threaded so N workers don't fight for the
    # 20 cores via OMP/MKL oversubscription.
    torch.set_num_threads(1)

    import numpy as np  # noqa: WPS433  (import after set_num_threads)

    rng = np.random.default_rng(seed)
    shared_model.eval()

    worker = SelfPlayWorker(
        config=config,
        model=shared_model,
        device=torch.device("cpu"),
        rng=rng,
    )

    while True:
        task = task_queue.get()
        if task is None:
            return

        num_games: int = task["num_games"]
        step_hint: int = task.get("step_hint", 0)

        games_examples: List[list] = []
        total_examples = 0
        for _ in range(num_games):
            examples = worker.generate_game()
            games_examples.append(examples)
            total_examples += len(examples)
            if write_game_files:
                worker.write_game_file(examples, step_hint=step_hint)

        result_queue.put(
            {
                "worker_id": worker_id,
                "games": games_examples,
                "total_examples": total_examples,
            }
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    default_workers = max(1, (os.cpu_count() or 4) - 2)

    p = argparse.ArgumentParser(
        description=(
            "GuuGo parallel automated training: many CPU self-play workers "
            "+ one trainer, alternating each cycle."
        )
    )
    # ---- I/O ----
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--replay-dir", type=Path, default=Path("replay"))
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for the TRAINER (workers always run on CPU).",
    )
    # ---- cadence ----
    p.add_argument(
        "--save-interval-seconds",
        type=float,
        default=DEFAULT_SAVE_INTERVAL_SECONDS,
        help="Wall-clock seconds between checkpoints (default: 3600 = 1h).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=default_workers,
        help=(
            f"Number of self-play processes "
            f"(default: os.cpu_count() - 2 = {default_workers})."
        ),
    )
    p.add_argument(
        "--games-per-worker",
        type=int,
        default=1,
        help=(
            "Self-play games played per worker per cycle. 1 keeps cycles "
            "short so workers pick up new weights often."
        ),
    )
    p.add_argument(
        "--train-steps-per-cycle",
        type=int,
        default=200,
        help="Gradient steps per cycle once the buffer has enough data.",
    )
    p.add_argument(
        "--log-every-cycles",
        type=int,
        default=1,
        help="How often to print a per-cycle summary line.",
    )
    p.add_argument(
        "--no-game-files",
        action="store_true",
        help=(
            "Skip persisting each completed game as a .pkl in --replay-dir. "
            "Faster, but drops crash-recovery and offline analysis. "
            "The in-memory buffer is unaffected."
        ),
    )
    # ---- MCTS / model knobs (passthrough to config) ----
    p.add_argument("--num-simulations", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--min-buffer", type=int, default=None)
    p.add_argument("--replay-capacity", type=int, default=None)
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> AlphaZeroConfig:
    config = AlphaZeroConfig(
        checkpoint_dir=args.checkpoint_dir,
        replay_dir=args.replay_dir,
        device=args.device,
    )
    if args.num_simulations is not None:
        config.num_simulations = args.num_simulations
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.min_buffer is not None:
        config.min_replay_to_train = args.min_buffer
    if args.replay_capacity is not None:
        config.replay_capacity = args.replay_capacity
    # Disable the trainer's step-based auto-save; this script owns
    # checkpointing on a time interval.
    config.checkpoint_every_steps = 10**9
    config.ensure_dirs()
    return config


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ParallelAutomatedTrainer:
    """Owns the trainer, the shared inference model, and the worker pool."""

    def __init__(
        self,
        config: AlphaZeroConfig,
        save_interval_seconds: float,
        num_workers: int,
        games_per_worker: int,
        write_game_files: bool,
    ) -> None:
        self.config = config
        self.save_interval = save_interval_seconds
        self.num_workers = num_workers
        self.games_per_worker = games_per_worker

        # ---- trainer (owns the authoritative model on the training device) ----
        self.buffer = ReplayBuffer(config.replay_capacity)
        self.trainer = Trainer(config, replay_buffer=self.buffer)

        resume_step = self.trainer.resume_from_checkpoint()
        if resume_step:
            print(f"[auto] resumed from checkpoint step={resume_step}")
        else:
            print("[auto] starting from fresh network")

        # ---- shared inference model for workers ----
        # This lives on CPU in shared memory. Workers hold handles to the
        # same physical tensors, so updating it here updates it everywhere
        # without any file I/O.
        self.worker_model = PolicyValueNet(config)
        self.worker_model.eval()
        self.worker_model.share_memory()
        self._sync_worker_model()

        # ---- spawn worker pool ----
        # Use ``spawn`` so CUDA (in the main process) is not forked, and
        # because torch.multiprocessing's tensor reducers handle the
        # shared model handle correctly across a spawn boundary.
        ctx = mp.get_context("spawn")
        self.task_queue: "mp.Queue" = ctx.Queue()
        self.result_queue: "mp.Queue" = ctx.Queue()
        self.processes: List[mp.Process] = []
        base_seed = int(time.time()) & 0xFFFFFF
        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_entry,
                args=(
                    i,
                    config,
                    self.worker_model,
                    self.task_queue,
                    self.result_queue,
                    base_seed * 1_000 + i,
                    write_game_files,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
        print(
            f"[auto] spawned {num_workers} self-play workers "
            f"(games_per_worker={games_per_worker}, "
            f"game_files={'on' if write_game_files else 'off'})"
        )

        # ---- bookkeeping ----
        self.start_time = time.time()
        self.last_save_time = time.time()
        self.total_games = 0
        self.total_examples = 0
        self.cycles = 0

    # ---------------- weight broadcast ----------------

    def _sync_worker_model(self) -> None:
        """Copy ``trainer.model`` weights into ``worker_model`` (shared mem).

        Since ``worker_model`` is in shared memory, this write is visible
        to all worker processes on their next forward pass. Safe to call
        only when workers are idle (i.e. between cycles).
        """
        src_state = self.trainer.model.state_dict()
        with torch.no_grad():
            dst_state = self.worker_model.state_dict()
            for name, dst_tensor in dst_state.items():
                src_tensor = src_state[name].detach()
                if src_tensor.device != dst_tensor.device:
                    src_tensor = src_tensor.to(dst_tensor.device)
                dst_tensor.copy_(src_tensor)

    # ---------------- lifecycle ----------------

    def shutdown(self) -> None:
        """Tell every worker to exit and join the processes."""
        for _ in self.processes:
            self.task_queue.put(None)
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    def run_forever(self, train_steps_per_cycle: int, log_every_cycles: int) -> None:
        try:
            while True:
                self._one_cycle(
                    train_steps_per_cycle=train_steps_per_cycle,
                    log_every_cycles=log_every_cycles,
                )
        except KeyboardInterrupt:
            print("\n[auto] KeyboardInterrupt - saving final checkpoint...")
            self._save_checkpoint(reason="ctrl-c")
            self.shutdown()
            print("[auto] goodbye")

    # ---------------- cycle ----------------

    def _one_cycle(self, train_steps_per_cycle: int, log_every_cycles: int) -> None:
        cycle_start = time.time()

        # --- self-play phase: fire off tasks and wait for all to return ---
        selfplay_start = time.time()
        step_hint = self.trainer.metrics.step
        for _ in range(self.num_workers):
            self.task_queue.put(
                {"num_games": self.games_per_worker, "step_hint": step_hint}
            )

        games_this_cycle = 0
        examples_this_cycle = 0
        for _ in range(self.num_workers):
            result = self.result_queue.get()
            for examples in result["games"]:
                self.buffer.add_examples(examples)
                games_this_cycle += 1
                examples_this_cycle += len(examples)

        self.total_games += games_this_cycle
        self.total_examples += examples_this_cycle
        selfplay_time = time.time() - selfplay_start

        # --- train phase ---
        train_start = time.time()
        stats_list = self.trainer.train_iteration(train_steps_per_cycle)
        train_time = time.time() - train_start

        # --- publish latest weights into the shared inference model ---
        self._sync_worker_model()

        self.cycles += 1

        # --- time-based checkpoint ---
        if time.time() - self.last_save_time >= self.save_interval:
            self._save_checkpoint(reason="interval")

        # --- log ---
        if self.cycles % max(1, log_every_cycles) == 0:
            self._log_cycle(
                cycle_start=cycle_start,
                selfplay_time=selfplay_time,
                train_time=train_time,
                games_this_cycle=games_this_cycle,
                examples_this_cycle=examples_this_cycle,
                stats_list=stats_list,
            )

    # ---------------- helpers ----------------

    def _save_checkpoint(self, reason: str) -> None:
        path = self.trainer.save_checkpoint(notes=f"auto:{reason}")
        self.last_save_time = time.time()
        uptime = _format_duration(time.time() - self.start_time)
        print(
            f"[auto] checkpoint saved: {path.name} "
            f"step={self.trainer.metrics.step} "
            f"buffer={len(self.buffer)} "
            f"total_games={self.total_games} "
            f"uptime={uptime}"
        )

    def _log_cycle(
        self,
        cycle_start: float,
        selfplay_time: float,
        train_time: float,
        games_this_cycle: int,
        examples_this_cycle: int,
        stats_list: list,
    ) -> None:
        wall = time.time() - cycle_start
        time_to_save = max(
            0.0, self.save_interval - (time.time() - self.last_save_time)
        )
        if stats_list:
            last = stats_list[-1]
            train_info = (
                f"trained={len(stats_list)} "
                f"loss={last.total:.4f} "
                f"policy={last.policy:.4f} "
                f"value={last.value:.4f}"
            )
        else:
            train_info = (
                f"waiting for data buffer={len(self.buffer)}/"
                f"{self.config.min_replay_to_train}"
            )
        games_per_sec = games_this_cycle / max(selfplay_time, 1e-6)
        print(
            f"[auto] cycle={self.cycles} "
            f"games={games_this_cycle} "
            f"({games_per_sec:.2f}/s) "
            f"examples={examples_this_cycle} "
            f"buffer={len(self.buffer)} "
            f"step={self.trainer.metrics.step} "
            f"{train_info} "
            f"sp={selfplay_time:.1f}s "
            f"tr={train_time:.1f}s "
            f"wall={wall:.1f}s "
            f"to_next_save={_format_duration(time_to_save)}"
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    config = _build_config(args)

    runner = ParallelAutomatedTrainer(
        config=config,
        save_interval_seconds=args.save_interval_seconds,
        num_workers=args.num_workers,
        games_per_worker=args.games_per_worker,
        write_game_files=not args.no_game_files,
    )

    print(
        f"[auto] starting loop "
        f"trainer_device={runner.trainer.device} "
        f"worker_device=cpu "
        f"num_workers={args.num_workers} "
        f"games_per_worker={args.games_per_worker} "
        f"games_per_cycle={args.num_workers * args.games_per_worker} "
        f"train_steps_per_cycle={args.train_steps_per_cycle} "
        f"save_every={_format_duration(args.save_interval_seconds)} "
        f"num_simulations={config.num_simulations} "
        f"batch_size={config.batch_size}"
    )
    runner.run_forever(
        train_steps_per_cycle=args.train_steps_per_cycle,
        log_every_cycles=args.log_every_cycles,
    )


if __name__ == "__main__":
    # Ensure ``spawn`` is the context for any Queues/Processes built
    # before ``ParallelAutomatedTrainer`` explicitly picks a context.
    mp.set_start_method("spawn", force=True)
    main()
