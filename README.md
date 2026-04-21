# GuuGo

A 9x9 Go trainer, written in Python. It ships two things built on the
same rules engine:

1. A beginner **PvP app** with a pygame GUI that satisfies the class
   assignment's harness interface.
2. An **AlphaZero-style training pipeline** (self-play + replay buffer
   + PyTorch trainer) that teaches a network to play 9x9 Go.

The system architecture, data flow, and scaling path are documented
separately in [`Architecture.md`](Architecture.md).

## Setup

Two machines, two different installation stories. **No virtualenvs are
used**: the PvP app uses a per-user pip install, and training uses the
NVIDIA NGC PyTorch container directly (virtualenvs don't carry the
right CUDA toolchain for Blackwell).

### Laptop (PvP GUI)

```bash
pip install --user -r requirements.txt
python main.py
```

On systems that block system-wide pip (PEP 668, e.g. recent Debian /
Ubuntu / Homebrew-managed Python), use `pipx install pygame` or add
`--break-system-packages` to the pip invocation. `torch` is **not**
needed for the PvP app; it is intentionally absent from
`requirements.txt`.

### Training box (Blackwell / DGX Spark)

Training lives inside a Docker container built on NVIDIA's NGC
PyTorch image. That image ships a `torch` wheel compiled with
Blackwell kernels (`sm_100` / `sm_120`); stock PyPI wheels don't, and
a plain `pip install torch` on Blackwell silently falls back to CPU
(or crashes with "no kernel image available for device"). Do not
create a venv on the training box — `pip install torch` will fight
the NGC build.

Build the image once:

```bash
docker build -t guugo-train .
```

Run training:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/workspace -w /workspace \
  guugo-train \
  python scripts/automated_training.py
```

Flags that matter:

- `--gpus all` — expose every GPU; replace with `--gpus '"device=0"'`
  to pin one.
- `--ipc=host` — required for `torch.multiprocessing` shared memory.
  The default 64 MB `/dev/shm` in Docker will kill the worker pool
  with bus errors. (Alternative: `--shm-size=8g`.)
- `-v "$PWD":/workspace` — keeps `checkpoints/` and `replay/` on the
  host so they survive container restarts and can be rsync'd to
  another machine.

Quick GPU sanity check inside the container:

```bash
docker run --rm --gpus all guugo-train \
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
```

On Blackwell you should see something like
`2.6.0a0+ecf3bae40a.nv25.03 12.8 True (10, 0)`.

---

## Running the PvP GUI

```bash
python main.py
```

This opens a small startup menu with two options:

- **Play vs Friend (PvP)** — the classic two-humans-at-one-keyboard mode.
- **Play vs Computer (PvE)** — plays against a trained AlphaZero model
  (see [Playing vs the AI](#playing-vs-the-ai) below).

Click an intersection to place a stone. The most recent move is
highlighted with a red ring. Capture counts and the current player are
shown above the board.

Controls:

- `Pass (Resign)` button or **P** key — pass your turn, which immediately
  concedes the game (per the assignment rules).
- `Score Game` button or **S** key — stop play and compute the final
  Chinese area score with a 2.5 komi for White.
- `New Game` button or **N** key — reset the board.
- **Esc** — quit the application.

### Playing vs the AI

The PvE panel on the startup menu takes one input:

- **Checkpoint directory** — path to a directory containing `latest.pt`
  (written by the training pipeline). Defaults to `checkpoints`.
  Checkpoints are gitignored (`*.pt`), so they do **not** sync
  automatically between machines. If you've been training on
  `spark-43f4`, rsync the directory over to your laptop first, e.g.

  ```bash
  rsync -av spark-43f4:~/GuuGo/checkpoints ./
  ```

The AI runs a fixed 100 MCTS simulations per move — strong enough to
play real moves, fast enough that you rarely wait more than a second or
two. If you want a different strength, change `DEFAULT_SIMULATIONS` in
[`go_game/menu.py`](go_game/menu.py).

When you click **Play vs Computer**, the app loads the checkpoint and
drops into a normal board view with the human as Black and the AI as
White. The title bar shows `(vs AI)` and a banner reads "AI is
thinking..." while MCTS runs. If loading fails (no `latest.pt`, torch
not installed, corrupted checkpoint, ...) the menu stays open and shows
the error — pick a different directory or install torch and try again.

PvE is the one place where the PvP app needs the training stack
installed (`pip install --user torch`). Plain PvP works with just
`pygame`.

### Engine interface

The assignment's harness requirements map directly onto
[`go_game.engine.GameEngine`](go_game/engine.py):

| Requirement                                         | API                                     |
| --------------------------------------------------- | --------------------------------------- |
| (a) Initialize a new game (empty 9x9 board)         | `GameEngine()` / `engine.reset()`       |
| (b) Place a stone at `(row, col)` for current player| `engine.play((row, col))`               |
| (c) Return whether a given move is legal            | `engine.is_legal((row, col))`           |
| (d) Return the current board state                  | `engine.board_state()`                  |
| (e) End the game when a player passes               | `engine.pass_turn()` (concedes)         |
| (f) Compute final score and winner                  | `engine.finish_by_score()` / `result`   |

---

## Running AlphaZero training

There are three entry points under [`scripts/`](scripts):

- [`scripts/automated_training.py`](scripts/automated_training.py) —
  infinite loop on a single box with **parallel self-play**. Spawns
  `--num-workers` dedicated worker processes that share a CPU
  inference model via `torch.multiprocessing` shared memory; the
  trainer runs in the main process. Each cycle alternates: all
  workers play in parallel → trainer does gradient steps → fresh
  weights are synced into the shared model for the next cycle. The
  network is persisted on a wall-clock schedule (default: every 1
  hour). This is the recommended way to run training on a single
  box (e.g. a 20-core / 200 GB machine).
- [`scripts/self_play.py`](scripts/self_play.py) — long-running
  self-play-only worker that watches for new checkpoints and writes
  replay files. Run multiple instances on a cluster beside a dedicated
  trainer.
- [`scripts/train.py`](scripts/train.py) — long-running trainer-only
  that ingests replay files, trains the network, and publishes new
  checkpoints. Pair with one or more `self_play.py` workers.

Recommended single-box run (Ctrl-C saves a final checkpoint):

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/workspace -w /workspace \
  guugo-train \
  python scripts/automated_training.py
```

That is equivalent to the fully-explicit form:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/workspace -w /workspace \
  guugo-train \
  python scripts/automated_training.py \
    --device cuda \
    --num-workers $(nproc) \
    --games-per-worker 1 \
    --train-steps-per-cycle 200 \
    --save-interval-seconds 3600 \
    --checkpoint-dir checkpoints \
    --replay-dir replay
```

Defaults: trainer on CUDA, one self-play worker per CPU core, one-hour
checkpoint cadence. Workers always run inference on CPU (small 9x9
net, dozens of cores, no contention for the GPU). On a machine
without a GPU you can run the script directly (outside Docker) with
`--device cpu`; on Apple Silicon, pass `--device auto` to resolve to
MPS.

Decoupled layout (separate shells / hosts share the same dirs). On a
single box you can open two containers bound to the same volume:

```bash
# terminal 1 - trainer
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/workspace -w /workspace guugo-train \
  python scripts/train.py --checkpoint-dir checkpoints --replay-dir replay

# terminal 2..N - workers (CPU-only is fine for workers)
docker run --rm -it --ipc=host \
  -v "$PWD":/workspace -w /workspace guugo-train \
  python scripts/self_play.py --checkpoint-dir checkpoints --replay-dir replay
```

Key knobs live in [`alphazero/config.py`](alphazero/config.py):
network size (`num_res_blocks`, `num_channels`), MCTS budget
(`num_simulations`, `c_puct`), replay capacity, optimizer settings,
checkpoint cadence, and paths. For DGX Spark / Blackwell deployment see
the scaling section of [`Architecture.md`](Architecture.md).

---

## Tests

```bash
python -m pytest tests/
```

Five focused suites:

- `tests/test_engine.py` — rules of Go (capture, suicide, ko, scoring).
- `tests/test_encoding.py` — state/action indexing and D4 symmetries.
- `tests/test_mcts.py` — MCTS invariants with a stubbed policy.
- `tests/test_replay_buffer.py` — FIFO, sampling, file ingestion.
- `tests/test_model_trainer.py` — PyTorch forward/backward and
  checkpoint roundtrip (auto-skipped if torch isn't installed).

---

## Project layout

```
main.py                       # PvP app entry point
go_game/                      # rules engine + pygame GUI
  __init__.py
  types.py                    # Color / MoveResult / GameResult
  board.py                    # grid storage, neighbors, groups, liberties
  scoring.py                  # Chinese area scoring (+ komi 2.5)
  engine.py                   # rules, turn order, ko, captures, pass
  gui.py                      # pygame PvP board
alphazero/                    # AlphaZero training stack (PyTorch)
  __init__.py
  config.py                   # all hyperparameters in one dataclass
  encoding.py                 # state/action encoding + D4 symmetries
  model.py                    # ResNet policy-value network + loss
  mcts.py                     # PUCT Monte Carlo Tree Search
  self_play.py                # worker that generates (state, pi, z)
  replay_buffer.py            # bounded FIFO + file ingestion
  trainer.py                  # batch sampling + SGD + checkpoints
  checkpoints.py              # atomic save / load helpers
scripts/
  automated_training.py       # single-process infinite self-play + train loop
  self_play.py                # long-running worker CLI
  train.py                    # long-running trainer CLI
tests/                        # pytest suites (see above)
Dockerfile                    # NGC PyTorch + pytest; training container
Architecture.md               # system design, data flow, scaling notes
```

---

## Rules implemented

- **Liberties / capture.** Groups with zero liberties are removed.
- **Ko.** A move is rejected if the resulting board matches the
  position that existed before the opponent's previous move.
- **Suicide.** A move that would leave the playing group with no
  liberties is illegal *unless* it first captures one or more enemy
  stones.
- **Passing.** Passing is treated as resignation; the passing player
  loses immediately. The AlphaZero pipeline trains on this rule, so
  the learned policy reflects "pass = concede" semantics.
- **Scoring.** Chinese area scoring: each player's stones on the board
  plus empty regions surrounded solely by their stones, with White
  receiving a fixed 2.5 komi.
