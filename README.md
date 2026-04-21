# GuuGo

A 9x9 Go trainer, written in Python. It ships two things built on the
same rules engine:

1. A beginner **PvP app** with a pygame GUI that satisfies the class
   assignment's harness interface.
2. An **AlphaZero-style training pipeline** (self-play + replay buffer
   + PyTorch trainer) that teaches a network to play 9x9 Go.

The system architecture, data flow, and scaling path are documented
separately in [`Architecture.md`](Architecture.md).

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` — installs:
  - `pygame` for the PvP GUI,
  - `numpy` and `torch` for the AlphaZero stack,
  - `pytest` for tests.

If you only want the PvP app you can skip installing `torch`; the
training imports are not loaded unless you actually run them.

Fresh setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the PvP GUI

```bash
python main.py
```

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
python scripts/automated_training.py
```

That is equivalent to the fully-explicit form:

```bash
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
net, dozens of cores, no contention for the GPU). On a machine without
a GPU, pass `--device cpu`; on Apple Silicon you can pass `--device
auto` to resolve to MPS.

Decoupled layout (separate shells / hosts share the same dirs):

```bash
# terminal 1 - trainer
python scripts/train.py --checkpoint-dir checkpoints --replay-dir replay

# terminal 2..N - workers
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
