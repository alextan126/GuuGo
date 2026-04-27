# GuuGo

A 9x9 Go trainer in Python. Two things built on the same rules engine:

1. A **PvP / PvE app** with a pygame GUI (satisfies the class assignment harness).
2. An **AlphaZero-style training pipeline** (self-play + replay buffer + PyTorch trainer).

Full system design, data flow, and scaling notes live in
[`Architecture.md`](Architecture.md).

---

## How to use

### Play the game (laptop)

```bash
pip install --user -r requirements.txt
python main.py
```

`main.py` opens a menu with two modes:

- **Play vs Friend (PvP)** — two humans at one keyboard.
- **Play vs Computer (PvE)** — plays against a trained checkpoint.
  Point the checkpoint field at a directory containing `latest.pt`.
  PvE requires `pip install --user torch`; plain PvP does not.

Controls: click to play, **P** = pass (resigns), **S** = score, **N** =
new game, **Esc** = quit. The AI runs 100 MCTS simulations per move
(change `DEFAULT_SIMULATIONS` in [`go_game/menu.py`](go_game/menu.py) to
taste).

Checkpoints are gitignored. If training is on another machine:

```bash
rsync -av <training-host>:~/GuuGo/checkpoints ./
```

### Train a network (GPU box)

Training runs in NVIDIA's NGC PyTorch container. **Do not use a venv** —
`pip install torch` replaces the NGC build that has Blackwell kernels
(`sm_100` / `sm_120`) with a stock wheel that doesn't.

```bash
docker build -t guugo-train .

docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/workspace -w /workspace \
  guugo-train \
  python scripts/automated_training.py
```

Flags that matter on the `docker run`:

- `--gpus all` — expose GPUs.
- `--ipc=host` — needed for `torch.multiprocessing` shared memory
  (alternative: `--shm-size=8g`).
- `-v "$PWD":/workspace` — keeps `checkpoints/` and `replay/` on the host.

Three training entry points live in [`scripts/`](scripts):

| Script                      | What it does                                                                    |
| --------------------------- | ------------------------------------------------------------------------------- |
| `automated_training.py`     | Single-box loop. Parallel CPU self-play workers + GPU trainer, shared-memory weight sync. **Recommended.** |
| `self_play.py`              | Long-running self-play-only worker. Run many alongside one trainer.             |
| `train.py`                  | Long-running trainer-only. Consumes replay files, publishes checkpoints.        |

All hyperparameters (network size, MCTS budget, optimizer, paths) live
in [`alphazero/config.py`](alphazero/config.py).

### Run tests

```bash
python -m pytest tests/
```

Five suites under `tests/`: engine rules, encoding + D4 symmetries,
MCTS invariants, replay buffer, and model / trainer roundtrip
(auto-skipped if torch isn't installed).

---

## Architecture overview

Two tracks share one rules engine and nothing else:

```
                        go_game.engine.GameEngine
                        (rules, ko, scoring, pass)
                          /                \
                         /                  \
              main.py + gui.py        alphazero/ training stack
              (pygame PvP/PvE)        (self-play, MCTS, trainer)
```

The training pipeline is deliberately decoupled — self-play workers and
the trainer never talk to each other directly, only through files:

```
  self_play workers  ──► replay/game_*.pkl ──►  trainer
         ▲                                          │
         │                                          ▼
         └──── checkpoints/latest.pt ◄───── publishes new weights
```

Key components (each in its own module under [`alphazero/`](alphazero)):

- **`model.py`** — `PolicyValueNet`, a small ResNet (3 blocks, 96
  channels) with a policy head (82 logits = 81 points + pass) and a
  scalar value head.
- **`mcts.py`** — PUCT Monte Carlo Tree Search, single-threaded,
  per-move. Dirichlet noise at the root.
- **`self_play.py`** — plays a game, records
  `(state, mcts_pi, to_move)` per move, back-fills `z` at game end,
  D4-augments, pickles to `replay/`.
- **`replay_buffer.py`** — bounded FIFO; ingests new game files by name.
- **`trainer.py`** — samples batches, runs SGD + Nesterov, atomically
  publishes new checkpoints.
- **`config.py`** — every hyperparameter in one dataclass.

This is the **self-improvement loop: MCTS teaches the network (search
 raw policy), and the stronger network makes MCTS stronger next
iteration. See [`Architecture.md`](Architecture.md) for data flow,
scaling to multi-node, and the DGX Spark / Blackwell deployment story.

### Project layout

```
main.py                      # PvP/PvE app entry point
go_game/                     # rules engine + pygame GUI
alphazero/                   # AlphaZero training stack (PyTorch)
scripts/                     # training CLIs (automated / self_play / train)
tests/                       # pytest suites
Dockerfile                   # NGC PyTorch training container
Architecture.md              # system design, data flow, scaling
```

---

## Rules implemented

- **Capture** — groups with zero liberties are removed.
- **Ko** — a move is rejected if the resulting board matches the
  position before the opponent's previous move.
- **Suicide** — illegal unless the move first captures enemy stones.
- **Pass = resignation** — the passing player loses immediately, per
  the assignment. The AlphaZero pipeline trains on this rule.
- **Scoring** — Chinese area scoring with a 2.5 komi for White.
