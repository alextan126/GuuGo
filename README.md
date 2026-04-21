# GuuGo

A 9x9 Go trainer, written in Python. This first milestone is a local
player-vs-player version with a pygame GUI and a clean rules engine that
will later be reused to power a bot.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` — installs `pygame` (for the GUI) and
  `pytest` (for the rule tests). No system packages required.

Fresh setup from scratch:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Running the PvP GUI

```bash
python main.py
```

Click an intersection to place a stone for the player to move. The most
recent move is highlighted with a red ring. Capture counts and the current
player are shown above the board.

Controls:

- `Pass (Resign)` button or **P** key — pass your turn, which immediately
  concedes the game (per the assignment rules).
- `Score Game` button or **S** key — stop play and compute the final
  Chinese area score with a 2.5 komi for White.
- `New Game` button or **N** key — reset the board.
- **Esc** — quit the application.

## Tests

```bash
pip install pytest
python3 -m pytest tests/
```

Tests cover single-stone capture, multi-stone group capture, suicide
rejection, the suicide-via-capture exception, the ko rule (both
preventing immediate recapture and allowing recapture after a ko threat),
pass-as-concession, and Chinese area scoring with komi.

## Engine interface

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

Additional helpers used by the GUI:

- `engine.current_player` — whose turn it is.
- `engine.last_move` — the coordinate of the most recent stone.
- `engine.captures_by(color)` — running capture total for a player.
- `engine.is_over`, `engine.result` — game-end status.

All engine methods are pure Python with no GUI dependencies, so the same
class can be driven from the test harness or from a future bot engine.

## Project layout

```
main.py                 # Tkinter entry point
go_game/
  __init__.py
  types.py              # Color / MoveResult / GameResult
  board.py              # Grid storage, neighbors, groups, liberties
  scoring.py            # Chinese area scoring (+ komi 2.5)
  engine.py             # Rules, turn order, ko, captures, pass
  gui.py                # Pygame PvP board
tests/
  test_engine.py        # Rule-focused unit tests
```

## Rules implemented

- **Liberties / capture.** Groups with zero liberties are removed.
- **Ko.** A move is rejected if the resulting board matches the
  position that existed before the opponent's previous move
  (prevents single-stone capture-recapture loops).
- **Suicide.** A move that would leave the playing group with no
  liberties is illegal *unless* it first captures one or more enemy
  stones.
- **Passing.** Passing is treated as resignation — the passing player
  loses immediately.
- **Scoring.** Chinese area scoring: each player's stones on the board
  plus empty regions surrounded solely by their stones, with White
  receiving a fixed 2.5 komi.
