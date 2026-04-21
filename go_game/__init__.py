"""GuuGo: a 9x9 Go engine and Tkinter GUI.

The package exposes the :class:`GameEngine` class as the primary entry point
for interacting with the rules engine. A separate Tkinter GUI lives in
``go_game.gui`` but the engine itself has no GUI dependencies, so the same
engine can be reused later by a bot or a test harness.
"""

from .types import Color, GameResult, MoveResult
from .engine import GameEngine

__all__ = ["Color", "GameResult", "MoveResult", "GameEngine"]
