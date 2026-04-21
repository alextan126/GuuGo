"""Shared types for the Go engine.

These are intentionally small, dependency-free dataclasses and enums so they
can be imported by both the engine and the GUI without creating a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class Color(Enum):
    """A point on the board is either empty or occupied by Black / White."""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def opponent(self) -> "Color":
        if self is Color.BLACK:
            return Color.WHITE
        if self is Color.WHITE:
            return Color.BLACK
        raise ValueError("EMPTY has no opponent")


Point = Tuple[int, int]  # (row, col)


@dataclass(frozen=True)
class MoveResult:
    """Outcome of attempting to place a stone.

    ``legal`` is the only field clients must check; the rest describe what
    actually happened on a legal move so the GUI can animate captures and
    highlight the most recent move.
    """

    legal: bool
    reason: str = ""
    captured: Tuple[Point, ...] = ()
    move: Optional[Point] = None


@dataclass(frozen=True)
class GameResult:
    """Final game outcome once the game has ended."""

    winner: Optional[Color]  # None means a tie, which cannot occur with komi=2.5
    black_score: float
    white_score: float
    reason: str  # "score" or "resignation"
