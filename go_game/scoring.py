"""Chinese (area) scoring for a finished 9x9 Go game.

A player's score is the number of their stones on the board plus the number
of empty intersections that are surrounded solely by their stones. White
additionally receives a fixed komi.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

from .board import BoardGrid, neighbors
from .types import Color

KOMI = 2.5


def area_score(board: BoardGrid) -> Tuple[float, float]:
    """Return ``(black_score, white_score)`` using Chinese area scoring.

    The komi is added to White. The function is pure: it does not mutate the
    board. Empty regions touching stones of both colors belong to neither
    player (dame) and contribute zero.
    """

    size = len(board)
    counts: Dict[Color, float] = {Color.BLACK: 0.0, Color.WHITE: 0.0}

    visited_empty: Set[Tuple[int, int]] = set()

    for r in range(size):
        for c in range(size):
            value = board[r][c]
            if value is Color.BLACK or value is Color.WHITE:
                counts[value] += 1
                continue
            if (r, c) in visited_empty:
                continue

            region: Set[Tuple[int, int]] = set()
            bordering: Set[Color] = set()
            stack = [(r, c)]
            while stack:
                cell = stack.pop()
                if cell in region:
                    continue
                region.add(cell)
                for nb in neighbors(cell, size):
                    nr, nc = nb
                    v = board[nr][nc]
                    if v is Color.EMPTY and nb not in region:
                        stack.append(nb)
                    elif v is Color.BLACK or v is Color.WHITE:
                        bordering.add(v)
            visited_empty |= region

            if len(bordering) == 1:
                owner = next(iter(bordering))
                counts[owner] += len(region)

    return counts[Color.BLACK], counts[Color.WHITE] + KOMI
