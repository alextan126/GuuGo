"""Low-level board storage and helpers for connected groups / liberties.

The board is stored as a tuple of tuples of :class:`Color` so that snapshots
are hashable and comparable. This matters for the ko rule, which compares the
post-move board against the immediately preceding position.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, List, Set, Tuple

from .types import Color, Point

BOARD_SIZE = 9


BoardGrid = Tuple[Tuple[Color, ...], ...]


def empty_board(size: int = BOARD_SIZE) -> BoardGrid:
    """Return an immutable empty board."""

    row = (Color.EMPTY,) * size
    return tuple(row for _ in range(size))


def in_bounds(point: Point, size: int = BOARD_SIZE) -> bool:
    r, c = point
    return 0 <= r < size and 0 <= c < size


def neighbors(point: Point, size: int = BOARD_SIZE) -> List[Point]:
    r, c = point
    candidates = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    return [p for p in candidates if in_bounds(p, size)]


def set_point(board: BoardGrid, point: Point, color: Color) -> BoardGrid:
    """Return a new board with ``point`` set to ``color``."""

    r, c = point
    row = board[r]
    new_row = row[:c] + (color,) + row[c + 1 :]
    return board[:r] + (new_row,) + board[r + 1 :]


def remove_points(board: BoardGrid, points: Iterable[Point]) -> BoardGrid:
    """Return a new board with the given points cleared."""

    current = board
    for p in points:
        current = set_point(current, p, Color.EMPTY)
    return current


def find_group(board: BoardGrid, point: Point) -> Tuple[FrozenSet[Point], FrozenSet[Point]]:
    """Flood-fill the connected group of same-color stones containing ``point``.

    Returns ``(group_stones, liberties)``. If ``point`` is empty, both sets are
    empty.
    """

    r, c = point
    color = board[r][c]
    if color is Color.EMPTY:
        return frozenset(), frozenset()

    size = len(board)
    group: Set[Point] = set()
    liberties: Set[Point] = set()
    stack = [point]
    while stack:
        current = stack.pop()
        if current in group:
            continue
        group.add(current)
        for nb in neighbors(current, size):
            nr, nc = nb
            value = board[nr][nc]
            if value is Color.EMPTY:
                liberties.add(nb)
            elif value is color and nb not in group:
                stack.append(nb)
    return frozenset(group), frozenset(liberties)
