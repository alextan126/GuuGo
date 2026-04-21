"""Game engine enforcing the 9x9 Go rules used by GuuGo.

This module is the public interface for the rules engine: the GUI, the test
harness, and a future bot all talk to :class:`GameEngine`. The engine is
deliberately self-contained: it owns the board, the player to move, capture
counts, the ko state, and the game result, and exposes a small set of
well-documented methods.

Rules implemented (per the assignment):
    * Liberties / capture of zero-liberty groups.
    * Positional superko against the immediately preceding board (simple ko).
    * Suicide is illegal unless the move first captures opposing stones.
    * Passing is treated as resignation.
    * Chinese area scoring with a komi of 2.5 for White.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

from .board import (
    BOARD_SIZE,
    BoardGrid,
    empty_board,
    find_group,
    neighbors,
    remove_points,
    set_point,
)
from .scoring import area_score
from .types import Color, GameResult, MoveResult, Point


class GameEngine:
    """Encapsulates the full state of a 9x9 Go game.

    The engine is single-threaded and mutated only through its public
    methods. Callers should treat :meth:`board_state` as read-only.
    """

    def __init__(self, size: int = BOARD_SIZE) -> None:
        self._size = size
        self._board: BoardGrid = empty_board(size)
        self._current: Color = Color.BLACK
        self._previous_board: Optional[BoardGrid] = None
        self._captured: dict = {Color.BLACK: 0, Color.WHITE: 0}
        self._last_move: Optional[Point] = None
        self._result: Optional[GameResult] = None
        self._move_number: int = 0

    # ---------------- public state accessors ----------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def current_player(self) -> Color:
        return self._current

    @property
    def last_move(self) -> Optional[Point]:
        return self._last_move

    @property
    def move_number(self) -> int:
        return self._move_number

    @property
    def is_over(self) -> bool:
        return self._result is not None

    @property
    def result(self) -> Optional[GameResult]:
        return self._result

    def captures_by(self, color: Color) -> int:
        """Number of stones of the opposite color that ``color`` has captured."""

        if color is Color.EMPTY:
            raise ValueError("EMPTY cannot capture stones")
        return self._captured[color]

    def board_state(self) -> List[List[Color]]:
        """Return a mutable copy of the board as a list of lists of Color."""

        return [list(row) for row in self._board]

    @property
    def board_grid(self) -> BoardGrid:
        """Return the raw immutable board grid.

        This is intended for hot paths like the AlphaZero state encoder /
        MCTS, which do not need a defensive copy. Do not mutate.
        """

        return self._board

    def clone(self) -> "GameEngine":
        """Return a new engine in the exact same state.

        Cheap because the board grid is already an immutable tuple of
        tuples. MCTS uses this to branch on trial moves without touching
        the real game state.
        """

        twin = GameEngine(self._size)
        twin._board = self._board
        twin._current = self._current
        twin._previous_board = self._previous_board
        twin._captured = dict(self._captured)
        twin._last_move = self._last_move
        twin._result = self._result
        twin._move_number = self._move_number
        return twin

    def iter_legal_points(self, color: Optional[Color] = None) -> Iterator[Point]:
        """Yield every board point that is a legal move for ``color``."""

        mover = color if color is not None else self._current
        if self._result is not None:
            return
        for r in range(self._size):
            for c in range(self._size):
                if self._board[r][c] is not Color.EMPTY:
                    continue
                legal, _ = self.is_legal((r, c), mover)
                if legal:
                    yield (r, c)

    def legal_points(self, color: Optional[Color] = None) -> List[Point]:
        """List form of :meth:`iter_legal_points`."""

        return list(self.iter_legal_points(color))

    def terminal_value(self, color: Color) -> float:
        """Return the game result as +1 / -1 / 0 from ``color``'s view.

        Raises if the game is not over; MCTS should gate this on
        :attr:`is_over`. A tie returns ``0`` (extremely rare with komi 2.5
        but the :class:`GameResult` type allows it).
        """

        if self._result is None:
            raise RuntimeError("terminal_value called on unfinished game")
        if self._result.winner is None:
            return 0.0
        return 1.0 if self._result.winner is color else -1.0

    # ---------------- rule checks ----------------

    def is_legal(self, point: Point, color: Optional[Color] = None) -> Tuple[bool, str]:
        """Return ``(True, "")`` if ``color`` may play at ``point``.

        When ``color`` is omitted the current player's turn is used. The
        second element of the return value is a short human-readable reason
        explaining an illegal move.
        """

        if self._result is not None:
            return False, "game is over"

        mover = color if color is not None else self._current

        r, c = point
        if not (0 <= r < self._size and 0 <= c < self._size):
            return False, "out of bounds"
        if self._board[r][c] is not Color.EMPTY:
            return False, "point is occupied"

        trial, _ = self._simulate(point, mover)
        if trial is None:
            return False, "suicide"
        if self._previous_board is not None and trial == self._previous_board:
            return False, "ko"
        return True, ""

    # ---------------- mutating operations ----------------

    def play(self, point: Point) -> MoveResult:
        """Attempt to place a stone for the current player.

        On success advances the turn and records capture counts. On failure
        the board is left untouched and ``MoveResult.legal`` is False.
        """

        legal, reason = self.is_legal(point, self._current)
        if not legal:
            return MoveResult(legal=False, reason=reason)

        new_board, captured = self._simulate(point, self._current)
        assert new_board is not None  # legality check guarantees this

        self._previous_board = self._board
        self._board = new_board
        self._captured[self._current] += len(captured)
        self._last_move = point
        self._move_number += 1
        self._current = self._current.opponent()

        return MoveResult(
            legal=True,
            reason="",
            captured=tuple(captured),
            move=point,
        )

    def pass_turn(self) -> GameResult:
        """Passing concedes the game per the assignment rules."""

        if self._result is not None:
            return self._result

        loser = self._current
        winner = loser.opponent()
        black_score, white_score = area_score(self._board)
        self._result = GameResult(
            winner=winner,
            black_score=black_score,
            white_score=white_score,
            reason="resignation",
        )
        return self._result

    def finish_by_score(self) -> GameResult:
        """End the game and compute the winner from Chinese area scoring.

        This is not required by the assignment (which ends games on pass),
        but it is a convenient entry point for the GUI's "Score Game" action
        and for the test harness.
        """

        if self._result is not None:
            return self._result

        black_score, white_score = area_score(self._board)
        if black_score > white_score:
            winner: Optional[Color] = Color.BLACK
        elif white_score > black_score:
            winner = Color.WHITE
        else:
            winner = None
        self._result = GameResult(
            winner=winner,
            black_score=black_score,
            white_score=white_score,
            reason="score",
        )
        return self._result

    def reset(self) -> None:
        """Return the engine to a fresh empty board with Black to move."""

        self._board = empty_board(self._size)
        self._current = Color.BLACK
        self._previous_board = None
        self._captured = {Color.BLACK: 0, Color.WHITE: 0}
        self._last_move = None
        self._result = None
        self._move_number = 0

    # ---------------- internals ----------------

    def _simulate(
        self, point: Point, color: Color
    ) -> Tuple[Optional[BoardGrid], List[Point]]:
        """Simulate placing ``color`` at ``point``.

        Returns the resulting board plus the list of captured stones. If the
        move would be suicide (own group has no liberties AND nothing is
        captured) returns ``(None, [])``.
        """

        tentative = set_point(self._board, point, color)
        opponent = color.opponent()

        captured: List[Point] = []
        seen_groups = set()
        for nb in neighbors(point, self._size):
            nr, nc = nb
            if tentative[nr][nc] is not opponent:
                continue
            group, liberties = find_group(tentative, nb)
            if not group or group in seen_groups:
                continue
            seen_groups.add(group)
            if not liberties:
                captured.extend(group)

        if captured:
            tentative = remove_points(tentative, captured)

        own_group, own_liberties = find_group(tentative, point)
        if not own_liberties:
            return None, []

        return tentative, captured
