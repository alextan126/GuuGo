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
    * Standard two-pass end-of-game (`pass_move`): the second consecutive pass
      enters the SCORING phase (dead stones auto-marked, then user-editable).
    * Explicit resignation (`resign`): the current player concedes, skipping
      the scoring phase.
    * Chinese area scoring with a komi of 2.5 for White, with dead-stone
      awareness when finalized via `confirm_score`.

Lifecycle:

    PLAYING --pass_move (2nd) / enter_scoring--> SCORING --confirm_score--> FINISHED
    PLAYING --resign-------------------------->                             FINISHED
    SCORING --cancel_scoring--> PLAYING
    SCORING --resign---------->                                             FINISHED

The legacy ``pass_turn`` method (single pass = resignation) is kept for the
AlphaZero training pipeline, which treats pass as a terminal action, along
with ``finish_by_score`` which skips the scoring phase entirely.
"""

from __future__ import annotations

from typing import FrozenSet, Iterator, List, Optional, Tuple

from .board import (
    BOARD_SIZE,
    BoardGrid,
    empty_board,
    find_group,
    neighbors,
    remove_points,
    set_point,
)
from .scoring import area_score, auto_dead_stones, group_at, score_breakdown
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
        # Number of consecutive passes seen at the tail of the move history.
        # Reset to zero on every successful stone placement; two in a row
        # enters the SCORING phase under the standard Go rule.
        self._consecutive_passes: int = 0
        # SCORING-phase state. When ``_in_scoring`` is True the engine
        # rejects new plays; ``_dead_stones`` is the user/auto-marked
        # dead set used by ``confirm_score``.
        self._in_scoring: bool = False
        self._dead_stones: FrozenSet[Point] = frozenset()

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
    def consecutive_passes(self) -> int:
        """How many passes have been played in a row at the tail of history."""

        return self._consecutive_passes

    @property
    def is_scoring(self) -> bool:
        """True while the engine is in the SCORING phase awaiting confirmation."""

        return self._in_scoring

    @property
    def dead_stones(self) -> FrozenSet[Point]:
        """Current set of stones marked dead (valid only during SCORING)."""

        return self._dead_stones

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
        twin._consecutive_passes = self._consecutive_passes
        twin._in_scoring = self._in_scoring
        twin._dead_stones = self._dead_stones
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
        if self._in_scoring:
            return False, "scoring phase"

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
        self._consecutive_passes = 0
        self._current = self._current.opponent()

        return MoveResult(
            legal=True,
            reason="",
            captured=tuple(captured),
            move=point,
        )

    def pass_move(self) -> None:
        """Standard Go pass.

        Two consecutive passes transition the engine into the SCORING
        phase (see :meth:`enter_scoring`). The caller should then let
        the player review the proposed dead stones and call
        :meth:`confirm_score` (or :meth:`cancel_scoring` to go back).

        This method returns ``None`` in all cases. Callers can check
        :attr:`is_scoring` / :attr:`consecutive_passes` to observe the
        transition.
        """

        if self._result is not None or self._in_scoring:
            return

        self._consecutive_passes += 1
        self._move_number += 1
        self._last_move = None
        self._current = self._current.opponent()

        if self._consecutive_passes >= 2:
            self.enter_scoring()

    def enter_scoring(self) -> None:
        """Transition PLAYING -> SCORING and auto-mark dead stones.

        Idempotent: calling while already in SCORING simply re-runs
        :meth:`auto_mark_dead`.
        """

        if self._result is not None:
            return
        self._in_scoring = True
        self.auto_mark_dead()

    def auto_mark_dead(self) -> None:
        """Recompute the heuristic dead-stone set from the current board.

        Used after ``enter_scoring`` to seed the proposal; callers (e.g.
        the GUI) may also invoke it explicitly to reset any manual
        overrides the user has made.
        """

        if not self._in_scoring:
            return
        self._dead_stones = auto_dead_stones(self._board)

    def toggle_group_dead(self, point: Point) -> None:
        """Flip the entire connected group at ``point`` dead <-> alive.

        No-op outside SCORING, on out-of-bounds points, or on empty
        points. The whole connected group is toggled together, matching
        how a Go player would mark dead stones.
        """

        if not self._in_scoring:
            return
        r, c = point
        if not (0 <= r < self._size and 0 <= c < self._size):
            return
        group = group_at(self._board, point)
        if group is None:
            return

        new_dead = set(self._dead_stones)
        if group & new_dead:
            # Any overlap -> treat group as currently-dead, remove it.
            new_dead.difference_update(group)
        else:
            new_dead.update(group)
        self._dead_stones = frozenset(new_dead)

    def confirm_score(self) -> GameResult:
        """Finalize the SCORING phase using the current dead-stone set.

        Computes Chinese area scoring with dead stones counted as
        territory for the surrounding color and sets the game result.
        Transitions SCORING -> FINISHED.
        """

        if self._result is not None:
            return self._result
        if not self._in_scoring:
            # Be forgiving: auto-enter scoring first so the caller gets
            # a sensible result even if they skipped the transition.
            self.enter_scoring()

        breakdown = score_breakdown(self._board, dead_stones=self._dead_stones)
        if breakdown.black_score > breakdown.white_score:
            winner: Optional[Color] = Color.BLACK
        elif breakdown.white_score > breakdown.black_score:
            winner = Color.WHITE
        else:
            winner = None
        self._result = GameResult(
            winner=winner,
            black_score=breakdown.black_score,
            white_score=breakdown.white_score,
            reason="score",
        )
        self._in_scoring = False
        return self._result

    def cancel_scoring(self) -> None:
        """Return SCORING -> PLAYING, discarding any dead-stone proposal.

        Leaves ``consecutive_passes`` at 1 so a single fresh pass is
        enough to re-enter scoring, matching the intuition that the
        previous pass-pass sequence was "almost" the end.
        """

        if not self._in_scoring or self._result is not None:
            return
        self._in_scoring = False
        self._dead_stones = frozenset()
        self._consecutive_passes = 1

    def resign(self) -> GameResult:
        """Current player resigns; the opponent wins by resignation.

        Area scores (with komi, dead-stone aware if we're in scoring)
        are still recorded for HUD display, but the winner is
        determined by the resignation, not the score. Works in either
        PLAYING or SCORING.
        """

        if self._result is not None:
            return self._result

        loser = self._current
        winner = loser.opponent()
        if self._in_scoring:
            breakdown = score_breakdown(self._board, dead_stones=self._dead_stones)
            black_score = breakdown.black_score
            white_score = breakdown.white_score
        else:
            black_score, white_score = area_score(self._board)
        self._result = GameResult(
            winner=winner,
            black_score=black_score,
            white_score=white_score,
            reason="resignation",
        )
        self._in_scoring = False
        return self._result

    def pass_turn(self) -> GameResult:
        """Legacy single-pass-resigns entry point.

        Retained because the AlphaZero training stack
        (``alphazero/self_play.py`` and ``alphazero/mcts.py``) treats a
        pass as a terminal action. New code (the GUI) should call
        :meth:`pass_move` for the standard two-pass rule or
        :meth:`resign` to concede explicitly.
        """

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
        """End the game with dead-stone-aware Chinese area scoring.

        This bypasses the interactive SCORING phase but still runs the
        :func:`auto_dead_stones` heuristic so the terminal score matches
        what a player would see after confirming the GUI's auto
        proposal. The AlphaZero training pipeline relies on this entry
        point for capped games and benefits from the stronger evaluator;
        the GUI's "skip dead-stone review" escape hatch is the same
        deterministic computation.
        """

        if self._result is not None:
            return self._result

        dead = auto_dead_stones(self._board)
        breakdown = score_breakdown(self._board, dead_stones=dead)
        if breakdown.black_score > breakdown.white_score:
            winner: Optional[Color] = Color.BLACK
        elif breakdown.white_score > breakdown.black_score:
            winner = Color.WHITE
        else:
            winner = None
        self._result = GameResult(
            winner=winner,
            black_score=breakdown.black_score,
            white_score=breakdown.white_score,
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
        self._consecutive_passes = 0
        self._in_scoring = False
        self._dead_stones = frozenset()

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
