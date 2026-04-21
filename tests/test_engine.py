"""Rule-focused tests for :class:`GameEngine`.

The tests here drive the engine with sequences of plays rather than
reaching into its internals. This keeps the tests honest about the public
interface used by the GUI and the test harness.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from go_game import Color, GameEngine


def _play_sequence(engine: GameEngine, moves):
    """Helper that plays a list of (row, col) moves, failing on illegal moves."""

    for move in moves:
        result = engine.play(move)
        assert result.legal, f"move {move} was expected legal, got {result.reason}"


def test_initial_state():
    engine = GameEngine()
    assert engine.current_player is Color.BLACK
    assert engine.last_move is None
    assert engine.captures_by(Color.BLACK) == 0
    assert engine.captures_by(Color.WHITE) == 0
    board = engine.board_state()
    assert all(cell is Color.EMPTY for row in board for cell in row)


def test_turn_alternates_and_rejects_occupied():
    engine = GameEngine()
    assert engine.play((4, 4)).legal
    assert engine.current_player is Color.WHITE
    res = engine.play((4, 4))
    assert not res.legal
    assert "occupied" in res.reason


def test_simple_single_stone_capture():
    engine = GameEngine()
    moves = [
        (0, 1),  # B
        (0, 0),  # W (the stone that will be captured)
        (1, 0),  # B  (captures W at (0,0))
    ]
    _play_sequence(engine, moves)
    board = engine.board_state()
    assert board[0][0] is Color.EMPTY
    assert engine.captures_by(Color.BLACK) == 1


def test_multi_stone_group_capture():
    """Two connected white stones in a corner are captured together."""

    engine = GameEngine()
    # Build the shape step by step with alternating play.
    # White stones at (0,0) and (0,1); black surrounds them.
    sequence = [
        (1, 0),  # B
        (0, 0),  # W
        (1, 1),  # B
        (0, 1),  # W
        (1, 2),  # B — threatens the group
        (8, 8),  # W — elsewhere, not relevant
        (0, 2),  # B — captures W group at (0,0) and (0,1)
    ]
    _play_sequence(engine, sequence)
    board = engine.board_state()
    assert board[0][0] is Color.EMPTY
    assert board[0][1] is Color.EMPTY
    assert engine.captures_by(Color.BLACK) == 2


def test_suicide_is_illegal():
    """White cannot play into a cell fully enclosed by Black."""

    engine = GameEngine()
    sequence = [
        (0, 1),  # B
        (8, 8),  # W (throwaway)
        (1, 0),  # B
        (8, 7),  # W (throwaway)
        (1, 2),  # B
        (8, 6),  # W (throwaway)
        (2, 1),  # B — fully encloses (1,1) with black on all sides
    ]
    _play_sequence(engine, sequence)
    # White to move. Playing (1,1) would be suicide.
    legal, reason = engine.is_legal((1, 1))
    assert not legal
    assert reason == "suicide"


def test_suicide_allowed_when_it_captures():
    """A move that would be suicide becomes legal when it captures first.

    Set up a position where playing white at (1,0) has zero liberties
    *before* removing captured stones, but captures the isolated black
    stone at (0,0) and so ends up with a liberty at (0,0).
    """

    engine = GameEngine()
    sequence = [
        (0, 0),  # B - victim
        (0, 1),  # W
        (2, 0),  # B
        (8, 8),  # W throwaway
        (1, 1),  # B
    ]
    _play_sequence(engine, sequence)
    # Sanity: black at (0,0) has exactly one liberty at (1,0).
    assert engine.current_player is Color.WHITE

    res = engine.play((1, 0))
    assert res.legal
    assert (0, 0) in res.captured
    board = engine.board_state()
    assert board[0][0] is Color.EMPTY
    assert board[1][0] is Color.WHITE
    assert engine.captures_by(Color.WHITE) == 1


def test_ko_rule_prevents_immediate_recapture():
    engine = GameEngine()
    setup = [
        (3, 1),  # B
        (3, 2),  # W
        (4, 0),  # B
        (4, 3),  # W
        (5, 1),  # B
        (5, 2),  # W
        (4, 2),  # B - sacrifice
        (4, 1),  # W - captures B at (4,2)
    ]
    _play_sequence(engine, setup)
    board = engine.board_state()
    assert board[4][2] is Color.EMPTY
    assert board[4][1] is Color.WHITE
    # Black tries to recapture W at (4,1) by playing (4,2). This is ko.
    res = engine.play((4, 2))
    assert not res.legal
    assert res.reason == "ko"


def test_ko_allowed_after_a_threat_elsewhere():
    engine = GameEngine()
    setup = [
        (3, 1), (3, 2),
        (4, 0), (4, 3),
        (5, 1), (5, 2),
        (4, 2), (4, 1),  # B sacrifice, then W captures -> ko shape
    ]
    _play_sequence(engine, setup)
    # Black plays a ko threat elsewhere, white responds, then the ko is
    # no longer "immediate" and black may retake.
    assert engine.play((8, 8)).legal   # B threat
    assert engine.play((7, 8)).legal   # W response
    retake = engine.play((4, 2))
    assert retake.legal
    assert len(retake.captured) == 1


def test_pass_concedes_game():
    engine = GameEngine()
    engine.play((4, 4))  # B
    result = engine.pass_turn()  # W passes -> resigns
    assert result.winner is Color.BLACK
    assert result.reason == "resignation"
    assert engine.is_over
    # Subsequent moves are illegal.
    assert not engine.play((0, 0)).legal


def test_area_score_with_komi():
    """Score an empty board: white wins by komi alone."""

    engine = GameEngine()
    result = engine.finish_by_score()
    assert result.black_score == pytest.approx(0.0)
    assert result.white_score == pytest.approx(2.5)
    assert result.winner is Color.WHITE


def test_area_score_territory_and_stones():
    """Black fills left half, White fills right half; each owns their region."""

    engine = GameEngine()
    size = engine.size
    # Columns 0..3 -> Black stones on column 3; columns 5..8 -> White stones on column 5.
    # Alternate plays.
    moves = []
    for r in range(size):
        moves.append((r, 3))   # B wall
        moves.append((r, 5))   # W wall
    _play_sequence(engine, moves)
    result = engine.finish_by_score()
    # Black: 9 stones on column 3 + territory cols 0..2 (3*9 = 27) = 36.
    # White: 9 stones on column 5 + territory cols 6..8 (3*9 = 27) = 36 + 2.5 komi.
    assert result.black_score == pytest.approx(36.0)
    assert result.white_score == pytest.approx(38.5)
    assert result.winner is Color.WHITE
