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
from go_game.board import empty_board, set_point
from go_game.scoring import (
    auto_dead_stones,
    find_real_eye_points,
    score_breakdown,
)


def _build_board(black=(), white=(), size=9):
    """Construct a board fixture with the listed black/white stones.

    Used for tests that need a precise board shape (false eyes,
    surrounded groups, seki) that would be impractical to reach
    through alternating legal play.
    """

    board = empty_board(size)
    for p in black:
        board = set_point(board, p, Color.BLACK)
    for p in white:
        board = set_point(board, p, Color.WHITE)
    return board


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


def test_two_passes_enter_scoring_then_confirm_finalizes():
    """Two consecutive passes transition the engine into the SCORING phase.

    Confirming the score (with no dead stones marked on this trivial
    position) finalizes the game with standard Chinese area scoring.
    """

    engine = GameEngine()
    assert engine.play((4, 4)).legal  # B places one stone
    engine.pass_move()                # W passes
    assert not engine.is_over
    assert not engine.is_scoring
    assert engine.consecutive_passes == 1
    engine.pass_move()                # B passes -> SCORING
    assert engine.is_scoring
    assert not engine.is_over

    result = engine.confirm_score()
    assert engine.is_over
    assert not engine.is_scoring
    assert engine.result is result
    assert result.reason == "score"
    # With only one live color on the board, the open region is treated as
    # unsettled/dame rather than gifting the whole board as territory.
    assert result.black_score == pytest.approx(1.0)
    assert result.white_score == pytest.approx(2.5)
    assert result.winner is Color.WHITE


def test_pass_then_play_resets_pass_count():
    """A stone played between two passes must not enter scoring."""

    engine = GameEngine()
    assert engine.play((4, 4)).legal   # B
    engine.pass_move()                 # W passes (counter -> 1)
    assert engine.consecutive_passes == 1
    assert not engine.is_scoring
    assert engine.play((3, 3)).legal   # B plays -> counter resets
    assert engine.consecutive_passes == 0
    engine.pass_move()                 # W passes again (counter -> 1)
    assert not engine.is_over
    assert not engine.is_scoring


def test_resign_sets_opponent_winner():
    """Explicit resignation hands the win to the opponent regardless of score."""

    engine = GameEngine()
    assert engine.play((4, 4)).legal   # B places, W to move
    result = engine.resign()           # W resigns
    assert engine.is_over
    assert result.winner is Color.BLACK
    assert result.reason == "resignation"
    # Subsequent moves are illegal.
    assert not engine.play((0, 0)).legal


def test_resign_during_scoring_finalizes_game():
    """Resigning from the scoring phase still ends the game by resignation."""

    engine = GameEngine()
    engine.play((4, 4))  # B -> W to move
    engine.pass_move()    # W pass -> B to move
    engine.pass_move()    # B pass -> scoring, current=W
    assert engine.is_scoring
    assert engine.current_player is Color.WHITE
    result = engine.resign()  # W resigns -> B wins
    assert engine.is_over
    assert not engine.is_scoring
    assert result.reason == "resignation"
    assert result.winner is Color.BLACK


def test_play_during_scoring_is_rejected():
    """No stones may be placed while the engine is in the SCORING phase."""

    engine = GameEngine()
    engine.play((4, 4))
    engine.pass_move()
    engine.pass_move()
    assert engine.is_scoring
    result = engine.play((0, 0))
    assert not result.legal
    assert result.reason == "scoring phase"


def test_cancel_scoring_returns_to_play():
    """cancel_scoring goes back to PLAYING and preserves almost-ended state."""

    engine = GameEngine()
    engine.play((4, 4))
    engine.pass_move()
    engine.pass_move()
    assert engine.is_scoring
    engine.cancel_scoring()
    assert not engine.is_scoring
    assert not engine.is_over
    # A single fresh pass should re-enter scoring.
    assert engine.consecutive_passes == 1
    engine.pass_move()
    assert engine.is_scoring


def test_toggle_group_dead_round_trip():
    """Toggling a group dead/alive reflects in dead_stones and score."""

    engine = GameEngine()
    # Place an uncontested black stone so auto_mark_dead leaves it alive.
    engine.play((4, 4))
    engine.pass_move()
    engine.pass_move()
    assert engine.is_scoring
    assert engine.dead_stones == frozenset()

    engine.toggle_group_dead((4, 4))
    assert engine.dead_stones == frozenset({(4, 4)})
    # With the lone stone marked dead, the board reads as fully empty ->
    # bordering set empty -> all 81 points are dame.
    breakdown = score_breakdown(engine.board_grid, dead_stones=engine.dead_stones)
    assert breakdown.black_score == pytest.approx(0.0)
    assert breakdown.white_score == pytest.approx(2.5)

    engine.toggle_group_dead((4, 4))
    assert engine.dead_stones == frozenset()


def test_auto_mark_dead_isolates_surrounded_group():
    """A small enemy group surrounded by a 2-eyed live group is auto-dead."""

    engine = GameEngine()
    # Layout: an 'L' of White stones penned inside a Black wall with
    # two eyes. Plays are interleaved B/W.
    #
    # Rough picture (rows 0..5, cols 0..5), '.' = empty, 'B'/'W' = stone:
    #
    #   B B B B B .
    #   B . W . B .
    #   B W W . B .
    #   B B B B B .
    #   . . . . . .
    #
    # Black wall around the rectangle rows 1..2, cols 1..3. Inside:
    # two empty eye points at (1,1) and (1,3), and a white 'L' group at
    # (1,2)/(2,1)/(2,2). The black wall group has 2 eyes -> alive. The
    # white L has no liberties outside the enclosure -> dead.
    black = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 4),
        (2, 0), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
    ]
    white = [(1, 2), (2, 1), (2, 2)]
    # Interleave. Need len(black) >= len(white); use "extra" white
    # stones far away so turn alternation stays legal.
    extra_white = [(8, 8), (8, 7), (8, 6), (8, 5), (8, 4), (8, 3), (8, 2), (8, 1), (8, 0), (7, 0), (7, 1)]
    wq = white + extra_white
    seq = []
    for i, b in enumerate(black):
        seq.append(b)
        if i < len(wq):
            seq.append(wq[i])
    _play_sequence(engine, seq)
    engine.pass_move()
    engine.pass_move()
    assert engine.is_scoring

    dead = engine.dead_stones
    for p in white:
        assert p in dead, f"expected white {p} auto-marked dead, got {sorted(dead)}"
    # Black wall stones should all be alive.
    for p in black:
        assert p not in dead


def test_auto_mark_dead_leaves_uncontested_group_alive():
    """A lone stone with no enemies nearby should not be auto-killed."""

    engine = GameEngine()
    engine.play((4, 4))
    engine.pass_move()
    engine.pass_move()
    assert engine.is_scoring
    # auto heuristic: group has <2 eyes but is not contested, so alive.
    assert (4, 4) not in engine.dead_stones


def test_score_breakdown_counts_dead_opponent_stones_as_territory():
    """Dead opponent stones become territory + captured_dead for the surrounder."""

    engine = GameEngine()
    # Same layout as ``test_auto_mark_dead_isolates_surrounded_group``.
    black = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 4),
        (2, 0), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
    ]
    white = [(1, 2), (2, 1), (2, 2)]
    extra_white = [(8, 8), (8, 7), (8, 6), (8, 5), (8, 4), (8, 3), (8, 2), (8, 1), (8, 0), (7, 0), (7, 1)]
    wq = white + extra_white
    seq = []
    for i, b in enumerate(black):
        seq.append(b)
        if i < len(wq):
            seq.append(wq[i])
    _play_sequence(engine, seq)

    dead = auto_dead_stones(engine.board_grid)
    for p in white:
        assert p in dead

    breakdown = score_breakdown(engine.board_grid, dead_stones=dead)
    # Enclosed points (1,1) and (1,3) + the 3 dead white stones => 5
    # territory points for Black.
    assert breakdown.black_captured_dead == 3
    assert breakdown.black_territory >= 5
    for p in white:
        assert breakdown.territory_owner[p] is Color.BLACK


def test_score_breakdown_reports_territory_and_dame():
    """Touching B/W walls produce two single-color regions and dame in between."""

    engine = GameEngine()
    # Two touching walls on columns 3 and 4 leave no empty cells between them
    # but split the board into B-only territory (cols 0..2) and W-only
    # territory (cols 5..8). To create dame, plant adjacent stones in the
    # same row instead.
    # Layout we drive:
    #   row 0: B at (0,0), W at (0,1) -> the rest of row 0 is empty and
    #   borders both colors via flood-fill, becoming dame along with the
    #   rest of the board.
    assert engine.play((0, 0)).legal   # B
    assert engine.play((0, 1)).legal   # W
    breakdown = score_breakdown(engine.board_grid)
    # 1 stone each, no territory, all 79 remaining empty points are dame.
    assert breakdown.black_stones == 1
    assert breakdown.white_stones == 1
    assert breakdown.black_territory == 0
    assert breakdown.white_territory == 0
    assert breakdown.dame == 79
    assert breakdown.black_score == pytest.approx(1.0)
    assert breakdown.white_score == pytest.approx(1.0 + 2.5)
    assert (4, 4) in breakdown.dame_points
    assert (0, 0) not in breakdown.territory_owner

    # A second case: a fully-walled-off black region produces territory, not dame.
    engine2 = GameEngine()
    # Build a 3x3 black box at top-left so (0,0)..(0,1)..(1,0) are inside it.
    walls = [
        (0, 2), (1, 2), (2, 2),  # right wall (B)
        (2, 0), (2, 1),          # bottom wall (B)
        # White stones placed far away so turn alternation works without
        # interfering with the black region.
        (8, 8), (8, 7), (8, 6), (7, 8), (6, 8),
    ]
    # Alternate B/W placements covering ``walls`` (B on B-wall coords, W on
    # W-coords). Order them by side so play() accepts them.
    seq = [
        (0, 2),  # B
        (8, 8),  # W
        (1, 2),  # B
        (8, 7),  # W
        (2, 2),  # B
        (8, 6),  # W
        (2, 0),  # B
        (7, 8),  # W
        (2, 1),  # B
        (6, 8),  # W
    ]
    _play_sequence(engine2, seq)
    breakdown2 = score_breakdown(engine2.board_grid)
    # The 2x2 region {(0,0),(0,1),(1,0),(1,1)} is bordered only by Black.
    assert breakdown2.territory_owner.get((0, 0)) is Color.BLACK
    assert breakdown2.territory_owner.get((1, 1)) is Color.BLACK
    assert breakdown2.black_territory >= 4


def test_single_live_color_open_region_counts_as_dame():
    """A lone live color should not inherit every remaining empty point."""

    engine = GameEngine()
    assert engine.play((4, 4)).legal
    breakdown = score_breakdown(engine.board_grid)
    assert breakdown.black_stones == 1
    assert breakdown.white_stones == 0
    assert breakdown.black_territory == 0
    assert breakdown.white_territory == 0
    assert breakdown.dame == 80
    assert breakdown.black_score == pytest.approx(1.0)
    assert breakdown.white_score == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# Real-eye / false-eye / seki regression tests
# --------------------------------------------------------------------------- #


def test_real_eye_in_corner_requires_diagonal():
    """A 1-1 corner eye must have its single on-board diagonal friendly."""

    # Black "owns" (0,0) as an eye: orthogonal neighbours (1,0) and (0,1)
    # are black, and the lone on-board diagonal (1,1) is also black.
    board = _build_board(black=[(1, 0), (0, 1), (1, 1)])
    eyes = find_real_eye_points(board)
    assert eyes.get((0, 0)) is Color.BLACK


def test_false_eye_on_edge_with_enemy_diagonal_is_excluded():
    """An edge point with an enemy diagonal stone is a false eye."""

    # (0,1) has all orthogonal neighbours black or off-board, but (1,0) is
    # white, breaking the "control both diagonals" rule for an edge eye.
    board = _build_board(
        black=[(0, 0), (0, 2), (1, 1)],
        white=[(1, 0)],
    )
    eyes = find_real_eye_points(board)
    assert (0, 1) not in eyes


def test_real_eye_in_center_allows_one_enemy_diagonal():
    """A center eye is real if at least 3 of 4 diagonals are friendly."""

    board = _build_board(
        black=[(3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5), (5, 3)],
        white=[(5, 5)],
    )
    eyes = find_real_eye_points(board)
    assert eyes.get((4, 4)) is Color.BLACK


def test_false_eye_in_center_with_two_enemy_diagonals_is_excluded():
    """Two enemy diagonals reduce a centre eye to false."""

    board = _build_board(
        black=[(3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5)],
        white=[(5, 3), (5, 5)],
    )
    eyes = find_real_eye_points(board)
    assert (4, 4) not in eyes


def test_auto_dead_kills_group_with_only_a_false_eye():
    """A group whose sole eye is false has 0 effective eyes.

    Set-up: a tiny black "eye" surrounded by white where the diagonal
    rule fails, so the eye is false. Black is contested by the
    surrounding white wall and has no other eye-shape, so the
    heuristic should consider Black dead. White, with plenty of
    liberties, is safeguarded as alive.
    """

    # Black stones forming an enclosure with a single 1-point empty at
    # (1,1). White owns the diagonal (2,2), making (1,1) a false eye.
    black = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1)]
    # White wall on the outside of black plus the killer diagonal.
    white = [
        (2, 0), (3, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 3),
        (2, 2),
    ]
    board = _build_board(black=black, white=white)
    eyes = find_real_eye_points(board)
    assert (1, 1) not in eyes  # false eye, not real
    dead = auto_dead_stones(board)
    for p in black:
        assert p in dead, f"expected black {p} marked dead, got {sorted(dead)}"
    for p in white:
        assert p not in dead


def test_auto_dead_seki_protection_keeps_short_shared_lib_groups_alive():
    """Two contested short groups sharing a liberty are treated as seki.

    The position is a constructed corner shape where a short black and
    a short white group both have low liberty counts and share at
    least one empty intersection in their borders. Neither has two
    eyes, but the seki rule (shared liberty + both under the safeguard
    threshold) should keep both groups alive.
    """

    # Black L: (2,1),(2,2),(3,1)
    # White L: (3,2),(4,2),(4,3)
    # They touch at (2,2)-(3,2) and (3,1)-(3,2). Their borders both
    # include (4,1), giving a shared empty boundary point.
    black = [(2, 1), (2, 2), (3, 1)]
    white = [(3, 2), (4, 2), (4, 3)]
    board = _build_board(black=black, white=white)
    dead = auto_dead_stones(board)
    for p in black + white:
        assert p not in dead, (
            f"expected {p} alive (seki), got dead set {sorted(dead)}"
        )


def test_auto_dead_kills_surrounded_white_with_dominant_black_wall():
    """Sanity: even with seki protection, a clearly-surrounded group dies.

    The black wall has many liberties (above the safeguard threshold)
    so it is alive on its own merits, which means it cannot be a seki
    partner for the surrounded white stones, which are correctly
    auto-marked dead.
    """

    black_wall = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 4),
        (2, 0), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
    ]
    white_inside = [(1, 2), (2, 1), (2, 2)]
    # An external white wall far away keeps Black "contested" overall
    # (so the heuristic actually evaluates Black) without changing the
    # surrounded group's life status.
    extra_white = [
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4),
        (8, 5), (8, 6), (8, 7), (8, 8),
        (7, 0), (7, 1),
    ]
    board = _build_board(
        black=black_wall,
        white=white_inside + extra_white,
    )
    dead = auto_dead_stones(board)
    for p in white_inside:
        assert p in dead
    for p in black_wall + extra_white:
        assert p not in dead


def test_finish_by_score_uses_dead_stone_aware_scoring():
    """The training-pipeline terminal scorer credits dead enemies as territory.

    A capped self-play game that ends with ``finish_by_score`` should
    apply the same dead-stone-aware Chinese area scoring the GUI uses,
    not naive area scoring. Without the auto-dead pass, the surrounded
    white stones below would still count toward White's score.
    """

    engine = GameEngine()
    black_wall = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 4),
        (2, 0), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
    ]
    white_inside = [(1, 2), (2, 1), (2, 2)]
    extra_white = [
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4),
        (8, 5), (8, 6), (8, 7), (8, 8),
        (7, 0), (7, 1),
    ]
    seq = []
    wq = white_inside + extra_white
    for i, b in enumerate(black_wall):
        seq.append(b)
        if i < len(wq):
            seq.append(wq[i])
    _play_sequence(engine, seq)

    result = engine.finish_by_score()
    assert result.reason == "score"
    # Auto-dead-aware breakdown for direct comparison.
    expected = score_breakdown(
        engine.board_grid, dead_stones=auto_dead_stones(engine.board_grid)
    )
    assert result.black_score == pytest.approx(expected.black_score)
    assert result.white_score == pytest.approx(expected.white_score)
    # And the surrounded white stones must be credited to Black.
    assert expected.black_captured_dead == len(white_inside)
    assert result.winner is Color.BLACK
