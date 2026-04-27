"""Chinese (area) scoring for a finished 9x9 Go game.

Two entry points:

* :func:`score_breakdown` -- the source of truth. Performs the flood fill
  exactly once and returns a :class:`ScoreBreakdown` with per-region
  ownership, dame points, stone counts, territory counts, and final scores.
  Accepts an optional ``dead_stones`` set that is treated as empty for
  region flood-fill but is still counted as territory for the
  surrounding color (dead opponent stones become territory).
* :func:`area_score` -- thin backwards-compatible wrapper that returns
  just ``(black_score, white_score)`` without dead-stone awareness.

Also exposes the life-and-death heuristic used by the GUI's scoring
phase to pre-populate dead stones (:func:`auto_dead_stones`) and the
helper that backs it (:func:`find_real_eye_points`).

The heuristic implements the textbook "two eyes live, one eye dies"
rule with three refinements aimed at avoiding both false positives and
false negatives in the GUI proposal:

* False eyes are recognised via the standard diagonal rule (center
  positions need 3 of 4 diagonals controlled by the owner; edge and
  corner positions need every on-board diagonal controlled, with
  off-board diagonals counted as friendly). A point that fails this
  rule is not counted as an eye, even if its orthogonal neighbours are
  all friendly.
* A group that already encloses a "big" single-color empty region (>=
  ``BIG_EYE_REGION_SIZE`` points) is treated as having two eyes' worth
  of potential, since the owner can almost always split such a region
  into two real eyes. This keeps a freshly enclosing wall alive after
  its surrounded enemy is auto-marked dead.
* A group with at least ``LIBERTY_SAFEGUARD`` liberties is treated as
  alive regardless of eye count, and a contested pair of opposite-color
  groups that share an empty intersection (a shared liberty) and both
  fall under the safeguard threshold is treated as a seki and left
  alive together.

Users can override any of this with the click-to-toggle dead-stone UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from .board import BoardGrid, neighbors
from .types import Color, Point

KOMI = 2.5

EMPTY_DEAD_SET: FrozenSet[Point] = frozenset()

# A single-color enclosed region of at least this many empty points is
# treated as enough space to make two real eyes. Tuned conservatively
# for 9x9: a 6-point enclosure can almost always be divided into two
# eyes by the owner if forced to.
BIG_EYE_REGION_SIZE = 6

# Groups with at least this many liberties are not auto-marked dead,
# even when the eye count is below 2. This is a guard against the
# heuristic killing unsettled but clearly-not-yet-dead groups during a
# mid-game "score game" preview.
LIBERTY_SAFEGUARD = 7


@dataclass(frozen=True)
class ScoreBreakdown:
    """Detailed Chinese-area-scoring decomposition for a board position.

    ``black_score`` / ``white_score`` are what callers should display as
    the final result. ``territory_owner`` and ``dame_points`` exist so
    the GUI can paint the post-game territory overlay.

    ``black_captured_dead`` is the number of opponent (white) stones
    marked dead inside black's enclosed regions and likewise for white.
    These are already rolled into ``black_territory`` / ``white_territory``.
    """

    black_stones: int
    white_stones: int
    black_territory: int
    white_territory: int
    black_captured_dead: int
    white_captured_dead: int
    dame: int
    komi: float
    black_score: float
    white_score: float
    territory_owner: Dict[Point, Color] = field(default_factory=dict)
    dame_points: FrozenSet[Point] = field(default_factory=frozenset)


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #


def _effective_value(
    board: BoardGrid, point: Point, dead_stones: FrozenSet[Point]
) -> Color:
    """Return the board value with dead stones coerced to :class:`Color.EMPTY`."""

    if point in dead_stones:
        return Color.EMPTY
    r, c = point
    return board[r][c]


def find_groups(board: BoardGrid) -> List[FrozenSet[Point]]:
    """Return every connected same-color stone group on ``board``."""

    size = len(board)
    seen: Set[Point] = set()
    groups: List[FrozenSet[Point]] = []
    for r in range(size):
        for c in range(size):
            start = (r, c)
            if start in seen:
                continue
            color = board[r][c]
            if color is Color.EMPTY:
                continue
            stack = [start]
            group: Set[Point] = set()
            while stack:
                cur = stack.pop()
                if cur in group:
                    continue
                group.add(cur)
                for nb in neighbors(cur, size):
                    nr, nc = nb
                    if board[nr][nc] is color and nb not in group:
                        stack.append(nb)
            seen |= group
            groups.append(frozenset(group))
    return groups


def find_eye_regions(
    board: BoardGrid,
    dead_stones: FrozenSet[Point] = EMPTY_DEAD_SET,
) -> Dict[FrozenSet[Point], Color]:
    """Empty regions whose only bordering live color is a single color.

    Dead stones count as empty for the flood-fill but they do not
    contribute a "bordering color" (so an enclosure full of dead
    opponents still counts as single-color territory).
    """

    size = len(board)
    visited: Set[Point] = set()
    result: Dict[FrozenSet[Point], Color] = {}

    for r in range(size):
        for c in range(size):
            start = (r, c)
            if start in visited:
                continue
            if _effective_value(board, start, dead_stones) is not Color.EMPTY:
                continue

            region: Set[Point] = set()
            bordering: Set[Color] = set()
            stack = [start]
            while stack:
                cur = stack.pop()
                if cur in region:
                    continue
                region.add(cur)
                for nb in neighbors(cur, size):
                    val = _effective_value(board, nb, dead_stones)
                    if val is Color.EMPTY and nb not in region:
                        stack.append(nb)
                    elif val is Color.BLACK or val is Color.WHITE:
                        bordering.add(val)
            visited |= region
            if len(bordering) == 1:
                result[frozenset(region)] = next(iter(bordering))
    return result


def _diagonals(point: Point, size: int) -> List[Point]:
    """Return the in-bounds diagonal neighbours of ``point``."""

    r, c = point
    candidates = ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1))
    return [p for p in candidates if 0 <= p[0] < size and 0 <= p[1] < size]


def _is_real_eye_point(
    board: BoardGrid,
    point: Point,
    color: Color,
    dead_stones: FrozenSet[Point],
    size: int,
) -> bool:
    """True if ``point`` is a real (non-false) eye for ``color``.

    Standard diagonal rule:

      * every orthogonal neighbour must be ``color`` (off-board edges
        count as friendly because the opponent can never play there);
      * for a center position (4 on-board diagonals) at least 3 of the
        4 diagonals must be friendly stones;
      * for an edge or corner position (fewer on-board diagonals) every
        on-board diagonal must be a friendly stone, with off-board
        diagonals counted as friendly.

    Dead stones are coerced to empty before the check, so a dead
    opponent at a diagonal does not count as a friendly stone.
    """

    if _effective_value(board, point, dead_stones) is not Color.EMPTY:
        return False

    for nb in neighbors(point, size):
        if _effective_value(board, nb, dead_stones) is not color:
            return False

    diags = _diagonals(point, size)
    off_board = 4 - len(diags)
    friendly = off_board
    for d in diags:
        if _effective_value(board, d, dead_stones) is color:
            friendly += 1

    if len(diags) == 4:
        return friendly >= 3
    return friendly >= 4


def find_real_eye_points(
    board: BoardGrid,
    dead_stones: FrozenSet[Point] = EMPTY_DEAD_SET,
) -> Dict[Point, Color]:
    """Return every empty intersection that is a real eye, mapped to its owner.

    A given intersection is an eye for at most one colour because the
    orthogonal-neighbour rule rules out the other.
    """

    dead = frozenset(dead_stones)
    size = len(board)
    out: Dict[Point, Color] = {}
    for r in range(size):
        for c in range(size):
            point = (r, c)
            if _effective_value(board, point, dead) is not Color.EMPTY:
                continue
            for color in (Color.BLACK, Color.WHITE):
                if _is_real_eye_point(board, point, color, dead, size):
                    out[point] = color
                    break
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_breakdown(
    board: BoardGrid,
    dead_stones: Iterable[Point] = EMPTY_DEAD_SET,
) -> ScoreBreakdown:
    """Compute a full Chinese-area-scoring breakdown for ``board``.

    ``dead_stones`` may be any iterable of points; they are treated as
    empty for the flood-fill and credited as territory for the
    surrounding (opposite) color.
    """

    dead: FrozenSet[Point] = frozenset(dead_stones)
    size = len(board)

    black_stones = 0
    white_stones = 0
    black_territory = 0
    white_territory = 0
    black_captured_dead = 0
    white_captured_dead = 0
    territory_owner: Dict[Point, Color] = {}
    dame_points: Set[Point] = set()

    for r in range(size):
        for c in range(size):
            effective = _effective_value(board, (r, c), dead)
            if effective is Color.BLACK:
                black_stones += 1
            elif effective is Color.WHITE:
                white_stones += 1

    visited_empty: Set[Point] = set()

    for r in range(size):
        for c in range(size):
            point = (r, c)
            effective = _effective_value(board, point, dead)

            if effective is Color.BLACK:
                continue
            if effective is Color.WHITE:
                continue

            # Effective EMPTY -- either truly empty, or a dead stone.
            if point in visited_empty:
                continue

            region: Set[Point] = set()
            bordering: Set[Color] = set()
            stack = [point]
            while stack:
                cur = stack.pop()
                if cur in region:
                    continue
                region.add(cur)
                for nb in neighbors(cur, size):
                    val = _effective_value(board, nb, dead)
                    if val is Color.EMPTY and nb not in region:
                        stack.append(nb)
                    elif val is Color.BLACK or val is Color.WHITE:
                        bordering.add(val)
            visited_empty |= region

            if len(bordering) == 1:
                owner = next(iter(bordering))
                opponent_has_live_stones = (
                    white_stones > 0 if owner is Color.BLACK else black_stones > 0
                )
                region_has_dead_opponent = False
                for p in region:
                    if p not in dead:
                        continue
                    pr, pc = p
                    if board[pr][pc] is owner.opponent():
                        region_has_dead_opponent = True
                        break
                if not opponent_has_live_stones and not region_has_dead_opponent:
                    dame_points.update(region)
                    continue
                # Every point in the region (empty or dead stone) is
                # territory for the sole bordering color.
                for p in region:
                    territory_owner[p] = owner
                    if p in dead:
                        # Count the dead opponent stone as captured.
                        pr, pc = p
                        original = board[pr][pc]
                        if original is Color.BLACK and owner is Color.WHITE:
                            white_captured_dead += 1
                        elif original is Color.WHITE and owner is Color.BLACK:
                            black_captured_dead += 1
                if owner is Color.BLACK:
                    black_territory += len(region)
                else:
                    white_territory += len(region)
            else:
                dame_points.update(region)

    black_score = float(black_stones + black_territory)
    white_score = float(white_stones + white_territory) + KOMI

    return ScoreBreakdown(
        black_stones=black_stones,
        white_stones=white_stones,
        black_territory=black_territory,
        white_territory=white_territory,
        black_captured_dead=black_captured_dead,
        white_captured_dead=white_captured_dead,
        dame=len(dame_points),
        komi=KOMI,
        black_score=black_score,
        white_score=white_score,
        territory_owner=territory_owner,
        dame_points=frozenset(dame_points),
    )


def area_score(board: BoardGrid) -> Tuple[float, float]:
    """Return ``(black_score, white_score)`` using naive Chinese area scoring.

    Thin wrapper over :func:`score_breakdown` that does not know about
    dead stones; callers that care about the scoring-phase flow should
    use :func:`score_breakdown` directly.
    """

    breakdown = score_breakdown(board)
    return breakdown.black_score, breakdown.white_score


# --------------------------------------------------------------------------- #
# Life-and-death heuristic
# --------------------------------------------------------------------------- #


def _group_neighbors(
    group: FrozenSet[Point], size: int
) -> Set[Point]:
    """All intersections adjacent to any stone of the group."""

    out: Set[Point] = set()
    for p in group:
        for nb in neighbors(p, size):
            if nb not in group:
                out.add(nb)
    return out


def _liberty_count(
    board: BoardGrid,
    group: FrozenSet[Point],
    dead_stones: FrozenSet[Point],
    size: int,
) -> int:
    """Number of distinct empty intersections adjacent to ``group``.

    Dead stones are treated as empty so the count reflects the
    effective (post-dead-removal) liberty picture.
    """

    libs: Set[Point] = set()
    for p in group:
        for nb in neighbors(p, size):
            if _effective_value(board, nb, dead_stones) is Color.EMPTY:
                libs.add(nb)
    return len(libs)


def auto_dead_stones(board: BoardGrid) -> FrozenSet[Point]:
    """Iterative life-and-death heuristic with false-eye and seki awareness.

    A group is marked dead iff it is *contested* (adjacent to at least
    one live opposing stone), has an effective eye count below 2, has
    fewer than ``LIBERTY_SAFEGUARD`` actual liberties, and is not in a
    potential seki with another candidate-kill opponent group.

    The eye count of a group ``G`` of colour ``C`` is

        max(real_eye_points_adjacent_to_G,
            2 if G borders a single-colour empty region of size
              >= BIG_EYE_REGION_SIZE else 0)

    A "real eye" is a single empty intersection passing the diagonal
    rule in :func:`_is_real_eye_point`; this is how false eyes are
    excluded from the count. The "big region" carve-out keeps a
    surrounding wall alive in cases where the wall has not yet shaped
    its interior into two distinct eye points but has plenty of space
    to do so.

    Two contested opposite-colour groups that both fall below the
    liberty safeguard *and* share at least one empty intersection in
    their borders are treated as a seki pair: neither is auto-marked
    dead, regardless of eye count. The user can still mark them dead
    manually.

    Smallest contested groups are evaluated first so a small enclosed
    group is killed before the surrounding enclosure is tested, which
    means the enclosure gets to count the freed interior as a big eye
    region on the next pass. Uncontested groups are never killed by the
    heuristic -- there is no opponent around to capture them, so we let
    the user mark them dead manually if they really are.

    The heuristic intentionally errs toward leaving groups alive: any
    misclassification can be corrected in the GUI with a single click.
    """

    size = len(board)
    groups = list(find_groups(board))
    # Smallest-first evaluation order.
    groups.sort(key=len)

    dead: Set[Point] = set()
    changed = True
    while changed:
        changed = False
        dead_frozen = frozenset(dead)

        eye_points = find_real_eye_points(board, dead_frozen)
        eye_by_color: Dict[Color, Set[Point]] = {
            Color.BLACK: set(),
            Color.WHITE: set(),
        }
        for p, c in eye_points.items():
            eye_by_color[c].add(p)

        single_color_regions = find_eye_regions(board, dead_frozen)
        regions_by_color: Dict[Color, List[FrozenSet[Point]]] = {
            Color.BLACK: [],
            Color.WHITE: [],
        }
        for region, color in single_color_regions.items():
            regions_by_color[color].append(region)

        # First pass: gather per-group eye_score and liberties for every
        # *contested* live group.
        eye_score: Dict[FrozenSet[Point], int] = {}
        liberties: Dict[FrozenSet[Point], int] = {}
        for group in groups:
            sample = next(iter(group))
            if sample in dead:
                continue
            r, c = sample
            color = board[r][c]
            opponent = color.opponent()
            border = _group_neighbors(group, size)

            is_contested = any(
                board[pr][pc] is opponent and (pr, pc) not in dead
                for (pr, pc) in border
            )
            if not is_contested:
                continue

            real_eye_count = sum(1 for p in border if p in eye_by_color[color])
            big_region_bonus = 0
            for region in regions_by_color[color]:
                if region & border and len(region) >= BIG_EYE_REGION_SIZE:
                    big_region_bonus = 2
                    break
            eye_score[group] = max(real_eye_count, big_region_bonus)
            liberties[group] = _liberty_count(board, group, dead_frozen, size)

        # A group is a kill candidate when it has < 2 effective eyes AND
        # fewer than the safeguard threshold of liberties.
        candidate_kill = {
            g
            for g, score in eye_score.items()
            if score < 2 and liberties[g] < LIBERTY_SAFEGUARD
        }

        # Seki pairing: two opposite-color candidates that share an
        # empty boundary point are left alive together.
        empty_borders: Dict[FrozenSet[Point], Set[Point]] = {}
        group_color: Dict[FrozenSet[Point], Color] = {}
        for g in candidate_kill:
            r, c = next(iter(g))
            group_color[g] = board[r][c]
            empty_borders[g] = {
                p
                for p in _group_neighbors(g, size)
                if _effective_value(board, p, dead_frozen) is Color.EMPTY
            }

        seki: Set[FrozenSet[Point]] = set()
        candidates = list(candidate_kill)
        for i, g1 in enumerate(candidates):
            for g2 in candidates[i + 1 :]:
                if group_color[g1] is group_color[g2]:
                    continue
                if empty_borders[g1] & empty_borders[g2]:
                    seki.add(g1)
                    seki.add(g2)

        for g in candidate_kill:
            if g in seki:
                continue
            dead.update(g)
            changed = True

    return frozenset(dead)


def group_at(board: BoardGrid, point: Point) -> Optional[FrozenSet[Point]]:
    """Return the connected group of stones containing ``point``, or None if empty."""

    r, c = point
    size = len(board)
    if not (0 <= r < size and 0 <= c < size):
        return None
    color = board[r][c]
    if color is Color.EMPTY:
        return None

    group: Set[Point] = set()
    stack = [point]
    while stack:
        cur = stack.pop()
        if cur in group:
            continue
        group.add(cur)
        for nb in neighbors(cur, size):
            nr, nc = nb
            if board[nr][nc] is color and nb not in group:
                stack.append(nb)
    return frozenset(group)
