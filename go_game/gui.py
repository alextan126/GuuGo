"""Pygame GUI for GuuGo (PvP + PvE).

The GUI is a thin presentation layer over :class:`GameEngine`. It draws
the board, handles click-to-play, highlights the most recent move,
shows capture counts, and surfaces the game result. All rules logic
lives in the engine.

We use pygame because it is a single pure-pip dependency (``pip install
pygame``) with no system packages required, which makes the app trivial
to ship to a grader: ``pip install --user -r requirements.txt`` then
``python main.py``. No virtualenvs needed.

PvE mode is opt-in via the ``ai_player`` argument on :class:`GoGUI`.
When set, the human plays Black and the AI plays White; the AI runs
MCTS on a background thread so the pygame loop keeps drawing while it
thinks.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Tuple

import pygame

from .engine import GameEngine
from .types import Color, Point

BOARD_MARGIN = 48
CELL_SIZE = 60
STONE_RADIUS = 26
HUD_HEIGHT = 110
BANNER_HEIGHT = 40

BOARD_BG = (221, 176, 107)
WINDOW_BG = (245, 238, 224)
LINE_COLOR = (34, 34, 34)
STAR_COLOR = (40, 40, 40)
BLACK_FILL = (17, 17, 17)
WHITE_FILL = (245, 245, 245)
WHITE_OUTLINE = (50, 50, 50)
HIGHLIGHT_COLOR = (226, 61, 61)
TEXT_COLOR = (28, 28, 28)
MUTED_COLOR = (80, 80, 80)
BUTTON_BG = (235, 225, 205)
BUTTON_BG_HOVER = (220, 205, 175)
BUTTON_BORDER = (120, 100, 60)
BUTTON_TEXT = (30, 30, 30)
BANNER_BG = (50, 50, 55)
BANNER_TEXT = (240, 240, 240)

STAR_POINTS = ((2, 2), (2, 4), (2, 6), (4, 2), (4, 4), (4, 6), (6, 2), (6, 4), (6, 6))


class _Button:
    """Minimal click-and-hover rectangular button used in the HUD."""

    def __init__(self, rect: pygame.Rect, label: str, action: str) -> None:
        self.rect = rect
        self.label = label
        self.action = action
        self.hover = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        bg = BUTTON_BG_HOVER if self.hover else BUTTON_BG
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, BUTTON_BORDER, self.rect, width=1, border_radius=6)
        text = font.render(self.label, True, BUTTON_TEXT)
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)


class GoGUI:
    """Pygame application window for PvP or PvE play.

    PvE is opt-in via the ``ai_player`` keyword argument: when provided,
    the AI plays White after every legal human (Black) move, from a
    background thread so the main loop never blocks on MCTS.
    """

    def __init__(
        self,
        engine: Optional[GameEngine] = None,
        *,
        ai_player: Optional[Any] = None,
    ) -> None:
        self.engine = engine or GameEngine()

        self._board_pixels = BOARD_MARGIN * 2 + CELL_SIZE * (self.engine.size - 1)
        self._width = self._board_pixels
        self._height = HUD_HEIGHT + self._board_pixels

        pygame.init()
        caption = "GuuGo - 9x9 Go"
        if ai_player is not None:
            caption += " (vs AI)"
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode((self._width, self._height))
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("helvetica", 22, bold=True)
        self.hud_font = pygame.font.SysFont("helvetica", 15)
        self.button_font = pygame.font.SysFont("helvetica", 14, bold=True)
        self.banner_font = pygame.font.SysFont("helvetica", 16, bold=True)

        self.buttons = self._build_buttons()
        self._banner_text: Optional[str] = None
        self._banner_timer_ms: int = 0

        # --- AI / PvE state ---
        # The AI plays the White side. MCTS runs in a daemon thread so
        # the GUI stays interactive. The main thread remains the only
        # place that mutates ``self.engine``, which keeps rendering and
        # rules logic race-free.
        self.ai_player = ai_player
        self._ai_color: Color = Color.WHITE
        self._ai_lock = threading.Lock()
        self._ai_thread: Optional[threading.Thread] = None
        self._ai_pending_move: Optional[Optional[Point]] = None
        self._ai_pending_game_id: int = -1
        self._ai_thinking = False
        # ``_game_id`` is bumped on every ``New Game``. The AI thread
        # stamps its result with the id it started under, so a reset
        # while the AI is thinking discards the stale move.
        self._game_id = 0

    # ---------------- layout ----------------

    def _build_buttons(self) -> list:
        labels = [
            ("Pass (Resign)", "pass"),
            ("Score Game", "score"),
            ("New Game", "new"),
        ]
        button_w, button_h = 130, 32
        gap = 8
        total_w = len(labels) * button_w + (len(labels) - 1) * gap
        start_x = self._width - total_w - 16
        y = HUD_HEIGHT - button_h - 14

        buttons = []
        for i, (label, action) in enumerate(labels):
            rect = pygame.Rect(start_x + i * (button_w + gap), y, button_w, button_h)
            buttons.append(_Button(rect, label, action))
        return buttons

    def _point_to_pixel(self, point: Point) -> Tuple[int, int]:
        r, c = point
        x = BOARD_MARGIN + c * CELL_SIZE
        y = HUD_HEIGHT + BOARD_MARGIN + r * CELL_SIZE
        return x, y

    def _pixel_to_point(self, x: int, y: int) -> Optional[Point]:
        size = self.engine.size
        board_y = y - HUD_HEIGHT
        if board_y < 0:
            return None
        col = round((x - BOARD_MARGIN) / CELL_SIZE)
        row = round((board_y - BOARD_MARGIN) / CELL_SIZE)
        if 0 <= row < size and 0 <= col < size:
            px, py = self._point_to_pixel((row, col))
            if abs(px - x) <= CELL_SIZE / 2 and abs(py - y) <= CELL_SIZE / 2:
                return (row, col)
        return None

    # ---------------- drawing ----------------

    def _draw_hud(self) -> None:
        pygame.draw.rect(self.screen, WINDOW_BG, pygame.Rect(0, 0, self._width, HUD_HEIGHT))

        if self.engine.is_over:
            result = self.engine.result
            assert result is not None
            if self.ai_player is not None:
                # PvE: frame the outcome from the human's (Black) POV.
                human_color = Color.BLACK
                if result.winner is None:
                    winner_text = "Tie"
                elif result.winner is human_color:
                    winner_text = "You win!"
                else:
                    winner_text = "You lose."
            else:
                if result.winner is None:
                    winner_text = "Tie"
                else:
                    winner_text = f"{result.winner.name.title()} wins"
            title = (
                f"Game over - {winner_text} "
                f"(Black {result.black_score:g}, White {result.white_score:g})"
            )
        else:
            player = self.engine.current_player
            if self.ai_player is not None and player is self._ai_color:
                title = f"{player.name.title()} (AI) to move"
            else:
                title = f"{player.name.title()} to move"

        title_surface = self.title_font.render(title, True, TEXT_COLOR)
        self.screen.blit(title_surface, (16, 14))

        captures_line = (
            f"Black captures: {self.engine.captures_by(Color.BLACK)}    "
            f"White captures: {self.engine.captures_by(Color.WHITE)}    "
            f"Move #: {self.engine.move_number}"
        )
        cap_surface = self.hud_font.render(captures_line, True, MUTED_COLOR)
        self.screen.blit(cap_surface, (16, 46))

        for button in self.buttons:
            button.draw(self.screen, self.button_font)

    def _draw_board(self) -> None:
        size = self.engine.size
        board_rect = pygame.Rect(
            0, HUD_HEIGHT, self._board_pixels, self._board_pixels
        )
        pygame.draw.rect(self.screen, BOARD_BG, board_rect)

        start = BOARD_MARGIN
        end = BOARD_MARGIN + CELL_SIZE * (size - 1)
        for i in range(size):
            offset = BOARD_MARGIN + i * CELL_SIZE
            pygame.draw.line(
                self.screen,
                LINE_COLOR,
                (start, HUD_HEIGHT + offset),
                (end, HUD_HEIGHT + offset),
                1,
            )
            pygame.draw.line(
                self.screen,
                LINE_COLOR,
                (offset, HUD_HEIGHT + start),
                (offset, HUD_HEIGHT + end),
                1,
            )
        for point in STAR_POINTS:
            x, y = self._point_to_pixel(point)
            pygame.draw.circle(self.screen, STAR_COLOR, (x, y), 3)

    def _draw_stones(self) -> None:
        board = self.engine.board_state()
        for r, row in enumerate(board):
            for c, value in enumerate(row):
                if value is Color.EMPTY:
                    continue
                x, y = self._point_to_pixel((r, c))
                if value is Color.BLACK:
                    pygame.draw.circle(self.screen, BLACK_FILL, (x, y), STONE_RADIUS)
                else:
                    pygame.draw.circle(self.screen, WHITE_FILL, (x, y), STONE_RADIUS)
                    pygame.draw.circle(
                        self.screen, WHITE_OUTLINE, (x, y), STONE_RADIUS, width=1
                    )

        last = self.engine.last_move
        if last is not None:
            x, y = self._point_to_pixel(last)
            pygame.draw.circle(
                self.screen, HIGHLIGHT_COLOR, (x, y), STONE_RADIUS // 2, width=3
            )

    def _draw_banner(self) -> None:
        if not self._banner_text:
            return
        rect = pygame.Rect(0, 0, self._width, BANNER_HEIGHT)
        pygame.draw.rect(self.screen, BANNER_BG, rect)
        text = self.banner_font.render(self._banner_text, True, BANNER_TEXT)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def _render(self) -> None:
        self.screen.fill(WINDOW_BG)
        self._draw_hud()
        self._draw_board()
        self._draw_stones()
        self._draw_banner()
        pygame.display.flip()

    # ---------------- interaction ----------------

    def _set_banner(self, text: str, duration_ms: int = 2200) -> None:
        self._banner_text = text
        self._banner_timer_ms = duration_ms

    def _handle_board_click(self, pos: Tuple[int, int]) -> None:
        if self.engine.is_over:
            return
        if self._ai_thinking:
            return
        if self.ai_player is not None and self.engine.current_player is self._ai_color:
            # Shouldn't normally happen (AI would already be thinking)
            # but guards against a stale click sneaking in.
            return
        point = self._pixel_to_point(pos[0], pos[1])
        if point is None:
            return
        result = self.engine.play(point)
        if not result.legal:
            self._set_banner(f"Illegal move: {result.reason}")
            return
        if result.captured:
            self._set_banner(f"Captured {len(result.captured)} stone(s)")
        self._maybe_start_ai_turn()

    def _handle_button(self, action: str) -> None:
        if action == "pass":
            if self.engine.is_over or self._ai_thinking:
                return
            loser = self.engine.current_player
            self.engine.pass_turn()
            self._set_banner(f"{loser.name.title()} passed - concedes game")
        elif action == "score":
            if self.engine.is_over or self._ai_thinking:
                return
            self.engine.finish_by_score()
            self._set_banner("Game scored")
        elif action == "new":
            with self._ai_lock:
                self._game_id += 1
                self._ai_pending_move = None
                self._ai_pending_game_id = -1
                self._ai_thinking = False
            self.engine.reset()
            self._set_banner("New game started")
            self._maybe_start_ai_turn()

    # ---------------- AI turn ----------------

    def _maybe_start_ai_turn(self) -> None:
        """Spawn MCTS on a daemon thread if it's the AI's move."""

        if self.ai_player is None:
            return
        if self.engine.is_over:
            return
        if self.engine.current_player is not self._ai_color:
            return
        if self._ai_thinking:
            return

        self._ai_thinking = True
        self._set_banner("AI is thinking...", duration_ms=60_000)
        engine_snapshot = self.engine.clone()
        game_id = self._game_id
        ai = self.ai_player

        def _run() -> None:
            try:
                move = ai.choose_move(engine_snapshot)
            except Exception:
                move = None
            with self._ai_lock:
                # Only publish if this thread started under the current
                # game; otherwise the user already pressed New Game.
                if game_id == self._game_id:
                    self._ai_pending_move = move
                    self._ai_pending_game_id = game_id

        thread = threading.Thread(target=_run, name="guugo-ai", daemon=True)
        self._ai_thread = thread
        thread.start()

    def _drain_ai_pending(self) -> None:
        """Apply a finished AI move on the main thread (pygame-safe)."""

        if not self._ai_thinking:
            return
        with self._ai_lock:
            pending_game_id = self._ai_pending_game_id
            move = self._ai_pending_move
            thread_done = self._ai_thread is None or not self._ai_thread.is_alive()
            has_pending = pending_game_id == self._game_id and thread_done
            if has_pending:
                self._ai_pending_move = None
                self._ai_pending_game_id = -1
        if not has_pending:
            return

        self._ai_thinking = False
        self._banner_text = None
        self._banner_timer_ms = 0

        if move is None:
            # AI had nothing to play. Two sub-cases:
            #   (a) engine is already over (someone resigned, scored, etc.)
            #   (b) AI has no legal board move and refuses to pass
            # In (b) we end the game by area scoring so the GUI never
            # gets stuck waiting on a phantom AI turn. The human picks
            # this up from the game-over HUD + banner.
            if not self.engine.is_over:
                self.engine.finish_by_score()
                self._set_banner(
                    "AI has no legal move - game scored.", duration_ms=4000
                )
            return
        if self.engine.is_over:
            return

        result = self.engine.play(move)
        if not result.legal:
            # choose_move filters to legal points; if we hit this it's a
            # bug. Surface it rather than silently hanging the AI turn.
            self._set_banner(f"AI suggested illegal move: {result.reason}")
            return
        if result.captured:
            self._set_banner(f"AI captured {len(result.captured)} stone(s)")

    # ---------------- main loop ----------------

    def run(self) -> None:
        # If the game starts on the AI's turn (e.g. configured to play
        # first in a future change), kick it off immediately.
        self._maybe_start_ai_turn()

        running = True
        while running:
            dt = self.clock.tick(60)
            if self._banner_timer_ms > 0:
                self._banner_timer_ms -= dt
                if self._banner_timer_ms <= 0:
                    self._banner_text = None

            self._drain_ai_pending()

            mouse_pos = pygame.mouse.get_pos()
            for button in self.buttons:
                button.hover = button.rect.collidepoint(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicked_button = None
                    for button in self.buttons:
                        if button.rect.collidepoint(event.pos):
                            clicked_button = button
                            break
                    if clicked_button is not None:
                        self._handle_button(clicked_button.action)
                    else:
                        self._handle_board_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_n:
                        self._handle_button("new")
                    elif event.key == pygame.K_p:
                        self._handle_button("pass")
                    elif event.key == pygame.K_s:
                        self._handle_button("score")

            self._render()

        pygame.quit()


def launch(ai_player: Optional[Any] = None) -> None:
    """Entry point. Shows a menu first, then the board.

    Passing ``ai_player`` skips the menu and drops straight into a PvE
    game with that AI. When ``ai_player`` is ``None`` (the default
    ``main.py`` flow), the startup menu decides between PvP and PvE and
    constructs the :class:`AIPlayer` as needed.
    """

    if ai_player is not None:
        GoGUI(ai_player=ai_player).run()
        return

    # Deferred import so ``go_game.gui`` stays importable when pygame
    # isn't present at module scan time (e.g. while running pytest
    # against the rules engine in a minimal environment).
    from .menu import MainMenu

    result = MainMenu().run()
    if result.mode == "quit":
        return
    if result.mode == "pvp":
        GoGUI().run()
        return
    if result.mode == "pve":
        GoGUI(ai_player=result.ai_player).run()
        return
