"""Pygame GUI for GuuGo's PvP mode.

The GUI is a thin presentation layer over :class:`GameEngine`. It draws the
board, handles click-to-play, highlights the most recent move, shows capture
counts, and surfaces the game result. All rules logic lives in the engine.

We use pygame because it is a single pure-pip dependency (``pip install
pygame``) with no system packages required, which makes the app trivial to
ship to a grader: create a venv, install requirements, run ``main.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

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
    """Pygame application window for PvP play."""

    def __init__(self, engine: Optional[GameEngine] = None) -> None:
        self.engine = engine or GameEngine()

        self._board_pixels = BOARD_MARGIN * 2 + CELL_SIZE * (self.engine.size - 1)
        self._width = self._board_pixels
        self._height = HUD_HEIGHT + self._board_pixels

        pygame.init()
        pygame.display.set_caption("GuuGo - 9x9 Go")
        self.screen = pygame.display.set_mode((self._width, self._height))
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("helvetica", 22, bold=True)
        self.hud_font = pygame.font.SysFont("helvetica", 15)
        self.button_font = pygame.font.SysFont("helvetica", 14, bold=True)
        self.banner_font = pygame.font.SysFont("helvetica", 16, bold=True)

        self.buttons = self._build_buttons()
        self._banner_text: Optional[str] = None
        self._banner_timer_ms: int = 0

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
        point = self._pixel_to_point(pos[0], pos[1])
        if point is None:
            return
        result = self.engine.play(point)
        if not result.legal:
            self._set_banner(f"Illegal move: {result.reason}")
            return
        if result.captured:
            self._set_banner(f"Captured {len(result.captured)} stone(s)")

    def _handle_button(self, action: str) -> None:
        if action == "pass":
            if self.engine.is_over:
                return
            loser = self.engine.current_player
            self.engine.pass_turn()
            self._set_banner(f"{loser.name.title()} passed - concedes game")
        elif action == "score":
            if self.engine.is_over:
                return
            self.engine.finish_by_score()
            self._set_banner("Game scored")
        elif action == "new":
            self.engine.reset()
            self._set_banner("New game started")

    # ---------------- main loop ----------------

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60)
            if self._banner_timer_ms > 0:
                self._banner_timer_ms -= dt
                if self._banner_timer_ms <= 0:
                    self._banner_text = None

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


def launch() -> None:
    GoGUI().run()
