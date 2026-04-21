"""Startup menu for GuuGo.

Shown before the board GUI. Two modes:

- **Play vs Friend** -- launches the existing PvP board with no AI.
- **Play vs Computer** -- loads a trained checkpoint into an
  :class:`alphazero.ai_player.AIPlayer` and hands it to the board GUI.

The menu owns its own pygame event loop so the board GUI does not have
to know anything about it; on a successful click it returns a
:class:`MenuResult` describing what to do next.

The import of :mod:`alphazero` is deliberately deferred until the user
clicks the PvE start button. This keeps ``go_game`` importable on
machines without torch installed (the PvP app should still work), and
keeps menu startup instant even on boxes with a slow torch import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pygame

# -- palette chosen to visually match go_game.gui --
WINDOW_BG = (245, 238, 224)
TEXT_COLOR = (28, 28, 28)
MUTED_COLOR = (80, 80, 80)
ERROR_COLOR = (170, 40, 40)
BORDER = (120, 100, 60)
BUTTON_BG = (235, 225, 205)
BUTTON_BG_HOVER = (220, 205, 175)
BUTTON_TEXT = (30, 30, 30)
FIELD_BG = (255, 250, 238)
FIELD_BG_FOCUS = (255, 245, 215)

WIDTH = 640
HEIGHT = 480

# Fixed MCTS simulations per AI move. Kept as a module constant rather
# than a UI knob because the plan calls for a single, predictable
# difficulty level.
DEFAULT_SIMULATIONS = 100


Mode = Literal["pvp", "pve", "quit"]


@dataclass
class MenuResult:
    mode: Mode
    ai_player: Optional[object] = None  # alphazero.ai_player.AIPlayer at runtime
    num_simulations: Optional[int] = None


# --------------------------------------------------------------------------- #
# Small reusable widgets
# --------------------------------------------------------------------------- #


class _Button:
    def __init__(self, rect: pygame.Rect, label: str, action: str) -> None:
        self.rect = rect
        self.label = label
        self.action = action
        self.hover = False
        self.enabled = True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        bg = BUTTON_BG_HOVER if self.hover and self.enabled else BUTTON_BG
        pygame.draw.rect(surface, bg, self.rect, border_radius=8)
        pygame.draw.rect(surface, BORDER, self.rect, width=1, border_radius=8)
        color = BUTTON_TEXT if self.enabled else MUTED_COLOR
        text = font.render(self.label, True, color)
        surface.blit(text, text.get_rect(center=self.rect.center))


class _TextField:
    """Single-line text input, enough for typing a short path."""

    def __init__(self, rect: pygame.Rect, initial: str = "") -> None:
        self.rect = rect
        self.text = initial
        self.focused = False
        self._caret_on = True
        self._caret_timer = 0

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focused = self.rect.collidepoint(event.pos)
            return
        if not self.focused:
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_TAB):
                self.focused = False
            elif event.key == pygame.K_ESCAPE:
                self.focused = False
            elif event.unicode and event.unicode.isprintable():
                if len(self.text) < 96:
                    self.text += event.unicode

    def tick(self, dt_ms: int) -> None:
        self._caret_timer += dt_ms
        if self._caret_timer >= 500:
            self._caret_timer = 0
            self._caret_on = not self._caret_on

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        bg = FIELD_BG_FOCUS if self.focused else FIELD_BG
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        border_w = 2 if self.focused else 1
        pygame.draw.rect(surface, BORDER, self.rect, width=border_w, border_radius=6)
        text_surface = font.render(self.text, True, TEXT_COLOR)
        text_y = self.rect.y + (self.rect.height - text_surface.get_height()) // 2
        surface.blit(text_surface, (self.rect.x + 10, text_y))
        if self.focused and self._caret_on:
            caret_x = self.rect.x + 10 + text_surface.get_width() + 1
            caret_y = self.rect.y + 8
            caret_h = self.rect.height - 16
            pygame.draw.line(
                surface,
                TEXT_COLOR,
                (caret_x, caret_y),
                (caret_x, caret_y + caret_h),
                2,
            )


# --------------------------------------------------------------------------- #
# Main menu
# --------------------------------------------------------------------------- #


class MainMenu:
    """Pygame startup menu for PvP / PvE selection."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("GuuGo - 9x9 Go")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("helvetica", 34, bold=True)
        self.section_font = pygame.font.SysFont("helvetica", 18, bold=True)
        self.label_font = pygame.font.SysFont("helvetica", 14)
        self.button_font = pygame.font.SysFont("helvetica", 15, bold=True)
        self.field_font = pygame.font.SysFont("menlo", 15)
        self.status_font = pygame.font.SysFont("helvetica", 14)

        self.pvp_button = _Button(
            pygame.Rect(WIDTH // 2 - 180, 130, 360, 52),
            "Play vs Friend (PvP)",
            "pvp",
        )

        field_y = 290
        self.dir_field = _TextField(
            pygame.Rect(WIDTH // 2 - 220, field_y, 440, 36),
            initial="checkpoints",
        )

        self.pve_button = _Button(
            pygame.Rect(WIDTH // 2 - 180, field_y + 60, 360, 48),
            "Play vs Computer (PvE)",
            "pve",
        )

        self.status_text: str = ""
        self.status_is_error: bool = False

    # -------------------- helpers --------------------

    def _set_status(self, text: str, *, error: bool) -> None:
        self.status_text = text
        self.status_is_error = error

    def _try_load_ai(self) -> Optional[MenuResult]:
        raw_dir = self.dir_field.text.strip() or "checkpoints"
        checkpoint_dir = Path(raw_dir).expanduser()

        if not checkpoint_dir.exists():
            self._set_status(
                f"Checkpoint directory not found: {checkpoint_dir}", error=True
            )
            return None

        self._set_status("Loading model...", error=False)
        self._render()  # flush status before the blocking import/load
        pygame.display.flip()

        try:
            # Deferred import so go_game stays torch-free until this moment.
            from alphazero.ai_player import AIPlayer  # type: ignore
        except ImportError as exc:
            self._set_status(
                f"PyTorch not installed: {exc}. Install the training stack first.",
                error=True,
            )
            return None

        try:
            ai = AIPlayer.load_from_checkpoint(
                checkpoint_dir=checkpoint_dir,
                num_simulations=DEFAULT_SIMULATIONS,
                device="auto",
            )
        except FileNotFoundError as exc:
            self._set_status(str(exc), error=True)
            return None
        except Exception as exc:  # model arch mismatch, corrupted file, ...
            self._set_status(f"Failed to load model: {exc}", error=True)
            return None

        return MenuResult(
            mode="pve", ai_player=ai, num_simulations=DEFAULT_SIMULATIONS
        )

    # -------------------- drawing --------------------

    def _render(self) -> None:
        self.screen.fill(WINDOW_BG)

        # Title
        title = self.title_font.render("GuuGo - 9x9 Go", True, TEXT_COLOR)
        self.screen.blit(title, title.get_rect(center=(WIDTH // 2, 60)))
        subtitle = self.label_font.render(
            "Beginner 9x9 Go trainer.  Pick a mode.", True, MUTED_COLOR
        )
        self.screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, 92)))

        # PvP
        self.pvp_button.draw(self.screen, self.button_font)

        # Divider
        divider_y = 220
        pygame.draw.line(
            self.screen, BORDER, (60, divider_y), (WIDTH - 60, divider_y), 1
        )
        header = self.section_font.render("Play vs Computer", True, TEXT_COLOR)
        self.screen.blit(
            header, (WIDTH // 2 - header.get_width() // 2, divider_y + 14)
        )

        dir_label = self.label_font.render(
            "Checkpoint directory (must contain latest.pt):", True, MUTED_COLOR
        )
        self.screen.blit(dir_label, (self.dir_field.rect.x, self.dir_field.rect.y - 20))
        self.dir_field.draw(self.screen, self.field_font)

        self.pve_button.draw(self.screen, self.button_font)

        # Status line
        if self.status_text:
            color = ERROR_COLOR if self.status_is_error else MUTED_COLOR
            status = self.status_font.render(self.status_text, True, color)
            self.screen.blit(
                status, status.get_rect(center=(WIDTH // 2, HEIGHT - 28))
            )

    # -------------------- main loop --------------------

    def run(self) -> MenuResult:
        running = True
        while running:
            dt = self.clock.tick(60)
            mouse_pos = pygame.mouse.get_pos()

            self.pvp_button.hover = self.pvp_button.rect.collidepoint(mouse_pos)
            self.pve_button.hover = self.pve_button.rect.collidepoint(mouse_pos)

            self.dir_field.tick(dt)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return MenuResult(mode="quit")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return MenuResult(mode="quit")

                self.dir_field.handle_event(event)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.pvp_button.rect.collidepoint(event.pos):
                        return MenuResult(mode="pvp")
                    if self.pve_button.rect.collidepoint(event.pos):
                        result = self._try_load_ai()
                        if result is not None:
                            return result

            self._render()
            pygame.display.flip()

        return MenuResult(mode="quit")
