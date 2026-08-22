"""PatchCraft interactive GUI package (Textual TUI)."""

from __future__ import annotations

from src.gui.app import PatchCraftApp


def launch_gui(model: str | None = None, max_retries: int | None = None) -> None:
    """Launch the PatchCraft Textual application.

    ``max_retries=None`` runs the goal-driven loop until tests pass
    (loop detection and budgets still apply).
    """
    PatchCraftApp(default_model=model, max_retries=max_retries).run()


__all__ = ["PatchCraftApp", "launch_gui"]