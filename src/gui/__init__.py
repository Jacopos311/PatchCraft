"""PatchCraft interactive GUI package (Textual TUI)."""

from __future__ import annotations

from src.gui.app import PatchCraftApp


def launch_gui(model: str | None = None, max_retries: int = 3) -> None:
    """Launch the PatchCraft Textual application."""
    PatchCraftApp(default_model=model, max_retries=max_retries).run()


__all__ = ["PatchCraftApp", "launch_gui"]