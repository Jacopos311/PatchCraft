"""OpenRouter credits widget for PatchCraft.

Queries the official endpoint ``GET https://openrouter.ai/api/v1/auth/key``
using the key in ``OPENROUTER_API_KEY`` and shows a ``rich`` panel with the
credit usage percentage (Unicode block bar).

The widget is **non-blocking** (it never interrupts the CLI flow), but it is
not swallowed by a silent ``except``:

* missing key  -> ``dim`` hint + INFO log explaining what to do;
* network/HTTP/JSON error -> ``dim`` line with the exact exception class
  and message + ERROR/WARNING log;
* every useful detail is in the ``src.core.credits`` logger
  (enable with ``logging.basicConfig(level=logging.DEBUG)``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)

OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/auth/key"
CREDITS_TIMEOUT_SECONDS = 5.0
DEFAULT_BAR_WIDTH = 20


class CreditsError(RuntimeError):
    """Error fetching OpenRouter credits (network, HTTP or key)."""


def _get_api_key() -> Optional[str]:
    """Read the OpenRouter key from the environment.

    The full key is NEVER exposed in logs (only its length).
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key or not key.strip():
        logger.info("OPENROUTER_API_KEY is not set or is empty in the environment.")
        return None
    logger.debug("OPENROUTER_API_KEY found (%d characters long).", len(key.strip()))
    return key.strip()


def fetch_credits(api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch the credit usage from the OpenRouter endpoint.

    Parameters
    ----------
    api_key : str | None
        API key to use; if ``None`` it is read from ``OPENROUTER_API_KEY``.

    Returns
    -------
    dict | None
        Dict with the keys ``usage`` (float), ``limit`` (float | None,
        when the plan has a limit) and ``is_free_tier`` (bool). ``None`` when
        the key is not configured (no error).

    Raises
    ------
    CreditsError
        On network errors, timeouts, non-200 HTTP responses or malformed JSON.
    """
    key = api_key or _get_api_key()
    if not key:
        logger.info("No OpenRouter key: the credits widget is disabled.")
        return None

    headers = {
        "Authorization": f"Bearer {key}",
        "User-Agent": "PatchCraft",
    }
    logger.debug("GET %s (timeout=%ss)", OPENROUTER_CREDITS_URL, CREDITS_TIMEOUT_SECONDS)
    try:
        response = requests.get(
            OPENROUTER_CREDITS_URL,
            headers=headers,
            timeout=CREDITS_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.error("Request to OpenRouter failed: %s: %s", type(exc).__name__, exc)
        raise CreditsError(f"Network error reaching OpenRouter: {exc}") from exc

    logger.debug("OpenRouter replied HTTP %d (%d bytes).",
                 response.status_code, len(response.content))
    if response.status_code == 401:
        logger.error("OpenRouter: invalid key (HTTP 401).")
        raise CreditsError("Invalid API key (401 Unauthorized).")
    if response.status_code == 429:
        logger.error("OpenRouter: rate limit exceeded (HTTP 429).")
        raise CreditsError("OpenRouter rate limit exceeded (429).")
    if response.status_code != 200:
        logger.error("OpenRouter: unexpected status HTTP %d.", response.status_code)
        raise CreditsError(f"OpenRouter replied with HTTP {response.status_code}.")

    try:
        body = response.json()
    except ValueError as exc:
        logger.error("OpenRouter: response is not JSON: %s: %s", type(exc).__name__, exc)
        raise CreditsError("OpenRouter response is not valid JSON.") from exc

    data = body.get("data") or {}
    if not data:
        logger.warning("OpenRouter: response without 'data' section: %r", body)
    else:
        logger.debug("Fields received in 'data': %s", sorted(data.keys()))
    usage = data.get("usage")
    limit = data.get("limit")
    return {
        "usage": float(usage) if usage is not None else 0.0,
        "limit": float(limit) if limit is not None else None,
        "is_free_tier": bool(data.get("is_free_tier", False)),
    }


def build_usage_bar(usage: float, limit: float, width: int = DEFAULT_BAR_WIDTH) -> str:
    """Build a usage bar from Unicode blocks.

    Returns something like ``"██████░░░░░░░░░░ 30%"`` (no rich markup).
    """
    if limit <= 0:
        return "░" * width
    pct = min(100.0, max(0.0, usage / limit * 100.0))
    filled = int(round(width * pct / 100.0))
    return f"{'█' * filled}{'░' * (width - filled)}"


def _format_amount(value: float) -> str:
    """Format an OpenRouter monetary value ($, adaptive precision)."""
    if abs(value) < 0.01:
        return f"${value:.6f}".rstrip("0").rstrip(".")
    if value >= 1000:
        return f"${value:,.2f}"
    return f"${value:.4f}".rstrip("0").rstrip(".")


def render_credits_panel(
    console: Optional[Console] = None,
    verbose: bool = False,
) -> bool:
    """Show the 💳 OpenRouter Credit Usage panel (never blocking).

    Parameters
    ----------
    console : Console | None
        ``rich`` console to print on. If ``None`` a new instance is created
        (robust default, no shared state).
    verbose : bool
        When ``True`` and the endpoint fails, also print the full traceback
        (useful for debugging). Default ``False``.

    Returns
    -------
    True if the panel was printed; ``False`` if the widget is inactive
    (missing key) or an error was handled gracefully.
    """
    _console = console or Console()
    try:
        credits = fetch_credits()
    except CreditsError as exc:
        logger.error("Credits widget not rendered: %s", exc)
        _console.print(f"[dim]💳 OpenRouter: {type(exc).__name__}: {exc}[/dim]")
        if verbose:
            _console.print_exception(show_locals=True)
        return False
    except Exception as exc:  # noqa: BLE001 - the widget must never block the CLI
        logger.exception("Credits widget: unexpected error while fetching.")
        _console.print(f"[dim]💳 OpenRouter: unexpected error {type(exc).__name__}: {exc}[/dim]")
        if verbose:
            _console.print_exception(show_locals=True)
        return False

    if credits is None:
        # Missing key: a discrete hint instead of total silence, so it is clear
        # why the widget does not appear (and how to enable it).
        _console.print(
            "[dim]💳 OpenRouter: OPENROUTER_API_KEY is not configured — "
            "credits widget disabled. Export the key to enable it.[/dim]"
        )
        return False

    usage: float = credits["usage"]
    limit: Optional[float] = credits["limit"]
    is_free_tier: bool = credits["is_free_tier"]

    title = "[bold yellow]💳 OpenRouter Credit Usage[/]"
    if limit and limit > 0:
        pct = min(100.0, max(0.0, usage / limit * 100.0))
        bar = build_usage_bar(usage, limit)
        content = (
            f"Credits used: [bold]{_format_amount(usage)}[/] "
            f"of [bold]{_format_amount(limit)}[/]\n\n"
            f"[cyan]{bar}[/] [bold]{pct:.1f}%[/]"
        )
    else:
        body = f"Total credits spent: [bold]{_format_amount(usage)}[/]"
        if is_free_tier:
            body += "\n[dim]Free tier — no limit applied.[/dim]"
        content = body

    _console.print(Panel(content, title=title, border_style="cyan"))
    return True


__all__ = [
    "fetch_credits",
    "build_usage_bar",
    "render_credits_panel",
    "CreditsError",
    "OPENROUTER_CREDITS_URL",
]