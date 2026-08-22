"""Test del widget crediti OpenRouter (requests mockato, nessuna rete)."""
from __future__ import annotations

import json
import os
from io import StringIO
from unittest import mock

import pytest
import requests
from rich.console import Console

from src.core.credits import (
    CreditsError,
    OPENROUTER_CREDITS_URL,
    build_usage_bar,
    fetch_credits,
    render_credits_panel,
)


def _response(payload, status_code: int = 200) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = json.dumps(payload).encode("utf-8")
    return resp


class TestFetchCredits:
    def test_extracts_usage_and_limit(self) -> None:
        payload = {"data": {"usage": 12.5, "limit": 100.0, "is_free_tier": False}}
        with mock.patch("requests.get", return_value=_response(payload)) as mocked:
            credits = fetch_credits(api_key="sk-test")

        assert credits == {"usage": 12.5, "limit": 100.0, "is_free_tier": False}
        url = mocked.call_args.args[0]
        assert url == OPENROUTER_CREDITS_URL
        assert mocked.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-test"

    def test_uses_environment_key(self) -> None:
        payload = {"data": {"usage": 1.0, "limit": None, "is_free_tier": True}}
        with mock.patch("requests.get", return_value=_response(payload)) as mocked:
            with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}, clear=True):
                credits = fetch_credits()

        assert credits["usage"] == 1.0
        assert credits["limit"] is None
        assert credits["is_free_tier"] is True
        assert mocked.call_args.kwargs["headers"]["Authorization"] == "Bearer env-key"

    def test_no_key_returns_none(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            assert fetch_credits() is None

    def test_401_raises(self) -> None:
        with mock.patch("requests.get", return_value=_response({"error": "bad"}, 401)):
            with pytest.raises(CreditsError, match="401"):
                fetch_credits(api_key="sk-bad")

    def test_network_error_raises(self) -> None:
        with mock.patch("requests.get", side_effect=requests.ConnectionError("offline")):
            with pytest.raises(CreditsError, match="Network error"):
                fetch_credits(api_key="sk-x")


class TestUsageBar:
    def test_half_bar(self) -> None:
        bar = build_usage_bar(50, 100, width=10)
        assert bar == "█████░░░░░"

    def test_no_limit(self) -> None:
        assert build_usage_bar(1.0, 0, width=5) == "░░░░░"

    def test_boundaries_clamped(self) -> None:
        assert build_usage_bar(0, 100, width=10) == "░░░░░░░░░░"
        assert build_usage_bar(150, 100, width=10) == "██████████"


class TestRenderPanel:
    @staticmethod
    def _capture() -> tuple[Console, StringIO]:
        io = StringIO()
        return Console(file=io), io

    def test_with_limit_renders_bar(self) -> None:
        console, io = self._capture()
        with mock.patch(
            "src.core.credits.fetch_credits",
            return_value={"usage": 30.0, "limit": 100.0, "is_free_tier": False},
        ):
            ok = render_credits_panel(console)

        out = io.getvalue()
        assert ok is True
        assert "OpenRouter Credit Usage" in out
        assert "█" in out and "░" in out
        assert "30.0%" in out

    def test_without_limit_shows_total(self) -> None:
        console, io = self._capture()
        with mock.patch(
            "src.core.credits.fetch_credits",
            return_value={"usage": 7.25, "limit": None, "is_free_tier": True},
        ):
            ok = render_credits_panel(console)

        out = io.getvalue()
        assert ok is True
        assert "Total credits spent" in out
        assert "7.25" in out
        assert "Free tier" in out

    def test_error_is_graceful_but_explicit(self) -> None:
        """L'errore esatto (classe + messaggio) viene stampato, non nascosto."""
        console, io = self._capture()
        with mock.patch(
            "src.core.credits.fetch_credits",
            side_effect=CreditsError("Invalid API key (401 Unauthorized)."),
        ):
            ok = render_credits_panel(console)

        out = io.getvalue()
        assert ok is False
        assert "CreditsError" in out
        assert "Invalid API key (401 Unauthorized)" in out

    def test_missing_key_shows_hint(self) -> None:
        """Missing key: explicit hint (no more total silence)."""
        console, io = self._capture()
        with mock.patch("src.core.credits.fetch_credits", return_value=None):
            ok = render_credits_panel(console)

        assert ok is False
        assert "OPENROUTER_API_KEY" in io.getvalue()


class TestCreditsLogging:
    def test_logs_debug_when_key_found(self, caplog) -> None:
        import logging

        from src.core.credits import fetch_credits

        payload = {"data": {"usage": 1.0, "limit": 100.0, "is_free_tier": False}}
        with (
            mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-secret"}, clear=True),
            mock.patch("requests.get", return_value=_response(payload)),
            caplog.at_level(logging.DEBUG, logger="src.core.credits"),
        ):
            fetch_credits()

        joined = caplog.text
        assert "OPENROUTER_API_KEY found" in joined
        assert "HTTP 200" in joined
        assert "Fields received" in joined
        # The key must NEVER appear in logs
        assert "sk-secret" not in joined

    def test_logs_error_on_401(self, caplog) -> None:
        import logging

        from src.core.credits import fetch_credits

        with (
            mock.patch("requests.get", return_value=_response({"error": "bad"}, 401)),
            caplog.at_level(logging.ERROR, logger="src.core.credits"),
        ):
            with pytest.raises(CreditsError):
                fetch_credits(api_key="sk-bad")

        assert "HTTP 401" in caplog.text


class TestCliIntegration:
    def test_cli_invokes_credit_panel_at_startup(self) -> None:
        """All'avvio di un comando reale (ask) il widget viene renderizzato."""
        from click.testing import CliRunner

        import main

        with (
            mock.patch("main.render_credits_panel") as render_mock,
            mock.patch("main.call_llm", return_value="ok"),
        ):
            result = CliRunner().invoke(main.cli, ["ask", "hello"])

        assert result.exit_code == 0
        render_mock.assert_called_once()

    def test_cli_skips_panel_on_help(self) -> None:
        """Con --help il widget viene saltato per non sporcare l'output."""
        from click.testing import CliRunner

        import main

        with mock.patch("main.render_credits_panel") as render_mock:
            _ = CliRunner().invoke(main.cli, ["select", "--help"])

        render_mock.assert_not_called()