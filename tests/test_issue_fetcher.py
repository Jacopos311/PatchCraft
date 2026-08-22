"""Test del fetcher issue GitHub (requests mockato, nessuna rete)."""
from __future__ import annotations

import json
from unittest import mock

import pytest
import requests

from src.github.issue_fetcher import GitHubAPIError, get_open_issues


def _response(payload, status_code: int = 200) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = json.dumps(payload).encode("utf-8")
    return resp


def _issue(number: int, title: str = "bug!", is_pull_request: bool = False) -> dict:
    item: dict = {"id": f"i-{number}", "number": number, "title": title, "body": f"body {number}"}
    if is_pull_request:
        item["pull_request"] = {"url": f"https://api.github.com/repos/o/r/pulls/{number}"}
    return item


def test_returns_only_issues_and_filters_pull_requests() -> None:
    """Le Pull Request (chiave `pull_request`) vengono escluse."""
    payload = [
        _issue(1),
        _issue(2, is_pull_request=True),
        _issue(3, "altro"),
    ]
    with mock.patch("requests.get", return_value=_response(payload)):
        issues = get_open_issues("owner/repo")
    assert [i["number"] for i in issues] == [1, 3]


def test_sends_expected_params_and_url() -> None:
    payload = [_issue(7)]
    captured: dict = {}

    def fake_get(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return _response(payload)

    with mock.patch("requests.get", fake_get):
        get_open_issues("owner/repo", label="bug", limit=5)

    assert captured["url"] == "https://api.github.com/repos/owner/repo/issues"
    assert captured["params"] == {"state": "open", "labels": "bug", "per_page": 5}
    assert captured["headers"]["User-Agent"] == "PatchCraft"
    assert captured["timeout"] == 10.0


def test_raises_on_404() -> None:
    with mock.patch("requests.get", return_value=_response({"message": "Not Found"}, 404)):
        with pytest.raises(GitHubAPIError, match="not found"):
            get_open_issues("owner/missing")


def test_raises_on_rate_limit_403() -> None:
    with mock.patch("requests.get", return_value=_response({"message": "rate limit"}, 403)):
        with pytest.raises(GitHubAPIError, match="rate limit"):
            get_open_issues("owner/repo")


def test_raises_on_network_error() -> None:
    with mock.patch("requests.get", side_effect=requests.ConnectionError("boom")):
        with pytest.raises(GitHubAPIError, match="Network error"):
            get_open_issues("owner/repo")


def test_invalid_repo_name() -> None:
    with pytest.raises(ValueError):
        get_open_issues("solo-un-nome")


def test_invalid_limit() -> None:
    with pytest.raises(ValueError):
        get_open_issues("owner/repo", limit=0)