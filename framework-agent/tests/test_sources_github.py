"""Tests for framework_agent.sources.github.

Hermetic - monkeypatches urlopen. Verifies best-effort policy
(returns [] on failure) and keyword-driven query composition.
"""

from __future__ import annotations

import io
import json
import urllib.error

from framework_agent.sources import github as gh
from framework_agent.sources._shared import GitHubPr


class _FakeResp:
    """Tiny urllib response stand-in usable as a context manager."""

    def __init__(self, status: int, body: bytes):
        """Capture status and body for later urlopen() reads."""
        self.status = status
        self._body = body
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self):
        """Context-manager enter; returns self so .read() works."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context-manager exit; nothing to clean up."""
        return None

    def read(self) -> bytes:
        """Return the canned body."""
        return self._body


def _install_urlopen(monkeypatch, handler) -> None:
    """Replace urllib.request.urlopen used by github backend."""
    def fake(req, timeout):  # noqa: ARG001
        return handler(req)

    monkeypatch.setattr(gh.urllib.request, "urlopen", fake)


# _build_query -----------------------------------------------------------


def test_build_query_uses_extracted_keywords() -> None:
    """gap_description keywords feed the OR clause; PERF_TERMS not used."""
    q = gh._build_query("sgl-project/sglang", "improve fp8 MoE attention")
    assert "repo:sgl-project/sglang" in q
    assert "is:pr" in q
    assert "is:open" in q
    assert "fp8" in q
    assert "moe" in q
    assert "attention" in q


def test_build_query_falls_back_to_perf_terms_when_empty() -> None:
    """Empty gap_description triggers the PERF_TERMS fallback."""
    q = gh._build_query("sgl-project/sglang", "")
    for t in gh.PERF_TERMS:
        assert t in q


# search_perf_prs --------------------------------------------------------


def test_search_perf_prs_parses_items(monkeypatch) -> None:
    """A normal GitHub-Search-shaped response is mapped to GitHubPr list."""
    body = json.dumps({
        "items": [
            {"number": 11, "title": "fp8 fix", "html_url": "u11"},
            {"number": 12, "title": "rocm tune", "html_url": "u12"},
        ]
    }).encode("utf-8")
    _install_urlopen(monkeypatch, lambda req: _FakeResp(200, body))
    out = gh.search_perf_prs("https://github.com/sgl-project/sglang.git", limit=5)
    assert out == [
        GitHubPr(number=11, title="fp8 fix", html_url="u11"),
        GitHubPr(number=12, title="rocm tune", html_url="u12"),
    ]


def test_search_perf_prs_best_effort_on_failure(monkeypatch) -> None:
    """Network / rate-limit errors must return [] (no exception)."""
    def boom(req):
        raise urllib.error.HTTPError(req.get_full_url(), 403, "rate", {}, io.BytesIO(b""))

    _install_urlopen(monkeypatch, boom)
    out = gh.search_perf_prs("https://github.com/sgl-project/sglang.git", limit=5)
    assert out == []


def test_search_perf_prs_non_github_returns_empty() -> None:
    """A non-GitHub remote should silently return [] (no network call)."""
    out = gh.search_perf_prs("https://gitlab.com/foo/bar.git")
    assert out == []
