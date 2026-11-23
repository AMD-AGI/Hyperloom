# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.sources.github. Hermetic - monkeypatches urlopen; verifies best-effort policy (returns [] on failure) and query composition."""

from __future__ import annotations

import email.message
import http.client
import io
import json
import urllib.error

import pytest

from hyperloom.agents.framework.sources import github as gh
from hyperloom.agents.framework.sources._shared import GitHubPr


class _FakeResp:
    """Tiny urllib response stand-in usable as a context manager."""

    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _install_urlopen(monkeypatch, handler) -> None:
    """Replace urllib.request.urlopen used by github backend."""

    def fake(req, timeout):
        return handler(req)

    monkeypatch.setattr(gh.urllib.request, "urlopen", fake)


# _build_query -----------------------------------------------------------


def test_build_query_scopes_repo_and_ors_the_terms() -> None:
    """The query is scoped to the repo's PRs and ORs exactly the given terms."""
    q = gh._build_query("ROCm/vllm", terms=("deepseekv4", "causallm"))
    assert "repo:ROCm/vllm" in q
    assert "is:pr" in q
    assert "(deepseekv4 OR causallm)" in q


def test_build_query_stays_within_the_search_operator_limit() -> None:
    """GitHub answers 422 past five OR operators, so only the leading terms are sent."""
    terms = ("glm5", "next", "glm", "flash", "model", "architecture", "support", "add")
    q = gh._build_query("vllm-project/vllm", terms=terms)
    assert q.count(" OR ") == 5
    assert "(glm5 OR next OR glm OR flash OR model OR architecture)" in q


def test_search_perf_prs_sends_the_given_terms(monkeypatch) -> None:
    """The composed query reaches the Search API with the caller's terms."""
    captured: dict[str, str] = {}

    def _open(req):
        captured["url"] = req.get_full_url()
        return _FakeResp(200, b'{"items": []}')

    _install_urlopen(monkeypatch, _open)
    gh.search_perf_prs("https://github.com/ROCm/vllm.git", terms=("deepseekv4",))
    assert "deepseekv4" in captured["url"]
    assert "throughput" not in captured["url"]
    # Best-match order: a recency sort would bury the PRs that match the terms best.
    assert "sort=" not in captured["url"]


def test_build_query_searches_every_pr_state() -> None:
    """The search carries no is:open qualifier, so merged and closed PRs are found too."""
    q = gh._build_query("sgl-project/sglang", terms=gh.PERF_TERMS)
    assert "is:open" not in q
    assert "is:pr" in q


# search_perf_prs --------------------------------------------------------


def test_search_perf_prs_parses_items(monkeypatch) -> None:
    """A normal GitHub-Search-shaped response is mapped to GitHubPr list."""
    body = json.dumps(
        {
            "items": [
                {"number": 11, "title": "fp8 fix", "html_url": "u11"},
                {"number": 12, "title": "rocm tune", "html_url": "u12"},
            ]
        }
    ).encode("utf-8")
    _install_urlopen(monkeypatch, lambda req: _FakeResp(200, body))
    out = gh.search_perf_prs("https://github.com/sgl-project/sglang.git", limit=5, terms=gh.PERF_TERMS)
    assert out == [
        GitHubPr(number=11, title="fp8 fix", html_url="u11"),
        GitHubPr(number=12, title="rocm tune", html_url="u12"),
    ]


def test_search_perf_prs_best_effort_on_failure(monkeypatch) -> None:
    """Network / rate-limit errors must return [] (no exception)."""

    def boom(req):
        raise urllib.error.HTTPError(req.get_full_url(), 403, "rate", {}, io.BytesIO(b""))

    _install_urlopen(monkeypatch, boom)
    out = gh.search_perf_prs("https://github.com/sgl-project/sglang.git", limit=5, terms=gh.PERF_TERMS)
    assert out == []


def test_search_perf_prs_non_github_returns_empty() -> None:
    """A non-GitHub remote should silently return [] (no network call)."""
    out = gh.search_perf_prs("https://gitlab.com/foo/bar.git", terms=gh.PERF_TERMS)
    assert out == []


# search_issues: a failure is raised, never an empty answer -------------------------------------------------


QUERY = "repo:ggml-org/llama.cpp is:pr (decode)"


def _refusing(code: int, headers: dict[str, str] | None = None, body: bytes = b""):
    def _open(req):
        message = email.message.Message()
        for name, value in (headers or {}).items():
            message[name] = value
        raise urllib.error.HTTPError(req.get_full_url(), code, "refused", message, io.BytesIO(body))

    return _open


NOW = 1_800_000_000
SPENT = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(NOW + 1500)}


@pytest.mark.parametrize(
    ("code", "headers", "body", "said"),
    [
        (403, SPENT, b"", "GitHub's rate limit for this host is spent (HTTP 403): ask again in 25 min"),
        (
            403,
            {"x-ratelimit-remaining": "4990"},
            b'{"message": "You have exceeded a secondary rate limit."}',
            "GitHub's rate limit for this host is spent (HTTP 403): ask again in 60 s",
        ),
    ],
    ids=["quota-spent", "no-wait-named"],
)
def test_a_rate_limit_that_names_no_retry_after_still_says_how_long_to_wait(
    monkeypatch, code: int, headers: dict[str, str], body: bytes, said: str
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "a-token")
    monkeypatch.setattr(gh.time, "time", lambda: NOW)
    _install_urlopen(monkeypatch, _refusing(code, headers, body))
    with pytest.raises(gh.GitHubUnavailable) as refused:
        gh.search_issues(QUERY, per_page=5)
    assert str(refused.value) == said


def _checking_headers_as_http_client_does(req):
    connection = http.client.HTTPSConnection("api.github.com")  # opens no socket before a request is sent
    connection.putrequest("GET", "/")
    for name, value in req.header_items():
        connection.putheader(name, value)
    raise AssertionError("http.client took the token as a header value")


@pytest.mark.parametrize("token", ["ghp_first_line\nsecond_line", "ghp_not_latin_✓"], ids=["line-break", "not-latin-1"])
def test_a_token_no_header_can_carry_is_named_without_its_value(monkeypatch, token: str) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", token)
    _install_urlopen(monkeypatch, _checking_headers_as_http_client_does)
    with pytest.raises(gh.GitHubUnavailable, match="GITHUB_TOKEN or GH_TOKEN. holds a line break") as refused:
        gh.search_issues(QUERY, per_page=5)
    assert not any(part in str(refused.value) for part in ("ghp_", "second_line", "✓"))
