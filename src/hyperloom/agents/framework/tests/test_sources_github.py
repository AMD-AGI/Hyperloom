# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.sources.github. Hermetic - monkeypatches urlopen; verifies query composition and that a failed search is raised."""

from __future__ import annotations

import email.message
import http.client
import io
import urllib.error

import pytest

from hyperloom.agents.framework.sources import github as gh


def _install_urlopen(monkeypatch, handler) -> None:
    """Replace urllib.request.urlopen used by github backend."""

    def fake(req, timeout):
        return handler(req)

    monkeypatch.setattr(gh.urllib.request, "urlopen", fake)


# build_query -----------------------------------------------------------


def test_build_query_scopes_repo_and_ors_the_terms() -> None:
    """The query is scoped to the repo's PRs and ORs exactly the given terms."""
    q = gh.build_query("ROCm/vllm", terms=("deepseekv4", "causallm"))
    assert "repo:ROCm/vllm" in q
    assert "is:pr" in q
    assert "(deepseekv4 OR causallm)" in q


def test_build_query_stays_within_the_search_operator_limit() -> None:
    """GitHub answers 422 past five OR operators, so only the leading terms are sent."""
    terms = ("glm5", "next", "glm", "flash", "model", "architecture", "support", "add")
    q = gh.build_query("vllm-project/vllm", terms=terms)
    assert q.count(" OR ") == 5
    assert "(glm5 OR next OR glm OR flash OR model OR architecture)" in q


def test_build_query_searches_every_pr_state() -> None:
    """The search carries no is:open qualifier, so merged and closed PRs are found too."""
    q = gh.build_query("sgl-project/sglang", terms=gh.PERF_TERMS)
    assert "is:open" not in q
    assert "is:pr" in q


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
