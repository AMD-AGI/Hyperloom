# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for framework_agent.sources.github. Hermetic - monkeypatches urlopen; verifies best-effort policy (returns [] on failure) and keyword-driven query composition."""

from __future__ import annotations

import io
import json
import urllib.error

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


def test_build_query_open_only_keeps_is_open() -> None:
    """Default open-only keeps the is:open qualifier."""
    q = gh._build_query("sgl-project/sglang", "fp8", states=("open",))
    assert "is:open" in q


def test_build_query_all_drops_is_open() -> None:
    """Explicit all-state search omits the is:open qualifier."""
    q = gh._build_query("sgl-project/sglang", "fp8", states=("all",))
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


# pr_patches / fetch_raw_file -------------------------------------------


class _RawResp(io.BytesIO):
    """Minimal context-manager response wrapping fixed bytes."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def _raw_urlopen_ok(payload: str):
    def _open(req, timeout=0):
        return _RawResp(payload.encode("utf-8"))

    return _open


def _raw_urlopen_fail(req, timeout=0):
    raise urllib.error.URLError("boom")


def test_pr_patches_returns_diff(monkeypatch):
    diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    monkeypatch.setattr(gh.urllib.request, "urlopen", _raw_urlopen_ok(diff))
    assert gh.pr_patches("ROCm/vllm", 1234) == diff


def test_pr_patches_network_error_returns_empty(monkeypatch):
    monkeypatch.setattr(gh.urllib.request, "urlopen", _raw_urlopen_fail)
    assert gh.pr_patches("ROCm/vllm", 1234) == ""


def test_pr_patches_invalid_args(monkeypatch):
    monkeypatch.setattr(gh.urllib.request, "urlopen", _raw_urlopen_ok("x"))
    assert gh.pr_patches("", 1) == ""
    assert gh.pr_patches("o/r", 0) == ""


def test_pr_patches_auth_header_when_token(monkeypatch):
    captured = {}

    def _open(req, timeout=0):
        captured["auth"] = req.headers.get("Authorization")
        return _RawResp(b"diff --git a/x b/x\n")

    monkeypatch.setenv("GITHUB_TOKEN", "secret123")
    monkeypatch.setattr(gh.urllib.request, "urlopen", _open)
    gh.pr_patches("o/r", 5)
    assert captured["auth"] == "Bearer secret123"


def test_pr_patches_anonymous_when_no_token(monkeypatch):
    captured = {}

    def _open(req, timeout=0):
        captured["auth"] = req.headers.get("Authorization")
        return _RawResp(b"diff --git a/x b/x\n")

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(gh.urllib.request, "urlopen", _open)
    gh.pr_patches("o/r", 5)
    assert captured["auth"] is None


def test_fetch_raw_file_returns_content(monkeypatch):
    monkeypatch.setattr(gh.urllib.request, "urlopen", _raw_urlopen_ok("print('hi')\n"))
    assert gh.fetch_raw_file("ROCm/vllm", "abc123", "vllm/model.py") == "print('hi')\n"


def test_fetch_raw_file_network_error_returns_empty(monkeypatch):
    monkeypatch.setattr(gh.urllib.request, "urlopen", _raw_urlopen_fail)
    assert gh.fetch_raw_file("o/r", "ref", "p.py") == ""


def test_fetch_raw_file_invalid_args(monkeypatch):
    monkeypatch.setattr(gh.urllib.request, "urlopen", _raw_urlopen_ok("x"))
    assert gh.fetch_raw_file("", "ref", "p") == ""
    assert gh.fetch_raw_file("o/r", "", "p") == ""
    assert gh.fetch_raw_file("o/r", "ref", "") == ""


def test_source_patch_exports():
    assert "pr_patches" in gh.__all__
    assert "fetch_raw_file" in gh.__all__
