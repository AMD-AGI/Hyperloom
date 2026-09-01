# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.sources.primus_cortex. Hermetic - no real HTTP."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from hyperloom.agents.framework.sources import primus_cortex as pc
from hyperloom.agents.framework.sources._shared import GitHubPr, _repo_slug


class _FakeResp:
    """Minimal urllib response stand-in usable as a context manager."""

    def __init__(self, status: int, body: bytes, content_type: str = "application/json"):
        self.status = status
        self._body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _install_urlopen(monkeypatch, handler) -> None:
    """Replace urllib.request.urlopen used by primus_cortex with handler."""

    def fake(req, timeout):  # noqa: ARG001
        return handler(req)

    monkeypatch.setattr(pc.urllib.request, "urlopen", fake)


def test_repo_slug_parses_https_ssh_and_git_suffix() -> None:
    """_repo_slug handles https/.git/ssh URLs uniformly."""
    assert _repo_slug("https://github.com/sgl-project/sglang.git") == "sgl-project/sglang"
    assert _repo_slug("https://github.com/sgl-project/sglang") == "sgl-project/sglang"
    assert _repo_slug("git@github.com:sgl-project/sglang.git") == "sgl-project/sglang"


def test_repo_slug_rejects_malformed() -> None:
    """Non-GitHub-shaped URLs raise ValueError."""
    with pytest.raises(ValueError):
        _repo_slug("not-a-url")


def test_repo_slug_rejects_github_substring_in_path() -> None:
    """URLs that embed github.com in the path must not be accepted."""
    with pytest.raises(ValueError):
        _repo_slug("https://evil.com/github.com/owner/repo.git")
    with pytest.raises(ValueError):
        _repo_slug("https://github.com.evil.com/owner/repo.git")


def test_list_perf_prs_parses_items_list(monkeypatch) -> None:
    """list_perf_prs accepts the ``{"items": [...]}`` wrapper and trims to limit."""
    body = json.dumps(
        {
            "items": [
                {"number": 1, "title": "a", "html_url": "u1"},
                {"number": 2, "title": "b", "html_url": "u2"},
                {"number": 3, "title": "c", "html_url": "u3"},
            ]
        }
    ).encode("utf-8")
    _install_urlopen(monkeypatch, lambda req: _FakeResp(200, body))
    prs = pc.list_perf_prs(
        "https://github.com/sgl-project/sglang.git",
        base_url="http://x",
        limit=2,
    )
    assert len(prs) == 2
    assert prs[0] == GitHubPr(number=1, title="a", html_url="u1")


def test_base_url_may_include_v1_suffix(monkeypatch) -> None:
    """The KB Store-derived PR Monitor URL may already include /v1."""
    seen: dict[str, str] = {}
    body = json.dumps({"items": [{"number": 1, "title": "a", "html_url": "u1"}]}).encode("utf-8")

    def handler(req):
        seen["url"] = req.get_full_url()
        return _FakeResp(200, body)

    _install_urlopen(monkeypatch, handler)
    pc.list_perf_prs(
        "https://github.com/sgl-project/sglang.git",
        base_url="https://global.primus-safe.amd.com/knowledge-base/pr-monitor/v1",
        limit=1,
    )
    assert "/v1/v1/" not in seen["url"]
    assert seen["url"].startswith("https://global.primus-safe.amd.com/knowledge-base/pr-monitor/v1/repos/")


def test_search_prs_unwraps_summary_match_records(monkeypatch) -> None:
    """Search results may wrap PR fields under summary with match metadata."""
    body = json.dumps(
        [
            {
                "summary": {
                    "repo_name": "ROCm/vllm",
                    "number": 1057,
                    "title": "Use AITER Backend for Dsv4",
                    "state": "closed",
                    "is_merged": True,
                },
                "matched_field": "title",
                "snippet": "Use AITER Backend for Dsv4",
            }
        ]
    ).encode("utf-8")
    _install_urlopen(monkeypatch, lambda req: _FakeResp(200, body))
    prs = pc.search_perf_prs_via_primus_search(
        "https://github.com/ROCm/vllm.git",
        base_url="http://x",
        query="dsv4",
        state="all",
        limit=5,
    )
    assert prs == [
        GitHubPr(
            number=1057,
            title="Use AITER Backend for Dsv4",
            html_url="https://github.com/ROCm/vllm/pull/1057",
        )
    ]


def test_list_perf_prs_hard_fails_on_http_error(monkeypatch) -> None:
    """HTTPError from urlopen propagates as PrimusCortexError."""

    def handler(req):
        raise urllib.error.HTTPError(req.get_full_url(), 503, "boom", {}, io.BytesIO(b"oops"))

    _install_urlopen(monkeypatch, handler)
    with pytest.raises(pc.PrimusCortexError, match="HTTP 503"):
        pc.list_perf_prs(
            "https://github.com/sgl-project/sglang.git",
            base_url="http://x",
        )


def test_list_perf_prs_hard_fails_on_bad_json(monkeypatch) -> None:
    """Non-JSON body propagates as PrimusCortexError."""
    _install_urlopen(monkeypatch, lambda req: _FakeResp(200, b"<html>not json"))
    with pytest.raises(pc.PrimusCortexError, match="non-JSON"):
        pc.list_perf_prs(
            "https://github.com/sgl-project/sglang.git",
            base_url="http://x",
        )


def test_list_perf_prs_hard_fails_on_url_error(monkeypatch) -> None:
    """URLError (DNS / unreachable) propagates as PrimusCortexError."""

    def handler(req):
        raise urllib.error.URLError("dns")

    _install_urlopen(monkeypatch, handler)
    with pytest.raises(pc.PrimusCortexError, match="unreachable"):
        pc.list_perf_prs(
            "https://github.com/sgl-project/sglang.git",
            base_url="http://x",
        )


def test_pr_patches_renders_unified_diff_from_json(monkeypatch) -> None:
    """pr_patches converts primus-cortex JSON array into a unified diff stream."""
    payload = [
        {
            "file": {"file_path": "a/x.py", "status": "modified"},
            "patch": "@@ -1 +1 @@\n-old\n+new",
            "patch_truncated": False,
        },
        {
            "file": {"file_path": "a/y.py", "status": "added"},
            "patch": "@@ -0,0 +1 @@\n+brand_new",
            "patch_truncated": False,
        },
        {
            "file": {
                "file_path": "a/old.py",
                "previous_path": "a/older.py",
                "status": "deleted",
            },
            "patch": "@@ -1 +0,0 @@\n-dropped",
        },
    ]
    body = json.dumps(payload).encode("utf-8")
    _install_urlopen(monkeypatch, lambda req: _FakeResp(200, body))
    text = pc.pr_patches("o/r", 1, base_url="http://x")
    assert "diff --git a/a/x.py b/a/x.py" in text
    assert "--- a/a/x.py" in text
    assert "+++ b/a/x.py" in text
    assert "+new" in text
    assert "--- /dev/null" in text
    assert "+++ /dev/null" in text


def test_pr_patches_handles_empty_list(monkeypatch) -> None:
    """An empty array yields an empty patch string (no crash)."""
    _install_urlopen(monkeypatch, lambda req: _FakeResp(200, b"[]"))
    text = pc.pr_patches("o/r", 1, base_url="http://x")
    assert text == ""


def test_pr_files_extracts_list(monkeypatch) -> None:
    """pr_files unwraps a dict-with-list payload into a clean list of dicts."""
    body = json.dumps({"files": [{"file_path": "a.py"}, {"file_path": "b.py"}]}).encode()
    _install_urlopen(monkeypatch, lambda req: _FakeResp(200, body))
    out = pc.pr_files("o/r", 1, base_url="http://x")
    assert out == [{"file_path": "a.py"}, {"file_path": "b.py"}]


def test_pr_get_requires_object(monkeypatch) -> None:
    """pr_get refuses non-object payloads."""
    _install_urlopen(monkeypatch, lambda req: _FakeResp(200, b"[]"))
    with pytest.raises(pc.PrimusCortexError, match="did not return an object"):
        pc.pr_get("o/r", 1, base_url="http://x")
