# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for github source pr_patches() + fetch_raw_file() (M2 §9.3)."""

from __future__ import annotations

import io
import urllib.error

import pytest

from hyperloom.agents.framework.sources import github as gh


class _FakeResp(io.BytesIO):
    """Minimal context-manager response wrapping fixed bytes."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def _fake_urlopen_ok(payload: str):
    def _open(req, timeout=0):
        return _FakeResp(payload.encode("utf-8"))

    return _open


def _fake_urlopen_fail(req, timeout=0):
    raise urllib.error.URLError("boom")


# ---------------------------------------------------------------------------
# pr_patches
# ---------------------------------------------------------------------------

def test_pr_patches_returns_diff(monkeypatch):
    diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    monkeypatch.setattr(gh.urllib.request, "urlopen", _fake_urlopen_ok(diff))
    out = gh.pr_patches("ROCm/vllm", 1234)
    assert out == diff


def test_pr_patches_network_error_returns_empty(monkeypatch):
    monkeypatch.setattr(gh.urllib.request, "urlopen", _fake_urlopen_fail)
    assert gh.pr_patches("ROCm/vllm", 1234) == ""


def test_pr_patches_invalid_args(monkeypatch):
    monkeypatch.setattr(gh.urllib.request, "urlopen", _fake_urlopen_ok("x"))
    assert gh.pr_patches("", 1) == ""
    assert gh.pr_patches("o/r", 0) == ""


def test_pr_patches_auth_header_when_token(monkeypatch):
    captured = {}

    def _open(req, timeout=0):
        captured["auth"] = req.headers.get("Authorization")
        return _FakeResp(b"diff --git a/x b/x\n")

    monkeypatch.setenv("GITHUB_TOKEN", "secret123")
    monkeypatch.setattr(gh.urllib.request, "urlopen", _open)
    gh.pr_patches("o/r", 5)
    assert captured["auth"] == "Bearer secret123"


def test_pr_patches_anonymous_when_no_token(monkeypatch):
    captured = {}

    def _open(req, timeout=0):
        captured["auth"] = req.headers.get("Authorization")
        return _FakeResp(b"diff --git a/x b/x\n")

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(gh.urllib.request, "urlopen", _open)
    gh.pr_patches("o/r", 5)
    assert captured["auth"] is None


# ---------------------------------------------------------------------------
# fetch_raw_file
# ---------------------------------------------------------------------------

def test_fetch_raw_file_returns_content(monkeypatch):
    monkeypatch.setattr(gh.urllib.request, "urlopen", _fake_urlopen_ok("print('hi')\n"))
    out = gh.fetch_raw_file("ROCm/vllm", "abc123", "vllm/model.py")
    assert out == "print('hi')\n"


def test_fetch_raw_file_network_error_returns_empty(monkeypatch):
    monkeypatch.setattr(gh.urllib.request, "urlopen", _fake_urlopen_fail)
    assert gh.fetch_raw_file("o/r", "ref", "p.py") == ""


def test_fetch_raw_file_invalid_args(monkeypatch):
    monkeypatch.setattr(gh.urllib.request, "urlopen", _fake_urlopen_ok("x"))
    assert gh.fetch_raw_file("", "ref", "p") == ""
    assert gh.fetch_raw_file("o/r", "", "p") == ""
    assert gh.fetch_raw_file("o/r", "ref", "") == ""


def test_exports():
    assert "pr_patches" in gh.__all__
    assert "fetch_raw_file" in gh.__all__
