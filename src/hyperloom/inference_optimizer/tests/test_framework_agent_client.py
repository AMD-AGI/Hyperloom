# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""framework_agent_client helper-utility tests (``repo_url_for_framework`` + ``_resolve_fa_binary``)."""

from __future__ import annotations

import stat
from pathlib import Path

from hyperloom.orchestrator.framework import client as fac


def _write_fake_fa(
    tmp_path: Path,
    *,
    name: str = "fa",
    body: str,
) -> Path:
    fa_path = tmp_path / name
    fa_path.write_text(body, encoding="utf-8")
    fa_path.chmod(fa_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return fa_path


def test_repo_url_for_framework_known():
    assert fac.repo_url_for_framework("sglang") == ("https://github.com/sgl-project/sglang.git")
    assert fac.repo_url_for_framework("vllm") == ("https://github.com/ROCm/vllm.git")


def test_repo_url_for_framework_atom():
    """atom resolves to ROCm/ATOM."""
    assert fac.repo_url_for_framework("atom") == ("https://github.com/ROCm/ATOM.git")


def test_repo_url_for_framework_unknown_returns_empty():
    assert fac.repo_url_for_framework("rust-burn") == ""
    assert fac.repo_url_for_framework("") == ""


def test_fac_repo_url_for_framework_is_the_canonical_repo_map():
    """``client.py``'s ``repo_url_for_framework`` is a direct re-export of the
    canonical ``hyperloom.agents.framework.repo_map`` implementation."""
    from hyperloom.agents.framework.repo_map import (
        _FRAMEWORK_TO_REPO_URL,
        repo_url_for_framework,
    )

    assert fac.repo_url_for_framework is repo_url_for_framework
    assert _FRAMEWORK_TO_REPO_URL.get("atom") == "https://github.com/ROCm/ATOM.git"
    assert set(_FRAMEWORK_TO_REPO_URL.keys()) == {"sglang", "vllm", "atom", "xdit"}


def test_resolve_fa_binary_prefers_env_var(tmp_path, monkeypatch):
    fa_path = _write_fake_fa(tmp_path, body="#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("FA_BIN", str(fa_path))
    monkeypatch.setenv("PATH", "")
    assert fac._resolve_fa_binary() == str(fa_path)


def test_resolve_fa_binary_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("FA_BIN", raising=False)
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    monkeypatch.setenv("PATH", "")
    assert fac._resolve_fa_binary() is None
