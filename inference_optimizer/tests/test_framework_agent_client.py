# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""framework_agent_client helper-utility tests (``repo_url_for_framework`` + ``_resolve_fa_binary``)."""

from __future__ import annotations

import stat
from pathlib import Path

from inference_optimizer.orchestrator import framework_agent_client as fac


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
    assert fac.repo_url_for_framework("sglang") == (
        "https://github.com/sgl-project/sglang.git"
    )
    assert fac.repo_url_for_framework("vllm") == (
        "https://github.com/ROCm/vllm.git"
    )


def test_repo_url_for_framework_atom():
    """atom resolves to ROCm/ATOM whether or not ``framework_agent`` is importable."""
    assert fac.repo_url_for_framework("atom") == (
        "https://github.com/ROCm/ATOM.git"
    )


def test_repo_url_for_framework_unknown_returns_empty():
    assert fac.repo_url_for_framework("rust-burn") == ""
    assert fac.repo_url_for_framework("") == ""


def test_fallback_dict_has_atom_entry():
    """Introspect the ImportError-branch fallback dict via AST so its contract is pinned regardless of PYTHONPATH."""
    import ast
    import pathlib

    src = pathlib.Path(fac.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fallback: dict[str, str] | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_FRAMEWORK_TO_REPO_URL"
            and node.value is not None
        ):
            fallback = ast.literal_eval(node.value)
            break
    assert fallback is not None
    assert fallback.get("atom") == "https://github.com/ROCm/ATOM.git", (
        f"IO fallback dict missing atom entry: {fallback!r}"
    )
    # Pin all three keys so a future drift trips this guard.
    assert set(fallback.keys()) == {"sglang", "vllm", "atom"}


def test_resolve_fa_binary_prefers_env_var(tmp_path, monkeypatch):
    fa_path = _write_fake_fa(tmp_path, body='#!/usr/bin/env bash\nexit 0\n')
    monkeypatch.setenv("FA_BIN", str(fa_path))
    monkeypatch.setenv("PATH", "")
    assert fac._resolve_fa_binary() == str(fa_path)


def test_resolve_fa_binary_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("FA_BIN", raising=False)
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    monkeypatch.setenv("PATH", "")
    assert fac._resolve_fa_binary() is None
