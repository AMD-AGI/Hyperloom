"""framework_agent_client helper-utility tests.

Coverage: ``repo_url_for_framework`` lookup table and ``_resolve_fa_binary``
env-var precedence. The legacy ``fetch_pr_candidates`` tests were retired
together with the function itself (no production caller after the
FRAMEWORK_PR phase migration).
"""

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
    """atom_plan/phase3_open_framework_agent 3.1: atom resolves to
    ROCm/ATOM regardless of whether ``framework_agent`` is on the
    import path. ``fac.repo_url_for_framework`` delegates to the
    canonical when available and to the inline fallback otherwise;
    both must produce the same URL."""
    assert fac.repo_url_for_framework("atom") == (
        "https://github.com/ROCm/ATOM.git"
    )


def test_repo_url_for_framework_unknown_returns_empty():
    assert fac.repo_url_for_framework("rust-burn") == ""
    assert fac.repo_url_for_framework("") == ""


def test_fallback_dict_has_atom_entry():
    """Introspect the ImportError-branch fallback dict directly so
    the IO-only path's contract is pinned independently of whether
    the canonical ``framework_agent.repo_map`` is on PYTHONPATH.

    Reads the source via AST so we exercise the literal in the
    fallback ``except ImportError`` arm — when ``framework_agent`` is
    importable (the normal install path), the symbol isn't actually
    bound in the module namespace."""
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
