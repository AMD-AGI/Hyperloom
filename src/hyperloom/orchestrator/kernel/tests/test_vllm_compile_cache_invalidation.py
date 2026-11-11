# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Invalidating Triton sources must also invalidate the active vLLM AOT cache."""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.kernel import apply_kernel_patch as akp


@pytest.mark.parametrize("root_kind", ["default", "vllm", "xdg", "tilde"])
def test_clears_active_vllm_compile_cache_with_triton_sources(tmp_path, monkeypatch, root_kind):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(akp.tempfile, "gettempdir", lambda: str(tmp_path / "isolated-tmp"))
    if root_kind == "vllm":
        cache_root = tmp_path / "custom-vllm"
        monkeypatch.setenv("VLLM_CACHE_ROOT", str(cache_root))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "unused-xdg"))
    elif root_kind == "xdg":
        cache_root = tmp_path / "custom-xdg" / "vllm"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root.parent))
    elif root_kind == "tilde":
        cache_root = home / "custom-vllm"
        monkeypatch.setenv("VLLM_CACHE_ROOT", "~/custom-vllm")
    else:
        cache_root = home / ".cache" / "vllm"
    compiled = cache_root / "torch_compile_cache"
    compiled.mkdir(parents=True)
    retained = cache_root / "other-cache-entry"
    retained.write_text("retain")
    source = home / ".triton" / "cache" / "kernel.source"
    source.parent.mkdir(parents=True)
    source.write_text("compiled Triton source")
    (compiled / "graph.json").write_text(json.dumps({"source": str(source)}))
    target = tmp_path / "kernel.py"
    target.write_text("VALUE = 1\n")

    result = akp._clear_python_kernel_caches(target)

    assert result["status"] == "ok"
    assert not source.exists()
    assert not compiled.exists()
    assert str(compiled) in result["removed"]
    assert retained.read_text() == "retain"
