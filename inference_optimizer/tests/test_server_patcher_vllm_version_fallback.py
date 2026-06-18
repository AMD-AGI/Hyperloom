"""vLLM TraceLens patch version-fallback resolution (Task 4 / P-version).

Covers the tolerant patch picker so a freshly-bumped vLLM image still gets a
nearby (backward-compatible) patch instead of silently losing roofline.
"""
from __future__ import annotations

from inference_optimizer.orchestrator.action_executors import _server_patcher as sp


def _mk_patches(tmp_path, versions):
    d = tmp_path / "vllm_patches"
    d.mkdir()
    for v in versions:
        (d / f"config_vllm_v{v}.patch").write_text("dummy\n", encoding="utf-8")
    return d


def test_exact_match_wins(tmp_path):
    d = _mk_patches(tmp_path, ["0.19.0", "0.20.0", "0.21.0"])
    got = sp._resolve_vllm_patch_file(d, "0.19.0")
    assert got is not None and got.name == "config_vllm_v0.19.0.patch"


def test_same_minor_fallback(tmp_path):
    d = _mk_patches(tmp_path, ["0.20.0", "0.21.0"])
    # 0.21.5 has no exact patch -> highest same-minor (0.21.*) wins.
    got = sp._resolve_vllm_patch_file(d, "0.21.5")
    assert got is not None and got.name == "config_vllm_v0.21.0.patch"


def test_same_minor_never_picks_newer_patch(tmp_path):
    d = _mk_patches(tmp_path, ["0.21.0", "0.21.10"])
    # 0.21.10 is same-minor but newer than running 0.21.5, so skip it.
    got = sp._resolve_vllm_patch_file(d, "0.21.5")
    assert got is not None and got.name == "config_vllm_v0.21.0.patch"


def test_nearest_lower_fallback(tmp_path):
    d = _mk_patches(tmp_path, ["0.20.0", "0.21.0"])
    # 0.22.0 -> nearest patch whose version is <= running -> 0.21.0.
    got = sp._resolve_vllm_patch_file(d, "0.22.0")
    assert got is not None and got.name == "config_vllm_v0.21.0.patch"


def test_only_higher_returns_none(tmp_path):
    # Never apply a NEWER patch to an OLDER vLLM (may reference absent symbols).
    d = _mk_patches(tmp_path, ["0.21.0"])
    assert sp._resolve_vllm_patch_file(d, "0.19.0") is None


def test_env_exact_pin(tmp_path, monkeypatch):
    d = _mk_patches(tmp_path, ["0.20.0", "0.21.0"])
    monkeypatch.setenv("HYPERLOOM_VLLM_PATCH_EXACT_VERSIONS", "0.20.0")
    # Running 0.22.0 would pick 0.21.0, but the env pin forces 0.20.0.
    got = sp._resolve_vllm_patch_file(d, "0.22.0")
    assert got is not None and got.name == "config_vllm_v0.20.0.patch"


def test_missing_dir_returns_none(tmp_path):
    assert sp._resolve_vllm_patch_file(tmp_path / "nope", "0.21.0") is None


def test_no_numeric_version_returns_none(tmp_path):
    d = _mk_patches(tmp_path, ["0.21.0"])
    assert sp._resolve_vllm_patch_file(d, "") is None
