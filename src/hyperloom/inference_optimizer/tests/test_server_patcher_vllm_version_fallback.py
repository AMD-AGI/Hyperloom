"""vLLM TraceLens patch version-fallback resolution."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors import _server_patcher as sp

_RC_DEV_LOCAL = "0.22.1rc1.dev259+g23d65ff98.localfp8.aiter1be6"


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


@pytest.mark.parametrize(
    "versions",
    [
        pytest.param(["0.20.0", "0.21.0"], id="highest-same-minor"),
        pytest.param(["0.21.0", "0.21.10"], id="never-a-newer-same-minor-patch"),
    ],
)
def test_same_minor_fallback(tmp_path, versions):
    d = _mk_patches(tmp_path, versions)
    # No exact patch -> highest same-minor patch that is not newer than the running version.
    got = sp._resolve_vllm_patch_file(d, "0.21.5")
    assert got is not None and got.name == "config_vllm_v0.21.0.patch"


@pytest.mark.parametrize(
    "running",
    [
        pytest.param(_RC_DEV_LOCAL, id="rc-dev-local-build"),
        pytest.param("0.22.0+rocm722", id="local-build-suffix"),
    ],
)
def test_suffixed_version_uses_same_minor_release_patch(tmp_path, running):
    d = _mk_patches(tmp_path, ["0.21.0", "0.22.0"])
    got = sp._resolve_vllm_patch_file(d, running)
    assert got is not None and got.name == "config_vllm_v0.22.0.patch"


@pytest.mark.parametrize(
    "running",
    [
        pytest.param("0.22.0", id="release"),
        pytest.param(_RC_DEV_LOCAL, id="rc-dev-local-build"),
    ],
)
def test_nearest_lower_when_minor_missing(tmp_path, running):
    d = _mk_patches(tmp_path, ["0.20.0", "0.21.0"])
    got = sp._resolve_vllm_patch_file(d, running)
    assert got is not None and got.name == "config_vllm_v0.21.0.patch"


@pytest.mark.parametrize(
    "running",
    [
        # Never apply a NEWER patch to an OLDER vLLM (may reference absent symbols).
        pytest.param("0.19.0", id="only-higher-patches"),
        pytest.param("", id="no-numeric-version"),
    ],
)
def test_no_patch_fits_returns_none(tmp_path, running):
    d = _mk_patches(tmp_path, ["0.21.0"])
    assert sp._resolve_vllm_patch_file(d, running) is None


def test_env_exact_pin(tmp_path, monkeypatch):
    d = _mk_patches(tmp_path, ["0.20.0", "0.21.0"])
    monkeypatch.setenv("HYPERLOOM_VLLM_PATCH_EXACT_VERSIONS", "0.20.0")
    # The env pin forces 0.20.0.
    got = sp._resolve_vllm_patch_file(d, "0.22.0")
    assert got is not None and got.name == "config_vllm_v0.20.0.patch"


def test_missing_dir_returns_none(tmp_path):
    assert sp._resolve_vllm_patch_file(tmp_path / "nope", "0.21.0") is None
