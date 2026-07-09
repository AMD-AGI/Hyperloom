# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Cover the canonical repo-URL mapping framework-agent owns; lives here so it runs without inference_optimizer on PYTHONPATH (standalone ``fa`` CLI)."""

from __future__ import annotations

import pytest

from hyperloom.agents.framework.repo_map import (
    KNOWN_FRAMEWORKS,
    _FRAMEWORK_TO_REPO_URL,
    bridge_repo_urls,
    repo_url_for_framework,
)


def test_repo_url_for_framework_known():
    assert repo_url_for_framework("sglang") == ("https://github.com/sgl-project/sglang.git")
    assert repo_url_for_framework("vllm") == ("https://github.com/ROCm/vllm.git")


def test_repo_url_for_atom():
    """atom must resolve to the public ROCm/ATOM repo so ``fa phase-discover --framework atom`` has a target to scout."""
    assert repo_url_for_framework("atom") == ("https://github.com/ROCm/ATOM.git")


def test_repo_url_for_framework_lowercases_and_strips():
    assert repo_url_for_framework("SGLang") == ("https://github.com/sgl-project/sglang.git")
    assert repo_url_for_framework("  vllm  ") == ("https://github.com/ROCm/vllm.git")
    assert repo_url_for_framework("ATOM") == ("https://github.com/ROCm/ATOM.git")
    assert repo_url_for_framework("  Atom  ") == ("https://github.com/ROCm/ATOM.git")


def test_repo_url_for_framework_unknown_returns_empty():
    assert repo_url_for_framework("rust-burn") == ""
    assert repo_url_for_framework("") == ""
    assert repo_url_for_framework("   ") == ""


# ---------------------------------------------------------------------------
# Cross-cutting static guards
# ---------------------------------------------------------------------------
def test_repo_url_for_xdit():
    """xdit must resolve to the upstream xDiT repo so framework discovery on the diffusion framework has a target."""
    assert repo_url_for_framework("xdit") == ("https://github.com/xdit-project/xDiT.git")


def test_repo_map_known_frameworks():
    """The canonical dict must enumerate exactly the supported frameworks (pinned so future additions update this test intentionally)."""
    assert set(_FRAMEWORK_TO_REPO_URL.keys()) == {"sglang", "vllm", "atom", "xdit"}


# ---------------------------------------------------------------------------
# Enablement bridge repos (separate from serving frameworks)
# ---------------------------------------------------------------------------
def test_bridge_repos_for_rocm_hip_layer():
    """rocm_hip failures scout aiter/HIP/ROCm for an enabling PR."""
    urls = bridge_repo_urls("rocm_hip")
    assert "https://github.com/ROCm/aiter.git" in urls
    assert "https://github.com/ROCm/HIP.git" in urls
    assert "https://github.com/ROCm/ROCm.git" in urls


def test_bridge_repos_build_layer_is_aiter():
    """build-layer failures scout aiter."""
    assert bridge_repo_urls("build") == ("https://github.com/ROCm/aiter.git",)


def test_bridge_repos_framework_layer_is_empty():
    """The framework layer needs no bridge repos (caller has the framework repo)."""
    assert bridge_repo_urls("framework") == ()
    assert bridge_repo_urls("") == ()
    assert bridge_repo_urls("nonsense") == ()


def test_bridge_repos_case_insensitive():
    """Layer lookup is case-insensitive / whitespace-tolerant."""
    assert bridge_repo_urls("  ROCM_HIP ") == bridge_repo_urls("rocm_hip")


def test_bridge_repos_do_not_pollute_known_frameworks():
    """Bridge repos must NOT leak into the serving-framework set."""
    assert "aiter" not in KNOWN_FRAMEWORKS
    assert "hip" not in KNOWN_FRAMEWORKS
    assert "rocm" not in KNOWN_FRAMEWORKS


def test_known_frameworks_constant_matches_dict():
    """KNOWN_FRAMEWORKS must derive from the URL dict so a new entry auto-propagates to validation sites that previously hardcoded the set."""
    assert KNOWN_FRAMEWORKS == frozenset(_FRAMEWORK_TO_REPO_URL.keys())
    assert "atom" in KNOWN_FRAMEWORKS


def test_repo_map_in_sync_with_io_fallback():
    """orchestrator's framework client re-exports this module's ``repo_url_for_framework``
    directly, since framework-agent is now always importable alongside orchestrator.
    Skipped when inference_optimizer is absent."""
    pytest.importorskip("hyperloom.orchestrator.framework.client")
    from hyperloom.orchestrator.framework import client as fac

    assert fac.repo_url_for_framework is repo_url_for_framework
