# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Cover the canonical repo-URL mapping framework-agent owns; lives here so it runs without inference_optimizer on PYTHONPATH (standalone ``fa`` CLI)."""

from __future__ import annotations

import pytest

from framework_agent.repo_map import (
    KNOWN_FRAMEWORKS,
    _FRAMEWORK_TO_REPO_URL,
    repo_url_for_framework,
)


def test_repo_url_for_framework_known():
    assert repo_url_for_framework("sglang") == (
        "https://github.com/sgl-project/sglang.git"
    )
    assert repo_url_for_framework("vllm") == (
        "https://github.com/ROCm/vllm.git"
    )


def test_repo_url_for_atom():
    """atom must resolve to the public ROCm/ATOM repo so ``fa phase-discover --framework atom`` has a target to scout."""
    assert repo_url_for_framework("atom") == (
        "https://github.com/ROCm/ATOM.git"
    )


def test_repo_url_for_framework_lowercases_and_strips():
    assert repo_url_for_framework("SGLang") == (
        "https://github.com/sgl-project/sglang.git"
    )
    assert repo_url_for_framework("  vllm  ") == (
        "https://github.com/ROCm/vllm.git"
    )
    assert repo_url_for_framework("ATOM") == (
        "https://github.com/ROCm/ATOM.git"
    )
    assert repo_url_for_framework("  Atom  ") == (
        "https://github.com/ROCm/ATOM.git"
    )


def test_repo_url_for_framework_unknown_returns_empty():
    assert repo_url_for_framework("rust-burn") == ""
    assert repo_url_for_framework("") == ""
    assert repo_url_for_framework("   ") == ""


# ---------------------------------------------------------------------------
# Cross-cutting static guards
# ---------------------------------------------------------------------------
def test_repo_map_known_frameworks():
    """The canonical dict must enumerate exactly the three supported frameworks (pinned so future additions update this test intentionally)."""
    assert set(_FRAMEWORK_TO_REPO_URL.keys()) == {"sglang", "vllm", "atom"}


def test_known_frameworks_constant_matches_dict():
    """KNOWN_FRAMEWORKS must derive from the URL dict so a new entry auto-propagates to validation sites that previously hardcoded the set."""
    assert KNOWN_FRAMEWORKS == frozenset(_FRAMEWORK_TO_REPO_URL.keys())
    assert "atom" in KNOWN_FRAMEWORKS


def test_repo_map_in_sync_with_io_fallback():
    """G1 — both ``_FRAMEWORK_TO_REPO_URL`` dicts must stay identical so a drift can't silently break the framework_pr loop. Skipped when inference_optimizer is absent."""
    pytest.importorskip("inference_optimizer.orchestrator.framework_agent_client")
    from inference_optimizer.orchestrator import framework_agent_client as fac

    # The IO fallback dict only lives in the ``except ImportError`` branch, so
    # when framework_agent is importable the symbol isn't module-bound; read source.
    fallback = getattr(fac, "_FRAMEWORK_TO_REPO_URL", None)
    if fallback is None:
        import ast
        import textwrap

        src = textwrap.dedent(
            (
                __import__("pathlib").Path(fac.__file__)
                .read_text(encoding="utf-8")
            )
        )
        tree = ast.parse(src)
        fallback = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "_FRAMEWORK_TO_REPO_URL"
                and node.value is not None
            ):
                fallback = ast.literal_eval(node.value)
                break
        assert fallback is not None, (
            "could not locate _FRAMEWORK_TO_REPO_URL fallback dict in "
            "framework_agent_client source"
        )

    assert fallback == _FRAMEWORK_TO_REPO_URL, (
        "IO fallback dict drifted from framework_agent.repo_map; "
        "update both in lock-step."
    )
