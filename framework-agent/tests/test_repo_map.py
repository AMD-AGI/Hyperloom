"""Cover the canonical repo-URL mapping that framework-agent owns.

Lives here (not in inference_optimizer/tests) so the test runs even when
inference_optimizer isn't on PYTHONPATH — the whole point of moving
``repo_url_for_framework`` to ``framework_agent.repo_map`` was so the
standalone ``fa`` CLI doesn't reverse-import the orchestrator side.
"""

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
    """atom_plan/phase3_open_framework_agent 3.1: atom must resolve to
    the public ROCm/ATOM repo so ``fa phase-discover --framework atom``
    has a target to scout."""
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
    # atom: case + whitespace tolerance follows the same contract.
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
# Phase 3.5 cross-cutting static guards
# ---------------------------------------------------------------------------
def test_repo_map_known_frameworks():
    """The canonical dict must enumerate exactly the three supported
    frameworks. Pinning the set as a fixture forces future additions
    to update this test intentionally (and to verify the IO fallback
    + KNOWN_FRAMEWORKS export stay in sync)."""
    assert set(_FRAMEWORK_TO_REPO_URL.keys()) == {"sglang", "vllm", "atom"}


def test_known_frameworks_constant_matches_dict():
    """KNOWN_FRAMEWORKS is the opt-in import for framework-validation
    sites that previously hardcoded the ``{"sglang", "vllm"}`` literal.
    It must derive from the URL dict so a new entry above auto-
    propagates."""
    assert KNOWN_FRAMEWORKS == frozenset(_FRAMEWORK_TO_REPO_URL.keys())
    assert "atom" in KNOWN_FRAMEWORKS


def test_repo_map_in_sync_with_io_fallback():
    """G1 — both ``_FRAMEWORK_TO_REPO_URL`` dicts must stay byte-for-byte
    identical so a future drift (e.g. adding a key to one but not the
    other) doesn't silently break Hyperloom's framework_pr loop.

    Skipped gracefully when ``inference_optimizer`` is not on the test
    path (framework-agent unit-test sandboxes don't always have it)."""
    pytest.importorskip("inference_optimizer.orchestrator.framework_agent_client")
    from inference_optimizer.orchestrator import framework_agent_client as fac

    # The IO fallback dict only exists in the ``except ImportError``
    # branch. When ``framework_agent`` is importable (the normal path),
    # the symbol ``_FRAMEWORK_TO_REPO_URL`` isn't bound in the
    # ``framework_agent_client`` module namespace — we have to read
    # the source to get at the fallback dict.
    fallback = getattr(fac, "_FRAMEWORK_TO_REPO_URL", None)
    if fallback is None:
        # Re-execute the fallback branch by introspecting the module
        # source. This keeps the test honest even when framework-agent
        # is importable in the active venv.
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
