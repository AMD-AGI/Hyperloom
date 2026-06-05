"""Pin the Orchestration system prompt's specialist / integrate_patch visibility.

Empirical evidence from a v0.8 12h DeepSeek-R1-0528 run showed that
``storage/coordinator.db tasks`` carried 0 rows of ``kind='specialist'``
for an entire session. Root cause: the orchestration system prompt's
ACTIONS catalogue did not render an entry for ``specialist`` because
no ``actions/_meta/specialist.yaml`` existed for ActionRegistry to load.
The synthetic action was wired into PolicyGate / SpecialistRunner / phase
allowlist, but the LLM literally did not know how to emit it.

PR-A1 (Arbor-into-Hyperloom) adds:

* ``actions/_meta/specialist.yaml`` + ``actions/specialist.md``
* ``actions/_meta/integrate_patch.yaml`` + ``actions/integrate_patch.md``
* ``_format_emit_hint`` clauses for both names in prompt_builder
* ``specialist`` / ``integrate_patch`` in
  ``FULL_ENABLED_ACTIONS`` / ``NO_KERNEL_ENABLED_ACTIONS``
* ``integrate_patch`` in ``PHASE_ALLOWED_ACTIONS[PHASE_EXPLORE]``
* EXPLORE specialist-informed contract in ``orchestration.md``

These tests enforce that those changes hold so a future refactor cannot
silently regress the visibility.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.phase_state import (
    PHASE_ALLOWED_ACTIONS,
    PHASE_EXPLORE,
)
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    NO_KERNEL_ENABLED_ACTIONS,
    build_orchestration_prompt,
)
from inference_optimizer.paths import asset_system_prompts_dir


@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry().load()


@pytest.fixture
def rules_path() -> Path:
    return asset_system_prompts_dir() / "orchestration.md"


def _build_prompt(
    registry: ActionRegistry,
    rules_path: Path,
    *,
    enabled: tuple[str, ...],
    kernel_enabled: bool,
) -> str:
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=enabled,
        framework="vllm",
        kernel_enabled=kernel_enabled,
        objective_kind="gain_pct",
        objective_value=100.0,
        max_minutes=660,
        rules_fragment_path=rules_path,
    )


# ---------------------------------------------------------------------------
# Registry side: both yaml files must load cleanly
# ---------------------------------------------------------------------------
def test_specialist_yaml_loads_into_registry(registry: ActionRegistry) -> None:
    meta = registry.get("specialist")
    assert meta is not None, (
        "actions/_meta/specialist.yaml missing — PR-A1 contract broken"
    )
    assert meta.family == "creative"
    assert "research_lane" in meta.requires_lanes
    assert "emit_intent" in meta.allowed_tools
    # PR-A1: specialists may write source patches into their worktree
    for tool in ("Edit", "Write", "MultiEdit"):
        assert tool in meta.allowed_tools, (
            f"specialist must advertise write tool {tool!r} so the per-domain "
            f"prompt builder advertises the patch-writing capability."
        )


def test_integrate_patch_yaml_loads_into_registry(registry: ActionRegistry) -> None:
    meta = registry.get("integrate_patch")
    assert meta is not None, (
        "actions/_meta/integrate_patch.yaml missing — PR-A1 contract broken"
    )
    assert "server_lifecycle" in meta.requires_lanes
    assert "benchmark_lane" in meta.requires_lanes
    assert "workspace_mutation" in meta.requires_lanes


# ---------------------------------------------------------------------------
# Enabled-actions visibility
# ---------------------------------------------------------------------------
def test_full_enabled_actions_contains_specialist_and_integrate_patch() -> None:
    assert "specialist" in FULL_ENABLED_ACTIONS
    assert "integrate_patch" in FULL_ENABLED_ACTIONS


def test_no_kernel_enabled_actions_contains_specialist_and_integrate_patch() -> None:
    assert "specialist" in NO_KERNEL_ENABLED_ACTIONS
    assert "integrate_patch" in NO_KERNEL_ENABLED_ACTIONS


# ---------------------------------------------------------------------------
# Phase allowlist visibility (PolicyGate R1)
# ---------------------------------------------------------------------------
def test_integrate_patch_allowed_in_explore_phase() -> None:
    allowed = PHASE_ALLOWED_ACTIONS[PHASE_EXPLORE]
    assert "integrate_patch" in allowed
    # ``specialist`` should already be there from v0.8 M5; assert that
    # to anchor the EXPLORE specialist-informed contract.
    assert "specialist" in allowed


# ---------------------------------------------------------------------------
# Rendered prompt: EMIT hints must be present in both kernel + no_kernel modes
# ---------------------------------------------------------------------------
def test_specialist_emit_hint_in_full_prompt(
    registry: ActionRegistry, rules_path: Path,
) -> None:
    prompt = _build_prompt(
        registry, rules_path,
        enabled=FULL_ENABLED_ACTIONS, kernel_enabled=True,
    )
    assert "EMIT: delegate{action_name='specialist'" in prompt, (
        "Orchestration prompt missing the specialist EMIT hint — the LLM "
        "will not learn how to fan out specialists."
    )
    # Required payload fields must be advertised so the LLM does not
    # produce a payload PolicyGate's ``specialist_dispatch_source`` rule
    # rejects.
    assert "domain=" in prompt
    assert "gap_canonical_id" in prompt


def test_integrate_patch_emit_hint_in_full_prompt(
    registry: ActionRegistry, rules_path: Path,
) -> None:
    prompt = _build_prompt(
        registry, rules_path,
        enabled=FULL_ENABLED_ACTIONS, kernel_enabled=True,
    )
    assert "EMIT: delegate{action_name='integrate_patch'" in prompt
    assert "specialist_task_id=" in prompt


def test_specialist_emit_hint_in_no_kernel_prompt(
    registry: ActionRegistry, rules_path: Path,
) -> None:
    prompt = _build_prompt(
        registry, rules_path,
        enabled=NO_KERNEL_ENABLED_ACTIONS, kernel_enabled=False,
    )
    assert "EMIT: delegate{action_name='specialist'" in prompt
    assert "EMIT: delegate{action_name='integrate_patch'" in prompt


# ---------------------------------------------------------------------------
# Orchestration rules fragment: EXPLORE specialist-informed contract
# ---------------------------------------------------------------------------
def test_orchestration_rules_mentions_explore_specialist_informed(
    rules_path: Path,
) -> None:
    text = rules_path.read_text(encoding="utf-8")
    # The fragment must explicitly enumerate ``specialist`` and
    # ``integrate_patch`` in the EXPLORE phase, otherwise the rendered
    # prompt's prose contradicts the catalogue.
    assert "EXPLORE" in text
    assert "specialist" in text
    assert "integrate_patch" in text
    assert "Specialist-informed" in text
    assert "llm_direct" in text
    assert "needs_gpu" in text
    assert "gpu_count" in text
    assert "specialist_gpu_pool_disabled" in text
    assert "All-llm_direct grids are denied" not in text
    assert "explore_requires_specialist_provenance" not in text
