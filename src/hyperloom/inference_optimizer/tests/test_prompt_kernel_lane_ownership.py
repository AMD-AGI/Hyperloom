# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The assembled KERNEL_AGENT prompt agrees with the request-kind ownership tables.

``run_optimization`` / ``run_gemm_tuning`` are Coordinator-owned lanes that
PolicyGate denies from an LLM, so no part of the prompt -- generated sections or
the ``orchestration.md`` rules fragment -- may advertise them as requestable.
"""

from __future__ import annotations

import re

import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import (
    ACTION_CATALOGUE,
    COORDINATOR_OWNED_KERNEL_REQUEST_KINDS,
    KERNEL_ACTION_REQUEST_KINDS,
    KERNEL_REQUEST_KIND_ALIASES,
    REQUEST_KIND_TO_OWNED_ACTION,
)
from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir
from hyperloom.orchestrator.phases import machine_state as _ps
from hyperloom.orchestrator.prompts.prompt_builder import (
    build_orchestration_prompt,
    default_enabled_actions,
)


# ``trace_analyze`` owns no action, so it is absent from the ownership tables
# and cannot be derived from them.
_UNOWNED_REQUESTABLE_KINDS = frozenset({"trace_analyze"})

_BACKTICKED = re.compile(r"`([a-z_]+)`")
_REQUESTABLE_KINDS_SENTENCE = re.compile(r"MUST be EXACTLY one of(?P<kinds>.*?)—", re.DOTALL)


@pytest.fixture(scope="module")
def kernel_prompt() -> str:
    return build_orchestration_prompt(
        action_registry=ACTION_CATALOGUE,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        kernel_enabled=True,
        framework_agent_phase_enabled=True,
        objective_kind="gain_pct",
        objective_value=15.0,
        max_minutes=480,
        phase=_ps.PHASE_KERNEL_AGENT,
        rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
    )


def _expected_requestable_kinds() -> frozenset[str]:
    """Request kinds an LLM may emit in KERNEL_AGENT, per the ownership tables."""
    kinds = set(KERNEL_ACTION_REQUEST_KINDS.values()) | set(KERNEL_REQUEST_KIND_ALIASES)
    kinds -= set(COORDINATOR_OWNED_KERNEL_REQUEST_KINDS)
    proposable = set(_ps.llm_proposable_actions_for(_ps.PHASE_KERNEL_AGENT))
    kinds = {k for k in kinds if REQUEST_KIND_TO_OWNED_ACTION[k] in proposable}
    return frozenset(kinds | _UNOWNED_REQUESTABLE_KINDS)


def test_requestable_kind_whitelist_matches_the_ownership_tables(kernel_prompt):
    """The whitelist the model reads must be exactly the kinds PolicyGate accepts."""
    match = _REQUESTABLE_KINDS_SENTENCE.search(kernel_prompt)
    assert match is not None, "KERNEL_AGENT prompt lost its requestable request-kind whitelist"

    advertised = frozenset(_BACKTICKED.findall(match.group("kinds")))
    assert advertised == _expected_requestable_kinds()


def test_no_request_template_exists_for_a_coordinator_owned_kind(kernel_prompt):
    """A payload template is an invitation; the owned lanes must have none."""
    for kind in sorted(COORDINATOR_OWNED_KERNEL_REQUEST_KINDS):
        for template in (f"kind: '{kind}'", f"kind='{kind}'", f'kind="{kind}"'):
            assert template not in kernel_prompt, f"{kind} still has a request template"

    # The requestable kinds keep theirs, so the assertions above cannot pass by
    # the whole reference section having vanished.
    assert "kind: 'trace_analyze'" in kernel_prompt
    assert "kind: 'integrate'" in kernel_prompt
    assert "## 6. KERNEL-OPT REQUEST REFERENCE" in kernel_prompt
    assert "### Kernel request kinds" in kernel_prompt


def test_trace_analyze_is_not_presented_as_a_run_optimization_prerequisite(kernel_prompt):
    """``run_optimization`` is dispatched at phase entry; nothing the model does gates it."""
    assert "must precede every `run_optimization`" not in kernel_prompt
    assert "### `trace_analyze` — read-only candidate analysis" in kernel_prompt


def test_the_owned_lanes_are_named_as_coordinator_owned(kernel_prompt):
    """Ownership must be stated, not left to be inferred from the catalogue."""
    assert "### `kernel_opt` and `gemm_tuning` — not yours to propose" in kernel_prompt

    owned_bullet = next(
        (
            block
            for block in kernel_prompt.split("\n* ")
            if block.startswith("`run_optimization`") and "NOT yours to request" in block
        ),
        None,
    )
    assert owned_bullet is not None, "the Coordinator-owned request kinds lost their rules-fragment bullet"
    for kind in sorted(COORDINATOR_OWNED_KERNEL_REQUEST_KINDS):
        assert f"`{kind}`" in owned_bullet


def test_no_analysis_recommendation_routes_to_an_owned_request_kind(kernel_prompt):
    """Analysis-driven targeting may only route to actions the model can emit."""
    assert "run `run_gemm_tuning` first" not in kernel_prompt
    assert "## Compute Kernel Optimizations" in kernel_prompt
