# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PolicyGate evolution tests."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.inference_optimizer.protocol.intent import (
    Intent,
    IntentType,
)
from hyperloom.orchestrator.policy.gate import (
    ALL_KNOWN_EXTERNAL_TOOL_NAMES,
    CORTEX_KB_READ_TOOL_NAMES,
    KB_WRITE_TOOL_NAMES,
    PR_MONITOR_TOOL_NAMES,
    PolicyDenied,
    PolicyGate,
    TOOL_WHITELIST_BY_ROLE,
    WEB_TOOL_NAMES,
)


def _gate(shared_state=None) -> PolicyGate:
    return PolicyGate(
        role_registry=default_role_registry(),
        shared_state=shared_state,
    )


class _State:
    """Minimal SharedState double for phase + denial bookkeeping."""

    def __init__(self, phase: str = "EXPLORE"):
        self.phase = phase
        self.tick = 0

    def record_policy_denial(self, **_kwargs):  # noqa: D401
        return 1


# 1. Tool registry surface
def test_tool_registry_constants_are_disjoint():
    """The four tool groups are mutually exclusive (Inv-11.3 orthogonality)."""
    pairs = [
        ("KB_WRITE", KB_WRITE_TOOL_NAMES),
        ("CORTEX_KB_READ", CORTEX_KB_READ_TOOL_NAMES),
        ("PR_MONITOR", PR_MONITOR_TOOL_NAMES),
        ("WEB", WEB_TOOL_NAMES),
    ]
    for i, (n1, a) in enumerate(pairs):
        for n2, b in pairs[i + 1 :]:
            overlap = a & b
            assert not overlap, f"{n1} and {n2} tool sets overlap: {overlap!r}"


def test_all_known_external_tool_names_equals_union():
    expected = KB_WRITE_TOOL_NAMES | CORTEX_KB_READ_TOOL_NAMES | PR_MONITOR_TOOL_NAMES | WEB_TOOL_NAMES
    assert ALL_KNOWN_EXTERNAL_TOOL_NAMES == expected


def test_specialist_role_owns_all_readonly_external_tools():
    """The specialist whitelist must contain every readonly external tool."""
    specialist_set = TOOL_WHITELIST_BY_ROLE["specialist"]
    assert WEB_TOOL_NAMES <= specialist_set
    assert PR_MONITOR_TOOL_NAMES <= specialist_set
    assert CORTEX_KB_READ_TOOL_NAMES <= specialist_set
    # KB write surfaces never appear in any role whitelist (R4).
    for tools in TOOL_WHITELIST_BY_ROLE.values():
        assert not (tools & KB_WRITE_TOOL_NAMES)


def test_primary_roles_have_empty_external_tool_set():
    for role_name in ("orchestration", "kernel_agent", "critic", "robustness"):
        assert TOOL_WHITELIST_BY_ROLE[role_name] == frozenset()


def test_no_tools_are_phase_restricted():
    """No tool carries a phase restriction — role isolation is the only R5 gate."""
    from hyperloom.orchestrator.policy import gate as policy_mod

    assert not hasattr(policy_mod, "PHASE_RESTRICTED_TOOLS")


# R4 — intent-level collision via propose_action / delegate / request
def test_propose_action_with_kb_write_name_denied():
    """A ``propose_action`` whose ``action_name`` is a KB write tool is denied with R4."""
    gate = _gate()
    for write_tool in sorted(KB_WRITE_TOOL_NAMES):
        intent = Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": write_tool, "predicted_gain_pct": 0.0},
        )
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_intent("orchestration", intent)
        assert excinfo.value.rule == "kb_write_unauthorized"


def test_delegate_with_kb_write_name_denied():
    gate = _gate()
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "mcp__cortex_kb__propose_point"},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "kb_write_unauthorized"


def test_request_kind_with_kb_write_name_denied():
    """A REQUEST smuggling a KB write tool as its ``kind`` is blocked with R4."""
    gate = _gate()
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
            "kind": "mcp__cortex_kb__propose_point",
        },
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "kb_write_unauthorized"


# 6. R5 — intent-level collision (external readonly tool as action_name)
def test_propose_action_with_external_readonly_tool_name_denied():
    """A ``propose_action`` whose ``action_name`` is a readonly external tool is denied with R5."""
    gate = _gate()
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "mcp__pr_monitor__pr_list",
            "predicted_gain_pct": 0.0,
        },
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "tool_whitelist_role"


def test_delegate_with_websearch_action_name_denied():
    gate = _gate()
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "WebSearch"},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "tool_whitelist_role"


# 7. Specialist runner imports stay closed against policy.py
def test_specialist_runner_constants_derive_from_policy():
    """Specialist runner re-exports the canonical tool sets via :mod:`policy`."""
    from hyperloom.orchestrator.specialists import runner as sr

    assert set(sr.PR_MONITOR_MCP_TOOLS) == PR_MONITOR_TOOL_NAMES
    assert set(sr.CORTEX_KB_READONLY_MCP_TOOLS) == CORTEX_KB_READ_TOOL_NAMES
    # Default whitelist includes emit_intent + web + PR Monitor MCP + the
    # read-only Cortex KB-graph MCP tools, and excludes KB write surfaces.
    # (The cortex_kb read tools are stripped at resolve time when the KB-graph
    # MCP server is not wired; see KnowledgePlane.cortex_enabled.)
    default_set = set(sr.DEFAULT_SPECIALIST_TOOLS)
    assert "emit_intent" in default_set
    assert WEB_TOOL_NAMES <= default_set
    assert PR_MONITOR_TOOL_NAMES <= default_set
    assert CORTEX_KB_READ_TOOL_NAMES <= default_set
    assert not (default_set & KB_WRITE_TOOL_NAMES)
    assert KB_WRITE_TOOL_NAMES <= sr.SPECIALIST_TOOL_DENYLIST


# 8. Inv-11.3 — R4 + R5 are orthogonal (a single intent never trips both)
def test_kb_write_tool_does_not_trip_r5_role_check():
    """A KB write tool in the intent fires ONLY R4; rule order returns R4 first."""
    gate = _gate()
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "mcp__cortex_kb__propose_point",
            "predicted_gain_pct": 0.0,
        },
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "kb_write_unauthorized"
    # The wording calls out KB writes, not tool-whitelist vocabulary.
    assert "cannot invoke tool" not in str(excinfo.value)
