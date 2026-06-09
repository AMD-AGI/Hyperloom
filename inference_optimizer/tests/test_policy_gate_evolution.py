# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""v0.8 §3.11 — PolicyGate evolution tests."""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.protocol.intent import (
    Intent, IntentType,
)
from inference_optimizer.orchestrator.policy import (
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
        for n2, b in pairs[i + 1:]:
            overlap = a & b
            assert not overlap, (
                f"{n1} and {n2} tool sets overlap: {overlap!r}"
            )


def test_all_known_external_tool_names_equals_union():
    expected = (
        KB_WRITE_TOOL_NAMES
        | CORTEX_KB_READ_TOOL_NAMES
        | PR_MONITOR_TOOL_NAMES
        | WEB_TOOL_NAMES
    )
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
    for role_name in ("orchestration", "kernel", "critic", "robustness"):
        assert TOOL_WHITELIST_BY_ROLE[role_name] == frozenset()


def test_no_tools_are_phase_restricted():
    """No tool carries a phase restriction — role isolation is the only R5 gate."""
    from inference_optimizer.orchestrator import policy as policy_mod
    assert not hasattr(policy_mod, "PHASE_RESTRICTED_TOOLS")


# 2. R4 — kb_write_unauthorized via validate_tool_invocation
@pytest.mark.parametrize("write_tool", sorted(KB_WRITE_TOOL_NAMES))
@pytest.mark.parametrize("role_name", [
    "specialist", "orchestration", "kernel", "critic", "robustness",
])
def test_validate_tool_invocation_blocks_kb_writes_for_every_role(
    write_tool: str, role_name: str,
):
    gate = _gate()
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_tool_invocation(write_tool, source_role=role_name)
    assert excinfo.value.rule == "kb_write_unauthorized"
    assert "KB write" in str(excinfo.value)


def test_validate_tool_invocation_empty_tool_name():
    gate = _gate()
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_tool_invocation("", source_role="specialist")
    assert excinfo.value.rule == "payload"


# 3. R5 — tool_whitelist_role
@pytest.mark.parametrize("read_tool", sorted(
    CORTEX_KB_READ_TOOL_NAMES | PR_MONITOR_TOOL_NAMES | WEB_TOOL_NAMES,
))
def test_validate_tool_invocation_blocks_readonly_external_tools_for_non_specialist(
    read_tool: str,
):
    gate = _gate(_State(phase="EXPLORE"))
    for role_name in ("orchestration", "kernel", "critic", "robustness"):
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_tool_invocation(read_tool, source_role=role_name)
        assert excinfo.value.rule == "tool_whitelist_role"


def test_specialist_can_invoke_readonly_external_tools_in_explore():
    gate = _gate(_State(phase="EXPLORE"))
    for tool in (
        sorted(CORTEX_KB_READ_TOOL_NAMES)
        + sorted(PR_MONITOR_TOOL_NAMES)
        + sorted(WEB_TOOL_NAMES)
    ):
        gate.validate_tool_invocation(tool, source_role="specialist")


# 4. R5 — Web tools are usable in any phase (no phase restriction)
@pytest.mark.parametrize("web_tool", sorted(WEB_TOOL_NAMES))
@pytest.mark.parametrize("phase", ["PRELUDE", "EXPLORE", "KERNEL", "SWEEP", "CLOSE"])
def test_specialist_web_tools_allowed_in_any_phase(
    web_tool: str, phase: str,
):
    gate = _gate(_State(phase=phase))
    gate.validate_tool_invocation(web_tool, source_role="specialist")


def test_specialist_pr_monitor_allowed_in_any_phase():
    """PR Monitor read tools are usable in any phase."""
    for phase in ("PRELUDE", "FRAMEWORK_PR", "EXPLORE", "KERNEL", "SWEEP", "CLOSE"):
        gate = _gate(_State(phase=phase))
        for tool in PR_MONITOR_TOOL_NAMES:
            gate.validate_tool_invocation(tool, source_role="specialist")


def test_validate_tool_invocation_phase_argument_is_noop():
    """The ``phase`` kwarg no longer gates any tool."""
    gate = _gate()
    gate.validate_tool_invocation("WebSearch", source_role="specialist")
    gate.validate_tool_invocation(
        "WebSearch", source_role="specialist", phase="KERNEL",
    )
    # Role isolation still applies regardless of phase.
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_tool_invocation(
            "WebSearch", source_role="orchestration", phase="EXPLORE",
        )
    assert excinfo.value.rule == "tool_whitelist_role"


# 5. R4 — intent-level collision via propose_action / delegate / request
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
            "target_agent": "kernel",
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
    from inference_optimizer.orchestrator import specialist_runner as sr

    assert set(sr.PR_MONITOR_MCP_TOOLS) == PR_MONITOR_TOOL_NAMES
    assert set(sr.CORTEX_KB_READONLY_MCP_TOOLS) == CORTEX_KB_READ_TOOL_NAMES
    # Default whitelist includes emit_intent + web + PR Monitor MCP, and excludes
    # KB write surfaces and the orphan Cortex KB read MCP names.
    default_set = set(sr.DEFAULT_SPECIALIST_TOOLS)
    assert "emit_intent" in default_set
    assert WEB_TOOL_NAMES <= default_set
    assert PR_MONITOR_TOOL_NAMES <= default_set
    assert not (default_set & CORTEX_KB_READ_TOOL_NAMES)
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
