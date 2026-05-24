"""F2-2 — framework_pr_scout sub_kind dispatch validation.

Verifies the per-domain ``sub_kinds`` catalogue field added on
:class:`SpecialistDomain` and the dispatch-time validation hooks in
:meth:`PolicyGate._validate_specialist_dispatch`.

The serving_specialist domain gains exactly one sub_kind so far —
``framework_pr_scout`` — gated end-to-end on
``SharedState.framework_agent_enabled`` (the F0-10 toggle).

Reference: ``plan_roofline_framework/F2_framework_agent.MD`` §F2-2.
"""
from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.policy import (
    PolicyDenied,
    PolicyGate,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.specialist_domains import (
    FRAMEWORK_AGENT_GATED_SUB_KINDS,
    SPECIALIST_DOMAINS,
    get_domain,
)


def _gate(state: SharedState | None) -> PolicyGate:
    """Build a minimal PolicyGate using the canonical 4-agent registry."""
    return PolicyGate(
        role_registry=default_role_registry(),
        action_registry=None,
        session_dir=None,
        strict_paths=False,
        shared_state=state,
    )


def _orch_role(gate: PolicyGate):
    """Return the Orchestration AgentRole (lowercase key per registry)."""
    return gate.role_registry["orchestration"]


def _payload(**extra) -> dict:
    base = {
        "params": {
            "domain": "serving_specialist",
            "gap_canonical_id": "gap.serving.cuda_graph_misses.session-test",
        },
    }
    base["params"].update(extra)
    return base


def test_serving_specialist_lists_framework_pr_scout_sub_kind():
    domain = get_domain("serving_specialist")
    assert domain is not None
    assert "framework_pr_scout" in domain.sub_kinds


def test_other_domains_have_no_sub_kinds_yet():
    """No other domain advertises sub_kinds in F2-2."""
    for d in SPECIALIST_DOMAINS:
        if d.key == "serving_specialist":
            continue
        assert d.sub_kinds == (), (
            f"unexpected sub_kinds on {d.key!r}: {d.sub_kinds!r}"
        )


def test_framework_pr_scout_in_gated_set():
    assert "framework_pr_scout" in FRAMEWORK_AGENT_GATED_SUB_KINDS


def test_dispatch_default_sub_kind_always_allowed():
    """Empty / missing sub_kind -> default per-domain prompt; never blocked."""
    s = SharedState()
    s.framework_agent_enabled = False
    gate = _gate(s)
    gate._validate_specialist_dispatch(_orch_role(gate), _payload())


def test_dispatch_framework_pr_scout_blocked_when_flag_off():
    s = SharedState()
    s.framework_agent_enabled = False
    gate = _gate(s)
    with pytest.raises(PolicyDenied, match="framework-agent-enabled"):
        gate._validate_specialist_dispatch(
            _orch_role(gate),
            _payload(sub_kind="framework_pr_scout"),
        )


def test_dispatch_framework_pr_scout_allowed_when_flag_on():
    s = SharedState()
    s.framework_agent_enabled = True
    gate = _gate(s)
    gate._validate_specialist_dispatch(
        _orch_role(gate),
        _payload(sub_kind="framework_pr_scout"),
    )


def test_dispatch_unknown_sub_kind_blocked_for_serving():
    s = SharedState()
    s.framework_agent_enabled = True
    gate = _gate(s)
    with pytest.raises(PolicyDenied, match="does not support sub_kind"):
        gate._validate_specialist_dispatch(
            _orch_role(gate),
            _payload(sub_kind="totally_unknown"),
        )


def test_dispatch_sub_kind_blocked_on_domain_without_sub_kinds():
    """kernel_switch_specialist has empty sub_kinds — any non-empty
    sub_kind is rejected with the same rule."""
    s = SharedState()
    s.framework_agent_enabled = True
    gate = _gate(s)
    payload = {
        "params": {
            "domain": "kernel_switch_specialist",
            "gap_canonical_id": "gap.kernel.attention",
            "sub_kind": "framework_pr_scout",
        },
    }
    with pytest.raises(PolicyDenied, match="does not support sub_kind"):
        gate._validate_specialist_dispatch(_orch_role(gate), payload)


def test_dispatch_framework_pr_scout_blocked_without_shared_state():
    """Defensive: if PolicyGate has no SharedState wired, the gated
    sub_kind is treated as flag-off and rejected."""
    gate = _gate(None)
    with pytest.raises(PolicyDenied, match="framework-agent-enabled"):
        gate._validate_specialist_dispatch(
            _orch_role(gate),
            _payload(sub_kind="framework_pr_scout"),
        )


# ---------------------------------------------------------------------------
# F2-3 — differential subprocess sandbox via _resolve_tools(sub_kind)
# ---------------------------------------------------------------------------


def _make_runner(extra_tools: tuple[str, ...] = ()):
    from inference_optimizer.orchestrator.specialist_runner import (
        DEFAULT_SPECIALIST_TOOLS,
        SpecialistRunner,
    )

    return SpecialistRunner(
        backend_factory=lambda d: None,
        default_tools=DEFAULT_SPECIALIST_TOOLS + extra_tools,
    )


def test_resolve_tools_default_strips_fa_mcp_tools():
    """Default sub_kind: any pre-loaded ``mcp__fa__*`` tool is stripped."""
    runner = _make_runner((
        "mcp__fa__candidates", "mcp__fa__fetch_pr",
    ))
    tools = runner._resolve_tools()
    assert not any(t.startswith("mcp__fa__") for t in tools), tools


def test_resolve_tools_framework_pr_scout_keeps_fa_mcp_tools():
    """framework_pr_scout sub_kind: ``mcp__fa__*`` tools preserved."""
    runner = _make_runner((
        "mcp__fa__candidates", "mcp__fa__fetch_pr",
    ))
    tools = runner._resolve_tools("framework_pr_scout")
    assert "mcp__fa__candidates" in tools
    assert "mcp__fa__fetch_pr" in tools
    # Bash stays — fa CLI runs through it.
    assert "Bash" in tools


def test_resolve_tools_other_sub_kind_strips_fa_mcp_tools():
    """Any sub_kind other than framework_pr_scout still strips fa MCP."""
    runner = _make_runner(("mcp__fa__candidates",))
    tools = runner._resolve_tools("some_unrelated_sub_kind")
    assert "mcp__fa__candidates" not in tools


def test_prepared_run_carries_sub_kind_field():
    """``_PreparedRun.sub_kind`` is the audit hook for downstream code
    (subprocess sandbox / prompt builder)."""
    from inference_optimizer.orchestrator.specialist_runner import _PreparedRun

    prep = _PreparedRun()
    assert prep.sub_kind == ""
    prep2 = _PreparedRun(sub_kind="framework_pr_scout")
    assert prep2.sub_kind == "framework_pr_scout"
