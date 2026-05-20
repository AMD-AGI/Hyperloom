"""Tests for the 5th Framework role: registry, policy, handlers (P1 PR-C).

Coverage:

* :func:`default_role_registry` includes ``framework`` (5 keys total).
* :func:`roles_for_run` 5-tuple ordering.
* PolicyGate ``FRAMEWORK_OWNED_ACTIONS`` enforcement (delegate denial,
  REQUEST allowlist orch -> framework).
* ``framework_request_handlers`` mock envelopes match the P1 e2e
  expectations (predicted_gain_pct >= 3 so orchestration re-proposes
  framework_integrate; KEEP verdict so the stack grows).
* SharedState framework_role_enabled / framework_ast_scan_enabled /
  framework_ast_frameworks default behaviour + JSON round-trip.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.agent_role import (
    BackendType,
    default_role_registry,
    roles_for_run,
)
from inference_optimizer.orchestrator.framework_request_handlers import (
    FRAMEWORK_REQUEST_HANDLERS,
    framework_integrate_handler,
    framework_optimize_handler,
    get_framework_handler,
)
from inference_optimizer.orchestrator.intent_parser import IntentType
from inference_optimizer.orchestrator.policy import (
    FRAMEWORK_OWNED_ACTIONS,
    KERNEL_OWNED_ACTIONS,
    PolicyDenied,
    PolicyGate,
    REQUEST_ROUTING,
)
from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# Registry + policy basics
# ---------------------------------------------------------------------------
def test_default_registry_includes_framework_5_total():
    reg = default_role_registry()
    assert set(reg) == {
        "orchestration", "kernel", "framework", "critic", "robustness",
    }


def test_framework_role_is_responder_only_claude_tool_using():
    role = default_role_registry()["framework"]
    assert role.backend_type == BackendType.CLAUDE
    assert role.no_tools is False
    assert role.can_delegate_side_effects is False
    assert role.can_mutate_core_state is False
    # Allowed intents — base set + RESPONSE + UPDATE_STATE only.
    assert IntentType.RESPONSE in role.allowed_intents
    assert IntentType.UPDATE_STATE in role.allowed_intents
    # Forbidden — no initiation primitives, no review.
    for forbidden in (
        IntentType.PROPOSE_ACTION,
        IntentType.DELEGATE,
        IntentType.REQUEST,
        IntentType.REVIEW_VERDICT,
    ):
        assert forbidden not in role.allowed_intents


def test_roles_for_run_inserts_framework_between_kernel_and_critic():
    """Framework sits between kernel and critic — symmetric to the
    design §5.2 capability matrix."""
    assert roles_for_run() == (
        "orchestration", "kernel", "framework", "critic", "robustness",
    )


def test_framework_owned_actions_two_disjoint_from_kernel():
    assert FRAMEWORK_OWNED_ACTIONS == frozenset({
        "framework_optimize", "framework_integrate",
    })
    assert FRAMEWORK_OWNED_ACTIONS.isdisjoint(KERNEL_OWNED_ACTIONS)


def test_request_routing_orch_can_target_framework():
    assert REQUEST_ROUTING["orchestration"] == frozenset({
        "kernel", "framework",
    })


# ---------------------------------------------------------------------------
# PolicyGate: framework-owned action delegate denial
# ---------------------------------------------------------------------------
@pytest.fixture
def gate() -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry())


def test_orchestration_cannot_delegate_framework_optimize(gate):
    role = default_role_registry()["orchestration"]
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_delegate(role, {"action_name": "framework_optimize"})
    assert exc.value.rule == "framework_owned_by_framework_agent"
    assert "REQUEST(target_agent='framework'" in str(exc.value)


def test_orchestration_cannot_delegate_framework_integrate(gate):
    role = default_role_registry()["orchestration"]
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_delegate(role, {"action_name": "framework_integrate"})
    assert exc.value.rule == "framework_owned_by_framework_agent"


def test_framework_role_cannot_delegate_at_all(gate):
    """Framework is responder-only — it cannot delegate ANY action."""
    role = default_role_registry()["framework"]
    with pytest.raises(PolicyDenied) as exc:
        gate._validate_delegate(role, {"action_name": "framework_optimize"})
    # Even before reaching the framework_owned check, the role.can_delegate
    # gate fires first.
    assert exc.value.rule in (
        "role", "framework_owned_by_framework_agent",
    )


def test_orchestration_can_request_framework(gate):
    role = default_role_registry()["orchestration"]
    # Should not raise — orch -> framework is in REQUEST_ROUTING.
    gate._validate_request(role, {
        "target_agent": "framework",
        "kind": "framework_optimize",
    })


# ---------------------------------------------------------------------------
# P1 mock handlers — envelopes match orchestration few-shot
# ---------------------------------------------------------------------------
def _run(coro):
    """Run an async coroutine in a fresh loop -- safer than
    asyncio.get_event_loop() under pytest-asyncio interaction.
    """
    return asyncio.new_event_loop().run_until_complete(coro)


def test_framework_optimize_handler_returns_optimize_success(tmp_path: Path):
    payload = {"target_framework": "sglang"}
    result = _run(framework_optimize_handler(payload, session_dir=tmp_path))
    assert result["status"] == "succeeded"
    assert result["payload_kind"] == "OptimizeSuccess"
    # predicted_gain >= 3% so orchestration re-proposes framework_integrate
    assert result["predicted_gain_pct"] >= 3.0
    # patch_path is non-empty under the session_dir runs/framework/ tree
    assert result["patch_path"].startswith(str(tmp_path))
    assert "runs/framework" in result["patch_path"]
    assert result["target_framework"] == "sglang"


def test_framework_integrate_handler_returns_keep_verdict(tmp_path: Path):
    payload = {"patch_id": "fw-test-001"}
    result = _run(framework_integrate_handler(payload, session_dir=tmp_path))
    assert result["status"] == "succeeded"
    assert result["payload_kind"] == "IntegrateSuccess"
    assert result["verdict"] == "KEEP"
    assert result["patch_id"] == "fw-test-001"


def test_handler_dispatch_table_resolves_both_kinds():
    assert get_framework_handler("framework_optimize") is framework_optimize_handler
    assert get_framework_handler("framework_integrate") is framework_integrate_handler
    assert get_framework_handler("unknown_kind") is None
    assert set(FRAMEWORK_REQUEST_HANDLERS) == {
        "framework_optimize", "framework_integrate",
    }


# ---------------------------------------------------------------------------
# SharedState — framework role / AST plumbing fields
# ---------------------------------------------------------------------------
def test_shared_state_default_framework_role_disabled():
    """PR-A1 dead-path default: framework role is OFF unless CLI opts in."""
    s = SharedState()
    assert s.framework_role_enabled is False
    assert s.framework_ast_scan_enabled is True
    assert s.framework_ast_frameworks == ()
    # Legacy framework_enabled (framework_pr arm) defaults True; distinct.
    assert s.framework_enabled is True


def test_shared_state_resume_lacking_new_fields_decodes_as_default(
    tmp_path: Path,
):
    """Older state.json without framework_role_enabled / ast_* must load
    as defaults (False / True / ())."""
    # Hand-craft a minimal state.json missing the new fields.
    sd = tmp_path / "session"
    sd.mkdir()
    legacy = {
        "session_id": "x",
        "model_name": "m",
        "kernel_enabled": True,
        "framework_enabled": True,
    }
    (sd / "state.json").write_text(json.dumps(legacy), encoding="utf-8")
    s = SharedState.load_or_init(sd)
    assert s.framework_role_enabled is False
    assert s.framework_ast_scan_enabled is True
    assert s.framework_ast_frameworks == ()


def test_shared_state_round_trip_keeps_framework_fields(tmp_path: Path):
    sd = tmp_path / "session"
    sd.mkdir()
    s = SharedState(
        session_id="x",
        framework_role_enabled=True,
        framework_ast_scan_enabled=False,
        framework_ast_frameworks=("sglang",),
    )
    s.save(sd)
    s2 = SharedState.load_or_init(sd)
    assert s2.framework_role_enabled is True
    assert s2.framework_ast_scan_enabled is False
    assert s2.framework_ast_frameworks == ("sglang",)
