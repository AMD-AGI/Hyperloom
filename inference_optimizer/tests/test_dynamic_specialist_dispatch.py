"""Regression tests for the absorbed PR #461 free-form dynamic specialist
dispatch.

PR #461 added ``dynamic_specialist`` / ``dynamic_specialist_check`` /
``dynamic_specialist_collect`` to the EXPLORE phase allowlist but never
registered them, so production delegates were denied with ``unknown_action``
and the actions were invisible in the orchestration prompt catalogue. These
tests pin the plumbing added on absorption:

* the three ``_meta`` yamls load into the ActionRegistry;
* the actions are prompt-visible (enabled sets + rendered emit hints);
* they are LLM-proposable in EXPLORE;
* PolicyGate accepts ``delegate{action_name='dynamic_specialist'}`` in
  EXPLORE (no ``unknown_action`` / ``phase_incompatible``);
* the Coordinator's liveness reaper kills an overdue subprocess and marks
  it complete so the run never leaks a zombie agent.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator import phase_state as ps
from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.policy import PolicyDenied, PolicyGate
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    NO_KERNEL_ENABLED_ACTIONS,
    build_orchestration_prompt,
)
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.paths import asset_system_prompts_dir, make_session_dir


_DYNAMIC_SPECIALIST_ACTIONS = (
    "dynamic_specialist",
    "dynamic_specialist_check",
    "dynamic_specialist_collect",
)


# ---------------------------------------------------------------------------
# Reachability: registry + prompt + phase
# ---------------------------------------------------------------------------
@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry().load()


@pytest.mark.parametrize("name", _DYNAMIC_SPECIALIST_ACTIONS)
def test_dynamic_specialist_yaml_loads(registry: ActionRegistry, name: str) -> None:
    meta = registry.get(name)
    assert meta is not None, f"actions/_meta/{name}.yaml missing from registry"
    assert meta.family == "creative"
    assert meta.verdict_class == "exploration"
    assert "emit_intent" in meta.allowed_tools


@pytest.mark.parametrize("name", _DYNAMIC_SPECIALIST_ACTIONS)
def test_dynamic_specialist_in_enabled_sets(name: str) -> None:
    assert name in FULL_ENABLED_ACTIONS
    assert name in NO_KERNEL_ENABLED_ACTIONS


@pytest.mark.parametrize("name", _DYNAMIC_SPECIALIST_ACTIONS)
def test_dynamic_specialist_proposable_in_explore(name: str) -> None:
    assert ps.is_action_allowed_in_phase(name, "EXPLORE")
    assert ps.is_action_llm_proposable_in_phase(name, "EXPLORE")


def test_dynamic_specialist_rendered_in_prompt(registry: ActionRegistry) -> None:
    prompt = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="vllm",
        kernel_enabled=True,
        objective_kind="gain_pct",
        objective_value=100.0,
        max_minutes=660,
        rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
    )
    for name in _DYNAMIC_SPECIALIST_ACTIONS:
        assert f"**{name}**" in prompt, f"{name} missing from action catalogue"
    # The dispatch emit hint must advertise the free-form tasks payload.
    assert "delegate{action_name='dynamic_specialist', params={" in prompt
    assert "tasks=" in prompt


# ---------------------------------------------------------------------------
# PolicyGate: EXPLORE delegate is reachable (no unknown_action / phase deny)
# ---------------------------------------------------------------------------
class _PhaseStub:
    """Minimal SharedState stand-in for the phase contract check."""

    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.tick = 0
        self.precision = "bf16"


def _explore_gate() -> PolicyGate:
    return PolicyGate(
        role_registry=default_role_registry(),
        action_registry=ActionRegistry().load(),
        shared_state=_PhaseStub("EXPLORE"),
        strict_phase=True,
    )


@pytest.mark.parametrize("name", _DYNAMIC_SPECIALIST_ACTIONS)
def test_explore_delegate_dynamic_specialist_not_denied(name: str) -> None:
    gate = _explore_gate()
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": name,
            "params": {
                "tasks": [
                    {"task_description": "investigate decode", "task_summary": "x"},
                ],
            },
        },
    )
    # Must not raise — registered action, EXPLORE-proposable.
    gate.validate_intent("orchestration", intent)


def test_unregistered_action_still_denied() -> None:
    """Control: the unknown_action guard still bites for a bogus name, so
    the registration above is what unblocks dynamic_specialist."""
    gate = _explore_gate()
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "dynamic_specialist_bogus", "params": {}},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "unknown_action"


# ---------------------------------------------------------------------------
# Liveness reap
# ---------------------------------------------------------------------------
@pytest.fixture
def session_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_coordinator(session_dir):
    from inference_optimizer.orchestrator.backends import MockBackend, ScriptedPlan
    from inference_optimizer.orchestrator.coordinator import Coordinator

    silent = ScriptedPlan(turns=[])
    return Coordinator(
        session_dir,
        backends={
            "orchestration": MockBackend(silent, name="o"),
            "kernel": MockBackend(silent, name="k"),
            "critic": MockBackend(silent, name="c"),
            "robustness": MockBackend(silent, name="r"),
        },
    )


@pytest.mark.asyncio
async def test_reap_kills_overdue_dynamic_specialist(session_dir):
    from inference_optimizer.orchestrator.dynamic_dispatch_comms import (
        TaskManifest,
        read_completion,
        write_task_manifest,
    )

    c = _silent_coordinator(session_dir)
    # Spawn a real, long-lived process in its own session so the reaper has
    # a genuine process group to tear down.
    proc = subprocess.Popen(["sleep", "300"], start_new_session=True)
    agent_id = "specialist-test-0001"
    agent_dir = Path(session_dir) / "agents" / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "process.log").write_text("started\n")
    # Dispatched 10 minutes ago with a 1-minute budget => overdue.
    write_task_manifest(
        str(session_dir),
        TaskManifest(
            agent_id=agent_id,
            task_description="overdue task",
            session_dir=str(session_dir),
            dispatched_at=(
                datetime.now(timezone.utc) - timedelta(minutes=10)
            ).isoformat(),
            timeout_minutes=1,
            pid=proc.pid,
        ),
    )

    try:
        c._reap_dynamic_specialists()

        # A synthetic completion report must now exist.
        report = read_completion(str(session_dir), agent_id)
        assert report is not None
        assert report.status == "timeout"

        # The process group must have been signalled; the child dies shortly.
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        assert proc.poll() is not None, "overdue subprocess was not killed"
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, OSError):
                pass
        await c.stop()
