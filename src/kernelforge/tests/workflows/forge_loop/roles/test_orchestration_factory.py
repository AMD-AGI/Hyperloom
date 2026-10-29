# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Building the planning chain: the context the planner is given and the agents the factory creates."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from kernelforge.workflows.forge_loop.roles import orchestration as orchestration_module
from kernelforge.workflows.forge_loop.roles.orchestration import make_orchestration_service
from kernelforge.tests.workflows.forge_loop.roles._orchestration_fakes import _context


def test_the_planning_context_states_the_editable_source_set() -> None:
    """The planner is told what it may edit, in campaign order."""
    campaign_source_files = (
        "/workspace/aiter/ops/triton/op.py",
        "/workspace/aiter/fused_moe.py",
        "/workspace/aiter/configs/tuned_shapes.csv",
        "/workspace/aiter/configs/dispatch.json",
    )
    context = replace(_context(), editable_sources=campaign_source_files)

    payload = context.to_prompt_dict()

    assert payload["editable_sources"] == list(campaign_source_files)
    assert payload["editable_sources"][0].endswith("op.py")
    assert any(not path.endswith(".py") for path in payload["editable_sources"])


def test_the_editable_source_set_defaults_to_empty_and_rejects_junk() -> None:
    assert _context().to_prompt_dict()["editable_sources"] == []
    with pytest.raises(ValueError, match="editable_sources"):
        replace(_context(), editable_sources=("/workspace/a.py", "  "))
    with pytest.raises(ValueError, match="duplicates"):
        replace(
            _context(),
            editable_sources=("/workspace/a.py", "/workspace/a.py"),
        )


def test_explicit_empty_specialist_registry_is_rejected() -> None:
    with pytest.raises(ValueError, match="definitions must not be empty"):
        make_orchestration_service(
            config=None,
            definitions={},
        )


def test_factory_uses_same_runtime_for_independent_critic_backend(
    monkeypatch,
) -> None:
    runtime = SimpleNamespace(timeout_sec=1800)
    backends = []

    class Backend:
        def __init__(self):
            self.runtime = runtime
            self.name = "test"

    def create_backend(selected_runtime, **_kwargs):
        assert selected_runtime is runtime
        backend = Backend()
        backends.append(backend)
        return backend

    monkeypatch.setattr(
        orchestration_module,
        "create_registered_backend",
        create_backend,
    )
    monkeypatch.setattr(orchestration_module, "agent_runtime", lambda _config: runtime)
    service = make_orchestration_service(
        config=SimpleNamespace(
            workspace="/workspace",
            max_turns=500,
            # The probe fields the factory now reads as attributes rather than through a ``getattr`` default that
            # restated them a fourth time.
            specialist_probe=True,
            specialist_probe_max=6,
            specialist_probe_budget_sec=600.0,
            specialist_probe_scratch_root="",
        ),
        enable_plan_critic=True,
    )

    assert len(backends) == 2 + len(service._definitions)
    assert service._agent.backend is backends[0]
    assert service._plan_critic is not None
    assert service._plan_critic.backend is backends[1]
    assert service._plan_critic.backend is not service._agent.backend
    assert service._plan_critic.backend.runtime is (service._agent.backend.runtime)
    assert service._plan_critic.max_turns == 100
    assert service._plan_critic.timeout_sec == 600
