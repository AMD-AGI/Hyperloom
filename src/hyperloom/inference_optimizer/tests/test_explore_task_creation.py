# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The explore task a delegated grid becomes: its whole grid, seeded with the stack it measures on."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType


@dataclass
class _BareSharedState:
    """SharedState double exposing the fields explore task creation touches."""

    recipe_kb_session_id: str = "sid-test"
    save_count: int = 0
    # Empty string means "nothing in flight"; the auto-roofline dispatch gate is a no-op.
    auto_roofline_pending_task_id: str = ""
    # Fields read by inject_explore_runtime_params and related helpers.
    baseline_runtime_sec: float = 0.0
    baseline_accuracy: float = 0.0
    baseline_warm_runtime_sec: float = 0.0
    explore_search: dict = field(default_factory=dict)
    macro_cycle: int = 0
    specialist_rounds: list = field(default_factory=list)

    def save(self, _session_dir: Path | None) -> None:
        self.save_count += 1


@dataclass
class _BusMessage:
    from_agent: str
    to_agent: str
    topic: str
    payload: dict[str, Any]
    msg_id: str = ""


class _StubBus:
    """MessageBus double that captures every appended message."""

    def __init__(self) -> None:
        self.messages: list[_BusMessage] = []

    async def append_and_seq(self, msg: Any) -> Any:
        bm = _BusMessage(
            from_agent=getattr(msg, "from_agent", ""),
            to_agent=getattr(msg, "to_agent", ""),
            topic=getattr(msg, "topic", ""),
            payload=dict(getattr(msg, "payload", {}) or {}),
            msg_id=getattr(msg, "msg_id", ""),
        )
        self.messages.append(bm)
        return None


class _StubRecipeKB:
    enabled: bool = True

    def __init__(self) -> None:
        self.verify_calls: list[dict[str, Any]] = []

    def verify(self, **kwargs: Any) -> None:
        self.verify_calls.append(dict(kwargs))


def _delegate_coord(tmp_path: Path):
    """Coordinator double reaching the direct explore-task creation path."""
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path

    @dataclass
    class _State(_BareSharedState):
        baseline_config_path: str = ""
        baseline_tput: float = 0.0
        current_best: dict = field(default_factory=dict)
        tick: int = 0

        def is_pruned(self, _action_name: str) -> bool:
            return False

    c.shared_state = _State()
    c.knowledge_plane = KnowledgePlane(recipe_kb=_StubRecipeKB())
    c.bus = _StubBus()
    c.bus.record_observation = AsyncMock()  # type: ignore[method-assign]
    c.router.record_policy_denied = AsyncMock()  # type: ignore[method-assign]
    c.dispatcher.admission_denial_for_action = lambda *a, **k: None  # type: ignore[method-assign]
    c.dispatcher.registry_lanes_ttl = lambda _name: (set(), 0)  # type: ignore[method-assign]
    c.policy = None
    c.rounds = None
    return c


@pytest.mark.asyncio
async def test_delegate_explore_with_grid_creates_task_directly(tmp_path: Path):
    """A delegate explore with a non-empty grid creates an explore task directly, grid forwarded verbatim."""
    coord = _delegate_coord(tmp_path)
    create_calls: list[dict[str, Any]] = []

    class _TaskRegistry:
        async def queued(self) -> list:
            return []

        async def running(self) -> list:
            return []

        async def find_by_idempotency_key(self, _idempotency_key: str) -> None:
            return None

        async def create_or_return_existing(self, spec):
            create_calls.append(dict(vars(spec)))
            from hyperloom.orchestrator.state.task_registry import Task

            return (
                Task(
                    task_id="t-explore-direct",
                    kind=spec.kind,
                    state="queued",
                    params=spec.params,
                    idempotency_key=spec.idempotency_key,
                ),
                False,
            )

    coord.tasks = _TaskRegistry()
    grid = [
        {"name": "v_a", "extra_args": "--flag-a", "provenance": "specialist:serving_specialist"},
        {"name": "v_b", "extra_args": "--flag-b", "provenance": "specialist:serving_specialist"},
    ]
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "explore",
            "params": {"grid": grid},
        },
    )
    await coord.router.handle_delegate("orchestration", intent)
    assert len(create_calls) == 1
    assert create_calls[0]["kind"] == "explore"
    assert create_calls[0]["params"]["grid"] == grid


@pytest.mark.asyncio
async def test_delegate_explore_seeds_the_stack_with_the_anchor(tmp_path: Path):
    """A delegate explore carries current_best's args/envs, not just its throughput."""
    coord = _delegate_coord(tmp_path)
    coord.shared_state.baseline_tput = 4663.79
    coord.shared_state.current_best = {
        "tput": 7725.6,
        "extra_server_args": "--kv-cache-dtype fp8_e4m3 --max-num-seqs 64",
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
    }
    created: list[dict[str, Any]] = []

    class _TaskRegistry:
        async def queued(self) -> list:
            return []

        async def running(self) -> list:
            return []

        async def find_by_idempotency_key(self, _idempotency_key: str) -> None:
            return None

        async def create_or_return_existing(self, spec):
            created.append(dict(spec.params))
            from hyperloom.orchestrator.state.task_registry import Task

            return (
                Task(
                    task_id="t-explore-stack",
                    kind=spec.kind,
                    state="queued",
                    params=spec.params,
                    idempotency_key=spec.idempotency_key,
                ),
                False,
            )

    coord.tasks = _TaskRegistry()
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "explore",
            "params": {"grid": [{"name": "v_a", "extra_args": "--flag-a"}]},
        },
    )
    await coord.router.handle_delegate("orchestration", intent)
    assert len(created) == 1
    params = created[0]
    assert params["base_tput"] == 7725.6
    assert params["base_extra_args"] == "--kv-cache-dtype fp8_e4m3 --max-num-seqs 64"
    assert params["base_extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}
