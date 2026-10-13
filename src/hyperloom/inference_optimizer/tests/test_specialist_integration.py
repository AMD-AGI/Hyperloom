# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist sub-agent integration smoke test exercising the cli → Coordinator → SubAgentRunner → SpecialistRunner chain with real wiring."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.shared_state import SharedState

from .conftest import use_fake_specialist_cli


@dataclass
class _StubTask:
    """Minimal Task-shaped stub (only ``task_id`` and ``params`` are inspected)."""

    task_id: str
    kind: str = "specialist"
    params: dict[str, Any] | None = None
    requires_lanes: tuple[str, ...] = ()
    lease_ttl_sec: int = 60


class _FakeKnowledgePlane:
    """Minimal KnowledgePlane double with both planes enabled."""

    pr_monitor_enabled = True
    recipe_kb_enabled = True


def _build_args(**overrides) -> argparse.Namespace:
    base = dict(
        claude_model="claude-3-5-sonnet-latest",
        specialist_model=None,
        specialist_max_turns=4,
        research_lane_capacity=1,
        specialist_mcp_config=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_specialist_executor_returns_callable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The cli factory must produce a callable executor."""
    from hyperloom.inference_optimizer.cli.executors import _build_specialist_executor

    use_fake_specialist_cli(tmp_path, monkeypatch, behavior="done_only")
    executor = _build_specialist_executor(
        _build_args(),
        session_dir=tmp_path,
        knowledge_plane=_FakeKnowledgePlane(),
    )
    assert callable(executor), "specialist executor must be a callable"


@pytest.mark.asyncio
async def test_register_executors_registers_specialist_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``_register_executors`` populates the ``specialist`` registry entry when capacity > 0."""
    from hyperloom.inference_optimizer.cli.executors import (
        _build_specialist_executor,
        _register_executors,
    )

    # A stand-in satisfies the _register_executors signature; target is the registry.
    class _StubSub:
        def __init__(self):
            self.registry: dict[str, Any] = {}

        def register_executor(self, kind: str, fn: Any) -> None:
            self.registry[kind] = fn

    class _StubCoord:
        sub = _StubSub()
        shared_state = object()  # RooflineExecutor refuses None; any truthy ref works.

    use_fake_specialist_cli(tmp_path, monkeypatch, behavior="done_only")
    coord = _StubCoord()
    spec_exec = _build_specialist_executor(
        _build_args(research_lane_capacity=1),
        session_dir=tmp_path,
        knowledge_plane=_FakeKnowledgePlane(),
    )
    _register_executors(
        coord,
        # skip kernel-only kinds
        session_dir=tmp_path,
        specialist_executor=spec_exec,
    )
    assert "specialist" in coord.sub.registry, (
        "specialist executor must be wired into SubAgentRunner — KB_gaps/Gap-01 root cause"
    )


@pytest.mark.asyncio
async def test_register_executors_omits_specialist_when_capacity_zero(
    tmp_path: Path,
):
    """``--research-lane-capacity 0`` leaves the specialist executor unregistered (fails closed)."""
    from hyperloom.inference_optimizer.cli.executors import _register_executors

    class _StubSub:
        def __init__(self):
            self.registry: dict[str, Any] = {}

        def register_executor(self, kind: str, fn: Any) -> None:
            self.registry[kind] = fn

    class _StubCoord:
        sub = _StubSub()
        shared_state = object()  # RooflineExecutor refuses None; any truthy ref works.

    coord = _StubCoord()
    # specialist_executor=None mirrors cli's gating when capacity == 0.
    _register_executors(
        coord,
        session_dir=tmp_path,
        specialist_executor=None,
    )
    assert "specialist" not in coord.sub.registry


# 3. SpecialistDispatchCollaborator.warm_specialist_params populates task params
@pytest.mark.asyncio
async def test_warm_specialist_params_fills_pr_monitor_available(tmp_path: Path):
    """Warmup populates pr_monitor_available and warm-start fields."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.knowledge_plane = _FakeKnowledgePlane()

    state = SharedState(
        gpu_type="MI300X",
        warm_start_recipe={"backend": "sglang", "tp": 8},
        warm_start_pitfalls=["avoid --max-num-seqs 1024 on MoE"],
    )
    coord.shared_state = state

    params: dict = {"domain": "serving_specialist"}
    await coord.specialist_dispatch.warm_specialist_params(params)

    assert params["pr_monitor_available"] is True
    assert params["warm_start_recipe"]["backend"] == "sglang"
    assert "avoid --max-num-seqs 1024 on MoE" in params["warm_start_pitfalls"]
    assert params["gpu_type"] == "MI300X"


@pytest.mark.asyncio
async def test_warm_specialist_params_graceful_when_plane_is_none(tmp_path: Path):
    """``--degraded-kb`` (knowledge_plane=None) sets pr_monitor_available=False."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.knowledge_plane = None

    coord.shared_state = SharedState()

    params: dict = {"domain": "serving_specialist"}
    await coord.specialist_dispatch.warm_specialist_params(params)
    assert params["pr_monitor_available"] is False


# 4. End-to-end: SubAgentRunner dispatches a specialist via the adapter
@pytest.mark.asyncio
async def test_specialist_adapter_run_returns_dict_via_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The cli adapter returns a dict carrying runner_status + specialist_done + on-disk artefacts."""
    from hyperloom.inference_optimizer.cli.executors import _build_specialist_executor

    done_payload = {
        "gap_canonical_id": "gap.scheduler.moe",
        "domain": "serving_specialist",
        "proposal_set": [
            {
                "variant_name": "moe_expert_parallel",
                "rationale": "expand expert parallelism per the merged PR",
                "predicted_impact": "5-8% throughput uplift",
                "extra_server_args": "--expert-parallel-size 8",
                "extra_envs": {},
                "kb_evidence": ["pr.sgl-project/sglang#1234"],
                "review_notes": "verified by warm PR feed",
                "confidence": 0.7,
            },
        ],
        "summary": ("Surveyed sglang main; expert-parallel scheduling looks safe"),
        "reason": "kb_evidence",
        "confidence": 0.7,
        "new_findings": [],
        "residual_questions": [],
    }
    use_fake_specialist_cli(tmp_path, monkeypatch, behavior="done_only", payload=done_payload)
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    executor = _build_specialist_executor(
        _build_args(),
        session_dir=session_dir,
        knowledge_plane=_FakeKnowledgePlane(),
    )

    task = _StubTask(
        task_id="task-int-1",
        params={
            "domain": "serving_specialist",
            "framework": "sglang",
            "gap_canonical_id": "gap.scheduler.moe",
            "max_turns": 4,
        },
    )
    ctx = RunnerContext(task=task, lease=None, extra={})
    result_dict = await executor(ctx)

    # Adapter contract — must be a dict (SubAgentRunner writes it to the bus).
    assert isinstance(result_dict, dict)
    assert result_dict["runner_status"] == "succeeded"
    assert result_dict["task_id"] == "task-int-1"
    assert result_dict["domain"] == "serving_specialist"
    assert result_dict["gap_canonical_id"] == "gap.scheduler.moe"

    sd = result_dict["specialist_done"]
    assert sd["proposal_set"]
    assert len(sd["proposal_set"]) == 1
    assert sd["proposal_set"][0]["variant_name"] == "moe_expert_parallel"

    # On-disk artefacts (a transcript per specialist is required).
    workspace = session_dir / "runs" / "specialist" / "task-int-1"
    assert (workspace / "prompt.md").exists()
    assert (workspace / "specialist_done.json").exists()
    assert (workspace / "transcript.jsonl").exists()
    on_disk = json.loads((workspace / "specialist_done.json").read_text())
    for key in (
        "gap_canonical_id",
        "domain",
        "proposal_set",
        "summary",
        "reason",
        "confidence",
    ):
        assert on_disk[key] == sd[key], f"on-disk vs return diff at {key!r}"


@pytest.mark.asyncio
async def test_specialist_adapter_synthesises_empty_done_on_runner_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When the specialist CLI dies without a specialist_done, the adapter synthesises a well-formed empty dict."""
    from hyperloom.inference_optimizer.cli.executors import _build_specialist_executor

    use_fake_specialist_cli(tmp_path, monkeypatch, behavior="crash")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    executor = _build_specialist_executor(
        _build_args(specialist_max_turns=2),
        session_dir=session_dir,
        knowledge_plane=_FakeKnowledgePlane(),
    )
    task = _StubTask(
        task_id="task-stale-1",
        params={
            "domain": "serving_specialist",
            "framework": "sglang",
            "gap_canonical_id": "gap.x",
            "max_turns": 2,
        },
    )
    ctx = RunnerContext(task=task, lease=None, extra={})
    result_dict = await executor(ctx)

    # A CLI that exits non-zero is a backend error, so the run is stale rather than merely empty.
    assert result_dict["runner_status"] == "stale"
    sd = result_dict["specialist_done"]
    assert sd["proposal_set"] == []
    # Some specialist_done is always written, even for a crashed CLI.
    workspace = session_dir / "runs" / "specialist" / "task-stale-1"
    assert (workspace / "specialist_done.json").exists()


# 5. CLI argparse surface — flags are wired
def test_cli_specialist_flags_present():
    """Smoke that the new CLI specialist flags parse."""
    from hyperloom.inference_optimizer import cli as cli_mod

    parser = cli_mod._build_parser()
    args = parser.parse_args(
        [
            "optimize",
            "--model",
            "/tmp/dummy-model",
            "--research-lane-capacity",
            "2",
            "--specialist-max-turns",
            "5",
            "--specialist-model",
            "claude-3-haiku-20240307",
        ]
    )
    assert args.research_lane_capacity == 2
    assert args.specialist_max_turns == 5
    assert args.specialist_model == "claude-3-haiku-20240307"


def test_cli_specialist_flags_have_safe_defaults(monkeypatch):
    from hyperloom.inference_optimizer import cli as cli_mod
    from hyperloom.orchestrator.policy import gate as policy_mod

    # research-lane-capacity default is GPU-derived; pin the GPU count for determinism.
    monkeypatch.setattr(policy_mod, "detect_gpu_count", lambda: 4)
    parser = cli_mod._build_parser()
    args = parser.parse_args(
        [
            "optimize",
            "--model",
            "/tmp/dummy-model",
        ]
    )
    # Default capacity is the research-lane ceiling (2 × visible GPU).
    assert args.research_lane_capacity == policy_mod.research_lane_ceiling()
    from hyperloom.orchestrator.specialists.domains import (
        DEFAULT_SPECIALIST_MAX_TURNS,
    )

    assert args.specialist_max_turns == DEFAULT_SPECIALIST_MAX_TURNS
    # Turn cap is effectively unbounded; the real stop is the wall-clock budget.
    assert args.specialist_max_turns == 1000
    # Specialist model defaults to None → cli falls back to --claude-model.
    assert args.specialist_model is None
