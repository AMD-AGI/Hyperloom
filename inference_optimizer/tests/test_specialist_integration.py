# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""v0.8 §3.5 / M5 / KB_gaps/Gap-01 — specialist sub-agent integration smoke test exercising the cli → Coordinator → SubAgentRunner → SpecialistRunner chain with real wiring."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends.mock_backend import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.protocol.intent import (
    Intent, IntentType,
)
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext


# Fixtures — mocks limited to network surfaces (LLM + KB)
@dataclass
class _StubTask:
    """Minimal Task-shaped stub (only ``task_id`` and ``params`` are inspected)."""
    task_id: str
    kind: str = "specialist"
    params: dict[str, Any] | None = None
    requires_lanes: tuple[str, ...] = ()
    lease_ttl_sec: int = 60


class _StubPRSummary:
    """PRSummary-shaped duck for the warmup adapter."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeKnowledgePlane:
    """Minimal KnowledgePlane double returning a deterministic PR feed with both planes enabled."""

    pr_monitor_enabled = True
    cortex_enabled = True

    def __init__(self, prs: list | None = None):
        self._prs = prs if prs is not None else [
            _StubPRSummary(
                repo="sgl-project/sglang",
                number=1234,
                title="Add MoE expert parallel scheduling",
                url="https://example.test/pr/1234",
                state="open",
                labels=("moe", "perf"),
                author="alice",
            ),
        ]

    def pr_feed_warm(self, domain: str, **_kwargs):
        # Mirrors KnowledgePlane.pr_feed_warm return shape (list[PRSummary], list[str]).
        return self._prs, []


# 1. cli._build_specialist_executor wiring
def _build_args(**overrides) -> argparse.Namespace:
    base = dict(
        claude_model="claude-3-5-sonnet-latest",
        specialist_model=None,
        specialist_max_turns=4,
        specialist_per_turn_max_seconds=300.0,
        research_lane_capacity=1,
        # PR-A2: use the in-process ClaudeBackend path so mocks work end-to-end.
        specialist_dispatch_mode="inprocess",
        specialist_mcp_config=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_specialist_executor_returns_callable(tmp_path: Path):
    """Gap-01 — the cli factory must produce a callable executor."""
    from inference_optimizer.cli import _build_specialist_executor

    plane = _FakeKnowledgePlane()
    args = _build_args()
    executor = _build_specialist_executor(
        args, session_dir=tmp_path, knowledge_plane=plane,
    )
    assert callable(executor), "specialist executor must be a callable"


# 2. cli._register_executors wires 'specialist' kind end-to-end
@pytest.mark.asyncio
async def test_register_executors_registers_specialist_kind(tmp_path: Path):
    """Gap-01 regression: ``_register_executors`` populates the ``specialist`` registry entry when capacity > 0."""
    from inference_optimizer.cli import (
        _build_specialist_executor,
        _register_executors,
    )

    # A tiny stand-in satisfies the `_register_executors` signature; the target is the registry.
    class _StubSub:
        def __init__(self):
            self.registry: dict[str, Any] = {}

        def register_executor(self, kind: str, fn: Any) -> None:
            self.registry[kind] = fn

    class _StubCoord:
        sub = _StubSub()
        shared_state = object()  # RooflineExecutor refuses None; any truthy ref works.

    coord = _StubCoord()
    args = _build_args(research_lane_capacity=1)
    plane = _FakeKnowledgePlane()
    spec_exec = _build_specialist_executor(
        args, session_dir=tmp_path, knowledge_plane=plane,
    )
    _register_executors(
        coord, no_kernel=True,  # skip kernel-only kinds (independent path)
        session_dir=tmp_path,
        specialist_executor=spec_exec,
    )
    assert "specialist" in coord.sub.registry, (
        "specialist executor must be wired into SubAgentRunner — "
        "KB_gaps/Gap-01 root cause"
    )


@pytest.mark.asyncio
async def test_register_executors_omits_specialist_when_capacity_zero(
    tmp_path: Path,
):
    """``--research-lane-capacity 0`` leaves the specialist executor unregistered (fails closed)."""
    from inference_optimizer.cli import _register_executors

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
        coord, no_kernel=True, session_dir=tmp_path,
        specialist_executor=None,
    )
    assert "specialist" not in coord.sub.registry


# 3. Coordinator._warm_specialist_params populates task params
@pytest.mark.asyncio
async def test_warm_specialist_params_fills_pr_feed_from_plane(tmp_path: Path):
    """Warmup mutates ``params`` with the flattened PR feed, pr_monitor_available, and warm-start fields."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.knowledge_plane = _FakeKnowledgePlane()

    @dataclass
    class _State:
        warm_start_recipe: dict = None
        warm_start_pitfalls: list = None
        warm_start_lessons: list = None
        gpu_type: str = "MI300X"
    state = _State(
        warm_start_recipe={"backend": "sglang", "tp": 8},
        warm_start_pitfalls=["avoid --max-num-seqs 1024 on MoE"],
    )
    coord.shared_state = state

    params: dict = {"domain": "serving_specialist"}
    await coord._warm_specialist_params(params)

    assert "pr_feed" in params
    assert isinstance(params["pr_feed"], list)
    assert len(params["pr_feed"]) == 1
    pr = params["pr_feed"][0]
    assert pr["title"].startswith("Add MoE expert")
    assert pr["repo"] == "sgl-project/sglang"
    assert "moe" in pr["labels"]
    assert params["pr_monitor_available"] is True
    assert params["warm_start_recipe"]["backend"] == "sglang"
    assert "avoid --max-num-seqs 1024 on MoE" in params["warm_start_pitfalls"]
    assert params["gpu_type"] == "MI300X"


@pytest.mark.asyncio
async def test_warm_specialist_params_graceful_when_plane_is_none(tmp_path: Path):
    """``--degraded-kb`` (knowledge_plane=None) still leaves a valid empty ``pr_feed``."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.knowledge_plane = None

    @dataclass
    class _State:
        warm_start_recipe: dict = None
        warm_start_pitfalls: list = None
        warm_start_lessons: list = None
        gpu_type: str = ""
    coord.shared_state = _State()

    params: dict = {"domain": "serving_specialist"}
    await coord._warm_specialist_params(params)
    assert params["pr_feed"] == []
    assert params["pr_monitor_available"] is False


@pytest.mark.asyncio
async def test_warm_specialist_params_respects_explicit_pr_feed(tmp_path: Path):
    """A pre-populated ``params['pr_feed']`` is not clobbered — the explicit value wins."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.knowledge_plane = _FakeKnowledgePlane()

    @dataclass
    class _State:
        warm_start_recipe: dict = None
        warm_start_pitfalls: list = None
        warm_start_lessons: list = None
        gpu_type: str = ""
    coord.shared_state = _State()

    explicit_pr_feed = [{"repo": "x/y", "title": "preset PR", "labels": []}]
    params: dict = {
        "domain": "serving_specialist",
        "pr_feed": explicit_pr_feed,
    }
    await coord._warm_specialist_params(params)
    assert params["pr_feed"] is explicit_pr_feed


# 4. End-to-end: SubAgentRunner dispatches a specialist via the adapter
@pytest.mark.asyncio
async def test_specialist_adapter_run_returns_dict_via_runner(tmp_path: Path):
    """The cli adapter returns a dict carrying runner_status + specialist_done + on-disk artefacts."""
    from inference_optimizer.cli import _build_specialist_executor

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
        "empty": False,
        "summary": (
            "Surveyed sglang main; expert-parallel scheduling looks safe"
        ),
        "reason": "kb_evidence",
        "confidence": 0.7,
        "new_findings": [],
        "residual_questions": [],
    }
    plan = ScriptedPlan(turns=[MockTurn(intents=[
        Intent(type=IntentType.SPECIALIST_DONE, payload=done_payload),
    ])])

    # Monkey-patch ClaudeBackend so the cli factory doesn't reach the real SDK.
    import inference_optimizer.cli_executors as cli_mod
    real_claude_cls = cli_mod.ClaudeBackend
    cli_mod.ClaudeBackend = lambda **_kw: MockBackend(
        plan, name="specialist-mock",
    )
    try:
        args = _build_args()
        executor = _build_specialist_executor(
            args,
            session_dir=tmp_path,
            knowledge_plane=_FakeKnowledgePlane(),
        )

        task = _StubTask(
            task_id="task-int-1",
            params={
                "domain": "serving_specialist",
                "gap_canonical_id": "gap.scheduler.moe",
                "max_turns": 4,
            },
        )
        ctx = RunnerContext(task=task, lease=None, extra={})
        result_dict = await executor(ctx)
    finally:
        cli_mod.ClaudeBackend = real_claude_cls

    # Adapter contract — must be a dict (SubAgentRunner writes it to the bus).
    assert isinstance(result_dict, dict)
    assert result_dict["runner_status"] == "succeeded"
    assert result_dict["task_id"] == "task-int-1"
    assert result_dict["domain"] == "serving_specialist"
    assert result_dict["gap_canonical_id"] == "gap.scheduler.moe"

    sd = result_dict["specialist_done"]
    assert sd["empty"] is False
    assert len(sd["proposal_set"]) == 1
    assert sd["proposal_set"][0]["variant_name"] == "moe_expert_parallel"

    # On-disk artefacts (Inv-5.3 requires a transcript per specialist).
    workspace = tmp_path / "runs" / "specialist" / "task-int-1"
    assert (workspace / "prompt.md").exists()
    assert (workspace / "specialist_done.json").exists()
    assert (workspace / "transcript.jsonl").exists()
    on_disk = json.loads((workspace / "specialist_done.json").read_text())
    for key in (
        "gap_canonical_id", "domain", "proposal_set", "empty",
        "summary", "reason", "confidence",
    ):
        assert on_disk[key] == sd[key], f"on-disk vs return diff at {key!r}"


@pytest.mark.asyncio
async def test_specialist_adapter_synthesises_empty_done_on_runner_failure(
    tmp_path: Path,
):
    """When the runner exhausts max_turns without a specialist_done, the adapter synthesises a well-formed empty dict (Inv-5.3)."""
    from inference_optimizer.cli import _build_specialist_executor

    # Backend keeps emitting heartbeats; never produces a done.
    heartbeat = Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "still working"},
    )
    plan = ScriptedPlan(
        turns=[MockTurn(intents=[heartbeat])],
        loop_last=True,
    )

    import inference_optimizer.cli_executors as cli_mod
    real_claude_cls = cli_mod.ClaudeBackend
    cli_mod.ClaudeBackend = lambda **_kw: MockBackend(plan, name="spec-stale")
    try:
        args = _build_args(specialist_max_turns=2)
        executor = _build_specialist_executor(
            args, session_dir=tmp_path, knowledge_plane=_FakeKnowledgePlane(),
        )
        task = _StubTask(
            task_id="task-stale-1",
            params={
                "domain": "serving_specialist",
                "gap_canonical_id": "gap.x",
                "max_turns": 2,
            },
        )
        ctx = RunnerContext(task=task, lease=None, extra={})
        result_dict = await executor(ctx)
    finally:
        cli_mod.ClaudeBackend = real_claude_cls

    assert result_dict["runner_status"] == "empty_synthesised"
    sd = result_dict["specialist_done"]
    assert sd["empty"] is True
    assert sd["proposal_set"] == []
    # Transcript + done file still on disk (Inv-5.3: some specialist_done is always written).
    workspace = tmp_path / "runs" / "specialist" / "task-stale-1"
    assert (workspace / "specialist_done.json").exists()


# 5. CLI argparse surface — flags are wired
def test_cli_specialist_flags_present():
    """Smoke that the new CLI specialist flags parse."""
    import inference_optimizer.cli as cli_mod

    parser = cli_mod._build_parser()
    args = parser.parse_args([
        "optimize",
        "--model", "/tmp/dummy-model",
        "--research-lane-capacity", "2",
        "--specialist-max-turns", "5",
        "--specialist-per-turn-max-seconds", "120",
        "--specialist-model", "claude-3-haiku-20240307",
    ])
    assert args.research_lane_capacity == 2
    assert args.specialist_max_turns == 5
    assert args.specialist_per_turn_max_seconds == 120.0
    assert args.specialist_model == "claude-3-haiku-20240307"


def test_cli_specialist_flags_have_safe_defaults(monkeypatch):
    import inference_optimizer.cli as cli_mod
    from inference_optimizer.orchestrator import policy as policy_mod

    # research-lane-capacity default is GPU-derived; pin the GPU count for determinism.
    monkeypatch.delenv(
        "INFERENCE_OPTIMIZER_RESEARCH_LANE_CAPACITY", raising=False,
    )
    monkeypatch.setattr(policy_mod, "detect_gpu_count", lambda: 4)
    parser = cli_mod._build_parser()
    args = parser.parse_args([
        "optimize", "--model", "/tmp/dummy-model",
    ])
    # Default capacity is the research-lane ceiling (2 × visible GPU).
    assert args.research_lane_capacity == policy_mod.research_lane_ceiling()
    from inference_optimizer.orchestrator.specialist_domains import (
        DEFAULT_SPECIALIST_MAX_TURNS,
    )
    assert args.specialist_max_turns == DEFAULT_SPECIALIST_MAX_TURNS
    assert args.specialist_max_turns == 12
    assert args.specialist_per_turn_max_seconds == 600.0
    # Specialist model defaults to None → cli falls back to --claude-model.
    assert args.specialist_model is None
    # PR-A2/A3: subprocess dispatch is the production default.
    assert args.specialist_dispatch_mode == "subprocess"
