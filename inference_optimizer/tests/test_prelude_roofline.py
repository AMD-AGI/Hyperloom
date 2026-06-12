# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PRELUDE-bootstrap analysis-task enqueue tests (kind/idempotency/benchmark-script wiring of ``_enqueue_internal_analysis_task``)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator


# Stubs — minimal SharedState + TaskRegistry doubles.
@dataclass
class _BareState:
    baseline_tput: float = 100.0
    cumulative_gain_validated: float = 0.0
    last_roofline_tput: float = 0.0
    auto_roofline_pending_task_id: str = ""
    enable_roofline: bool = True
    current_best: dict[str, Any] = field(default_factory=dict)
    last_baseline: dict[str, Any] = field(default_factory=dict)

    def save(self, _session_dir: Path | None) -> None:
        pass


class _StubTaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, Any] = {}
        self._by_idem: dict[str, Any] = {}

    async def create_or_return_existing(
        self,
        *,
        kind: str,
        params: dict,
        idempotency_key: str,
        **_extras: Any,
    ):
        existing = self._by_idem.get(idempotency_key)
        if existing is not None:
            return existing, True
        import uuid as _uuid
        from inference_optimizer.orchestrator.task_registry import Task

        task = Task(
            task_id=_uuid.uuid4().hex,
            kind=kind,
            state="queued",
            params=dict(params),
            idempotency_key=idempotency_key,
        )
        self._tasks[task.task_id] = task  # type: ignore[assignment]
        self._by_idem[idempotency_key] = task  # type: ignore[assignment]
        return task, False


@pytest.fixture
def coord(tmp_path: Path) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.tasks = _StubTaskRegistry()
    c.knowledge_plane = None
    return c


def test_prelude_initial_roofline_task_contract(coord: Coordinator):
    """The PRELUDE-bootstrap roofline represents baseline, not current_best."""
    coord.shared_state.current_best = {
        "extra_server_args": "--tp 8 --enable-mla",
    }
    coord.shared_state.last_baseline = {
        "benchmark_script": "magpie_serving_bench.sh",
    }

    task = asyncio.run(
        coord._enqueue_internal_analysis_task(reason="prelude_initial"),
    )

    assert task.kind == "roofline"
    assert task.idempotency_key == "internal-analysis-prelude_initial"
    assert task.params["reason"] == "prelude_initial"
    assert task.params["source"] == "coordinator_internal"
    assert "base_extra_args" not in task.params
    assert task.params["benchmark_script"] == "magpie_serving_bench.sh"


def test_prelude_initial_roofline_uses_baseline_server_args(
    coord: Coordinator, monkeypatch,
):
    """PRELUDE roofline injects baseline's own server args, never current_best's."""
    coord.shared_state.current_best = {
        "extra_server_args": "--enable-torch-compile --quantization fp8",
    }
    import inference_optimizer.orchestrator.roofline_ceiling as rc
    monkeypatch.setattr(
        rc, "_read_baseline_yaml_server_args",
        lambda _state: "--attention-backend AITER",
    )

    task = asyncio.run(
        coord._enqueue_internal_analysis_task(reason="prelude_initial"),
    )

    assert task.params["base_extra_args"] == "--attention-backend AITER"
    assert "--enable-torch-compile" not in task.params["base_extra_args"]
    assert "fp8" not in task.params["base_extra_args"]


@pytest.mark.asyncio
async def test_prelude_initial_roofline_is_idempotent(coord: Coordinator):
    """A second call with the same reason returns the same task (no double-enqueue on resume)."""
    first = await coord._enqueue_internal_analysis_task(reason="prelude_initial")
    second = await coord._enqueue_internal_analysis_task(reason="prelude_initial")
    assert first.task_id == second.task_id
    assert len(coord.tasks._tasks) == 1


@pytest.mark.asyncio
async def test_distinct_reasons_produce_distinct_tasks(coord: Coordinator):
    """A watermark-driven roofline is a separate task; the idempotency key is reason-scoped."""
    prelude = await coord._enqueue_internal_analysis_task(
        reason="prelude_initial",
    )
    watermark = await coord._enqueue_internal_analysis_task(
        reason="explore_keep_watermark",
    )
    assert prelude.task_id != watermark.task_id
    assert "internal-analysis-prelude_initial" in coord.tasks._by_idem
    assert "internal-analysis-explore_keep_watermark" in coord.tasks._by_idem


def test_watermark_roofline_inherits_current_best_args(coord: Coordinator):
    """Watermark roofline still profiles the optimized current_best config."""
    coord.shared_state.current_best = {
        "extra_server_args": "--tp 8 --enable-mla",
    }

    task = asyncio.run(
        coord._enqueue_internal_analysis_task(reason="explore_keep_watermark"),
    )

    assert task.params["reason"] == "explore_keep_watermark"
    assert task.params["base_extra_args"] == "--tp 8 --enable-mla"


@pytest.mark.asyncio
async def test_enable_roofline_false_picks_profile_kind(coord: Coordinator):
    """When ``enable_roofline`` is False, the task switches kind to ``profile`` keeping the reason-scoped key."""
    coord.shared_state.enable_roofline = False
    task = await coord._enqueue_internal_analysis_task(reason="prelude_initial")
    assert task.kind == "profile"
    assert task.idempotency_key == "internal-analysis-prelude_initial"
