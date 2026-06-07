# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PRELUDE-bootstrap analysis-task enqueue test.

The PRELUDE phase ends with an auto-enqueued analysis task driven by
the baseline-completion hook inside ``_promote_to_shared_state``.
That hook delegates the actual enqueue to
:meth:`Coordinator._enqueue_internal_analysis_task` with the fixed
``reason='prelude_initial'`` idempotency key. The task kind is
``roofline`` when ``shared_state.enable_roofline`` is True (default)
and ``profile`` when False — picked by ``_internal_analysis_kind``.

This file pins:

* The internal-analysis task contract: kind, idempotency key,
  reason param, and benchmark-script wiring from ``last_baseline``.
* Idempotency: a second enqueue with the same reason returns the
  existing task instead of creating a duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator


# ---------------------------------------------------------------------------
# Stubs — minimal SharedState + TaskRegistry doubles.
# ---------------------------------------------------------------------------
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


@pytest.mark.asyncio
async def test_prelude_initial_roofline_task_contract(coord: Coordinator):
    """The PRELUDE-bootstrap enqueue produces a ``roofline`` task with
    idempotency key ``internal-analysis-prelude_initial`` and carries
    forward the baseline's benchmark-script + current_best extra args
    so the profile sub-step bench against the same workload."""
    coord.shared_state.current_best = {
        "extra_server_args": "--tp 8 --enable-mla",
    }
    coord.shared_state.last_baseline = {
        "benchmark_script": "magpie_serving_bench.sh",
    }

    task = await coord._enqueue_internal_analysis_task(reason="prelude_initial")

    assert task.kind == "roofline"
    assert task.idempotency_key == "internal-analysis-prelude_initial"
    assert task.params["reason"] == "prelude_initial"
    assert task.params["source"] == "coordinator_internal"
    assert task.params["base_extra_args"] == "--tp 8 --enable-mla"
    assert task.params["benchmark_script"] == "magpie_serving_bench.sh"


@pytest.mark.asyncio
async def test_prelude_initial_roofline_is_idempotent(coord: Coordinator):
    """A second call with ``reason='prelude_initial'`` returns the
    same task — resume after the baseline-completion edge must not
    double-enqueue."""
    first = await coord._enqueue_internal_analysis_task(reason="prelude_initial")
    second = await coord._enqueue_internal_analysis_task(reason="prelude_initial")
    assert first.task_id == second.task_id
    assert len(coord.tasks._tasks) == 1


@pytest.mark.asyncio
async def test_distinct_reasons_produce_distinct_tasks(coord: Coordinator):
    """A subsequent watermark-driven roofline (reason
    ``explore_keep_watermark``) is a separate task from the PRELUDE
    initial — the idempotency key is reason-scoped so the two never
    collapse."""
    prelude = await coord._enqueue_internal_analysis_task(
        reason="prelude_initial",
    )
    watermark = await coord._enqueue_internal_analysis_task(
        reason="explore_keep_watermark",
    )
    assert prelude.task_id != watermark.task_id
    assert "internal-analysis-prelude_initial" in coord.tasks._by_idem
    assert "internal-analysis-explore_keep_watermark" in coord.tasks._by_idem


@pytest.mark.asyncio
async def test_enable_roofline_false_picks_profile_kind(coord: Coordinator):
    """When ``shared_state.enable_roofline`` is False, the internal
    analysis task switches kind to ``profile`` (the lighter
    Coordinator-internal analysis path) while keeping the same
    reason-scoped idempotency key prefix."""
    coord.shared_state.enable_roofline = False
    task = await coord._enqueue_internal_analysis_task(reason="prelude_initial")
    assert task.kind == "profile"
    assert task.idempotency_key == "internal-analysis-prelude_initial"
