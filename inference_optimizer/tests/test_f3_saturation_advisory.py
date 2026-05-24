"""F3-4 — Roofline saturation soft advisory.

Three surfaces:

* :func:`derive_saturation_per_direction` — pure parser over an
  Executive Summary table; missing aliases / unparseable cells degrade
  silently to ``0.0``.
* :class:`RooflineExecutor` — appends the parsed snapshot to
  :attr:`SharedState.roofline_saturation_history` (capped at 10) when
  the ``roofline_saturation_advisory`` toggle is on.
* :meth:`SharedState._format_roofline_saturation_advisory` — renders
  the soft prompt advisory only for directions ≥ threshold, gated on
  the toggle, with a ``=== Roofline Saturation Advisory ===`` bookend.

Reference: ``plan_roofline_framework/F3_policygate_advisory.MD`` §F3-4.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors.roofline import (
    make_roofline_executor,
)
from inference_optimizer.orchestrator.roofline_snapshot import (
    SATURATION_ADVISORY_THRESHOLD_PCT,
    derive_saturation_per_direction,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# Parser — derive_saturation_per_direction
# ---------------------------------------------------------------------------

_FULL_EXEC_TABLE = """\
# Executive Summary

| Metric                  | Value           |
| ----------------------- | --------------- |
| Compute %               | 92.5%           |
| Memory %                | 35.0%           |
| Idle %                  | 12.0%           |
| Exposed Communication % | 5.0%            |
| Top Bottleneck Category | MoE_fused (28%) |

## Compute Kernel Optimizations
...
"""


def test_derive_saturation_extracts_all_four_directions():
    sat = derive_saturation_per_direction(_FULL_EXEC_TABLE)
    assert sat == {
        "compute": 92.5,
        "memory": 35.0,
        "host_overhead": 12.0,
        "comm": 5.0,
    }


def test_derive_saturation_falls_back_to_alias_label():
    """``Communication %`` is an accepted fallback for ``Exposed Communication %``."""
    text = (
        "| Metric            | Value |\n"
        "| ----------------- | ----- |\n"
        "| Compute %         | 50.0% |\n"
        "| Communication %   | 99.0% |\n"
    )
    sat = derive_saturation_per_direction(text)
    assert sat["comm"] == 99.0
    assert sat["compute"] == 50.0
    # Missing labels degrade to 0.0.
    assert sat["memory"] == 0.0
    assert sat["host_overhead"] == 0.0


def test_derive_saturation_empty_input_returns_zero_dict():
    sat = derive_saturation_per_direction("")
    assert sat == {
        "compute": 0.0, "memory": 0.0, "host_overhead": 0.0, "comm": 0.0,
    }


def test_derive_saturation_handles_text_without_tables():
    sat = derive_saturation_per_direction("# just a header\nplain prose")
    for v in sat.values():
        assert v == 0.0


# ---------------------------------------------------------------------------
# Prompt advisory — SharedState._format_roofline_saturation_advisory
# ---------------------------------------------------------------------------


def test_advisory_off_by_default_returns_empty():
    s = SharedState()
    assert s._format_roofline_saturation_advisory() == ""


def test_advisory_empty_when_toggle_on_but_no_history():
    s = SharedState()
    s.roofline_saturation_advisory = True
    assert s._format_roofline_saturation_advisory() == ""


def test_advisory_renders_only_directions_above_threshold():
    s = SharedState()
    s.roofline_saturation_advisory = True
    s.roofline_saturation_history = [
        {"snapshot_id": 3, "compute": 92.0, "memory": 35.0,
         "host_overhead": 12.0, "comm": 85.0},
    ]
    out = s._format_roofline_saturation_advisory()
    assert out.startswith("=== Roofline Saturation Advisory (snapshot #3)")
    assert "compute: 92.0% saturated" in out
    assert "comm: 85.0% saturated" in out
    # Sub-threshold directions are absent.
    assert "memory" not in out
    assert "host_overhead" not in out
    assert out.rstrip().endswith("=== End Saturation Advisory ===")


def test_advisory_empty_when_no_direction_above_threshold():
    s = SharedState()
    s.roofline_saturation_advisory = True
    s.roofline_saturation_history = [
        {"snapshot_id": 1, "compute": 50.0, "memory": 60.0,
         "host_overhead": 30.0, "comm": 10.0},
    ]
    assert s._format_roofline_saturation_advisory() == ""


def test_advisory_uses_latest_snapshot_only():
    s = SharedState()
    s.roofline_saturation_advisory = True
    s.roofline_saturation_history = [
        {"snapshot_id": 1, "compute": 95.0, "memory": 0.0,
         "host_overhead": 0.0, "comm": 0.0},
        {"snapshot_id": 2, "compute": 30.0, "memory": 30.0,
         "host_overhead": 30.0, "comm": 30.0},
    ]
    assert s._format_roofline_saturation_advisory() == ""


def test_advisory_threshold_constant_is_80():
    """Locks the threshold so test_advisory_renders / docs stay in sync."""
    assert SATURATION_ADVISORY_THRESHOLD_PCT == 80.0


def test_to_prompt_summary_emits_advisory_when_toggle_on():
    s = SharedState()
    s.roofline_saturation_advisory = True
    s.roofline_saturation_history = [
        {"snapshot_id": 5, "compute": 95.0, "memory": 0.0,
         "host_overhead": 0.0, "comm": 0.0},
    ]
    summary = s.to_prompt_summary()
    assert "=== Roofline Saturation Advisory (snapshot #5)" in summary


def test_to_prompt_summary_omits_advisory_when_toggle_off():
    s = SharedState()
    s.roofline_saturation_advisory = False
    s.roofline_saturation_history = [
        {"snapshot_id": 5, "compute": 95.0},
    ]
    summary = s.to_prompt_summary()
    assert "Roofline Saturation Advisory" not in summary


# ---------------------------------------------------------------------------
# RooflineExecutor — saturation history append
# ---------------------------------------------------------------------------


def _make_ctx(session_dir: Path) -> RunnerContext:
    task = Task(
        task_id="t-sat",
        kind="roofline",
        state="running",
        params={},
        idempotency_key="ik-sat-1",
        requires_lanes=["profile_lane"],
        allowed_tools=["emit_intent"],
        side_effects=["reads_server", "writes_results"],
        lease_ttl_sec=2700,
    )
    return RunnerContext(
        task=task, lease=None, extra={"session_dir": str(session_dir)},
    )


@pytest.mark.asyncio
async def test_executor_appends_saturation_history_when_toggle_on(
    tmp_path: Path,
):
    md = tmp_path / "analysis.md"
    md.write_text(_FULL_EXEC_TABLE, encoding="utf-8")

    profile_result = {
        "status": "succeeded",
        "main_trace_path": str(tmp_path / "trace.json"),
        "workspace": str(tmp_path / "ws"),
    }
    ta_result = {
        "status": "ok",
        "trace_report_path": str(md),
        "candidates_path": str(tmp_path / "candidates.json"),
        "hot_kernels": [],
    }

    state = SharedState()
    state.roofline_saturation_advisory = True
    executor = make_roofline_executor(shared_state=state)

    async def _fake_profile(ctx):
        return profile_result

    async def _fake_ta(payload, *, session_dir):
        return ta_result

    with patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        side_effect=_fake_profile,
    ), patch(
        "inference_optimizer.orchestrator.kernel_request_handlers.trace_analyze_handler",
        side_effect=_fake_ta,
    ):
        result = await executor(_make_ctx(tmp_path))
    assert result["status"] == "succeeded"
    assert len(state.roofline_saturation_history) == 1
    record = state.roofline_saturation_history[0]
    assert record["snapshot_id"] == 1
    assert record["compute"] == 92.5
    assert record["memory"] == 35.0


@pytest.mark.asyncio
async def test_executor_history_omitted_when_toggle_off(tmp_path: Path):
    md = tmp_path / "analysis.md"
    md.write_text(_FULL_EXEC_TABLE, encoding="utf-8")

    profile_result = {"status": "succeeded",
                      "main_trace_path": str(tmp_path / "trace.json")}
    ta_result = {"status": "ok", "trace_report_path": str(md),
                 "candidates_path": "", "hot_kernels": []}

    state = SharedState()
    state.roofline_saturation_advisory = False
    executor = make_roofline_executor(shared_state=state)

    with patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        return_value=profile_result,
    ), patch(
        "inference_optimizer.orchestrator.kernel_request_handlers.trace_analyze_handler",
        return_value=ta_result,
    ):
        # use side_effect coroutines:
        pass

    async def _fake_profile(ctx):
        return profile_result

    async def _fake_ta(payload, *, session_dir):
        return ta_result

    with patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        side_effect=_fake_profile,
    ), patch(
        "inference_optimizer.orchestrator.kernel_request_handlers.trace_analyze_handler",
        side_effect=_fake_ta,
    ):
        await executor(_make_ctx(tmp_path))
    assert state.roofline_saturation_history == []


@pytest.mark.asyncio
async def test_executor_history_capped_at_10(tmp_path: Path):
    md = tmp_path / "analysis.md"
    md.write_text(_FULL_EXEC_TABLE, encoding="utf-8")

    state = SharedState()
    state.roofline_saturation_advisory = True
    state.roofline_saturation_history = [
        {"snapshot_id": i, "compute": 0.0, "memory": 0.0,
         "host_overhead": 0.0, "comm": 0.0}
        for i in range(10)
    ]

    profile_result = {"status": "succeeded",
                      "main_trace_path": str(tmp_path / "trace.json")}
    ta_result = {"status": "ok", "trace_report_path": str(md),
                 "candidates_path": "", "hot_kernels": []}
    executor = make_roofline_executor(shared_state=state)

    async def _fake_profile(ctx):
        return profile_result

    async def _fake_ta(payload, *, session_dir):
        return ta_result

    with patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        side_effect=_fake_profile,
    ), patch(
        "inference_optimizer.orchestrator.kernel_request_handlers.trace_analyze_handler",
        side_effect=_fake_ta,
    ):
        await executor(_make_ctx(tmp_path))
    # Capped at 10 — oldest record evicted.
    assert len(state.roofline_saturation_history) == 10
    # Newest record is the just-recorded one (compute 92.5 from full table).
    assert state.roofline_saturation_history[-1]["compute"] == 92.5
