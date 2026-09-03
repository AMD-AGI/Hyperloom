# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the post-optimization concurrency sweep runner."""

from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    VariantResult,
)
from hyperloom.orchestrator.kernel.conc_sweep import (
    DEFAULT_CONCS,
    DEFAULT_TOTAL_BUDGET_SEC,
    DEFAULT_VARIANT_TIMEOUT_SEC,
    _build_arm_grid,
    _flush_conc_sweep_report,
    _flush_partial_conc_sweep_report,
    _granted_cap_sec,
    _has_optimization,
    _order_concs_desc,
    _point_from_variant,
    conc_sweep_declined_to_run,
    run_conc_sweep,
)
from hyperloom.common.gain_math import conc_pair_comparison as _build_comparison
from hyperloom.common.perf_metric import graded_metric_key
from hyperloom.orchestrator.state.shared_state import SharedState


# Fixtures
def _make_state(
    *,
    baseline_tput: float = 100.0,
    isl: int = 1024,
    osl: int = 1024,
    tp: int = 8,
    current_best: dict[str, Any] | None = None,
    baseline_config_path: str = "",
) -> SharedState:
    s = SharedState()
    s.baseline_tput = baseline_tput
    s.isl = isl
    s.osl = osl
    s.tp = tp
    s.current_best = (
        current_best
        if current_best is not None
        else {
            "extra_server_args": "--enable-torch-compile",
            "extra_envs": {"SGLANG_FOO": "1"},
        }
    )
    s.baseline_config_path = baseline_config_path
    return s


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "Qwen-Test" / "20260101T000000Z"
    sd.mkdir(parents=True)
    (sd / "reports").mkdir()
    return sd


@pytest.fixture
def baseline_yaml(tmp_path: Path) -> Path:
    """Minimal YAML on disk so ``run_conc_sweep`` passes the missing-config gate."""
    p = tmp_path / "baseline.yaml"
    p.write_text("benchmark:\n  benchmark_script: bench.sh\n")
    return p


# _has_optimization
def test_has_optimization_args_only():
    s = SharedState()
    s.current_best = {"extra_server_args": "--a 1", "extra_envs": {}}
    has, args, envs = _has_optimization(s)
    assert has is True
    assert args == "--a 1"
    assert envs == {}


def test_has_optimization_envs_only():
    s = SharedState()
    s.current_best = {"extra_server_args": "", "extra_envs": {"X": "1"}}
    has, args, envs = _has_optimization(s)
    assert has is True
    assert args == ""
    assert envs == {"X": "1"}


def test_has_optimization_both_empty():
    s = SharedState()
    s.current_best = {"extra_server_args": "", "extra_envs": {}}
    has, _args, _envs = _has_optimization(s)
    assert has is False


def test_has_optimization_missing_current_best():
    s = SharedState()
    s.current_best = {}
    has, _, _ = _has_optimization(s)
    assert has is False


# _build_comparison
def test_build_comparison_simple_speedup():
    baseline = [
        {"conc": 1, "output_throughput": 100.0, "status": "succeeded"},
        {"conc": 4, "output_throughput": 200.0, "status": "succeeded"},
    ]
    optimized = [
        {"conc": 1, "output_throughput": 150.0, "status": "succeeded"},
        {"conc": 4, "output_throughput": 220.0, "status": "succeeded"},
    ]
    rows, summary = _build_comparison(baseline, optimized)
    assert len(rows) == 2
    assert rows[0]["speedup"] == pytest.approx(1.5)
    assert rows[1]["speedup"] == pytest.approx(1.1)
    assert summary["successful_pairs"] == 2
    assert summary["failed_pairs"] == 0
    assert summary["best_conc"] == 1
    assert summary["best_speedup"] == pytest.approx(1.5)
    assert summary["median_speedup"] == pytest.approx(1.3)


def test_build_comparison_partial_failures():
    """One arm fails on conc=4 → that pair is counted failed, others still pair up."""
    baseline = [
        {"conc": 1, "output_throughput": 100.0, "status": "succeeded"},
        {"conc": 4, "output_throughput": 200.0, "status": "succeeded"},
    ]
    optimized = [
        {"conc": 1, "output_throughput": 150.0, "status": "succeeded"},
        {"conc": 4, "output_throughput": None, "status": "failed"},
    ]
    rows, summary = _build_comparison(baseline, optimized)
    assert summary["successful_pairs"] == 1
    assert summary["failed_pairs"] == 1
    assert rows[1]["speedup"] is None
    assert rows[1]["optimized_status"] == "failed"


def test_build_comparison_all_failed_no_summary_numbers():
    baseline = [{"conc": 1, "output_throughput": None, "status": "failed"}]
    optimized = [{"conc": 1, "output_throughput": None, "status": "failed"}]
    rows, summary = _build_comparison(baseline, optimized)
    assert summary["successful_pairs"] == 0
    assert summary["best_speedup"] is None
    assert summary["median_speedup"] is None
    assert rows[0]["speedup"] is None


def test_build_comparison_mismatched_concs_outer_join():
    """Baseline ran 1/4, optimized ran 4/16 — outer join over CONC."""
    baseline = [
        {"conc": 1, "output_throughput": 100.0, "status": "succeeded"},
        {"conc": 4, "output_throughput": 200.0, "status": "succeeded"},
    ]
    optimized = [
        {"conc": 4, "output_throughput": 220.0, "status": "succeeded"},
        {"conc": 16, "output_throughput": 400.0, "status": "succeeded"},
    ]
    rows, summary = _build_comparison(baseline, optimized)
    assert [r["conc"] for r in rows] == [1, 4, 16]
    assert rows[0]["optimized_tput"] is None
    assert rows[2]["baseline_tput"] is None
    assert summary["successful_pairs"] == 1


# Skip paths
@pytest.mark.parametrize(
    "override, reason",
    [
        ({"baseline_tput": 0.0}, "no_baseline_tput"),
        ({"isl": 0}, "missing_workload_shape"),
        ({"osl": 0}, "missing_workload_shape"),
        (
            {"current_best": {"extra_server_args": "", "extra_envs": {}}},
            "no_optimization_to_compare",
        ),
    ],
)
def test_run_conc_sweep_skip_short_circuits(
    session_dir: Path,
    baseline_yaml: Path,
    override: dict[str, Any],
    reason: str,
):
    state = _make_state(baseline_config_path=str(baseline_yaml))
    for k, v in override.items():
        setattr(state, k, v)

    with patch(
        "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
    ) as mock_run_grid:
        payload = asyncio.run(run_conc_sweep(state, session_dir))

    assert payload["status"] == "skipped"
    assert payload["skip_reason"] == reason
    mock_run_grid.assert_not_called()
    assert not (session_dir / "reports" / "conc_sweep_summary.json").exists()


def test_run_conc_sweep_skip_empty_conc_list(
    session_dir: Path,
    baseline_yaml: Path,
):
    state = _make_state(baseline_config_path=str(baseline_yaml))
    with patch(
        "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
    ) as mock_run_grid:
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[]))
    assert payload["status"] == "skipped"
    assert payload["skip_reason"] == "no_optimization_to_compare" or payload["skip_reason"] == "empty_conc_list"
    # has_opt passes, so empty list is the reason.
    assert payload["skip_reason"] == "empty_conc_list"
    mock_run_grid.assert_not_called()


def test_run_conc_sweep_skip_missing_config(
    session_dir: Path,
    tmp_path: Path,
):
    state = _make_state(
        baseline_config_path=str(tmp_path / "does_not_exist.yaml"),
    )
    with patch(
        "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
    ) as mock_run_grid:
        payload = asyncio.run(run_conc_sweep(state, session_dir))
    assert payload["status"] == "skipped"
    assert payload["skip_reason"] == "baseline_config_missing"
    mock_run_grid.assert_not_called()


# Happy path with mocked run_grid
def _fake_variant(
    name: str,
    *,
    throughput: float | None,
    status: str = "succeeded",
    envs: dict[str, str] | None = None,
    error: str | None = None,
) -> VariantResult:
    return VariantResult(
        name=name,
        extra_server_args="",
        extra_envs=envs or {},
        status=status,
        output_throughput=throughput,
        request_throughput=throughput,
        total_token_throughput=throughput,
        completed_requests=None,
        duration_seconds=10.0 if status == "succeeded" else None,
        ttft_mean_ms=25.0 if status == "succeeded" else None,
        e2el_mean_ms=200.0 if status == "succeeded" else None,
        workspace=f"/tmp/{name}",
        report_path=None,
        raw_result_path=None,
        reported_success=status == "succeeded",
        returncode=0 if status == "succeeded" else 1,
        nonfatal_warnings=[],
        error=error,
        error_class="" if status == "succeeded" else "magpie_timeout",
        note="",
        runtime_sec=12.0 if status == "succeeded" else None,
        killed_overtime=False,
    )


def _fake_materialize(src, out_dir, **_kw):
    """Copy the base YAML into the run dir (mock for materialize_config_with_envs)."""
    out = Path(out_dir) / "conc_sweep_base.with_envs.yaml"
    out.write_text(Path(src).read_text())
    return out


def test_run_conc_sweep_happy_path_writes_reports(
    session_dir: Path,
    baseline_yaml: Path,
):
    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        out: list[VariantResult] = []
        for v in grid:
            conc = int(v.extra_envs["CONC"])
            if v.name.startswith("baseline_"):
                tput = 100.0 * conc / (1 + conc * 0.05)
            else:
                tput = 130.0 * conc / (1 + conc * 0.05)
            out.append(
                _fake_variant(
                    v.name,
                    throughput=tput,
                    envs=v.extra_envs,
                )
            )
        return out

    with (
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
            side_effect=_fake_run_grid,
        ),
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs",
            side_effect=_fake_materialize,
        ),
    ):
        payload = asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[1, 4, 16],
            )
        )

    assert payload["status"] == "succeeded"
    assert payload["isl"] == 1024
    assert payload["osl"] == 1024
    assert payload["tp"] == 8
    assert payload["concs_requested"] == [1, 4, 16]
    assert len(payload["baseline"]["points"]) == 3
    assert len(payload["optimized"]["points"]) == 3
    speedups = [r["speedup"] for r in payload["comparison"]]
    for s in speedups:
        assert s == pytest.approx(1.30)
    assert payload["summary"]["successful_pairs"] == 3
    assert payload["summary"]["best_speedup"] == pytest.approx(1.30)
    assert payload["summary"]["median_speedup"] == pytest.approx(1.30)

    summary_path = session_dir / "reports" / "conc_sweep_summary.json"
    csv_path = session_dir / "reports" / "conc_sweep_raw.csv"
    assert summary_path.exists()
    assert csv_path.exists()

    disk = json.loads(summary_path.read_text())
    assert disk["status"] == "succeeded"
    assert disk["summary"]["successful_pairs"] == 3
    # Self-referential paths must land in the on-disk JSON.
    assert disk["report_json_path"] == summary_path.as_posix()
    assert disk["report_csv_path"] == csv_path.as_posix()

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 6  # 3 baseline + 3 optimized
    assert {r["arm"] for r in rows} == {"baseline", "optimized"}

    # final.json is owned by report.py at CLOSE; conc_sweep must not touch it.
    final_json_path = session_dir / "reports" / "final.json"
    assert not final_json_path.exists()


def test_run_conc_sweep_canonicalizes_gpu_type_to_runner(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch,
):
    """On MI325X/MI308X conc-sweep must select the mi300x runner script, like
    every other executor — not state.gpu_type's real type."""
    state = _make_state(baseline_config_path=str(baseline_yaml))
    state.gpu_type = "mi325x"
    monkeypatch.setenv("GPU_TYPE", "mi300x")

    seen: dict[str, str] = {}

    def _fake_materialize(src, out_dir, **kw):
        seen["materialize_gpu"] = kw.get("gpu_type")
        out = Path(out_dir) / "conc_sweep_base.with_envs.yaml"
        out.write_text(Path(src).read_text())
        return out

    async def _fake_run_grid(*, grid: list[GridVariant], **kw):
        seen["run_grid_gpu"] = kw.get("gpu_type")
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
            side_effect=_fake_run_grid,
        ),
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs",
            side_effect=_fake_materialize,
        ),
    ):
        asyncio.run(run_conc_sweep(state, session_dir, concs=[1]))

    assert seen["materialize_gpu"] == "mi300x"
    assert seen["run_grid_gpu"] == "mi300x"


def test_run_conc_sweep_optimized_oom_yields_failed_pair(
    session_dir: Path,
    baseline_yaml: Path,
):
    """An optimized variant crashing yields a failed pair but doesn't abort the summary."""
    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        out = []
        for v in grid:
            conc = int(v.extra_envs["CONC"])
            if v.name.startswith("baseline_"):
                out.append(
                    _fake_variant(
                        v.name,
                        throughput=100.0 * conc,
                        envs=v.extra_envs,
                    )
                )
            else:
                if conc == 16:
                    out.append(
                        _fake_variant(
                            v.name,
                            throughput=None,
                            status="failed",
                            envs=v.extra_envs,
                            error="CUDA OOM",
                        )
                    )
                else:
                    out.append(
                        _fake_variant(
                            v.name,
                            throughput=120.0 * conc,
                            envs=v.extra_envs,
                        )
                    )
        return out

    with (
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
            side_effect=_fake_run_grid,
        ),
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs",
            side_effect=_fake_materialize,
        ),
    ):
        payload = asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[1, 4, 16],
            )
        )

    # Two successful pairs (conc 1, 4) + one failed pair (conc 16).
    assert payload["summary"]["successful_pairs"] == 2
    assert payload["summary"]["failed_pairs"] == 1
    fail_row = next(r for r in payload["comparison"] if r["conc"] == 16)
    assert fail_row["speedup"] is None
    assert fail_row["optimized_status"] == "failed"
    assert payload["status"] == "succeeded"


def test_run_conc_sweep_args_only_optimization_triggers_run(
    session_dir: Path,
    baseline_yaml: Path,
):
    """An optimized config with only ``extra_server_args`` still triggers the sweep."""
    state = _make_state(
        baseline_config_path=str(baseline_yaml),
        current_best={
            "extra_server_args": "--enable-torch-compile",
            "extra_envs": {},
        },
    )

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
            side_effect=_fake_run_grid,
        ) as mock_run,
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs",
            side_effect=_fake_materialize,
        ),
    ):
        payload = asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[1, 4],
            )
        )

    assert payload["status"] in ("succeeded", "failed")
    assert mock_run.call_count == 4  # 2 arms × 2 concs
    assert payload["optimized"]["extra_server_args"] == "--enable-torch-compile"
    assert payload["optimized"]["extra_envs"] == {}


def test_run_conc_sweep_envs_only_optimization_triggers_run(
    session_dir: Path,
    baseline_yaml: Path,
):
    """Symmetric: envs-only ``current_best`` also triggers the run."""
    state = _make_state(
        baseline_config_path=str(baseline_yaml),
        current_best={
            "extra_server_args": "",
            "extra_envs": {"SGLANG_MOE_ENABLE": "1"},
        },
    )

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
            side_effect=_fake_run_grid,
        ) as mock_run,
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs",
            side_effect=_fake_materialize,
        ),
    ):
        payload = asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[1],
            )
        )

    assert mock_run.call_count == 2  # 2 arms × 1 conc
    assert payload["optimized"]["extra_envs"] == {"SGLANG_MOE_ENABLE": "1"}


def test_run_conc_sweep_does_not_touch_final_json(
    session_dir: Path,
    baseline_yaml: Path,
):
    """conc_sweep runs before CLOSE and must never create or modify final.json."""
    state = _make_state(baseline_config_path=str(baseline_yaml))
    final_json_path = session_dir / "reports" / "final.json"
    assert not final_json_path.exists()

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
            side_effect=_fake_run_grid,
        ),
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs",
            side_effect=_fake_materialize,
        ),
    ):
        payload = asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[1],
            )
        )

    assert payload["status"] in ("succeeded", "failed")
    assert (session_dir / "reports" / "conc_sweep_summary.json").exists()
    assert not final_json_path.exists()


class TestTheAgentXLadderIsDefaultOn:
    """The sweep is what an agentic session produces; it used to default off.

    It was disabled under AgentX because sixteen synthetic rungs would spend the
    whole session without tuning a server parameter. The ladder is now the
    deliverable rather than a postscript to one, and it is seven rungs, not
    sixteen.
    """

    def test_the_state_default_is_on(self):
        assert SharedState().conc_sweep_enabled is True

    def test_agentx_does_not_turn_it_off(self, monkeypatch: pytest.MonkeyPatch):
        from argparse import Namespace

        from hyperloom.inference_optimizer.cli import bootstrap as cb

        monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
        args = Namespace(enable_conc_sweep=True)
        assert bool(getattr(args, "enable_conc_sweep", True)) is True
        assert not hasattr(cb, "_flag_explicitly_set")


def test_the_agentx_budget_funds_the_whole_ladder():
    """Seven rungs on each arm; a smaller default would truncate every run."""
    from hyperloom.inference_optimizer.cli import (
        _AGENTX_CONC_SWEEP_TIMEOUT_SEC,
        _AGENTX_CONC_SWEEP_TOTAL_BUDGET_SEC,
    )
    from hyperloom.orchestrator.kernel.conc_sweep import AGENTX_DEFAULT_CONCS

    rungs = len(AGENTX_DEFAULT_CONCS) * 2
    assert rungs == 14
    measured_round_sec = 111 * 60
    assert _AGENTX_CONC_SWEEP_TOTAL_BUDGET_SEC >= rungs * measured_round_sec
    assert _AGENTX_CONC_SWEEP_TIMEOUT_SEC < _AGENTX_CONC_SWEEP_TOTAL_BUDGET_SEC


def test_the_engine_resolves_the_ladder_from_the_session_mode(monkeypatch: pytest.MonkeyPatch):
    """`concs=None` reaches the engine from the SDK and from a bare task alike."""
    from hyperloom.orchestrator.kernel.conc_sweep import default_concs_for_mode

    state = SharedState()
    state.benchmark_mode = "agentx"
    assert default_concs_for_mode(state.benchmark_mode) == [1, 4, 8, 10, 14, 20, 28]


def test_default_concs_is_powers_of_two():
    """Doc-pin: default ladder is [256,128,64,32,16,8,4,2] (high-to-low for single-server reuse)."""
    assert DEFAULT_CONCS == [256, 128, 64, 32, 16, 8, 4, 2]


def test_default_total_budget_is_two_and_half_hours():
    """Doc-pin: default total wall-clock budget is 2.5h."""
    assert DEFAULT_TOTAL_BUDGET_SEC == 9000


# Total wall-clock budget
def test_run_conc_sweep_budget_exhausted_marks_remaining_skipped(
    session_dir: Path,
    baseline_yaml: Path,
):
    """When remaining budget cannot cover another variant, the tail is skipped."""
    state = _make_state(baseline_config_path=str(baseline_yaml))
    calls = {"n": 0}

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        import time as _t

        _t.sleep(1.2)
        calls["n"] += 1
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
            side_effect=_fake_run_grid,
        ),
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs",
            side_effect=_fake_materialize,
        ),
    ):
        payload = asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[1, 4, 16, 64],
                variant_timeout_sec=1,
                total_budget_sec=2,
            )
        )

    all_points = payload["baseline"]["points"] + payload["optimized"]["points"]
    statuses = [p["status"] for p in all_points]
    assert "succeeded" in statuses
    assert "skipped" in statuses
    skipped_pts = [p for p in all_points if p["status"] == "skipped"]
    for p in skipped_pts:
        assert p["error_class"] == "budget_exhausted"
    assert payload["budget_exhausted"] is True
    assert payload["budget_skip_reason"] == "insufficient_remaining_for_variant"
    assert payload["budget_remaining_sec"] < 1
    assert payload["total_budget_sec"] == 2
    assert calls["n"] < 8


def test_run_conc_sweep_none_budget_disables_gate(
    session_dir: Path,
    baseline_yaml: Path,
):
    """``total_budget_sec=None`` disables the gate; every variant is launched."""
    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
            side_effect=_fake_run_grid,
        ) as mock_run,
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs",
            side_effect=_fake_materialize,
        ),
    ):
        payload = asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[1, 4],
                total_budget_sec=None,
            )
        )

    assert payload["budget_exhausted"] is False
    assert payload["total_budget_sec"] is None
    assert mock_run.call_count == 4  # 2 arms × 2 concs


def test_run_conc_sweep_zero_budget_skips_without_running(
    session_dir: Path,
    baseline_yaml: Path,
):
    """``total_budget_sec=0`` is "no time left": skip, never run the ladder."""
    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
            side_effect=_fake_run_grid,
        ) as mock_run,
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs",
            side_effect=_fake_materialize,
        ),
    ):
        payload = asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[1, 4],
                total_budget_sec=0,
            )
        )

    assert payload["status"] == "skipped"
    assert payload["skip_reason"] == "no_time_budget_remaining"
    assert payload["total_budget_sec"] == 0
    assert mock_run.call_count == 0
    # Nothing ran, so this reads as a sweep that declined rather than one that
    # spent its budget.
    assert conc_sweep_declined_to_run({**payload, "was_skipped": True}) is True


def test_run_conc_sweep_skips_when_initial_budget_below_variant_timeout(
    session_dir: Path,
    baseline_yaml: Path,
):
    """A too-small budget is reported as skipped instead of a timeout-prone run."""
    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.run_grid",
            side_effect=_fake_run_grid,
        ),
        patch(
            "hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs",
            side_effect=_fake_materialize,
        ),
    ):
        payload = asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[1],
                variant_timeout_sec=3600,
                total_budget_sec=120,
            )
        )

    all_points = payload["baseline"]["points"] + payload["optimized"]["points"]
    assert {p["status"] for p in all_points} == {"skipped"}
    assert {p["error_class"] for p in all_points} == {"budget_exhausted"}
    assert payload["status"] == "skipped"
    assert payload["was_skipped"] is True
    assert payload["skip_reason"] == "budget_exhausted_no_successful_pairs"
    assert payload["budget_exhausted"] is True
    assert payload["budget_skip_reason"] == "insufficient_remaining_for_variant"
    # This sweep started; only the pre-flight envelope means "declined to run".
    assert conc_sweep_declined_to_run(payload) is False


# ActionExecutor integration (SWEEP-phase dispatch)
def test_conc_sweep_executor_loads_state_and_dispatches(
    session_dir: Path,
    baseline_yaml: Path,
):
    """ConcSweepExecutor reloads SharedState and threads registry fields through."""
    from hyperloom.orchestrator.actions.executors.conc_sweep import (
        ConcSweepExecutor,
    )

    state = _make_state(baseline_config_path=str(baseline_yaml))
    state.conc_sweep_enabled = True
    state.conc_sweep_concs = [1, 4]
    state.conc_sweep_total_budget_sec = 60
    state.conc_sweep_variant_timeout_sec = 30
    state.save(session_dir)

    class _Task:
        params = {}  # executor should fall back to SharedState values

    class _Ctx:
        task = _Task()
        extra = {"session_dir": str(session_dir)}

    captured: dict = {}

    async def _fake_run(state_arg, sd, *, concs, variant_timeout_sec, total_budget_sec, **_kw):
        captured["concs"] = list(concs)
        captured["timeout"] = variant_timeout_sec
        captured["budget"] = total_budget_sec
        return {
            "status": "succeeded",
            "summary": {"successful_pairs": 2},
        }

    with patch(
        "hyperloom.orchestrator.actions.executors.conc_sweep.run_conc_sweep",
        side_effect=_fake_run,
    ):
        result = asyncio.run(ConcSweepExecutor()(_Ctx()))

    assert result["status"] == "succeeded"
    assert captured["concs"] == [1, 4]
    assert captured["timeout"] == 30
    assert captured["budget"] == 60


def test_conc_sweep_executor_missing_session_dir_yields_failure():
    from hyperloom.orchestrator.actions.executors.conc_sweep import (
        ConcSweepExecutor,
    )

    class _Task:
        params = {}

    class _Ctx:
        task = _Task()
        extra: dict = {}

    result = asyncio.run(ConcSweepExecutor()(_Ctx()))
    assert result["status"] == "failed"
    assert result["error_class"] == "missing_session_dir"


def test_conc_sweep_executor_state_load_failure_yields_failure(monkeypatch):
    from hyperloom.orchestrator.actions.executors.conc_sweep import (
        ConcSweepExecutor,
    )

    class _Task:
        params = {}

    class _Ctx:
        task = _Task()
        extra = {"session_dir": "/tmp/does-not-matter"}

    def _boom(_session_dir):
        raise RuntimeError("state is unreadable")

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.conc_sweep.SharedState.load_or_init",
        _boom,
    )

    result = asyncio.run(ConcSweepExecutor()(_Ctx()))
    assert result["status"] == "failed"
    assert result["error_class"] == "shared_state_load_failed"
    assert "state is unreadable" in result["error"]


def test_conc_sweep_executor_task_params_override_state(
    session_dir: Path,
    baseline_yaml: Path,
):
    from hyperloom.orchestrator.actions.executors.conc_sweep import (
        ConcSweepExecutor,
    )

    state = _make_state(baseline_config_path=str(baseline_yaml))
    state.conc_sweep_concs = [1]
    state.conc_sweep_total_budget_sec = 60
    state.conc_sweep_variant_timeout_sec = 30
    state.save(session_dir)

    class _Task:
        params = {
            "concs": ["2", "8"],
            "variant_timeout_sec": "45",
            "total_budget_sec": "120",
        }

    class _Ctx:
        task = _Task()
        extra = {"session_dir": str(session_dir)}

    captured: dict = {}

    async def _fake_run(state_arg, sd, *, concs, variant_timeout_sec, total_budget_sec, **_kw):
        captured["concs"] = list(concs)
        captured["timeout"] = variant_timeout_sec
        captured["budget"] = total_budget_sec
        return {"status": "succeeded", "summary": {"successful_pairs": 1}}

    with patch(
        "hyperloom.orchestrator.actions.executors.conc_sweep.run_conc_sweep",
        side_effect=_fake_run,
    ):
        result = asyncio.run(ConcSweepExecutor()(_Ctx()))

    assert result["status"] == "succeeded"
    assert captured == {"concs": [2, 8], "timeout": 45, "budget": 120}


def test_conc_sweep_executor_keeps_none_budget_unbounded(
    session_dir: Path,
    baseline_yaml: Path,
):
    """An explicit ``None`` budget stays None: coercing it to 0 would skip the sweep."""
    from hyperloom.orchestrator.actions.executors.conc_sweep import (
        ConcSweepExecutor,
    )

    state = _make_state(baseline_config_path=str(baseline_yaml))
    state.conc_sweep_total_budget_sec = 9000
    state.save(session_dir)

    class _Task:
        params = {"total_budget_sec": None}

    class _Ctx:
        task = _Task()
        extra = {"session_dir": str(session_dir)}

    captured: dict = {}

    async def _fake_run(state_arg, sd, *, concs, variant_timeout_sec, total_budget_sec, **_kw):
        captured["budget"] = total_budget_sec
        return {"status": "succeeded", "summary": {"successful_pairs": 1}}

    with patch(
        "hyperloom.orchestrator.actions.executors.conc_sweep.run_conc_sweep",
        side_effect=_fake_run,
    ):
        asyncio.run(ConcSweepExecutor()(_Ctx()))

    assert captured["budget"] is None


def test_conc_sweep_executor_remaps_skip_to_succeeded(
    session_dir: Path,
    baseline_yaml: Path,
):
    """A run_conc_sweep skip surfaces ``was_skipped=True`` + ``status='succeeded'``."""
    from hyperloom.orchestrator.actions.executors.conc_sweep import (
        ConcSweepExecutor,
    )

    state = _make_state(baseline_config_path=str(baseline_yaml))
    state.save(session_dir)

    class _Task:
        params = {}

    class _Ctx:
        task = _Task()
        extra = {"session_dir": str(session_dir)}

    async def _fake_run(*_a, **_kw):
        return {"status": "skipped", "skip_reason": "no_baseline_tput"}

    with patch(
        "hyperloom.orchestrator.actions.executors.conc_sweep.run_conc_sweep",
        side_effect=_fake_run,
    ):
        result = asyncio.run(ConcSweepExecutor()(_Ctx()))
    assert result["status"] == "succeeded"
    assert result["was_skipped"] is True
    assert result["skip_reason"] == "no_baseline_tput"
    assert conc_sweep_declined_to_run(result) is True


# record_conc_sweep writes state.last_conc_sweep for SWEEP completion detection.
def test_record_conc_sweep_writes_last_conc_sweep():
    s = SharedState()
    assert s.last_conc_sweep == {}
    assert s.last_conc_sweep_watermark == {}
    s.cumulative_gain_validated = 3.25
    s.record_conc_sweep(
        {
            "status": "succeeded",
            "skip_reason": "",
            "was_skipped": False,
            "budget_exhausted": False,
            "summary": {"successful_pairs": 8, "best_speedup": 1.05},
            "workspace": "/tmp/conc_sweep_xyz",
        }
    )
    assert s.last_conc_sweep.get("status") == "succeeded"
    assert s.last_conc_sweep.get("summary", {}).get("successful_pairs") == 8
    assert s.last_conc_sweep.get("ts")
    assert s.last_conc_sweep_watermark.get("status") == "succeeded"
    assert s.last_conc_sweep_watermark.get("cumulative_gain_validated_at_record") == 3.25
    watermark = dict(s.last_conc_sweep_watermark)
    # Skip cases also recorded so SWEEP exits cleanly even on skip.
    s.record_conc_sweep(
        {
            "status": "skipped",
            "skip_reason": "no_optimization_to_compare",
            "was_skipped": True,
        }
    )
    assert s.last_conc_sweep.get("status") == "skipped"
    assert s.last_conc_sweep.get("skip_reason") == "no_optimization_to_compare"
    assert s.last_conc_sweep.get("was_skipped") is True
    assert s.last_conc_sweep_watermark == watermark


def test_exit_normal_sweep_reads_the_ladder_as_the_sweep():
    """The concurrency ladder is the only sweep, so its status is the phase's."""
    from hyperloom.orchestrator.phases.machine_state import exit_normal_sweep

    class _State:
        last_conc_sweep = {}
        phase = "SWEEP"
        phase_started_ts = "2026-06-02T10:00:00+00:00"
        max_minutes = 360
        phase_budget_pct = {"SWEEP": 0.50}

    # Nothing recorded => don't exit (budget remaining).
    assert exit_normal_sweep(_State()) is None

    _State.last_conc_sweep = {"status": "succeeded"}
    result = exit_normal_sweep(_State())
    assert result is not None
    reason, evidence = result
    assert reason == "sweep_done", reason
    assert evidence.get("sweep_status") == "succeeded"

    # Skipped also counts as "done" (the action reached a terminal decision).
    for terminal in ("partial", "completed", "skipped"):
        _State.last_conc_sweep = {"status": terminal}
        result = exit_normal_sweep(_State())
        assert result is not None and result[0] == "sweep_done", terminal

    _State.last_conc_sweep = {"status": "failed"}
    result = exit_normal_sweep(_State())
    assert result is not None
    reason, evidence = result
    assert reason == "sweep_failed"
    assert evidence.get("sweep_status") == "failed"


def test_the_sweep_exit_evidence_separates_a_skip_from_a_spent_budget():
    """``was_skipped`` covers both outcomes, so the row must carry what tells them apart."""
    from hyperloom.orchestrator.phases.machine_state import exit_normal_sweep

    state = SharedState(
        phase="SWEEP",
        phase_started_ts="2026-06-02T10:00:00+00:00",
        max_minutes=360,
        phase_budget_pct={"SWEEP": 0.50},
    )
    state.record_conc_sweep({"status": "skipped", "was_skipped": True, "skip_reason": "no_optimization_to_compare"})
    _, declined = exit_normal_sweep(state)
    assert declined["sweep_was_skipped"] is True
    assert declined["sweep_skip_budget_exhausted"] is False

    state.record_conc_sweep(
        {
            "status": "skipped",
            "was_skipped": True,
            "budget_exhausted": True,
            "skip_reason": "budget_exhausted_no_successful_pairs",
        }
    )
    _, spent = exit_normal_sweep(state)
    assert spent["sweep_was_skipped"] is True
    assert spent["sweep_skip_budget_exhausted"] is True


def test_on_enter_sweep_drains_pending_keep_integrates(monkeypatch):
    """Bug #7: KERNEL→SWEEP must drain pending KEEP integrates before closeout."""
    from unittest.mock import AsyncMock, MagicMock
    from hyperloom.orchestrator.kernel import request_handlers as kernel_request_handlers

    fake_integrate = AsyncMock(
        return_value={
            "status": "ok",
            "decision": "KEEP",
            "kernel_id": "k001",
            "gain_pct": 1.5,
        },
    )
    monkeypatch.setattr(
        kernel_request_handlers,
        "integrate_handler",
        fake_integrate,
    )
    from hyperloom.orchestrator.loop import coordinator as coord_mod

    if hasattr(coord_mod, "integrate_handler"):
        monkeypatch.setattr(coord_mod, "integrate_handler", fake_integrate)

    coord = MagicMock()
    coord.shared_state = MagicMock()
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.rejected_kernel_ids = []
    coord.shared_state.save = MagicMock()
    pending_queue = [
        {"kernel_id": "k001", "integration_id": "integration-1"},
        {"kernel_id": "k002", "integration_id": "integration-2"},
    ]
    coord.shared_state.pending_kernel_integration_records = lambda: [pending_queue.pop(0)] if pending_queue else []
    coord._record_integrate_keep = AsyncMock()
    coord.session_dir = Path("/tmp/sess")

    from hyperloom.orchestrator.loop.coordinator import Coordinator

    asyncio.run(Coordinator._drain_pending_keep_integrates(coord))

    assert fake_integrate.await_count == 2, fake_integrate.await_args_list
    assert coord.shared_state.save.call_count >= 2


# NOTE: the former ``test_conc_sweep_phase_singleton_denies_after_auto_enqueue``
# was retired together with PolicyGate._validate_conc_sweep_singleton. conc_sweep
# is now a Coordinator-internal action (COORDINATOR_INTERNAL_ACTIONS), so an LLM
# conc_sweep proposal is rejected as Coordinator-managed (phase_incompatible),
# not via a per-action singleton rule. That behaviour is covered by
# test_sweep_phase_auto.py::test_validate_intent_denies_llm_conc_sweep_propose_as_coordinator_managed
# and ::test_conc_sweep_is_coordinator_internal_action.


# ─────────────────────────────────────────────────────────────────────────────
# Change 2 extras: _order_concs_desc / _build_arm_grid
# ─────────────────────────────────────────────────────────────────────────────


def test_order_concs_desc_deduplicates_and_sorts():
    assert _order_concs_desc([4, 1, 16, 4, 2]) == [16, 4, 2, 1]


def test_order_concs_desc_already_sorted():
    assert _order_concs_desc([8, 4, 2]) == [8, 4, 2]


def test_order_concs_desc_single():
    assert _order_concs_desc([7]) == [7]


def test_build_arm_grid_single_arm_descending():
    grid = _build_arm_grid(
        "baseline",
        [64, 32, 16],
        isl=512,
        osl=512,
        num_prompts_factor=5,
        arm_args="",
        arm_envs={},
    )
    assert [v.name for v in grid] == ["baseline_conc64", "baseline_conc32", "baseline_conc16"]
    assert grid[0].extra_envs["CONC"] == "64"
    assert grid[0].extra_envs["RUN_EVAL"] == "false"


def test_build_arm_grid_optimized_arm_carries_args():
    grid = _build_arm_grid(
        "optimized",
        [4],
        isl=1024,
        osl=512,
        num_prompts_factor=3,
        arm_args="--my-flag",
        arm_envs={"MY_ENV": "1"},
    )
    assert len(grid) == 1
    assert grid[0].extra_server_args == "--my-flag"
    assert grid[0].extra_envs["MY_ENV"] == "1"
    assert grid[0].extra_envs["CONC"] == "4"
    assert int(grid[0].extra_envs["NUM_PROMPTS"]) >= 4


# ─────────────────────────────────────────────────────────────────────────────
# Change 1: soft switch + arm-major orchestration
# ─────────────────────────────────────────────────────────────────────────────


def test_run_conc_sweep_single_server_arm_major_order(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """In single-server mode optimized arm runs before baseline."""
    state = _make_state(baseline_config_path=str(baseline_yaml))
    call_log: list[str] = []

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        for v in grid:
            call_log.append(v.name)
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[4, 16]))

    assert payload["status"] == "succeeded"
    # optimized_ variants should appear before baseline_ variants.
    first_opt = next((i for i, n in enumerate(call_log) if n.startswith("optimized_")), None)
    first_base = next((i for i, n in enumerate(call_log) if n.startswith("baseline_")), None)
    assert first_opt is not None and first_base is not None
    assert first_opt < first_base, f"expected optimized before baseline; got {call_log}"


def test_run_conc_sweep_single_server_concs_descending(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """CONCs within each arm are visited highest-to-lowest."""
    state = _make_state(baseline_config_path=str(baseline_yaml))
    call_log: list[str] = []

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        for v in grid:
            call_log.append(v.name)
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        asyncio.run(run_conc_sweep(state, session_dir, concs=[1, 4, 16]))

    opt_calls = [n for n in call_log if n.startswith("optimized_")]
    opt_concs = [int(n.split("conc")[1]) for n in opt_calls]
    assert opt_concs == sorted(opt_concs, reverse=True), f"expected descending, got {opt_concs}"

    base_calls = [n for n in call_log if n.startswith("baseline_")]
    base_concs = [int(n.split("conc")[1]) for n in base_calls]
    assert base_concs == sorted(base_concs, reverse=True), f"expected descending, got {base_concs}"


# ─────────────────────────────────────────────────────────────────────────────
# Change 5: _flush_conc_sweep_report / _flush_partial_conc_sweep_report
# ─────────────────────────────────────────────────────────────────────────────


def test_flush_conc_sweep_report_writes_json_and_csv(session_dir: Path):
    rdir = session_dir / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    json_path = rdir / "conc_sweep_summary.json"
    csv_path = rdir / "conc_sweep_raw.csv"
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "succeeded",
        "report_json_path": str(json_path),
        "report_csv_path": str(csv_path),
        "baseline": {"points": [{"arm": "baseline", "conc": 4, "status": "succeeded", "output_throughput": 100.0}]},
        "optimized": {"points": []},
    }
    _flush_conc_sweep_report(payload, session_dir)
    assert json_path.exists()
    assert csv_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["status"] == "succeeded"
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert rows[0]["arm"] == "baseline"


class TestTheSummaryIsTakenOnTheChartsAxis:
    """The headline speedup and the curve beside it have to be one quantity.

    On the agentic corpus output throughput is about 1% of the token budget, so
    a summary left on that axis reports a number the chart contradicts.
    """

    def _pts(self, arm: str, out: float, total: float) -> list[dict[str, Any]]:
        return [
            {"arm": arm, "conc": 8, "status": "succeeded", "output_throughput": out, "total_token_throughput": total}
        ]

    def test_agentx_grades_on_total_token_throughput(self):
        comparison, summary = _build_comparison(
            self._pts("baseline", 183.0, 20000.0),
            self._pts("optimized", 183.0, 26000.0),
            metric_key=graded_metric_key(benchmark_mode="agentx"),
        )
        assert summary["metric"] == "total_token_throughput"
        assert summary["best_speedup"] == pytest.approx(26000.0 / 20000.0)
        assert comparison[0]["baseline_tput"] == pytest.approx(20000.0)

    def test_synthetic_stays_on_output_throughput(self):
        _comparison, summary = _build_comparison(
            self._pts("baseline", 100.0, 20000.0),
            self._pts("optimized", 130.0, 26000.0),
            metric_key=graded_metric_key(benchmark_mode=""),
        )
        assert summary["metric"] == "output_throughput"
        assert summary["best_speedup"] == pytest.approx(1.3)

    def test_the_key_follows_the_mode(self):
        assert graded_metric_key(benchmark_mode="agentx") == "total_token_throughput"
        assert graded_metric_key(benchmark_mode="AgentX") == "total_token_throughput"
        assert graded_metric_key(benchmark_mode="synthetic") == "output_throughput"
        assert graded_metric_key(benchmark_mode="") == "output_throughput"

    def test_an_explicit_grading_override_wins_over_the_mode(self, monkeypatch):
        """The summary follows the axis the KEEP verdicts were taken on."""
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
        assert graded_metric_key(benchmark_mode="agentx") == "output_throughput"
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
        assert graded_metric_key(benchmark_mode="synthetic") == "total_token_throughput"

    def test_the_ambient_agentx_signal_reaches_the_summary(self, monkeypatch):
        """A session whose mode never persisted still grades on the total axis."""
        monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
        monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
        assert graded_metric_key(benchmark_mode="") == "total_token_throughput"


class TestAnUnreportedTotalComesFromItsHalves:
    """A row graded on the total axis must not read as unmeasured."""

    def _variant(self, **kw: Any) -> VariantResult:
        return VariantResult(
            name="baseline_c8", extra_server_args="", extra_envs={"CONC": "8"}, status="succeeded", **kw
        )

    def test_the_halves_sum_when_the_parser_named_no_total(self):
        point = _point_from_variant(
            self._variant(input_throughput=24000.0, output_throughput=180.0),
            arm="baseline",
        )
        assert point["total_token_throughput"] == pytest.approx(24180.0)

    def test_a_reported_total_is_not_recomputed(self):
        point = _point_from_variant(
            self._variant(input_throughput=1.0, output_throughput=1.0, total_token_throughput=25984.8),
            arm="baseline",
        )
        assert point["total_token_throughput"] == pytest.approx(25984.8)

    def test_a_missing_half_leaves_the_total_unmeasured(self):
        point = _point_from_variant(self._variant(output_throughput=180.0), arm="baseline")
        assert point["total_token_throughput"] is None


# --- the chart's data contract ---


class TestTheCurveCarriesBothAxisPairs:
    """A point has to carry whichever pair its mode is plotted on.

    Synthetic is plotted on output throughput against ``output_throughput/conc``;
    an agentic run on token throughput per chip against p90 interactivity. Both
    pairs live on the same record because the mode is a property of the session,
    not of the point.
    """

    def _variant(self, **kw: Any) -> VariantResult:
        base: dict[str, Any] = {
            "name": "optimized_conc8",
            "extra_server_args": "",
            "extra_envs": {"CONC": "8"},
            "status": "succeeded",
        }
        base.update(kw)
        return VariantResult(**base)

    def test_the_agentic_pair_reaches_the_point(self):
        point = _point_from_variant(
            self._variant(
                output_throughput=183.44,
                total_token_throughput=25984.8,
                input_throughput=25801.36,
                intvty_p90=447.2,
                tpot_p90_ms=2.4,
            ),
            arm="optimized",
        )
        assert point["total_token_throughput"] == pytest.approx(25984.8)
        assert point["intvty_p90"] == pytest.approx(447.2)
        assert point["input_throughput"] == pytest.approx(25801.36)
        assert point["tpot_p90_ms"] == pytest.approx(2.4)

    def test_the_synthetic_pair_is_unaffected(self):
        point = _point_from_variant(
            self._variant(output_throughput=1200.0, e2el_mean_ms=850.0),
            arm="baseline",
        )
        assert point["output_throughput"] == pytest.approx(1200.0)
        assert point["e2el_mean_ms"] == pytest.approx(850.0)
        assert point["intvty_p90"] is None

    def test_the_csv_carries_the_agentic_axes_too(self, session_dir: Path):
        """The CSV is the download button; it has to draw the same chart."""
        rdir = session_dir / "reports"
        rdir.mkdir(parents=True, exist_ok=True)
        json_path = rdir / "conc_sweep_summary.json"
        csv_path = rdir / "conc_sweep_raw.csv"
        point = _point_from_variant(
            self._variant(total_token_throughput=25984.8, intvty_p90=447.2),
            arm="optimized",
        )
        _flush_conc_sweep_report(
            {
                "schema_version": "1.0",
                "status": "succeeded",
                "report_json_path": str(json_path),
                "report_csv_path": str(csv_path),
                "baseline": {"points": []},
                "optimized": {"points": [point]},
            },
            session_dir,
        )
        row = next(iter(csv.DictReader(csv_path.open())))
        assert row["intvty_p90"] == "447.2"
        assert row["total_token_throughput"] == "25984.8"


def test_flush_conc_sweep_report_is_atomic(session_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """_flush_conc_sweep_report silently catches IO errors."""
    rdir = session_dir / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    json_path = rdir / "conc_sweep_summary.json"
    csv_path = rdir / "conc_sweep_raw.csv"
    payload = {
        "status": "succeeded",
        "report_json_path": str(json_path),
        "report_csv_path": str(csv_path),
        "baseline": {"points": []},
        "optimized": {"points": []},
    }
    # Should not raise even if atomic_write_text itself raises.
    from hyperloom.common import io as _common_io

    monkeypatch.setattr(_common_io, "atomic_write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    _flush_conc_sweep_report(payload, session_dir)  # must not raise


def test_flush_partial_conc_sweep_report_marks_in_progress(session_dir: Path):
    """_flush_partial_conc_sweep_report writes status=in_progress."""
    rdir = session_dir / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    json_path = rdir / "conc_sweep_summary.json"
    csv_path = rdir / "conc_sweep_raw.csv"
    state = _make_state()
    result = _fake_variant(
        "baseline_conc4", throughput=100.0, envs={"CONC": "4", "ISL": "512", "OSL": "512", "NUM_PROMPTS": "20"}
    )
    _flush_partial_conc_sweep_report(
        results=[result],
        state=state,
        session_dir=session_dir,
        json_path=json_path,
        csv_path=csv_path,
        concs=[4, 8],
        isl=512,
        osl=512,
        opt_args="--x",
        opt_envs={},
        workspace=session_dir / "ws",
        started_at=0.0,
        total_budget_sec=9000,
        has_budget=True,
        budget_exhausted=False,
        budget_skip_reason="",
        budget_remaining_sec=None,
        partial=True,
    )
    assert json_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["status"] == "in_progress"


# ─────────────────────────────────────────────────────────────────────────────
# Change 3: plotting
# ─────────────────────────────────────────────────────────────────────────────


def test_render_conc_sweep_curve_no_data_returns_none(tmp_path: Path):
    from hyperloom.orchestrator.kernel.conc_sweep_plot import render_conc_sweep_curve

    payload = {
        "baseline": {"points": []},
        "optimized": {"points": []},
    }
    result = render_conc_sweep_curve(payload, tmp_path / "out.png", model_label="M", gpu_label="GPU", tp=1)
    assert result is None


def test_render_conc_sweep_curve_writes_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from hyperloom.orchestrator.kernel.conc_sweep_plot import render_conc_sweep_curve

    payload = {
        "baseline": {
            "points": [
                {"conc": 16, "output_throughput": 1600.0},
                {"conc": 4, "output_throughput": 600.0},
            ]
        },
        "optimized": {
            "points": [
                {"conc": 16, "output_throughput": 2000.0},
                {"conc": 4, "output_throughput": 750.0},
            ]
        },
        "roofline_ceiling": {
            "rows": [
                {"conc": 4, "t_peak_tok_s": 800.0},
                {"conc": 16, "t_peak_tok_s": 2500.0},
            ]
        },
    }
    out_path = tmp_path / "curve.png"
    result = render_conc_sweep_curve(payload, out_path, model_label="TestModel", gpu_label="MI300X", tp=8)
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_conc_sweep_curve_from_file(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from hyperloom.orchestrator.kernel.conc_sweep_plot import render_conc_sweep_curve

    payload = {
        "baseline": {"points": [{"conc": 8, "output_throughput": 800.0}]},
        "optimized": {"points": []},
    }
    json_file = tmp_path / "summary.json"
    json_file.write_text(json.dumps(payload))
    result = render_conc_sweep_curve(json_file, tmp_path / "out.png", tp=1)
    assert result is not None
    assert result.exists()


class TestTheChartFollowsTheMode:
    """Two workloads, two rankings, two axis pairs.

    Plotting an agentic ladder on ``output_throughput / conc`` would label a
    number that is ~1% of the token budget as the thing being optimised.
    """

    def _payload(self, mode: str) -> dict[str, Any]:
        return {
            "benchmark_mode": mode,
            "baseline": {
                "points": [
                    {
                        "conc": 8,
                        "output_throughput": 183.44,
                        "total_token_throughput": 25984.8,
                        "intvty_p90": 447.2,
                    }
                ]
            },
            "optimized": {"points": []},
        }

    def test_agentx_reads_interactivity_and_total(self):
        from hyperloom.orchestrator.kernel import conc_sweep_plot as plot

        axes = plot._resolve_axes("agentx", tp_eff=8.0)
        xs, ys = plot._arm_series(self._payload("agentx")["baseline"]["points"], 8.0, axes)
        assert xs == [pytest.approx(447.2)]
        assert ys == [pytest.approx(25984.8 / 8.0)]
        assert "P90 Interactivity" in axes.x_label
        assert "per Chip" in axes.y_label

    def test_synthetic_keeps_the_output_pair(self):
        from hyperloom.orchestrator.kernel import conc_sweep_plot as plot

        axes = plot._resolve_axes("synthetic", tp_eff=8.0)
        xs, ys = plot._arm_series(self._payload("synthetic")["baseline"]["points"], 8.0, axes)
        assert xs == [pytest.approx(183.44 / 8)]
        assert ys == [pytest.approx(183.44 / 8.0)]

    def test_an_unset_mode_reads_as_synthetic(self):
        from hyperloom.orchestrator.kernel import conc_sweep_plot as plot

        assert plot._resolve_axes("", tp_eff=1.0).agentic is False
        assert plot._resolve_axes(None, tp_eff=1.0).agentic is False

    def test_a_rung_missing_its_axis_is_dropped_not_zeroed(self):
        from hyperloom.orchestrator.kernel import conc_sweep_plot as plot

        axes = plot._resolve_axes("agentx", tp_eff=1.0)
        points = [
            {"conc": 8, "total_token_throughput": 25984.8},
            {"conc": 4, "total_token_throughput": 20000.0, "intvty_p90": 500.0},
        ]
        xs, ys = plot._arm_series(points, 1.0, axes)
        assert xs == [pytest.approx(500.0)]
        assert ys == [pytest.approx(20000.0)]


def test_render_conc_sweep_curve_missing_matplotlib_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When matplotlib is missing, render_conc_sweep_curve returns None gracefully."""
    import builtins

    _real_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("no module named matplotlib (mocked)")
        return _real_import(name, *args, **kwargs)

    from hyperloom.orchestrator.kernel.conc_sweep_plot import render_conc_sweep_curve

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    result = render_conc_sweep_curve(
        {"baseline": {"points": [{"conc": 4, "output_throughput": 400.0}]}, "optimized": {"points": []}},
        tmp_path / "out.png",
        model_label="M",
        gpu_label="G",
        tp=1,
    )
    assert result is None


def test_conc_sweep_plot_series_helpers_filter_and_sort_points():
    from hyperloom.orchestrator.kernel import conc_sweep_plot

    axes = conc_sweep_plot._resolve_axes("synthetic", 2.0)
    xs, ys = conc_sweep_plot._arm_series(
        [
            {"conc": 4, "output_throughput": 800.0},
            {"conc": 2, "output_throughput": 300.0},
            {"conc": 0, "output_throughput": 1000.0},
            {"conc": 1, "output_throughput": None},
            {"conc": 8, "output_throughput": -1},
        ],
        2.0,
        axes,
    )
    assert xs == [150.0, 200.0]
    assert ys == [150.0, 400.0]
    assert conc_sweep_plot._arm_series([{"conc": 0, "output_throughput": 0}], 1.0, axes) == ([], [])

    cx, cy = conc_sweep_plot._ceiling_series(
        {
            "rows": [
                {"conc": 4, "t_peak_tok_s": 1000},
                {"conc": 2, "t_peak_tok_s": "600"},
                {"conc": None, "t_peak_tok_s": 1},
                {"conc": "bad", "t_peak_tok_s": 1},
                {"conc": 1, "t_peak_tok_s": -1},
            ]
        },
        tp_eff=4.0,
    )
    assert cx == [250.0, 300.0]
    assert cy == [250.0, 150.0]
    assert conc_sweep_plot._ceiling_series({"rows": [{"conc": 0, "t_peak_tok_s": 0}]}, 1.0) == ([], [])


def test_render_conc_sweep_curve_with_fake_matplotlib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from hyperloom.orchestrator.kernel.conc_sweep_plot import render_conc_sweep_curve

    calls: dict[str, Any] = {"plots": [], "annotations": [], "labels": [], "titles": [], "closed": False}

    class _FakePatch:
        def set_facecolor(self, value):
            calls["fig_facecolor"] = value

    class _FakeFig:
        patch = _FakePatch()

        def tight_layout(self):
            calls["tight_layout"] = True

        def savefig(self, path, **kwargs):
            calls["saved"] = (path, kwargs)
            Path(path).write_bytes(b"fake-png")

    class _FakeSpine:
        def set_color(self, value):
            calls.setdefault("spines", []).append(value)

    class _FakeLegendText:
        def set_color(self, value):
            calls.setdefault("legend_text_colors", []).append(value)

    class _FakeLegend:
        def get_texts(self):
            return [_FakeLegendText(), _FakeLegendText()]

    class _FakeAx:
        spines = {"left": _FakeSpine(), "right": _FakeSpine()}

        def set_facecolor(self, value):
            calls["ax_facecolor"] = value

        def plot(self, *args, **kwargs):
            calls["plots"].append((args, kwargs))

        def annotate(self, *args, **kwargs):
            calls["annotations"].append((args, kwargs))

        def set_xlabel(self, value, **kwargs):
            calls["labels"].append(("x", value, kwargs))

        def set_ylabel(self, value, **kwargs):
            calls["labels"].append(("y", value, kwargs))

        def set_title(self, value, **kwargs):
            calls["titles"].append((value, kwargs))

        def grid(self, *args, **kwargs):
            calls["grid"] = (args, kwargs)

        def tick_params(self, **kwargs):
            calls["ticks"] = kwargs

        def legend(self, **kwargs):
            calls["legend"] = kwargs
            return _FakeLegend()

    fake_matplotlib = ModuleType("matplotlib")
    fake_matplotlib.use = lambda backend: calls.setdefault("backend", backend)
    fake_pyplot = ModuleType("matplotlib.pyplot")
    fake_pyplot.subplots = lambda figsize: (_FakeFig(), _FakeAx())
    fake_pyplot.close = lambda fig: calls.update({"closed": fig is not None})
    fake_matplotlib.pyplot = fake_pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)

    payload = {
        "baseline": {"points": [{"conc": 4, "output_throughput": 800.0}]},
        "optimized": {"points": [{"conc": 8, "output_throughput": 1200.0}]},
        "roofline_ceiling": {"rows": [{"conc": 4, "t_peak_tok_s": 1600.0}]},
    }
    out_path = tmp_path / "plots" / "curve.png"
    result = render_conc_sweep_curve(
        payload,
        out_path,
        model_label="M",
        gpu_label="MI300X",
        tp=4,
        isl=1024,
        osl=512,
        draw_ceiling=True,
    )

    assert result == out_path
    assert out_path.read_bytes() == b"fake-png"
    assert calls["backend"] == "Agg"
    assert len(calls["plots"]) == 3
    assert len(calls["annotations"]) == 2
    assert "ISL=1024 OSL=512" in calls["titles"][0][0]
    assert calls["closed"] is True


def test_format_conc_sweep_curve_section_empty_summary():
    from hyperloom.orchestrator.actions.executors.report import _format_conc_sweep_curve_section

    assert _format_conc_sweep_curve_section({}) == []


def test_format_conc_sweep_curve_section_with_png():
    from hyperloom.orchestrator.actions.executors.report import _format_conc_sweep_curve_section

    lines = _format_conc_sweep_curve_section({"conc_sweep_curve_png": "reports/conc_sweep_curve.png"})
    embed = next(line for line in lines if line.startswith("!["))
    # final.md lives in reports/, so the embed must use the basename, not the
    # session-root-relative "reports/conc_sweep_curve.png" (which would resolve
    # to reports/reports/... and 404).
    assert embed == "![Concurrency sweep curve](conc_sweep_curve.png)"
    assert "reports/conc_sweep_curve.png" not in embed


# ─────────────────────────────────────────────────────────────────────────────
# Change 1: single-server Option A boot/reuse path (lifecycle-eligible)
# ─────────────────────────────────────────────────────────────────────────────


def _patch_lifecycle_eligible(monkeypatch: pytest.MonkeyPatch, teardown_log: list[tuple]):
    """Patch resolve_lifecycle_params → eligible and track teardown calls."""
    import hyperloom.orchestrator.actions.executors._server_lifecycle as _sl

    def _fake_resolve(_cfg_path):
        return {"eligible": True, "framework": "vllm", "port": 8888, "reason": ""}

    def _fake_teardown(*, pid_dir, framework, port):
        teardown_log.append((str(pid_dir), framework, port))

    monkeypatch.setattr(_sl, "resolve_lifecycle_params", _fake_resolve)
    monkeypatch.setattr(_sl, "teardown_lifecycle_server", _fake_teardown)


def test_single_server_option_a_boot_and_reuse(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Option A: highest-CONC boot round + reuse rounds; one teardown per arm."""
    teardown_log: list[tuple] = []
    _patch_lifecycle_eligible(monkeypatch, teardown_log)

    state = _make_state(baseline_config_path=str(baseline_yaml))
    grid_calls: list[dict] = []

    async def _fake_run_grid(*, grid: list[GridVariant], **kw):
        grid_calls.append(
            {
                "name": grid[0].name,
                "server_lifecycle": kw.get("server_lifecycle"),
                "server_already_ready": kw.get("server_already_ready"),
                "preclean_before_run": kw.get("preclean_before_run"),
            }
        )
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[4, 16, 64]))

    assert payload["status"] == "succeeded"
    # 2 arms × 3 concs = 6 run_grid calls.
    assert len(grid_calls) == 6
    # One teardown per arm.
    assert len(teardown_log) == 2

    # First call per arm = boot round: highest conc, server_already_ready=False, cleanup=False.
    opt_calls = [c for c in grid_calls if c["name"].startswith("optimized_")]
    boot = opt_calls[0]
    assert boot["name"] == "optimized_conc64", f"boot should be highest conc; got {boot['name']}"
    assert boot["server_already_ready"] is False
    assert boot["server_lifecycle"]["cleanup"] is False
    assert boot["preclean_before_run"] is True

    # Middle reuse round: server_already_ready=True, cleanup=False.
    mid = opt_calls[1]
    assert mid["server_already_ready"] is True
    assert mid["server_lifecycle"]["cleanup"] is False
    assert mid["preclean_before_run"] is False

    # Last reuse round: cleanup=True.
    last = opt_calls[-1]
    assert last["server_lifecycle"]["cleanup"] is True


def test_single_server_boot_retry_descend(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """If the highest-CONC boot fails, the next lower CONC is tried as boot."""
    teardown_log: list[tuple] = []
    _patch_lifecycle_eligible(monkeypatch, teardown_log)

    state = _make_state(baseline_config_path=str(baseline_yaml))
    boot_attempts: list[str] = []

    async def _fake_run_grid(*, grid: list[GridVariant], **kw):
        name = grid[0].name
        is_boot = kw.get("server_already_ready") is False
        if is_boot:
            boot_attempts.append(name)
            # Fail the very first (highest-conc) boot attempt only.
            if name.endswith("conc64"):
                return [
                    _fake_variant(name, throughput=None, status="failed", envs=grid[0].extra_envs, error="boot fail")
                ]
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[16, 64]))

    # optimized arm: boot conc64 fails → retries boot at conc16.
    opt_boots = [b for b in boot_attempts if b.startswith("optimized_")]
    assert "optimized_conc64" in opt_boots
    assert "optimized_conc16" in opt_boots
    assert payload["status"] in {"succeeded", "failed"}


def test_single_server_all_boot_fail_falls_back_option_b(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When every boot attempt fails, remaining variants run via Option B."""
    teardown_log: list[tuple] = []
    _patch_lifecycle_eligible(monkeypatch, teardown_log)

    state = _make_state(baseline_config_path=str(baseline_yaml))
    option_b_calls: list[str] = []

    async def _fake_run_grid(*, grid: list[GridVariant], **kw):
        name = grid[0].name
        # Boot rounds (server_lifecycle set, not already ready) always fail.
        if kw.get("server_already_ready") is False and kw.get("server_lifecycle"):
            return [_fake_variant(name, throughput=None, status="failed", envs=grid[0].extra_envs, error="boot fail")]
        # Option B path = no server_lifecycle kwarg.
        if kw.get("server_lifecycle") is None:
            option_b_calls.append(name)
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[8, 16]))

    # After all boots fail, Option B (no server_lifecycle) runs the remaining variants.
    assert len(option_b_calls) > 0, "expected Option B fallback calls"
    assert payload["status"] in {"succeeded", "failed"}


def test_single_server_not_eligible_uses_option_b(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A non-lifecycle-eligible framework routes straight to Option B."""
    import hyperloom.orchestrator.actions.executors._server_lifecycle as _sl

    monkeypatch.setattr(
        _sl,
        "resolve_lifecycle_params",
        lambda _p: {"eligible": False, "framework": "", "port": 8888, "reason": "not a builtin"},
    )

    state = _make_state(baseline_config_path=str(baseline_yaml))
    calls: list[dict] = []

    async def _fake_run_grid(*, grid: list[GridVariant], **kw):
        calls.append({"name": grid[0].name, "server_lifecycle": kw.get("server_lifecycle")})
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[4, 16]))

    assert payload["status"] == "succeeded"
    # Option B: no server_lifecycle kwarg on any call.
    assert all(c["server_lifecycle"] is None for c in calls)


def test_single_server_reuse_loop_stops_on_closing_phase(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """closing_phase set mid-arm skips the remaining reuse points."""
    teardown_log: list[tuple] = []
    _patch_lifecycle_eligible(monkeypatch, teardown_log)

    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **kw):
        # After the boot round (server_already_ready=False), flip closing_phase.
        if kw.get("server_already_ready") is False:
            state.closing_phase = True
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[4, 16, 64]))

    all_points = payload["baseline"]["points"] + payload["optimized"]["points"]
    skipped = [p for p in all_points if p["status"] == "skipped"]
    assert len(skipped) > 0
    assert payload["budget_skip_reason"] == "session_deadline_reserve"


def test_single_server_reuse_exception_recorded_as_failed(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """An exception in a reuse round is captured as a failed point, not raised."""
    teardown_log: list[tuple] = []
    _patch_lifecycle_eligible(monkeypatch, teardown_log)

    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **kw):
        # Boot round succeeds; reuse rounds raise.
        if kw.get("server_already_ready") is True:
            raise RuntimeError("reuse boom")
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[4, 16]))

    all_points = payload["baseline"]["points"] + payload["optimized"]["points"]
    failed = [p for p in all_points if p["status"] == "failed"]
    # Each arm: boot(conc16) ok, reuse(conc4) raises → at least 2 failed points.
    assert len(failed) >= 2
    assert any((p.get("error_class") or "").startswith("single_server_reuse") for p in failed)


def test_single_server_reuse_loop_budget_exhausted(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Task budget exhausted mid-arm skips remaining reuse points."""
    teardown_log: list[tuple] = []
    _patch_lifecycle_eligible(monkeypatch, teardown_log)

    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **kw):
        import time as _t

        # Boot round consumes the whole budget so reuse points get skipped.
        if kw.get("server_already_ready") is False:
            _t.sleep(1.2)
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(
            run_conc_sweep(
                state,
                session_dir,
                concs=[4, 16, 64],
                variant_timeout_sec=1,
                total_budget_sec=2,
            )
        )

    all_points = payload["baseline"]["points"] + payload["optimized"]["points"]
    skipped = [p for p in all_points if p["status"] == "skipped"]
    assert len(skipped) > 0
    assert payload["budget_exhausted"] is True


def test_single_server_boot_exception_falls_back(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A boot round raising an exception is caught and boot-retry-descend continues."""
    teardown_log: list[tuple] = []
    _patch_lifecycle_eligible(monkeypatch, teardown_log)

    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **kw):
        name = grid[0].name
        # Highest-conc boot raises; lower-conc boot succeeds.
        if kw.get("server_already_ready") is False and name.endswith("conc64"):
            raise RuntimeError("boot boom")
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[16, 64]))

    all_points = payload["baseline"]["points"] + payload["optimized"]["points"]
    # conc64 boot exception → recorded as failed; conc16 boots and succeeds.
    failed = [p for p in all_points if p["status"] == "failed" and p["conc"] == 64]
    assert len(failed) >= 1
    assert any((p.get("error_class") or "").startswith("single_server_boot") for p in failed)


def test_single_server_pre_arm_skip_on_closing_phase(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """closing_phase set before the sweep skips every arm's variants."""
    teardown_log: list[tuple] = []
    _patch_lifecycle_eligible(monkeypatch, teardown_log)

    state = _make_state(baseline_config_path=str(baseline_yaml))
    state.closing_phase = True
    ran: list[str] = []

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        ran.append(grid[0].name)
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[4, 16]))

    # No run_grid calls; all variants skipped before any arm starts.
    assert ran == []
    all_points = payload["baseline"]["points"] + payload["optimized"]["points"]
    assert all(p["status"] == "skipped" for p in all_points)
    assert payload["budget_skip_reason"] == "session_deadline_reserve"


# --- budget arithmetic must price a variant at the cap it will be granted -------


def test_the_admission_price_is_the_declared_cap_on_the_synthetic_path(
    monkeypatch: pytest.MonkeyPatch,
):
    """Zero regression: with AgentX off the sweep prices variants exactly as before."""
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    assert _granted_cap_sec(DEFAULT_VARIANT_TIMEOUT_SEC) == float(DEFAULT_VARIANT_TIMEOUT_SEC)


def test_the_admission_price_follows_the_agentx_raise(monkeypatch: pytest.MonkeyPatch):
    """Pricing a round at 1800s while granting it 10800s admits what cannot be paid for.

    The round is then clamped back to the remaining budget and killed mid-warmup
    -- the failure the cap-raise exists to prevent, moved into the sweep's own
    admission check.
    """
    from hyperloom.orchestrator.actions.executors.baseline import agentx_baseline_timeout_sec

    for k in (
        "AGENTX_BASELINE_TIMEOUT_SEC",
        "AGENTX_BASELINE_OVERHEAD_SEC",
        "AGENTX_WARMUP_GRACE_PERIOD",
        "CONC",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    raised = _granted_cap_sec(DEFAULT_VARIANT_TIMEOUT_SEC)
    assert raised == float(agentx_baseline_timeout_sec())
    assert raised > float(DEFAULT_VARIANT_TIMEOUT_SEC)


def test_the_admission_price_never_lowers_an_operator_raised_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    """An operator who asked for longer than AgentX derives keeps what they asked for."""
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    assert _granted_cap_sec(99_999) == 99_999.0


# ─────────────────────────────────────────────────────────────────────────────
# Post-run orphan reap (AMD-AGI/Hyperloom#1354)
# ─────────────────────────────────────────────────────────────────────────────


def test_run_conc_sweep_reaps_stale_servers_after_both_arms(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """After both arms finish (happy path), run_conc_sweep must reap any
    lingering server via the same broad /proc scan used elsewhere: a
    per-variant timeout that fires before a server_lifecycle pidfile is
    written leaves nothing for that pidfile-based teardown to find, so this
    is the safety net that catches it (AMD-AGI/Hyperloom#1354)."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    kill_calls = {"n": 0}

    def _fake_kill():
        kill_calls["n"] += 1

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
        patch("hyperloom.orchestrator.kernel.conc_sweep._kill_stale_servers", side_effect=_fake_kill),
    ):
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[4, 16]))

    assert payload["status"] == "succeeded"
    assert kill_calls["n"] == 1


def test_run_conc_sweep_reaps_stale_servers_even_when_an_arm_raises(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The reap must fire from a ``finally`` -- even when an arm blows up
    with an exception that escapes its own internal handling, not just on
    the happy path. ``_sweep_one_arm_single_server`` itself is mocked out
    (rather than the ``run_grid`` it calls) so its own per-variant error
    handling can't swallow the exception before it reaches run_conc_sweep."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _boom(*_a, **_kw):
        raise RuntimeError("arm blew up")

    kill_calls = {"n": 0}

    def _fake_kill():
        kill_calls["n"] += 1

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep._sweep_one_arm_single_server", side_effect=_boom),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
        patch("hyperloom.orchestrator.kernel.conc_sweep._kill_stale_servers", side_effect=_fake_kill),
    ):
        with pytest.raises(RuntimeError):
            asyncio.run(run_conc_sweep(state, session_dir, concs=[4, 16]))

    assert kill_calls["n"] == 1


def test_run_conc_sweep_skips_reap_under_pytest(
    session_dir: Path,
    baseline_yaml: Path,
):
    """Direct guard: the reap must NOT fire while ``PYTEST_CURRENT_TEST`` is
    set (pytest always sets it for a running test), mirroring the guard on
    the per-launch preclean in ``_grid_runner.py``."""
    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    kill_calls = {"n": 0}

    def _fake_kill():
        kill_calls["n"] += 1

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
        patch("hyperloom.orchestrator.kernel.conc_sweep._kill_stale_servers", side_effect=_fake_kill),
    ):
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[4, 16]))

    assert payload["status"] == "succeeded"
    assert kill_calls["n"] == 0, "must be a no-op while PYTEST_CURRENT_TEST is set"
