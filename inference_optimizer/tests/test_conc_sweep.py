# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the post-optimization concurrency sweep runner."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors._grid_runner import (
    GridVariant,
    VariantResult,
)
from inference_optimizer.orchestrator.conc_sweep import (
    DEFAULT_CONCS,
    DEFAULT_TOTAL_BUDGET_SEC,
    _build_comparison,
    _build_grid,
    _has_optimization,
    format_summary_line,
    run_conc_sweep,
)
from inference_optimizer.orchestrator.shared_state import SharedState


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
    s.current_best = current_best if current_best is not None else {
        "extra_server_args": "--enable-torch-compile",
        "extra_envs": {"SGLANG_FOO": "1"},
    }
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


# _build_grid
def test_build_grid_two_arms_per_conc():
    grid = _build_grid(
        concs=[1, 4, 16],
        isl=1024, osl=1024,
        num_prompts_factor=5,
        optimized_args="--enable-torch-compile",
        optimized_envs={"FOO": "1"},
    )
    assert len(grid) == 6
    names = [v.name for v in grid]
    assert names == [
        "baseline_conc1", "baseline_conc4", "baseline_conc16",
        "optimized_conc1", "optimized_conc4", "optimized_conc16",
    ]
    baseline = next(v for v in grid if v.name == "baseline_conc4")
    assert baseline.extra_server_args == ""
    assert baseline.extra_envs == {
        "CONC": "4", "ISL": "1024", "OSL": "1024", "NUM_PROMPTS": "20",
    }
    optimized = next(v for v in grid if v.name == "optimized_conc16")
    assert optimized.extra_server_args == "--enable-torch-compile"
    assert optimized.extra_envs == {
        "FOO": "1",
        "CONC": "16", "ISL": "1024", "OSL": "1024", "NUM_PROMPTS": "80",
    }


def test_build_grid_num_prompts_floor():
    """When NUM_PROMPTS_FACTOR=1, NUM_PROMPTS still >= CONC (sanity)."""
    grid = _build_grid(
        concs=[64],
        isl=1024, osl=1024,
        num_prompts_factor=1,
        optimized_args="--x",
        optimized_envs={},
    )
    baseline = next(v for v in grid if v.name.startswith("baseline_"))
    assert int(baseline.extra_envs["NUM_PROMPTS"]) >= 64


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
        {"conc": 4, "output_throughput": None,  "status": "failed"},
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
        {"conc": 4,  "output_throughput": 220.0, "status": "succeeded"},
        {"conc": 16, "output_throughput": 400.0, "status": "succeeded"},
    ]
    rows, summary = _build_comparison(baseline, optimized)
    assert [r["conc"] for r in rows] == [1, 4, 16]
    assert rows[0]["optimized_tput"] is None
    assert rows[2]["baseline_tput"] is None
    assert summary["successful_pairs"] == 1


# Skip paths
@pytest.mark.parametrize("override, reason", [
    ({"baseline_tput": 0.0}, "no_baseline_tput"),
    ({"isl": 0},             "missing_workload_shape"),
    ({"osl": 0},             "missing_workload_shape"),
    (
        {"current_best": {"extra_server_args": "", "extra_envs": {}}},
        "no_optimization_to_compare",
    ),
])
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
        "inference_optimizer.orchestrator.conc_sweep.run_grid",
    ) as mock_run_grid:
        payload = asyncio.run(run_conc_sweep(state, session_dir))

    assert payload["status"] == "skipped"
    assert payload["skip_reason"] == reason
    mock_run_grid.assert_not_called()
    assert not (session_dir / "reports" / "conc_sweep_summary.json").exists()


def test_run_conc_sweep_skip_empty_conc_list(
    session_dir: Path, baseline_yaml: Path,
):
    state = _make_state(baseline_config_path=str(baseline_yaml))
    with patch(
        "inference_optimizer.orchestrator.conc_sweep.run_grid",
    ) as mock_run_grid:
        payload = asyncio.run(run_conc_sweep(state, session_dir, concs=[]))
    assert payload["status"] == "skipped"
    assert payload["skip_reason"] == "no_optimization_to_compare" or \
           payload["skip_reason"] == "empty_conc_list"
    # has_opt passes (non-empty fixture), so empty list is the reason.
    assert payload["skip_reason"] == "empty_conc_list"
    mock_run_grid.assert_not_called()


def test_run_conc_sweep_skip_missing_config(
    session_dir: Path, tmp_path: Path,
):
    state = _make_state(
        baseline_config_path=str(tmp_path / "does_not_exist.yaml"),
    )
    with patch(
        "inference_optimizer.orchestrator.conc_sweep.run_grid",
    ) as mock_run_grid:
        payload = asyncio.run(run_conc_sweep(state, session_dir))
    assert payload["status"] == "skipped"
    assert payload["skip_reason"] == "baseline_config_missing"
    mock_run_grid.assert_not_called()


# Happy path with mocked run_grid
def _fake_variant(
    name: str, *, throughput: float | None, status: str = "succeeded",
    envs: dict[str, str] | None = None, error: str | None = None,
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


def test_run_conc_sweep_happy_path_writes_reports(
    session_dir: Path, baseline_yaml: Path,
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
            out.append(_fake_variant(
                v.name, throughput=tput, envs=v.extra_envs,
            ))
        return out

    def _fake_materialize(src, out_dir, **_kw):
        out = Path(out_dir) / "conc_sweep_base.with_envs.yaml"
        out.write_text(Path(src).read_text())
        return out

    with patch(
        "inference_optimizer.orchestrator.conc_sweep.run_grid",
        side_effect=_fake_run_grid,
    ), patch(
        "inference_optimizer.orchestrator.conc_sweep.materialize_config_with_envs",
        side_effect=_fake_materialize,
    ):
        payload = asyncio.run(run_conc_sweep(
            state, session_dir, concs=[1, 4, 16],
        ))

    assert payload["status"] == "succeeded"
    assert payload["isl"] == 1024
    assert payload["osl"] == 1024
    assert payload["tp"] == 8
    assert payload["concs_requested"] == [1, 4, 16]
    assert len(payload["baseline"]["points"]) == 3
    assert len(payload["optimized"]["points"]) == 3
    # All three pairs should produce a 1.30x speedup.
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
    # Regression for a payload-mutation-after-write bug: self-referential
    # paths must land in the on-disk JSON.
    assert disk["report_json_path"] == summary_path.as_posix()
    assert disk["report_csv_path"] == csv_path.as_posix()

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 6  # 3 baseline + 3 optimized
    assert {r["arm"] for r in rows} == {"baseline", "optimized"}

    # final.json pointer is owned by report.py at CLOSE; conc_sweep must not touch it.
    final_json_path = session_dir / "reports" / "final.json"
    assert not final_json_path.exists()


def test_run_conc_sweep_optimized_oom_yields_failed_pair(
    session_dir: Path, baseline_yaml: Path,
):
    """An optimized variant crashing yields a failed pair but doesn't abort the summary."""
    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        out = []
        for v in grid:
            conc = int(v.extra_envs["CONC"])
            if v.name.startswith("baseline_"):
                out.append(_fake_variant(
                    v.name, throughput=100.0 * conc, envs=v.extra_envs,
                ))
            else:
                if conc == 16:
                    out.append(_fake_variant(
                        v.name, throughput=None, status="failed",
                        envs=v.extra_envs, error="CUDA OOM",
                    ))
                else:
                    out.append(_fake_variant(
                        v.name, throughput=120.0 * conc, envs=v.extra_envs,
                    ))
        return out

    def _fake_materialize(src, out_dir, **_kw):
        out = Path(out_dir) / "conc_sweep_base.with_envs.yaml"
        out.write_text(Path(src).read_text())
        return out

    with patch(
        "inference_optimizer.orchestrator.conc_sweep.run_grid",
        side_effect=_fake_run_grid,
    ), patch(
        "inference_optimizer.orchestrator.conc_sweep.materialize_config_with_envs",
        side_effect=_fake_materialize,
    ):
        payload = asyncio.run(run_conc_sweep(
            state, session_dir, concs=[1, 4, 16],
        ))

    # Two successful pairs (conc 1, 4) + one failed pair (conc 16).
    assert payload["summary"]["successful_pairs"] == 2
    assert payload["summary"]["failed_pairs"] == 1
    fail_row = next(r for r in payload["comparison"] if r["conc"] == 16)
    assert fail_row["speedup"] is None
    assert fail_row["optimized_status"] == "failed"
    assert payload["status"] == "succeeded"


def test_run_conc_sweep_args_only_optimization_triggers_run(
    session_dir: Path, baseline_yaml: Path,
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
        return [
            _fake_variant(v.name, throughput=100.0, envs=v.extra_envs)
            for v in grid
        ]

    def _fake_materialize(src, out_dir, **_kw):
        out = Path(out_dir) / "conc_sweep_base.with_envs.yaml"
        out.write_text(Path(src).read_text())
        return out

    with patch(
        "inference_optimizer.orchestrator.conc_sweep.run_grid",
        side_effect=_fake_run_grid,
    ) as mock_run, patch(
        "inference_optimizer.orchestrator.conc_sweep.materialize_config_with_envs",
        side_effect=_fake_materialize,
    ):
        payload = asyncio.run(run_conc_sweep(
            state, session_dir, concs=[1, 4],
        ))

    assert payload["status"] in ("succeeded", "failed")
    assert mock_run.call_count == 4  # 2 arms × 2 concs
    assert payload["optimized"]["extra_server_args"] == "--enable-torch-compile"
    assert payload["optimized"]["extra_envs"] == {}


def test_run_conc_sweep_envs_only_optimization_triggers_run(
    session_dir: Path, baseline_yaml: Path,
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
        return [
            _fake_variant(v.name, throughput=100.0, envs=v.extra_envs)
            for v in grid
        ]

    def _fake_materialize(src, out_dir, **_kw):
        out = Path(out_dir) / "conc_sweep_base.with_envs.yaml"
        out.write_text(Path(src).read_text())
        return out

    with patch(
        "inference_optimizer.orchestrator.conc_sweep.run_grid",
        side_effect=_fake_run_grid,
    ) as mock_run, patch(
        "inference_optimizer.orchestrator.conc_sweep.materialize_config_with_envs",
        side_effect=_fake_materialize,
    ):
        payload = asyncio.run(run_conc_sweep(
            state, session_dir, concs=[1],
        ))

    assert mock_run.call_count == 2  # 2 arms × 1 conc
    assert payload["optimized"]["extra_envs"] == {"SGLANG_MOE_ENABLE": "1"}


def test_run_conc_sweep_does_not_touch_final_json(
    session_dir: Path, baseline_yaml: Path,
):
    """conc_sweep runs before CLOSE and must never create or modify final.json."""
    state = _make_state(baseline_config_path=str(baseline_yaml))
    final_json_path = session_dir / "reports" / "final.json"
    assert not final_json_path.exists()

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [
            _fake_variant(v.name, throughput=100.0, envs=v.extra_envs)
            for v in grid
        ]

    def _fake_materialize(src, out_dir, **_kw):
        out = Path(out_dir) / "conc_sweep_base.with_envs.yaml"
        out.write_text(Path(src).read_text())
        return out

    with patch(
        "inference_optimizer.orchestrator.conc_sweep.run_grid",
        side_effect=_fake_run_grid,
    ), patch(
        "inference_optimizer.orchestrator.conc_sweep.materialize_config_with_envs",
        side_effect=_fake_materialize,
    ):
        payload = asyncio.run(run_conc_sweep(
            state, session_dir, concs=[1],
        ))

    assert payload["status"] in ("succeeded", "failed")
    assert (session_dir / "reports" / "conc_sweep_summary.json").exists()
    assert not final_json_path.exists()


# format_summary_line
def test_format_summary_line_skip():
    payload = {"status": "skipped", "skip_reason": "no_baseline_tput"}
    line = format_summary_line(payload)
    assert "skipped" in line
    assert "no_baseline_tput" in line


def test_format_summary_line_success():
    payload = {
        "status": "succeeded",
        "summary": {
            "successful_pairs": 7,
            "failed_pairs":     1,
            "best_speedup":     1.55,
            "best_conc":        32,
            "median_speedup":   1.32,
        },
    }
    line = format_summary_line(payload)
    assert "succeeded" in line
    assert "7+1f" in line
    assert "1.55x" in line
    assert "conc=32" in line
    assert "median=1.32x" in line


def test_default_concs_is_powers_of_two():
    """Doc-pin: default ladder is [1,2,4,8,16,32,64,128]."""
    assert DEFAULT_CONCS == [1, 2, 4, 8, 16, 32, 64, 128]


def test_default_total_budget_is_two_and_half_hours():
    """Doc-pin: default total wall-clock budget is 2.5h."""
    assert DEFAULT_TOTAL_BUDGET_SEC == 9000


# Total wall-clock budget
def test_run_conc_sweep_budget_exhausted_marks_remaining_skipped(
    session_dir: Path, baseline_yaml: Path,
):
    """When the wall-clock budget runs out, remaining variants are skipped."""
    state = _make_state(baseline_config_path=str(baseline_yaml))
    calls = {"n": 0}

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        import time as _t
        _t.sleep(0.6)
        calls["n"] += 1
        return [
            _fake_variant(v.name, throughput=100.0, envs=v.extra_envs)
            for v in grid
        ]

    def _fake_materialize(src, out_dir, **_kw):
        out = Path(out_dir) / "conc_sweep_base.with_envs.yaml"
        out.write_text(Path(src).read_text())
        return out

    with patch(
        "inference_optimizer.orchestrator.conc_sweep.run_grid",
        side_effect=_fake_run_grid,
    ), patch(
        "inference_optimizer.orchestrator.conc_sweep.materialize_config_with_envs",
        side_effect=_fake_materialize,
    ):
        payload = asyncio.run(run_conc_sweep(
            state, session_dir,
            concs=[1, 4, 16, 64],
            total_budget_sec=1,
        ))

    all_points = payload["baseline"]["points"] + payload["optimized"]["points"]
    statuses = [p["status"] for p in all_points]
    assert "succeeded" in statuses
    assert "skipped" in statuses
    skipped_pts = [p for p in all_points if p["status"] == "skipped"]
    for p in skipped_pts:
        assert p["error_class"] == "budget_exhausted"
    assert payload["budget_exhausted"] is True
    assert payload["total_budget_sec"] == 1
    assert calls["n"] < 8


def test_run_conc_sweep_zero_budget_disables_gate(
    session_dir: Path, baseline_yaml: Path,
):
    """``total_budget_sec <= 0`` disables the gate; every variant is launched."""
    state = _make_state(baseline_config_path=str(baseline_yaml))

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [
            _fake_variant(v.name, throughput=100.0, envs=v.extra_envs)
            for v in grid
        ]

    def _fake_materialize(src, out_dir, **_kw):
        out = Path(out_dir) / "conc_sweep_base.with_envs.yaml"
        out.write_text(Path(src).read_text())
        return out

    with patch(
        "inference_optimizer.orchestrator.conc_sweep.run_grid",
        side_effect=_fake_run_grid,
    ) as mock_run, patch(
        "inference_optimizer.orchestrator.conc_sweep.materialize_config_with_envs",
        side_effect=_fake_materialize,
    ):
        payload = asyncio.run(run_conc_sweep(
            state, session_dir,
            concs=[1, 4],
            total_budget_sec=0,
        ))

    assert payload["budget_exhausted"] is False
    assert payload["total_budget_sec"] is None
    assert mock_run.call_count == 4  # 2 arms × 2 concs


def test_run_conc_sweep_per_variant_timeout_clamped_to_remaining_budget(
    session_dir: Path, baseline_yaml: Path,
):
    """Per-variant timeout is clamped to the remaining budget."""
    state = _make_state(baseline_config_path=str(baseline_yaml))
    recorded_timeouts: list[int] = []

    async def _fake_run_grid(*, grid: list[GridVariant], variant_timeout_sec: int, **_kw):
        recorded_timeouts.append(variant_timeout_sec)
        return [
            _fake_variant(v.name, throughput=100.0, envs=v.extra_envs)
            for v in grid
        ]

    def _fake_materialize(src, out_dir, **_kw):
        out = Path(out_dir) / "conc_sweep_base.with_envs.yaml"
        out.write_text(Path(src).read_text())
        return out

    with patch(
        "inference_optimizer.orchestrator.conc_sweep.run_grid",
        side_effect=_fake_run_grid,
    ), patch(
        "inference_optimizer.orchestrator.conc_sweep.materialize_config_with_envs",
        side_effect=_fake_materialize,
    ):
        asyncio.run(run_conc_sweep(
            state, session_dir,
            concs=[1],
            variant_timeout_sec=3600,
            total_budget_sec=120,
        ))

    assert all(t <= 120 for t in recorded_timeouts)
    assert all(t >= 1 for t in recorded_timeouts)


# ActionExecutor integration (SWEEP-phase dispatch)
def test_conc_sweep_executor_loads_state_and_dispatches(
    session_dir: Path, baseline_yaml: Path,
):
    """ConcSweepExecutor reloads SharedState and threads registry fields through."""
    from inference_optimizer.orchestrator.action_executors.conc_sweep import (
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

    async def _fake_run(state_arg, sd, *, concs, variant_timeout_sec,
                       total_budget_sec, **_kw):
        captured["concs"] = list(concs)
        captured["timeout"] = variant_timeout_sec
        captured["budget"] = total_budget_sec
        return {
            "status": "succeeded", "summary": {"successful_pairs": 2},
        }

    with patch(
        "inference_optimizer.orchestrator.action_executors.conc_sweep.run_conc_sweep",
        side_effect=_fake_run,
    ):
        result = asyncio.run(ConcSweepExecutor()(_Ctx()))

    assert result["status"] == "succeeded"
    assert captured["concs"] == [1, 4]
    assert captured["timeout"] == 30
    assert captured["budget"] == 60


def test_conc_sweep_executor_missing_session_dir_yields_failure():
    from inference_optimizer.orchestrator.action_executors.conc_sweep import (
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


def test_conc_sweep_executor_remaps_skip_to_succeeded(
    session_dir: Path, baseline_yaml: Path,
):
    """A run_conc_sweep skip surfaces ``was_skipped=True`` + ``status='succeeded'``."""
    from inference_optimizer.orchestrator.action_executors.conc_sweep import (
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
        "inference_optimizer.orchestrator.action_executors.conc_sweep.run_conc_sweep",
        side_effect=_fake_run,
    ):
        result = asyncio.run(ConcSweepExecutor()(_Ctx()))
    assert result["status"] == "succeeded"
    assert result["was_skipped"] is True
    assert result["skip_reason"] == "no_baseline_tput"


def test_format_summary_line_budget_exhausted_suffix():
    payload = {
        "status": "succeeded",
        "budget_exhausted": True,
        "total_budget_sec": 7200,
        "summary": {
            "successful_pairs": 4, "failed_pairs": 0,
            "best_speedup": 1.20, "best_conc": 8, "median_speedup": 1.15,
        },
    }
    line = format_summary_line(payload)
    assert "budget_exhausted" in line
    assert "@7200s" in line


# Bug #12: record_conc_sweep writes state.last_conc_sweep for SWEEP completion detection.
def test_record_conc_sweep_writes_last_conc_sweep():
    s = SharedState()
    assert s.last_conc_sweep == {}
    s.record_conc_sweep({
        "status":           "succeeded",
        "skip_reason":      "",
        "was_skipped":      False,
        "budget_exhausted": False,
        "summary":          {"successful_pairs": 8, "best_speedup": 1.05},
        "workspace":        "/tmp/conc_sweep_xyz",
    })
    assert s.last_conc_sweep.get("status") == "succeeded"
    assert s.last_conc_sweep.get("summary", {}).get("successful_pairs") == 8
    assert s.last_conc_sweep.get("ts")
    # Skip cases also recorded so SWEEP exits cleanly even on skip.
    s.record_conc_sweep({
        "status":       "skipped",
        "skip_reason":  "no_optimization_to_compare",
        "was_skipped":  True,
    })
    assert s.last_conc_sweep.get("status") == "skipped"
    assert s.last_conc_sweep.get("skip_reason") == "no_optimization_to_compare"
    assert s.last_conc_sweep.get("was_skipped") is True


def test_exit_normal_sweep_returns_conc_sweep_done():
    """Bug #12: SWEEP→CLOSE must fire on conc_sweep completion, not only sweep_done."""
    from inference_optimizer.orchestrator.phase_state import exit_normal_sweep

    class _State:
        last_sweep = {}             # no sweep recorded
        last_conc_sweep = {}
        phase = "SWEEP"
        phase_started_ts = "2026-06-02T10:00:00+00:00"
        max_minutes = 360
        phase_budget_pct = {"SWEEP": 0.50}

    # No sweep, no conc_sweep => don't exit (budget remaining).
    assert exit_normal_sweep(_State()) is None

    _State.last_conc_sweep = {"status": "succeeded"}
    result = exit_normal_sweep(_State())
    assert result is not None
    reason, evidence = result
    assert reason == "conc_sweep_done", reason
    assert evidence.get("conc_sweep_status") == "succeeded"

    # Skipped also counts as "done" (action ran to its terminal decision).
    for terminal in ("partial", "completed", "skipped"):
        _State.last_conc_sweep = {"status": terminal}
        result = exit_normal_sweep(_State())
        assert result is not None and result[0] == "conc_sweep_done", terminal


def test_on_enter_sweep_drains_pending_keep_integrates(monkeypatch):
    """Bug #7: KERNEL→SWEEP must drain pending KEEP integrates before enqueuing sweep."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from inference_optimizer.orchestrator import kernel_request_handlers

    fake_integrate = AsyncMock(
        return_value={
            "status": "ok", "decision": "KEEP",
            "kernel_id": "k001", "gain_pct": 1.5,
        },
    )
    monkeypatch.setattr(
        kernel_request_handlers, "integrate_handler", fake_integrate,
    )
    from inference_optimizer.orchestrator import coordinator as coord_mod
    if hasattr(coord_mod, "integrate_handler"):
        monkeypatch.setattr(coord_mod, "integrate_handler", fake_integrate)

    coord = MagicMock()
    coord.shared_state = MagicMock()
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.rejected_kernel_ids = []
    coord.shared_state.save = MagicMock()
    pending_queue = ["k001", "k002"]
    coord.shared_state.next_pending_keep_kernel_id = lambda: (
        pending_queue.pop(0) if pending_queue else ""
    )
    coord.session_dir = Path("/tmp/sess")

    from inference_optimizer.orchestrator.coordinator import Coordinator
    asyncio.run(Coordinator._drain_pending_keep_integrates(coord))

    assert fake_integrate.await_count == 2, fake_integrate.await_args_list
    assert coord.shared_state.save.call_count >= 2


def test_conc_sweep_phase_singleton_denies_after_auto_enqueue():
    """Bug #11: ``conc_sweep_phase_singleton`` denies LLM conc_sweep proposals after auto-enqueue."""
    from inference_optimizer.orchestrator.policy import PolicyGate, PolicyDenied

    class _State:
        phase = "SWEEP"
        phase_history = [{
            "to_phase": "SWEEP",
            "evidence": {"auto_conc_sweep_task_id": "cs-abc-123"},
        }]

    gate = PolicyGate.__new__(PolicyGate)
    gate.shared_state = _State()

    with pytest.raises(PolicyDenied) as exc_info:
        gate._validate_conc_sweep_singleton(
            {"params": {}}, intent_kind="propose_action",
        )
    assert exc_info.value.rule == "conc_sweep_phase_singleton"
    assert "auto_conc_sweep_task_id" in str(exc_info.value)

    # Operator bypass works.
    gate._validate_conc_sweep_singleton(
        {"params": {"bypass_conc_sweep_singleton": True}},
        intent_kind="propose_action",
    )

    # No evidence stamp -> rule is inert.
    _State.phase_history = [{"to_phase": "SWEEP", "evidence": {}}]
    gate._validate_conc_sweep_singleton(
        {"params": {}}, intent_kind="propose_action",
    )

    # Outside SWEEP -> rule is inert.
    _State.phase_history = [{
        "to_phase": "EXPLORE",
        "evidence": {"auto_conc_sweep_task_id": "cs-x"},
    }]
    gate._validate_conc_sweep_singleton(
        {"params": {}}, intent_kind="propose_action",
    )
