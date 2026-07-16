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

from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    VariantResult,
)
from hyperloom.orchestrator.kernel.conc_sweep import (
    DEFAULT_CONCS,
    DEFAULT_TOTAL_BUDGET_SEC,
    _build_arm_grid,
    _build_comparison,
    _build_grid,
    _conc_sweep_single_server_enabled,
    _flush_conc_sweep_report,
    _flush_partial_conc_sweep_report,
    _has_optimization,
    _order_concs_desc,
    run_conc_sweep,
)
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


# _build_grid
def test_build_grid_two_arms_per_conc():
    grid = _build_grid(
        concs=[1, 4, 16],
        isl=1024,
        osl=1024,
        num_prompts_factor=5,
        optimized_args="--enable-torch-compile",
        optimized_envs={"FOO": "1"},
    )
    assert len(grid) == 6
    names = [v.name for v in grid]
    # CONC-major with arms interleaved; no anchor preserves ladder order.
    assert names == [
        "baseline_conc1",
        "optimized_conc1",
        "baseline_conc4",
        "optimized_conc4",
        "baseline_conc16",
        "optimized_conc16",
    ]
    baseline = next(v for v in grid if v.name == "baseline_conc4")
    assert baseline.extra_server_args == ""
    assert baseline.extra_envs == {
        "CONC": "4",
        "ISL": "1024",
        "OSL": "1024",
        "NUM_PROMPTS": "20",
        # Accuracy eval off by default for conc_sweep.
        "RUN_EVAL": "false",
    }
    optimized = next(v for v in grid if v.name == "optimized_conc16")
    assert optimized.extra_server_args == "--enable-torch-compile"
    assert optimized.extra_envs == {
        "FOO": "1",
        "CONC": "16",
        "ISL": "1024",
        "OSL": "1024",
        "NUM_PROMPTS": "80",
        "RUN_EVAL": "false",
    }


def test_build_grid_anchor_conc_runs_first():
    """The operating-point CONC emits its complete A/B pair before any other."""
    grid = _build_grid(
        concs=[1, 4, 16, 64],
        isl=1024,
        osl=1024,
        num_prompts_factor=5,
        optimized_args="--x",
        optimized_envs={},
        anchor_conc=64,
    )
    names = [v.name for v in grid]
    assert names[:2] == ["baseline_conc64", "optimized_conc64"]
    # Remaining concs keep requested order, arms interleaved.
    assert names[2:] == [
        "baseline_conc1",
        "optimized_conc1",
        "baseline_conc4",
        "optimized_conc4",
        "baseline_conc16",
        "optimized_conc16",
    ]


def test_build_grid_anchor_absent_is_noop():
    """An anchor not present in the ladder (or <=0) leaves the order untouched."""
    for anchor in (0, 999):
        grid = _build_grid(
            concs=[1, 8],
            isl=512,
            osl=512,
            num_prompts_factor=5,
            optimized_args="--x",
            optimized_envs={},
            anchor_conc=anchor,
        )
        assert [v.name for v in grid] == [
            "baseline_conc1",
            "optimized_conc1",
            "baseline_conc8",
            "optimized_conc8",
        ]


def test_build_grid_disables_eval_by_default(monkeypatch):
    """Every conc_sweep variant carries RUN_EVAL=false unless opted back in."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SWEEP_RUN_EVAL", raising=False)
    grid = _build_grid(
        concs=[1, 8],
        isl=512,
        osl=512,
        num_prompts_factor=5,
        optimized_args="--x",
        optimized_envs={},
    )
    assert grid, "expected a non-empty grid"
    assert all(v.extra_envs.get("RUN_EVAL") == "false" for v in grid)


def test_build_grid_eval_opt_in(monkeypatch):
    """INFERENCE_OPTIMIZER_SWEEP_RUN_EVAL=1 re-enables the per-point accuracy gate."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SWEEP_RUN_EVAL", "1")
    grid = _build_grid(
        concs=[1, 8],
        isl=512,
        osl=512,
        num_prompts_factor=5,
        optimized_args="--x",
        optimized_envs={},
    )
    assert grid, "expected a non-empty grid"
    assert all("RUN_EVAL" not in v.extra_envs for v in grid)


def test_build_grid_num_prompts_floor():
    """When NUM_PROMPTS_FACTOR=1, NUM_PROMPTS still >= CONC (sanity)."""
    grid = _build_grid(
        concs=[64],
        isl=1024,
        osl=1024,
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
        return [
            _fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid
        ]

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


def test_run_conc_sweep_zero_budget_disables_gate(
    session_dir: Path,
    baseline_yaml: Path,
):
    """``total_budget_sec <= 0`` disables the gate; every variant is launched."""
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

    assert payload["budget_exhausted"] is False
    assert payload["total_budget_sec"] is None
    assert mock_run.call_count == 4  # 2 arms × 2 concs


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


def test_exit_normal_sweep_returns_conc_sweep_done():
    """SWEEP→CLOSE must fire on conc_sweep completion, not only sweep_done."""
    from hyperloom.orchestrator.phases.machine_state import exit_normal_sweep

    class _State:
        last_sweep = {}  # no sweep recorded
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

    # Skipped also counts as "done" (action reached a terminal decision).
    for terminal in ("partial", "completed", "skipped"):
        _State.last_conc_sweep = {"status": terminal}
        result = exit_normal_sweep(_State())
        assert result is not None and result[0] == "conc_sweep_done", terminal

    _State.last_conc_sweep = {"status": "failed"}
    result = exit_normal_sweep(_State())
    assert result is not None
    reason, evidence = result
    assert reason == "conc_sweep_failed"
    assert evidence.get("conc_sweep_status") == "failed"


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
    pending_queue = ["k001", "k002"]
    coord.shared_state.next_pending_keep_kernel_id = lambda: pending_queue.pop(0) if pending_queue else ""
    coord.session_dir = Path("/tmp/sess")

    from hyperloom.orchestrator.loop.coordinator import Coordinator

    asyncio.run(Coordinator._drain_pending_keep_integrates(coord))

    assert fake_integrate.await_count == 2, fake_integrate.await_args_list
    assert coord.shared_state.save.call_count >= 2


def test_conc_sweep_phase_singleton_denies_after_auto_enqueue():
    """``conc_sweep_phase_singleton`` denies LLM conc_sweep proposals after auto-enqueue."""
    from hyperloom.orchestrator.policy.gate import PolicyGate, PolicyDenied

    class _State:
        phase = "SWEEP"
        phase_history = [
            {
                "to_phase": "SWEEP",
                "evidence": {"auto_conc_sweep_task_id": "cs-abc-123"},
            }
        ]

    gate = PolicyGate.__new__(PolicyGate)
    gate.shared_state = _State()

    with pytest.raises(PolicyDenied) as exc_info:
        gate._validate_conc_sweep_singleton(
            {"params": {}},
            intent_kind="propose_action",
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
        {"params": {}},
        intent_kind="propose_action",
    )

    # Outside SWEEP -> rule is inert.
    _State.phase_history = [
        {
            "to_phase": "EXPLORE",
            "evidence": {"auto_conc_sweep_task_id": "cs-x"},
        }
    ]
    gate._validate_conc_sweep_singleton(
        {"params": {}},
        intent_kind="propose_action",
    )


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

def test_conc_sweep_single_server_enabled_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
    assert _conc_sweep_single_server_enabled() is True


def test_conc_sweep_single_server_enabled_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", "0")
    assert _conc_sweep_single_server_enabled() is False


def test_conc_sweep_single_server_enabled_false_str(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", "false")
    assert _conc_sweep_single_server_enabled() is False


def test_run_conc_sweep_single_server_arm_major_order(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """In single-server mode optimized arm runs before baseline."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
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
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
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


def test_run_conc_sweep_legacy_path_with_env_off(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER=0 uses the legacy CONC-major path."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", "0")
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
        asyncio.run(run_conc_sweep(state, session_dir, concs=[4, 16]))

    # Legacy: CONC-major interleaving → baseline_ and optimized_ are interleaved per CONC.
    # The first two calls should be the same CONC (one arm each).
    assert len(call_log) == 4
    first_conc = call_log[0].split("conc")[1]
    second_conc = call_log[1].split("conc")[1]
    assert first_conc == second_conc, f"expected CONC-major interleaving; got {call_log}"


def test_run_conc_sweep_partial_sweep_writes_incremental_checkpoint(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """After each conc point a partial checkpoint is written to disk."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", "0")
    state = _make_state(baseline_config_path=str(baseline_yaml))
    checkpoints: list[dict] = []
    json_path = session_dir / "reports" / "conc_sweep_summary.json"

    real_flush = _flush_partial_conc_sweep_report

    def _intercept_flush(*args, **kwargs):
        real_flush(*args, **kwargs)
        if json_path.exists():
            try:
                checkpoints.append(json.loads(json_path.read_text()))
            except Exception:
                pass

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
        patch("hyperloom.orchestrator.kernel.conc_sweep._flush_partial_conc_sweep_report", side_effect=_intercept_flush),
    ):
        asyncio.run(run_conc_sweep(state, session_dir, concs=[4, 16], write_reports=True))

    # Intermediate checkpoints should carry status=in_progress.
    assert any(c.get("status") == "in_progress" for c in checkpoints), (
        f"no in_progress checkpoint found; checkpoints={[c.get('status') for c in checkpoints]}"
    )
    # Final file should have a terminal status.
    final = json.loads(json_path.read_text())
    assert final["status"] in {"succeeded", "failed", "skipped"}


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
    result = _fake_variant("baseline_conc4", throughput=100.0, envs={"CONC": "4", "ISL": "512", "OSL": "512", "NUM_PROMPTS": "20"})
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
# Change 4: session deadline detection
# ─────────────────────────────────────────────────────────────────────────────

def test_run_conc_sweep_stops_on_closing_phase(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When state.closing_phase is True, remaining variants are skipped."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", "0")
    state = _make_state(baseline_config_path=str(baseline_yaml))
    calls: list[str] = []

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        for v in grid:
            calls.append(v.name)
        # Flip closing_phase after first call so remaining are skipped.
        state.closing_phase = True
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(
            run_conc_sweep(state, session_dir, concs=[1, 4, 16, 64])
        )

    all_points = payload["baseline"]["points"] + payload["optimized"]["points"]
    skipped = [p for p in all_points if p["status"] == "skipped"]
    assert len(skipped) > 0, "expected some variants to be skipped after closing_phase set"
    assert payload["budget_exhausted"] is True
    assert payload["budget_skip_reason"] == "session_deadline_reserve"


def test_run_conc_sweep_session_deadline_via_remaining_minutes(
    session_dir: Path,
    baseline_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When remaining_minutes() returns a tiny value, variants are skipped."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", "0")
    state = _make_state(baseline_config_path=str(baseline_yaml))
    # Force remaining_minutes to return 0 (no budget left).
    monkeypatch.setattr(type(state), "remaining_minutes", lambda self: 0.0)
    # Override max_minutes so remaining_minutes is called (>0).
    state.max_minutes = 60.0

    async def _fake_run_grid(*, grid: list[GridVariant], **_kw):
        return [_fake_variant(v.name, throughput=100.0, envs=v.extra_envs) for v in grid]

    with (
        patch("hyperloom.orchestrator.kernel.conc_sweep.run_grid", side_effect=_fake_run_grid),
        patch("hyperloom.orchestrator.kernel.conc_sweep.materialize_config_with_envs", side_effect=_fake_materialize),
    ):
        payload = asyncio.run(
            run_conc_sweep(state, session_dir, concs=[1, 4, 16])
        )

    all_points = payload["baseline"]["points"] + payload["optimized"]["points"]
    # With remaining=0 and reserve=120s, all points should be skipped.
    assert all(p["status"] == "skipped" for p in all_points), (
        f"expected all skipped; statuses={[p['status'] for p in all_points]}"
    )


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


def test_format_conc_sweep_curve_section_empty_summary():
    from hyperloom.orchestrator.actions.executors.report import _format_conc_sweep_curve_section

    assert _format_conc_sweep_curve_section({}) == []


def test_format_conc_sweep_curve_section_with_png():
    from hyperloom.orchestrator.actions.executors.report import _format_conc_sweep_curve_section

    lines = _format_conc_sweep_curve_section({"conc_sweep_curve_png": "reports/conc_sweep_curve.png"})
    assert any("conc_sweep_curve.png" in line for line in lines)
    assert any("![" in line for line in lines)


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
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
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
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
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
                return [_fake_variant(name, throughput=None, status="failed", envs=grid[0].extra_envs, error="boot fail")]
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
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
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
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
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
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
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
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
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
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
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
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
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
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER", raising=False)
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
