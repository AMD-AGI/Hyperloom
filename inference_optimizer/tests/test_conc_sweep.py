"""Tests for the post-optimization concurrency sweep runner.

Pins the contract for ``orchestrator.conc_sweep.run_conc_sweep``:

* Skip cases short-circuit before ``run_grid`` is invoked and return
  a stable envelope with ``status="skipped"`` + ``skip_reason``.
* Both ``extra_server_args`` and ``extra_envs`` independently count as
  "optimized" — either being non-empty triggers the run.
* Two-arm grid (baseline + optimized) × N concs is built correctly,
  including CONC / ISL / OSL / NUM_PROMPTS env overrides.
* Aggregation pairs by CONC and produces correct speedup / median.
* JSON + CSV outputs land under ``reports/`` and final.json gets
  a ``conc_sweep_summary`` pointer merged in.
* Aggregation handles failed/missing optimized points gracefully.
"""

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
    _build_comparison,
    _build_grid,
    _has_optimization,
    format_summary_line,
    run_conc_sweep,
)
from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """Minimal YAML file that exists on disk so ``run_conc_sweep``
    passes the ``baseline_config_missing`` skip gate."""
    p = tmp_path / "baseline.yaml"
    p.write_text("benchmark:\n  benchmark_script: bench.sh\n")
    return p


# ---------------------------------------------------------------------------
# _has_optimization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _build_grid
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _build_comparison
# ---------------------------------------------------------------------------


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
    """One arm fails on conc=4 → that pair is counted failed,
    others still pair up."""
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


# ---------------------------------------------------------------------------
# Skip paths
# ---------------------------------------------------------------------------


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
    # No reports written on skip.
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
    # The empty-list gate fires AFTER has_opt check; if has_opt passes,
    # empty list should be the reason. With our default fixture state
    # current_best is non-empty so we expect "empty_conc_list".
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


# ---------------------------------------------------------------------------
# Happy path with mocked run_grid
# ---------------------------------------------------------------------------


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


def test_run_conc_sweep_happy_path_writes_reports_and_patches_final_json(
    session_dir: Path, baseline_yaml: Path,
):
    state = _make_state(baseline_config_path=str(baseline_yaml))
    # Seed an existing final.json so the pointer-patch path is exercised.
    final_json_path = session_dir / "reports" / "final.json"
    final_json_path.write_text(
        json.dumps({"stop_reason": "target_reached", "cumulative_gain": 12.3}),
    )

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

    # Also bypass materialize_config_with_envs which would otherwise
    # try to PyYAML-parse the toy fixture.
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
    # All three pairs should produce a 1.30x speedup (130 / 100 with
    # the same shape factor in numerator + denominator).
    speedups = [r["speedup"] for r in payload["comparison"]]
    for s in speedups:
        assert s == pytest.approx(1.30)
    assert payload["summary"]["successful_pairs"] == 3
    assert payload["summary"]["best_speedup"] == pytest.approx(1.30)
    assert payload["summary"]["median_speedup"] == pytest.approx(1.30)

    # Reports landed.
    summary_path = session_dir / "reports" / "conc_sweep_summary.json"
    csv_path = session_dir / "reports" / "conc_sweep_raw.csv"
    assert summary_path.exists()
    assert csv_path.exists()

    disk = json.loads(summary_path.read_text())
    assert disk["status"] == "succeeded"
    assert disk["summary"]["successful_pairs"] == 3

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 6  # 3 baseline + 3 optimized
    assert {r["arm"] for r in rows} == {"baseline", "optimized"}

    # final.json got the pointer.
    final_disk = json.loads(final_json_path.read_text())
    assert final_disk["stop_reason"] == "target_reached"
    assert "conc_sweep_summary" in final_disk
    assert final_disk["conc_sweep_summary"]["status"] == "succeeded"
    assert final_disk["conc_sweep_summary"]["summary"]["best_speedup"] == \
        pytest.approx(1.30)
    # Pointer is a session-relative path.
    assert final_disk["conc_sweep_summary"]["report_path"] == \
        "reports/conc_sweep_summary.json"


def test_run_conc_sweep_optimized_oom_yields_failed_pair(
    session_dir: Path, baseline_yaml: Path,
):
    """An optimized variant crashing should yield a failed pair but
    not abort the overall summary (other concs still pair up)."""
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
    # Status overall is still "succeeded" because at least one pair worked.
    assert payload["status"] == "succeeded"


def test_run_conc_sweep_args_only_optimization_triggers_run(
    session_dir: Path, baseline_yaml: Path,
):
    """An optimized config with only ``extra_server_args`` (no envs)
    should still trigger the sweep — A/B/C combinations all run, only D
    (both empty) skips."""
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
    mock_run.assert_called_once()
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

    mock_run.assert_called_once()
    assert payload["optimized"]["extra_envs"] == {"SGLANG_MOE_ENABLE": "1"}


def test_run_conc_sweep_missing_final_json_does_not_raise(
    session_dir: Path, baseline_yaml: Path,
):
    """Pointer-patch must no-op cleanly when final.json is absent
    (e.g. close-sequence skipped)."""
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
    # final.json was never created.
    assert not final_json_path.exists()


# ---------------------------------------------------------------------------
# format_summary_line
# ---------------------------------------------------------------------------


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
