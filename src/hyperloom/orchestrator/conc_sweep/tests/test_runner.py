# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.conc_sweep import points, runner
from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
from hyperloom.orchestrator.actions.executors.grid._grid_runner import GridVariant, VariantResult
from hyperloom.orchestrator.state.shared_state import SharedState
from inference_suite.contracts.base_launch import SourceKind


def _state(session_dir: Path, **fields: Any) -> SharedState:
    state = SharedState()
    state.framework = "sglang"
    state.isl, state.osl, state.tp, state.conc = 1024, 1024, 8, 16
    state.baseline_tput = 100.0
    state.baseline_runtime_sec = 600.0
    state.cumulative_gain_validated = 12.0
    state.current_best = {"extra_server_args": "--enable-torch-compile", "extra_envs": {}}
    base_yaml = session_dir / "baseline.yaml"
    base_yaml.write_text("benchmark:\n  benchmark_script: bench.sh\n")
    state.baseline_config_path = str(base_yaml)
    for name, value in fields.items():
        setattr(state, name, value)
    state.save(session_dir)
    return state


def _base(kind: SourceKind, concs: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(source=SimpleNamespace(kind=kind), point=SimpleNamespace(source_concs=concs))


def _result(variant: GridVariant, *, status: str = "succeeded") -> VariantResult:
    conc = int(variant.extra_envs["CONC"])
    tput = conc * (130.0 if variant.name.startswith("optimized") else 100.0)
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status=status,
        output_throughput=tput if status == "succeeded" else None,
        total_token_throughput=tput if status == "succeeded" else None,
        duration_seconds=10.0,
    )


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    path = tmp_path / "session"
    (path / "reports").mkdir(parents=True)
    return path


@pytest.fixture
def grid(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The rounds the sweep runs and the rounds told to fail, with its base and a held-server framework patched in."""
    calls: list[dict[str, Any]] = []
    failing: set[str] = set()

    async def run_grid(*, grid: SimpleNamespace[GridVariant], launch: Any, **_: Any) -> list[VariantResult]:
        (variant,) = grid
        calls.append({"name": variant.name, "lifecycle": launch.server_lifecycle, "ready": launch.server_already_ready})
        return [_result(variant, status="failed" if variant.name in failing else "succeeded")]

    def materialize(src: Path, out_dir: Path, **_: Any) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "conc_sweep_base.with_envs.yaml"
        out.write_text(Path(src).read_text())
        return out

    monkeypatch.setattr(runner, "run_grid", run_grid)
    monkeypatch.setattr(runner, "prepare_grid_benchmark_config", materialize)
    monkeypatch.setattr(runner, "resolve_lifecycle_params", lambda _: {"eligible": True, "port": 9000})
    monkeypatch.setattr(runner, "teardown_variant_server", lambda _: [])
    frozen = SimpleNamespace(launch=_base(SourceKind.INFERENCEX, (4, 8, 32)))
    monkeypatch.setattr(runner, "frozen_base", lambda _: frozen)
    return SimpleNamespace(calls=calls, failing=failing)


def _summary(session_dir: Path) -> dict[str, Any]:
    return json.loads((session_dir / "reports" / "conc_sweep_summary.json").read_text())


def test_an_inferencex_base_sweeps_its_arm_with_the_session_conc():
    swept = points.sweep_points(_base(SourceKind.INFERENCEX, (4, 8, 32)), 16, [2, 4])

    assert swept == points.SweepPoints((4, 8, 16, 32), points.SOURCE_INFERENCEX)


def test_a_base_no_inferencex_arm_gave_sweeps_the_operators_points_or_none():
    user_file = _base(SourceKind.USER_FILE, ())

    assert points.sweep_points(user_file, 16, [64, 4]) == points.SweepPoints((4, 16, 64), points.SOURCE_OPERATOR)
    assert points.sweep_points(user_file, 16, []) is None


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"nodes": 2}, "multi_node"),
        ({"framework": "xdit"}, "non_serving_workload"),
        ({"framework": "llamacpp"}, "checkout_built_server"),
        ({"benchmark_mode": "agentx", "agentx_backend": "mlperf"}, "mlperf_backend"),
    ],
)
def test_a_workload_the_sweep_does_not_serve_is_declined(fields: dict[str, Any], reason: str):
    state = SharedState(framework="sglang")
    for name, value in fields.items():
        setattr(state, name, value)

    assert runner.decline_reason(state) == reason


def test_a_session_without_a_validated_gain_records_a_skip_and_runs_nothing(session_dir: Path, grid: SimpleNamespace):
    _state(session_dir, cumulative_gain_validated=0.0)

    payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert (payload["status"], payload["skip_reason"]) == ("skipped", "no_optimization")
    assert _summary(session_dir)["status"] == "skipped" and grid.calls == []


def test_each_arm_runs_the_session_conc_first_then_low_to_high_on_one_held_server(
    session_dir: Path, grid: SimpleNamespace
):
    _state(session_dir)

    payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert [call["name"] for call in grid.calls] == [
        *(f"baseline_conc{conc}" for conc in (16, 4, 8, 32)),
        *(f"optimized_conc{conc}" for conc in (16, 4, 8, 32)),
    ]
    arm = grid.calls[:4]
    assert [call["ready"] for call in arm] == [False, True, True, True]
    assert [call["lifecycle"]["cleanup"] for call in arm] == [False, False, False, True]
    assert payload["status"] == "succeeded" and payload["summary"]["successful_pairs"] == 4
    assert payload["concs_source"] == points.SOURCE_INFERENCEX
    rows = (session_dir / "reports" / "conc_sweep_raw.csv").read_text().splitlines()
    assert len(rows) == 1 + 8


def test_an_agentx_replay_boots_a_cold_server_for_every_point(session_dir: Path, grid: SimpleNamespace):
    _state(session_dir, benchmark_mode="agentx")

    asyncio.run(runner.run_conc_sweep(session_dir))

    assert grid.calls and all(call["lifecycle"] is None for call in grid.calls)


def test_a_point_that_failed_leaves_the_sweep_partial(session_dir: Path, grid: SimpleNamespace):
    _state(session_dir)
    grid.failing.add("optimized_conc32")

    payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert payload["status"] == "partial" and payload["summary"]["successful_pairs"] == 3


def test_an_optimized_overlay_that_cannot_load_skips_the_sweep(session_dir: Path, grid: SimpleNamespace):
    missing = session_dir / "overlay-gone"
    _state(session_dir, current_best={"extra_server_args": "", "extra_envs": {}, "final_overlay": str(missing)})

    payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert (payload["status"], payload["skip_reason"]) == ("skipped", "optimized_overlay_unavailable")
    assert grid.calls == []


def test_a_later_run_measures_only_the_points_not_yet_measured(session_dir: Path, grid: SimpleNamespace):
    _state(session_dir)
    grid.failing.add("optimized_conc32")
    asyncio.run(runner.run_conc_sweep(session_dir))
    grid.failing.clear()
    grid.calls.clear()

    payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert [call["name"] for call in grid.calls] == ["optimized_conc32"]
    assert payload["status"] == "succeeded"


def test_a_later_run_measures_again_the_arm_whose_launch_changed(session_dir: Path, grid: SimpleNamespace):
    _state(session_dir)
    asyncio.run(runner.run_conc_sweep(session_dir))
    grid.calls.clear()
    _state(session_dir, current_best={"extra_server_args": "--max-num-seqs 64", "extra_envs": {}})

    payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert [call["name"] for call in grid.calls] == [
        "optimized_conc16",
        "optimized_conc4",
        "optimized_conc8",
        "optimized_conc32",
    ]
    assert [point["conc"] for point in payload["baseline"]["points"]] == [4, 8, 16, 32]


def test_a_later_run_measures_again_the_arm_whose_launch_controls_changed(session_dir: Path, grid: SimpleNamespace):
    best = {"extra_server_args": "--enable-torch-compile", "extra_envs": {}}
    _state(session_dir, current_best=best)
    asyncio.run(runner.run_conc_sweep(session_dir))
    grid.calls.clear()
    _state(session_dir, current_best={**best, "remove_args": ["--max-running-requests"]})

    asyncio.run(runner.run_conc_sweep(session_dir))

    assert [call["name"] for call in grid.calls] == [
        "optimized_conc16",
        "optimized_conc4",
        "optimized_conc8",
        "optimized_conc32",
    ]


def test_a_cancel_keeps_the_points_measured_before_it(
    session_dir: Path, grid: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    _state(session_dir)
    scope = CancelScope()
    measured = runner._Sweep.record

    def record_then_cancel(self: Any, arm: Any, result: VariantResult) -> None:
        measured(self, arm, result)
        scope.cancel(reason="SIGTERM")

    monkeypatch.setattr(runner._Sweep, "record", record_then_cancel)

    with use_cancel_scope(scope):
        payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert [call["name"] for call in grid.calls] == ["baseline_conc16"]
    assert (payload["status"], payload["cancel_reason"]) == ("cancelled", "SIGTERM")
    assert [point["conc"] for point in _summary(session_dir)["baseline"]["points"]] == [16]


def _breakdown(session_dir: Path) -> dict[str, Any]:
    return json.loads((session_dir / "session_breakdown.json").read_text())


def test_the_session_breakdown_carries_the_sweep_as_each_point_lands(
    session_dir: Path, grid: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    _state(session_dir)
    (session_dir / "session_breakdown.json").write_text(json.dumps({"schema_version": "v6.1", "outcome": {}}))
    carried: list[int] = []
    measured = runner._Sweep.record

    def record_then_read(self: Any, arm: Any, result: VariantResult) -> None:
        measured(self, arm, result)
        carried.append(len(_breakdown(session_dir)["conc_sweep_summary"][arm.name]["points"]))

    monkeypatch.setattr(runner._Sweep, "record", record_then_read)

    asyncio.run(runner.run_conc_sweep(session_dir))

    breakdown = _breakdown(session_dir)
    assert carried == [1, 2, 3, 4, 1, 2, 3, 4]
    assert breakdown["conc_sweep_summary"] == _summary(session_dir)
    assert breakdown["conc_sweep_summary"]["status"] == "succeeded"
    assert (breakdown["schema_version"], breakdown["outcome"]) == ("v6.1", {})


def test_a_skipped_sweep_is_carried_in_the_breakdown_too(session_dir: Path, grid: SimpleNamespace):
    _state(session_dir, cumulative_gain_validated=0.0)
    (session_dir / "session_breakdown.json").write_text(json.dumps({"schema_version": "v6.1"}))

    asyncio.run(runner.run_conc_sweep(session_dir))

    carried = _breakdown(session_dir)["conc_sweep_summary"]
    assert (carried["status"], carried["skip_reason"]) == ("skipped", "no_optimization")


@pytest.mark.parametrize("content", ["{truncated", "[]"], ids=["unparseable", "not_an_object"])
def test_a_breakdown_the_sweep_cannot_carry_its_report_in_does_not_stop_the_sweep(
    session_dir: Path, grid: SimpleNamespace, content: str
):
    _state(session_dir)
    (session_dir / "session_breakdown.json").write_text(content)

    payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert payload["status"] == "succeeded" and len(grid.calls) == 8
    assert _summary(session_dir)["status"] == "succeeded"
    assert (session_dir / "session_breakdown.json").read_text() == content


def test_a_breakdown_that_cannot_be_replaced_does_not_stop_the_sweep(
    session_dir: Path, grid: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    from hyperloom.inference_optimizer.breakdown import exporter

    _state(session_dir)
    (session_dir / "session_breakdown.json").write_text(json.dumps({"schema_version": "v6.1"}))

    def refuse(path: Path, _text: str) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(exporter, "atomic_write_text", refuse)

    payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert payload["status"] == "succeeded" and _summary(session_dir)["status"] == "succeeded"
    assert _breakdown(session_dir) == {"schema_version": "v6.1"}


_REPORT_LOGGER = "hyperloom.orchestrator.conc_sweep.report"


def test_a_session_without_a_breakdown_still_gets_its_report(
    session_dir: Path, grid: SimpleNamespace, caplog: pytest.LogCaptureFixture
):
    _state(session_dir)

    with caplog.at_level(logging.WARNING, logger=_REPORT_LOGGER):
        asyncio.run(runner.run_conc_sweep(session_dir))

    assert _summary(session_dir)["status"] == "succeeded"
    assert not (session_dir / "session_breakdown.json").exists()
    missing = [r for r in caplog.records if r.name == _REPORT_LOGGER]
    assert missing and all("no breakdown at" in r.getMessage() and r.exc_info is None for r in missing)


def test_the_session_report_points_at_the_sweep(session_dir: Path, grid: SimpleNamespace):
    _state(session_dir)
    final = session_dir / "reports" / "final.json"
    final.write_text(json.dumps({"stop_reason": "cycle_end_done"}))

    asyncio.run(runner.run_conc_sweep(session_dir))

    pointer = json.loads(final.read_text())["conc_sweep_summary"]
    assert pointer["report_path"] == "reports/conc_sweep_summary.json" and pointer["status"] == "succeeded"
