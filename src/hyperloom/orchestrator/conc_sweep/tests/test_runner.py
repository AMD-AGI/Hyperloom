# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.conc_sweep import points, runner
from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
from hyperloom.orchestrator.actions.executors.grid._grid_runner import GridVariant, VariantResult
from hyperloom.orchestrator.state import base_launch as state_base_launch
from hyperloom.orchestrator.state.shared_state import SharedState
from inference_suite.contracts.base_launch import SourceKind, SourcePointLaunch


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


def _base(
    kind: SourceKind, concs: tuple[int, ...], launches: tuple[SourcePointLaunch, ...] = (), args: str = ""
) -> SimpleNamespace:
    source = SimpleNamespace(kind=kind, repo="InferenceX", commit="c22b4b2", entry="entry", locator="recipe.yaml:c16")
    image = SimpleNamespace(run="vllm/vllm-openai-rocm:v0.32.0", recipe=None)
    point = SimpleNamespace(source_concs=concs, source_launches=launches)
    return SimpleNamespace(
        source=source, point=point, image=image, args=tuple(args.split()), env={}, golden_al=None, cards=2
    )


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
        calls.append(
            {
                "name": variant.name,
                "lifecycle": launch.server_lifecycle,
                "ready": launch.server_already_ready,
                "args": variant.extra_server_args,
                "remove": variant.remove_args,
                "mode": variant.args_mode,
            }
        )
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
    monkeypatch.setattr(state_base_launch, "frozen_base", lambda _: frozen)
    return SimpleNamespace(calls=calls, failing=failing)


def _summary(session_dir: Path) -> dict[str, Any]:
    return json.loads((session_dir / "reports" / "conc_sweep_summary.json").read_text())


def test_an_inferencex_base_sweeps_its_arm_with_the_session_conc():
    swept = points.sweep_points(_base(SourceKind.INFERENCEX, (4, 8, 32)), 16, [])

    assert swept == points.SweepPoints((4, 8, 16, 32), points.SOURCE_INFERENCEX)


def test_the_operators_points_replace_the_recipes():
    swept = points.sweep_points(_base(SourceKind.INFERENCEX, (4, 8, 32)), 16, [2, 64])

    assert swept == points.SweepPoints((2, 16, 64), points.SOURCE_OPERATOR)


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


def test_the_session_report_points_at_the_sweep(session_dir: Path, grid: SimpleNamespace):
    _state(session_dir)
    final = session_dir / "reports" / "final.json"
    final.write_text(json.dumps({"stop_reason": "cycle_end_done"}))

    asyncio.run(runner.run_conc_sweep(session_dir))

    pointer = json.loads(final.read_text())["conc_sweep_summary"]
    assert pointer["report_path"] == "reports/conc_sweep_summary.json" and pointer["status"] == "succeeded"


def test_under_a_deadline_points_run_in_pairs_and_one_that_would_end_past_it_is_not_started(
    session_dir: Path, grid: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    """Each point is priced at the session's baseline round (600 s); a deadline 1500 s out fits two points, which run
    as a pair though the workload could hold a server per arm."""
    _state(session_dir)
    clock = [1000.0]
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    measure = runner.run_grid

    async def run_grid(**kwargs: Any) -> list[VariantResult]:
        clock[0] += 600.0
        return await measure(**kwargs)

    monkeypatch.setattr(runner, "run_grid", run_grid)

    payload = asyncio.run(runner.run_conc_sweep(session_dir, deadline_unix=2500.0))

    assert [call["name"] for call in grid.calls] == ["baseline_conc16", "optimized_conc16"]
    assert all(call["lifecycle"] is None for call in grid.calls)
    statuses = {arm: {p["conc"]: p["status"] for p in payload[arm]["points"]} for arm in ("baseline", "optimized")}
    expected = {4: "skipped_budget", 8: "skipped_budget", 16: "succeeded", 32: "skipped_budget"}
    assert statuses == {"baseline": expected, "optimized": expected}


def test_the_operators_points_are_swept_in_place_of_the_recipes(session_dir: Path, grid: SimpleNamespace):
    _state(session_dir)

    payload = asyncio.run(runner.run_conc_sweep(session_dir, concs=(2,)))

    assert (payload["concs"], payload["concs_source"]) == ([2, 16], points.SOURCE_OPERATOR)
    assert [call["name"] for call in grid.calls][:2] == ["baseline_conc16", "baseline_conc2"]


def test_the_report_names_the_hardware_the_image_and_the_recipe_and_who_wrote_it(
    session_dir: Path, grid: SimpleNamespace
):
    _state(session_dir, precision="fp8", gpu_type="mi355x", ep=1)

    payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert {k: payload[k] for k in ("framework", "precision", "gpu_type", "gpu_count", "tp", "ep")} == {
        "framework": "sglang",
        "precision": "fp8",
        "gpu_type": "mi355x",
        "gpu_count": 2,
        "tp": 8,
        "ep": 1,
    }
    assert payload["base_image"] == "vllm/vllm-openai-rocm:v0.32.0"
    assert payload["recipe"]["commit"] == "c22b4b2" and payload["recipe"]["locator"] == "recipe.yaml:c16"
    assert payload["pid"] == os.getpid() and payload["host"] and payload["updated_at"]
    header = (session_dir / "reports" / "conc_sweep_raw.csv").read_text().splitlines()[0].split(",")
    assert "e2e_norm_intvty_p50" in header


def _per_conc_base(monkeypatch: pytest.MonkeyPatch, base_args: str, by_conc: dict[int, str]) -> None:
    """A base whose arm launches each concurrency with its own args, the session's CONC (16) with ``base_args``."""
    launches = tuple(SourcePointLaunch(conc, tuple(args.split()), {}) for conc, args in by_conc.items())
    frozen = SimpleNamespace(launch=_base(SourceKind.INFERENCEX, tuple(by_conc), launches, base_args))
    monkeypatch.setattr(runner, "frozen_base", lambda _: frozen)
    monkeypatch.setattr(state_base_launch, "frozen_base", lambda _: frozen)


_SLOTS = {4: "--max-running-requests 8", 16: "--max-running-requests 32", 32: "--max-running-requests 64"}


def test_each_point_launches_the_recipes_own_args_at_its_concurrency_under_the_optimization(
    session_dir: Path, grid: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    _state(session_dir)
    _per_conc_base(monkeypatch, _SLOTS[16], _SLOTS)

    asyncio.run(runner.run_conc_sweep(session_dir))

    args = {call["name"]: call["args"] for call in grid.calls}
    assert args["baseline_conc16"] == "" and args["optimized_conc16"] == "--enable-torch-compile"
    assert args["baseline_conc4"] == "--max-running-requests 8"
    assert args["optimized_conc32"] == "--max-running-requests 64 --enable-torch-compile"


def test_what_the_optimization_removes_stays_removed_at_every_concurrency(
    session_dir: Path, grid: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    best = {"extra_server_args": "--enable-torch-compile", "extra_envs": {}, "remove_args": ["--max-running-requests"]}
    _state(session_dir, current_best=best)
    _per_conc_base(monkeypatch, _SLOTS[16], _SLOTS)

    asyncio.run(runner.run_conc_sweep(session_dir))

    optimized = {call["name"]: call for call in grid.calls if call["name"].startswith("optimized")}
    assert optimized["optimized_conc32"]["args"] == "--enable-torch-compile"
    assert "--max-running-requests" in optimized["optimized_conc32"]["remove"]


def test_a_held_server_is_booted_again_where_the_next_point_launches_otherwise(
    session_dir: Path, grid: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    _state(session_dir)
    _per_conc_base(monkeypatch, _SLOTS[16], {4: "--a 1", 8: "--a 1", 16: _SLOTS[16], 32: "--a 2"})

    asyncio.run(runner.run_conc_sweep(session_dir))

    baseline = [call for call in grid.calls if call["name"].startswith("baseline")]
    assert [call["name"] for call in baseline] == [f"baseline_conc{c}" for c in (16, 4, 8, 32)]
    assert [call["ready"] for call in baseline] == [False, False, True, False]


def test_a_replacing_argv_keeps_its_changes_and_takes_the_recipes_values_at_each_concurrency(
    session_dir: Path, grid: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    """The optimization replaced the argv: it dropped ``--a`` and added ``--enable-x``, and left the cap as it was."""
    _state(session_dir, current_best={"extra_server_args": "--max-num-seqs 64 --enable-x", "args_mode": "replace"})
    by_conc = {16: "--max-num-seqs 64 --a 1", 32: "--max-num-seqs 256 --a 1"}
    _per_conc_base(monkeypatch, by_conc[16], by_conc)

    asyncio.run(runner.run_conc_sweep(session_dir))

    calls = {call["name"]: call for call in grid.calls}
    assert (calls["optimized_conc16"]["args"], calls["optimized_conc16"]["mode"]) == (
        "--max-num-seqs 64 --enable-x",
        "replace",
    )
    assert (calls["optimized_conc32"]["args"], calls["optimized_conc32"]["mode"]) == (
        "--max-num-seqs 256 --enable-x",
        "replace",
    )
    assert (calls["baseline_conc32"]["args"], calls["baseline_conc32"]["mode"]) == ("--max-num-seqs 256", "append")


def test_a_later_run_measures_again_a_report_an_earlier_schema_wrote(session_dir: Path, grid: SimpleNamespace):
    """A report before per-concurrency launches measured every point on the session's own launch."""
    _state(session_dir)
    asyncio.run(runner.run_conc_sweep(session_dir))
    summary = session_dir / "reports" / "conc_sweep_summary.json"
    summary.write_text(json.dumps({**json.loads(summary.read_text()), "schema_version": "2.0"}))
    grid.calls.clear()

    asyncio.run(runner.run_conc_sweep(session_dir))

    assert len(grid.calls) == 8


def test_cold_points_run_in_pairs_so_a_deadline_leaves_pairs_measured(
    session_dir: Path, grid: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    """Each point is priced at 600 s and boots its own server; a deadline 1300 s out fits one concurrency's pair."""
    _state(session_dir, benchmark_mode="agentx")
    clock = [1000.0]
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    measure = runner.run_grid

    async def run_grid(**kwargs: Any) -> list[VariantResult]:
        clock[0] += 600.0
        return await measure(**kwargs)

    monkeypatch.setattr(runner, "run_grid", run_grid)

    payload = asyncio.run(runner.run_conc_sweep(session_dir, deadline_unix=2300.0))

    assert [call["name"] for call in grid.calls] == ["baseline_conc16", "optimized_conc16"]
    measured = {
        arm: [p["conc"] for p in payload[arm]["points"] if p["status"] == "succeeded"]
        for arm in ("baseline", "optimized")
    }
    assert measured == {"baseline": [16], "optimized": [16]}
    skipped = {
        arm: sorted(p["conc"] for p in payload[arm]["points"] if p["status"] == "skipped_budget")
        for arm in ("baseline", "optimized")
    }
    assert skipped == {"baseline": [4, 8, 32], "optimized": [4, 8, 32]}


def test_a_later_run_over_other_points_keeps_the_points_both_runs_name(session_dir: Path, grid: SimpleNamespace):
    _state(session_dir)
    asyncio.run(runner.run_conc_sweep(session_dir, concs=(2,)))
    grid.calls.clear()

    asyncio.run(runner.run_conc_sweep(session_dir))

    assert not any(call["name"].endswith("_conc16") for call in grid.calls)
    assert {call["name"].rsplit("conc", 1)[1] for call in grid.calls} == {"4", "8", "32"}
