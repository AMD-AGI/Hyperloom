# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests pinning what each sweep round launches with and the axis its rungs are ranked on."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_OUTPUT
from hyperloom.orchestrator.actions.executors.grid._grid_runner import (
    GridRecipe,
    GridVariant,
    VariantResult,
    _build_variant_yaml,
)
from hyperloom.orchestrator.conc_sweep import runner
from hyperloom.orchestrator.state.shared_state import SharedState

_BASE_ARGS = "--trust-remote-code --max-running-requests 64"
_CHILD = (
    "import json, os, sys; print(json.dumps({'argv': sys.argv[1:], 'env': {k: os.environ.get(k) for k in "
    "('SGLANG_SWEEP_TEST', 'GEAK_SWEEP_OVERLAY_MARKER', 'NUM_PROMPTS')}}))"
)


def _state(session_dir: Path, base: dict[str, Any], **fields: Any) -> None:
    state = SharedState()
    state.framework = "sglang"
    state.isl, state.osl, state.tp, state.conc = 1024, 1024, 8, 1
    state.conc_sweep_concs = [2]
    state.baseline_runtime_sec = 600.0
    state.cumulative_gain_validated = 12.0
    state.current_best = {"extra_server_args": "--enable-torch-compile", "extra_envs": {}}
    base_yaml = session_dir / "baseline.yaml"
    base_yaml.write_text(yaml.safe_dump(base))
    state.baseline_config_path = str(base_yaml)
    for name, value in fields.items():
        setattr(state, name, value)
    state.save(session_dir)


def _sglang_base(envs: dict[str, str]) -> dict[str, Any]:
    return {"benchmark": {"framework": "sglang", "envs": envs}}


def _result(variant: GridVariant) -> VariantResult:
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="succeeded",
        output_throughput=100.0,
        total_token_throughput=100.0,
        duration_seconds=10.0,
    )


def _launch(recipe: GridRecipe, variant: GridVariant, out_dir: Path) -> dict[str, Any]:
    """Materialize the round's config and start a process with its server argv and environment."""
    config = _build_variant_yaml(recipe, variant, output_subdir=out_dir)
    envs = {str(k): str(v) for k, v in yaml.safe_load(config.read_text())["benchmark"]["envs"].items()}
    command = shlex.join([sys.executable, "-B", "-c", _CHILD]) + " " + envs.get("EXTRA_SGLANG_ARGS", "")
    child = subprocess.run(
        ["bash", "-c", "exec " + command],
        env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1", **envs},
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(child.stdout)


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    path = tmp_path / "session"
    (path / "reports").mkdir(parents=True)
    return path


@pytest.fixture
def rounds(session_dir: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """What each round's process was started with, by round name, and the server dirs torn down."""
    children: dict[str, dict[str, Any]] = {}
    teardowns: list[Path] = []

    async def run_grid(*, grid: list[GridVariant], recipe: GridRecipe, **_: Any) -> list[VariantResult]:
        (variant,) = grid
        children[variant.name] = _launch(recipe, variant, session_dir / "children" / variant.name)
        return [_result(variant)]

    def materialize(src: Path, out_dir: Path, **_: Any) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "conc_sweep_base.with_envs.yaml"
        out.write_text(Path(src).read_text())
        return out

    def teardown(pid_dir: Path) -> list[int]:
        teardowns.append(pid_dir)
        return []

    monkeypatch.setattr(runner, "run_grid", run_grid)
    monkeypatch.setattr(runner, "prepare_grid_benchmark_config", materialize)
    monkeypatch.setattr(runner, "teardown_variant_server", teardown)
    monkeypatch.setattr(runner, "frozen_base", lambda _: None)
    return SimpleNamespace(children=children, teardowns=teardowns)


def _hold_server(monkeypatch: pytest.MonkeyPatch, held: bool) -> None:
    monkeypatch.setattr(runner, "resolve_lifecycle_params", lambda _: {"eligible": held, "port": 9000})


@pytest.mark.parametrize("persistent_server", [False, True])
@pytest.mark.parametrize(
    ("current_best", "expected_args", "expected_env"),
    [
        (
            {"extra_server_args": "--trust-remote-code --cuda-graph-max-bs 64", "args_mode": "replace"},
            "--trust-remote-code --cuda-graph-max-bs 64",
            "stock",
        ),
        ({"extra_server_args": "", "args_mode": "replace"}, "", "stock"),
        ({"remove_args": ["--max-running-requests"]}, "--trust-remote-code", "stock"),
        ({"unset_envs": ["SGLANG_SWEEP_TEST"]}, _BASE_ARGS, None),
        (
            {"unset_envs": ["SGLANG_SWEEP_TEST"], "extra_envs": {"SGLANG_SWEEP_TEST": "accepted 'literal'"}},
            _BASE_ARGS,
            "accepted 'literal'",
        ),
        ({"extra_server_args": "--disable-radix-cache"}, f"{_BASE_ARGS} --disable-radix-cache", "stock"),
    ],
    ids=["complete-args", "empty-replacement", "remove-args", "unset-env", "unset-then-assign", "legacy-delta"],
)
def test_conc_sweep_preserves_retained_launch_controls(
    session_dir: Path,
    rounds: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    persistent_server: bool,
    current_best: dict[str, Any],
    expected_args: str,
    expected_env: str | None,
):
    """The optimized arm's server starts as the retained stack would; the baseline arm's starts stock."""
    _hold_server(monkeypatch, persistent_server)
    _state(
        session_dir,
        _sglang_base({"EXTRA_SGLANG_ARGS": _BASE_ARGS, "SGLANG_SWEEP_TEST": "stock"}),
        current_best=current_best,
    )

    payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert payload["status"] == "succeeded"
    assert sorted(rounds.children) == [f"{arm}_conc{conc}" for arm in ("baseline", "optimized") for conc in (1, 2)]
    for name, child in rounds.children.items():
        optimized = name.startswith("optimized_")
        assert child["argv"] == shlex.split(expected_args if optimized else _BASE_ARGS)
        assert child["env"]["SGLANG_SWEEP_TEST"] == (expected_env if optimized else "stock")


@pytest.mark.parametrize("extra_args", ["", "--max-running-requests 64"])
@pytest.mark.parametrize("persistent_server", [False, True])
def test_conc_sweep_loads_retained_overlay_only_in_optimized_arm(
    session_dir: Path,
    rounds: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: str,
    persistent_server: bool,
):
    """An overlay-only session still sweeps, and only the optimized arm's server imports the overlay."""
    overlay = session_dir / "retained_overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("import os\nos.environ['GEAK_SWEEP_OVERLAY_MARKER'] = 'candidate'\n")
    _hold_server(monkeypatch, persistent_server)
    _state(
        session_dir,
        _sglang_base({}),
        current_best={"extra_server_args": extra_args, "extra_envs": {}, "final_overlay": str(overlay)},
    )

    payload = asyncio.run(runner.run_conc_sweep(session_dir))

    assert payload["status"] == "succeeded"
    assert {name: child["env"]["GEAK_SWEEP_OVERLAY_MARKER"] for name, child in rounds.children.items()} == {
        "baseline_conc1": None,
        "baseline_conc2": None,
        "optimized_conc1": "candidate",
        "optimized_conc2": "candidate",
    }
    assert len(rounds.teardowns) == (2 if persistent_server else 0)


@pytest.mark.parametrize(("isl", "osl", "per_conc"), [(1024, 8192, 3), (1024, 1024, 5)])
def test_each_rung_sends_the_request_window_the_session_measures_with(
    session_dir: Path, rounds: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, isl: int, osl: int, per_conc: int
):
    """The session's requests per unit of concurrency for its ISL/OSL is every rung's too."""
    _hold_server(monkeypatch, False)
    _state(session_dir, _sglang_base({}), isl=isl, osl=osl, conc=4, conc_sweep_concs=[16])

    asyncio.run(runner.run_conc_sweep(session_dir))

    assert {name: child["env"]["NUM_PROMPTS"] for name, child in rounds.children.items()} == {
        f"{arm}_conc{conc}": str(conc * per_conc) for arm in ("baseline", "optimized") for conc in (4, 16)
    }


class TestTheSweepGradesOnTheAxisTheSessionKeepsOn:
    """A curve drawn on one axis beside promotions decided on another is two answers to one question."""

    @pytest.fixture(autouse=True)
    def _no_ambient_grading(self, monkeypatch: pytest.MonkeyPatch):
        for name in (
            "HYPERLOOM_PERF_METRIC",
            "HYPERLOOM_PERF_NOISE_PCT",
            "HYPERLOOM_AGENTX",
            "HYPERLOOM_BENCHMARK_CONFIG",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_the_recorded_axis_beats_the_environment(self, monkeypatch: pytest.MonkeyPatch):
        """A resume is a new process, and the shell it landed in is not evidence about the axis."""
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", GRADED_OUTPUT)
        state = SharedState(benchmark_mode="agentx", grading={"objective": GRADED_INTVTY, "noise_pct": 3.5})

        assert runner._graded_on(state) == (GRADED_INTVTY, 3.5)

    def test_a_session_with_nothing_recorded_falls_back_to_its_mode(self):
        assert runner._graded_on(SharedState(benchmark_mode="agentx")) == (GRADED_INTVTY, None)
        assert runner._graded_on(SharedState(benchmark_mode="synthetic")) == (GRADED_OUTPUT, None)

    def test_a_scriptable_framework_is_carved_out(self):
        """An image framework reports no interactivity axis, so ranking every rung on it fails the whole sweep."""
        assert runner._graded_on(SharedState(benchmark_mode="agentx", framework="xdit")) == (GRADED_OUTPUT, None)

    def test_an_agentx_sweep_ranks_its_rungs_on_interactivity(
        self, session_dir: Path, rounds: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ):
        """Its optimized arm traded output throughput for interactivity, which an output ranking would call a loss."""

        async def run_grid(*, grid: list[GridVariant], **_: Any) -> list[VariantResult]:
            (variant,) = grid
            optimized = variant.name.startswith("optimized_")
            result = _result(variant)
            result.intvty_p90 = 30.0 if optimized else 20.0
            result.output_throughput = 100.0 if optimized else 200.0
            return [result]

        monkeypatch.setattr(runner, "run_grid", run_grid)
        _hold_server(monkeypatch, False)
        _state(session_dir, _sglang_base({}), benchmark_mode="agentx")

        payload = asyncio.run(runner.run_conc_sweep(session_dir))

        assert payload["summary"]["metric"] == GRADED_INTVTY
        assert [row["speedup"] for row in payload["comparison"]] == [1.5, 1.5]
        assert payload["summary"]["best_speedup"] == 1.5
