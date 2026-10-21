# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local backend against a stubbed ``run_grid``: the parameter mapping of each kind of test (design 4.4.7)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.common.env import is_truthy
from hyperloom.contracts.measurement import AccuracyValue, AttachmentRef
from hyperloom.measure import local
from hyperloom.measure.local import SUPPORTED_KINDS, LocalMeasurementBackend, TestCallRefused
from hyperloom.measure.repro import read_verdict
from hyperloom.measure.socket import RoundPlan, SessionFacts, TestCall
from hyperloom.orchestrator.actions.executors import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    VariantResult,
)
from hyperloom.orchestrator.actions.executors._server_lifecycle import REUSE_PORT_DEFAULT


def _point(directory: Path, name: str, *, conc: int = 4, run_eval: object = "false", **counts: object) -> Path:
    envs: dict[str, object] = {"TP": 1, "CONC": conc, **counts}
    if run_eval is not None:
        envs["RUN_EVAL"] = run_eval
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": envs}}), encoding="utf-8")
    return path


def _call(tmp_path: Path, *points: Path, **kwargs: Any) -> TestCall:
    defaults: dict[str, Any] = {
        "points": points or (_point(tmp_path, "p"),),
        "output_dir": tmp_path / "out",
        "timeout_sec": 900,
        "cwd": str(tmp_path),
        "magpie_python": "",
        "task_id": "t-1",
    }
    return TestCall(**{**defaults, **kwargs})


class _Grid:
    """A ``run_grid`` stand-in: records each call and answers from a script keyed by the variant's name."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.calls: list[dict[str, Any]] = []
        self.answers: dict[str, VariantResult] = {}
        self.env_seen: list[dict[str, str | None]] = []
        self.configs: list[dict[str, Any]] = []  # what each call's point held while it ran: a copy is gone after

    def result(self, name: str, **kwargs: Any) -> VariantResult:
        workspace = self.tmp_path / "ws" / name
        workspace.mkdir(parents=True, exist_ok=True)
        fields: dict[str, Any] = {
            "name": name,
            "extra_server_args": "",
            "extra_envs": {},
            "status": "succeeded",
            "returncode": 0,
            "workspace": str(workspace),
        }
        return VariantResult(**{**fields, **kwargs})

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        self.calls.append(kwargs)
        self.configs.append(yaml.safe_load(Path(kwargs["base_yaml_path"]).read_text()))
        self.env_seen.append(
            {
                key: os.environ.get(key)
                for key in (
                    "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR",
                    "INFERENCEX_PATH",
                    "HL_TEST_FLAG",
                    "PROFILE_EXTRA_BODY",
                )
            }
        )
        name = kwargs["grid"][0].name
        return [self.answers.get(name) or self.result(name)]


@pytest.fixture(autouse=True)
def magpie_backend(monkeypatch) -> None:
    """The benchmark backend is read from the process environment; a test that does not pick one runs on Magpie."""
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "magpie")


@pytest.fixture
def grid(monkeypatch, tmp_path) -> _Grid:
    fake = _Grid(tmp_path)
    monkeypatch.setattr(local, "run_grid", fake)
    return fake


class _Server:
    """Stands in for the lifecycle helpers: a server that is eligible and records its teardowns."""

    def __init__(self, monkeypatch, *, eligible: bool = True) -> None:
        self.teardowns: list[dict[str, Any]] = []
        monkeypatch.setattr(
            local,
            "resolve_lifecycle_params",
            lambda path: {"eligible": eligible, "port": 8123, "framework": "sglang", "reason": "not eligible"},
        )
        monkeypatch.setattr(local, "teardown_lifecycle_server", lambda **kw: self.teardowns.append(kw))


@pytest.fixture
def server(monkeypatch) -> _Server:
    return _Server(monkeypatch)


def _run(call: TestCall):
    return LocalMeasurementBackend().run_test(call)


def _own_port(config: dict[str, Any]) -> int:
    """The port a round's point carries, taken out of it: what is left is the point the test was asked to run."""
    port = int(config["benchmark"]["envs"].pop("PORT"))
    assert port != REUSE_PORT_DEFAULT, "the default is the port every round without one shares"
    return port


# --- single-point test --------------------------------------------------------------------------------------------


def test_a_single_point_test_is_one_run_grid_call_with_one_variant(grid, tmp_path):
    call = _call(tmp_path, result_dir="/res", session_deadline_sec=123.0, serving_lease="lease")

    outcome = _run(call)

    (kwargs,) = grid.calls
    assert [v.name for v in kwargs["grid"]] == ["test"]
    assert kwargs["output_root"] == tmp_path / "out"
    assert (kwargs["result_dir"], kwargs["session_deadline_sec"], kwargs["serving_lease"]) == ("/res", 123.0, "lease")
    assert kwargs["magpie_python"] is None
    assert "server_lifecycle" not in kwargs
    (config,) = grid.configs
    _own_port(config)
    assert config == yaml.safe_load(call.points[0].read_text()), "the point is run as it is, with a port of its own"
    assert [(r.point, r.phase, r.returncode) for r in outcome.rounds] == [(0, "measure", 0)]
    assert (outcome.returncode, outcome.failure_class, outcome.request_id) == (0, None, "t-1")
    assert outcome.sandbox.backend == "local"


@pytest.mark.parametrize(("plan", "expected"), [(None, None), (0, False), (1, True)])
def test_the_warmup_plan_reaches_run_grid_as_it_was_stated(grid, tmp_path, plan, expected):
    _run(_call(tmp_path, rounds=RoundPlan(warmup_rounds=plan)))

    assert grid.calls[0]["warmup_before_measure"] is expected


def test_a_discarded_warmup_round_is_reported_as_its_own_round(grid, tmp_path):
    grid.answers["test"] = grid.result("test", nonfatal_warnings=["run_grid_warmup_discarded_first"])

    outcome = _run(_call(tmp_path))

    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("measure", 0)]
    assert outcome.rounds[0].output_dir == outcome.rounds[1].output_dir.parent / "warmup_round"


def test_the_warnings_run_grid_wrote_ride_on_the_first_round_of_the_point_once(grid, tmp_path):
    warnings = ["warmup_round_seed:1", "run_grid_warmup_discarded_first", "warmup_round_seed:1"]
    grid.answers["test"] = grid.result("test", nonfatal_warnings=warnings)

    outcome = _run(_call(tmp_path))

    assert [r.nonfatal_warnings for r in outcome.rounds] == [tuple(warnings), ()]
    assert [w for r in outcome.rounds for w in r.nonfatal_warnings] == warnings


def test_a_point_that_run_grid_did_not_warn_about_carries_none(grid, tmp_path):
    outcome = _run(_call(tmp_path))
    assert [r.nonfatal_warnings for r in outcome.rounds] == [()]


def test_an_accuracy_test_reports_the_accuracy_round_with_the_eval_exit_code(grid, tmp_path):
    result = grid.result("test", returncode=0)
    (Path(result.workspace) / "eval_returncode").write_text("3\n", encoding="utf-8")
    grid.answers["test"] = result
    point = _point(tmp_path, "acc", run_eval="true")

    outcome = _run(_call(tmp_path, point, accuracy=True, kind="accuracy"))

    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("measure", 0), ("accuracy", 3)]
    assert outcome.returncode == 3


def _lm_eval_results(directory: Path, score: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    document = {"results": {"gsm8k": {"exact_match,flexible-extract": score}}}
    (directory / "results_2026-10-14T10-00-00.json").write_text(json.dumps(document), encoding="utf-8")


def test_the_accuracy_round_carries_the_measured_rounds_scores_not_the_warmups(grid, tmp_path):
    result = grid.result("test", nonfatal_warnings=["run_grid_warmup_discarded_first"])
    workspace = Path(result.workspace)
    _lm_eval_results(workspace / "eval_output", 0.64)
    _lm_eval_results(workspace.parent / "warmup_round" / "bench" / "eval_output", 0.10)
    grid.answers["test"] = result
    point = _point(tmp_path, "acc", run_eval="true")

    outcome = _run(_call(tmp_path, point, accuracy=True, kind="accuracy"))

    by_phase = {r.phase: r.accuracy for r in outcome.rounds}
    assert by_phase["accuracy"] == (AccuracyValue("gsm8k", "exact_match,flexible-extract", 0.64),)
    assert by_phase["warmup"] == by_phase["measure"] == ()


def test_the_accuracy_round_carries_the_version_lm_eval_reported(grid, tmp_path):
    result = grid.result("test")
    workspace = Path(result.workspace)
    workspace.joinpath("eval_output").mkdir(parents=True, exist_ok=True)
    document = {"results": {"gsm8k": {"exact_match,flexible-extract": 0.64}}, "versions": {"gsm8k": 3.0}}
    document["lm_eval_version"] = "0.4.13"
    (workspace / "eval_output" / "results_2026-10-14T10-00-00.json").write_text(json.dumps(document), encoding="utf-8")
    grid.answers["test"] = result
    point = _point(tmp_path, "acc", run_eval="true")

    outcome = _run(_call(tmp_path, point, accuracy=True, kind="accuracy"))

    (value,) = [r.accuracy for r in outcome.rounds if r.phase == "accuracy"][0]
    assert value.version == "lm_eval=0.4.13;gsm8k=3"


def test_an_accuracy_round_with_only_a_warmup_evaluation_carries_no_score(grid, tmp_path):
    result = grid.result("test", nonfatal_warnings=["run_grid_warmup_discarded_first"])
    _lm_eval_results(Path(result.workspace).parent / "warmup_round" / "eval_output", 0.10)
    grid.answers["test"] = result
    point = _point(tmp_path, "acc", run_eval="true")

    outcome = _run(_call(tmp_path, point, accuracy=True, kind="accuracy"))

    assert [r.accuracy for r in outcome.rounds if r.phase == "accuracy"] == [()]


def test_a_test_that_did_not_ask_for_accuracy_reports_no_accuracy_round(grid, tmp_path):
    outcome = _run(_call(tmp_path))

    assert [r.phase for r in outcome.rounds] == ["measure"]


# --- a graded test evaluates once, in its warmup round ---------------------------------------------------------------


def _graded_call(tmp_path: Path, **kwargs: Any) -> TestCall:
    """A graded single-point test whose warmup is planned: the case that used to evaluate twice."""
    point = _point(tmp_path, "graded", run_eval="true")
    return _call(
        tmp_path, point, **{"accuracy": True, "kind": "accuracy", "rounds": RoundPlan(warmup_rounds=1), **kwargs}
    )


def _evaluates(kwargs: dict[str, Any], config: dict[str, Any]) -> bool:
    """Whether a round evaluates, as ``run_grid`` would build it: the round's extra envs laid over the point's own."""
    envs = {**config["benchmark"]["envs"], **(kwargs.get("base_extra_envs") or {})}
    return is_truthy(envs["RUN_EVAL"], default=True)


def _graded_answers(grid: _Grid, score: float = 0.64) -> VariantResult:
    """The scripted answers of a graded test: the evaluation's score sits in the warmup round's directory."""
    warmup = grid.result("warmup", output_throughput=1111.0)
    _lm_eval_results(Path(warmup.workspace) / "eval_output", score)
    grid.answers.update(warmup=warmup, test=grid.result("test"))
    return warmup


def test_a_graded_test_runs_one_evaluation_and_it_is_in_the_warmup_round(grid, server, tmp_path):
    _graded_answers(grid)

    _run(_graded_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup", "test"]
    assert [_evaluates(c, config) for c, config in zip(grid.calls, grid.configs)] == [True, False]


def test_the_two_rounds_of_a_graded_test_share_one_server_that_the_measured_round_reclaims(grid, server, tmp_path):
    _graded_answers(grid)

    _run(_graded_call(tmp_path))

    warmup, measured = grid.calls
    assert warmup["server_lifecycle"] == {
        "cleanup": False,
        "pid_dir": warmup["server_lifecycle"]["pid_dir"],
        "port": 8123,
    }
    assert measured["server_lifecycle"] == {**warmup["server_lifecycle"], "cleanup": True}
    assert "server_already_ready" not in warmup and measured["server_already_ready"] is True
    assert warmup["warmup_before_measure"] is measured["warmup_before_measure"] is False, "run_grid does not warm up"
    assert "lifecycle_boot_only" not in warmup, "the warmup round is a whole round, it measures and evaluates"
    assert measured["base_extra_envs"] == {"RUN_EVAL": "false"} and "base_extra_envs" not in warmup
    assert len(server.teardowns) == 1, "and whatever is still up is reclaimed when the test ends"


def test_the_accuracy_of_a_graded_test_is_the_one_evaluation_of_the_warmup_round(grid, server, tmp_path):
    warmup = _graded_answers(grid, 0.64)

    outcome = _run(_graded_call(tmp_path))

    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("accuracy", 0), ("measure", 0)]
    warmed, accuracy, measured = outcome.rounds
    assert warmed.output_dir == accuracy.output_dir == Path(warmup.workspace)
    assert measured.output_dir == Path(grid.answers["test"].workspace) != warmed.output_dir
    assert accuracy.accuracy == (AccuracyValue("gsm8k", "exact_match,flexible-extract", 0.64),)
    assert warmed.accuracy == measured.accuracy == ()
    assert outcome.returncode == 0


def test_the_warmup_round_of_a_graded_test_says_what_run_grid_says_of_a_discarded_one(grid, server, tmp_path):
    _graded_answers(grid)
    grid.answers["warmup"].nonfatal_warnings.append("harvested_leaked_artifact:x")
    grid.answers["test"].nonfatal_warnings.append("served_slowly")

    outcome = _run(_graded_call(tmp_path))

    assert [w for r in outcome.rounds for w in r.nonfatal_warnings] == [
        "run_grid_warmup_discarded_first",
        "warmup_round_tput:1111.0",
        "harvested_leaked_artifact:x",
        "served_slowly",
    ]


def test_the_eval_exit_code_of_a_graded_test_is_the_one_the_warmup_round_recorded(grid, server, tmp_path):
    warmup = _graded_answers(grid)
    (Path(warmup.workspace) / "eval_returncode").write_text("3\n", encoding="utf-8")

    outcome = _run(_graded_call(tmp_path))

    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("accuracy", 3), ("measure", 0)]
    assert outcome.returncode == 3


def test_a_warmup_that_fails_ends_a_graded_test_with_nothing_measured_and_no_accuracy(grid, server, tmp_path):
    grid.answers["warmup"] = grid.result("warmup", status="failed", returncode=7, error="boom", error_class="x")
    _lm_eval_results(Path(grid.answers["warmup"].workspace) / "eval_output", 0.64)

    outcome = _run(_graded_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup"]
    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 7)] and outcome.returncode == 7
    assert len(server.teardowns) == 1


def test_a_measured_round_that_fails_keeps_the_evaluation_the_warmup_made(grid, server, tmp_path):
    _graded_answers(grid)
    grid.answers["test"] = grid.result("test", status="failed", returncode=5, error="boom", error_class="x")

    outcome = _run(_graded_call(tmp_path))

    assert [(r.phase, r.returncode) for r in outcome.rounds] == [("warmup", 0), ("accuracy", 0), ("measure", 5)]
    assert outcome.returncode == 5


def test_a_graded_test_the_run_stops_during_its_warmup_reports_the_stop_and_measures_nothing(grid, server, tmp_path):
    grid.answers["warmup"] = grid.result(
        "warmup", status="failed", returncode=None, error_class="session_time_exhausted", error="out of time"
    )

    outcome = _run(_graded_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup"]
    assert outcome.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE


def test_a_graded_test_decodes_after_its_measured_round(grid, server, tmp_path, monkeypatch):
    _graded_answers(grid)
    point = _point(tmp_path, "graded", run_eval="true", ISL=8, OSL=4)
    monkeypatch.setattr(local, "run_pass", lambda *a, **k: ())
    monkeypatch.setattr(local, "write_outputs", lambda *a, **k: None)
    monkeypatch.setattr(local, "vocab_size_of", lambda model: 1000)
    grid.answers["decode_server"] = grid.result("decode_server", note="server_lifecycle_boot_only")

    _run(_call(tmp_path, point, accuracy=True, kind="accuracy", decode_pass=True, rounds=RoundPlan(warmup_rounds=1)))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup", "test", "decode_server"]


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"rounds": RoundPlan(warmup_rounds=0)}, "no warmup is planned, so the one round evaluates"),
        ({"rounds": RoundPlan(warmup_rounds=None)}, "pytest's own default is no warmup"),
    ],
)
def test_a_graded_test_without_a_warmup_evaluates_in_its_one_round_as_before(grid, server, tmp_path, kwargs, why):
    result = grid.result("test")
    _lm_eval_results(Path(result.workspace) / "eval_output", 0.64)
    grid.answers["test"] = result
    point = _point(tmp_path, "graded", run_eval="true")

    outcome = _run(_call(tmp_path, point, accuracy=True, kind="accuracy", **kwargs))

    assert [c["grid"][0].name for c in grid.calls] == ["test"], why
    assert "server_lifecycle" not in grid.calls[0] and "base_extra_envs" not in grid.calls[0]
    assert [r.phase for r in outcome.rounds] == ["measure", "accuracy"]
    assert outcome.rounds[1].accuracy[0].score == 0.64


def test_a_graded_test_on_a_framework_that_cannot_hold_a_server_keeps_run_grids_warmup(monkeypatch, grid, tmp_path):
    _Server(monkeypatch, eligible=False)

    _run(_graded_call(tmp_path))

    (kwargs,) = grid.calls
    assert kwargs["grid"][0].name == "test" and kwargs["warmup_before_measure"] is True
    assert "server_lifecycle" not in kwargs and "base_extra_envs" not in kwargs


def test_a_test_that_is_not_graded_keeps_run_grids_warmup_and_never_holds_a_server(grid, server, tmp_path):
    _run(_call(tmp_path, rounds=RoundPlan(warmup_rounds=1)))

    (kwargs,) = grid.calls
    assert kwargs["grid"][0].name == "test" and kwargs["warmup_before_measure"] is True
    assert "server_lifecycle" not in kwargs and "base_extra_envs" not in kwargs and server.teardowns == []


def test_the_default_warmup_of_run_grid_decides_for_a_graded_test_that_states_none(grid, server, tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
    _graded_answers(grid)

    _run(_graded_call(tmp_path, rounds=RoundPlan(warmup_rounds=None)))

    assert [c["grid"][0].name for c in grid.calls] == ["warmup", "test"]


# --- refusals -----------------------------------------------------------------------------------------------------


def test_a_point_that_leaves_run_eval_unset_is_refused_because_unset_means_on(grid, tmp_path):
    point = _point(tmp_path, "unset", run_eval=None)

    with pytest.raises(TestCallRefused, match="does not set RUN_EVAL"):
        _run(_call(tmp_path, point))
    assert grid.calls == []


@pytest.mark.parametrize(("yaml_value", "accuracy"), [("true", False), ("false", True), (True, False), ("0", True)])
def test_a_point_whose_run_eval_contradicts_the_call_is_refused(grid, tmp_path, yaml_value, accuracy):
    point = _point(tmp_path, "mismatch", run_eval=yaml_value)

    with pytest.raises(TestCallRefused, match=f"asks for accuracy={accuracy}"):
        _run(_call(tmp_path, point, accuracy=accuracy))
    assert grid.calls == []


@pytest.mark.parametrize("kind", ["sweep", "bogus"])
def test_a_kind_the_local_backend_does_not_run_is_refused(grid, tmp_path, kind):
    with pytest.raises(TestCallRefused, match="runs kinds"):
        _run(_call(tmp_path, kind=kind))


def test_an_accuracy_kind_without_accuracy_is_refused(grid, tmp_path):
    with pytest.raises(TestCallRefused, match="must ask for accuracy"):
        _run(_call(tmp_path, kind="accuracy"))


def test_a_single_point_test_with_two_points_is_refused(grid, tmp_path):
    with pytest.raises(TestCallRefused, match="exactly one point"):
        _run(_call(tmp_path, _point(tmp_path, "a"), _point(tmp_path, "b")))


def test_env_overrides_may_not_set_a_blocked_name(grid, tmp_path):
    with pytest.raises(TestCallRefused, match="PYTHONPATH"):
        _run(_call(tmp_path, env_overrides={"PYTHONPATH": "/x"}))


@pytest.mark.parametrize("levels", [(4, 4), (4, 0)])
def test_a_sweep_needs_a_distinct_positive_level_on_every_point(grid, server, tmp_path, levels):
    points = [_point(tmp_path, f"c{i}", conc=level) for i, level in enumerate(levels)]

    with pytest.raises(TestCallRefused, match="names its own CONC"):
        _run(_call(tmp_path, *points, prep="conc_sweep"))


# --- the process environment --------------------------------------------------------------------------------------


def test_the_session_is_a_fresh_empty_directory_and_the_environment_is_put_back(grid, tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", "/the/real/session")
    monkeypatch.delenv("HL_TEST_FLAG", raising=False)
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    facts = SessionFacts(reference_unset_envs=("A_PIN",), inferencex_path="/ix")

    _run(_call(tmp_path, env_overrides={"HL_TEST_FLAG": "on"}, unset_envs=("B_PIN",), session_facts=facts))

    seen = grid.env_seen[0]
    assert seen["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] != "/the/real/session"
    assert (seen["HL_TEST_FLAG"], seen["INFERENCEX_PATH"]) == ("on", "/ix")
    assert grid.calls[0]["grid"][0].unset_envs == ["A_PIN", "B_PIN"]
    assert os.environ["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] == "/the/real/session"
    assert "HL_TEST_FLAG" not in os.environ and "INFERENCEX_PATH" not in os.environ
    assert not Path(seen["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] or "x").exists()


def test_the_environment_is_put_back_when_the_test_raises(grid, tmp_path, monkeypatch):
    monkeypatch.delenv("HL_TEST_FLAG", raising=False)
    grid.answers["test"] = grid.result("test", status="failed", error_class="magpie_timeout")

    with pytest.raises(subprocess.TimeoutExpired):
        _run(_call(tmp_path, env_overrides={"HL_TEST_FLAG": "on"}))

    assert "HL_TEST_FLAG" not in os.environ


def test_the_reference_launch_the_old_flow_keeps_reads_empty_in_the_test_session(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors._workload_envs import resolve_reference_launch

    real = tmp_path / "real"
    (real).mkdir()
    (real / "state.json").write_text('{"reference_server_args": "--from-the-old-flow"}', encoding="utf-8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(real))
    seen: list[tuple] = []

    async def probe(**kwargs):
        seen.append(resolve_reference_launch())
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", probe)

    _run(_call(tmp_path))

    assert seen == [("", {}, {})]


def test_the_per_round_cap_and_the_leak_scope_come_from_the_call(grid, tmp_path, monkeypatch):
    captured: dict[str, str | None] = {}

    async def look(**kwargs):
        captured.update(
            {
                k: os.environ.get(k)
                for k in ("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "INFERENCE_OPTIMIZER_LEAK_ROOTS")
            }
        )
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", look)

    _run(_call(tmp_path, timeout_sec=321))

    assert captured == {
        "INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC": "321",
        "INFERENCE_OPTIMIZER_LEAK_ROOTS": str(tmp_path),
    }


def test_two_tests_in_one_process_do_not_overlap(monkeypatch, tmp_path):
    active, peak = [0], [0]
    gate = threading.Lock()

    async def slow(**kwargs):
        with gate:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        await __import__("asyncio").sleep(0.05)
        with gate:
            active[0] -= 1
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", slow)
    threads = [threading.Thread(target=_run, args=(_call(tmp_path),)) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak[0] == 1


# --- one arm of a concurrency sweep -------------------------------------------------------------------------------


def _sweep_call(tmp_path: Path, *levels: int, **kwargs: Any) -> TestCall:
    points = [_point(tmp_path, f"c{level}", conc=level) for level in levels]
    return _call(tmp_path, *points, prep="conc_sweep", **kwargs)


def _lifecycle(call: dict[str, Any]) -> dict[str, Any] | None:
    return call.get("server_lifecycle")


#: What a boot round adds to the point on Magpie: no evaluation, and one request of the client round Magpie runs after
#: its server phase whatever the phase says. Not the phase itself: Magpie sets that, and a variable it does not know
#: would make its reuse check refuse every round that measures (``test_local_reuse_identity``).
BOOT_ENVS = {"RUN_EVAL": "false", "NUM_PROMPTS": "1"}


def _envs_handed_to_magpie(tmp_path: Path, call: dict[str, Any]) -> dict[str, Any]:
    """The ``benchmark.envs`` ``run_grid`` makes of one recorded call: its own layering of the point and ``base_extra_envs``."""
    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

    candidate = _build_variant_yaml(
        Path(call["base_yaml_path"]),
        "",
        call["grid"][0],
        output_subdir=tmp_path / "candidates" / str(len(list((tmp_path / "candidates").glob("*")))),
        base_extra_envs=call.get("base_extra_envs"),
    )
    return yaml.safe_load(candidate.read_text())["benchmark"]["envs"]


def test_a_sweep_boots_the_server_once_and_measures_the_levels_from_high_to_low_on_it(grid, server, tmp_path):
    outcome = _run(_sweep_call(tmp_path, 2, 8, 4))

    names = [c["grid"][0].name for c in grid.calls]
    assert names == ["c8", "c8", "c4", "c2"]
    boot, *reuse = grid.calls
    assert boot["lifecycle_boot_only"] is True and boot["base_extra_envs"] == BOOT_ENVS
    assert all("base_extra_envs" not in c for c in reuse), "the rounds that measure carry nothing of the boot"
    assert _lifecycle(boot)["cleanup"] is False
    assert all(c["server_already_ready"] is True for c in reuse)
    assert [_lifecycle(c)["cleanup"] for c in reuse] == [False, False, True]
    assert {(_lifecycle(c)["pid_dir"], _lifecycle(c)["port"]) for c in grid.calls} == {
        (str(tmp_path / "out" / "server_arm"), 8123)
    }
    assert all(c["warmup_before_measure"] is False for c in grid.calls)
    assert [r.point for r in outcome.rounds] == [1, 0 + 2, 0]  # c8, c4, c2 by their positions in the call
    assert len(server.teardowns) == 1


def _graded_sweep(tmp_path: Path, *levels: int, run_eval: str = "true") -> TestCall:
    """A sweep whose points hold the request counts the point generator gives them (an accuracy sweep by default)."""
    points = [
        _point(tmp_path, f"g{n}", conc=n, run_eval=run_eval, NUM_PROMPTS=n * 10, NUM_WARMUPS=min(n, 8)) for n in levels
    ]
    return _call(tmp_path, *points, prep="conc_sweep", accuracy=run_eval == "true")


def test_every_boot_of_a_sweep_arm_does_not_evaluate_and_asks_one_request_and_no_measured_round_changes(
    grid, server, tmp_path
):
    grid.answers["c8"] = grid.result("c8", status="failed", returncode=1, error_class="no_benchmark_workspace")
    call = _graded_sweep(tmp_path, 8, 4, 2)
    points = [Path(p).read_text() for p in call.points]

    outcome = _run(call)

    boots = [c for c in grid.calls if c.get("lifecycle_boot_only")]
    measured = [c for c in grid.calls if not c.get("lifecycle_boot_only")]
    assert [c["grid"][0].name for c in boots] == ["c8", "c4"], "the failed boot and the one it stepped down to"
    assert [c["grid"][0].name for c in measured] == ["c4", "c2"]
    assert all(c["base_extra_envs"] == BOOT_ENVS for c in boots)
    assert all("base_extra_envs" not in c for c in measured)
    assert [Path(p).read_text() for p in call.points] == points, "the points are the ones the call was accepted on"
    for index, boot in enumerate(boots):
        level = (8, 4)[index]
        envs = _envs_handed_to_magpie(tmp_path, boot)
        assert (envs["RUN_EVAL"], envs["NUM_PROMPTS"]) == ("false", "1"), f"the boot at {level}"
        assert "MAGPIE_RUN_PHASE" not in envs and envs["CONC"] == level
        assert envs["NUM_WARMUPS"] == min(level, 8), "a variable outside the reuse check is the point's own"
    for level, round_ in zip((4, 2), measured, strict=True):
        envs = _envs_handed_to_magpie(tmp_path, round_)
        assert (envs["RUN_EVAL"], envs["NUM_PROMPTS"], envs["NUM_WARMUPS"]) == ("true", level * 10, min(level, 8))
        assert "MAGPIE_RUN_PHASE" not in envs
    assert [r.phase for r in outcome.rounds if r.point == 1] == ["measure", "accuracy"]
    assert outcome.returncode == 0


def test_a_sweep_that_does_not_run_accuracy_boots_the_same_way(grid, server, tmp_path):
    _run(_graded_sweep(tmp_path, 8, 4, run_eval="false"))

    boot, *measured = grid.calls
    assert boot["base_extra_envs"] == BOOT_ENVS and all("base_extra_envs" not in c for c in measured)


def test_a_boot_does_not_pin_what_the_point_pins_for_a_graded_round(grid, server, tmp_path):
    """The boot's overrides live in its round's environment; the call's own validation reads the point."""
    call = _graded_sweep(tmp_path, 8, 4)

    _run(call)  # an accuracy sweep whose points say RUN_EVAL=true is accepted, and its boots run

    assert all(yaml.safe_load(Path(p).read_text())["benchmark"]["envs"]["RUN_EVAL"] == "true" for p in call.points)
    assert grid.calls[0]["base_yaml_path"] == call.points[0]


@pytest.mark.parametrize(
    ("backend", "boot_envs"), [("magpie", {}), ("bypass", {"MAGPIE_RUN_PHASE": "server"})], ids=["magpie", "bypass"]
)
def test_an_agentx_boot_carries_only_what_its_backend_says(grid, server, tmp_path, monkeypatch, backend, boot_envs):
    """An AgentX point has no evaluation round and no prompt count: its boot round keeps only the backend's phase."""
    point = _agentx_point(tmp_path, monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", backend)

    _run(_call(tmp_path, point, prep="conc_sweep"))

    boot, *measured = grid.calls
    assert boot["base_extra_envs"] == boot_envs
    assert all("base_extra_envs" not in c for c in measured)


def test_no_call_of_a_sweep_runs_without_the_shared_server(grid, server, tmp_path):
    _run(_sweep_call(tmp_path, 8, 4, 2))

    assert all(_lifecycle(c) is not None for c in grid.calls)


def test_a_round_the_lifecycle_holds_runs_from_the_point_itself_on_the_lifecycles_port(grid, server, tmp_path):
    call = _sweep_call(tmp_path, 8, 4)

    _run(call)

    assert [c["base_yaml_path"] for c in grid.calls] == [call.points[0], call.points[0], call.points[1]]
    assert all("PORT" not in config["benchmark"]["envs"] for config in grid.configs)
    assert {_lifecycle(c)["port"] for c in grid.calls} == {8123}


def test_a_sweep_that_restarts_per_level_gives_every_level_a_port_of_its_own(monkeypatch, grid, tmp_path):
    _Server(monkeypatch, eligible=False)
    call = _sweep_call(tmp_path, 8, 4)

    _run(call)

    ports = [_own_port(config) for config in grid.configs]
    assert len(ports) == 2 and all(0 < port < 65536 for port in ports)
    assert all(config == yaml.safe_load(point.read_text()) for config, point in zip(grid.configs, call.points[:2]))


def test_a_boot_that_fails_steps_down_one_level_and_that_level_is_measured_on_the_new_server(grid, server, tmp_path):
    grid.answers["c8"] = grid.result(
        "c8", status="failed", returncode=1, error_class="no_benchmark_workspace", error="oom"
    )

    outcome = _run(_sweep_call(tmp_path, 8, 4, 2))

    assert [c["grid"][0].name for c in grid.calls] == ["c8", "c4", "c4", "c2"]
    assert [r.point for r in outcome.rounds] == [1, 2]
    assert "stepping down" in outcome.stderr
    assert len(server.teardowns) == 2  # after the failed boot, and at the end


def test_when_no_level_boots_the_ladder_falls_back_to_one_server_per_level(grid, server, tmp_path):
    for level in (8, 4):
        grid.answers[f"c{level}"] = grid.result(f"c{level}", status="failed", returncode=1, error_class="x")

    outcome = _run(_sweep_call(tmp_path, 8, 4))

    restart_calls = grid.calls[2:]
    assert [c["grid"][0].name for c in restart_calls] == ["c8", "c4"]
    assert all(_lifecycle(c) is None for c in restart_calls)
    assert "every boot attempt failed" in outcome.stderr
    assert outcome.returncode == 1


def test_a_framework_that_cannot_hold_a_server_restarts_per_level_and_says_so(monkeypatch, grid, tmp_path):
    _Server(monkeypatch, eligible=False)

    outcome = _run(_sweep_call(tmp_path, 8, 4))

    assert [c["grid"][0].name for c in grid.calls] == ["c8", "c4"]
    assert all(_lifecycle(c) is None for c in grid.calls)
    assert "server restarted per level: not eligible" in outcome.stderr


def test_a_sweep_stops_when_the_run_stops_it_and_reports_the_stop_code(grid, server, tmp_path):
    grid.answers["c4"] = grid.result(
        "c4", status="skipped", returncode=None, error_class="session_time_exhausted", error="out of time"
    )

    outcome = _run(_sweep_call(tmp_path, 8, 4, 2))

    assert [c["grid"][0].name for c in grid.calls] == ["c8", "c8", "c4"]
    assert outcome.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE
    assert outcome.failure_class is None and outcome.failure_reason == "session_time_exhausted"
    assert len(server.teardowns) == 1


def test_a_cancel_during_the_boot_ends_the_sweep_without_stepping_down(grid, server, tmp_path):
    grid.answers["c8"] = grid.result("c8", status="skipped", returncode=None, error_class="orchestrator_cancelled")

    outcome = _run(_sweep_call(tmp_path, 8, 4))

    assert [c["grid"][0].name for c in grid.calls] == ["c8"]
    assert outcome.returncode == ORCHESTRATOR_CANCELLED_RETURNCODE


# --- a compiled environment ---------------------------------------------------------------------------------------

OLD_SOURCE = "def f():\n    return 1\n"
NEW_SOURCE = "def f():\n    return 10\n"


@pytest.fixture
def installed_framework(tmp_path):
    """A real interpreter with a framework 'sglang' installed in it, as the image has one."""
    import subprocess
    import sys

    from hyperloom.measure.tests._recipes import write_tree

    venv = tmp_path / "image-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, capture_output=True)
    site = next((venv / "lib").glob("python*/site-packages"))
    package = write_tree(site / "sglang", {"__init__.py": "", "srt/__init__.py": "", "srt/a.py": OLD_SOURCE})
    return venv / "bin" / "python", package


def _compile(tmp_path, installed_framework):
    from hyperloom.contracts.recipe import RecipeDelta
    from hyperloom.measure.compile import ComponentLayout, LocalBlobStore, PackageBase, compile_environment
    from hyperloom.measure.tests._recipes import IMAGE, PREFIX, git_patch, put_patch, recipe_with

    _, package = installed_framework
    blobs = LocalBlobStore(tmp_path / "blobs")
    patch = git_patch(tmp_path / "pw", {f"{PREFIX}/srt/a.py": OLD_SOURCE}, {f"{PREFIX}/srt/a.py": NEW_SOURCE})
    recipe = recipe_with((put_patch(blobs, "a.patch", patch),))
    base = PackageBase("sglang", package, "sglang", "0.5.4")

    def locate(layout: ComponentLayout) -> PackageBase:
        return base

    return compile_environment(recipe, RecipeDelta(), blobs=blobs, active_image_digest=IMAGE, locator=locate)


def _point_with_python(tmp_path, python) -> Path:
    path = tmp_path / "patched.yaml"
    envs = {"TP": 1, "CONC": 4, "RUN_EVAL": "false", "HYPERLOOM_FRAMEWORK_PYTHON": str(python)}
    path.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": envs}}), encoding="utf-8")
    return path


def test_a_patched_test_measures_the_rebuilt_framework_that_the_compile_produced(
    monkeypatch, tmp_path, installed_framework
):
    import subprocess

    from hyperloom.measure.tests._recipes import IMAGE

    python, package = installed_framework
    compiled = _compile(tmp_path, installed_framework)
    imported: dict[str, str] = {}

    async def look(**kwargs):
        envs = yaml.safe_load(Path(kwargs["base_yaml_path"]).read_text())["benchmark"]["envs"]
        done = subprocess.run(
            [str(python), "-c", "import sglang.srt.a as a; print(a.f())"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={"PYTHONPATH": envs["PYTHONPATH"], "PATH": ""},
        )
        imported.update(value=done.stdout.strip(), path=envs["PYTHONPATH"].split(":")[0])
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", look)

    outcome = _run(_call(tmp_path, _point_with_python(tmp_path, python), environment=compiled))

    assert imported["value"] == "10"  # the server's interpreter imported the patched file, not the image's
    assert (package / "srt" / "a.py").read_text() == OLD_SOURCE  # and the image's was not written
    assert not Path(imported["path"]).exists()  # the farm is removed when the test ends
    assert outcome.sandbox.image_digest == IMAGE


def _agentx_point(tmp_path, monkeypatch, **launch_envs: str) -> Path:
    """The point the toolbox renders for an AgentX session whose recipe launches with ``launch_envs``."""
    from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
    from hyperloom.inference_optimizer.cli.benchmark_config import AgentXWorkload, write_agentx_benchmark
    from hyperloom.orchestrator.actions.executors import materialize_benchmark_config

    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.setenv(BENCHMARK_CONFIG_ENV, "")
    source = tmp_path / "agentx.yaml"
    workload = AgentXWorkload(framework="sglang", model="Qwen/Qwen3-8B", precision="fp8", tp=2, ep=1, conc=4)
    write_agentx_benchmark(source, workload)
    return materialize_benchmark_config(source, tmp_path / "agentx", extra_envs={**launch_envs, "RUN_EVAL": "false"})


def _magpie_launch(monkeypatch, tmp_path, python) -> dict[str, Any]:
    """What ``run_grid`` hands Magpie for the point, and what the server it launches then imports from ``a.f()``."""
    import subprocess

    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

    launched: dict[str, Any] = {}

    async def look(**kwargs):
        candidate = _build_variant_yaml(
            Path(kwargs["base_yaml_path"]), "", kwargs["grid"][0], output_subdir=tmp_path / "candidate"
        )
        launch = yaml.safe_load(candidate.read_text())["benchmark"]["agentx"]["launch_overrides"]
        done = subprocess.run(
            [launch.get("executable") or "python3", "-c", "import sglang.srt.a as a; print(a.f())"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={"PYTHONPATH": launch["env"]["PYTHONPATH"], "PATH": str(Path(python).parent)},
        )
        launched.update(value=done.stdout.strip(), launch=launch)
        return [VariantResult(name="test", extra_server_args="", extra_envs={}, status="succeeded", returncode=0)]

    monkeypatch.setattr(local, "run_grid", look)
    return launched


def test_a_patched_agentx_test_hands_the_rebuilt_framework_to_the_server_magpie_launches(
    monkeypatch, tmp_path, installed_framework
):
    """``benchmark.envs`` never reaches an AgentX server: its environment and interpreter are the launch's."""
    python, package = installed_framework
    launched = _magpie_launch(monkeypatch, tmp_path, python)
    point = _agentx_point(tmp_path, monkeypatch, HYPERLOOM_FRAMEWORK_PYTHON=str(python))

    _run(_call(tmp_path, point, environment=_compile(tmp_path, installed_framework)))

    launch = launched["launch"]
    assert launched["value"] == "10"  # the interpreter Magpie launches imported the patched file, not the image's
    assert launch["executable"] == str(python) and "HYPERLOOM_FRAMEWORK_PYTHON" not in launch["env"]
    assert launch["env"]["PYTHONPATH"].split(":")[0] in "".join(launch["source_files"]), "rebuilt files are evidence"
    assert (package / "srt" / "a.py").read_text() == OLD_SOURCE
    assert "PYTHONPATH" not in yaml.safe_load(point.read_text())["benchmark"]["envs"]


def test_a_patched_agentx_test_that_names_no_interpreter_is_rebuilt_for_the_python3_magpie_launches(
    monkeypatch, tmp_path, installed_framework
):
    """Magpie starts the server with ``python3`` unless told otherwise; the farm is built and checked against that."""
    python, _ = installed_framework
    monkeypatch.setenv("PATH", f"{python.parent}{os.pathsep}{os.environ['PATH']}")
    launched = _magpie_launch(monkeypatch, tmp_path, python)
    point = _agentx_point(tmp_path, monkeypatch)

    _run(_call(tmp_path, point, environment=_compile(tmp_path, installed_framework)))

    assert launched["value"] == "10" and "executable" not in launched["launch"]


def test_a_test_on_the_image_as_it_is_runs_the_point_unchanged(grid, tmp_path, installed_framework):
    from hyperloom.contracts.recipe import RecipeDelta
    from hyperloom.measure.compile import LocalBlobStore, compile_environment
    from hyperloom.measure.tests._recipes import IMAGE, recipe_with

    compiled = compile_environment(
        recipe_with(), RecipeDelta(), blobs=LocalBlobStore(tmp_path / "blobs"), active_image_digest=IMAGE
    )
    call = _call(tmp_path, environment=compiled)

    _run(call)

    _own_port(grid.configs[0])
    assert compiled.is_base and grid.configs[0] == yaml.safe_load(call.points[0].read_text())


def test_an_environment_that_cannot_be_rebuilt_is_refused_with_the_reason(grid, tmp_path, installed_framework):
    compiled = _compile(tmp_path, installed_framework)
    point = _point_with_python(tmp_path, "/no/such/python")

    with pytest.raises(TestCallRefused, match="cannot be rebuilt here"):
        _run(_call(tmp_path, point, environment=compiled))
    assert grid.calls == []


# --- profile test -------------------------------------------------------------------------------------------------


def _profile_point(directory: Path, *, profile: object = "1", enabled: bool = True, body: str | None = None) -> Path:
    envs: dict[str, object] = {"TP": 1, "CONC": 4, "RUN_EVAL": "false"}
    if body is not None:
        envs["PROFILE_EXTRA_BODY"] = body
    if profile is not None:
        envs["PROFILE"] = profile
    config = {"benchmark": {"framework": "sglang", "envs": envs, "profiler": {"torch_profiler": {"enabled": enabled}}}}
    path = directory / "profile.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _with_trace(grid: _Grid, name: str = "test", where: str = "torch_trace") -> VariantResult:
    result = grid.result(name)
    trace = Path(result.workspace) / where / "rank0.pt.trace.json.gz"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_bytes(b"trace")
    grid.answers[name] = result
    return result


def test_a_profile_test_is_one_profiled_round_with_no_warmup_before_it(grid, tmp_path):
    _with_trace(grid)
    call = _call(tmp_path, _profile_point(tmp_path), kind="profile", rounds=RoundPlan(warmup_rounds=1))

    outcome = _run(call)

    (kwargs,) = grid.calls
    assert [v.name for v in kwargs["grid"]] == ["test"]
    assert kwargs["warmup_before_measure"] is False, "a profiled round is not a statistic: no warmup before it"
    assert outcome.returncode == 0 and [r.phase for r in outcome.rounds] == ["measure"]


def test_a_profile_test_puts_the_points_start_profile_body_in_the_process_environment(grid, tmp_path, monkeypatch):
    """The benchmark client reads PROFILE_EXTRA_BODY from its own environment; the point's envs never reach it."""
    monkeypatch.delenv("PROFILE_EXTRA_BODY", raising=False)
    _with_trace(grid)
    body = '{"start_step": 320, "num_steps": 128, "profile_by_stage": false}'

    _run(_call(tmp_path, _profile_point(tmp_path, body=body), kind="profile"))

    assert [seen["PROFILE_EXTRA_BODY"] for seen in grid.env_seen] == [body]
    assert "PROFILE_EXTRA_BODY" not in os.environ, "the test's environment is put back afterwards"


def test_the_traces_of_a_profile_test_move_into_the_output_profile_directory(grid, tmp_path):
    result = _with_trace(grid)
    sidecar = Path(result.workspace).parent / "capture_traces" / "graph.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}")

    _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    profile = tmp_path / "out" / "profile"
    assert (profile / "torch_trace" / "rank0.pt.trace.json.gz").read_bytes() == b"trace"
    assert (profile / "capture_traces" / "graph.json").is_file(), "the capture sidecars keep their directory name"
    assert not (Path(result.workspace) / "torch_trace").exists(), "the bundle must carry each trace once"


def test_a_profile_test_that_left_no_trace_does_not_end_as_a_success(grid, tmp_path):
    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert outcome.returncode != 0 and "profile_no_trace" in outcome.stderr
    assert not (tmp_path / "out" / "profile").exists()


def test_a_failed_profile_round_moves_no_trace(grid, tmp_path):
    result = _with_trace(grid)
    grid.answers["test"] = VariantResult(**{**result.__dict__, "status": "failed", "returncode": 1})

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile"))

    assert outcome.returncode == 1 and "profile_no_trace" not in outcome.stderr
    assert (Path(result.workspace) / "torch_trace").exists()


@pytest.mark.parametrize(
    ("point", "kwargs", "match"),
    [
        (dict(profile=None), {}, "does not turn the torch profiler on"),
        (dict(profile="0"), {}, "does not turn the torch profiler on"),
        (dict(enabled=False), {}, "does not turn the torch profiler on"),
        ({}, dict(accuracy=True), "does not run accuracy"),
        ({}, dict(prep="conc_sweep"), "not a sweep"),
    ],
)
def test_a_profile_call_that_would_not_profile_or_asks_for_more_is_refused(grid, tmp_path, point, kwargs, match):
    run_eval = "true" if kwargs.get("accuracy") else "false"
    path = _profile_point(tmp_path, **point)
    if run_eval == "true":
        config = yaml.safe_load(path.read_text())
        config["benchmark"]["envs"]["RUN_EVAL"] = "true"
        path.write_text(yaml.safe_dump(config))

    with pytest.raises(TestCallRefused, match=match):
        _run(_call(tmp_path, path, kind="profile", **kwargs))
    assert grid.calls == []


def test_an_agentx_profile_test_is_one_profiled_round_that_turns_the_profiler_on_in_the_yaml(
    grid, tmp_path, monkeypatch
):
    from hyperloom.orchestrator.actions.executors.magpie_agentx import enable_profile

    point = _agentx_point(tmp_path, monkeypatch)
    enable_profile(point)
    capture = "0123456789abcdef0123456789abcdef"
    _with_trace(grid, where=f"torch_trace/{capture}")

    outcome = _run(_call(tmp_path, point, kind="profile", rounds=RoundPlan(warmup_rounds=1)))

    (kwargs,) = grid.calls
    _own_port(grid.configs[0])
    assert kwargs["warmup_before_measure"] is False and grid.configs[0] == yaml.safe_load(point.read_text())
    assert outcome.returncode == 0
    assert (tmp_path / "out" / "profile" / "torch_trace" / capture / "rank0.pt.trace.json.gz").is_file()


def test_an_agentx_profile_call_whose_point_leaves_the_profiler_off_is_refused(grid, tmp_path, monkeypatch):
    point = _agentx_point(tmp_path, monkeypatch)

    with pytest.raises(TestCallRefused, match="does not turn the torch profiler on"):
        _run(_call(tmp_path, point, kind="profile"))
    assert grid.calls == []


def test_a_plain_test_does_not_gather_traces(grid, tmp_path):
    _with_trace(grid)

    _run(_call(tmp_path))

    assert not (tmp_path / "out" / "profile").exists()


class _Machine:
    """One machine's ports: ``run_grid`` as it is for a round whose server is up, listening on the port the round's own
    config names in ``benchmark.envs.PORT`` (the default when it names none).

    The InferenceX script reads it from the environment it is run with; Magpie's AgentX recipe reads it into the
    ``--port`` of the server's argv (``agentx_recipe._port``, at the pinned ref) and refuses ``PORT`` in
    ``launch_overrides.env`` as a protocol variable, so the AgentX server's port can come from nowhere else.

    The listeners stay open to the end of the test, as the servers of tests running at the same time do; a round whose
    port is taken fails the way the real one did: its server dies.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.ports: list[int] = []
        self.launches: list[dict[str, Any]] = []
        self.listeners: list[socket.socket] = []

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

        variant = kwargs["grid"][0]
        slot = self.tmp_path / "slots" / str(len(self.ports))
        launched = _build_variant_yaml(Path(kwargs["base_yaml_path"]), "", variant, output_subdir=slot)
        self.launches.append(benchmark := yaml.safe_load(launched.read_text())["benchmark"])
        self.ports.append(port := int((benchmark.get("envs") or {}).get("PORT", REUSE_PORT_DEFAULT)))
        workspace = slot / "benchmark_stub"
        (workspace / "torch_trace").mkdir(parents=True)
        (workspace / "torch_trace" / "rank0.pt.trace.json.gz").write_bytes(b"trace")
        listener = socket.socket()
        try:
            listener.bind(("0.0.0.0", port))
            listener.listen()
        except OSError as exc:
            listener.close()
            error = f"Server process died during benchmark: {exc}"
            return [
                VariantResult(
                    variant.name, "", {}, status="failed", returncode=1, workspace=str(workspace), error=error
                )
            ]
        self.listeners.append(listener)
        return [VariantResult(variant.name, "", {}, status="succeeded", returncode=0, workspace=str(workspace))]


@pytest.fixture
def machine(monkeypatch, tmp_path):
    fake = _Machine(tmp_path)
    monkeypatch.setattr(local, "run_grid", fake)
    yield fake
    for listener in fake.listeners:
        listener.close()


def _profile_agentx_point(tmp_path: Path, monkeypatch) -> Path:
    from hyperloom.orchestrator.actions.executors.magpie_agentx import enable_profile

    point = _agentx_point(tmp_path, monkeypatch)
    enable_profile(point)
    return point


@pytest.mark.parametrize("make_point", [_profile_point, _profile_agentx_point], ids=["inferencex", "agentx"])
def test_two_profile_tests_on_one_machine_launch_their_servers_on_ports_of_their_own(
    machine, tmp_path, monkeypatch, make_point
):
    """A profile round is not held by the server lifecycle, so no port was assigned to it: both listened on 8888."""
    point = make_point(tmp_path) if make_point is _profile_point else make_point(tmp_path, monkeypatch)
    shared = point.read_text()

    first = _run(_call(tmp_path, point, kind="profile", output_dir=tmp_path / "first"))
    second = _run(_call(tmp_path, point, kind="profile", output_dir=tmp_path / "second"))

    assert [first.returncode, second.returncode] == [0, 0], "the second server found its port taken"
    assert len(set(machine.ports)) == 2 and REUSE_PORT_DEFAULT not in machine.ports
    server_envs = [((b.get("agentx") or {}).get("launch_overrides") or {}).get("env") or {} for b in machine.launches]
    assert all("PORT" not in env for env in server_envs), "Magpie refuses PORT among an AgentX server's variables"
    assert point.read_text() == shared, "the point is shared by every test that runs it: it is not written to"


# --- repro test ---------------------------------------------------------------------------------------------------

_CASE = AttachmentRef("case", "dir", "a" * 64)


class _Case:
    """A repro case unpacked where the runner leaves it, and the order the server and the case ran in."""

    def __init__(self, tmp_path: Path, monkeypatch, body: str) -> None:
        script = tmp_path / "attach" / _CASE.id / "repro.sh"
        script.parent.mkdir(parents=True)
        script.write_text(f"{body}\n", encoding="utf-8")
        self.ran = tmp_path / "case-ran"
        self.events: list[str] = []

        def teardown(**_: Any) -> None:
            self.events.append("teardown after the case" if self.ran.exists() else "teardown before the case")

        monkeypatch.setattr(local, "teardown_lifecycle_server", teardown)


def _repro_point(directory: Path) -> Path:
    path = directory / "repro-point.yaml"
    config = {"benchmark": {"framework": "sglang", "model": "/models/m", "envs": {"TP": 1, "RUN_EVAL": "false"}}}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _repro_call(tmp_path: Path, **kwargs: Any) -> TestCall:
    defaults: dict[str, Any] = {"kind": "repro", "attachments": (_CASE,), "timeout_sec": 30}
    return _call(tmp_path, _repro_point(tmp_path), **{**defaults, **kwargs})


def _boot_only(grid: _Grid) -> None:
    grid.answers["repro"] = grid.result("repro", note="server_lifecycle_boot_only")


def test_repro_is_a_kind_the_local_backend_runs():
    assert "repro" in SUPPORTED_KINDS


@pytest.mark.parametrize(("code", "verdict"), [(0, "not_reproduced"), (1, "reproduced")])
@pytest.mark.usefixtures("server")
def test_a_repro_test_boots_the_server_runs_the_case_against_it_and_reports_the_verdict(
    grid, monkeypatch, tmp_path, code, verdict
):
    _boot_only(grid)
    case = _Case(tmp_path, monkeypatch, f'echo "$HL_BASE_URL $HL_MODEL" > {tmp_path}/case-ran\nexit {code}')

    outcome = _run(_repro_call(tmp_path))

    (kwargs,) = grid.calls
    assert [v.name for v in kwargs["grid"]] == ["repro"]
    assert kwargs["lifecycle_boot_only"] is True and kwargs["warmup_before_measure"] is False
    assert kwargs["base_extra_envs"] == BOOT_ENVS, "the repro boot is not an evaluation"
    assert kwargs["server_lifecycle"]["cleanup"] is False and kwargs["server_lifecycle"]["port"] == 8123
    assert (tmp_path / "case-ran").read_text() == "http://127.0.0.1:8123 /models/m\n"
    assert case.events == ["teardown after the case"], "the server stays up for the case and is reclaimed after it"
    assert read_verdict(tmp_path / "out") == verdict
    assert (outcome.returncode, outcome.failure_class, outcome.rounds) == (0, None, ())


def test_a_repro_test_runs_no_warmup_even_when_the_call_plans_one(grid, server, monkeypatch, tmp_path):
    _boot_only(grid)
    _Case(tmp_path, monkeypatch, "exit 0")

    _run(_repro_call(tmp_path, rounds=RoundPlan(warmup_rounds=1)))

    assert [c["warmup_before_measure"] for c in grid.calls] == [False]


@pytest.mark.parametrize(
    ("body", "reason"),
    [("exit 2", "exited with 2"), ("sleep 30", "did not finish within"), (None, "no attachment holds repro.sh")],
)
@pytest.mark.usefixtures("server")
def test_a_case_that_cannot_run_to_a_verdict_is_a_failure_of_the_measurement_not_a_verdict(
    grid, monkeypatch, tmp_path, body, reason
):
    _boot_only(grid)
    case = _Case(tmp_path, monkeypatch, body or "")
    if body is None:
        (tmp_path / "attach" / _CASE.id / "repro.sh").unlink()

    outcome = _run(_repro_call(tmp_path, timeout_sec=1))

    assert outcome.failure_class == "infra" and outcome.returncode == local.HOSTED_INFRA_FAILURE_RC
    assert reason in outcome.stderr and "repro_case_error" in outcome.stderr
    assert not (tmp_path / "out" / "repro" / "verdict.json").exists()
    assert len(case.events) == 1, "the server is reclaimed whatever became of the case"


def test_a_server_that_does_not_boot_fails_the_test_like_any_test_and_runs_no_case(grid, server, monkeypatch, tmp_path):
    grid.answers["repro"] = grid.result("repro", status="failed", returncode=1, error="server died")
    case = _Case(tmp_path, monkeypatch, f"touch {tmp_path}/case-ran\nexit 0")

    outcome = _run(_repro_call(tmp_path))

    assert outcome.returncode == 1 and outcome.failure_class in ("candidate", "unclear")
    assert not (tmp_path / "case-ran").exists() and not (tmp_path / "out" / "repro").exists()
    assert case.events == ["teardown before the case"]


def test_a_server_that_cannot_be_held_up_is_a_failure_of_the_measurement_and_boots_nothing(grid, monkeypatch, tmp_path):
    _Server(monkeypatch, eligible=False)
    _Case(tmp_path, monkeypatch, "exit 0")

    outcome = _run(_repro_call(tmp_path))

    assert (
        outcome.failure_class == "infra" and "cannot be held up" in outcome.stderr and "not eligible" in outcome.stderr
    )
    assert grid.calls == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"attachments": ()}, "needs the reproduction case as an attachment"),
        ({"accuracy": True}, "does not run accuracy"),
        ({"prep": "conc_sweep"}, "not a sweep"),
    ],
)
def test_a_repro_call_that_is_not_one_case_against_one_server_is_refused(grid, server, tmp_path, kwargs, match):
    with pytest.raises(TestCallRefused, match=match):
        _run(_repro_call(tmp_path, **kwargs))
    assert grid.calls == []


def _failed_round_with_logs(grid, tmp_path, *, server_log: str, benchmark_stderr: str = "") -> VariantResult:
    workspace = tmp_path / "ws" / "test"
    workspace.mkdir(parents=True)
    (workspace / "server.log").write_text(server_log, encoding="utf-8")
    if benchmark_stderr:
        (workspace / "benchmark_stderr.log").write_text(benchmark_stderr, encoding="utf-8")
    return grid.result("test", status="failed", returncode=1, error="", server_log_path=str(workspace / "server.log"))


def test_a_failed_round_is_classified_from_the_flag_error_in_its_server_log(grid, tmp_path):
    flag_error = "launch_server.py: error: ambiguous option: --chunked-prefill could match --chunked-prefill-size"
    grid.answers["test"] = _failed_round_with_logs(
        grid, tmp_path, server_log=f"usage: launch_server.py [-h]\n{flag_error}\n"
    )

    outcome = _run(_call(tmp_path))

    assert (outcome.returncode, outcome.failure_class) == (1, "candidate")
    assert outcome.failure_reason == f"serve_flag: {flag_error}"


def test_a_failed_round_is_classified_from_the_refused_request_in_its_benchmark_stderr(grid, tmp_path):
    refusal = 'API request failed with error message: {"message":"maximum context length is 5120 tokens","code":400}'
    grid.answers["test"] = _failed_round_with_logs(
        grid, tmp_path, server_log="INFO server booted\n", benchmark_stderr=f"WARNING {refusal}\n"
    )

    outcome = _run(_call(tmp_path))

    assert (outcome.failure_class, outcome.failure_reason) == ("unclear", f"lm_eval_http_400: WARNING {refusal}")
