# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local backend against a stubbed ``run_grid``: the parameter mapping of each kind of test (design 4.4.7)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.contracts.measurement import AccuracyValue
from hyperloom.measure import local
from hyperloom.measure.local import LocalMeasurementBackend, TestCallRefused
from hyperloom.measure.socket import RoundPlan, SessionFacts, TestCall
from hyperloom.orchestrator.actions.executors import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    VariantResult,
)


def _point(directory: Path, name: str, *, conc: int = 4, run_eval: object = "false") -> Path:
    envs: dict[str, object] = {"TP": 1, "CONC": conc}
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
        self.env_seen.append(
            {
                key: os.environ.get(key)
                for key in ("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", "INFERENCEX_PATH", "HL_TEST_FLAG")
            }
        )
        name = kwargs["grid"][0].name
        return [self.answers.get(name) or self.result(name)]


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


# --- single-point test --------------------------------------------------------------------------------------------


def test_a_single_point_test_is_one_run_grid_call_with_one_variant(grid, tmp_path):
    call = _call(tmp_path, result_dir="/res", session_deadline_sec=123.0, serving_lease="lease")

    outcome = _run(call)

    (kwargs,) = grid.calls
    assert kwargs["base_yaml_path"] == call.points[0]
    assert [v.name for v in kwargs["grid"]] == ["test"]
    assert kwargs["output_root"] == tmp_path / "out"
    assert (kwargs["result_dir"], kwargs["session_deadline_sec"], kwargs["serving_lease"]) == ("/res", 123.0, "lease")
    assert kwargs["magpie_python"] is None
    assert "server_lifecycle" not in kwargs
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


@pytest.mark.parametrize("kind", ["repro"])
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


def test_a_sweep_boots_the_server_once_and_measures_the_levels_from_high_to_low_on_it(grid, server, tmp_path):
    outcome = _run(_sweep_call(tmp_path, 2, 8, 4))

    names = [c["grid"][0].name for c in grid.calls]
    assert names == ["c8", "c8", "c4", "c2"]
    boot, *reuse = grid.calls
    assert boot["lifecycle_boot_only"] is True and boot["base_extra_envs"] == {"MAGPIE_RUN_PHASE": "server"}
    assert _lifecycle(boot)["cleanup"] is False
    assert all(c["server_already_ready"] is True for c in reuse)
    assert [_lifecycle(c)["cleanup"] for c in reuse] == [False, False, True]
    assert {(_lifecycle(c)["pid_dir"], _lifecycle(c)["port"]) for c in grid.calls} == {
        (str(tmp_path / "out" / "server_arm"), 8123)
    }
    assert all(c["warmup_before_measure"] is False for c in grid.calls)
    assert [r.point for r in outcome.rounds] == [1, 0 + 2, 0]  # c8, c4, c2 by their positions in the call
    assert len(server.teardowns) == 1


def test_no_call_of_a_sweep_runs_without_the_shared_server(grid, server, tmp_path):
    _run(_sweep_call(tmp_path, 8, 4, 2))

    assert all(_lifecycle(c) is not None for c in grid.calls)


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


def test_a_test_on_the_image_as_it_is_runs_the_point_unchanged(grid, tmp_path, installed_framework):
    from hyperloom.contracts.recipe import RecipeDelta
    from hyperloom.measure.compile import LocalBlobStore, compile_environment
    from hyperloom.measure.tests._recipes import IMAGE, recipe_with

    compiled = compile_environment(
        recipe_with(), RecipeDelta(), blobs=LocalBlobStore(tmp_path / "blobs"), active_image_digest=IMAGE
    )
    call = _call(tmp_path, environment=compiled)

    _run(call)

    assert compiled.is_base and grid.calls[0]["base_yaml_path"] == call.points[0]


def test_an_environment_that_cannot_be_rebuilt_is_refused_with_the_reason(grid, tmp_path, installed_framework):
    compiled = _compile(tmp_path, installed_framework)
    point = _point_with_python(tmp_path, "/no/such/python")

    with pytest.raises(TestCallRefused, match="cannot be rebuilt here"):
        _run(_call(tmp_path, point, environment=compiled))
    assert grid.calls == []


# --- profile test -------------------------------------------------------------------------------------------------


def _profile_point(directory: Path, *, profile: object = "1", enabled: bool = True) -> Path:
    envs: dict[str, object] = {"TP": 1, "CONC": 4, "RUN_EVAL": "false"}
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


def test_a_plain_test_does_not_gather_traces(grid, tmp_path):
    _with_trace(grid)

    _run(_call(tmp_path))

    assert not (tmp_path / "out" / "profile").exists()
