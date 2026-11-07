# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A profile point is finished before it is routed, and its traces are read from the run_dir it ran into."""

from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _profile_setup, baseline, profile
from hyperloom.orchestrator.actions.executors._profile_setup import ProfilePatch
from hyperloom.orchestrator.actions.executors.benchmark_run.records import BenchmarkAttempt
from inference_suite.contracts import RunLayout


def _write(path: Path, bench: dict) -> Path:
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


def _agentx_point(path: Path) -> Path:
    return _write(path, {"framework": "sglang", "model": "org/model", "agentx": {"enabled": True}, "envs": {}})


@pytest.fixture
def unpatched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_profile_setup, "patch_for_profile", lambda envs, framework: ProfilePatch(True, False, None))


def test_an_agentx_profile_point_turns_its_profiler_on_before_it_is_routed(tmp_path, unpatched) -> None:
    point = _agentx_point(tmp_path / "point.yaml")

    profile.ProfileExecutor()._prepare_point(point, tmp_path)

    torch_profiler = yaml.safe_load(point.read_text())["benchmark"]["profiler"]["torch_profiler"]
    assert torch_profiler["enabled"] is True
    window = (torch_profiler["num_steps"], torch_profiler["num_profiles"], torch_profiler["start_seconds"])
    assert window == (256, 1, 300)


def test_a_fixed_profile_point_arms_the_host_probe_and_states_its_instrumentation(tmp_path, monkeypatch) -> None:
    point = _write(tmp_path / "point.yaml", {"framework": "sglang", "model": "m", "envs": {"PROFILE": "1"}})
    executor = profile.ProfileExecutor()
    armed: list[Path] = []
    monkeypatch.setattr(executor, "_inject_host_probe", lambda path, out: armed.append(path) or "/probe")

    executor._prepare_point(point, tmp_path)

    assert armed == [point]
    assert executor._host_probe_dir == "/probe"
    assert executor._instrumentation_preflight is not None


def test_the_point_is_prepared_before_its_route_is_read(tmp_path, monkeypatch, unpatched) -> None:
    point = _agentx_point(tmp_path / "point.yaml")
    routed: list[bool] = []

    class Routed(Exception):
        pass

    def route(session_dir, config_path):
        document = yaml.safe_load(Path(config_path).read_text())
        routed.append(document["benchmark"]["profiler"]["torch_profiler"]["enabled"])
        raise Routed

    monkeypatch.setattr(baseline, "suite_route", route)
    attempt = BenchmarkAttempt(7200, None, "m", point, "", "", {}, SimpleNamespace(extra={}))

    with pytest.raises(Routed):
        asyncio.run(profile.ProfileExecutor()._run_attempt_rounds(attempt, tmp_path))

    assert routed == [True]


def _run_dir(root: Path) -> Path:
    layout = RunLayout(root / "suite_run_0")
    layout.profile_torch_trace.mkdir(parents=True)
    layout.report.write_text("{}", encoding="utf-8")
    trace = layout.profile_torch_trace / "host-TP-0.1700000000.pt.trace.json.gz"
    trace.write_bytes(gzip.compress(json.dumps({"traceEvents": []}).encode()))
    return layout.root


def test_an_agentx_profile_round_on_inference_suite_certifies_the_traces_in_its_run_dir(tmp_path, monkeypatch) -> None:
    run_dir = _run_dir(tmp_path)

    async def ran(self, ctx):
        return {"status": "succeeded", "workspace": str(run_dir), "framework": "sglang"}

    monkeypatch.setattr(baseline.BenchmarkRunExecutor, "__call__", ran)

    result = asyncio.run(profile.ProfileExecutor()._run_native_profile(SimpleNamespace()))

    assert result["status"] == "succeeded"
    assert result["trace_dir"] == str(RunLayout(run_dir).profile_torch_trace)
    assert result["trace_input_ready"] is True
    assert "trace_capture" not in result


def test_an_agentx_profile_round_whose_run_dir_holds_no_trace_fails(tmp_path, monkeypatch) -> None:
    run_dir = _run_dir(tmp_path)
    for trace in RunLayout(run_dir).profile_torch_trace.iterdir():
        trace.unlink()

    async def ran(self, ctx):
        return {"status": "succeeded", "workspace": str(run_dir), "framework": "sglang"}

    monkeypatch.setattr(baseline.BenchmarkRunExecutor, "__call__", ran)

    result = asyncio.run(profile.ProfileExecutor()._run_native_profile(SimpleNamespace()))

    assert (result["status"], result["error_class"], result["trace_input_ready"]) == ("failed", "no_trace_files", False)
