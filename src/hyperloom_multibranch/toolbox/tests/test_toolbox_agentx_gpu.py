# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real toolbox on a GPU machine with an AgentX workload: a measurement, a patched recipe and a profile (W71).

The points are rendered by the toolbox's own ``HostedPlanner`` (through ``build_local_toolbox``) and run by the pinned
Magpie, which resolves the recipe from the pinned InferenceX, owns the server and replays the traces. Each test reads
back what Magpie wrote (``benchmark_report.json``) and holds it to the validity Magpie itself defines.

Opt in with the environment of ``test_toolbox_kinds_gpu.py`` and ``HL_W13_BACKEND=magpie``: ``HL_W13_MODEL`` (the
checkpoint directory of the recipe's model), ``HL_W13_INFERENCEX`` (a git checkout of the InferenceX the Magpie pin
names, the ``inferencex-e2e`` layout), ``HL_W13_OUT`` (an empty directory) and ``HL_W13_CARDS``. ``HL_W13_AGENTX_MODEL``
(default Qwen3.5-397B MXFP4), ``HL_W13_AGENTX_PRECISION`` (``fp4``), ``HL_W13_AGENTX_TP`` (``2``) and
``HL_W13_AGENTX_CONC`` (``4``) select the recipe arm: the matrix must have exactly one arm for that model, topology and
concurrency on the GPU type. Anywhere else the module is skipped.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.contracts.measurement import Purpose, TestKind, TestResult
from hyperloom.contracts.recipe import ComponentChange, RecipeDelta
from hyperloom.inference_optimizer.agentx.report import agentx_measurement
from hyperloom.inference_optimizer.cli.benchmark_config import AgentXWorkload, write_agentx_benchmark
from hyperloom.measure.profile_traces import PROFILE_DIR, is_trace_file
from hyperloom.measure.tests._recipes import put_patch
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.tests.helpers import req
from hyperloom_multibranch.toolbox.tests.test_toolbox_kinds_gpu import (
    _CLEARED,
    BACKEND,
    CARDS,
    INFERENCEX,
    MODEL,
    TARGET,
    _marker_patch,
    _probe,
    _run_root,
    _submit,
    _toolbox,
)

AGENTX_MODEL = os.environ.get("HL_W13_AGENTX_MODEL", "amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2")
PRECISION = os.environ.get("HL_W13_AGENTX_PRECISION", "fp4")
TP = int(os.environ.get("HL_W13_AGENTX_TP", "2"))
CONC = int(os.environ.get("HL_W13_AGENTX_CONC", "4"))
#: A canonical AgentX replay is one hour of traces (Magpie's ``_DEFAULT_AGENTIC_DURATION_SECONDS``), after a server
#: boot of a model this size and the replay client's own set-up: the attempt and the wait get the room.
ATTEMPT_SEC = 4 * 3600
WAIT_SEC = ATTEMPT_SEC + 1800
USES_REAL_GPUS = True  # keep the host's amd-smi and rocm-smi: see hyperloom.measure.tests.gpu_free_host
pytestmark = pytest.mark.skipif(
    BACKEND != "magpie"
    or importlib.util.find_spec("sglang") is None
    or not Path(MODEL).is_dir()
    or not Path(INFERENCEX, ".git").exists()
    or len(CARDS) != TP,
    reason="needs HL_W13_BACKEND=magpie, SGLang, HL_W13_MODEL, HL_W13_INFERENCEX (a git checkout) and TP cards",
)


@pytest.fixture(autouse=True)
def workload_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The process environment the points are rendered from: nothing of an InferenceX workload is in it."""
    for name in (*_CLEARED, "ISL", "OSL", "FRAMEWORK", "NUM_PROMPTS", "NUM_WARMUPS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")


def _workload(root: Path) -> Path:
    """The benchmark YAML an AgentX session binds, written as ``hyperloom optimize`` writes it."""
    path = root / "bench" / "benchmark.yaml"
    path.parent.mkdir(parents=True)
    workload = AgentXWorkload(
        framework="sglang",
        model=AGENTX_MODEL,
        precision=PRECISION,
        tp=TP,
        ep=1,
        conc=CONC,
        model_path=MODEL,
        gpu_type="MI355X",
    )
    write_agentx_benchmark(path, workload)
    return path


def _agentx_toolbox(root: Path, name: str):
    return _toolbox(
        root,
        name,
        tp=TP,
        config_path=_workload(root),
        primary=CONC,
        timeout_sec=ATTEMPT_SEC,
        workload_kind=WorkloadKind.AGENTX,
        inferencex_path=INFERENCEX,
    )


def _points(root: Path) -> list[dict[str, Any]]:
    """The benchmark mappings of the points the planner rendered for this test."""
    return [yaml.safe_load(path.read_text())["benchmark"] for path in sorted((root / "points").rglob("*.y*ml"))]


def _report(landed: Path) -> dict[str, Any]:
    """The one report Magpie wrote for the test's round."""
    found = sorted(landed.rglob("benchmark_report.json"))
    assert len(found) == 1, f"benchmark_report.json under {landed}: {[str(path) for path in found]}"
    return json.loads(found[0].read_text())


def _assert_rendered_the_agentx_way(root: Path) -> None:
    """What the planner hands Magpie: no InferenceX evaluation or script, the launch in ``agentx.launch_overrides``."""
    points = _points(root)
    assert points, f"no point rendered under {root / 'points'}"
    for bench in points:
        envs = bench["envs"]
        assert bench["agentx"]["enabled"] is True and "launch_overrides" in bench["agentx"], bench["agentx"]
        assert str(envs.get("RUN_EVAL", "false")).lower() == "false", envs
        assert not {"NUM_PROMPTS", "NUM_WARMUPS", "EXTRA_SGLANG_ARGS"} & envs.keys(), envs
        assert "benchmark_script" not in bench and envs["CONC"] == CONC and str(envs["TP"]) == str(TP), bench


def _assert_valid_on_magpies_terms(report: dict[str, Any]) -> dict[str, Any]:
    """Magpie's AgentX validity: success, benchmark_valid and publishable; not diagnostic; Magpie owns the run."""
    metrics = report["agentx_metrics"]
    assert report.get("scenario") == "agentx"
    assert all(report.get(key) is True for key in ("success", "benchmark_valid", "publishable")), {
        key: report.get(key) for key in ("success", "benchmark_valid", "publishable", "errors")
    }
    assert metrics.get("diagnostic_only") is not True and metrics.get("execution_owner") == "magpie"
    assert metrics["recipe"]["conc"] == CONC and str(metrics["recipe"]["tp"]) == str(TP), metrics["recipe"]
    assert agentx_measurement(report)["valid_measurement"] is True
    return metrics


async def test_an_agentx_point_the_planner_renders_is_run_by_magpie_to_a_valid_measurement() -> None:
    root = _run_root("agentx-measure")
    toolbox, blackboard, *_ = _agentx_toolbox(root, "w71-measure")
    try:
        request = req(purpose=Purpose.EXPLORE, change=RecipeDelta())
        request_id, result, landed = await _submit(toolbox, blackboard, request, root, WAIT_SEC)
        report = _report(landed)
        metrics = _assert_valid_on_magpies_terms(report)
        _probe(
            "agentx-measure",
            request_id=request_id,
            recipe=metrics["recipe"],
            throughput=metrics.get("throughput"),
            requests=metrics.get("requests"),
            valid={key: report.get(key) for key in ("success", "benchmark_valid", "publishable")},
            execution_owner=metrics.get("execution_owner"),
            rounds=[(r.phase, r.concurrency, r.metrics.get("output_throughput")) for r in result.rounds],
        )
        _assert_rendered_the_agentx_way(root)
        assert result.rounds and result.rounds[0].metrics.get("output_throughput")
    finally:
        await toolbox.close()


async def test_a_patched_agentx_recipe_reaches_the_server_magpie_launches() -> None:
    root = _run_root("agentx-patched")
    toolbox, blackboard, blobs = _agentx_toolbox(root, "w71-patched")
    try:
        marker = root / "marker.txt"
        diff, patched = _marker_patch(root / "patch-repo", marker)
        patch = put_patch(blobs, "w71-marker.patch", diff)
        delta = RecipeDelta(components=(ComponentChange("sglang", add_patches=(patch,)),))
        request_id, result, landed = await _submit(toolbox, blackboard, req(change=delta), root, WAIT_SEC)
        report = _report(landed)
        metrics = _assert_valid_on_magpies_terms(report)
        launch = metrics["server_launch"]
        site = "/farm/site"
        sandbox = toolbox.backend.journal.latest(request_id).sandbox_id
        # an attempt that ended on an infrastructure fault leaves its own lines in the marker: the landed one is read
        imported = {line for line in marker.read_text().split() if sandbox in line}
        _probe(
            "agentx-patched",
            request_id=request_id,
            imported=sorted(imported),
            pythonpath=launch["effective_env"].get("PYTHONPATH"),
            source_files=launch["source_files"],
            executable=launch["effective_argv"][0],
        )
        # the marker: the patched file was imported from the farm the rebuilt framework was placed in
        (farm_file,) = imported
        assert site + "/sglang/" + TARGET in farm_file and sandbox in farm_file
        # Magpie's launch evidence: the farm site leads the server's PYTHONPATH and the patched file is bound by hash
        first = launch["effective_env"]["PYTHONPATH"].split(":")[0]
        assert first.endswith(site) and sandbox in first, launch["effective_env"]["PYTHONPATH"]
        digest = hashlib.sha256(patched.encode()).hexdigest()
        bound = {
            name: value for name, value in launch["source_files"].items() if name.endswith(f"{site}/sglang/{TARGET}")
        }
        assert list(bound.values()) == [digest], launch["source_files"]
        _assert_rendered_the_agentx_way(root)
    finally:
        await toolbox.close()


def _assert_profile_landed(session: Path, result: TestResult) -> list[Path]:
    """The profile's location holds the traces Magpie captured."""
    location = session / result.profile_location
    assert result.profile_location and location.is_dir(), f"profile_location {result.profile_location!r}"
    assert location.name == PROFILE_DIR and location.is_relative_to(session / result.output_dir)
    traces = sorted(path for path in location.rglob("*") if path.is_file() and is_trace_file(path.name))
    ranks = [path for path in traces if ".trace.json" in path.name]
    assert len(ranks) >= TP and all(trace.stat().st_size > 0 for trace in ranks), f"trace files in {location}: {traces}"
    return traces


async def test_an_agentx_profile_captures_a_complete_trace_that_lands_under_the_profile_location() -> None:
    root = _run_root("agentx-profile")
    toolbox, blackboard, _ = _agentx_toolbox(root, "w71-profile")
    try:
        request = req(kind=TestKind.PROFILE, purpose=Purpose.PROFILE, change=RecipeDelta())
        request_id, result, landed = await _submit(toolbox, blackboard, request, root, WAIT_SEC)
        traces = _assert_profile_landed(root / "session", result)
        report = _report(landed)
        capture = report["agentx_metrics"]["profile_capture"]
        _probe(
            "agentx-profile",
            request_id=request_id,
            profile_location=result.profile_location,
            capture_status=capture.get("status"),
            profiles=len(capture.get("profiles") or []),
            traces=[str(trace.relative_to(root / "session")) for trace in traces],
        )
        assert capture.get("status") == "complete", capture
        assert report["agentx_metrics"].get("diagnostic_only") is True
        points = _points(root)
        assert points and all(bench["profiler"]["torch_profiler"]["enabled"] is True for bench in points)
    finally:
        await toolbox.close()
