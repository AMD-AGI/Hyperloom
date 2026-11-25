# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``knob_tune`` tool over fake builds and a fake microbenchmark: what it refuses, and how it holds its card;
and over the real store, planner and build runner child with a stand-in llama.cpp compile: each value's patch is
built."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.outside_facts import LLAMACPP_BACKEND_OPS, kernel_knobs
from hyperloom.measure.runner.native import recipes as recipes_module
from hyperloom.measure.runner.native import session_builds
from hyperloom.measure.runner.native.isolated import Isolation
from hyperloom.measure.runner.native.session import open_session
from hyperloom.measure.tests import _native_llamacpp as stand_in
from hyperloom.measure.tests._recipes import write_tree
from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
from hyperloom_multibranch.contracts.toolbox import BranchLease, LeaseCharge
from hyperloom_multibranch.toolbox.analysis import AnalysisAsk, analysis_request
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.toolbox import local_hosted
from hyperloom_multibranch.toolbox.tests.test_analysis_route import _stack, _text
from hyperloom_multibranch.toolbox.tools import knob_tune
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolFailure, ToolRequest
from hyperloom.orchestrator.actions.executors.candidate_builds import TreeBuild, TreeBuildStatus

_MMVQ = "int f() {\n    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {\n        return 1;\n    }\n}\n"
_HEADER = (
    "enum ggml_type { GGML_TYPE_F32 = 0, GGML_TYPE_Q8_0 = 8, };\nenum ggml_op { GGML_OP_NONE, GGML_OP_MUL_MAT, };\n"
)
_PARAMS = {
    "knob": "mmvq_nwarps_rdna3",
    "axes": {"type": ["Q8_0"], "ncols": ["1", "2"]},
    "values": ["2", "4"],
    "shapes": [[64, 128]],
    "rounds": 3,
}


class _Leases:
    def __init__(self) -> None:
        self.leased: list[tuple[str, str]] = []
        self.released: list[str] = []

    def check(self, cards: int) -> None:
        assert cards == 1

    def can_lease(self, cards: int) -> bool:
        return cards == 1

    async def lease(self, holder: str, cards: int, root: Path, *, charge: LeaseCharge) -> BranchLease:
        self.leased.append((holder, charge.branch))
        return BranchLease(f"lease-{len(self.leased)}", charge.branch, (5,), root, {}, 0)

    def release(self, lease_id: str) -> None:
        self.released.append(lease_id)


class _Builds:
    def __init__(self, root: Path, gfx: str = "gfx1100") -> None:
        self.root = root
        self.session = SimpleNamespace(
            planner=SimpleNamespace(site=SimpleNamespace(machine=SimpleNamespace(gfx_arch=gfx)))
        )

    def read_upstream(self, path: str) -> str:
        return _HEADER if path.endswith(".h") else _MMVQ

    def build_patch(self, patch: bytes) -> TreeBuild:
        forced = re.findall(rb"return (\d+); }", patch)
        bin_dir = self.root / f"bin-{forced[0].decode() if forced else 'base'}"
        bin_dir.mkdir(parents=True, exist_ok=True)
        return TreeBuild(TreeBuildStatus.BUILT, key=bin_dir.name, bin=bin_dir)


@pytest.fixture
def envs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    seen: list[dict[str, str]] = []

    def timed(executable: Path, cases: Path, names, env, timeout: float) -> dict[str, float]:
        seen.append(dict(env))
        faster = executable.parent.name == "bin-2"
        return {name: (60.0 if faster and "_1_" in name else 100.0) for name in names}

    monkeypatch.setattr(knob_tune, "time_cases", timed)
    monkeypatch.setattr(knob_tune, "check_cases", lambda exe, cases, names, env, timeout: dict.fromkeys(names, "OK"))
    return seen


async def test_a_sweep_leases_a_card_for_each_round_and_gives_it_back(
    tmp_path: Path, envs: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    leases = _Leases()
    tool = knob_tune.KnobTuneTool(leases, {"llamacpp": _Builds(tmp_path / "builds")})
    request = ToolRequest(knob_tune.KNOB_TUNE_TOOL, "b1", _PARAMS)

    tool.validate(request)
    output = await tool.run(request, tmp_path / "job-7")

    assert output.ok and output.result["winners"] == {"type=Q8_0,ncols=1": "2"}
    assert "+        if (type == GGML_TYPE_Q8_0 && ncols_dst == 1) { return 2; }" in output.result["patch"]
    assert leases.leased == [("tool:job-7", "b1")] * (1 + 3 + 1)
    assert leases.released == [f"lease-{index}" for index in range(1, 6)]
    assert {env["ROCR_VISIBLE_DEVICES"] for env in envs} == {"5"}
    assert not any({"HF_TOKEN", "HIP_VISIBLE_DEVICES"} & set(env) for env in envs)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"shapes": [[64]]}, "weight shapes"),
        ({"axes": {"ncols": "1"}}, "is a list"),
        ({"rounds": 99}, "rounds"),
        ({"copies": True}, "whole numbers"),
        ({"timeout_sec": 10}, "takes"),
    ],
)
def test_a_sweep_no_toolbox_can_run_is_refused_when_asked(changes, message: str) -> None:
    with pytest.raises(InvalidRequest, match=message):
        knob_tune.check_params({**_PARAMS, **changes})


def test_a_request_that_names_no_knob_is_refused_with_the_knobs_it_takes() -> None:
    params = {key: value for key, value in _PARAMS.items() if key != "knob"}

    with pytest.raises(InvalidRequest, match="no kernel knob") as refused:
        knob_tune.check_params(params)

    said = str(refused.value)
    assert kernel_knobs()
    for name, knob in kernel_knobs().items():
        assert f"{name} of {knob.component}" in said
        for axis, held in knob.axes:
            assert f"{axis} {list(held)}" in said
        assert f"values {list(knob.values)}" in said


def test_a_knob_of_another_architecture_or_component_is_refused(tmp_path: Path) -> None:
    request = ToolRequest(knob_tune.KNOB_TUNE_TOOL, "b1", _PARAMS)

    with pytest.raises(InvalidRequest, match="not for gfx942"):
        knob_tune.KnobTuneTool(_Leases(), {"llamacpp": _Builds(tmp_path, "gfx942")}).validate(request)
    with pytest.raises(InvalidRequest, match="does not build"):
        knob_tune.KnobTuneTool(_Leases(), {}).validate(request)


async def test_a_job_cancelled_in_a_round_leases_no_card_for_the_next(tmp_path: Path, envs) -> None:
    leases, scope = _Leases(), CancelScope()
    given_back = leases.release
    leases.release = lambda lease_id: (scope.cancel(reason="tool_job_cancelled"), given_back(lease_id))
    tool = knob_tune.KnobTuneTool(leases, {"llamacpp": _Builds(tmp_path / "builds")})

    with use_cancel_scope(scope), pytest.raises(ToolFailure, match="cancelled"):
        await tool.run(ToolRequest(knob_tune.KNOB_TUNE_TOOL, "b1", _PARAMS), tmp_path / "job-7")

    assert leases.leased == [("tool:job-7", "b1")]


async def test_a_job_cancelled_in_a_benchmark_runs_no_other_and_gives_its_card_back(
    tmp_path: Path, envs, monkeypatch: pytest.MonkeyPatch
) -> None:
    leases, scope, timed = _Leases(), CancelScope(), knob_tune.time_cases
    monkeypatch.setattr(
        knob_tune, "time_cases", lambda *args, **kw: (scope.cancel(reason="tool_job_cancelled"), timed(*args, **kw))[1]
    )
    tool = knob_tune.KnobTuneTool(leases, {"llamacpp": _Builds(tmp_path / "builds")})

    with use_cancel_scope(scope), pytest.raises(ToolFailure, match="cancelled"):
        await tool.run(ToolRequest(knob_tune.KNOB_TUNE_TOOL, "b1", _PARAMS), tmp_path / "job")

    assert len(envs) == 1 and leases.released == ["lease-1"]


async def test_a_job_cancelled_before_its_builds_start_builds_nothing(tmp_path: Path, envs) -> None:
    leases, scope = _Leases(), CancelScope()
    scope.cancel(reason="tool_job_cancelled")
    tool = knob_tune.KnobTuneTool(leases, {"llamacpp": _Builds(tmp_path / "builds")})

    with use_cancel_scope(scope), pytest.raises(ToolFailure, match="cancelled"):
        await tool.run(ToolRequest(knob_tune.KNOB_TUNE_TOOL, "b1", _PARAMS), tmp_path / "job")

    assert not (tmp_path / "builds").exists() and leases.leased == []


async def test_a_value_s_patch_is_built_by_the_session_whose_blob_store_is_not_below_the_tool_s_root(
    tmp_path: Path, envs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        recipes_module.RECIPE_FACTORIES, "llamacpp", f"{stand_in.__name__}:{stand_in.stand_in_recipe.__name__}"
    )
    monkeypatch.setattr(session_builds, "native_recipes", lambda: {"llamacpp": stand_in.stand_in_recipe()})
    monkeypatch.setattr(session_builds, "open_session", partial(open_session, isolation=lambda: Isolation((), None)))
    checkout = stand_in.llama_checkout(tmp_path / "llama.cpp")
    write_tree(checkout, {kernel_knobs()[_PARAMS["knob"]].file: _MMVQ, LLAMACPP_BACKEND_OPS["enum_header"]: _HEADER})
    stand_in.git(checkout, "add", "-A")
    stand_in.git(checkout, "commit", "-qm", "kernel")
    environ = {"FRAMEWORK_REPO_PATH": str(checkout), **stand_in.IMAGE, "HYPERLOOM_GFX_ARCH": "gfx1100"}
    native = session_builds.open_native_builds(tmp_path / "session", "llamacpp", environ)
    tool = knob_tune.knob_tune_tool(_Leases(), native.session, tmp_path / "knob-tune")

    output = await tool.run(ToolRequest(knob_tune.KNOB_TUNE_TOOL, "b1", _PARAMS), tmp_path / "job")

    assert [(arm["status"], arm["detail"]) for arm in output.result["arms"]] == [("timed", "")] * 3
    assert len(output.result["table"]) == 4


async def test_the_mcp_route_checks_a_sweep_when_it_is_asked_for() -> None:
    request = await analysis_request(None, None, AnalysisAsk("b1", knob_tune.KNOB_TUNE_TOOL, _PARAMS))

    assert (request.tool, request.branch, dict(request.params)) == (knob_tune.KNOB_TUNE_TOOL, "b1", _PARAMS)
    with pytest.raises(RequestRejected, match="no kernel knob"):
        await analysis_request(None, None, AnalysisAsk("b1", knob_tune.KNOB_TUNE_TOOL, {**_PARAMS, "knob": "x"}))
    with pytest.raises(RequestRejected, match="reads no profile test"):
        await analysis_request(None, None, AnalysisAsk("b1", knob_tune.KNOB_TUNE_TOOL, _PARAMS, "p1"))


async def test_a_specialist_reads_the_sweep_s_document_from_the_toolbox_and_another_branch_does_not(
    tmp_path: Path, envs
) -> None:
    tool = knob_tune.KnobTuneTool(_Leases(), {"llamacpp": _Builds(tmp_path / "builds")})
    async with contextlib.asynccontextmanager(_stack)(tmp_path, analyses={tool.name: tool}) as stack:
        job = json.loads(_text(await stack.start(tool=tool.name, params=_PARAMS)))["job_id"]
        await stack.jobs.wait(job)
        inbox = json.loads(_text(await asyncio.to_thread(stack.call, "b1", "poll_inbox")))
        own = await asyncio.to_thread(stack.call, "b1", "analysis_result", job_id=job)
        other = await asyncio.to_thread(stack.call, "b2", "analysis_result", job_id=job)

    (item,) = inbox["items"]
    assert item["pointer"] == job and item["summary"].endswith(
        "succeeded: mmvq_nwarps_rdna3: 1 of 2 keys beat the default beyond the noise; "
        "microbenchmark times are advice only"
    )
    document = json.loads(_text(own))
    assert (document["job_id"], document["status"]) == (job, "succeeded")
    assert document["result"]["winners"] == {"type=Q8_0,ncols=1": "2"}
    assert "+        if (type == GGML_TYPE_Q8_0 && ncols_dst == 1) { return 2; }" in document["result"]["patch"]
    assert len(document["result"]["table"]) == 4 and document["result"]["noise"]
    assert other["error"]["message"] == "a specialist reads the jobs of its own branch only"


@pytest.mark.parametrize(
    ("pinned", "gfx", "served"),
    [
        ("llamacpp", "gfx1100", True),
        ("llamacpp", "gfx1201", False),
        ("sgl_kernel", "gfx1100", False),
        (None, "", False),
    ],
)
def test_a_local_toolbox_serves_the_sweep_where_its_native_builds_pin_a_component_with_a_knob_of_their_architecture(
    tmp_path: Path, pinned: str | None, gfx: str, served: bool
) -> None:
    machine = SimpleNamespace(gfx_arch=gfx)
    site = SimpleNamespace(commits={pinned: "a" * 40}, sources=tmp_path / "src", machine=machine)
    native = SimpleNamespace(session=SimpleNamespace(planner=SimpleNamespace(site=site))) if pinned else None
    options = local_hosted.LocalOptions(framework_configs=str(tmp_path / "configs"))

    tools = local_hosted._local_tools(_Leases(), options, native, tmp_path / "toolbox")

    assert (knob_tune.KNOB_TUNE_TOOL in [tool.name for tool in tools]) is served
