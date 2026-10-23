# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``start_analysis`` and the kind check on a toolbox built by the real local factory, over real HTTP (CPU only)."""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import AttachmentRef, Purpose, TestKind, TestResult, Validity
from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom.measure.local import SUPPORTED_KINDS
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.measure.tests.test_hosted_real_runner import _point
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, RequestRejected, WorkloadKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.acceptance import check_kind
from hyperloom_multibranch.toolbox.fake import FakeToolbox
from hyperloom_multibranch.toolbox.local_hosted import (
    LocalHostedToolbox,
    LocalOptions,
    LocalSession,
    build_local_toolbox,
)
from hyperloom_multibranch.toolbox.mcp_tools import (
    McpSpecialistTokens,
    SpecialistToolOptions,
    register_specialist_tools,
)
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, PARENT, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_analysis_route import SHARES, Stack, _Analysis
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup
from hyperloom_multibranch.toolbox.tools import op_shares as op_shares_module

LOCATION = "tests/p1/attempt-1/trace"
_ATTACHMENT = AttachmentRef("case", "file", "a" * 64)


def _attachments_of(kind: TestKind) -> tuple[AttachmentRef, ...]:
    """A repro request must carry its case."""
    return (_ATTACHMENT,) if kind is TestKind.REPRO else ()


@dataclass
class Local:
    stack: Stack
    toolbox: LocalHostedToolbox
    notebook: LocalNotebook
    session: Path
    tracelens: _Analysis

    async def profile_test(self, location: str = LOCATION, branch: str = "b1") -> str:
        """A PROFILE test of ``branch`` on its head commit that left ``location``, as the notebook holds it."""
        request = replace(req(branch, purpose=Purpose.PROFILE, kind=TestKind.PROFILE), request_id="p1")
        await self.notebook.write_request(request)
        done = TestResult(request_id="p1", validity=Validity.SUCCEEDED, profile_location=location)
        await self.notebook.write_result(done)
        return "p1"


def _toolbox(tmp_path: Path, notebook: LocalNotebook, repo: FakeRepo, **options: Any) -> LocalHostedToolbox:
    session = tmp_path / "session"
    return build_local_toolbox(
        LocalSession(session, "run-1", range(2)),
        notebook,
        AcceptanceSetup(repo, PointPlanner(_point(tmp_path)), IMAGE, WorkloadKind.AGENTX),
        spec_context(tmp_path, blob_root=str(session / "hosted" / "blobs")),
        options=LocalOptions(**options),
    )


@pytest.fixture(autouse=True)
def _shares(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(op_shares_module, "read_op_shares", lambda result: dict(SHARES))


@pytest.fixture
async def local(tmp_path: Path) -> AsyncIterator[Local]:
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    repo = FakeRepo()
    repo.heads["b1"] = PARENT
    tracelens = _Analysis("tracelens")
    toolbox = _toolbox(tmp_path, notebook, repo, tools=[tracelens])
    await toolbox.start()
    server = McpServer(McpServerConfig(host_token=secrets.token_hex(16)), campaign="w38")
    register_specialist_tools(
        server,
        toolbox=toolbox,
        notebooks=lambda b: LocalNotebook(tmp_path / "notebook", Writer(Role.SPECIALIST_RUNTIME, b, b)),
        options=SpecialistToolOptions(analysis=toolbox, toolbox_loop=asyncio.get_running_loop()),
    )
    server.start()
    try:
        stack = Stack(None, toolbox._tools, None, {}, server, McpSpecialistTokens(server))  # type: ignore[arg-type]
        yield Local(stack, toolbox, notebook, tmp_path / "session", tracelens)
    finally:
        server.stop()
        await toolbox.close()


async def test_a_specialist_analyses_the_trace_of_its_profile_test_on_the_real_local_toolbox(local: Local) -> None:
    profile = await local.profile_test()
    reply = await local.stack.start(tool="tracelens", profile_request_id=profile)
    assert reply["result"]["isError"] is False, reply
    job = json.loads(reply["result"]["content"][0]["text"])["job_id"]
    await local.toolbox._tools.wait(job)  # type: ignore[union-attr]
    (request,) = local.tracelens.requests
    assert request.params["trace_input"] == str((local.session / LOCATION).resolve())
    page = await LocalNotebook(local.notebook.root, Writer(Role.SPECIALIST_RUNTIME, "b1", "b1")).poll("b1")
    assert [(item.kind.value, item.pointer) for item in page.items if item.pointer == job] == [("tool_result", job)]


async def test_the_local_backend_advertises_only_the_analysis_tool_it_serves(local: Local) -> None:
    entry = local.stack.advertised()
    assert local.toolbox._tools.tool_names == ("tracelens",)  # type: ignore[union-attr]
    assert entry["inputSchema"]["properties"]["tool"]["enum"] == list(local.toolbox.analysis_tools) == ["tracelens"]
    assert "roofline" not in entry["description"] and "roofline" not in json.dumps(entry["inputSchema"])


async def test_roofline_is_refused_on_the_local_backend(local: Local) -> None:
    assert "must be one of ['tracelens']" in await local.stack.refused(tool="roofline")
    with pytest.raises(RequestRejected) as stale:  # the toolbox's own guard, for a client with an older tool list
        await local.toolbox.submit_analysis("b1", "roofline", {})
    assert stale.value.code == "tool_unavailable_on_backend"


@pytest.mark.parametrize("with_profile", [True, False])
async def test_a_trace_input_of_the_callers_is_refused_with_or_without_a_profile_request_id(
    local: Local, with_profile: bool
) -> None:
    profile = await local.profile_test() if with_profile else ""
    text = await local.stack.refused(tool="tracelens", params={"trace_input": "/etc"}, profile_request_id=profile)
    assert "server_owned_param" in text
    assert local.tracelens.requests == []


async def test_tracelens_without_a_profile_request_id_is_refused_by_name(local: Local) -> None:
    text = await local.stack.refused(tool="tracelens")
    assert "profile_required" in text and local.tracelens.requests == []


async def test_an_absolute_trace_location_outside_the_session_is_refused(local: Local, tmp_path: Path) -> None:
    profile = await local.profile_test(str(tmp_path))  # a real directory, but not the session's
    text = await local.stack.refused(tool="tracelens", profile_request_id=profile)
    assert "profile_without_trace" in text and local.tracelens.requests == []


async def test_an_absolute_trace_location_inside_the_session_is_accepted_resolved(local: Local) -> None:
    inside = local.session / LOCATION
    profile = await local.profile_test(str(inside))
    reply = await local.stack.start(tool="tracelens", profile_request_id=profile)
    await local.toolbox._tools.wait(json.loads(reply["result"]["content"][0]["text"])["job_id"])  # type: ignore[union-attr]
    (request,) = local.tracelens.requests
    assert request.params["trace_input"] == str(inside.resolve())


async def test_a_trace_location_that_leaves_the_session_is_refused(local: Local) -> None:
    profile = await local.profile_test("../../outside")
    text = await local.stack.refused(tool="tracelens", profile_request_id=profile)
    assert "profile_without_trace" in text and local.tracelens.requests == []


async def test_a_profile_test_without_a_trace_location_is_refused_by_name(local: Local) -> None:
    profile = await local.profile_test()
    bare = tuple(
        replace(r, result=replace(r.result, profile_location="")) for r in await local.notebook.profiles(PARENT)
    )

    async def profiles(commit: str) -> Any:
        return bare

    local.toolbox._records.profiles = profiles  # type: ignore[method-assign,union-attr]
    text = await local.stack.refused(tool="tracelens", profile_request_id=profile)
    assert "profile_without_trace" in text and local.tracelens.requests == []


async def test_the_default_options_serve_tracelens_and_the_tuner_not_roofline(tmp_path: Path) -> None:
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _toolbox(tmp_path, notebook, FakeRepo())
    try:
        assert toolbox._tools is not None and toolbox._tools.tool_names == ("kernel_tune", "tracelens")
        assert toolbox.analysis_tools == ("tracelens", "kernel_tune")
    finally:
        await toolbox.close()


async def test_the_fake_toolbox_still_accepts_a_profile_test() -> None:
    fake = FakeToolbox()
    try:
        assert await fake.submit_test(req(kind=TestKind.PROFILE, purpose=Purpose.PROFILE, server_args="--p"))
    finally:
        await fake.close()


@pytest.mark.parametrize("kind", list(TestKind))
def test_the_kinds_come_from_the_backend_and_a_sweep_is_a_test(kind: TestKind) -> None:
    assert sorted(SUPPORTED_KINDS) == ["accuracy", "profile", "repro", "test"]
    request = req(
        kind=kind, purpose=Purpose.EXPLORE, accuracy=kind is TestKind.ACCURACY, attachments=_attachments_of(kind)
    )
    check_kind(request, SUPPORTED_KINDS)
    check_kind(request, None)  # a builder that models every kind accepts every kind


def test_a_backend_that_runs_fewer_kinds_refuses_the_rest_before_anything_is_queued() -> None:
    request = req(kind=TestKind.REPRO, purpose=Purpose.REPRO, attachments=(_ATTACHMENT,))

    with pytest.raises(RequestRejected) as error:
        check_kind(request, ("test", "accuracy", "profile"))

    assert error.value.code == "unsupported_kind" and "unsupported_kind" in REJECTION_CODES
