# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A repro test on a toolbox built by the local factory (CPU only).

The shipped runner and the real local backend run in a local sandbox; only the inference server is replaced, by a
stand-in that is "up" and measures nothing. What this proves is the wiring: acceptance accepts the kind, the case
travels as an attachment into the sandbox, runs against the server's URL, and its verdict reaches the signed bundle,
the result and the verification again.
"""

from __future__ import annotations

import asyncio
import io
import json
import stat
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, Purpose, TestKind, TestResult, Validity
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.measure.tests.repro_fixtures import case_attachment
from hyperloom.measure.tests.test_hosted_real_runner import _git_repo, _point
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom.measure.hosted_spec import MAX_ATTACHMENT_BYTES
from hyperloom_multibranch.contracts.toolbox import RequestRejected, WorkloadKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.toolbox.local_hosted import (
    LocalHostedToolbox,
    LocalOptions,
    LocalSession,
    build_local_toolbox,
)
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from hyperloom_multibranch.toolbox.tests.test_local_hosted import _require_tool_checkout
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

_BASE_URL = "http://127.0.0.1:8123"


def _runner_python(tmp_path: Path) -> str:
    """An interpreter whose ``-m hyperloom.measure.runner`` is the runner with the inference server replaced."""
    script = tmp_path / "repro-python"
    script.write_text(f"#!/bin/sh\nexec {sys.executable} -m hyperloom.measure.tests.repro_runner_entry\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


class _Rig:
    def __init__(
        self, tmp_path: Path, blackboard: LocalBlackboard, commit: str, repo: Path, timeout_sec: int, **envs: str
    ) -> None:
        session = tmp_path / "session"
        self.session = session
        self.spec = spec_context(
            tmp_path, inferencex_commit=commit, benchmark_backend="bypass", blob_root=str(session / "hosted" / "blobs")
        )
        self.toolbox: LocalHostedToolbox = build_local_toolbox(
            LocalSession(session, "run-1", range(2)),
            blackboard,
            AcceptanceSetup(FakeRepo(), PointPlanner(_point(tmp_path, **envs)), IMAGE, WorkloadKind.AGENTX),
            self.spec,
            options=LocalOptions(
                policy=QueuePolicy(per_branch_k=2, global_concurrency=64),
                runner_envs={"INFERENCEX_REPO": str(repo)},
                python=_runner_python(tmp_path),
                poll_interval_sec=0.1,
                timeout_sec=timeout_sec,
                max_infra_retries=0,
            ),
        )

    async def repro(self, script: str) -> TestResult:
        case = case_attachment(self.spec.blobs, script)
        request_id = await self.toolbox.submit_test(
            req(kind=TestKind.REPRO, purpose=Purpose.REPRO, attachments=(case,))
        )
        await asyncio.wait_for(self.toolbox.drain(), 120)
        entry = self.toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry
        return TestResult.from_document(entry.outcome["result"])


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    _require_tool_checkout(repo, commit, tmp_path / "home", monkeypatch)
    return repo, commit


async def _rig(tmp_path: Path, checkout: tuple[Path, str], timeout_sec: int = 120, **envs: str) -> _Rig:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    rig = _Rig(tmp_path, blackboard, checkout[1], checkout[0], timeout_sec, **envs)
    await rig.toolbox.start()
    return rig


@pytest.mark.parametrize(("code", "verdict"), [(0, "not_reproduced"), (1, "reproduced")])
async def test_a_repro_test_runs_its_attached_case_against_the_server_and_reports_the_signed_verdict(
    tmp_path: Path, checkout: tuple[Path, str], code: int, verdict: str
) -> None:
    rig = await _rig(tmp_path, checkout)
    try:
        result = await rig.repro(f'echo "case saw $HL_BASE_URL $HL_MODEL"\nexit {code}\n')

        assert result.validity is Validity.SUCCEEDED and result.repro_verdict == verdict, result
        assert result.rounds == () and result.accuracy == () and result.manifest_sha256
        landed = rig.session / result.output_dir / "repro"
        assert (landed / "case.log").read_text() == f"case saw {_BASE_URL} /nonexistent/model\n"
        assert json.loads((landed / "verdict.json").read_text())["verdict"] == verdict
        await rig.toolbox._executor.reverify(result)  # type: ignore[attr-defined]
    finally:
        await rig.toolbox.close()


@pytest.mark.parametrize(("script", "timeout_sec"), [("exit 2\n", 120), ("sleep 600\n", 120)])
async def test_a_case_that_cannot_run_to_a_verdict_is_a_failed_measurement_with_its_reason(
    tmp_path: Path, checkout: tuple[Path, str], script: str, timeout_sec: int
) -> None:
    rig = await _rig(tmp_path, checkout, timeout_sec)
    try:
        result = await rig.repro(script)

        assert result.validity is Validity.MEASUREMENT_FAILED and result.repro_verdict == "", result
        assert result.failure_class is FailureClass.INFRA
        assert "repro_case_error" in (result.failure_reason or ""), result.failure_reason
    finally:
        await rig.toolbox.close()


async def test_a_server_that_does_not_boot_is_a_failure_of_the_candidate_as_for_a_test(
    tmp_path: Path, checkout: tuple[Path, str]
) -> None:
    rig = await _rig(tmp_path, checkout, FAKE_BOOT="fail")
    try:
        result = await rig.repro("exit 0\n")

        assert result.validity is Validity.CANDIDATE_FAILED and result.repro_verdict == "", result
        assert result.failure_class is FailureClass.CANDIDATE
    finally:
        await rig.toolbox.close()


async def test_a_tampered_verdict_is_not_verified(tmp_path: Path, checkout: tuple[Path, str]) -> None:
    rig = await _rig(tmp_path, checkout)
    try:
        result = await rig.repro("exit 1\n")  # reproduced
        executor = rig.toolbox._executor  # type: ignore[attr-defined]

        with pytest.raises(ResultUnverified, match="repro_verdict: recorded 'not_reproduced'"):
            await executor.reverify(replace(result, repro_verdict="not_reproduced"))

        archived = next(
            Path(rig.toolbox.backend.journal.latest(result.request_id).extra["archive"]).rglob("verdict.json")
        )
        archived.chmod(0o644)
        archived.write_text(json.dumps({"verdict": "not_reproduced", "exit_code": 0}))
        with pytest.raises(ResultUnverified, match="bundle was rejected"):
            await executor.reverify(result)
    finally:
        await rig.toolbox.close()


def _tar(script: str) -> bytes:
    data, buffer = script.encode(), io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo("repro.sh")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


async def test_an_attachment_put_through_the_toolbox_is_the_case_a_repro_request_runs(
    tmp_path: Path, checkout: tuple[Path, str]
) -> None:
    rig = await _rig(tmp_path, checkout)
    try:
        case = await rig.toolbox.put_attachment("case-9", "dir", _tar("exit 1\n"))
        assert await rig.toolbox.put_attachment("case-9", "dir", _tar("exit 1\n")) == case  # content-addressed

        request_id = await rig.toolbox.submit_test(req(kind=TestKind.REPRO, purpose=Purpose.REPRO, attachments=(case,)))
        await asyncio.wait_for(rig.toolbox.drain(), 120)
        entry = rig.toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry
        result = TestResult.from_document(entry.outcome["result"])
        assert result.validity is Validity.SUCCEEDED and result.repro_verdict == "reproduced", result
    finally:
        await rig.toolbox.close()


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"\0" * (MAX_ATTACHMENT_BYTES + 1), "attachment_too_large"),
        (b"this is not a tar archive", "attachment_unreadable"),
    ],
    ids=["oversized", "not_a_tar"],
)
async def test_an_oversized_or_non_tar_directory_attachment_is_refused_and_not_stored(
    tmp_path: Path, checkout: tuple[Path, str], data: bytes, code: str
) -> None:
    rig = await _rig(tmp_path, checkout)
    try:
        with pytest.raises(RequestRejected) as refused:
            await rig.toolbox.put_attachment("case-1", "dir", data)

        assert refused.value.code == code
        assert not any(rig.spec.blobs.root.iterdir())
    finally:
        await rig.toolbox.close()
