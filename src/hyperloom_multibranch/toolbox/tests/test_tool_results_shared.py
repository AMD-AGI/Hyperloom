# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the tool jobs leave in the session's ``tool-results/`` is the group's on shared storage, where a later leg of the
run may be another account of the group: the shared-job ledger and each job's result file ``664`` (``600`` while one
holds a credential), the directories ``2775``, and the files a job copies back from its scratch directory ``664``.
Anywhere else they stay as private as before. The other account itself is in
:mod:`~hyperloom_multibranch.scheduling.tests.test_peers_other_account`."""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from hyperloom_common.shared_storage import SHARED_DIR_MODE, SHARED_FILE_MODE
from hyperloom_multibranch.contracts.blackboard.records import BuildNotice, ToolResultPointer
from hyperloom_multibranch.toolbox.tools import ToolJobs
from hyperloom_multibranch.toolbox.tools.base import RESULT_FILE, ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.executor_tool import _copy_back
from hyperloom_multibranch.toolbox.tools.shared_jobs import SHARED_JOBS_FILE

#: A summary that holds a credential: a URL with userinfo.
CREDENTIALED = "pushed to https://bot:ghp_secret@github.example/fork.git"
OWNER_ONLY = 0o600


class _Says:
    """A tool whose summary is ``summary``."""

    name = "says"

    def __init__(self, summary: str) -> None:
        self.summary = summary

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        return ToolOutput({}, self.summary)


class _Inbox:
    async def notify_tool_result(self, pointer: ToolResultPointer) -> None:
        pass

    async def notify_build(self, notice: BuildNotice) -> None:
        pass


def _shared(path: Path) -> Path:
    """``path`` made a directory set up to be shared: setgid, the group's to write, no one else's."""
    path.mkdir()
    path.chmod(0o2775)
    return path


def _ran(results: Path, summary: str) -> str:
    async def run() -> str:
        jobs = ToolJobs([_Says(summary)], _Inbox(), results)  # type: ignore[arg-type]
        job = await jobs.share(ToolRequest("says", "b1"), "key")
        await jobs.wait(job)
        await jobs.close()
        return job

    return asyncio.run(run())


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_the_ledger_and_a_result_on_shared_storage_are_the_groups_in_directories_of_the_group(tmp_path: Path) -> None:
    results = _shared(tmp_path / "toolbox") / "tool-results"

    job = _ran(results, "analysed")

    assert _mode(results / SHARED_JOBS_FILE) == _mode(results / job / RESULT_FILE) == SHARED_FILE_MODE
    assert _mode(results) == _mode(results / job) == SHARED_DIR_MODE


def test_a_ledger_and_a_result_that_hold_a_credential_are_the_owners_alone_on_shared_storage_too(
    tmp_path: Path,
) -> None:
    results = _shared(tmp_path / "toolbox") / "tool-results"

    job = _ran(results, CREDENTIALED)

    assert _mode(results / SHARED_JOBS_FILE) == _mode(results / job / RESULT_FILE) == OWNER_ONLY


def test_off_shared_storage_the_ledger_and_a_result_stay_the_owners_alone(tmp_path: Path) -> None:
    results = tmp_path / "tool-results"

    job = _ran(results, "analysed")

    assert _mode(results / SHARED_JOBS_FILE) == _mode(results / job / RESULT_FILE) == OWNER_ONLY


def test_a_file_copied_back_from_the_scratch_directory_to_shared_storage_is_the_groups(tmp_path: Path) -> None:
    scratch, workdir = tmp_path / "scratch", _shared(tmp_path / "toolbox") / "job"
    report = scratch / "out" / "analysis.md"
    report.parent.mkdir(parents=True)
    report.write_text("report", encoding="utf-8")
    report.chmod(OWNER_ONLY)

    out = _copy_back({"analysis_md_path": str(report)}, scratch, workdir)

    landed = Path(out["analysis_md_path"])
    assert landed == workdir / "out" / "analysis.md" and landed.read_text(encoding="utf-8") == "report"
    assert (_mode(landed), _mode(landed.parent), _mode(workdir)) == (SHARED_FILE_MODE, SHARED_DIR_MODE, SHARED_DIR_MODE)
