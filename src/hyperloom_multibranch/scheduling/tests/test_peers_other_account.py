# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A later leg of a hosted run may be another account of the run's group: it reads and adds to what an earlier leg's
toolbox wrote in ``MB-toolbox`` on shared storage: the shared tool-job ledger and the jobs' results, the archived
bundles of its tests and their landed files."""

from __future__ import annotations

import asyncio
import hashlib
import json
import stat
import sys
import tempfile
from pathlib import Path

import pytest

import hyperloom_multibranch
from hyperloom.measure.archive import BUNDLE_DIR_MODE, DirResultArchive
from hyperloom.measure.bundle import LANDED_DIR_MODE, LANDED_FILE_MODE, materialize, verify_bundle
from hyperloom.measure.tests.bundle_factory import REQUEST_ID, make_bundle, verify_args
from hyperloom_common.shared_storage import SHARED_FILE_MODE
from hyperloom_common.tests.shared_accounts import Accounts, other_accounts
from hyperloom_multibranch.contracts.blackboard.records import BuildNotice, ToolResultPointer
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.toolbox.hosted_executor import OUTPUTS_DIR
from hyperloom_multibranch.toolbox.leased_toolbox import session_tools
from hyperloom_multibranch.toolbox.tools import ToolJobs
from hyperloom_multibranch.toolbox.tools.base import RESULT_FILE, ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.shared_jobs import SHARED_JOBS_FILE

MODEL = "meta-llama/Llama-3-8B"


def _as_account(accounts: Accounts, uid: int, call: str) -> dict[str, str]:
    source = Path(hyperloom_multibranch.__file__).parents[1]
    code = (
        f"import json, sys\nsys.path.insert(0, {str(source)!r})\n"
        "from hyperloom_multibranch.scheduling.tests import test_peers_other_account as here\n"
        f"print(json.dumps(here.{call}))"
    )
    return json.loads(accounts.run(uid, code).splitlines()[-1])


#: The key of the analysis the first leg ran and the second one is given as it is, and of the one the second one runs.
EARLIER_KEY, LATER_KEY = "profile-r1/op_shares", "profile-r2/op_shares"
#: Where the first leg's test landed, below ``MB-toolbox``.
LANDED = f"{OUTPUTS_DIR}/{REQUEST_ID}/attempt-1"


class _Echo:
    """A tool that answers with the job's directory and counts its runs in ``ran``."""

    name = "echo"
    ran = 0

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        _Echo.ran += 1
        return ToolOutput({"workdir": str(workdir)}, f"answered {request.branch}")


class _Inbox:
    """The blackboard's push of tool results, kept in a list."""

    def __init__(self) -> None:
        self.pointers: list[ToolResultPointer] = []

    async def notify_tool_result(self, pointer: ToolResultPointer) -> None:
        self.pointers.append(pointer)

    async def notify_build(self, notice: BuildNotice) -> None:
        raise AssertionError(f"no build here: {notice}")


async def _share(toolbox: Path, key: str, branch: str) -> ToolResultPointer:
    """``key`` asked by ``branch`` of the session's tool jobs in ``toolbox``, as a toolbox of this account runs them:
    the pointer ``branch`` is pushed."""
    inbox = _Inbox()
    jobs: ToolJobs = session_tools([_Echo()], inbox, toolbox)  # type: ignore[arg-type]
    job = await jobs.share(ToolRequest("echo", branch), key)
    await jobs.wait(job)
    await jobs.close()
    [pointer] = inbox.pointers
    return pointer


def leave_tool_jobs_and_tests(workspace: str) -> dict[str, str]:
    """As the toolbox of a leg leaves them in a new session in ``workspace``: a shared analysis run to the end, and one
    test's bundle archived and landed. Where the session is and the bundle's key."""
    layout = create_session(Path(workspace), Path(MODEL).name, Clock())
    pointer = asyncio.run(_share(layout.toolbox, EARLIER_KEY, "b01"))
    staged = make_bundle(Path(tempfile.mkdtemp()) / "staging")
    verified = verify_bundle(staged.staging, verify_args(staged))
    DirResultArchive(layout.toolbox / "archive").put_bundle(REQUEST_ID, verified.manifest_sha256, staged.staging)
    materialize(verified, layout.toolbox / LANDED)
    return {"root": str(layout.root), "job": pointer.job_id, "manifest": verified.manifest_sha256}


def _digests(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in _files(root)}


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def take_up_tool_jobs_and_tests(left: dict[str, str]) -> dict[str, object]:
    """What a later leg reads of what :func:`leave_tool_jobs_and_tests` ``left``, and adds to it: the earlier analysis
    given as it is, a new one run, the archived bundle stored again and read, and the landed files read."""
    toolbox = SessionLayout(Path(left["root"])).toolbox
    _Echo.ran = 0
    given = asyncio.run(_share(toolbox, EARLIER_KEY, "b02"))
    earlier = json.loads(Path(given.location).read_text(encoding="utf-8"))
    ran = asyncio.run(_share(toolbox, LATER_KEY, "b02"))
    archive = DirResultArchive(toolbox / "archive")
    staged = archive.bundle_path(REQUEST_ID, left["manifest"])
    copy = Path(tempfile.mkdtemp()) / "staging"
    copy.mkdir()
    for path in _files(staged):
        (copy / path.relative_to(staged)).parent.mkdir(parents=True, exist_ok=True)
        (copy / path.relative_to(staged)).write_bytes(path.read_bytes())
    stored_again = archive.put_bundle(REQUEST_ID, left["manifest"], copy)
    return {
        "given": given.job_id,
        "earlier_status": earlier["status"],
        "runs": _Echo.ran,
        "ran": ran.job_id,
        "stored_again": stored_again,
        "landed": _digests(toolbox / LANDED),
    }


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_another_account_of_the_group_takes_up_the_tool_jobs_and_tests_one_account_left() -> None:
    with other_accounts() as accounts:
        source = Path(hyperloom_multibranch.__file__).parents[1]
        if any(not directory.stat().st_mode & stat.S_IXOTH for directory in (source, *source.parents)):
            pytest.skip(f"another account cannot reach this checkout's {source}")
        if accounts.python != sys.executable:
            pytest.skip(f"another account cannot run {sys.executable}, whose environment holds the project's packages")
        workspace = accounts.shared_dir("workspace")
        left = _as_account(accounts, accounts.first, f"leave_tool_jobs_and_tests({str(workspace)!r})")
        toolbox = SessionLayout(Path(left["root"])).toolbox
        landed = _digests(toolbox / LANDED)

        later = _as_account(accounts, accounts.second, f"take_up_tool_jobs_and_tests({left!r})")

        assert (later["given"], later["earlier_status"]) == (left["job"], "succeeded"), "given as the first leg left it"
        assert later["runs"] == 1 and later["ran"] != left["job"], "only the new analysis ran"
        assert later["landed"] == landed != {}, "the second account read every file the first one landed"
        bundle = toolbox / "archive" / REQUEST_ID / left["manifest"]
        assert later["stored_again"] == str(bundle), "the second account read the stored bundle to store it again"
        ledger = toolbox / "tool-results" / SHARED_JOBS_FILE
        assert (ledger.stat().st_uid, _mode(ledger)) == (accounts.second, SHARED_FILE_MODE)
        result = toolbox / "tool-results" / left["job"] / RESULT_FILE
        assert (result.stat().st_uid, _mode(result)) == (accounts.first, SHARED_FILE_MODE)
        assert {_mode(path) for path in (bundle, *bundle.rglob("*")) if path.is_dir()} == {BUNDLE_DIR_MODE}
        assert {_mode(path) for path in (toolbox / LANDED).rglob("*") if path.is_dir()} == {LANDED_DIR_MODE}
        assert {_mode(path) for path in _files(toolbox / LANDED)} == {LANDED_FILE_MODE}
