# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The asynchronous tool contract and the runner that guarantees every tool result reaches the inbox.

A tool is one coroutine: a request in, a JSON-serialisable result out. :class:`ToolJobs` accepts a request and returns
a job id at once; when the tool ends, whatever way it ends (success, failure, cancellation), the runner writes the
result file and pushes a :class:`~hyperloom_multibranch.contracts.blackboard.records.ToolResultPointer` to the branch
that asked. A push the sink refuses is retried and, when it still fails, kept for :meth:`ToolJobs.redeliver`, so a
finished job is never silent.

Work whose answer is the same whoever asks is started once (:meth:`ToolJobs.share`, keyed by the caller): a branch that
asks while the job runs joins it, and the job's pointer then goes to every branch that joined; one that asks after it
succeeded gets its pointer at once, as it is; a failed or cancelled job is not reused. The keys and their jobs are kept in
the results directory (:mod:`.shared_jobs`), so a restarted runner shares what the one before it finished.

A job runs in its own cancel scope, the channel the orchestrator's subprocess helpers watch: a tool that hands its work
to a thread (``run_with_session_kill``) has that thread end the child's process group when the job is cancelled or
the loop goes away, instead of leaving it for the interpreter to wait on at exit.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from hyperloom.common.io import atomic_write_json
from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
from hyperloom.orchestrator.actions.executors import COOPERATIVE_REAP_BUDGET_SEC
from hyperloom_multibranch.contracts.blackboard.records import BuildNotice, ToolResultPointer
from hyperloom_multibranch.toolbox.tools.shared_jobs import SHARED_JOBS_FILE, SharedEntry, SharedJobLedger

log = logging.getLogger(__name__)

RESULT_FILE = "result.json"
_SUMMARY_LIMIT = 400
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_REAP_POLL_SECONDS = 0.05


class JobStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UnknownTool(LookupError):
    """No tool is registered under the requested name."""


class InvalidRequest(ValueError):
    """The request cannot be accepted: a missing field, a bad value or a duplicate job id."""


class ToolFailure(Exception):
    """A tool ran and failed; the message is the reason the branch is told."""


@dataclass(frozen=True)
class ToolRequest:
    """What a specialist asked for: the tool, the branch that gets the answer, and the tool's own parameters."""

    tool: str
    branch: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolOutput:
    """What a tool returns. ``ok`` is False when the tool ran to the end but its own verdict is a failure."""

    result: Mapping[str, Any]
    summary: str
    ok: bool = True
    commit_id: str = ""
    profile_request_id: str = ""
    op_shares: Mapping[str, float] = field(default_factory=dict)


class AsyncTool(Protocol):
    name: str

    # An optional ``validate(request) -> None`` raises :class:`InvalidRequest`; it runs at submit time, so a request
    # that can never work is refused instead of queued.

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        """Run to completion inside ``workdir``; raise :class:`ToolFailure` (or any exception) to fail the job."""


class ToolResultSink(Protocol):
    """The inbox pushes; the blackboard's ``TestRecordSink`` satisfies it."""

    async def notify_tool_result(self, pointer: ToolResultPointer) -> None: ...

    async def notify_build(self, notice: BuildNotice) -> None: ...


@dataclass
class _Job:
    job_id: str
    request: ToolRequest
    workdir: Path
    status: JobStatus = JobStatus.RUNNING
    pointer: ToolResultPointer | None = None
    #: The branches the result goes to, in the order they asked, and those whose push the sink accepted.
    recipients: list[str] = field(default_factory=list)
    delivered: set[str] = field(default_factory=set)
    #: The shared key this job answers (:meth:`ToolJobs.share`), or ``""``.
    key: str = ""
    cancel_requested: bool = False
    scope: CancelScope = field(default_factory=CancelScope)
    inner: asyncio.Task[ToolOutput] | None = None
    task: asyncio.Task[None] | None = None


def _short(text: str) -> str:
    one_line = " ".join(str(text).split())
    return one_line if len(one_line) <= _SUMMARY_LIMIT else one_line[: _SUMMARY_LIMIT - 3] + "..."


def _share_fields(output: ToolOutput | None) -> dict[str, Any]:
    if output is None or not output.op_shares:
        return {}
    return {
        "commit_id": output.commit_id,
        "profile_request_id": output.profile_request_id,
        "op_shares": dict(output.op_shares),
    }


class ToolJobs:
    """Accepts tool requests, runs them in the background and pushes each result to the inbox."""

    def __init__(
        self,
        tools: Sequence[AsyncTool],
        sink: ToolResultSink,
        results_dir: Path,
        *,
        push_attempts: int = 3,
        push_delay_seconds: float = 0.5,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._sink = sink
        self._results_dir = Path(results_dir)
        self._attempts = max(1, push_attempts)
        self._delay = push_delay_seconds
        self._jobs: dict[str, _Job] = {}
        self._shared = SharedJobLedger(self._results_dir / SHARED_JOBS_FILE)
        self._closed = False

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def submit(self, request: ToolRequest, *, job_id: str | None = None) -> str:
        """Accept ``request`` and return its job id; the result is pushed to ``request.branch`` when the tool ends.

        Raises :class:`UnknownTool` or :class:`InvalidRequest`; nothing is started then.
        """
        if self._closed:
            raise InvalidRequest("the tool runner is closed")
        if request.tool not in self._tools:
            raise UnknownTool(f"no tool named {request.tool!r}; known: {', '.join(self.tool_names)}")
        if not request.branch:
            raise InvalidRequest("a tool request names the branch that gets the result")
        validate = getattr(self._tools[request.tool], "validate", None)
        if validate is not None:
            validate(request)
        job_id = job_id or f"{request.tool}-{uuid.uuid4().hex[:12]}"
        if not _JOB_ID.match(job_id):
            raise InvalidRequest(f"job id {job_id!r} is not a plain file name")
        if job_id in self._jobs:
            raise InvalidRequest(f"job id {job_id!r} is already in use")
        job = _Job(job_id, request, self._workdir(job_id), recipients=[request.branch])
        self._jobs[job_id] = job
        job.task = asyncio.get_running_loop().create_task(self._run(job), name=f"tool-job-{job_id}")
        return job_id

    def _workdir(self, job_id: str) -> Path:
        return self._results_dir / quote(job_id, safe="")  # shared storage rejects ":" in names

    async def share(self, request: ToolRequest, key: str) -> str:
        """The job that answers ``key``, for ``request.branch``: a running one is joined and a succeeded one shared (see
        :meth:`join`); without either, ``request`` starts a new job, which ``key`` then names. Returns its id.

        The lookup and the start are one step of the event loop, so callers that ask at once start one job. Raises as
        :meth:`submit` does when it starts one.
        """
        found = self._answering(key)
        if found is not None:
            await self.join(found, request.branch)
            return found
        job_id = self.submit(request)
        self._jobs[job_id].key = key
        self._shared.record(key, SharedEntry(job_id, request.tool, JobStatus.RUNNING.value))
        return job_id

    def _answering(self, key: str) -> str | None:
        """The id of the job ``key`` names when it runs or succeeded; ``None`` when there is none to share."""
        entry = self._shared.get(key)
        if entry is None:
            return None
        job = self._jobs.get(entry.job_id)
        if job is None:  # a job of a runner before this one: only one that succeeded has a result to share
            job = self._adopted(entry) if entry.succeeded else None
        return job.job_id if job is not None and job.status in (JobStatus.RUNNING, JobStatus.SUCCEEDED) else None

    def _adopted(self, entry: SharedEntry) -> _Job:
        """The settled job of an earlier runner, its pointer delivered to the branch it was pushed to then."""
        assert entry.pointer is not None
        branch = entry.pointer.branch
        job = _Job(entry.job_id, ToolRequest(entry.tool, branch), self._workdir(entry.job_id), JobStatus.SUCCEEDED)
        job.pointer, job.recipients, job.delivered = entry.pointer, [branch], {branch}
        self._jobs[job.job_id] = job
        return job

    async def join(self, job_id: str, branch: str) -> None:
        """Send the job's result to ``branch`` too: when it ends, or now, as it is, when it already ended. A branch the
        result already goes to is not sent it twice."""
        job = self._job(job_id)
        if branch in job.recipients:
            return
        job.recipients.append(branch)
        if job.pointer is not None:
            await self._push(job)

    def leave(self, job_id: str, branch: str) -> bool:
        """Stop sending the job's result to ``branch``; the job is cancelled when no branch is left to send it to.
        Returns whether it was cancelled (False for a job that already ended or that other branches still wait on)."""
        job = self._job(job_id)
        if branch in job.recipients:
            job.recipients.remove(branch)
        return False if job.recipients else self.cancel(job_id)

    def has_job(self, job_id: str) -> bool:
        return job_id in self._jobs

    def status(self, job_id: str) -> JobStatus:
        return self._job(job_id).status

    def cancel(self, job_id: str) -> bool:
        """Ask the job to stop; False when it already ended. The branch still gets a pointer, saying cancelled."""
        job = self._job(job_id)
        if job.status is not JobStatus.RUNNING:
            return False
        job.cancel_requested = True
        if job.inner is not None:
            job.inner.cancel()
        return True

    async def close(self) -> None:
        """Cancel every running job, then wait up to one reap budget for the work any job handed to threads to end.

        A job cancelled earlier counts: it may be settled while its thread still reaps the child. Each job's own end
        (its pointer is pushed, saying cancelled) is awaited too; nothing is started afterwards.
        """
        self._closed = True
        jobs = list(self._jobs.values())
        for job in jobs:
            if job.status is JobStatus.RUNNING:
                self.cancel(job.job_id)
        await asyncio.gather(*(job.task for job in jobs if job.task is not None), return_exceptions=True)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + COOPERATIVE_REAP_BUDGET_SEC
        while loop.time() < deadline and any(job.scope.has_listeners for job in jobs):
            await asyncio.sleep(_REAP_POLL_SECONDS)

    async def wait(self, job_id: str) -> JobStatus:
        """Wait until the job ended and its pointer was handed to the sink (or the sink gave up)."""
        job = self._job(job_id)
        if job.task is not None:
            await asyncio.shield(job.task)
        return job.status

    def undelivered(self) -> tuple[str, ...]:
        """Jobs that ended but whose pointer the sink never accepted for some branch it goes to."""
        return tuple(job.job_id for job in self._jobs.values() if _owed(job))

    async def redeliver(self) -> tuple[str, ...]:
        """Push every undelivered pointer again; returns the job ids still undelivered."""
        for job in list(self._jobs.values()):
            if _owed(job):
                await self._push(job)
        return self.undelivered()

    def _job(self, job_id: str) -> _Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise LookupError(f"no job {job_id!r}") from None

    async def _run(self, job: _Job) -> None:
        output: ToolOutput | None = None
        error = ""
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
        else:
            output, error = await self._execute(job)
        job.workdir.mkdir(parents=True, exist_ok=True)
        summary = self._settle(job, output, error)
        record = {
            "job_id": job.job_id,
            "tool": job.request.tool,
            "branch": job.request.branch,
            "status": job.status.value,
            "summary": summary,
            "result": dict(output.result) if output is not None else None,
            "error": error or None,
        }
        atomic_write_json(job.workdir / RESULT_FILE, record)
        job.pointer = ToolResultPointer(
            branch=job.request.branch,
            job_id=job.job_id,
            location=str(job.workdir / RESULT_FILE),
            summary=_short(f"{job.status.value}: {summary}"),
            **_share_fields(output),
        )
        if job.key:
            pointer = job.pointer if job.status is JobStatus.SUCCEEDED else None
            self._shared.record(job.key, SharedEntry(job.job_id, job.request.tool, job.status.value, pointer))
        await self._push(job)

    async def _execute(self, job: _Job) -> tuple[ToolOutput | None, str]:
        job.workdir.mkdir(parents=True, exist_ok=True)
        tool = self._tools[job.request.tool]
        job.inner = asyncio.get_running_loop().create_task(self._in_scope(job, tool))
        try:
            return await job.inner, ""
        except asyncio.CancelledError:
            if job.inner.cancelled() and job.cancel_requested:
                job.status = JobStatus.CANCELLED
                return None, "cancelled"
            raise
        except ToolFailure as exc:
            return None, str(exc) or "the tool failed"
        except Exception as exc:
            log.exception("tool job %s crashed", job.job_id)
            return None, f"{type(exc).__name__}: {exc}"

    @staticmethod
    async def _in_scope(job: _Job, tool: AsyncTool) -> ToolOutput:
        """Run ``tool`` with the job's cancel scope published; the scope fires when the task is cancelled."""
        with use_cancel_scope(job.scope):
            try:
                return await tool.run(job.request, job.workdir)
            except asyncio.CancelledError:
                job.scope.cancel(reason="tool_job_cancelled")
                raise

    @staticmethod
    def _settle(job: _Job, output: ToolOutput | None, error: str) -> str:
        if job.status is JobStatus.CANCELLED:
            return "cancelled before it finished"
        if output is None:
            job.status = JobStatus.FAILED
            return error
        job.status = JobStatus.SUCCEEDED if output.ok else JobStatus.FAILED
        return output.summary

    async def _push(self, job: _Job) -> None:
        """Push the job's pointer to every branch it goes to that has not accepted it, each with its own branch."""
        assert job.pointer is not None
        for branch in [branch for branch in job.recipients if branch not in job.delivered]:
            if await self._pushed(replace(job.pointer, branch=branch)):
                job.delivered.add(branch)
            else:
                log.error("job %s ended but its result could not be pushed; redeliver() will retry", job.job_id)

    async def _pushed(self, pointer: ToolResultPointer) -> bool:
        for attempt in range(1, self._attempts + 1):
            try:
                await self._sink.notify_tool_result(pointer)
            except Exception:
                log.warning("pushing the result of job %s failed (attempt %d)", pointer.job_id, attempt, exc_info=True)
                if attempt < self._attempts:
                    await asyncio.sleep(self._delay)
                continue
            return True
        return False


def _owed(job: _Job) -> bool:
    """Whether the job ended and some branch it goes to has not accepted its pointer."""
    return job.pointer is not None and any(branch not in job.delivered for branch in job.recipients)
