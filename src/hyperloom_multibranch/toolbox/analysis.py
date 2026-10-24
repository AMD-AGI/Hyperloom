# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist may start from the MCP service: ``tracelens`` and ``roofline`` on its own branch's commit,
``kernel_tune`` on the landed output of a test, and ``upstream_pr_search`` on an upstream repository, those of them the
toolbox's tool registry serves (:attr:`AnalysisStarter.analysis_tools`).

The caller names the tool and its own parameters. The commit the analysis is about is never the caller's: it is the
head of the caller's branch in the experiment repository, read here. ``profile_request_id`` is kept only when the
notebook holds a PROFILE test of that id on that commit, whichever branch asked for it: a profile describes a commit,
and ``submit_test`` refuses a second profile of a commit with the id of the first (``profile_exists``). The trace and
the trace splitter's workload hints are that profile test's own: the caller names neither. The scheduler's own request,
for the trace the PRELUDE child profiled, is built here too (:func:`root_profile_request`), its hints read from the
child's point config by the same rule. A ``kernel_tune`` request names a test (``test_request_id``) and gets the
directory its output landed in, resolved inside the session, from the notebook's record of it, never from the caller.
An ``upstream_pr_search`` request is about no commit and no test: it names a repository (one the rule pack lists) and what to
look for, and the service the search asks is the host's, never the caller's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hyperloom_multibranch.contracts.notebook.interfaces import RecordSearch, UnknownRequest
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolRequest
from hyperloom_multibranch.toolbox.tools.kernel_tune import KERNEL_TUNE_TOOL, LANDED_PARAM, TEST_PARAM, check_params
from hyperloom_multibranch.toolbox.tools.op_shares import COMMIT_PARAM, PROFILE_REQUEST_PARAM
from hyperloom_multibranch.toolbox.tools.roofline import ROOFLINE_TOOL, TRACELENS_TOOL
from hyperloom_multibranch.toolbox.tools.tracelens_report import REPORT_ROUTE, WORKLOAD_HINTS, split_hint
from hyperloom_multibranch.toolbox.tools.tune_inputs import POINT_CONFIG_GLOB, read_benchmark
from hyperloom_multibranch.toolbox.tools.upstream_pr import UPSTREAM_PR_TOOL, build_search

#: The ceiling of what a specialist may start; the toolbox serves those of them its registry holds, never more.
ANALYSIS_TOOLS = (TRACELENS_TOOL, ROOFLINE_TOOL, KERNEL_TUNE_TOOL, UPSTREAM_PR_TOOL)
CODE_TOOL_NOT_ALLOWED = "tool_not_allowed"
CODE_SERVER_OWNED_PARAM = "server_owned_param"
CODE_COMMIT_UNKNOWN = "commit_unknown"
CODE_PROFILE_UNVERIFIED = "profile_request_unverified"
CODE_PROFILE_REQUIRED = "profile_required"
CODE_NO_EXECUTOR = "no_tool_executor"
CODE_TOOL_UNAVAILABLE = "tool_unavailable_on_backend"
CODE_PROFILE_WITHOUT_TRACE = "profile_without_trace"
CODE_TEST_REQUIRED = "test_required"
CODE_TEST_UNKNOWN = "test_unknown"
CODE_TEST_WITHOUT_OUTPUT = "test_without_output"
TRACE_INPUT_PARAM = "trace_input"
ANALYSIS_ROUTE_PARAM = "analysis_route"
ROOT_PROFILE_JOB_PREFIX = "prelude:"
TRACE_GLOB = "*.trace.json*"
SPLIT_HINT_PARAMS = tuple(key for _, key, _ in WORKLOAD_HINTS)
#: The monitor service an upstream PR search may also read: only the host names one.
PR_MONITOR_PARAM = "pr_monitor_url"


class AnalysisStarter(Protocol):
    """What the MCP route calls; :class:`~hyperloom_multibranch.toolbox.toolbox.RealToolbox` implements it."""

    @property
    def analysis_tools(self) -> tuple[str, ...]:
        """The tools of :data:`ANALYSIS_TOOLS` this toolbox serves: the ones its tool registry holds, in that order."""

    async def submit_analysis(
        self, branch: str, tool: str, params: Mapping[str, Any], profile_request_id: str = ""
    ) -> str:
        """Start ``tool`` for ``branch`` on the branch's head commit and return the job id; raises RequestRejected."""


async def _head(repo: ExperimentRepo, branch: str) -> str:
    try:
        head = await repo.head(branch)
    except Exception as exc:  # the repository knows no such branch, or cannot be read: there is no commit to profile
        raise RequestRejected(
            f"the head commit of branch {branch!r} cannot be read: {exc}", CODE_COMMIT_UNKNOWN
        ) from exc
    if not head:
        raise RequestRejected(f"branch {branch!r} has no head commit", CODE_COMMIT_UNKNOWN)
    return head


async def _verified_profile(
    records: RecordSearch | None, branch: str, commit: str, request_id: str
) -> TestRecord | None:
    """The PROFILE record ``request_id`` of the head ``commit`` of ``branch``, or ``None`` when no id was given.

    What a profile describes is a commit, so any branch's profile of ``commit`` qualifies (``records.profiles`` is the
    same commit identity ``profile_exists`` keys on); the branch that ran it does not matter. A profile of another
    commit, or an id the notebook holds no profile for, is refused.
    """
    if not request_id:
        return None
    found = await records.profiles(commit) if records is not None else ()
    for record in found:
        if record.request_id == request_id:
            return record
    raise RequestRejected(
        f"{request_id!r} is not a profile test of {commit}, the head commit of branch {branch!r}",
        CODE_PROFILE_UNVERIFIED,
    )


def _in_session(location: str | Path, root: Path) -> Path | None:
    """``location`` (relative to the session directory ``root``, or absolute) resolved with symlinks followed; ``None``
    when it does not lie inside ``root``."""
    resolved = (root / location).resolve()  # an absolute location replaces ``root`` in the join, then is checked
    return resolved if resolved.is_relative_to(root.resolve()) else None


def _trace_input(record: TestRecord, root: Path | None) -> str:
    """Where the host reads the trace of a verified PROFILE record: its ``profile_location``.

    The location is relative to the toolbox session directory (``root``), where landed test output lives
    (``tests/<request_id>/...``). A relative or absolute location that does not resolve (symlinks followed) to
    inside ``root`` is refused. Without a ``root`` the location is passed on unchanged.
    """
    location = record.result.profile_location if record.result is not None else ""
    if not location:
        raise RequestRejected(
            f"profile test {record.request_id!r} left no trace location to analyse", CODE_PROFILE_WITHOUT_TRACE
        )
    if root is None:
        return location
    resolved = _in_session(location, root)
    if resolved is None:
        raise RequestRejected(
            f"the trace location of profile test {record.request_id!r} leaves the session", CODE_PROFILE_WITHOUT_TRACE
        )
    return str(resolved)


def _point_envs(path: Path, root: Path) -> Mapping[str, Any]:
    """``benchmark.envs`` of the point YAML at ``path``; empty when it is not a plain file inside ``root`` that parses."""
    try:
        if not path.resolve().is_relative_to(root):
            return {}
    except OSError:
        return {}
    envs = read_benchmark(path).get("envs")
    return envs if isinstance(envs, Mapping) else {}


def point_split_hints(point: Path, root: Path) -> dict[str, str]:
    """The workload hints (the splitter's, and the prompt length the op shares are weighted by), by request key, read from
    the point YAML at ``point`` inside ``root``.

    A point that cannot be read, a missing variable or a value the splitter would not take gives no hint for it, and the
    splitter then picks its window without that hint (a report without the workload has no op shares to weight).
    """
    envs = _point_envs(point, root.resolve())
    found = {key: split_hint(flag, envs[name]) for flag, key, name in WORKLOAD_HINTS if name in envs}
    return {key: hint for key, hint in found.items() if hint is not None}


def _split_hints(record: TestRecord, root: Path | None) -> dict[str, str]:
    """The splitter's hints from the point the PROFILE ``record`` measured.

    The record's landed test directory (``output_dir``, relative to ``root``) holds the point YAML the profile test ran,
    in the signed bundle. A profile test runs one point: a directory without exactly one gives no hint.
    """
    if root is None or record.result is None or not record.result.output_dir:
        return {}
    directory = _in_session(record.result.output_dir, root)
    points = sorted(directory.glob(POINT_CONFIG_GLOB)) if directory is not None else []
    return point_split_hints(points[0], root) if len(points) == 1 else {}


@dataclass(frozen=True)
class AnalysisAsk:
    """What the caller named: the branch (from its token), the tool, the tool's own params and a profile test id."""

    branch: str
    tool: str
    params: Mapping[str, Any] = field(default_factory=dict)
    profile_request_id: str = ""


async def analysis_request(
    repo: ExperimentRepo, records: RecordSearch | None, ask: AnalysisAsk, trace_root: Path | None = None
) -> ToolRequest:
    """The tool request for ``ask.branch``, its ``commit_id`` read from the repository; raises RequestRejected.

    With a verified ``profile_request_id`` the ``trace_input`` is the record's own trace (see :func:`_trace_input`,
    resolved against ``trace_root``); the caller never names one, because the trace must be that profile's. The same
    holds for the ``split_*`` hints tracelens' splitter takes: they are the workload of the point that profile measured
    (see :func:`_split_hints`).
    """
    if ask.tool not in ANALYSIS_TOOLS:
        allowed = f"{', '.join(ANALYSIS_TOOLS[:-1])} and {ANALYSIS_TOOLS[-1]}"
        raise RequestRejected(f"a specialist starts only {allowed} here, not {ask.tool!r}", CODE_TOOL_NOT_ALLOWED)
    owned = sorted(
        {COMMIT_PARAM, PROFILE_REQUEST_PARAM, TRACE_INPUT_PARAM, LANDED_PARAM, *SPLIT_HINT_PARAMS} & set(ask.params)
    )
    if owned:
        raise RequestRejected(
            f"{', '.join(owned)} in params is set by the server (from the branch's own commit, or the test or profile "
            "test it names), not by the caller",
            CODE_SERVER_OWNED_PARAM,
        )
    if ask.tool == KERNEL_TUNE_TOOL:
        return await _tune_request(records, ask, trace_root)
    if ask.tool == UPSTREAM_PR_TOOL:
        return _pr_search_request(ask)
    if not ask.profile_request_id:
        raise RequestRejected(
            f"{ask.tool} reads the trace of a profile test: name the profile test with profile_request_id",
            CODE_PROFILE_REQUIRED,
        )
    commit = await _head(repo, ask.branch)
    record = await _verified_profile(records, ask.branch, commit, ask.profile_request_id)
    params = {**ask.params, COMMIT_PARAM: commit, PROFILE_REQUEST_PARAM: ask.profile_request_id}
    if record is not None:
        params[TRACE_INPUT_PARAM] = _trace_input(record, trace_root)
        if ask.tool == TRACELENS_TOOL:
            params.update(_split_hints(record, trace_root))
    return ToolRequest(ask.tool, ask.branch, params)


def _pr_search_request(ask: AnalysisAsk) -> ToolRequest:
    """The ``upstream_pr_search`` request: the caller's search, checked now so a search that can never run is refused."""
    if ask.profile_request_id:
        raise RequestRejected(f"{UPSTREAM_PR_TOOL} reads no profile test: leave profile_request_id out")
    if PR_MONITOR_PARAM in ask.params:
        raise RequestRejected(
            f"{PR_MONITOR_PARAM} in params is not the caller's to name: the host does not ask a service a caller names",
            CODE_SERVER_OWNED_PARAM,
        )
    try:
        build_search(ask.params)
    except InvalidRequest as exc:
        raise RequestRejected(str(exc)) from None
    return ToolRequest(ask.tool, ask.branch, dict(ask.params))


async def _tune_request(records: RecordSearch | None, ask: AnalysisAsk, root: Path | None) -> ToolRequest:
    """The ``kernel_tune`` request: the caller's parameters and the directory the named test's output landed in.

    The test must be one the notebook holds a result for with an output directory inside the session ``root``; its server
    logs are read by the tool when it runs, with no card held, and a test that landed none is a failure of that job.
    """
    try:
        check_params(ask.params)
    except InvalidRequest as exc:
        raise RequestRejected(str(exc)) from None
    test_id = str(ask.params.get(TEST_PARAM) or "")
    if not test_id:
        raise RequestRejected(
            f"{KERNEL_TUNE_TOOL} reads the output of a test: name it with {TEST_PARAM}", CODE_TEST_REQUIRED
        )
    if records is None or root is None:
        raise RequestRejected("this toolbox cannot read the landed output of a test", CODE_TEST_WITHOUT_OUTPUT)
    try:
        record = await records.record(test_id)
    except UnknownRequest:
        raise RequestRejected(f"the notebook holds no test {test_id!r}", CODE_TEST_UNKNOWN) from None
    landed = _in_session(record.result.output_dir, root) if record.result and record.result.output_dir else None
    if landed is None or not landed.is_dir():
        raise RequestRejected(f"test {test_id!r} has no landed output to read", CODE_TEST_WITHOUT_OUTPUT)
    return ToolRequest(ask.tool, ask.branch, {**ask.params, LANDED_PARAM: str(landed)})


def root_profile_request(root: Path | None, commit_id: str, trace_dir: Path, point_config: Path) -> ToolRequest:
    """The deterministic ``tracelens`` report of the trace the PRELUDE child profiled, for the integration branch.

    ``trace_dir`` and ``point_config`` must lie inside the session directory ``root``. The splitter's hints are read
    from ``point_config`` by the rule a PROFILE test's own point is read by; the caller names none. Raises
    RequestRejected: ``invalid_request`` outside the session, ``profile_without_trace`` when ``trace_dir`` holds no trace.
    """
    if root is None:
        raise RequestRejected("this toolbox has no session directory to read the root profile from")
    trace, point = _in_session(trace_dir, root), _in_session(point_config, root)
    if trace is None or point is None:
        raise RequestRejected("the root profile's trace directory and point config must lie inside the session")
    if next(trace.rglob(TRACE_GLOB), None) is None:
        raise RequestRejected(f"no {TRACE_GLOB} under {trace_dir}", CODE_PROFILE_WITHOUT_TRACE)
    params = {
        ANALYSIS_ROUTE_PARAM: REPORT_ROUTE,
        COMMIT_PARAM: commit_id,
        PROFILE_REQUEST_PARAM: "",
        TRACE_INPUT_PARAM: str(trace),
        **point_split_hints(point, root),
    }
    return ToolRequest(TRACELENS_TOOL, INTEGRATION_BRANCH, params)


__all__ = [
    "ANALYSIS_TOOLS",
    "ROOT_PROFILE_JOB_PREFIX",
    "AnalysisAsk",
    "AnalysisStarter",
    "analysis_request",
    "root_profile_request",
]
