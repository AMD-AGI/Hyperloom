# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist may start from the MCP service: ``tracelens`` and ``roofline`` on its own branch's commit, those
of them the toolbox's tool registry serves (:attr:`AnalysisStarter.analysis_tools`).

The caller names the tool and its own parameters. The commit the analysis is about is never the caller's: it is the
head of the caller's branch in the experiment repository, read here. ``profile_request_id`` is kept only when the
notebook holds a PROFILE test of that id on that commit, whichever branch asked for it: a profile describes a commit,
and ``submit_test`` refuses a second profile of a commit with the id of the first (``profile_exists``). The trace and
the trace splitter's workload hints are that profile test's own: the caller names neither. The scheduler's own request,
for the trace the PRELUDE child profiled, is built here too (:func:`root_profile_request`), its hints read from the
child's point config by the same rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from hyperloom_multibranch.contracts.notebook.interfaces import RecordSearch
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.toolbox.tools.base import ToolRequest
from hyperloom_multibranch.toolbox.tools.op_shares import COMMIT_PARAM, PROFILE_REQUEST_PARAM
from hyperloom_multibranch.toolbox.tools.roofline import ROOFLINE_TOOL, TRACELENS_TOOL
from hyperloom_multibranch.toolbox.tools.tracelens_report import REPORT_ROUTE, WORKLOAD_HINTS, split_hint

#: The ceiling of what a specialist may start; the toolbox serves those of them its registry holds, never more.
ANALYSIS_TOOLS = (TRACELENS_TOOL, ROOFLINE_TOOL)
CODE_TOOL_NOT_ALLOWED = "tool_not_allowed"
CODE_SERVER_OWNED_PARAM = "server_owned_param"
CODE_COMMIT_UNKNOWN = "commit_unknown"
CODE_PROFILE_UNVERIFIED = "profile_request_unverified"
CODE_PROFILE_REQUIRED = "profile_required"
CODE_NO_EXECUTOR = "no_tool_executor"
CODE_TOOL_UNAVAILABLE = "tool_unavailable_on_backend"
CODE_PROFILE_WITHOUT_TRACE = "profile_without_trace"
TRACE_INPUT_PARAM = "trace_input"
ANALYSIS_ROUTE_PARAM = "analysis_route"
ROOT_PROFILE_JOB_PREFIX = "prelude:"
TRACE_GLOB = "*.trace.json*"
SPLIT_HINT_PARAMS = tuple(key for _, key, _ in WORKLOAD_HINTS)
#: The YAML the grid runner wrote for the point a test ran, in the test's landed directory; a profile test runs one.
POINT_CONFIG_GLOB = "variant_*/config.yaml"
_POINT_CONFIG_MAX_BYTES = 1 << 20


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
        if not path.resolve().is_relative_to(root) or path.stat().st_size > _POINT_CONFIG_MAX_BYTES:
            return {}
        envs = (yaml.safe_load(path.read_text(encoding="utf-8")) or {})["benchmark"]["envs"]
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return {}
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
        raise RequestRejected(
            f"a specialist starts only {' and '.join(ANALYSIS_TOOLS)} here, not {ask.tool!r}", CODE_TOOL_NOT_ALLOWED
        )
    owned = sorted({COMMIT_PARAM, PROFILE_REQUEST_PARAM, TRACE_INPUT_PARAM, *SPLIT_HINT_PARAMS} & set(ask.params))
    if owned:
        raise RequestRejected(
            f"{', '.join(owned)} in params is set by the server (from the branch's own commit or its profile test), "
            "not by the caller",
            CODE_SERVER_OWNED_PARAM,
        )
    if ask.tool == TRACELENS_TOOL and not ask.profile_request_id:
        raise RequestRejected(
            f"{TRACELENS_TOOL} reads the trace of a profile test: name the profile test with profile_request_id",
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
