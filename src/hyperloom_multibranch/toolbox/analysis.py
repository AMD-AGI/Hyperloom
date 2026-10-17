# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist may start from the MCP service: ``tracelens`` and ``roofline`` on its own branch's commit.

The caller names the tool and its own parameters. The commit the analysis is about is never the caller's: it is the
head of the caller's branch in the experiment repository, read here. ``profile_request_id`` is kept only when the
notebook holds a PROFILE test of that id, asked by the same branch, on that commit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hyperloom_multibranch.contracts.notebook.interfaces import RecordSearch
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.toolbox.tools.base import ToolRequest
from hyperloom_multibranch.toolbox.tools.op_shares import COMMIT_PARAM, PROFILE_REQUEST_PARAM
from hyperloom_multibranch.toolbox.tools.roofline import ROOFLINE_TOOL, TRACELENS_TOOL

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


class AnalysisStarter(Protocol):
    """What the MCP route calls; :class:`~hyperloom_multibranch.toolbox.toolbox.RealToolbox` implements it."""

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
    """The PROFILE record ``request_id`` (branch and head commit checked), or ``None`` when no id was given."""
    if not request_id:
        return None
    found = await records.profiles(commit) if records is not None else ()
    for record in found:
        if record.request_id == request_id and record.request.branch_id == branch:
            return record
    raise RequestRejected(
        f"{request_id!r} is not a profile test of branch {branch!r} on its head commit {commit}",
        CODE_PROFILE_UNVERIFIED,
    )


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
    resolved = (root / location).resolve()  # an absolute location replaces ``root`` in the join, then is checked
    if not resolved.is_relative_to(root.resolve()):
        raise RequestRejected(
            f"the trace location of profile test {record.request_id!r} leaves the session", CODE_PROFILE_WITHOUT_TRACE
        )
    return str(resolved)


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
    resolved against ``trace_root``); the caller never names one, because the trace must be that profile's.
    """
    if ask.tool not in ANALYSIS_TOOLS:
        raise RequestRejected(
            f"a specialist starts only {' and '.join(ANALYSIS_TOOLS)} here, not {ask.tool!r}", CODE_TOOL_NOT_ALLOWED
        )
    owned = sorted({COMMIT_PARAM, PROFILE_REQUEST_PARAM, TRACE_INPUT_PARAM} & set(ask.params))
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
    return ToolRequest(ask.tool, ask.branch, params)


__all__ = ["ANALYSIS_TOOLS", "AnalysisAsk", "AnalysisStarter", "analysis_request"]
