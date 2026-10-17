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
from typing import Any, Protocol

from hyperloom_multibranch.contracts.notebook.interfaces import RecordSearch
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
CODE_NO_EXECUTOR = "no_tool_executor"


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


async def _verified_profile(records: RecordSearch | None, branch: str, commit: str, request_id: str) -> str:
    if not request_id:
        return ""
    found = await records.profiles(commit) if records is not None else ()
    if any(r.request_id == request_id and r.request.branch_id == branch for r in found):
        return request_id
    raise RequestRejected(
        f"{request_id!r} is not a profile test of branch {branch!r} on its head commit {commit}",
        CODE_PROFILE_UNVERIFIED,
    )


@dataclass(frozen=True)
class AnalysisAsk:
    """What the caller named: the branch (from its token), the tool, the tool's own params and a profile test id."""

    branch: str
    tool: str
    params: Mapping[str, Any] = field(default_factory=dict)
    profile_request_id: str = ""


async def analysis_request(repo: ExperimentRepo, records: RecordSearch | None, ask: AnalysisAsk) -> ToolRequest:
    """The tool request for ``ask.branch``, its ``commit_id`` read from the repository; raises RequestRejected."""
    if ask.tool not in ANALYSIS_TOOLS:
        raise RequestRejected(
            f"a specialist starts only {' and '.join(ANALYSIS_TOOLS)} here, not {ask.tool!r}", CODE_TOOL_NOT_ALLOWED
        )
    owned = sorted({COMMIT_PARAM, PROFILE_REQUEST_PARAM} & set(ask.params))
    if owned:
        raise RequestRejected(
            f"{', '.join(owned)} in params is set by the server from the branch's own commit, not by the caller",
            CODE_SERVER_OWNED_PARAM,
        )
    commit = await _head(repo, ask.branch)
    profile = await _verified_profile(records, ask.branch, commit, ask.profile_request_id)
    return ToolRequest(ask.tool, ask.branch, {**ask.params, COMMIT_PARAM: commit, PROFILE_REQUEST_PARAM: profile})


__all__ = ["ANALYSIS_TOOLS", "AnalysisAsk", "AnalysisStarter", "analysis_request"]
