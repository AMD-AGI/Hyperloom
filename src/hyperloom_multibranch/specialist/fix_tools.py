# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Declaring a correctness fix as a tool on the session's MCP service: ``fix_case`` stores the fix's reproduction
case in the session's blob store and returns the attachment the specialist's ``submit_test`` of the fix carries.

The rule gate runs the case on the candidate's parent and on the candidate (``rulegate.fix_case`` is its format); a
candidate whose test carries no case is judged on its performance alone.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol

from hyperloom.contracts.measurement import AttachmentRef
from hyperloom.hosted.mcp_server import CallContext, McpServer, ToolError
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST, RequestRejected
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID, CaseError, pack_case

_TEXT = "string"
_FIX_CASE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "repro_sh": {
            "type": _TEXT,
            "minLength": 1,
            "description": (
                "A bash script run against the test's server ($HL_BASE_URL, serving $HL_MODEL): exit 1 when the "
                "defect occurs, 0 when it does not."
            ),
        },
        "diagnosis": {
            "type": _TEXT,
            "minLength": 1,
            "description": "The defect, its trigger and how the fix removes it.",
        },
    },
    "required": ["repro_sh", "diagnosis"],
    "additionalProperties": False,
}


class AttachmentWriter(Protocol):
    """Stores ``data`` in the session's blob store and returns the reference a test request carries it by; raises
    :class:`RequestRejected` and stores nothing when it refuses the data."""

    async def put_attachment(self, attachment_id: str, kind: Literal["file", "dir"], data: bytes) -> AttachmentRef: ...


def register_fix_tools(server: McpServer, *, attachments: AttachmentWriter) -> None:
    """Register ``fix_case``."""
    server.register(
        "fix_case",
        description=(
            "Store the reproduction case of a correctness fix; pass the returned attachment in the attachments of "
            "the submit_test of the fix."
        ),
        schema=_FIX_CASE,
        handler=_fix_case(attachments),
        scopes=frozenset({SCOPE_SPECIALIST}),
    )


def _fix_case(attachments: AttachmentWriter) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], _ctx: CallContext) -> dict[str, object]:
        try:
            data = pack_case(arguments["repro_sh"], arguments["diagnosis"])
        except CaseError as error:
            raise ToolError(str(error)) from error
        attachment_id, kind = FIX_CASE_ID, "dir"
        try:
            ref = await attachments.put_attachment(attachment_id, kind, data)
        except RequestRejected as exc:
            raise ToolError(f"rejected ({exc.code}): {exc.reason}") from exc
        return ref.to_document()

    return handle


__all__ = ["AttachmentWriter", "register_fix_tools"]
