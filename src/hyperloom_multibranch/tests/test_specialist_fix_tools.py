# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
from types import SimpleNamespace

import pytest

from hyperloom.contracts.measurement import AttachmentRef
from hyperloom.hosted.mcp_server import ToolError
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST, RequestRejected
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID, FixCase, read_case
from hyperloom_multibranch.specialist.fix_tools import register_fix_tools

SCRIPT = '#!/bin/bash\ncurl -sf "$HL_BASE_URL/v1/models" >/dev/null || exit 1\nexit 0\n'
DIAGNOSIS = "fp8 scales overflow at 8k context; clamp them before the cast"


class _Writer:
    """Keeps every attachment it is given, addressed by content as the session's blob store does."""

    def __init__(self) -> None:
        self.stored: dict[str, tuple[str, bytes]] = {}

    async def put_attachment(self, attachment_id, kind, data):
        self.stored[attachment_id] = (kind, data)
        return AttachmentRef(attachment_id, kind, hashlib.sha256(data).hexdigest())


class _Server:
    def __init__(self) -> None:
        self.tools = {}

    def register(self, name, *, description, schema, handler, scopes):
        self.tools[name] = SimpleNamespace(schema=schema, handler=handler, scopes=scopes)

    async def call(self, name: str, **arguments):
        context = SimpleNamespace(principal=SimpleNamespace(branch="s1"))
        return await self.tools[name].handler(arguments, context)


@pytest.fixture
def rig():
    server, writer = _Server(), _Writer()
    register_fix_tools(server, attachments=writer)
    return server, writer


async def test_a_case_is_stored_as_the_attachment_a_test_carries(rig):
    server, writer = rig

    answer = await server.call("fix_case", repro_sh=SCRIPT, diagnosis=DIAGNOSIS)

    kind, data = writer.stored[FIX_CASE_ID]
    assert AttachmentRef.from_document(answer) == AttachmentRef(FIX_CASE_ID, kind, hashlib.sha256(data).hexdigest())
    assert kind == "dir"
    assert read_case(data) == FixCase(SCRIPT, DIAGNOSIS)


async def test_the_tool_is_a_specialist_s_and_takes_exactly_a_script_and_a_diagnosis(rig):
    server, _writer = rig
    tool = server.tools["fix_case"]

    assert tool.scopes == frozenset({SCOPE_SPECIALIST})
    assert (set(tool.schema["required"]), tool.schema["additionalProperties"]) == ({"repro_sh", "diagnosis"}, False)


@pytest.mark.parametrize(("script", "diagnosis"), [(SCRIPT, " \n"), ("\t", DIAGNOSIS)])
async def test_a_blank_script_or_diagnosis_is_refused_and_nothing_stored(rig, script, diagnosis):
    server, writer = rig

    with pytest.raises(ToolError, match="needs both"):
        await server.call("fix_case", repro_sh=script, diagnosis=diagnosis)

    assert writer.stored == {}


class _Refusing:
    async def put_attachment(self, attachment_id, kind, data):
        raise RequestRejected("the archive expands past 64 MiB", "attachment_too_large")


async def test_a_case_the_toolbox_refuses_comes_back_as_the_tool_s_error():
    server = _Server()
    register_fix_tools(server, attachments=_Refusing())

    with pytest.raises(ToolError, match=r"rejected \(attachment_too_large\): the archive expands past 64 MiB"):
        await server.call("fix_case", repro_sh=SCRIPT, diagnosis=DIAGNOSIS)
