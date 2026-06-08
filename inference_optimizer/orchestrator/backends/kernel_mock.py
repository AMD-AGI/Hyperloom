# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Mock Kernel backend — auto-responds to every REQUEST it sees.

Used in P0 main-path tests so Coordinator + Orchestration can exercise the
Plan A REQUEST/RESPONSE protocol without a real Claude Kernel agent.

Behaviour:

* When the prompt contains an inbox row with ``topic=request`` and a
  parseable ``msg_id`` + ``kind``, emit ``response{in_reply_to=msg_id,
  kind=kind, status="ok", result={"source": "mock"}}``.
* Multiple requests in one inbox window → one response per request.
* Otherwise emit a heartbeat.
"""

from __future__ import annotations

import re
from typing import Any

from ...protocol.intent import Intent, IntentType
from .base import BackendTurnResult


# Coordinator renders inbox rows as:
#   seq=N msg_id=<hex32> from=orchestration topic=request payload={'target_agent': 'kernel', 'kind': 'trace_analyze', ...}
_REQUEST_RE = re.compile(
    r"^\s*seq=(\d+)\s+msg_id=([a-f0-9]+)\s+from=(\w+)\s+topic=request\s+payload=(.*)$",
    re.MULTILINE,
)
_KIND_RE = re.compile(r"['\"]kind['\"]\s*:\s*['\"]([\w-]+)['\"]")


class MockKernelBackend:
    """Auto-respond Kernel adapter. Implements :class:`Backend`."""

    def __init__(self, name: str = "kernel-mock"):
        """Initialise the mock Kernel backend.

        Args:
            name (str): Human-readable backend name used in logs and metadata.
        """
        self.name = name
        self.calls: list[dict[str, Any]] = []
        # Don't double-respond if the same request appears in two consecutive
        # inbox windows (which can happen during reactor fan-out).
        self._answered_msg_ids: set[str] = set()

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Auto-respond to every visible request, else emit a heartbeat.

        Scans the rendered inbox in ``prompt`` for request rows and emits one
        ``response`` with ``status="ok"`` per not-yet-answered request, echoing
        the request ``kind`` as ``<kind>_done``. When no request is visible,
        emits a single heartbeat message.

        Args:
            prompt (str): The composed turn prompt containing the rendered inbox.
            system_prompt (str | None): Unused; accepted for protocol parity.
            tools (list[str] | None): Unused; accepted for protocol parity.
            max_turns (int): Unused; accepted for protocol parity.

        Returns:
            BackendTurnResult: The response and/or heartbeat intents for this
            turn.
        """
        self.calls.append({"prompt": prompt})
        intents: list[Intent] = []
        for match in _REQUEST_RE.finditer(prompt):
            seq, msg_id, from_agent, raw_payload = match.groups()
            if msg_id in self._answered_msg_ids:
                continue
            kind_match = _KIND_RE.search(raw_payload)
            kind = kind_match.group(1) if kind_match else "unknown"
            self._answered_msg_ids.add(msg_id)
            intents.append(Intent(
                type=IntentType.RESPONSE,
                payload={
                    "in_reply_to": msg_id,
                    "kind": f"{kind}_done",
                    "status": "ok",
                    "result": {"source": "mock", "chosen": ["mock_kernel_1"]},
                },
            ))
        if not intents:
            intents.append(Intent(
                type=IntentType.SEND_MESSAGE,
                payload={"topic": "heartbeat", "body_md": "ok (mock kernel)"},
            ))
        return BackendTurnResult(intents=intents, raw_text="(mock kernel)")


__all__ = ["MockKernelBackend"]
