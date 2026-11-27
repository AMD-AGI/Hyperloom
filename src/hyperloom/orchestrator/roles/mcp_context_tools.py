"""In-process MCP server exposing read-only ``context`` tools."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable

from hyperloom.orchestrator.measurement.runtime_findings import render_runtime_findings
from hyperloom.orchestrator.state.failure_packets import read_failure, recent_failures
from hyperloom.orchestrator.mcp_tool_names import (
    CONTEXT_MCP_SERVER_NAME as MCP_SERVER_NAME,
)
from hyperloom.orchestrator.mcp_tool_names import (
    CONTEXT_TOOL_NAMES,
    CONTEXT_TOOL_QUALIFIED_NAMES,
    qualified_context_tool_name,
)

from .mcp_sdk import build_sdk_mcp_server

log = logging.getLogger(__name__)


@dataclass
class ContextProvider:
    """Read-only accessor over live session context for the pull tools."""

    shared_state: Any
    session_dir: Path
    inbox_reader: Callable[[int], str]
    recent_outcomes_reader: Callable[[int], str]
    running_tasks_reader: Callable[[], str]
    reference_reader: Callable[[str], str]

    def _safe(self, fn: Callable[[], str], label: str) -> str:
        """Invoke a projection callable; an empty result reads as empty."""
        out = fn()
        return out if isinstance(out, str) and out else f"({label}: empty)"

    def gaps(self) -> str:
        """Return the open-gaps summary projection."""
        return self._safe(lambda: self.shared_state.to_gaps_summary(max_attempts=5), "gaps")

    def runtime_findings(self) -> str:
        """Return the runtime findings scanned from the current best's server log."""
        return self._safe(
            lambda: render_runtime_findings(self.shared_state.current_best_measurement),
            "runtime_findings",
        )

    def inbox(self, since_seq: int = 0) -> str:
        """Return inbox messages newer than a sequence number."""
        return self._safe(lambda: self.inbox_reader(since_seq), "inbox")

    def recent_outcomes(self, top_k: int = 8) -> str:
        """Return a summary of recent action outcomes."""
        return self._safe(lambda: self.recent_outcomes_reader(top_k), "recent_outcomes")

    def running_tasks(self) -> str:
        """Return the in-flight task set with the resources each one holds."""
        return self._safe(self.running_tasks_reader, "running_tasks")

    def read_reference(self, name: str = "") -> str:
        """Return the full text of a named on-demand reference document."""
        return self._safe(lambda: self.reference_reader(name), "read_reference")

    def get_failure(self, failure_id: str = "") -> str:
        """Return one failure evidence packet as JSON."""
        fid = str(failure_id or "").strip()
        if not fid:
            return "(get_failure: failure_id is required)"

        def _read() -> str:
            packet = read_failure(self.session_dir, fid)
            if packet is None:
                return f"(get_failure: no failure evidence was written as {fid!r})"
            return json.dumps(packet, default=str, indent=2)

        return self._safe(_read, "get_failure")

    def get_variant_failures(self, task_id: str = "", top_k: int = 10) -> str:
        """Return recent failure evidence packets, newest first."""
        k = max(1, min(top_k, 50))

        def _read() -> str:
            packets = islice(recent_failures(self.session_dir, task_id=task_id), k)
            return "\n".join(json.dumps(packet, default=str) for packet in packets) or "(no failure entries)"

        return self._safe(_read, "get_variant_failures")


# Tool descriptors: (tool_name, description, input_schema, provider-method).
_NO_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}
_TOPK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"top_k": {"type": "integer", "minimum": 1, "maximum": 50}},
    "additionalProperties": False,
}
_SINCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"since_seq": {"type": "integer", "minimum": 0}},
    "additionalProperties": False,
}
_REFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}
_FAILURE_ID_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"failure_id": {"type": "string"}},
    "required": ["failure_id"],
    "additionalProperties": False,
}
_VARIANT_FAILURES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
    },
    "additionalProperties": False,
}
CONTEXT_TOOL_SPECS: tuple[tuple[str, str, dict[str, Any], str], ...] = (
    (
        "get_gaps",
        "Return the structured gaps[] ledger: canonical_id / layer / "
        "severity / symptom / attempt count and up to 5 recent attempts "
        "(each with action, outcome, error_class, and failure_id when present).",
        _NO_ARGS_SCHEMA,
        "gaps",
    ),
    (
        "get_runtime_findings",
        "Return what the current best's server.log shows the server actually "
        "did: unknown env vars, features the framework disabled, capability "
        "fallbacks, settings the engine adjusted, AITER tuned-config misses, "
        "tracebacks. Each rule is detected / not_detected / unknown(reason).",
        _NO_ARGS_SCHEMA,
        "runtime_findings",
    ),
    (
        "get_inbox",
        "Return inbox events addressed to orchestration. Pass since_seq to "
        "page from a given sequence; omit for the full history.",
        _SINCE_SCHEMA,
        "inbox",
    ),
    (
        "get_recent_outcomes",
        "Return the most recent action outcomes — the async results of "
        "prior delegate/request work (delegated_result: kind / state / "
        "status / gain / error) — so "
        "you can close the act->observe loop within this turn instead of "
        "waiting for the next-tick delta. Pass top_k to widen the window.",
        _TOPK_SCHEMA,
        "recent_outcomes",
    ),
    (
        "get_running_tasks",
        "Re-read the turn's Tasks in flight block: each running task and "
        "queued explore round with its elapsed seconds, specialist domain / "
        "gap, idempotency key, lease TTL and remaining time, held lanes, "
        "leased GPU ids, heartbeat age and explore grid. Use it when a "
        "decision later in the turn depends on whether that work has moved.",
        _NO_ARGS_SCHEMA,
        "running_tasks",
    ),
    (
        "read_reference",
        "Return the full text of a named on-demand reference document "
        "listed in the ON-DEMAND REFERENCE INDEX section of this prompt. "
        "Pass the bare stem from that index (e.g. name='failure_recovery'). "
        "Use when an action has just failed or you need the detailed "
        "decision rules for a situation flagged in the index.",
        _REFERENCE_SCHEMA,
        "read_reference",
    ),
    (
        "get_failure",
        "Return the full structured failure evidence packet for a single "
        "failure_id (from a delegated_result failure line or gap attempt). "
        "Carries variant args, stage, error_class, error_excerpt, "
        "server_log_path, and workspace so you can then `Read` the actual "
        "log file.  The packet is the file "
        "`$SESSION_DIR/reports/failures/<failure_id>.json`.",
        _FAILURE_ID_SCHEMA,
        "get_failure",
    ),
    (
        "get_variant_failures",
        "Return the most recent failure evidence entries across the session, "
        "optionally filtered to a single task_id.  Each entry is a JSON "
        "object with failure_id, variant_name, stage, error_class, "
        "error_excerpt, server_log_path, and workspace.  Use this to find "
        f"failure_ids you can then inspect with {qualified_context_tool_name('get_failure')}.",
        _VARIANT_FAILURES_SCHEMA,
        "get_variant_failures",
    ),
)


def _make_handler(
    provider: ContextProvider,
    method_name: str,
) -> Callable[[dict[str, Any]], Any]:
    """Build an async MCP handler returning the provider method's string."""

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        """Invoke the bound provider method and wrap its string result."""
        method = getattr(provider, method_name)
        kwargs: dict[str, Any] = {}
        if isinstance(args, dict):
            if "top_k" in args:
                kwargs["top_k"] = int(args["top_k"])
            if "since_seq" in args:
                kwargs["since_seq"] = int(args["since_seq"])
            if "name" in args:
                kwargs["name"] = str(args["name"])
            if "failure_id" in args:
                kwargs["failure_id"] = str(args["failure_id"])
            if "task_id" in args:
                kwargs["task_id"] = str(args["task_id"])
        try:
            text = method(**kwargs)
        except Exception as exc:
            log.exception("context tool handler %s raised", method_name)
            return {
                "content": [{"type": "text", "text": f"error: {exc!r}"}],
                "is_error": True,
            }
        if not isinstance(text, str):
            text = json.dumps(text, default=str)
        log.info(
            "context_tool pull: %s args=%s -> %d chars",
            method_name,
            kwargs or {},
            len(text),
        )
        return {"content": [{"type": "text", "text": text}]}

    return _handler


def build_context_tools_server(sdk: Any, provider: ContextProvider) -> Any:
    """Build the in-process MCP server exposing the read-only context tools with the imported ``claude_agent_sdk``."""
    return build_sdk_mcp_server(
        sdk,
        MCP_SERVER_NAME,
        [
            (tool_name, description, schema, _make_handler(provider, method_name))
            for tool_name, description, schema, method_name in CONTEXT_TOOL_SPECS
        ],
    )


__all__ = [
    "CONTEXT_TOOL_NAMES",
    "CONTEXT_TOOL_QUALIFIED_NAMES",
    "CONTEXT_TOOL_SPECS",
    "ContextProvider",
    "MCP_SERVER_NAME",
    "build_context_tools_server",
    "qualified_context_tool_name",
]
