"""In-process MCP server exposing read-only ``context`` tools.

Plan Step 2 — "context by pull, not push". In the persistent-conversation
ReAct design the Orchestration agent no longer receives a full
``SharedState`` dump every tick (that was the stateless-reactor model).
Instead the per-tick prompt is a thin *delta* and the agent **pulls** the
context it actually needs via these read-only tools.

Each tool simply re-exposes an existing ``SharedState.to_*_summary``
projection (the same text the old prompt pushed), so there is no new
serialization logic to drift — the projections remain the single source
of truth. A few tools (inbox, analysis.md) read through small callables
the Coordinator injects because they need the bus / session_dir.

Design mirrors :mod:`.mcp_emit_intent`:

* In-process MCP server (zero extra processes, synchronous lookup).
* Test seams: ``build_context_tools_server`` accepts factory overrides so
  tests don't have to import ``claude_agent_sdk``.

Unlike ``emit_intent`` (whose handler only validates + acks because the
intent is captured from the trajectory), these handlers **return the real
data** to the model — the return value IS the tool result.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)


MCP_SERVER_NAME = "inference_optimizer_context"


def _qualified(tool_name: str) -> str:
    return f"mcp__{MCP_SERVER_NAME}__{tool_name}"


@dataclass
class ContextProvider:
    """Read-only accessor over live session context for the pull tools.

    The Coordinator constructs one of these bound to its live
    ``SharedState`` (and small callables for bus / analysis reads) and
    passes it to :func:`build_context_tools_server`. Every method returns
    a plain string (the projection text) so the MCP handlers can hand it
    straight back to the model.

    All methods are defensive: a projection failure returns a short error
    string rather than raising, because a context-pull must never crash
    the reactor turn.
    """

    shared_state: Any
    # Optional callables injected by the Coordinator for context that
    # lives outside SharedState. Each takes no positional args beyond the
    # documented kwargs and returns a string. ``None`` => feature absent.
    inbox_reader: Callable[[int], str] | None = None
    analysis_reader: Callable[[], str] | None = None
    denial_reader: Callable[[int], str] | None = None

    def _safe(self, fn: Callable[[], str], label: str) -> str:
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001 — never crash a pull
            log.exception("context tool %s failed", label)
            return f"(context tool {label} unavailable: {exc!r})"
        return out if isinstance(out, str) and out else f"({label}: empty)"

    # -- projections backed by SharedState.to_*_summary --------------
    def mission_status(self) -> str:
        return self._safe(self.shared_state.to_mission_summary, "mission_status")

    def shared_state_summary(self) -> str:
        return self._safe(self.shared_state.to_prompt_summary, "shared_state")

    def gaps(self) -> str:
        return self._safe(self.shared_state.to_gaps_summary, "gaps")

    def warm_start(self) -> str:
        return self._safe(self.shared_state.to_warm_start_summary, "warm_start")

    def proposal_scores(self) -> str:
        return self._safe(
            self.shared_state.to_proposal_scores_summary, "proposal_scores"
        )

    def intervention_mix(self) -> str:
        return self._safe(
            self.shared_state.to_intervention_mix_summary, "intervention_mix"
        )

    def why_denied(self, top_k: int = 6) -> str:
        if self.denial_reader is not None:
            return self._safe(lambda: self.denial_reader(top_k), "why_denied")
        return self._safe(
            lambda: self.shared_state.to_policy_denial_summary(top_k=top_k),
            "why_denied",
        )

    def analysis_md(self) -> str:
        if self.analysis_reader is None:
            return "(analysis.md reader not wired)"
        return self._safe(self.analysis_reader, "analysis_md")

    def inbox(self, since_seq: int = 0) -> str:
        if self.inbox_reader is None:
            return "(inbox reader not wired)"
        return self._safe(lambda: self.inbox_reader(since_seq), "inbox")


# Tool descriptors: (tool_name, description, input_schema, provider-method).
# input_schema follows the JSON-schema subset the SDK accepts. Methods are
# resolved by name at server-build time so the descriptor table stays
# declarative.
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


CONTEXT_TOOL_SPECS: tuple[tuple[str, str, dict[str, Any], str], ...] = (
    (
        "get_mission_status",
        "Return the mission-progress snapshot: baseline, current best, "
        "raw vs validated cumulative gain, optimization-stack freshness, "
        "untried hot kernels, and time budget.",
        _NO_ARGS_SCHEMA,
        "mission_status",
    ),
    (
        "get_shared_state",
        "Return the full shared session-state summary (the verbose "
        "execution detail: best config, stack entries, per-action attempt "
        "history, discovered flags, warnings).",
        _NO_ARGS_SCHEMA,
        "shared_state_summary",
    ),
    (
        "get_gaps",
        "Return the structured gaps[] ledger: canonical_id / layer / "
        "severity / symptom / attempts for each open performance gap.",
        _NO_ARGS_SCHEMA,
        "gaps",
    ),
    (
        "get_warm_start",
        "Return the Cortex T0 warm-start snapshot (cross-session priors: "
        "what worked / failed in prior runs on a similar stack).",
        _NO_ARGS_SCHEMA,
        "warm_start",
    ),
    (
        "get_proposal_scores",
        "Return the advisory multi-rater proposal scores for the most "
        "recent specialist round (0-10 likelihood-of-gain priors). "
        "Advisory only — never a ranking directive.",
        _NO_ARGS_SCHEMA,
        "proposal_scores",
    ),
    (
        "get_intervention_mix",
        "Return the config-vs-code_patch intervention-mix telemetry "
        "(how many config keeps vs source-patch keeps so far).",
        _NO_ARGS_SCHEMA,
        "intervention_mix",
    ),
    (
        "why_denied",
        "Return the most recent PolicyGate denials (action / rule / hint) "
        "so you can see why a proposed intent was rejected and self-correct.",
        _TOPK_SCHEMA,
        "why_denied",
    ),
    (
        "show_analysis_md",
        "Return the latest TraceLens analysis.md snapshot (executive "
        "summary, top operations with kernel_ids, recommendations, "
        "priority markers).",
        _NO_ARGS_SCHEMA,
        "analysis_md",
    ),
    (
        "get_inbox",
        "Return inbox events addressed to orchestration. Pass since_seq to "
        "page from a given sequence; omit for the recent tail.",
        _SINCE_SCHEMA,
        "inbox",
    ),
)


CONTEXT_TOOL_NAMES: tuple[str, ...] = tuple(s[0] for s in CONTEXT_TOOL_SPECS)
CONTEXT_TOOL_QUALIFIED_NAMES: tuple[str, ...] = tuple(
    _qualified(n) for n in CONTEXT_TOOL_NAMES
)


def _resolve_sdk(sdk_module: Any | None) -> Any | None:
    if sdk_module is not None:
        return sdk_module
    try:
        return importlib.import_module("claude_agent_sdk")
    except ImportError:
        return None


def _make_handler(
    provider: ContextProvider, method_name: str,
) -> Callable[[dict[str, Any]], Any]:
    """Build an async MCP handler that calls the provider method and
    returns its string result as the tool result content."""

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        method = getattr(provider, method_name)
        kwargs: dict[str, Any] = {}
        if isinstance(args, dict):
            if "top_k" in args:
                kwargs["top_k"] = int(args["top_k"])
            if "since_seq" in args:
                kwargs["since_seq"] = int(args["since_seq"])
        try:
            text = method(**kwargs)
        except Exception as exc:  # noqa: BLE001 — never crash a pull
            log.exception("context tool handler %s raised", method_name)
            return {
                "content": [{"type": "text", "text": f"error: {exc!r}"}],
                "is_error": True,
            }
        if not isinstance(text, str):
            text = json.dumps(text, default=str)
        # Observability (plan Step 2): make on-demand context pulls visible
        # so we can confirm the agent actually pulls context instead of
        # relying on a full state push.
        log.info(
            "context_tool pull: %s args=%s -> %d chars",
            method_name, kwargs or {}, len(text),
        )
        return {"content": [{"type": "text", "text": text}]}

    return _handler


def build_context_tools_server(
    provider: ContextProvider,
    *,
    sdk_module: Any | None = None,
    tool_factory: Callable[..., Any] | None = None,
    server_factory: Callable[..., Any] | None = None,
) -> Any | None:
    """Build the in-process MCP server exposing the read-only context tools.

    Returns the SDK ``McpSdkServerConfig`` for
    :class:`ClaudeAgentOptions.mcp_servers`, or ``None`` if the SDK lacks
    in-process MCP helpers (handled gracefully by the caller).
    """
    sdk = _resolve_sdk(sdk_module)
    if tool_factory is None:
        tool_factory = getattr(sdk, "tool", None) if sdk is not None else None
    if server_factory is None:
        server_factory = (
            getattr(sdk, "create_sdk_mcp_server", None) if sdk is not None else None
        )
    if tool_factory is None or server_factory is None:
        log.info(
            "context-tools MCP server unavailable (sdk=%s).",
            getattr(sdk, "__name__", "<none>"),
        )
        return None

    decorated_tools = []
    for tool_name, description, schema, method_name in CONTEXT_TOOL_SPECS:
        decorator = tool_factory(tool_name, description, schema)
        decorated_tools.append(decorator(_make_handler(provider, method_name)))
    return server_factory(MCP_SERVER_NAME, "1.0.0", decorated_tools)


__all__ = [
    "CONTEXT_TOOL_NAMES",
    "CONTEXT_TOOL_QUALIFIED_NAMES",
    "CONTEXT_TOOL_SPECS",
    "ContextProvider",
    "MCP_SERVER_NAME",
    "build_context_tools_server",
]
