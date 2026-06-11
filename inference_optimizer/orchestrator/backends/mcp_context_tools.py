"""In-process MCP server exposing read-only ``context`` tools.

Plan Step 2 — "context by pull, not push": instead of a full ``SharedState``
dump each tick, the agent pulls context via these tools, each re-exposing an
existing ``SharedState.to_*_summary`` projection (the single source of truth).
Unlike ``emit_intent``, these handlers return the real data as the tool
result. ``build_context_tools_server`` accepts factory overrides for tests.
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
    """Return the fully-qualified MCP tool name.

    Args:
        tool_name: Bare tool name.

    Returns:
        The name prefixed with ``mcp__<server>__``.
    """
    return f"mcp__{MCP_SERVER_NAME}__{tool_name}"


@dataclass
class ContextProvider:
    """Read-only accessor over live session context for the pull tools.

    Bound by the Coordinator to its live ``SharedState``; every method
    returns a plain string and is defensive (a projection failure returns a
    short error string rather than crashing the reactor turn).
    """

    shared_state: Any
    # Optional callables for context outside SharedState; ``None`` => absent.
    inbox_reader: Callable[[int], str] | None = None
    analysis_reader: Callable[[], str] | None = None
    denial_reader: Callable[[int], str] | None = None
    recent_outcomes_reader: Callable[[int], str] | None = None
    # A3: run a whitelisted lane-light action inline (gated side effect, so
    # the Coordinator injects a bridge callable). ``None`` => unavailable.
    action_runner: Callable[[str, dict[str, Any]], str] | None = None

    def _safe(self, fn: Callable[[], str], label: str) -> str:
        """Invoke a projection callable, never letting it crash the reactor.

        Args:
            fn: Zero-argument projection returning a summary string.
            label: Short name used in log and fallback messages.

        Returns:
            The projection output, or a short error/empty marker string when
            ``fn`` raises or returns nothing usable.
        """
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001 — never crash a pull
            log.exception("context tool %s failed", label)
            return f"(context tool {label} unavailable: {exc!r})"
        return out if isinstance(out, str) and out else f"({label}: empty)"

    # Projections backed by SharedState.to_*_summary.
    def mission_status(self) -> str:
        """Return the mission-status summary projection.

        Returns:
            The mission-status summary string.
        """
        return self._safe(self.shared_state.to_mission_summary, "mission_status")

    def shared_state_summary(self) -> str:
        """Return the prompt-oriented shared-state summary projection.

        Returns:
            The shared-state summary string.
        """
        return self._safe(self.shared_state.to_prompt_summary, "shared_state")

    def gaps(self) -> str:
        """Return the open-gaps summary projection.

        Returns:
            The open-gaps summary string.
        """
        return self._safe(self.shared_state.to_gaps_summary, "gaps")

    def warm_start(self) -> str:
        """Return the warm-start summary projection.

        Returns:
            The warm-start summary string.
        """
        return self._safe(self.shared_state.to_warm_start_summary, "warm_start")

    def proposal_scores(self) -> str:
        """Return the proposal-scores summary projection.

        Returns:
            The proposal-scores summary string.
        """
        return self._safe(
            self.shared_state.to_proposal_scores_summary, "proposal_scores"
        )

    def intervention_mix(self) -> str:
        """Return the intervention-mix summary projection.

        Returns:
            The intervention-mix summary string.
        """
        return self._safe(
            self.shared_state.to_intervention_mix_summary, "intervention_mix"
        )

    def why_denied(self, top_k: int = 6) -> str:
        """Return a summary of recent policy denials.

        Args:
            top_k: Maximum number of denial entries to include.

        Returns:
            The denial summary from the denial reader when wired, otherwise the
            shared-state policy-denial projection.
        """
        if self.denial_reader is not None:
            return self._safe(lambda: self.denial_reader(top_k), "why_denied")
        return self._safe(
            lambda: self.shared_state.to_policy_denial_summary(top_k=top_k),
            "why_denied",
        )

    def analysis_md(self) -> str:
        """Return the current ``analysis.md`` contents.

        Returns:
            The analysis text, or a not-wired marker when no reader is bound.
        """
        if self.analysis_reader is None:
            return "(analysis.md reader not wired)"
        return self._safe(self.analysis_reader, "analysis_md")

    def inbox(self, since_seq: int = 0) -> str:
        """Return inbox messages newer than a sequence number.

        Args:
            since_seq: Only messages with a sequence greater than this are
                returned.

        Returns:
            The inbox text, or a not-wired marker when no reader is bound.
        """
        if self.inbox_reader is None:
            return "(inbox reader not wired)"
        return self._safe(lambda: self.inbox_reader(since_seq), "inbox")

    def recent_outcomes(self, top_k: int = 8) -> str:
        """Return a summary of recent action outcomes.

        Args:
            top_k: Maximum number of recent outcomes to include.

        Returns:
            The outcomes summary, or a not-wired marker when no reader is bound.
        """
        if self.recent_outcomes_reader is None:
            return "(recent outcomes reader not wired)"
        return self._safe(
            lambda: self.recent_outcomes_reader(top_k), "recent_outcomes"
        )

    def run_action_now(
        self, action_name: str = "", params: dict[str, Any] | None = None,
    ) -> str:
        """Run a whitelisted lane-light action inline.

        Args:
            action_name: Name of the action to run.
            params: Optional action parameters.

        Returns:
            The action result string, or a not-wired marker when no action
            runner is bound.
        """
        if self.action_runner is None:
            return "(run_action_now not wired)"
        return self._safe(
            lambda: self.action_runner(action_name, dict(params or {})),
            "run_action_now",
        )


# Tool descriptors: (tool_name, description, input_schema, provider-method);
# methods resolved by name at server-build time.
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
_RUN_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_name": {"type": "string"},
        "params": {"type": "object"},
    },
    "required": ["action_name"],
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
    (
        "get_recent_outcomes",
        "Return the most recent action outcomes — the async results of "
        "prior delegate/request work (delegated_result: kind / state / "
        "status / kept / gain / tput / error) plus review verdicts — so "
        "you can close the act->observe loop within this turn instead of "
        "waiting for the next-tick delta. Pass top_k to widen the window.",
        _TOPK_SCHEMA,
        "recent_outcomes",
    ),
    (
        "run_action_now",
        "Run a CHEAP, lane-light action synchronously and get its result "
        "back IN THIS TURN (closes the act->observe loop without waiting "
        "for the next tick). Only a small whitelist of fast, non-GPU / "
        "non-serving actions is eligible; anything heavy must still go "
        "through emit_intent delegate (async). PolicyGate still gates the "
        "run (phase / role / paths). Args: action_name (str), optional "
        "params (object). For deep multi-step investigation, delegate to "
        "a specialist sub-agent instead.",
        _RUN_ACTION_SCHEMA,
        "run_action_now",
    ),
)


CONTEXT_TOOL_NAMES: tuple[str, ...] = tuple(s[0] for s in CONTEXT_TOOL_SPECS)
CONTEXT_TOOL_QUALIFIED_NAMES: tuple[str, ...] = tuple(
    _qualified(n) for n in CONTEXT_TOOL_NAMES
)


def _resolve_sdk(sdk_module: Any | None) -> Any | None:
    """Resolve the Claude Agent SDK module.

    Args:
        sdk_module: Explicit module to use (for tests), or ``None`` to import
            the real SDK.

    Returns:
        The provided or imported SDK module, or ``None`` when it is not
        installed.
    """
    if sdk_module is not None:
        return sdk_module
    try:
        return importlib.import_module("claude_agent_sdk")
    except ImportError:
        return None


def _make_handler(
    provider: ContextProvider, method_name: str,
) -> Callable[[dict[str, Any]], Any]:
    """Build an async MCP handler returning the provider method's string.

    Args:
        provider: The context provider whose method backs the handler.
        method_name: Name of the provider method to invoke per tool call.

    Returns:
        An async MCP handler callable that invokes the bound method and wraps
        its result.
    """

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        """Invoke the bound provider method and wrap its string result.

        Args:
            args: Tool call arguments; recognized keys (``top_k``,
                ``since_seq``, ``action_name``, ``params``) are forwarded.

        Returns:
            An MCP tool result dict carrying the provider method's output.
        """
        method = getattr(provider, method_name)
        kwargs: dict[str, Any] = {}
        if isinstance(args, dict):
            if "top_k" in args:
                kwargs["top_k"] = int(args["top_k"])
            if "since_seq" in args:
                kwargs["since_seq"] = int(args["since_seq"])
            if "action_name" in args:
                kwargs["action_name"] = str(args["action_name"])
            if "params" in args and isinstance(args["params"], dict):
                kwargs["params"] = args["params"]
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
        # Observability: make on-demand context pulls visible.
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

    Args:
        provider: The context provider backing every tool handler.
        sdk_module: Explicit SDK module to use, or ``None`` to import the real
            one.
        tool_factory: Override for the SDK ``tool`` decorator factory (tests).
        server_factory: Override for the SDK ``create_sdk_mcp_server`` factory
            (tests).

    Returns:
        The constructed in-process MCP server config, or ``None`` when the SDK
        lacks the required in-process MCP helpers.
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
