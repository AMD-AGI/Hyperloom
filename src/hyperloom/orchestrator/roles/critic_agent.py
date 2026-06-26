# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CriticAgentBackend — bridges the ``hyperloom.agents.critic`` runtime into
the Coordinator as a real Critic Backend.

Runs the two-phase loop from ``src/hyperloom/agents/critic/AGENTS.md``
(prepare-review → Codex review.json → commit-review), giving KB priors,
per-session memory, review_constraints injection, and emergency fallbacks.
The returned envelope is re-validated locally so malformed replies surface as
backend-tagged errors. ``codex_client_factory`` / ``runtime_caller_factory``
are test seams.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from hyperloom.common.jsonio import extract_first_json_with_key
from hyperloom.inference_optimizer.protocol.intent import (
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from hyperloom.inference_optimizer.session.session_paths import allocate_turn_workdir, manifest_path
from ..trace.conversation_trace import ConversationRecord, append_conversation
from ..trace.llm_trace import LLMCallRecord, append_llm_call
from .base import BackendError, BackendTurnResult, build_chat_messages, parse_call_timeout_env
from ._runtime_bridge import RuntimeCall, RuntimeCaller, invoke_runtime_cli


if TYPE_CHECKING:  # pragma: no cover
    from hyperloom.agents.critic.runtime.web_tools import WebToolClients, WebToolsConfig


log = logging.getLogger(__name__)


CRITIC_AGENT_RUNTIME_TIMEOUT_SEC = 30  # prepare-review / commit-review wall cap
CRITIC_AGENT_MAX_COMPLETION_TOKENS = 2000

# Cap on per-turn workdirs kept on disk; older ones are pruned each turn.
CRITIC_AGENT_WORKDIR_KEEP_COUNT = 50

# Output instructions for the exact ``review.json`` shape commit-review validates.
_REVIEW_OUTPUT_INSTRUCTIONS = """
==== OUTPUT FORMAT (REQUIRED) ====
Reply with EXACTLY ONE JSON object that matches this review schema:

{
  "review_verdicts": [
    {
      "target_proposal_msg_id": "<the proposal's msg_id from the bundle>",
      "verdict": "approve" | "reject" | "redirect" | "advise" | "needs_review",
      "source": "critic" | "critic_unavailable",
      "reasoning": "<short, explicit reasoning>",
      "confidence": "low" | "medium" | "high",
      "predicted_gain_pct": <number or null>,
      "kb_evidence": ["<kb_id>", ...],
      "packet_evidence": ["<dotted.path.in.packet>", ...],
      "risks": [{"severity": "blocker|major|minor", "summary": "..."}],
      "required_evidence": ["<key>", ...],
      "notes": ["..."],
      "persist_to_kb": false,
      "topic": "<slug>"
    }
  ],
  "advice": [
    { "target_proposal_msg_id": "<msg_id>", "body_md": "..." }
  ]
}

Rules (mirror SKILL.md Hard Rules + Approve Standard):
- Wrap the JSON in a ```json fenced block. Bare JSON is also accepted.
- Free text outside the JSON is ignored.
- Emit one verdict object PER proposal in `judge_bundle.proposals`.
- If `judge_bundle.required_context` is non-empty, every verdict MUST be
  `needs_review` with `source = "critic_unavailable"` and list the
  missing keys in `notes`.
- If `judge_bundle.kb_read_skipped_reason == "kb_unreachable"`, prefer
  `advise` / `needs_review` over `approve` and mention the missing KB
  recall in `notes`.
- If there are no proposals, return `{"review_verdicts": []}` — the
  runtime falls back to a heartbeat.
- `approve` requires comparable before/after benchmark, accuracy gate
  (or waiver), active-path proof when relevant, and a clear rollback.
- If `review_constraints.known_actions` is non-empty, any
  `alternative_action` MUST be drawn from it; otherwise omit
  `alternative_action`.
==== END OUTPUT FORMAT ====
""".strip()


# Bare {...} fallback carrying "review_verdicts" (fenced case handled by helper).
_BARE_JSON_RE = re.compile(r"(\{[^{}]*\"review_verdicts\"[\s\S]*\})", re.DOTALL)


def _extract_review_json(text: str) -> dict[str, Any] | None:
    """Pull the first valid ``{"review_verdicts": ...}`` object out of a reply."""
    return extract_first_json_with_key(text, "review_verdicts", _BARE_JSON_RE)


def _assistant_message_with_tool_calls(msg: Any) -> dict[str, Any]:
    """Re-serialize an OpenAI assistant message that issued tool_calls
    (minimal dict shape, pydantic v1/v2 compatible).

    Args:
        msg: The OpenAI assistant message object carrying ``content`` and
            ``tool_calls``.

    Returns:
        A minimal assistant-message dict with role, content, and a normalized
        ``tool_calls`` list suitable for re-sending to the API.
    """
    return {
        "role": "assistant",
        "content": getattr(msg, "content", None),
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (getattr(msg, "tool_calls", None) or [])
            if tc.function is not None
        ],
    }


def _default_runtime_caller(call: RuntimeCall) -> None:
    """Real implementation — runs ``python -m hyperloom.agents.critic.runtime.cli <phase> ...``.

    Args:
        call (RuntimeCall): The invocation descriptor with phase, request /
            review / output paths, working directory, and subprocess env.

    Raises:
        BackendError: If a ``commit-review`` call is missing its review path,
            the subprocess times out, cannot start, or exits non-zero.
    """
    extra_args: list[str] = []
    if call.phase == "commit-review":
        if call.review_path is None:
            raise BackendError("commit-review invocation missing --review path")
        extra_args = ["--review", str(call.review_path)]

    # Exit codes: 0 success; 2 adapter bug (host → needs_review).
    invoke_runtime_cli(
        call,
        module="hyperloom.agents.critic.runtime.cli",
        agent_label="critic-agent",
        timeout_sec=CRITIC_AGENT_RUNTIME_TIMEOUT_SEC,
        extra_args=extra_args,
    )


# Cross-domain enrichment helper for cross-domain (scope=domains) proposals.
def _reviewed_msg_ids_from_bundle(judge_bundle: dict[str, Any]) -> list[str] | None:
    """Pull the proposal ``msg_id``s out of a judge bundle, or ``None``.

    The bundle's ``proposals`` is a list of proposal dicts each carrying a
    ``msg_id`` (see critic-agent ``inbox_parser.Proposal``). Returns the
    de-duplicated, order-preserving list of non-empty ids, or ``None`` when the
    bundle carries none — so a non-review turn leaves the trace field unset.
    """
    proposals = judge_bundle.get("proposals") if isinstance(judge_bundle, dict) else None
    if not isinstance(proposals, list):
        return None
    out: list[str] = []
    seen: set[str] = set()
    for p in proposals:
        if not isinstance(p, dict):
            continue
        mid = str(p.get("msg_id") or "").strip()
        if mid and mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out or None


def _proposal_scope_literal(proposal: dict[str, Any]) -> str:
    """Read the ``scope`` dial off a proposal (top-level or nested ``params``).

    Args:
        proposal: A proposal dict that may carry ``scope`` at the top level or
            under ``params``.

    Returns:
        The stripped scope string, or an empty string when absent or the
        proposal is not a dict.
    """
    if not isinstance(proposal, dict):
        return ""
    top = proposal.get("scope")
    if isinstance(top, str) and top.strip():
        return top.strip()
    params = proposal.get("params") or {}
    if isinstance(params, dict):
        nested = params.get("scope")
        if isinstance(nested, str):
            return nested.strip()
    return ""


def _verdict_references_kb(review: dict[str, Any] | None) -> bool:
    """Whether any final review verdict cites KB evidence.

    Scans ``review_verdicts[].kb_evidence`` for a truthy reference. Used by the
    KB trace to record whether the decision actually leaned on KB data.

    Args:
        review (dict[str, Any] | None): The parsed review object.

    Returns:
        bool: ``True`` if at least one verdict references KB evidence.
    """
    if not isinstance(review, dict):
        return False
    for v in review.get("review_verdicts") or []:
        if isinstance(v, dict) and v.get("kb_evidence"):
            return True
    return False


def _maybe_inject_cross_domain_constraints(judge_bundle: dict[str, Any]) -> None:
    """Set ``review_constraints.cross_domain`` + rule descriptors when any
    proposal is cross-domain (unified ``scope == 'domains'`` dial). Idempotent.

    Args:
        judge_bundle: The judge bundle to enrich in place; its
            ``review_constraints`` are updated when a cross-domain proposal is
            present.
    """
    proposals = judge_bundle.get("proposals") or []
    if not isinstance(proposals, list):
        return
    from ..specialists.patch_safety import (
        SCOPE_DOMAINS_LITERAL,
        cross_domain_rule_descriptors,
    )

    has_cross_domain = any(_proposal_scope_literal(p) == SCOPE_DOMAINS_LITERAL for p in proposals)
    if not has_cross_domain:
        return
    rc = judge_bundle.setdefault("review_constraints", {})
    if not isinstance(rc, dict):
        rc = {}
        judge_bundle["review_constraints"] = rc
    rc["cross_domain"] = True
    rc["cross_domain_rules"] = cross_domain_rule_descriptors()


@dataclass
class CriticAgentBackend:
    """Real Critic backend that drives the critic-agent runtime.

    Parameters
    ----------
    critic_agent_root:
        Directory containing ``runtime/cli.py`` (``src/hyperloom/agents/critic/``).
        The CLI is invoked as the package-qualified
        ``python -m hyperloom.agents.critic.runtime.cli``, resolved via the
        normal installed ``hyperloom`` namespace; ``cwd=critic_agent_root``
        is still set so relative asset reads (``SKILL.md``, ``actions/*.md``)
        resolve without an absolute path.
    session_dir:
        Coordinator session directory. Used to scope per-turn workdirs
        and the per-session critic memory store.
    codex_model:
        OpenAI / Codex chat-completion model id (e.g. ``gpt-5.4``).
    codex_client_factory:
        Optional callable returning an ``AsyncOpenAI``-compatible client.
        Mirrors :class:`CodexBackend.client_factory` for tests.
    kb_mode:
        ``inmemory`` (default) keeps KB writes / reads off the wire so
        the optimizer doesn't need a real KB service. ``live`` requires
        ``kb_env`` (or the surrounding process env) to provide
        ``KB_BASE_URL``.
    kb_env:
        Extra env vars merged into the runtime.cli subprocess env when
        ``kb_mode == "live"``. Caller is responsible for filling
        ``KB_BASE_URL`` / ``KB_TIMEOUT_MS`` / ``KB_DEAD_LETTER_DIR`` etc.
    cortex_kb_url:
        Optional cortex kb-service base URL (the ``--cortex-kb-url`` flag).
        When set, it is exported as ``CORTEX_KB_URL`` into the runtime.cli
        subprocess env (unless already present) so the critic runtime's
        optional ``/v2/reasoning/assess`` enrichment can reach the same KB
        recipe-snapshot uses. ``None`` leaves env-based config untouched.
    runtime_caller_factory:
        Test seam returning a :data:`RuntimeCaller`. Tests override this
        to bypass the real Python subprocess.
    static_context:
        Optional explicit per-session context injected as
        ``request.context``. When ``None``, derived from ``manifest.json``
        under ``session_dir``; ``{}`` is a valid "no context" override.
    name:
        Backend instance name surfaced in the Coordinator startup banner.
    """

    critic_agent_root: Path
    session_dir: Path
    codex_model: str = "gpt-5.4"
    codex_client_factory: Callable[[], Any] | None = None
    kb_mode: Literal["inmemory", "live"] = "inmemory"
    kb_env: dict[str, str] | None = None
    cortex_kb_url: str | None = None
    runtime_caller_factory: Callable[[], RuntimeCaller] | None = None
    static_context: dict[str, Any] | None = None
    known_actions: tuple[str, ...] = ()
    # Per-action verdict policy (action_name -> archival/exploration/promotion)
    # enriched onto ``review_constraints.action_verdict_policy`` post prepare-review.
    action_verdict_policy: dict[str, str] = field(default_factory=dict)
    # Substrate KB assess injection switch. ``None`` reads the
    # ``CORTEX_KB_ASSESS_INJECT`` env at turn time (default OFF = dry-run: the
    # assess verdicts are still fetched + traced, but withheld from the LLM
    # prompt so they don't steer the decision). ``True``/``False`` force it.
    kb_assess_inject: bool | None = None
    name: str = "critic-agent"
    # Optional web tools (web_search / web_fetch). ``web_tools_config``
    # None reads from env; ``web_tool_clients_factory`` injects test clients.
    web_tools_config: "WebToolsConfig | None" = None
    web_tool_clients_factory: Callable[["WebToolsConfig"], "WebToolClients"] | None = None

    # Runtime state. ``_runtime_caller`` is assigned on the instance in
    # __post_init__ (not as a dataclass field) to avoid descriptor binding
    # of a module-level function as a method.
    _client: Any = field(default=None, init=False, repr=False)
    _turn_idx: int = field(default=0, init=False, repr=False)
    # Trace context (tick / phase) the Coordinator sets before each reactor
    # ``run()`` so the critic's self-written llm_calls row carries the same
    # timeline keys the in-process reactor trace would have stamped, instead of
    # relying on ts-window backfill.
    _trace_tick: int | None = field(default=None, init=False, repr=False)
    _trace_phase: str | None = field(default=None, init=False, repr=False)
    # Proposal msg_ids reviewed by the current turn, snapshotted from the
    # judge bundle so the self-written llm_calls row can be attributed (via
    # proposal_task_map) to the decision the reviewed proposal became.
    _trace_reviewed_msg_ids: list[str] | None = field(
        default=None, init=False, repr=False,
    )
    _skill_preamble: str | None = field(default=None, init=False, repr=False)
    _static_context: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _web_tool_clients: Any = field(default=None, init=False, repr=False)
    _web_tool_schemas: list[dict[str, Any]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _web_tool_max_turns: int = field(default=0, init=False, repr=False)
    calls: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate config, wire transports, and resolve static context.

        Normalises paths, verifies ``runtime/cli.py`` exists under
        ``critic_agent_root`` (used as the subprocess ``cwd`` for
        ``runtime.cli`` invocations and to load ``SKILL.md`` / action
        prompts), validates ``kb_mode``, selects the real or test runtime
        caller, constructs the Codex/OpenAI client (or its factory), resolves
        the per-session static context (explicit or from ``manifest.json``), and
        initialises optional web tools.

        Raises:
            BackendError: If ``runtime/cli.py`` is missing, ``kb_mode`` is
                invalid, the OpenAI SDK is unavailable, or no API key is set.
        """
        self.critic_agent_root = Path(self.critic_agent_root)
        self.session_dir = Path(self.session_dir)
        if not (self.critic_agent_root / "runtime" / "cli.py").is_file():
            raise BackendError(
                f"CriticAgentBackend: runtime/cli.py not found under "
                f"{self.critic_agent_root!s} — set CRITIC_AGENT_ROOT or "
                f"check the install"
            )
        if self.kb_mode not in ("inmemory", "live"):
            raise BackendError(f"CriticAgentBackend: kb_mode={self.kb_mode!r} not in {{'inmemory','live'}}")

        if self.runtime_caller_factory is not None:
            # Assign on the instance to avoid descriptor binding as a method.
            object.__setattr__(
                self,
                "_runtime_caller",
                self.runtime_caller_factory(),
            )
        else:
            object.__setattr__(
                self,
                "_runtime_caller",
                _default_runtime_caller,
            )

        if self.codex_client_factory is not None:
            self._client = self.codex_client_factory()
        else:
            try:
                from openai import AsyncOpenAI  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover
                raise BackendError("openai SDK not installed; run `pip install openai>=1.50`") from exc
            # Codex review reasoning speaks the OpenAI protocol, so prefer the
            # OpenAI-side key (split entrypoints); ANTHROPIC_AUTH_TOKEN is the
            # single-gateway fallback.
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            if not api_key:
                raise BackendError(
                    "OPENAI_API_KEY / ANTHROPIC_AUTH_TOKEN not set "
                    "(CriticAgentBackend cannot reach Codex for review reasoning)"
                )
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            try:
                import httpx

                connect_timeout_s = parse_call_timeout_env(
                    "CRITIC_AGENT_LLM_CONNECT_TIMEOUT_S",
                    default=CRITIC_AGENT_LLM_CONNECT_TIMEOUT_SEC,
                )
                rw_timeout_s = parse_call_timeout_env(
                    "CRITIC_AGENT_LLM_RW_TIMEOUT_S",
                    default=CRITIC_AGENT_LLM_RW_TIMEOUT_SEC,
                )
                kwargs["timeout"] = httpx.Timeout(
                    connect=connect_timeout_s,
                    read=rw_timeout_s,
                    write=rw_timeout_s,
                    pool=rw_timeout_s,
                )
            except Exception:
                # Keep a best-effort fallback to SDK defaults when timeout wiring
                # cannot be constructed (e.g., import edge cases).
                pass                
            self._client = AsyncOpenAI(**kwargs)

        # Resolve static per-session context once; absent model/framework keys
        # make prepare-review fall back to needs_review + critic_unavailable.
        if self.static_context is not None:
            self._static_context = dict(self.static_context)
        else:
            self._static_context = self._load_static_context_from_manifest()
        log.info(
            "critic_agent_backend static_context source=%s keys=%s",
            "explicit" if self.static_context is not None else "manifest",
            sorted(self._static_context.keys()),
        )

        # Initialize web tools (no-op by default).
        self._init_web_tools()

    def _init_web_tools(self) -> None:
        """Resolve :class:`WebToolsConfig` + build clients + freeze schemas;
        never raises (failure falls back to no-tool reasoning)."""
        try:
            from hyperloom.agents.critic.runtime.web_tools import (
                WebToolsConfig as _Cfg,
                build_clients as _build_clients,
                build_tool_schemas as _build_schemas,
            )
        except ImportError as exc:
            log.warning(
                "critic_agent_backend: hyperloom.agents.critic.runtime.web_tools not importable (%s); web tools disabled",
                exc,
            )
            return

        cfg = self.web_tools_config or _Cfg.from_env()
        if not cfg.critic_web_tools_enabled:
            log.info("critic_agent_backend: web tools disabled by config")
            return

        try:
            clients = (
                self.web_tool_clients_factory(cfg) if self.web_tool_clients_factory is not None else _build_clients(cfg)
            )
        except Exception as exc:  # noqa: BLE001 — never let setup kill critic
            log.warning(
                "critic_agent_backend: failed to construct web tool clients (%s); web tools disabled",
                exc,
            )
            return

        schemas = _build_schemas(cfg)
        available_names: set[str] = set()
        if clients.search is not None:
            available_names.add("web_search")
        if clients.fetch is not None:
            available_names.add("web_fetch")
        schemas = [s for s in schemas if s.get("function", {}).get("name") in available_names]
        if not schemas or (clients.search is None and clients.fetch is None):
            log.info(
                "critic_agent_backend: web tools enabled by config but no "
                "usable client/schema; falling back to no-tool reasoning",
            )
            return

        self._web_tool_clients = clients
        self._web_tool_schemas = schemas
        self._web_tool_max_turns = cfg.critic_web_max_tool_turns
        log.info(
            "critic_agent_backend: web tools enabled tools=%s max_turns=%d",
            [s["function"]["name"] for s in schemas],
            self._web_tool_max_turns,
        )

    # Public API — Backend.run
    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """One Critic turn — run the prepare → reason → commit pipeline.

        Writes the ``coordinator_inbox`` request, runs ``prepare-review`` to get
        a judge bundle, enriches its ``review_constraints`` (action policy and
        cross-domain rules), drives Codex reasoning when proposals exist, runs
        ``commit-review`` to produce the intent envelope, validates it, and
        records per-turn telemetry.

        Args:
            prompt (str): The Coordinator-rendered inbox prompt for this turn.
            system_prompt (str | None): Optional system prompt forwarded into
                the Codex reasoning call.
            tools (list[str] | None): Unused; the Critic exposes no tool palette
                to the Coordinator.
            max_turns (int): Unused; the Critic is single-turn.

        Returns:
            BackendTurnResult: The validated review intents plus model, KB, and
            session metadata.

        Raises:
            BackendError: If the judge bundle or emit file cannot be read, or
                ``emit.json`` is missing a dict ``intent_envelope``.
            NoIntentEmitted: If the committed envelope fails intent validation.
        """
        del tools, max_turns  # Critic is single-turn / no tool palette.

        turn_idx = self._turn_idx
        self._turn_idx += 1

        workdir = allocate_turn_workdir(
            self.session_dir, "critic-workdir", turn_idx, keep=CRITIC_AGENT_WORKDIR_KEEP_COUNT
        )
        request_path = workdir / "request.json"
        judge_path = workdir / "judge_bundle.json"
        review_path = workdir / "review.json"
        emit_path = workdir / "emit.json"

        session_id = self.session_dir.name
        request: dict[str, Any] = {
            "kind": "coordinator_inbox",
            "session_id": session_id,
            "raw_prompt": prompt,
            "context": dict(self._static_context),
        }
        if self.known_actions:
            request["options"] = {
                "known_actions": list(self.known_actions),
            }
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        env = self._build_runtime_env()

        # Stage 1 — prepare-review (off-thread because subprocess.run blocks).
        await asyncio.to_thread(
            self._runtime_caller,
            RuntimeCall(
                phase="prepare-review",
                request_path=request_path,
                review_path=None,
                out_path=judge_path,
                cwd=self.critic_agent_root,
                env=env,
            ),
        )
        try:
            judge_bundle = json.loads(judge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"CriticAgentBackend: failed to read judge_bundle from {judge_path}: {exc}") from exc

        # Snapshot the reviewed proposal msg_ids for trace attribution: the
        # reasoning loop writes its token row mid-``run()`` (before the verdict
        # intents exist), so we stash them here for ``_trace_critic_llm_call``.
        self._trace_reviewed_msg_ids = _reviewed_msg_ids_from_bundle(judge_bundle)

        # Layer per-action verdict policy onto review_constraints.
        if self.action_verdict_policy:
            rc = judge_bundle.setdefault("review_constraints", {})
            if not isinstance(rc, dict):
                rc = {}
                judge_bundle["review_constraints"] = rc
            rc["action_verdict_policy"] = dict(self.action_verdict_policy)

        _maybe_inject_cross_domain_constraints(judge_bundle)

        # Stage 2 — Codex reasoning; short-circuit when there are no proposals.
        proposals = judge_bundle.get("proposals") or []
        if not proposals:
            review = {"review_verdicts": []}
            llm_text = "(skipped — no proposals)"
            llm_finish = "skipped"
        else:
            review, llm_text, llm_finish = await self._reason(
                judge_bundle=judge_bundle,
                system_prompt=system_prompt,
            )

        review_path.write_text(
            json.dumps(review, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Stage 3 — commit-review.
        await asyncio.to_thread(
            self._runtime_caller,
            RuntimeCall(
                phase="commit-review",
                request_path=request_path,
                review_path=review_path,
                out_path=emit_path,
                cwd=self.critic_agent_root,
                env=env,
            ),
        )
        try:
            emit = json.loads(emit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"CriticAgentBackend: failed to read emit.json from {emit_path}: {exc}") from exc

        envelope = emit.get("intent_envelope")
        if not isinstance(envelope, dict):
            raise BackendError(
                f"CriticAgentBackend: emit.json missing intent_envelope (got keys={sorted(emit.keys())!r})"
            )
        try:
            intents = validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(f"critic_agent_envelope_invalid: {exc}") from exc

        kb_skipped = judge_bundle.get("kb_read_skipped_reason")
        required_context = list(judge_bundle.get("required_context") or [])
        verdicts_summary = [
            (i.payload.get("verdict"), i.payload.get("source")) for i in intents if i.type.value == "review_verdict"
        ]
        assess_injected = self._kb_assess_inject_enabled()
        kb_assess_trace = self._build_kb_assess_trace(
            judge_bundle,
            review,
            injected=assess_injected,
        )
        kb_priors_trace = self._build_kb_priors_trace(judge_bundle, review)
        log.info(
            "critic_agent_backend turn=%d session=%s proposals=%d "
            "verdicts=%s kb_skipped=%s required_context=%s finish=%s "
            "kb_assess_mode=%s kb_assess_verdicts=%d kb_priors=%d",
            turn_idx,
            session_id,
            len(proposals),
            verdicts_summary,
            kb_skipped,
            required_context,
            llm_finish,
            kb_assess_trace.get("mode") or "n/a",
            kb_assess_trace.get("verdict_count") or 0,
            kb_priors_trace.get("prior_count") or 0,
        )
        self.calls.append(
            {
                "turn_idx": turn_idx,
                "proposals": len(proposals),
                "verdicts": verdicts_summary,
                "kb_skipped": kb_skipped,
                "required_context": required_context,
                "finish_reason": llm_finish,
                "workdir": str(workdir),
                "kb_assess_mode": kb_assess_trace.get("mode"),
                "kb_assess_verdicts": kb_assess_trace.get("verdict_count") or 0,
                "kb_priors_count": kb_priors_trace.get("prior_count") or 0,
            }
        )

        # Author-time breakdown capture: record this critic iteration before the
        # workdir can be pruned (composed into critic_robustness at assembly).
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_critic_iteration(
                self.session_dir,
                iter_n=turn_idx,
                review=review,
                emit=emit,
                workdir=workdir,
                kb_assess=kb_assess_trace,
                kb_priors=kb_priors_trace,
            )
        except Exception:  # noqa: BLE001
            pass

        # Second sink: mirror the KB integration trace into Langfuse (opt-in,
        # best-effort) so it lands on the same trace as the critic generations.
        self._mirror_kb_trace_to_langfuse(
            turn_idx=turn_idx,
            kb_assess=kb_assess_trace,
            kb_priors=kb_priors_trace,
        )

        return BackendTurnResult(
            intents=intents,
            raw_text=llm_text,
            metadata={
                "model": self.codex_model,
                "finish_reason": llm_finish,
                "judge_bundle_path": str(judge_path),
                "kb_read_skipped_reason": kb_skipped,
                "required_context": required_context,
                "kb_writes": [w.get("result", {}).get("status") for w in (emit.get("kb_writes") or [])],
                "session_id": session_id,
                "turn_idx": turn_idx,
            },
        )

    # Helpers

    def _load_static_context_from_manifest(self) -> dict[str, Any]:
        """Derive per-session context for ``request.context`` from
        manifest.json (model / framework / gpu_type / model_path / tp /
        workload / precision); empty values dropped. Any read error logs a
        WARNING and returns ``{}``.

        Returns:
            A context dict built from the manifest's non-empty fields, or an
            empty dict when the manifest is missing or unreadable.
        """
        path = manifest_path(self.session_dir)
        try:
            raw = path.read_text(encoding="utf-8")
            manifest = json.loads(raw)
        except FileNotFoundError:
            log.warning(
                "critic_agent_backend: manifest.json not found at %s — "
                "request.context will be empty; critic-agent runtime will "
                "report missing_critical_context for every verdict",
                path,
            )
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "critic_agent_backend: failed to load manifest.json at %s (%s: %s); request.context will be empty",
                path,
                type(exc).__name__,
                exc,
            )
            return {}

        ctx: dict[str, Any] = {}
        # CRITICAL keys — runtime hard-fails the verdict if either is missing.
        if manifest.get("model_name"):
            ctx["model"] = manifest["model_name"]
        if manifest.get("framework"):
            ctx["framework"] = manifest["framework"]
        if manifest.get("gpu_type"):
            ctx["gpu_type"] = manifest["gpu_type"]
        if manifest.get("model_path"):
            ctx["model_path"] = manifest["model_path"]
        tp = manifest.get("tp")
        if isinstance(tp, int) and tp > 0:
            ctx["tp"] = tp
        workload = manifest.get("workload")
        if isinstance(workload, dict):
            cleaned = {k: v for k, v in workload.items() if v not in (None, "")}
            if cleaned:
                ctx["workload"] = cleaned
                if cleaned.get("precision"):
                    ctx["precision"] = cleaned["precision"]
        return ctx

    def _build_runtime_env(self) -> dict[str, str]:
        """Build the subprocess environment for ``runtime.cli`` invocations.

        Co-locates session memory and the KB dead-letter dir under the session,
        sets the KB client mode and the robustness session-dir hint, and in
        ``live`` KB mode merges ``kb_env`` and requires ``KB_BASE_URL``.

        Returns:
            dict[str, str]: A copy of the current environment with the
            critic-agent runtime variables applied.

        Raises:
            BackendError: If ``kb_mode == "live"`` but ``KB_BASE_URL`` is unset.
        """
        env = dict(os.environ)
        # Co-locate session memory inside the Coordinator session.
        memory_dir = self.session_dir / "critic-session-memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("CRITIC_SESSION_MEMORY_DIR", str(memory_dir))
        env["CRITIC_KB_CLIENT_MODE"] = self.kb_mode

        # Point the runtime at the sibling robustness findings JSONL;
        # set here because the robustness CLI's env never reaches us.
        env.setdefault(
            "ROBUSTNESS_AGENT_SESSION_DIR",
            str(self.session_dir),
        )

        # Dead-letter dir under the session so cron replays don't cross sessions.
        dlq_dir = self.session_dir / "critic-kb-dead-letter"
        env.setdefault("KB_DEAD_LETTER_DIR", str(dlq_dir))

        # Propagate the --cortex-kb-url flag into the subprocess so the critic
        # runtime's optional /v2/reasoning/assess enrichment can reach the same
        # cortex KB recipe-snapshot uses. An explicit env var still wins.
        cortex_kb_url = (self.cortex_kb_url or "").strip()
        if cortex_kb_url and not env.get("CORTEX_KB_URL"):
            env["CORTEX_KB_URL"] = cortex_kb_url

        if self.kb_mode == "live":
            for k, v in (self.kb_env or {}).items():
                env[k] = v
            if not env.get("KB_BASE_URL"):
                raise BackendError(
                    "CriticAgentBackend: kb_mode=live but KB_BASE_URL is not set (export it or pass via kb_env=...)"
                )
        return env

    def _mirror_kb_trace_to_langfuse(
        self,
        *,
        turn_idx: int,
        kb_assess: dict[str, Any],
        kb_priors: dict[str, Any],
    ) -> None:
        """Mirror the per-iteration KB trace into Langfuse (best-effort).

        Emits one span per non-empty trace under the ``critic`` agent so the
        "was KB used / request / response / influenced decision" evidence is
        visible alongside the critic generations. No-op when Langfuse is
        disabled; never raises into the review path.

        Args:
            turn_idx (int): The critic iteration index.
            kb_assess (dict[str, Any]): The assess trace (may be empty).
            kb_priors (dict[str, Any]): The priors trace (may be empty).
        """
        try:
            from ..trace.langfuse_emitter import get_emitter

            emitter = get_emitter(self.session_dir)
            if not emitter.enabled:
                return
            if kb_assess:
                emitter.record_kb_span(
                    name=f"kb_assess:iter_{turn_idx}",
                    agent="critic",
                    output=kb_assess,
                    metadata={
                        "kind": "kb_assess",
                        "iter": turn_idx,
                        "mode": kb_assess.get("mode"),
                        "verdict_count": kb_assess.get("verdict_count") or 0,
                        "referenced_in_verdict": bool(kb_assess.get("referenced_in_verdict")),
                    },
                )
            if kb_priors:
                emitter.record_kb_span(
                    name=f"kb_priors:iter_{turn_idx}",
                    agent="critic",
                    output=kb_priors,
                    metadata={
                        "kind": "kb_priors",
                        "iter": turn_idx,
                        "prior_count": kb_priors.get("prior_count") or 0,
                        "referenced_in_verdict": bool(kb_priors.get("referenced_in_verdict")),
                    },
                )
        except Exception:  # noqa: BLE001 — trace must never break the review
            log.debug("critic_agent: langfuse kb mirror failed", exc_info=True)

    def _kb_assess_inject_enabled(self) -> bool:
        """Whether substrate assess verdicts are fed to the LLM this turn.

        ``kb_assess_inject`` (when not ``None``) forces the mode; otherwise the
        ``CORTEX_KB_ASSESS_INJECT`` env decides, defaulting to OFF (dry-run).

        Returns:
            bool: ``True`` to inject the verdicts into the prompt.
        """
        if self.kb_assess_inject is not None:
            return bool(self.kb_assess_inject)
        return os.environ.get(
            "CORTEX_KB_ASSESS_INJECT",
            "",
        ).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _build_kb_assess_trace(
        judge_bundle: dict[str, Any],
        review: dict[str, Any] | None,
        *,
        injected: bool,
    ) -> dict[str, Any]:
        """Assemble the substrate-assess KB trace for one critic iteration.

        Folds the runtime-captured ``kb_assess_trace`` (configured? skip reason?
        focus? per-proposal requests) with the resolved verdict count, whether
        the verdicts were injected into the prompt this turn (vs dry-run), and
        whether the final verdict referenced KB evidence.

        Args:
            judge_bundle (dict[str, Any]): The prepared judge bundle.
            review (dict[str, Any] | None): The parsed review object.
            injected (bool): Whether the verdicts were fed to the LLM.

        Returns:
            dict[str, Any]: The assess trace (empty when nothing was captured).
        """
        trace = dict(judge_bundle.get("kb_assess_trace") or {})
        verdicts = judge_bundle.get("kb_assess_by_proposal") or {}
        if not trace and not verdicts:
            return {}
        return {
            "configured": bool(trace.get("configured")),
            "skipped_reason": trace.get("skipped_reason"),
            "focus": trace.get("focus") or {},
            "requests": trace.get("requests") or [],
            "verdict_count": len(verdicts),
            "injected": bool(injected),
            "mode": "injected" if injected else "dry_run",
            "referenced_in_verdict": (_verdict_references_kb(review) if injected else False),
        }

    @staticmethod
    def _build_kb_priors_trace(
        judge_bundle: dict[str, Any],
        review: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Assemble the historical-priors KB trace for one critic iteration.

        Folds the runtime-captured ``kb_priors_trace`` (scope/limit/per-request
        cache+count) with the total prior count, the skip reason (if any), and
        whether the final verdict referenced KB evidence. Priors are always
        injected, so ``referenced_in_verdict`` is unconditional here.

        Args:
            judge_bundle (dict[str, Any]): The prepared judge bundle.
            review (dict[str, Any] | None): The parsed review object.

        Returns:
            dict[str, Any]: The priors trace (empty when nothing was captured).
        """
        trace = dict(judge_bundle.get("kb_priors_trace") or {})
        by_proposal = judge_bundle.get("kb_priors_by_proposal") or {}
        for_decision = judge_bundle.get("kb_priors_for_decision") or []
        skipped = judge_bundle.get("kb_read_skipped_reason")
        if not trace and not by_proposal and not for_decision and not skipped:
            return {}
        total = sum(len(v) for v in by_proposal.values() if isinstance(v, list)) + len(for_decision)
        return {
            "configured": bool(trace.get("configured")),
            "mode": trace.get("mode"),
            "client_mode": trace.get("client_mode"),
            "scope_filter": trace.get("scope_filter") or {},
            "limit": trace.get("limit"),
            "requests": trace.get("requests") or [],
            "skipped_reason": skipped,
            "prior_count": total,
            "referenced_in_verdict": _verdict_references_kb(review),
        }

    async def _reason(
        self,
        *,
        judge_bundle: dict[str, Any],
        system_prompt: str | None,
    ) -> tuple[dict[str, Any], str, str | None]:
        """Drive Codex with the judge bundle and parse a review object.

        Builds the skill-preamble + judge-bundle + output-format user prompt,
        runs the (optionally tool-using) reasoning loop, and extracts the review
        JSON, falling back to an empty verdict list when nothing parses.

        Args:
            judge_bundle (dict[str, Any]): The prepared judge bundle to reason
                over.
            system_prompt (str | None): Optional system prompt sent as the
                leading system message.

        Returns:
            tuple[dict[str, Any], str, str | None]: The parsed review dict, the
            raw model text, and the final finish reason.
        """
        preamble = self._load_skill_preamble()
        bundle_view: dict[str, Any] = {
            "kind": judge_bundle.get("kind"),
            "session_id": judge_bundle.get("session_id"),
            "decision_id": judge_bundle.get("decision_id"),
            "merged_context": judge_bundle.get("merged_context"),
            "missing_context": judge_bundle.get("missing_context"),
            "required_context": judge_bundle.get("required_context"),
            "proposals": judge_bundle.get("proposals"),
            "kb_priors_by_proposal": judge_bundle.get("kb_priors_by_proposal"),
            "kb_priors_for_decision": judge_bundle.get("kb_priors_for_decision"),
            "kb_read_skipped_reason": judge_bundle.get("kb_read_skipped_reason"),
            "review_constraints": judge_bundle.get("review_constraints"),
            "notes": judge_bundle.get("notes"),
        }
        # Dry-run gate: only feed the substrate assess verdicts to the LLM when
        # injection is explicitly enabled. When OFF the verdicts are still
        # fetched + traced (see _build_kb_assess_trace) but kept out of the
        # prompt so they cannot steer the decision.
        if self._kb_assess_inject_enabled():
            bundle_view["kb_assess_by_proposal"] = judge_bundle.get("kb_assess_by_proposal")
        bundle_text = json.dumps(bundle_view, ensure_ascii=False, indent=2)
        user_prompt = (
            f"{preamble}\n\n"
            f"==== JUDGE BUNDLE ====\n{bundle_text}\n==== END JUDGE BUNDLE ====\n\n"
            f"{_REVIEW_OUTPUT_INSTRUCTIONS}"
        )
        messages = build_chat_messages(system_prompt, user_prompt)

        text, finish = await self._run_reasoning_loop(messages)

        # Full-trace conversation: the reasoning loop already folded token
        # spend onto llm_calls.jsonl; mirror the full prompt + reply onto
        # conversations.jsonl so the critic turn is replayable alongside the
        # orchestration / specialist turns. The system + user prompt is the
        # full request the critic saw; ``text`` is its externally-visible
        # reply. Best-effort and never raised into the review path.
        self._record_critic_conversation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=text,
        )

        review = _extract_review_json(text)
        if review is None:
            # Empty list → commit-review emits a heartbeat; don't raise.
            review = {"review_verdicts": []}
            log.warning(
                "critic_agent_backend: model reply contained no parseable review_verdicts JSON (chars=%d, finish=%s)",
                len(text),
                finish,
            )
        return review, text, finish

    async def _run_reasoning_loop(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str, str | None]:
        """Run the Codex chat-completions loop, optionally interleaving
        web_search / web_fetch up to ``self._web_tool_max_turns`` before a
        final text-only reply. Returns ``(text, finish_reason)``; tool-exec
        failures are reported back to the model, never raised.

        Args:
            messages: The running chat-completions message list; tool-use turns
                are appended in place.

        Returns:
            A tuple of the final reply text and the final finish reason.

        Raises:
            BackendError: If any Codex chat-completions API call fails.
        """
        tools = self._web_tool_schemas
        max_turns = self._web_tool_max_turns if tools else 0
        # Accumulate token usage across every Codex call in
        # this reasoning loop (initial + tool-use rounds + forced final).
        # OpenAI has no prompt-cache split, so only in/out counters move.
        usage_acc = {"input_tokens": 0, "output_tokens": 0}
        # Sum the wall-clock of every Codex call in this reasoning loop so the
        # trace reports the critic's real end-to-end latency (tool turns incl.).
        latency_ms_acc = 0

        for turn in range(max_turns + 1):
            kwargs: dict[str, Any] = {
                "model": self.codex_model,
                "messages": messages,
                "max_completion_tokens": CRITIC_AGENT_MAX_COMPLETION_TOKENS,
            }
            if tools and turn < max_turns:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            _t0 = time.perf_counter()
            try:
                resp = await self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise BackendError(
                    f"Codex API call failed (critic-agent reasoning): {exc!r}",
                ) from exc
            latency_ms_acc += int((time.perf_counter() - _t0) * 1000)

            self._accumulate_usage(usage_acc, getattr(resp, "usage", None))
            choice = resp.choices[0]
            msg = choice.message
            finish = getattr(choice, "finish_reason", None)
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                self._trace_critic_llm_call(usage_acc, latency_ms=latency_ms_acc)
                return msg.content or "", finish

            log.info(
                "critic_agent_backend tool-call turn=%d count=%d tools=%s",
                turn,
                len(tool_calls),
                [tc.function.name for tc in tool_calls if tc.function],
            )
            messages.append(_assistant_message_with_tool_calls(msg))
            for tc in tool_calls:
                tool_result = await self._execute_tool_call(tc)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )

        # Exhausted max_turns mid-tool-use: force a final no-tool reply.
        _t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self.codex_model,
                messages=messages,
                max_completion_tokens=CRITIC_AGENT_MAX_COMPLETION_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001
            raise BackendError(
                f"Codex API call failed (critic-agent reasoning final turn): {exc!r}",
            ) from exc
        latency_ms_acc += int((time.perf_counter() - _t0) * 1000)
        self._accumulate_usage(usage_acc, getattr(resp, "usage", None))
        self._trace_critic_llm_call(usage_acc, latency_ms=latency_ms_acc)
        final = resp.choices[0]
        return final.message.content or "", getattr(final, "finish_reason", None)

    @staticmethod
    def _accumulate_usage(
        acc: dict[str, int],
        usage: Any,
    ) -> None:
        """Fold one OpenAI ``resp.usage`` into the running token accumulator.

        OpenAI reports ``prompt_tokens`` / ``completion_tokens``; map them
        onto the canonical in/out counters. Missing / bad values contribute
        0 so a single malformed response never corrupts the running sum.

        Args:
            acc: The running accumulator with ``input_tokens`` /
                ``output_tokens`` keys, updated in place.
            usage: An OpenAI usage object (or ``None``) to fold into ``acc``.
        """
        if usage is None:
            return
        try:
            acc["input_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        except (TypeError, ValueError):
            pass
        try:
            acc["output_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        except (TypeError, ValueError):
            pass

    def set_trace_context(
        self, *, tick: int | None = None, phase: str | None = None,
    ) -> None:
        """Stamp the timeline keys for the next reactor turn's trace row.

        The Coordinator calls this before ``run()`` (it owns ``shared_state``)
        so the critic's self-written ``llm_calls`` row carries the same
        tick/phase the in-process reactor trace would have. Best-effort: a
        bad value degrades to ``None`` rather than raising.
        """
        try:
            self._trace_tick = int(tick) if tick is not None else None
        except (TypeError, ValueError):
            self._trace_tick = None
        self._trace_phase = (str(phase) or None) if phase else None

    def _trace_critic_llm_call(
        self, usage_acc: dict[str, int], *, latency_ms: int | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a critic reasoning loop.

        Records the accumulated Codex token spend (and summed wall-clock
        ``latency_ms``) under ``component=critic``. tick/phase come from the
        trace context the Coordinator stamped via :meth:`set_trace_context`
        before the reactor ``run()`` (``None`` when unset → collector ts-window
        backfill). Best-effort: never raises into the review path.

        Args:
            usage_acc: Accumulated token counts with ``input_tokens`` /
                ``output_tokens`` keys for this reasoning loop.
            latency_ms: Summed wall-clock latency of the reasoning loop, when
                measured.
        """
        try:
            record = LLMCallRecord(
                session_id=self.session_dir.name,
                component="critic",
                role="critic",
                model=self.codex_model,
                tick=self._trace_tick,
                phase=self._trace_phase,
                input_tokens=usage_acc.get("input_tokens"),
                output_tokens=usage_acc.get("output_tokens"),
                latency_ms=latency_ms,
                reviewed_msg_ids=self._trace_reviewed_msg_ids,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break review
            log.debug(
                "full-trace: critic llm_call append failed",
                exc_info=True,
            )

    def _record_critic_conversation(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        response: str,
    ) -> None:
        """Append one ``conversations.jsonl`` row for a critic reasoning loop.

        Persists the full (redacted) prompt + reply under
        ``component=critic``. The prompt is the system message (when present)
        joined to the judge-bundle user message — the complete request the
        critic reasoned over. tick/phase are unknown to the critic backend
        (it runs as its own reactor); the collector backfills from the ts
        window. Best-effort: never raises into the review path. No-op when
        both prompt and reply are empty.

        Args:
            system_prompt: Optional system prompt prepended to the recorded
                prompt.
            user_prompt: The judge-bundle user prompt the critic reasoned over.
            response: The model's externally-visible reply text.
        """
        try:
            prompt = f"{system_prompt}\n---\n{user_prompt}" if system_prompt else user_prompt
            if not prompt and not response:
                return
            record = ConversationRecord(
                session_id=self.session_dir.name,
                component="critic",
                role="critic",
                model=self.codex_model,
                prompt=prompt or "",
                response=response or "",
            )
            append_conversation(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break review
            log.debug(
                "full-trace: critic conversation append failed",
                exc_info=True,
            )

    async def _execute_tool_call(self, tool_call: Any) -> str:
        """Dispatch one OpenAI tool_call to the configured web client.

        Args:
            tool_call (Any): The OpenAI tool-call object carrying the function
                name and JSON-encoded arguments.

        Returns:
            str: The tool's result text, or an ``Error: ...`` string when the
            arguments are malformed or the tool is unknown/disabled.
        """
        fn = getattr(tool_call, "function", None)
        name = getattr(fn, "name", "") if fn else ""
        raw_args = getattr(fn, "arguments", "") if fn else ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            return f"Error: tool arguments are not valid JSON: {exc}"
        if not isinstance(args, dict):
            return "Error: tool arguments must be a JSON object"

        clients = self._web_tool_clients
        if clients is None:
            return f"Error: tool {name!r} is not available (web tools disabled)"

        if name == "web_search" and clients.search is not None:
            return await asyncio.to_thread(clients.search.execute, args)
        if name == "web_fetch" and clients.fetch is not None:
            return await asyncio.to_thread(clients.fetch.execute, args)
        return f"Error: unknown or disabled tool {name!r}"

    def _load_skill_preamble(self) -> str:
        """Load and cache the critic-agent skill/action markdown preamble.

        Reads ``SKILL.md`` and ``actions/review_coordinator_inbox.md`` from the
        critic-agent root, concatenating whatever is readable. Missing files are
        skipped (the prompt just gets thinner). The result is memoised.

        Returns:
            str: The combined preamble text, or an empty string when no source
            files could be read.
        """
        if self._skill_preamble is not None:
            return self._skill_preamble
        parts: list[str] = []
        for rel in ("SKILL.md", "actions/review_coordinator_inbox.md"):
            path = self.critic_agent_root / rel
            try:
                parts.append(f"==== {rel} ====\n{path.read_text(encoding='utf-8').strip()}")
            except OSError:
                continue
        self._skill_preamble = "\n\n".join(parts) if parts else ""
        return self._skill_preamble


__all__ = [
    "CRITIC_AGENT_MAX_COMPLETION_TOKENS",
    "CRITIC_AGENT_RUNTIME_TIMEOUT_SEC",
    "CRITIC_AGENT_WORKDIR_KEEP_COUNT",
    "CriticAgentBackend",
    "RuntimeCall",
    "RuntimeCaller",
    "_assistant_message_with_tool_calls",
    "_default_runtime_caller",
    "_extract_review_json",
]
