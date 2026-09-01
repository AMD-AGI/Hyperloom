# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CriticAgentBackend — bridges the ``hyperloom.agents.critic`` runtime into
the Coordinator as a real Critic Backend.

Runs the two-phase loop from ``src/hyperloom/agents/critic/README.md``
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
from typing import Any, Callable, Literal

from hyperloom.common.llm_config import (
    LLMConfigError,
    aanthropic_completion,
    achat_completion,
    anthropic_transport_ready,
    apply_reasoning_effort,
    build_http_timeout,
    get_async_openai_client,
)
from hyperloom.inference_optimizer.breakdown.agent_ownership import (
    LEVER_CONFIG,
    LEVER_ENABLEMENT,
    LEVER_SOURCE_PATCH,
    LEVER_UPSTREAM_PR,
    patch_lever_kind,
)
from hyperloom.common.jsonio import extract_first_json_with_key
from hyperloom.inference_optimizer.protocol.intent import (
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from hyperloom.inference_optimizer.session.session_paths import allocate_turn_workdir, manifest_path
from ..trace.conversation_trace import ConversationRecord, append_conversation
from ..trace.llm_trace import LLMCallRecord, append_llm_call, new_call_id
from ..trace.parse_usage import reasoning_output_tokens
from .base import BackendError, BackendTurnResult, LLMCallFailed, build_chat_messages, parse_call_timeout_env
from ._runtime_bridge import RuntimeCall, RuntimeCaller, invoke_runtime_cli


log = logging.getLogger(__name__)


CRITIC_AGENT_RUNTIME_TIMEOUT_SEC = 30  # prepare-review / commit-review wall cap
# Output-token cap for both review paths. The Anthropic side spends it as a
# request field or through the CLI environment, depending on the transport.
#
# The cap is a ceiling, not a budget: headroom left unused is never billed,
# while a reply cut off at the cap bills the whole call and yields nothing. It
# is therefore sized for the largest review a batch could ever need rather than
# the typical one. The number is measured, not guessed: the three-proposal
# batch that deadlocked session 100162 needed roughly 2.4k output tokens to
# write out in full, so this leaves about 13x that — a batch would have to grow
# by an order of magnitude before the cap is what binds. Use the env var of the
# same name to lower it for a model whose own output limit is smaller, or to
# raise it further.
CRITIC_AGENT_MAX_COMPLETION_TOKENS = 32000
# One retry at this multiple of the cap when a reply stops at the limit.
CRITIC_AGENT_TRUNCATION_RETRY_FACTOR = 2
# Finish/stop reasons that mean "cut off at the output cap": OpenAI reports
# ``length``, the Anthropic Messages API reports ``max_tokens``.
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})
# Anthropic usage counters carried through to the trace row, each in its own
# column so critic rows stay comparable with the orchestration ones.
_ANTHROPIC_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _accumulate_reasoning_tokens(acc: dict[str, int], usage: Any) -> None:
    """Fold a reply's reasoning-output tokens into the accumulator, when reported.

    The key is only created when the provider reported a count, so a model
    without a reasoning split still writes ``None`` (not ``0``) to the ledger —
    the documented difference between "no reasoning concept" and "no reasoning
    tokens spent".

    Args:
        acc: The running accumulator, updated in place.
        usage: A provider usage payload (mapping or SDK object).
    """
    count = reasoning_output_tokens(usage)
    if count is None:
        return
    acc["reasoning_output_tokens"] = acc.get("reasoning_output_tokens", 0) + count


# HTTP client timeout defaults for critic review calls. A completion is not
# streamed, so the read half is what bounds generation: the server holds the
# connection open until the whole reply exists.
#
# It is therefore coupled to the output cap above and has to move with it. The
# 120s this used to sit at is the budget a completion is elsewhere given, and it
# was the right one while the cap was 2000 tokens: a reply that short always
# came back inside it, so the cap doubled as a latency bound. Raising the
# ceiling removed that guarantee, and a batch large enough to need the new room
# would now hit the timeout instead of finishing — a timeout that then repeats
# on identical input, which is the failure shape this whole fix exists to end.
# 300s is what the orchestration paths floor a multi-turn budget at, and it
# covers any review that could plausibly be worth waiting for.
CRITIC_AGENT_LLM_CONNECT_TIMEOUT_SEC = 10.0
CRITIC_AGENT_LLM_RW_TIMEOUT_SEC = 300.0

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
      "failure_reason_code": "<failure_reason_code of the review_constraints rule this verdict rests on, else \"\">",
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
- Keep `reasoning`/`notes` to new, decision-relevant points; do not restate the
  proposal or context already in the judge_bundle.
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
- When a verdict rests on a rule from `review_constraints`, copy that
  rule's `failure_reason_code` verbatim into the verdict's own
  `failure_reason_code`; leave it `""` when the verdict rests on your
  own judgement. Some of those rules declare `advise` as their
  `failure_verdict`, and naming the rule is how the Coordinator tells
  a verdict resting on one apart from a substantive refusal.
==== END OUTPUT FORMAT ====
""".strip()


# Bare {...} fallback carrying "review_verdicts" (fenced case handled by helper).
_BARE_JSON_RE = re.compile(r"(\{[^{}]*\"review_verdicts\"[\s\S]*\})", re.DOTALL)


def _extract_review_json(text: str) -> dict[str, Any] | None:
    """Pull the Critic's own ``{"review_verdicts": ...}`` object out of a reply.

    Uses ``last=True`` so the model's final answer wins over any earlier
    fenced block echoed from the (attacker-influenceable) proposal payload:
    the genuine verdict is the last block the Critic emits, an echoed block
    can only appear before it.
    """
    return extract_first_json_with_key(text, "review_verdicts", _BARE_JSON_RE, last=True)


def _is_truncated_finish(finish: str | None) -> bool:
    """Report whether a finish/stop reason means the reply hit the output cap.

    Args:
        finish: The finish/stop reason a transport reported, or ``None`` when
            it supplied none.

    Returns:
        ``True`` when the reason names the output cap, ``False`` otherwise.
    """
    return isinstance(finish, str) and finish.strip().lower() in _TRUNCATED_FINISH_REASONS


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

    invoke_runtime_cli(
        call,
        module="hyperloom.agents.critic.runtime.cli",
        agent_label="critic-agent",
        timeout_sec=CRITIC_AGENT_RUNTIME_TIMEOUT_SEC,
        extra_args=extra_args,
    )


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


# Per-phase review orientation; only the live phase's entry is injected.
_PHASE_ORIENTATION: dict[str, str] = {
    "PRELUDE": (
        "Typical proposals are `target_analysis` and `baseline`. If something "
        "else slips through (PolicyGate R1 should already have blocked it), "
        "`advise` with a phase hint rather than reject."
    ),
    "FRAMEWORK_AGENT": (
        "Typical proposals are `explore`, `specialist` and `integrate_patch`. "
        "Specialist-style proposal_set packets arrive as "
        "`propose_action='explore'` with a `variants` array — return one "
        "verdict dict per variant msg_id; missing entries are treated as "
        "`needs_review`. What a KEEP has to clear depends on the lever the "
        "proposal moves, not on the phase — see `review_constraints."
        "lever_orientation` when it is present."
    ),
    "KERNEL_AGENT": (
        "Typical proposals are the KERNEL_AGENT_OWNED_ACTIONS (proxied via "
        "REQUEST) plus auto-managed `profile` / `roofline`. Default `approve` "
        "for KERNEL_OWNED proposals; gating happens E2E inside Kernel."
    ),
    "SWEEP": ("Typical proposal is `sweep`. Mismatches → `advise` with the phase hint."),
    "CLOSE": (
        "Typical proposals are `report` and `session_breakdown`. Both are "
        "archival: they transcribe existing state and introduce no new "
        "measurement, so the before/after gate does not apply."
    ),
}


#: Orientation by the lever a proposal moves. The phase used to carry this,
#: which worked only while each phase held one lever: the FRAMEWORK entry told
#: the Critic that flat gain was a legitimate KEEP, and merging the phases would
#: have silently extended that to configuration search. The deterministic layer
#: already routes on payload markers rather than phase; this matches it.
_LEVER_ORIENTATION: dict[str, str] = {
    LEVER_UPSTREAM_PR: (
        "This lands an upstream diff nobody here wrote. Judge whether it is "
        "worth measuring and whether it can be rolled back — the measurement "
        "itself is the executor's gate, not yours."
    ),
    LEVER_ENABLEMENT: (
        "The gate here is runnability plus the accuracy floor, not throughput: "
        "a candidate that boots and holds accuracy is a legitimate KEEP even at "
        "flat gain. Pre-boot, the production evidence cannot exist yet."
    ),
    LEVER_SOURCE_PATCH: (
        "A patch written for this session. It changes the source tree, so "
        "rollback and blast radius carry the weight; throughput is measured "
        "afterwards and is not yours to predict."
    ),
    LEVER_CONFIG: (
        "Server arguments and environment only — nothing on disk changes and a "
        "revert is a non-composition. Judge the reasoning and the accuracy "
        "risk; the cost of being wrong is one bench."
    ),
}


def _inject_lever_orientation(judge_bundle: dict[str, Any], payload: dict[str, Any] | None) -> None:
    """Stamp the orientation for the lever this proposal moves, when known.

    Args:
        judge_bundle: The judge bundle to enrich in place.
        payload: The proposal payload, read for a lever stamp or its markers.
    """
    lever = patch_lever_kind(payload if isinstance(payload, dict) else None)
    if not lever:
        params = (payload or {}).get("params") if isinstance(payload, dict) else None
        lever = patch_lever_kind(params if isinstance(params, dict) else None)
    orientation = _LEVER_ORIENTATION.get(lever)
    if not orientation:
        return
    rc = judge_bundle.setdefault("review_constraints", {})
    rc["lever_kind"] = lever
    rc["lever_orientation"] = orientation


def _inject_phase_constraints(judge_bundle: dict[str, Any], phase: str) -> None:
    """Stamp the live phase and its review orientation onto the judge bundle.

    Args:
        judge_bundle: The judge bundle to enrich in place.
        phase: Coordinator pipeline phase; an unrecognised one leaves the bundle
            untouched so nothing asserts a phase that was never delivered.
    """
    normalized = (phase or "").strip().upper()
    if normalized not in _PHASE_ORIENTATION:
        return
    judge_bundle["phase"] = normalized
    rc = judge_bundle.setdefault("review_constraints", {})
    rc["phase"] = normalized
    rc["phase_orientation"] = _PHASE_ORIENTATION[normalized]


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


def _maybe_inject_quantitative_claim_constraint(judge_bundle: dict[str, Any]) -> None:
    """Set ``review_constraints.quantitative_claim_rule`` from the enforced list.

    Delivering the rule as data keeps the Critic's field list identical to the
    one the runner strips, instead of a hand-copied prose list that drifts. It
    is sent only when the bundle holds a proposal the rule is about, on the same
    principle as the cross-domain rules above: a Critic handed a rule that
    cannot apply to anything under review can still cite it, and a citation is
    what the verdict path reads.

    Args:
        judge_bundle: The judge bundle to enrich in place; unchanged when no
            proposal is one of the kinds the rule governs.
    """
    from ..specialists.patch_safety import (
        advisory_rules_govern,
        quantitative_claim_rule_descriptor,
    )

    proposals = judge_bundle.get("proposals") or []
    if not isinstance(proposals, list):
        return
    governed = any(advisory_rules_govern(str(p.get("action_name") or "")) for p in proposals if isinstance(p, dict))
    if not governed:
        return
    rc = judge_bundle.setdefault("review_constraints", {})
    if not isinstance(rc, dict):
        rc = {}
        judge_bundle["review_constraints"] = rc
    rc["quantitative_claim_rule"] = quantitative_claim_rule_descriptor()


@dataclass
class CriticAgentBackend:
    """Real Critic backend that drives the critic-agent runtime.

    Parameters
    ----------
    critic_agent_root:
        Directory containing ``runtime/cli.py``. The CLI is invoked as
        ``python -m hyperloom.agents.critic.runtime.cli`` with
        ``cwd=critic_agent_root`` so relative asset reads resolve.
    session_dir:
        Coordinator session directory. Scopes per-turn workdirs and the
        per-session critic memory store.
    codex_model:
        OpenAI / Codex chat-completion model id (e.g. ``gpt-5.6-sol``).
    codex_client_factory:
        Optional callable returning an ``AsyncOpenAI``-compatible client
        (test seam).
    kb_mode:
        ``inmemory`` (default) keeps KB writes / reads off the wire.
        ``live`` requires ``kb_env`` (or process env) to provide ``KB_BASE_URL``.
    kb_env:
        Extra env vars merged into the runtime.cli subprocess env when
        ``kb_mode == "live"``.
    runtime_caller_factory:
        Test seam returning a :data:`RuntimeCaller`.
    static_context:
        Optional explicit per-session context injected as ``request.context``.
        When ``None``, derived from ``manifest.json``; ``{}`` is a valid
        "no context" override.
    name:
        Backend instance name surfaced in the Coordinator startup banner.
    """

    critic_agent_root: Path
    session_dir: Path
    codex_model: str = "gpt-5.6-sol"
    codex_client_factory: Callable[[], Any] | None = None
    kb_mode: Literal["inmemory", "live"] = "inmemory"
    kb_env: dict[str, str] | None = None
    runtime_caller_factory: Callable[[], RuntimeCaller] | None = None
    static_context: dict[str, Any] | None = None
    known_actions: tuple[str, ...] = ()
    # Per-action verdict policy enriched onto
    # ``review_constraints.action_verdict_policy`` post prepare-review.
    action_verdict_policy: dict[str, str] = field(default_factory=dict)
    name: str = "critic-agent"
    # Review inference protocol. ``openai`` drives Codex chat.completions;
    # ``anthropic`` drives llm_config's single-shot Anthropic entry point.
    protocol: Literal["openai", "anthropic"] = "openai"
    # Claude model id used when ``protocol == "anthropic"`` (falls back to
    # ``codex_model`` when unset).
    claude_model: str | None = None

    # ``_runtime_caller`` is assigned on the instance in __post_init__ (not as a
    # dataclass field) to avoid descriptor binding as a method.
    _client: Any = field(default=None, init=False, repr=False)
    _turn_idx: int = field(default=0, init=False, repr=False)
    # Trace context the Coordinator stamps before each reactor
    # ``run()`` so the critic's self-written llm_calls row carries the timeline
    # keys.
    _trace_tick: int | None = field(default=None, init=False, repr=False)
    _trace_phase: str | None = field(default=None, init=False, repr=False)
    _trace_macro_cycle: int | None = field(default=None, init=False, repr=False)
    # Proposal msg_ids reviewed by the current turn, snapshotted for llm_calls
    # attribution.
    _trace_reviewed_msg_ids: list[str] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _skill_preamble: str | None = field(default=None, init=False, repr=False)
    _static_context: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    # Resolved review model id (protocol-aware).
    _review_model: str = field(default="", init=False, repr=False)
    calls: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate config, wire transports, and resolve static context.

        Normalises paths, verifies ``runtime/cli.py`` exists under
        ``critic_agent_root``, validates ``kb_mode``, selects the real or test
        runtime caller, constructs the Codex/OpenAI review client (or checks the
        Anthropic credential when ``protocol == "anthropic"``, where llm_config
        owns the transport), and resolves the per-session static context
        (explicit or from ``manifest.json``).

        Raises:
            BackendError: If ``runtime/cli.py`` is missing, ``kb_mode`` is
                invalid, the review SDK/transport is unavailable, or no API key
                is set.
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

        self._review_model = (
            (self.claude_model or self.codex_model) if self.protocol == "anthropic" else self.codex_model
        )
        if self.protocol == "anthropic":
            self._require_anthropic_transport()
        elif self.codex_client_factory is not None:
            self._client = self.codex_client_factory()
        else:
            connect_timeout_s, rw_timeout_s = self._resolve_llm_timeouts()
            try:
                self._client = get_async_openai_client(
                    timeout=build_http_timeout(connect=connect_timeout_s, read=rw_timeout_s),
                )
            except LLMConfigError as exc:
                raise BackendError(
                    str(exc).replace(
                        "OpenAI-compatible client",
                        "CriticAgentBackend cannot reach Codex for review reasoning",
                    )
                ) from exc

        # Resolve static per-session context once.
        if self.static_context is not None:
            self._static_context = dict(self.static_context)
        else:
            self._static_context = self._load_static_context_from_manifest()
        log.info(
            "critic_agent_backend static_context source=%s keys=%s",
            "explicit" if self.static_context is not None else "manifest",
            sorted(self._static_context.keys()),
        )

    @staticmethod
    def _resolve_llm_timeouts() -> tuple[float, float]:
        """Return the ``(connect, read/write/pool)`` review-call timeouts in seconds.

        The OpenAI-compatible transport spends both halves as HTTP timeouts, as
        does the Anthropic Messages API; the Claude CLI transport reuses only
        the read/write value as its wall-clock call budget.
        """
        return (
            parse_call_timeout_env(
                "CRITIC_AGENT_LLM_CONNECT_TIMEOUT_S",
                default=CRITIC_AGENT_LLM_CONNECT_TIMEOUT_SEC,
            ),
            parse_call_timeout_env(
                "CRITIC_AGENT_LLM_RW_TIMEOUT_S",
                default=CRITIC_AGENT_LLM_RW_TIMEOUT_SEC,
            ),
        )

    @staticmethod
    def _resolve_max_completion_tokens() -> int:
        """Return the output-token cap one review call may spend.

        The env override exists so a deployment that hits the cap can raise it
        without a code change; a malformed value falls back to the default
        rather than failing the turn.

        Returns:
            The positive cap in tokens.
        """
        raw = os.environ.get("CRITIC_AGENT_MAX_COMPLETION_TOKENS")
        if raw is None or not raw.strip():
            return CRITIC_AGENT_MAX_COMPLETION_TOKENS
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value <= 0:
            log.warning(
                "CRITIC_AGENT_MAX_COMPLETION_TOKENS=%r is not a positive integer; using default %d",
                raw,
                CRITIC_AGENT_MAX_COMPLETION_TOKENS,
            )
            return CRITIC_AGENT_MAX_COMPLETION_TOKENS
        return value

    def _require_anthropic_transport(self) -> None:
        """Fail fast when the Anthropic side cannot serve a review call.

        No client is built here: :func:`aanthropic_completion` owns transport
        selection. The check covers the transport and not just the credential,
        because a subscription token resolves to the Claude CLI — a host with
        the token but without the CLI would pass a credential-only check and
        then fail at the first review, which is what this backend promises not
        to do.

        Raises:
            BackendError: If no Anthropic-side credential is configured, or the
                transport it implies is unavailable.
        """
        if anthropic_transport_ready():
            return
        raise BackendError(
            "CriticAgentBackend(protocol=anthropic) review reasoning requires a usable "
            "Anthropic transport: an Anthropic-side credential (ANTHROPIC_API_KEY / "
            "ANTHROPIC_AUTH_TOKEN / CLAUDE_CODE_OAUTH_TOKEN), plus the claude CLI when "
            "that credential is a subscription token"
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
        cross-domain rules), drives the review inference call when proposals
        exist (Codex chat-completions, or llm_config's single-shot Anthropic
        entry point when ``protocol == 'anthropic'``), runs
        ``commit-review`` to produce the intent envelope, validates it, and
        records per-turn telemetry.

        Args:
            prompt (str): The Coordinator-rendered inbox prompt for this turn.
            system_prompt (str | None): Optional system prompt forwarded into
                the review reasoning call.
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
        # prepare-review runs as a subprocess; the phase reaches it via context.
        context = dict(self._static_context)
        if self._trace_phase:
            context["phase"] = str(self._trace_phase).strip().upper()
        if self._trace_macro_cycle is not None:
            context["macro_cycle"] = self._trace_macro_cycle
        request: dict[str, Any] = {
            "kind": "coordinator_inbox",
            "session_id": session_id,
            "raw_prompt": prompt,
            "context": context,
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

        # prepare-review runs off-thread because subprocess.run blocks.
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

        # Snapshot the reviewed proposal msg_ids for trace attribution.
        self._trace_reviewed_msg_ids = _reviewed_msg_ids_from_bundle(judge_bundle)

        # Layer per-action verdict policy onto review_constraints.
        if self.action_verdict_policy:
            rc = judge_bundle.setdefault("review_constraints", {})
            if not isinstance(rc, dict):
                rc = {}
                judge_bundle["review_constraints"] = rc
            rc["action_verdict_policy"] = dict(self.action_verdict_policy)

        _inject_phase_constraints(judge_bundle, self._trace_phase or "")
        # The lever says what a KEEP has to clear; the phase no longer can,
        # now that one phase carries every lever.
        _proposals = judge_bundle.get("proposals") or []
        _first = _proposals[0] if isinstance(_proposals, list) and _proposals else None
        _inject_lever_orientation(judge_bundle, _first if isinstance(_first, dict) else None)
        _maybe_inject_cross_domain_constraints(judge_bundle)
        _maybe_inject_quantitative_claim_constraint(judge_bundle)

        # Codex reasoning; short-circuit when there are no proposals.
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

        # commit-review.
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
        kb_priors_trace = self._build_kb_priors_trace(judge_bundle, review)
        log.info(
            "critic_agent_backend turn=%d session=%s proposals=%d "
            "verdicts=%s kb_skipped=%s required_context=%s finish=%s kb_priors=%d",
            turn_idx,
            session_id,
            len(proposals),
            verdicts_summary,
            kb_skipped,
            required_context,
            llm_finish,
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
                "kb_priors_count": kb_priors_trace.get("prior_count") or 0,
            }
        )

        # Record this critic iteration before the workdir can be pruned.
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_critic_iteration(
                self.session_dir,
                iter_n=turn_idx,
                request=request,
                judge_bundle=judge_bundle,
                review=review,
                emit=emit,
                workdir=workdir,
                kb_priors=kb_priors_trace,
            )
        except Exception:  # noqa: BLE001
            pass

        # Mirror the KB integration trace into Langfuse (opt-in, best-effort).
        self._mirror_kb_trace_to_langfuse(
            turn_idx=turn_idx,
            kb_priors=kb_priors_trace,
        )

        return BackendTurnResult(
            intents=intents,
            raw_text=llm_text,
            metadata={
                "model": self._review_model,
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

        # Point the runtime at the sibling robustness findings JSONL.
        env.setdefault(
            "ROBUSTNESS_AGENT_SESSION_DIR",
            str(self.session_dir),
        )

        # Dead-letter dir under the session so cron replays don't cross sessions.
        dlq_dir = self.session_dir / "critic-kb-dead-letter"
        env.setdefault("KB_DEAD_LETTER_DIR", str(dlq_dir))

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
        kb_priors: dict[str, Any],
    ) -> None:
        """Mirror the per-iteration KB trace into Langfuse (best-effort).

        Emits one span per non-empty trace under the ``critic`` agent so the
        "was KB used / request / response / influenced decision" evidence is
        visible alongside the critic generations. No-op when Langfuse is
        disabled; never raises into the review path.

        Args:
            turn_idx (int): The critic iteration index.
            kb_priors (dict[str, Any]): The priors trace (may be empty).
        """
        try:
            from ..trace.langfuse_emitter import get_emitter

            emitter = get_emitter(self.session_dir)
            if not emitter.enabled:
                return
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
        runs the single-shot reasoning call, and extracts the review JSON. A
        reply cut off at the output cap is retried once with more room; a reply
        that still carries no review JSON fails the turn.

        Args:
            judge_bundle (dict[str, Any]): The prepared judge bundle to reason
                over.
            system_prompt (str | None): Optional system prompt sent as the
                leading system message.

        Returns:
            tuple[dict[str, Any], str, str | None]: The parsed review dict, the
            raw model text, and the final finish reason.

        Raises:
            BackendError: If no reply yields parseable ``review_verdicts``
                JSON. The turn ends here, so ``commit-review``, the breakdown
                record and the Langfuse mirror are all skipped — there is no
                half-built envelope to hand on. The prompt and the unusable
                reply are already on ``conversations.jsonl``, which is where a
                post-mortem finds them.
        """
        preamble = self._load_skill_preamble()
        bundle_view: dict[str, Any] = {
            "kind": judge_bundle.get("kind"),
            "session_id": judge_bundle.get("session_id"),
            "decision_id": judge_bundle.get("decision_id"),
            "phase": judge_bundle.get("phase"),
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
        bundle_text = json.dumps(bundle_view, ensure_ascii=False, separators=(",", ":"))
        user_prompt = (
            f"{preamble}\n\n"
            f"==== JUDGE BUNDLE ====\n{bundle_text}\n==== END JUDGE BUNDLE ====\n\n"
            f"{_REVIEW_OUTPUT_INSTRUCTIONS}"
        )
        max_tokens = self._resolve_max_completion_tokens()
        # One id per review call, shared by its token row and its conversation
        # row so the two halves pair on the call rather than on a ts second.
        call_id = new_call_id()
        text, finish = await self._run_reasoning_loop(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            call_id=call_id,
        )

        # Mirror the full prompt + reply onto conversations.jsonl so the critic
        # turn is replayable. Best-effort; never raised into the review path.
        self._record_critic_conversation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=text,
            call_id=call_id,
        )
        review = _extract_review_json(text)

        if review is None and _is_truncated_finish(finish):
            # Retrying a cap-truncated reply under the same cap would truncate
            # again at the same byte, so the retry only makes sense with more
            # room. One is enough: a bundle that overflows twice this much is a
            # sizing problem, not a flaky call.
            #
            # Only the HTTP transports always name the reason. The Claude CLI
            # reports a stop reason only when the model supplies one, so a
            # truncated reply can arrive as finish=None and fall through to the
            # raise below. That path loses the retry but still fails loudly,
            # which is the half of this fix that matters.
            retry_tokens = max_tokens * CRITIC_AGENT_TRUNCATION_RETRY_FACTOR
            log.warning(
                "critic_agent_backend: review reply stopped at the %d-token cap "
                "(chars=%d, finish=%s); retrying once with %d",
                max_tokens,
                len(text),
                finish,
                retry_tokens,
            )
            retry_call_id = new_call_id()
            try:
                text, finish = await self._run_reasoning_loop(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=retry_tokens,
                    call_id=retry_call_id,
                )
            except BackendError as exc:
                # A provider whose own output limit sits below the doubled cap
                # rejects the retry outright. Letting that transport error
                # surface on its own would name the retry as the problem and
                # bury the truncation that caused it, pointing the reader at
                # the wrong thing.
                raise BackendError(
                    f"CriticAgentBackend: review reply was truncated at {max_tokens} tokens "
                    f"and the retry at {retry_tokens} was rejected: {exc}"
                ) from exc
            self._record_critic_conversation(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=text,
                call_id=retry_call_id,
            )
            review = _extract_review_json(text)
            max_tokens = retry_tokens

        if review is None:
            # A reply carrying no verdicts is a review that failed to arrive,
            # not a review that found nothing to say. Reporting it as an empty
            # verdict list makes the two indistinguishable downstream: the turn
            # looks successful, the proposals it was asked about stay pending,
            # and the loop re-asks the same question forever. Raising instead
            # puts a backend_error on the record and lets the streak guard trip.
            raise BackendError(
                "CriticAgentBackend: review reply carried no parseable review_verdicts JSON "
                f"(chars={len(text)}, finish={finish!r}, max_tokens={max_tokens})"
            )
        return review, text, finish

    async def _run_reasoning_loop(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        max_tokens: int,
        call_id: str | None = None,
    ) -> tuple[str, str | None]:
        """Issue one review inference call and return ``(text, finish_reason)``.

        The critic reasons single-shot over the judge bundle (no tool use).
        Both prompt segments are passed through unmerged so each transport can
        map them onto its own request shape.

        Args:
            system_prompt: The system instruction, or ``None``.
            user_prompt: The judge bundle plus output instructions.
            max_tokens: Output-token cap for this call.
            call_id: Per-call id stamped on the token row this call writes.

        Returns:
            A tuple of the reply text and the finish/stop reason.

        Raises:
            BackendError: If the review API call fails.
        """
        if self.protocol == "anthropic":
            return await self._run_anthropic_reasoning(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                call_id=call_id,
            )
        return await self._run_openai_reasoning(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            call_id=call_id,
        )

    async def _run_openai_reasoning(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        max_tokens: int,
        call_id: str | None = None,
    ) -> tuple[str, str | None]:
        """Issue one Codex chat-completions call and return ``(text, finish_reason)``.

        Args:
            system_prompt: The system instruction, or ``None``.
            user_prompt: The judge bundle plus output instructions.
            max_tokens: Output-token cap for this call.

        Returns:
            A tuple of the reply text and the finish reason.

        Raises:
            BackendError: If the Codex chat-completions API call fails.
        """
        kwargs: dict[str, Any] = {
            "model": self._review_model,
            "messages": build_chat_messages(system_prompt, user_prompt),
            "max_completion_tokens": max_tokens,
        }
        apply_reasoning_effort(kwargs)
        usage_acc = {"input_tokens": 0, "output_tokens": 0}
        _t0 = time.perf_counter()
        try:
            result = await achat_completion(
                self._client,
                component="critic",
                operation="review",
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._llm_call_failed(
                f"Codex API call failed (critic-agent reasoning): {exc!r}",
                latency_ms=int((time.perf_counter() - _t0) * 1000),
            ) from exc
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        self._accumulate_usage(usage_acc, result.usage)
        self._trace_critic_llm_call(usage_acc, latency_ms=latency_ms, call_id=call_id)
        return result.text, result.finish_reason

    async def _run_anthropic_reasoning(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        max_tokens: int,
        call_id: str | None = None,
    ) -> tuple[str, str | None]:
        """Issue one single-shot Anthropic completion for the review.

        ``llm_config`` picks the transport from the configured credential, so
        this path accepts an API key, a gateway bearer token, or a Max/Pro
        subscription token alike. Token counts fold into the same trace row as
        the OpenAI path, and both paths carry the same output-token cap.

        Args:
            system_prompt: The system instruction, or ``None``.
            user_prompt: The judge bundle plus output instructions.
            max_tokens: Output-token cap for this call.

        Returns:
            A tuple of the reply text and the stop reason. The CLI transport
            reports one only when the model supplies it, so unlike the OpenAI
            path it is not guaranteed.

        Raises:
            LLMCallFailed: If the completion fails.
        """
        connect_timeout_s, rw_timeout_s = self._resolve_llm_timeouts()
        _t0 = time.perf_counter()
        try:
            result = await aanthropic_completion(
                component="critic",
                operation="review",
                model=self._review_model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                timeout=build_http_timeout(connect=connect_timeout_s, read=rw_timeout_s),
                timeout_s=rw_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._llm_call_failed(
                f"Anthropic completion failed (critic-agent reasoning): {exc!r}",
                latency_ms=int((time.perf_counter() - _t0) * 1000),
            ) from exc
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        usage_acc = {"input_tokens": 0, "output_tokens": 0}
        self._accumulate_anthropic_usage(usage_acc, result.usage)
        self._trace_critic_llm_call(usage_acc, latency_ms=latency_ms, call_id=call_id)
        stop_reason = result.stop_reason
        return (result.text or "", stop_reason if isinstance(stop_reason, str) and stop_reason else None)

    @staticmethod
    def _accumulate_anthropic_usage(
        acc: dict[str, int],
        usage: Any,
    ) -> None:
        """Fold one Anthropic ``usage`` block into the running accumulator.

        Cache counters keep their own keys instead of being folded into
        ``input_tokens``. That matches the rows ``ClaudeBackend`` writes for
        orchestration, so a reader can compare or sum the two components
        without knowing which one produced a row. The judge bundle repeats
        across turns and reliably hits the prompt cache, so the split is most
        of the input side, not a rounding detail.

        Missing / bad values contribute 0 so a malformed reply never corrupts
        the token sum.

        Args:
            acc: The running accumulator, updated in place.
            usage: An Anthropic usage dict (or ``None``) to fold into ``acc``.
        """
        if not isinstance(usage, dict):
            return
        for key in _ANTHROPIC_USAGE_KEYS:
            try:
                acc[key] = acc.get(key, 0) + int(usage.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue
        _accumulate_reasoning_tokens(acc, usage)

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
        _accumulate_reasoning_tokens(acc, usage)

    def set_trace_context(
        self,
        *,
        tick: int | None = None,
        phase: str | None = None,
        macro_cycle: int | None = None,
    ) -> None:
        """Stamp the timeline keys for the next reactor turn and request.

        The Coordinator calls this before ``run()`` (it owns ``shared_state``)
        so the critic request carries the current phase/macro-cycle and its
        self-written ``llm_calls`` row carries the same tick/phase as the
        in-process reactor trace. Best-effort: a
        bad value degrades to ``None`` rather than raising.
        """
        try:
            self._trace_tick = int(tick) if tick is not None else None
        except (TypeError, ValueError):
            self._trace_tick = None
        self._trace_phase = (str(phase) or None) if phase else None
        try:
            self._trace_macro_cycle = int(macro_cycle) if macro_cycle is not None else None
        except (TypeError, ValueError):
            self._trace_macro_cycle = None

    def _trace_critic_llm_call(
        self,
        usage_acc: dict[str, int],
        *,
        latency_ms: int | None = None,
        call_id: str | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a critic reasoning loop.

        Records the accumulated review-model token spend (and summed wall-clock
        ``latency_ms``) under ``component=critic`` for whichever transport
        :attr:`protocol` selected, using the tick/phase from
        :meth:`set_trace_context`. The stamped ``model`` is
        :attr:`_review_model`. Best-effort: never raises into the review path.

        Cache counters are absent on the OpenAI path, which has no prompt-cache
        split, so they stay ``None`` there — the documented meaning of the
        column — rather than being reported as zero.

        Args:
            usage_acc: Accumulated token counts for this reasoning loop.
            latency_ms: Summed wall-clock latency of the reasoning loop, when
                measured.
            call_id: Per-call id shared with this call's conversation row.
        """
        try:
            record = LLMCallRecord(
                session_id=self.session_dir.name,
                component="critic",
                role="critic",
                call_id=call_id,
                model=self._review_model,
                tick=self._trace_tick,
                phase=self._trace_phase,
                input_tokens=usage_acc.get("input_tokens"),
                output_tokens=usage_acc.get("output_tokens"),
                cache_read_input_tokens=usage_acc.get("cache_read_input_tokens"),
                cache_creation_input_tokens=usage_acc.get("cache_creation_input_tokens"),
                reasoning_output_tokens=usage_acc.get("reasoning_output_tokens"),
                latency_ms=latency_ms,
                reviewed_msg_ids=self._trace_reviewed_msg_ids,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break review
            log.debug(
                "full-trace: critic llm_call append failed",
                exc_info=True,
            )

    def _llm_call_failed(
        self,
        message: str,
        *,
        latency_ms: int | None = None,
    ) -> LLMCallFailed:
        """Record a failed review-model call and return the error to raise.

        The critic self-writes its trace rows, so a review call that never
        returned has to be recorded here — the Coordinator only sees the raised
        error, and by then the token accounting the success path relies on does
        not exist. Returning the exception (rather than raising) keeps each call
        site a single ``raise ... from exc``.

        Args:
            message: The failure description carried by the raised error.
            latency_ms: Time spent before failing, when measured.

        Returns:
            The :class:`LLMCallFailed` for the caller to raise.
        """
        error = LLMCallFailed(message)
        self._trace_llm_failure(error, latency_ms=latency_ms)
        return error

    def _trace_llm_failure(
        self,
        error: BaseException,
        *,
        latency_ms: int | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a call that never returned.

        Args:
            error: The exception describing the failure.
            latency_ms: Time spent before failing, when measured.
        """
        try:
            record = LLMCallRecord.for_failure(
                session_id=self.session_dir.name,
                component="critic",
                role="critic",
                error=error,
                model=self._review_model,
                tick=self._trace_tick,
                phase=self._trace_phase,
                latency_ms=latency_ms,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break review
            log.debug(
                "full-trace: critic llm_call failure append failed",
                exc_info=True,
            )

    def _record_critic_conversation(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        response: str,
        call_id: str | None = None,
    ) -> None:
        """Append one ``conversations.jsonl`` row for a critic reasoning loop.

        Persists the full (redacted) prompt + reply under ``component=critic``.
        Best-effort: never raises into the review path. No-op when both prompt
        and reply are empty.

        Args:
            system_prompt: Optional system prompt prepended to the recorded
                prompt.
            user_prompt: The judge-bundle user prompt the critic reasoned over.
            response: The model's externally-visible reply text.
            call_id: Per-call id shared with this call's token row.
        """
        try:
            prompt = f"{system_prompt}\n---\n{user_prompt}" if system_prompt else user_prompt
            if not prompt and not response:
                return
            record = ConversationRecord(
                session_id=self.session_dir.name,
                component="critic",
                role="critic",
                call_id=call_id,
                model=self._review_model,
                prompt=prompt or "",
                response=response or "",
            )
            append_conversation(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break review
            log.debug(
                "full-trace: critic conversation append failed",
                exc_info=True,
            )

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
    "CRITIC_AGENT_TRUNCATION_RETRY_FACTOR",
    "CRITIC_AGENT_WORKDIR_KEEP_COUNT",
    "CriticAgentBackend",
    "RuntimeCall",
    "RuntimeCaller",
    "_default_runtime_caller",
    "_extract_review_json",
    "_is_truncated_finish",
]
