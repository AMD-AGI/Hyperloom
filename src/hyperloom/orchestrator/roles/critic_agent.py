# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CriticAgentBackend — bridges the ``hyperloom.agents.critic`` runtime into the Coordinator as a real Critic Backend."""

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
# Output-token cap for both review paths.
CRITIC_AGENT_MAX_COMPLETION_TOKENS = 32000
# One retry at this multiple of the cap when a reply stops at the limit.
CRITIC_AGENT_TRUNCATION_RETRY_FACTOR = 2
# Finish/stop reasons that mean "cut off at the output cap": OpenAI reports ``length``, the Anthropic Messages API
# reports ``max_tokens``.
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})
# Anthropic usage counters carried through to the trace row, each in its own column so critic rows stay comparable
# with the orchestration ones.
_ANTHROPIC_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _accumulate_reasoning_tokens(acc: dict[str, int], usage: Any) -> None:
    """Fold a reply's reasoning-output tokens into the accumulator, when reported."""
    count = reasoning_output_tokens(usage)
    if count is None:
        return
    acc["reasoning_output_tokens"] = acc.get("reasoning_output_tokens", 0) + count


# HTTP client timeout defaults for critic review calls.
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
    """Pull the Critic's own ``{\"review_verdicts\": ...}`` object out of a reply."""
    return extract_first_json_with_key(text, "review_verdicts", _BARE_JSON_RE, last=True)


def _is_truncated_finish(finish: str | None) -> bool:
    """Report whether a finish/stop reason means the reply hit the output cap."""
    return isinstance(finish, str) and finish.strip().lower() in _TRUNCATED_FINISH_REASONS


def _default_runtime_caller(call: RuntimeCall) -> None:
    """Real implementation — runs ``python -m hyperloom.agents.critic.runtime.cli <phase> ...``."""
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
    """Pull the proposal ``msg_id``s out of a judge bundle, or ``None``."""
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
    """Read the ``scope`` dial off a proposal (top-level or nested ``params``)."""
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


def _review_subjects(judge_bundle: dict[str, Any]) -> dict[str, str]:
    """Map each reviewed proposal's message id to the row it is recorded under.

    The two arms identify a proposal differently -- a configuration grid by the
    bus message that raised it, an upstream candidate by its candidate id --
    and evidence filed under the wrong one opens a second, near-empty row
    beside the proposal it was about.

    Args:
        judge_bundle (dict[str, Any]): The bundle of proposals reviewed.

    Returns:
        dict[str, str]: ``{msg_id: row_id}``, holding only the proposals whose
            row id is not their message id.
    """
    out: dict[str, str] = {}
    for proposal in judge_bundle.get("proposals") or []:
        if not isinstance(proposal, dict):
            continue
        msg_id = str(proposal.get("msg_id") or "")
        payload = proposal.get("payload") if isinstance(proposal.get("payload"), dict) else {}
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        candidate = str(
            payload.get("framework_agent_candidate_id") or params.get("framework_agent_candidate_id") or ""
        ).strip()
        if msg_id and candidate:
            out[msg_id] = candidate
    return out


def _verdict_references_kb(review: dict[str, Any] | None) -> bool:
    """Whether any final review verdict cites KB evidence."""
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
    """Stamp the orientation for the lever this proposal moves, when known."""
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
    """Stamp the live phase and its review orientation onto the judge bundle."""
    normalized = (phase or "").strip().upper()
    if normalized not in _PHASE_ORIENTATION:
        return
    judge_bundle["phase"] = normalized
    rc = judge_bundle.setdefault("review_constraints", {})
    rc["phase"] = normalized
    rc["phase_orientation"] = _PHASE_ORIENTATION[normalized]


def _maybe_inject_cross_domain_constraints(judge_bundle: dict[str, Any]) -> None:
    """Set ``review_constraints.cross_domain`` + rule descriptors when any proposal is cross-domain (unified ``scope == 'domains'`` dial)."""
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
    """Set ``review_constraints.quantitative_claim_rule`` from the enforced list."""
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
    """Real Critic backend that drives the critic-agent runtime."""

    critic_agent_root: Path
    session_dir: Path
    codex_model: str = "gpt-5.6-sol"
    codex_client_factory: Callable[[], Any] | None = None
    kb_mode: Literal["inmemory", "live"] = "inmemory"
    kb_env: dict[str, str] | None = None
    runtime_caller_factory: Callable[[], RuntimeCaller] | None = None
    static_context: dict[str, Any] | None = None
    known_actions: tuple[str, ...] = ()
    # Per-action verdict policy enriched onto ``review_constraints.action_verdict_policy`` post prepare-review.
    action_verdict_policy: dict[str, str] = field(default_factory=dict)
    name: str = "critic-agent"
    # Review inference protocol.
    protocol: Literal["openai", "anthropic"] = "openai"
    # Claude model id used when ``protocol == "anthropic"`` (falls back to ``codex_model`` when unset).
    claude_model: str | None = None

    # ``_runtime_caller`` is assigned on the instance in __post_init__ (not as a dataclass field) to avoid descriptor
    # binding as a method.
    _client: Any = field(default=None, init=False, repr=False)
    _turn_idx: int = field(default=0, init=False, repr=False)
    # Trace context the Coordinator stamps before each reactor ``run()`` so the critic's self-written llm_calls row
    # carries the timeline keys.
    _trace_tick: int | None = field(default=None, init=False, repr=False)
    _trace_phase: str | None = field(default=None, init=False, repr=False)
    _trace_macro_cycle: int | None = field(default=None, init=False, repr=False)
    # Proposal msg_ids reviewed by the current turn, snapshotted for llm_calls attribution.
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
        """Validate config, wire transports, and resolve static context."""
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
        """Return the ``(connect, read/write/pool)`` review-call timeouts in seconds."""
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
        """Return the output-token cap one review call may spend."""
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
        """Fail fast when the Anthropic side cannot serve a review call."""
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
        """One Critic turn — run the prepare → reason → commit pipeline."""
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
        # The lever says what a KEEP has to clear; the phase no longer can, now that one phase carries every lever.
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
            from hyperloom.inference_optimizer.breakdown.recorder import critic_out

            critic_out.record_critic_iteration(
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

        # Attach what each ruling was grounded in to the ruling itself, on the
        # proposal it judged. Done here because these are the turn's own facts:
        # the artifacts are this runtime's files, and a KB write's result only
        # comes back on the emit.
        self._record_review_evidence(
            request=request,
            judge_bundle=judge_bundle,
            review=review,
            emit=emit,
            workdir=workdir,
            kb_priors=kb_priors_trace,
        )

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
        """Derive per-session context for ``request.context`` from manifest.json (model / framework / gpu_type / model_path / tp / workload / precision); empty values dropped."""
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
        """Build the subprocess environment for ``runtime.cli`` invocations."""
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
        """Mirror the per-iteration KB trace into Langfuse (best-effort)."""
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

    def _record_review_evidence(
        self,
        *,
        request: dict[str, Any],
        judge_bundle: dict[str, Any],
        review: dict[str, Any] | None,
        emit: dict[str, Any],
        workdir: Path,
        kb_priors: dict[str, Any],
    ) -> None:
        """Record what each of this turn's rulings was grounded in.

        Written onto the proposal each verdict targets, because a ruling and
        its grounds are one fact about one proposal: the alternative is a
        per-turn stream a reader has to join back to the proposals, keyed on a
        turn index that resume reuses.

        The KB write is matched to its verdict by target, not spread across
        them: the Critic asks for a lesson to be persisted per verdict, and a
        turn that reviewed six proposals and wrote one lesson would otherwise
        report the write six times.

        Args:
            request (dict[str, Any]): The review request, read for the cycle.
            judge_bundle (dict[str, Any]): The bundle reviewed, read for the
                row each verdict's target is recorded under.
            review (dict[str, Any] | None): The parsed review object.
            emit (dict[str, Any]): The commit emit, read for the KB writes.
            workdir (Path): This turn's workdir, holding the artifacts.
            kb_priors (dict[str, Any]): The priors trace for the turn.
        """
        try:
            from hyperloom.inference_optimizer.breakdown.recorder.framework_event import record_review_evidence

            context = request.get("context") if isinstance(request.get("context"), dict) else {}
            macro_cycle = context.get("macro_cycle")
            if macro_cycle is None:
                return
            subjects = _review_subjects(judge_bundle)
            writes: dict[str, dict[str, Any]] = {}
            for write in emit.get("kb_writes") or []:
                if not isinstance(write, dict):
                    continue
                target = str(write.get("target_proposal_msg_id") or "")
                result = write.get("result") if isinstance(write.get("result"), dict) else {}
                if target:
                    writes[target] = {
                        "trigger": str(write.get("trigger") or ""),
                        "status": str(result.get("status") or ""),
                        "detail": str(result.get("detail") or result.get("error") or ""),
                    }
            artifacts = {
                name: str(workdir / filename)
                for name, filename in (
                    ("request_path", "request.json"),
                    ("judge_bundle_path", "judge_bundle.json"),
                    ("review_path", "review.json"),
                    ("emit_path", "emit.json"),
                )
            }
            for verdict in (review or {}).get("review_verdicts") or []:
                if not isinstance(verdict, dict):
                    continue
                target = str(verdict.get("target_proposal_msg_id") or "")
                if not target:
                    continue
                kb: dict[str, Any] = {"persist_requested": bool(verdict.get("persist_to_kb"))}
                if kb_priors:
                    kb["priors"] = kb_priors
                if target in writes:
                    kb["write"] = writes[target]
                record_review_evidence(
                    macro_cycle=macro_cycle,
                    proposal_id=subjects.get(target) or target,
                    artifacts=artifacts,
                    kb=kb,
                )
        except Exception:  # noqa: BLE001 — observability cannot break the review
            log.debug("critic_agent: review evidence record failed", exc_info=True)

    @staticmethod
    def _build_kb_priors_trace(
        judge_bundle: dict[str, Any],
        review: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Assemble the historical-priors KB trace for one critic iteration."""
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
        """Drive Codex with the judge bundle and parse a review object."""
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
        # One id per review call, shared by its token row and its conversation row so the two halves pair on the call
        # rather than on a ts second.
        call_id = new_call_id()
        text, finish = await self._run_reasoning_loop(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            call_id=call_id,
        )

        # Mirror the full prompt + reply onto conversations.jsonl so the critic turn is replayable.
        self._record_critic_conversation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=text,
            call_id=call_id,
        )
        review = _extract_review_json(text)

        if review is None and _is_truncated_finish(finish):
            # Retrying a cap-truncated reply under the same cap would truncate again at the same byte, so the retry
            # only makes sense with more room.
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
                # A provider whose own output limit sits below the doubled cap rejects the retry outright.
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
            # A reply carrying no verdicts is a review that failed to arrive, not a review that found nothing to say.
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
        """Issue one review inference call and return ``(text, finish_reason)``."""
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
        """Issue one Codex chat-completions call and return ``(text, finish_reason)``."""
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
        """Issue one single-shot Anthropic completion for the review."""
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
        """Fold one Anthropic ``usage`` block into the running accumulator."""
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
        """Fold one OpenAI ``resp.usage`` into the running token accumulator."""
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
        """Stamp the timeline keys for the next reactor turn and request."""
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
        """Append one ``llm_calls.jsonl`` row for a critic reasoning loop."""
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
        """Record a failed review-model call and return the error to raise."""
        error = LLMCallFailed(message)
        self._trace_llm_failure(error, latency_ms=latency_ms)
        return error

    def _trace_llm_failure(
        self,
        error: BaseException,
        *,
        latency_ms: int | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a call that never returned."""
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
        """Append one ``conversations.jsonl`` row for a critic reasoning loop."""
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
        """Load and cache the critic-agent skill/action markdown preamble."""
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
