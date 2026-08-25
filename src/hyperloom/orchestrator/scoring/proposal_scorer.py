# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ProposalScorer — purely advisory multi-model scorer for specialist proposals.

Scores each variant with one or more LLM models (advisory only; never
sorts/ranks/auto-selects, never touches Critic/PolicyGate). Each model scored
independently via ``asyncio.gather``.

Output schema (``score`` return value)::

    {"scale": "0-10", "models": {"<slug>": {"<name>": {"score": <0-10>, "reason": "<str>"}}}, "errors": {...}}

Each proposal is scored under a stable ``proposal_<index>`` id in the model
prompt; results are keyed by the proposal's display ``name`` in the output.

Test seam: pass ``client_factory`` to bypass real client construction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hyperloom.common.llm_config import (
    LLMConfigError,
    apply_reasoning_effort,
    astream_chat_completion_text,
    get_async_openai_client,
)
from ..roles.base import parse_call_timeout_env
from ..loop.coordinator_helpers import format_exc_brief
from ..trace.conversation_trace import ConversationRecord, append_conversation
from ..trace.llm_trace import LLMCallRecord, append_llm_call, new_call_id
from ..trace.parse_usage import reasoning_output_tokens

log = logging.getLogger(__name__)


DEFAULT_SCORER_MODELS: tuple[str, ...] = (
    "claude-opus-5",
    "gpt-5.6-sol",
    # Gemini MUST carry the ``gemini/`` prefix (bare slug routes to a broken Vertex ADC path).
    "dvue-aoai-005-Kimi-K2.6",
    "gemini/gemini-3.1-pro-preview",
)

# Soft cap on proposals fed to each model.
_MAX_PROPOSALS_SCORED: int = 16
_MAX_FIELD_CHARS: int = 600


_SCORING_INSTRUCTIONS = """
You are scoring candidate serving-config variants proposed by a perf
specialist for an LLM-inference optimization run on AMD GPUs.

For EACH proposal below, output a single composite score from 0 to 10:
how likely / how much this variant improves serving THROUGHPUT for the
stated gap, grounded ONLY in the evidence shown. 10 = very likely a
large win; 0 = irrelevant or likely harmful. Be calibrated, not generous.

Output ONLY one compact JSON object, no prose, no markdown fence needed:

{"scores": {"<proposal_id>": {"score": <0-10 number>, "reason": "<= 15 words"}}}

Rules:
- One entry per proposal, keyed by its exact "id" (not the human-readable name).
- "reason" MUST be <= 15 words. Keep it terse to keep the reply short.
- Do not add keys other than "score" and "reason".
""".strip()


def _extract_scores_json(text: str) -> dict[str, Any] | None:
    """Pull the last valid ``{"scores": {...}}`` object out of a reply."""
    from hyperloom.common.jsonio import extract_last_json_with_key

    return extract_last_json_with_key(text, "scores")


@dataclass(frozen=True)
class _ScoringProposal:
    """One proposal prepared for multi-model scoring."""

    stable_id: str
    output_name: str
    label_name: str
    proposal: dict[str, Any]


def _prepare_scoring_proposals(proposals: list[dict[str, Any]]) -> list[_ScoringProposal]:
    """Assign stable ids and reject duplicate display names.

    Args:
        proposals: Candidate variants to score.

    Returns:
        Prepared scoring entries with stable ids and unique label names.

    Raises:
        ValueError: When two proposals share the same canonical display name.
    """
    seen_labels: set[str] = set()
    out: list[_ScoringProposal] = []
    for i, proposal in enumerate(proposals):
        raw_name = proposal.get("name")
        if raw_name is not None and str(raw_name).strip():
            output_name = str(raw_name)
            label_name = output_name.strip()
        else:
            output_name = f"proposal_{i}"
            label_name = output_name
        if label_name in seen_labels:
            raise ValueError(f"duplicate proposal name: {label_name!r}")
        seen_labels.add(label_name)
        out.append(
            _ScoringProposal(
                stable_id=f"proposal_{i}",
                output_name=output_name,
                label_name=label_name,
                proposal=proposal,
            )
        )
    return out


def _clip(value: Any, *, limit: int = _MAX_FIELD_CHARS) -> str:
    """Stringify a value and truncate it to a maximum length.

    Args:
        value: Value to stringify (``None`` becomes an empty string).
        limit: Maximum number of characters to keep.

    Returns:
        The string, suffixed with an ellipsis if it was truncated.
    """
    s = "" if value is None else str(value)
    return s if len(s) <= limit else (s[:limit] + "…")


def _coerce_score(raw: Any) -> float | None:
    """Coerce a model-emitted score into a clamped [0, 10] float.

    Args:
        raw: Raw score value emitted by a model.

    Returns:
        The score clamped to ``[0, 10]``, or ``None`` if ``raw`` is not numeric
        or is NaN. Infinities are clamped to the bounds. Boolean values are
        rejected (they would otherwise coerce via ``float(True) == 1.0``).
    """
    if isinstance(raw, bool):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    return max(0.0, min(10.0, val))


def _normalise_model_scores(
    parsed: dict[str, Any],
    *,
    scoring_entries: list[_ScoringProposal],
) -> dict[str, dict[str, Any]]:
    """Project parsed ``{"scores": {...}}`` onto known stable ids.

    Args:
        parsed: A parsed ``{"scores": {...}}`` dict from a model reply.
        scoring_entries: Prepared proposals keyed by stable id in the prompt.

    Returns:
        A mapping of proposal display name to its clamped score and truncated
        reason.
    """
    out: dict[str, dict[str, Any]] = {}
    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        return out
    id_to_name = {entry.stable_id: entry.output_name for entry in scoring_entries}
    known = set(id_to_name)
    for proposal_id, entry in scores.items():
        key = str(proposal_id)
        if key not in known or not isinstance(entry, dict):
            continue
        score = _coerce_score(entry.get("score"))
        if score is None:
            continue
        out[id_to_name[key]] = {
            "score": score,
            "reason": _clip(entry.get("reason"), limit=160),
        }
    return out


@dataclass
class ProposalScorer:
    """Advisory multi-model scorer (see module docstring)."""

    models: tuple[str, ...] = DEFAULT_SCORER_MODELS
    # Scorer talks the OpenAI protocol, so prefer the OpenAI-side key/URL.
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    # Large cap so reasoning raters can finish reasoning and still emit the scores JSON.
    max_completion_tokens: int = 4096
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_PROPOSAL_SCORER_TIMEOUT_SEC",
            default=120.0,
        )
    )

    # Test seam — set to bypass real OpenAI client construction.
    client_factory: Callable[[], Any] | None = None

    # When set, each call appends token usage to the trace; ``None`` disables trace writes.
    session_dir: Path | None = None

    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize the model list and optionally eager-build the client.

        Strips and de-blanks the configured model names. If a
        ``client_factory`` is provided the client is built immediately;
        otherwise it is constructed lazily on first use so an
        unconfigured environment degrades per-call rather than at boot.
        """
        self.models = tuple(m for m in (str(x).strip() for x in (self.models or ())) if m)
        if len(self.models) != len(set(self.models)):
            raise ValueError("duplicate scorer model slug(s) in models")
        if self.client_factory is not None:
            self._client = self.client_factory()
            return
        # Lazy: construct on first use so an unconfigured env degrades per-call.
        self._client = None

    def _ensure_client(self) -> Any:
        """Return the cached client, constructing one on first use.

        Returns:
            The OpenAI-compatible async client.

        Raises:
            RuntimeError: If the ``openai`` SDK is missing or no API key
                is configured in the environment.
        """
        if self._client is not None:
            return self._client
        try:
            self._client = get_async_openai_client(
                api_key_env=self.api_key_env,
                base_url_env=self.base_url_env,
            )
        except LLMConfigError as exc:
            raise RuntimeError(str(exc).replace("OpenAI-compatible client", "ProposalScorer")) from exc
        return self._client

    def _build_prompt(
        self,
        *,
        gap: dict[str, Any],
        scoring_entries: list[_ScoringProposal],
    ) -> str:
        """Build ONE group-scoring prompt covering every proposal.

        Args:
            gap: The gap being addressed (domain, symptom, evidence, etc.).
            scoring_entries: Prepared proposals with stable ids.

        Returns:
            The assembled prompt text describing the gap and proposals.
        """
        lines: list[str] = ["=== Gap ==="]
        lines.append(f"domain: {_clip(gap.get('domain'), limit=80)}")
        lines.append(f"gap_canonical_id: {_clip(gap.get('gap_canonical_id'), limit=160)}")
        symptom = gap.get("gap_symptom") or gap.get("summary")
        if symptom:
            lines.append(f"symptom: {_clip(symptom)}")
        evidence = gap.get("gap_evidence")
        if evidence:
            lines.append(f"evidence: {_clip(json.dumps(evidence, sort_keys=True))}")

        lines.append("")
        lines.append("=== Proposals to score ===")
        for entry in scoring_entries:
            p = entry.proposal
            lines.append(f"- id: {entry.stable_id}")
            lines.append(f"  name: {entry.label_name}")
            if p.get("extra_args"):
                lines.append(f"  extra_args: {_clip(p.get('extra_args'))}")
            if p.get("extra_envs"):
                lines.append(f"  extra_envs: {_clip(json.dumps(p.get('extra_envs'), sort_keys=True))}")
            if p.get("reason"):
                lines.append(f"  reason: {_clip(p.get('reason'))}")
            if p.get("kb_evidence"):
                lines.append(f"  kb_evidence: {_clip(json.dumps(p.get('kb_evidence'), sort_keys=True))}")
        return "\n".join(lines)

    async def _score_one_model(
        self,
        model: str,
        prompt: str,
        scoring_entries: list[_ScoringProposal],
        *,
        task_id: str | None = None,
        tick: int | None = None,
        phase: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Score every proposal with a single model (raises on failure; caller records the per-model error).

        Args:
            model: The model slug to score with.
            prompt: The base scoring prompt (instructions are appended).
            scoring_entries: Prepared proposals with stable ids.

        Returns:
            A mapping of proposal name to its normalised score and reason.

        Raises:
            RuntimeError: If the call times out or the reply has no
                parseable scores JSON.
        """
        client = self._ensure_client()
        full_prompt = f"{prompt}\n\n{_SCORING_INSTRUCTIONS}"
        messages = [{"role": "user", "content": full_prompt}]
        _t0 = time.perf_counter()

        # The proxy only accepts streamed requests; the shared helper accumulates
        # the deltas and pulls usage from the final chunk. The deadline wraps both
        # stream creation and the chunk-consumption loop (a proxy can stall
        # mid-body), via a single ``asyncio.wait_for`` (``asyncio.timeout`` is 3.11+).
        create_params = apply_reasoning_effort(
            {
                "model": model,
                "messages": messages,
                "max_completion_tokens": self.max_completion_tokens,
            }
        )
        try:
            text, usage = await asyncio.wait_for(
                astream_chat_completion_text(client, **create_params),
                timeout=self.call_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            error = RuntimeError(f"timed out after {self.call_timeout_s:.0f}s")
            self._trace_scorer_llm_failure(
                model,
                error,
                latency_ms=int((time.perf_counter() - _t0) * 1000),
                task_id=task_id,
                tick=tick,
                phase=phase,
            )
            raise error from exc
        except Exception as exc:
            # Anything else out of the stream (transport, proxy 5xx, malformed
            # chunk) is still a model call that produced nothing usable.
            # Cancellation is a BaseException and deliberately not caught.
            self._trace_scorer_llm_failure(
                model,
                exc,
                latency_ms=int((time.perf_counter() - _t0) * 1000),
                task_id=task_id,
                tick=tick,
                phase=phase,
            )
            raise
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        # One id for both halves of this call, so the emitter pairs them on the
        # call itself instead of on a ts-second bucket shared with the other
        # models being scored concurrently.
        call_id = new_call_id()
        # Record this model's token spend before parsing (best-effort).
        self._trace_scorer_llm_call(
            model,
            usage,
            latency_ms=latency_ms,
            task_id=task_id,
            tick=tick,
            phase=phase,
            call_id=call_id,
        )
        # Persist the full (redacted) prompt + reply alongside the token row.
        self._record_scorer_conversation(
            model,
            full_prompt,
            text,
            task_id=task_id,
            tick=tick,
            phase=phase,
            call_id=call_id,
        )
        parsed = _extract_scores_json(text)
        if parsed is None:
            raise RuntimeError(f"no parseable scores JSON (reply_chars={len(text)})")
        return _normalise_model_scores(parsed, scoring_entries=scoring_entries)

    def _trace_scorer_llm_call(
        self,
        model: str,
        usage: Any,
        *,
        latency_ms: int | None = None,
        task_id: str | None = None,
        tick: int | None = None,
        phase: str | None = None,
        call_id: str | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a proposal-scoring call.

        No-op when ``session_dir`` is unset. ``task_id`` ties the scoring spend
        back to the specialist round it scored, and ``tick`` / ``phase`` place
        it on the timeline. Best-effort: never raises into the scoring path.

        Args:
            model: The model slug whose usage is being recorded.
            usage: The OpenAI usage object from the response, or ``None``.
            latency_ms: Wall-clock latency of the scoring call, when measured.
            task_id: The specialist round this scoring spend is attributed to.
            tick: Timeline tick threaded from the coordinator dispatch point.
            phase: Optimization phase threaded from the coordinator dispatch point.
            call_id: Per-call id shared with this call's conversation row.
        """
        if self.session_dir is None:
            return
        try:
            input_tokens = None
            output_tokens = None
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", None)
                ct = getattr(usage, "completion_tokens", None)
                input_tokens = int(pt) if pt is not None else None
                output_tokens = int(ct) if ct is not None else None
            record = LLMCallRecord(
                session_id=self.session_dir.name,
                component="proposal_scorer",
                role="proposal_scorer",  # must match the conversation row's role
                call_id=call_id,
                model=str(model),
                task_id=task_id,
                tick=tick,
                phase=phase,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                # A scoring model with a reasoning split bills it separately;
                # reading it here keeps the ledger consistent with the backends
                # that report it on their turn metadata.
                reasoning_output_tokens=reasoning_output_tokens(usage),
                latency_ms=latency_ms,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break scoring
            log.debug(
                "full-trace: proposal_scorer llm_call append failed for model=%s",
                model,
                exc_info=True,
            )

    def _trace_scorer_llm_failure(
        self,
        model: str,
        error: BaseException,
        *,
        latency_ms: int | None = None,
        task_id: str | None = None,
        tick: int | None = None,
        phase: str | None = None,
    ) -> None:
        """Append one ``status="error"`` row for a scoring call that never returned.

        Recorded here rather than at the caller because the per-model context
        (model slug, task/tick/phase) is only known inside this coroutine — by
        the time :func:`asyncio.gather` has folded the exception into the
        ``errors`` map, which model failed is all that survives.

        Args:
            model: The model slug whose call failed.
            error: The exception that ended the call.
            latency_ms: Time spent before failing, when measured.
            task_id: The specialist round this scoring spend is attributed to.
            tick: Timeline tick threaded from the coordinator dispatch point.
            phase: Optimization phase threaded from the coordinator dispatch point.
        """
        if self.session_dir is None:
            return
        try:
            record = LLMCallRecord.for_failure(
                session_id=self.session_dir.name,
                component="proposal_scorer",
                role="proposal_scorer",
                error=error,
                model=str(model),
                task_id=task_id,
                tick=tick,
                phase=phase,
                latency_ms=latency_ms,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break scoring
            log.debug(
                "full-trace: proposal_scorer llm_call failure append failed for model=%s",
                model,
                exc_info=True,
            )

    def _record_scorer_conversation(
        self,
        model: str,
        prompt: str,
        response: str,
        *,
        task_id: str | None = None,
        tick: int | None = None,
        phase: str | None = None,
        call_id: str | None = None,
    ) -> None:
        """Append one ``conversations.jsonl`` row for a proposal-scoring call.

        Persists the full (redacted) scoring prompt + model reply under
        ``component=proposal_scorer``, mirroring the per-call token row from
        :meth:`_trace_scorer_llm_call`. No-op when ``session_dir`` is unset or
        when both prompt and reply are empty. Best-effort: never raises into
        the scoring path.

        Args:
            model: The model slug whose conversation is being recorded.
            prompt: The full (redacted) scoring prompt sent to the model.
            response: The model's reply text.
            task_id: The specialist round this scoring spend is attributed to.
            tick: Timeline tick threaded from the coordinator dispatch point.
            phase: Optimization phase threaded from the coordinator dispatch point.
            call_id: Per-call id shared with this call's token row.
        """
        if self.session_dir is None:
            return
        if not prompt and not response:
            return
        try:
            record = ConversationRecord(
                session_id=self.session_dir.name,
                component="proposal_scorer",
                role="proposal_scorer",
                call_id=call_id,
                task_id=task_id,
                tick=tick,
                phase=phase,
                model=str(model),
                prompt=prompt or "",
                response=response or "",
            )
            append_conversation(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break scoring
            log.debug(
                "full-trace: proposal_scorer conversation append failed for model=%s",
                model,
                exc_info=True,
            )

    async def score(
        self,
        *,
        gap: dict[str, Any],
        proposals: list[dict[str, Any]],
        task_id: str | None = None,
        tick: int | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        """Score ``proposals`` against ``gap`` with every configured model (per-model failures land in ``errors``, never raised).

        ``task_id`` (the specialist round being scored) is stamped on every
        per-model trace row for attribution; ``tick`` / ``phase`` place the rows
        on the timeline.

        Args:
            gap: The gap the proposals are meant to address.
            proposals: Candidate variants to score (non-dict entries ignored).
            task_id: The specialist round being scored, stamped on trace rows.
            tick: Timeline tick threaded from the coordinator dispatch point.
            phase: Optimization phase threaded from the coordinator dispatch point.

        Returns:
            A dict with the scoring ``scale``, per-model ``models`` scores,
            and per-model ``errors``.
        """
        proposals = [p for p in (proposals or []) if isinstance(p, dict)]
        if not proposals or not self.models:
            return {"scale": "0-10", "models": {}, "errors": {}}
        if len(proposals) > _MAX_PROPOSALS_SCORED:
            proposals = proposals[:_MAX_PROPOSALS_SCORED]
        try:
            scoring_entries = _prepare_scoring_proposals(proposals)
        except ValueError as exc:
            return {
                "scale": "0-10",
                "models": {},
                "errors": {"input": format_exc_brief(exc, limit=200)},
            }
        prompt = self._build_prompt(gap=gap, scoring_entries=scoring_entries)

        results = await asyncio.gather(
            *(
                self._score_one_model(
                    m,
                    prompt,
                    scoring_entries,
                    task_id=task_id,
                    tick=tick,
                    phase=phase,
                )
                for m in self.models
            ),
            return_exceptions=True,
        )
        models_out: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for model, res in zip(self.models, results):
            if isinstance(res, BaseException):
                errors[model] = format_exc_brief(res, limit=200)
                log.warning(
                    "ProposalScorer: model=%s failed: %r",
                    model,
                    res,
                )
                continue
            if res:
                models_out[model] = res
            else:
                errors[model] = "no usable scores returned"
        return {"scale": "0-10", "models": models_out, "errors": errors}


__all__ = ["ProposalScorer", "DEFAULT_SCORER_MODELS", "_extract_scores_json"]
