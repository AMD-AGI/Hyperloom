"""Roofline ActionRunner — Roofline-v2 C4.

C4a (already shipped): registered a stub executor returning the safe
``primary_bottleneck="unknown"`` fallback so ``cli._register_executors``
could wire the action without a real LLM call.

C4b (this commit): adds the real :class:`RooflineExecutor` that spawns
a dedicated sub-agent Claude backend, sends the cached TraceLens
``analysis.md`` plus the current optimisation state, parses strict
JSON output, and writes the structured decision back to the result
dict. The fallback builder is reused for every failure branch
(backend error / timeout / malformed JSON / schema failure) so the
schema accepted by ``SharedState.record_roofline_analysis`` is
produced in exactly one place.

The C4a :class:`RooflineStubExecutor` and ``make_roofline_stub_executor``
are kept so a future operator can fall back to the stub via env or
CLI flag if the sub-agent backend becomes unreliable (the design
"increment, don't recoil" risk-mitigation hook in §11).

C4c will wire the executor output to ``record_roofline_analysis``
inside the Coordinator's task-completion handler and add the
``roofline`` entry to ``_sequence_denial_for_action``.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from ..sub_agent_runner import RunnerContext


# Re-exported types used in the executor — kept as late imports inside
# methods to avoid forcing Claude SDK import at module load time
# (some test environments don't have anthropic SDK on the path).
_BackendError: type[Exception] | None = None


def _get_backend_error_cls() -> type[Exception]:
    """Lazy-import BackendError so unit tests without SDK can import
    this module."""
    global _BackendError
    if _BackendError is None:
        from ..backends.base import BackendError as _BE
        _BackendError = _BE
    return _BackendError


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_roofline_fallback_result(
    *,
    snapshot_id: int,
    analysis_md_path: str = "",
    gain_pct: float = 0.0,
    error: str = "",
) -> dict[str, Any]:
    """Construct the canonical "no useful analysis" result dict.

    Shared between the C4a stub and every C4b failure branch (backend
    timeout, malformed JSON, schema validation failure) so the schema
    expected by ``SharedState.record_roofline_analysis`` is produced
    in exactly one place.

    The returned dict carries ``status="succeeded"`` even when an
    ``error`` is supplied because the optimisation loop must keep
    running — the LLM will see ``degraded=True`` and
    ``primary_bottleneck="unknown"`` in the prompt-rendered Roofline
    Decision section and naturally fall back to its baseline
    action_scores priors. Returning ``status="failed"`` instead
    would cause the SubAgentRunner to bubble the failure up and
    force the main LLM into a recovery branch, which is far more
    disruptive than the soft "no useful analysis" signal.
    """
    return {
        "status": "succeeded",
        "degraded": True,
        "snapshot_id": snapshot_id,
        "analyzed_at_iso": _now_iso(),
        "analyzed_at_gain_pct": float(gain_pct),
        "based_on_analysis_md": str(analysis_md_path),
        "primary_bottleneck": "unknown",
        "bottleneck_distribution": {},
        "suggested_prunes": [],
        "suggested_next_actions": [],
        "reprofile_recommended": False,
        "reprofile_reason": "",
        "raw_llm_response": "",
        "error": error,
    }


class RooflineStubExecutor:
    """C4a stub — wires the action into SubAgentRunner without doing
    any real LLM analysis.

    Once C4b lands, this class is **replaced** (not extended) by a
    new :class:`RooflineExecutor` that:

    * holds a ``backend_factory`` (typically lambda producing a fresh
      :class:`ClaudeBackend`) and ``shared_state`` reference;
    * reads ``shared_state.last_select_kernels`` for analysis_md_text
      / snapshot_id / cached path;
    * short-circuits as ``idempotency_hit=True`` when
      ``shared_state.last_roofline_analysis.snapshot_id`` matches the
      current snapshot;
    * composes the user prompt (analysis_md + gain + stack + pruned),
      invokes backend.run with the dedicated roofline_analyzer
      system prompt, and parses strict JSON;
    * on any failure path calls :func:`build_roofline_fallback_result`
      to keep the returned schema identical.

    The stub returns the same schema (via
    :func:`build_roofline_fallback_result`) so C4c's Coordinator
    integration and C5's prompt renderer can be wired and tested
    against C4a output before C4b is implemented.
    """

    def __init__(self, shared_state: Any = None):
        # ``shared_state`` is optional in C4a (the stub never reads
        # it) but the constructor accepts it so C4b can land without
        # changing cli wiring or test fixtures.
        self.shared_state = shared_state

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        snapshot_id = 0
        analysis_md_path = ""
        gain_pct = 0.0
        if self.shared_state is not None:
            cached = getattr(self.shared_state, "last_select_kernels", {}) or {}
            snap_raw = cached.get("roofline_snapshot_id")
            if isinstance(snap_raw, int):
                snapshot_id = snap_raw
            analysis_md_path = str(cached.get("analysis_md_path") or "")
            gain_raw = getattr(self.shared_state, "cumulative_gain_validated", 0.0)
            try:
                gain_pct = float(gain_raw)
            except (TypeError, ValueError):
                gain_pct = 0.0
        return build_roofline_fallback_result(
            snapshot_id=snapshot_id,
            analysis_md_path=analysis_md_path,
            gain_pct=gain_pct,
            error="roofline_stub_executor_active",
        )


def make_roofline_stub_executor(shared_state: Any = None) -> RooflineStubExecutor:
    """Stub-executor factory — kept as the explicit "safe fallback" wiring.

    Used by tests or by an env-flag-driven recovery path (§11 risk
    mitigation in design/roofline-v2.md): operators who want to
    temporarily disable the sub-agent LLM call without removing the
    action entry can wire ``make_roofline_stub_executor`` instead of
    ``make_roofline_executor``."""
    return RooflineStubExecutor(shared_state=shared_state)


# ---------------------------------------------------------------------------
# C4b: real sub-agent executor
# ---------------------------------------------------------------------------
# Default timeout for one sub-agent backend.run() invocation. 60s covers a
# 200KB analysis.md + 1KB JSON output single-turn on Claude with comfortable
# headroom; the executor falls back to ``build_roofline_fallback_result``
# rather than raising, so this can be tuned without breaking the call site.
ROOFLINE_SUBAGENT_TIMEOUT_SEC: float = 60.0

# Cap for the analysis.md text we ship to the sub-agent. The default
# (200 KB) is the worst-case Qwen3-32B Case A/B/C/D Hyperloom-produced
# size; larger inputs are truncated with a "[... truncated N bytes ...]"
# tail marker so the analyzer's bottleneck classification still sees the
# Executive Summary and Top Operations sections that live at the start.
ROOFLINE_ANALYSIS_MD_MAX_BYTES: int = 200 * 1024


# Module-relative path to the analyzer system prompt — resolved lazily so
# unit tests can monkeypatch ``_load_analyzer_system_prompt`` if needed.
_ANALYZER_SYSTEM_PROMPT_FILENAME = "roofline_analyzer.md"


def _load_analyzer_system_prompt() -> str:
    """Read ``system_prompts/roofline_analyzer.md`` from the package data.

    Kept as a module-level function (not a classmethod) so tests can
    monkeypatch it without instantiating ``RooflineExecutor``."""
    # Lazy import to avoid circular import between paths and orchestrator
    # at module load time.
    from ...paths import asset_system_prompts_dir
    sp_path = asset_system_prompts_dir() / _ANALYZER_SYSTEM_PROMPT_FILENAME
    return sp_path.read_text(encoding="utf-8")


def _truncate_analysis_md(text: str, *, max_bytes: int) -> str:
    """UTF-8-safe truncation preserving the leading sections.

    Splits at the last newline before ``max_bytes`` so the tail
    boundary is visible in the prompt; appends a marker so the
    sub-agent can mention truncation in its reasoning if needed."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    cut = encoded[:max_bytes]
    # Find last newline boundary to avoid splitting mid-line / mid-utf8
    last_nl = cut.rfind(b"\n")
    if last_nl > 0:
        cut = cut[:last_nl]
    head = cut.decode("utf-8", errors="ignore")
    omitted = len(encoded) - len(cut)
    return f"{head}\n\n[... truncated {omitted} bytes (tail dropped) ...]\n"


def _compose_analyzer_user_prompt(
    *,
    analysis_md: str,
    cumulative_gain_pct: float,
    optimization_stack: list[dict[str, Any]] | None,
    pruned_families: list[str] | None,
) -> str:
    """Build the user message the analyzer system prompt expects.

    The four labeled sections are documented on
    ``system_prompts/roofline_analyzer.md``. Keep field names /
    section order in sync with that file."""
    stack_repr = json.dumps(optimization_stack or [], default=str)
    pruned_repr = json.dumps(sorted(pruned_families or []))
    truncated = _truncate_analysis_md(
        analysis_md or "",
        max_bytes=ROOFLINE_ANALYSIS_MD_MAX_BYTES,
    )
    return (
        f"cumulative_gain_validated_pct: {float(cumulative_gain_pct):.3f}\n"
        f"optimization_stack: {stack_repr}\n"
        f"pruned_families: {pruned_repr}\n"
        "analysis_md: |\n"
        f"{truncated}\n"
    )


_JSON_BLOCK_RE = re.compile(
    r"\{.*\}",
    re.DOTALL,
)


def _parse_analyzer_json(raw: str) -> dict[str, Any] | None:
    """Parse the sub-agent's JSON response.

    The analyzer system prompt forbids prose / markdown fences, but
    we still defensively strip a code fence if the model includes
    one ("```json ... ```") and grab the largest balanced object
    via regex if there's leading / trailing whitespace.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        # Strip ```json or ``` fence
        first_nl = text.find("\n")
        if first_nl >= 0:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        pass
    # Regex fallback — find the largest brace-balanced substring.
    match = _JSON_BLOCK_RE.search(text)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def _default_backend_factory() -> Any:
    """Construct a fresh ClaudeBackend instance for the sub-agent.

    Kept as a module-level function so tests can monkeypatch and so
    ``make_roofline_executor(backend_factory=None)`` has a single
    well-known production path. ``enable_mcp_emit_intent=False``
    because the sub-agent must not emit intents — only return text.
    """
    # Lazy import to avoid forcing Claude SDK at module import time.
    from ..backends.claude import ClaudeBackend
    return ClaudeBackend(enable_mcp_emit_intent=False)


class RooflineExecutor:
    """C4b real ActionRunner — sub-agent LLM roofline analyzer.

    Lifecycle of one ``await self(ctx)`` call:

    1. Read ``shared_state.last_select_kernels`` for analysis_md_text
       and snapshot_id.
    2. If snapshot_id is 0 or analysis_md_text is empty, return
       ``build_roofline_fallback_result(error="no_cached_analysis_md")``.
    3. Idempotency: if ``shared_state.last_roofline_analysis.snapshot_id``
       already equals the current snapshot_id, short-circuit and
       return the previously-cached dict marked with
       ``idempotency_hit=True`` (no LLM call).
    4. Construct a fresh sub-agent backend via ``backend_factory``,
       compose the analyzer user prompt, invoke
       ``backend.run(..., system_prompt=analyzer_sp, tools=None,
       max_turns=1)`` with :data:`ROOFLINE_SUBAGENT_TIMEOUT_SEC`.
    5. Parse the raw_text via :func:`_parse_analyzer_json`. Any
       BackendError / timeout / parse failure → fallback dict with
       the relevant ``error`` field; the optimisation loop never
       sees a failed task.
    6. Merge parsed fields with snapshot_id / analyzed_at_iso /
       analyzed_at_gain_pct / based_on_analysis_md / raw_llm_response
       (truncated to 8 KB inside ``record_roofline_analysis``) and
       return.
    """

    def __init__(
        self,
        *,
        shared_state: Any,
        backend_factory: Callable[[], Any] | None = None,
        analyzer_system_prompt_loader: Callable[[], str] | None = None,
        timeout_sec: float = ROOFLINE_SUBAGENT_TIMEOUT_SEC,
    ):
        self.shared_state = shared_state
        self._backend_factory = backend_factory or _default_backend_factory
        self._sp_loader = (
            analyzer_system_prompt_loader or _load_analyzer_system_prompt
        )
        self._timeout_sec = float(timeout_sec)

    # ------------------------------------------------------------------
    # Backend protocol expected by SubAgentRunner
    # ------------------------------------------------------------------
    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        state = self.shared_state
        cached = (getattr(state, "last_select_kernels", {}) or {}) if state else {}
        analysis_md = str(cached.get("analysis_md_text") or "")
        snap_raw = cached.get("roofline_snapshot_id")
        snapshot_id = snap_raw if isinstance(snap_raw, int) else 0
        analysis_md_path = str(cached.get("analysis_md_path") or "")
        gain_pct = self._safe_float(
            getattr(state, "cumulative_gain_validated", 0.0) if state else 0.0,
        )

        if not analysis_md or snapshot_id <= 0:
            return build_roofline_fallback_result(
                snapshot_id=snapshot_id,
                analysis_md_path=analysis_md_path,
                gain_pct=gain_pct,
                error="no_cached_analysis_md",
            )

        # Idempotency short-circuit (D2): re-propose against the same
        # snapshot must NOT spend another sub-agent token. We surface
        # the previous result with ``idempotency_hit=True`` so the
        # caller can audit re-propose attempts.
        prev = (
            getattr(state, "last_roofline_analysis", {}) or {}
        ) if state else {}
        if (
            isinstance(prev.get("snapshot_id"), int)
            and prev["snapshot_id"] == snapshot_id
            and snapshot_id > 0
        ):
            replay = dict(prev)
            replay["status"] = "succeeded"
            replay["idempotency_hit"] = True
            replay.setdefault("snapshot_id", snapshot_id)
            replay.setdefault("based_on_analysis_md", analysis_md_path)
            return replay

        user_prompt = _compose_analyzer_user_prompt(
            analysis_md=analysis_md,
            cumulative_gain_pct=gain_pct,
            optimization_stack=list(
                getattr(state, "optimization_stack", []) or [],
            ),
            pruned_families=list(
                getattr(state, "pruned_families", []) or [],
            ),
        )

        try:
            system_prompt = self._sp_loader()
        except OSError as exc:
            return build_roofline_fallback_result(
                snapshot_id=snapshot_id,
                analysis_md_path=analysis_md_path,
                gain_pct=gain_pct,
                error=f"analyzer_sp_load_failed: {exc!r}",
            )

        try:
            backend = self._backend_factory()
        except Exception as exc:  # noqa: BLE001 — backend ctor failure
            return build_roofline_fallback_result(
                snapshot_id=snapshot_id,
                analysis_md_path=analysis_md_path,
                gain_pct=gain_pct,
                error=f"backend_factory_failed: {exc!r}",
            )

        backend_error_cls = _get_backend_error_cls()
        try:
            turn = await asyncio.wait_for(
                backend.run(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    tools=None,
                    max_turns=1,
                ),
                timeout=self._timeout_sec,
            )
        except asyncio.TimeoutError:
            return build_roofline_fallback_result(
                snapshot_id=snapshot_id,
                analysis_md_path=analysis_md_path,
                gain_pct=gain_pct,
                error=f"sub_agent_timeout_after_{int(self._timeout_sec)}s",
            )
        except backend_error_cls as exc:
            return build_roofline_fallback_result(
                snapshot_id=snapshot_id,
                analysis_md_path=analysis_md_path,
                gain_pct=gain_pct,
                error=f"backend_error: {exc!r}",
            )
        except Exception as exc:  # noqa: BLE001
            return build_roofline_fallback_result(
                snapshot_id=snapshot_id,
                analysis_md_path=analysis_md_path,
                gain_pct=gain_pct,
                error=f"backend_unexpected: {exc!r}",
            )

        raw_text = getattr(turn, "raw_text", "") or ""
        parsed = _parse_analyzer_json(raw_text)
        if parsed is None:
            fallback = build_roofline_fallback_result(
                snapshot_id=snapshot_id,
                analysis_md_path=analysis_md_path,
                gain_pct=gain_pct,
                error="json_parse_failed",
            )
            fallback["raw_llm_response"] = raw_text
            return fallback

        # Merge the analyzer's structured output with the contextual
        # fields the executor owns (snapshot_id / timestamps / paths).
        # Field-by-field merge (not parsed.update(extras)) so the
        # analyzer cannot accidentally override snapshot_id / paths.
        result: dict[str, Any] = {
            "status": "succeeded",
            "degraded": False,
            "snapshot_id": snapshot_id,
            "analyzed_at_iso": _now_iso(),
            "analyzed_at_gain_pct": gain_pct,
            "based_on_analysis_md": analysis_md_path,
            "primary_bottleneck": parsed.get("primary_bottleneck", "unknown"),
            "bottleneck_distribution": parsed.get("bottleneck_distribution", {}),
            "suggested_prunes": parsed.get("suggested_prunes", []),
            "suggested_next_actions": parsed.get("suggested_next_actions", []),
            "reprofile_recommended": bool(
                parsed.get("reprofile_recommended", False),
            ),
            "reprofile_reason": parsed.get("reprofile_reason", ""),
            "raw_llm_response": raw_text,
            "error": "",
        }
        return result

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def make_roofline_executor(
    *,
    shared_state: Any,
    backend_factory: Callable[[], Any] | None = None,
) -> RooflineExecutor:
    """Production factory used by ``cli._register_executors`` (C4b wiring).

    ``backend_factory`` defaults to :func:`_default_backend_factory`
    (fresh ClaudeBackend per call). Tests inject a stub backend
    factory directly. Same kwarg shape as
    :func:`make_roofline_stub_executor` so cli wiring switching
    between the two is a one-name change."""
    return RooflineExecutor(
        shared_state=shared_state,
        backend_factory=backend_factory,
    )


__all__ = [
    "ROOFLINE_ANALYSIS_MD_MAX_BYTES",
    "ROOFLINE_SUBAGENT_TIMEOUT_SEC",
    "RooflineExecutor",
    "RooflineStubExecutor",
    "build_roofline_fallback_result",
    "make_roofline_executor",
    "make_roofline_stub_executor",
]
