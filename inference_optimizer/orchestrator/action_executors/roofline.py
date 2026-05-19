"""Roofline ActionRunner — Roofline-v2 C4.

C4a (this commit): registers a **safe stub executor** that returns
``primary_bottleneck="unknown"`` with empty advice lists. This lets
``cli._register_executors`` wire the action and the
``MODEL_CLASS_ACTION_PRIORS`` row for ``roofline`` resolves to a real
runnable kind (no SubAgentRunner "no_executor" failure). The stub
records the snapshot_id from the cached ``last_select_kernels`` so
C4c's Coordinator integration can already exercise the
``record_roofline_analysis`` write path even before C4b lands.

C4b will replace ``RooflineStubExecutor`` with the real
``RooflineExecutor`` that spawns a sub-agent Claude backend, reads
``analysis.md`` from ``shared_state.last_select_kernels``, and
produces structured suggestions. The contract — call signature,
returned dict schema, ``status`` semantics, and ``degraded`` flag —
is documented on :class:`RooflineStubExecutor.__call__` and
:func:`build_roofline_fallback_result` and must be honoured by C4b.

The fallback builder is exported so C4b can reuse it for the
backend-failure / JSON-parse-failure / timeout paths without
duplicating the schema. See ``design/roofline-v2.md`` §8.4 for the
full pseudocode of the real executor that will replace this stub.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..sub_agent_runner import RunnerContext


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
    """Factory used by ``cli._register_executors`` (C4a wiring).

    Kept as a function so C4b can swap to ``make_roofline_executor``
    without touching the cli registration call-site signature."""
    return RooflineStubExecutor(shared_state=shared_state)


__all__ = [
    "RooflineStubExecutor",
    "build_roofline_fallback_result",
    "make_roofline_stub_executor",
]
