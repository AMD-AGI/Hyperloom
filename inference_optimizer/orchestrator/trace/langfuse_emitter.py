# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Live Langfuse push for the trace subsystem (opt-in, best-effort).

This is the *second* of the two parallel trace sinks. The first -- the
local ``reports/trace/*.jsonl`` ledger -- is always written and its format
is unchanged. This module mirrors each in-process LLM call into Langfuse as
a Generation *while the run is live*, plus a session-end ``flush_session``
that backfills the out-of-process children (geak / oob / robustness /
specialist subprocess, which only surface their tokens in ``ext/*.jsonl``
after the parent parses them) and the KEEP/REVERT decision Scores.

Three gates decide whether anything is sent (all must pass, else no-op):

1. master switch ``HYPERLOOM_LANGFUSE_ENABLE`` is on (default off);
2. the three ``LANGFUSE_*`` connection vars are all set;
3. the ``langfuse`` SDK is importable.

Fault posture mirrors :func:`..llm_trace.append_llm_call`: every send is
best-effort and any exception is logged-and-swallowed. A Langfuse outage,
a missing SDK, or a malformed row must never break the optimization loop or
the local jsonl write.

The Generation needs model + usage + prompt/response together, but those
arrive as two separate calls (``record_llm_call`` for tokens,
``record_conversation`` for text) a few ms apart. The emitter buffers by
:func:`..langfuse_mapping.pair_key` and emits as soon as both halves are in
(or at ``flush_session`` for whichever half never paired).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from ...session_paths import (
    decision_trace_path,
    trace_ext_dir,
)
from . import langfuse_mapping as lfmap
from .trace_env import (
    langfuse_credentials,
    langfuse_credentials_complete,
    langfuse_live_enabled,
)

log = logging.getLogger(__name__)


def _manifest_path(session_dir: Path) -> Path:
    return session_dir / "manifest.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    import json

    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


class LangfuseEmitter:
    """Per-session live emitter. No-op unless all three gates pass.

    One instance is created lazily per ``session_dir`` via
    :func:`get_emitter`. Thread-safe: the trace appenders may be called from
    the Coordinator loop and from worker threads, so the pairing buffer is
    guarded by a lock.
    """

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self._lock = threading.Lock()
        # pair_key -> partial generation parts ({"llm": row} / {"conv": row}).
        self._pending: dict[tuple, dict[str, dict[str, Any]]] = {}
        self._client: Any = None
        self._trace_id: str | None = None
        self._session_label: str | None = None
        self._manifest: dict[str, Any] = {}
        # Span hierarchy caches (lazy): root trace span; one span per phase;
        # one span per (phase, agent). Each Generation nests in its agent span.
        self._root_span: Any = None
        self._phase_spans: dict[str, Any] = {}
        self._agent_spans: dict[tuple[str, str], Any] = {}
        self._trace_attrs_set = False
        self._enabled = self._init_client()

    # -- gating / client setup ------------------------------------------
    def _init_client(self) -> bool:
        """Resolve the three gates and build the SDK client; False -> no-op."""
        if not langfuse_live_enabled():
            return False
        if not langfuse_credentials_complete():
            log.warning(
                "langfuse: HYPERLOOM_LANGFUSE_ENABLE is on but LANGFUSE_HOST/"
                "PUBLIC_KEY/SECRET_KEY are not all set; live push disabled.",
            )
            return False
        try:
            from langfuse import get_client  # type: ignore
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "langfuse: SDK not importable (%s: %s); live push disabled. "
                "Install the optional dependency: pip install 'hyperloom-"
                "inference_optimizer[trace]'.",
                type(exc).__name__, exc,
            )
            return False
        try:
            creds = langfuse_credentials()
            self._client = get_client()
            self._manifest = _load_json(_manifest_path(self.session_dir))
            # Correlate on the PrimusClaw session id (claw_session_id) so live
            # push, offline backfill, and any claw-side upload all land on one
            # Langfuse trace; fall back to the internal session id locally.
            self._session_label = lfmap.langfuse_session_id(
                self._manifest, self.session_dir.name,
            )
            self._trace_id = lfmap.derive_trace_id(
                lfmap.correlation_seed(self._manifest, self.session_dir.name),
            )
            log.info(
                "langfuse: live push enabled (host=%s, session=%s, trace_id=%s)",
                creds.get("LANGFUSE_HOST"), self._session_label, self._trace_id,
            )
            return True
        except Exception:  # noqa: BLE001
            log.warning("langfuse: client init failed; live push disabled.", exc_info=True)
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -- span hierarchy (trace -> phase -> agent -> generation) ---------
    def _trace_name(self) -> str:
        return str(self._manifest.get("model_name") or self._session_label or "hyperloom")

    def _ensure_root(self, start: Any) -> Any:
        """Lazily open the root span and stamp trace-level attrs once."""
        if self._root_span is None:
            self._root_span = self._client.start_observation(
                name=self._trace_name(),
                as_type="span",
                start_time=start,
                trace_context={"trace_id": self._trace_id},
                metadata=lfmap.trace_metadata(self._manifest),
            )
            if not self._trace_attrs_set:
                try:
                    self._root_span.update_trace(
                        name=self._trace_name(),
                        session_id=self._session_label,
                        metadata=lfmap.trace_metadata(self._manifest),
                    )
                except Exception:  # noqa: BLE001 — older SDKs may lack it
                    log.debug("langfuse: update_trace unavailable", exc_info=True)
                self._trace_attrs_set = True
        return self._root_span

    def _ensure_phase_span(self, phase: str, start: Any) -> Any:
        span = self._phase_spans.get(phase)
        if span is None:
            root = self._ensure_root(start)
            span = root.start_observation(
                name=f"phase:{phase}", as_type="span", start_time=start,
                metadata={"phase": phase},
            )
            self._phase_spans[phase] = span
        return span

    def _ensure_agent_span(self, phase: str, agent: str, start: Any) -> Any:
        """Get-or-create the per-(phase, agent) span. This is the 'which agent
        did what' layer; Generations and decision Scores attach here."""
        key = (phase, agent)
        span = self._agent_spans.get(key)
        if span is None:
            phase_span = self._ensure_phase_span(phase, start)
            span = phase_span.start_observation(
                name=f"agent:{agent}", as_type="span", start_time=start,
                metadata={"phase": phase, "agent": agent},
            )
            self._agent_spans[key] = span
        return span

    # -- live ingest ----------------------------------------------------
    def record_llm_call(self, row: dict[str, Any]) -> None:
        """Buffer a token row; emit the Generation if its text half is in."""
        if not self._enabled:
            return
        try:
            self._buffer(row, half="llm")
        except Exception:  # noqa: BLE001 — trace must never break the loop
            log.debug("langfuse: record_llm_call failed", exc_info=True)

    def record_conversation(self, row: dict[str, Any]) -> None:
        """Buffer a conversation row; emit the Generation if its tokens are in."""
        if not self._enabled:
            return
        try:
            self._buffer(row, half="conv")
        except Exception:  # noqa: BLE001
            log.debug("langfuse: record_conversation failed", exc_info=True)

    def _buffer(self, row: dict[str, Any], *, half: str) -> None:
        key = lfmap.pair_key(row)
        emit_parts: dict[str, dict[str, Any]] | None = None
        with self._lock:
            parts = self._pending.setdefault(key, {})
            parts[half] = row
            if "llm" in parts and "conv" in parts:
                emit_parts = self._pending.pop(key)
        if emit_parts is not None:
            self._emit_generation(
                token_row=emit_parts.get("llm"),
                conv_row=emit_parts.get("conv"),
            )

    def _emit_generation(
        self,
        *,
        token_row: dict[str, Any] | None,
        conv_row: dict[str, Any] | None,
    ) -> None:
        """Emit one Generation, nested under its phase -> agent span."""
        base = token_row or conv_row or {}
        phase = lfmap.phase_of(base)
        agent = lfmap.agent_of(base)
        start = lfmap.parse_ts(base.get("ts"))
        has_text = conv_row is not None
        try:
            parent = self._ensure_agent_span(phase, agent, start)
            gen = parent.start_observation(
                name=lfmap.generation_name(base),
                as_type="generation",
                start_time=start,
                model=base.get("model"),
                input=(conv_row or {}).get("prompt"),
                output=(conv_row or {}).get("response"),
                metadata=lfmap.generation_metadata(base, phase=phase, has_text=has_text),
                usage_details=lfmap.usage_details(token_row or {}),
            )
            gen.end(end_time=start)
        except Exception:  # noqa: BLE001
            log.debug("langfuse: emit generation failed", exc_info=True)

    # -- session-end reconcile ------------------------------------------
    def flush_session(self) -> None:
        """Emit leftovers + out-of-process Generations + decision Scores, then flush.

        Run once at session end (from the Coordinator/CLI). Safe to call when
        disabled (no-op) and safe to call more than once (idempotent-ish: the
        derived trace_id keeps re-runs on one trace).
        """
        if not self._enabled:
            return
        try:
            self._flush_pending_halves()
            self._flush_ext_shards()
            self._flush_decision_scores()
            self._close_spans()
        except Exception:  # noqa: BLE001
            log.debug("langfuse: flush_session reconcile failed", exc_info=True)
        finally:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                log.debug("langfuse: client.flush failed", exc_info=True)

    def _close_spans(self) -> None:
        """End every open span, innermost first (agent -> phase -> root)."""
        for span in list(self._agent_spans.values()):
            self._safe_end(span)
        for span in list(self._phase_spans.values()):
            self._safe_end(span)
        if self._root_span is not None:
            self._safe_end(self._root_span)

    @staticmethod
    def _safe_end(span: Any) -> None:
        try:
            span.end()
        except Exception:  # noqa: BLE001
            log.debug("langfuse: span end failed", exc_info=True)

    def _flush_pending_halves(self) -> None:
        """Emit any buffered call that only ever got one half (token XOR text)."""
        with self._lock:
            leftovers = list(self._pending.values())
            self._pending.clear()
        for parts in leftovers:
            self._emit_generation(
                token_row=parts.get("llm"),
                conv_row=parts.get("conv"),
            )

    def _flush_ext_shards(self) -> None:
        """Backfill out-of-process children's token rows from ext/*.jsonl.

        Children (geak / oob / robustness / specialist subprocess) never
        connect to Langfuse; their tokens land in ``ext/<component>-<pid>.
        jsonl`` once the parent parses them. We emit those as text-less
        Generations here so the trace still accounts their spend.
        """
        ext_dir = trace_ext_dir(self.session_dir)
        if not ext_dir.is_dir():
            return
        for shard in sorted(ext_dir.glob("*.jsonl")):
            for row in _load_jsonl(shard):
                self._emit_generation(token_row=row, conv_row=None)

    def _flush_decision_scores(self) -> None:
        """Convert each decision_trace row into Langfuse Score(s).

        Each score targets the agent span that owns the decision -- (phase,
        component) from the decision metadata -- so the KEEP/REVERT/gain_pct
        attaches to "which agent did this". When no matching span exists (the
        agent produced a decision but no LLM call, or phase/component is
        missing), it falls back to a trace-level score.
        """
        for drow in _load_jsonl(decision_trace_path(self.session_dir)):
            for score in lfmap.decision_to_scores(drow):
                meta = score.get("metadata") or {}
                phase = str(meta.get("phase") or lfmap.UNPHASED)
                agent = str(meta.get("component") or lfmap.UNKNOWN_AGENT)
                self._create_score(score, phase=phase, agent=agent)

    def _create_score(
        self, score: dict[str, Any], *, phase: str, agent: str,
    ) -> None:
        span = self._agent_spans.get((phase, agent))
        try:
            if span is not None and hasattr(span, "score"):
                span.score(
                    name=score["name"],
                    value=score["value"],
                    data_type=score["data_type"],
                    comment=score.get("comment") or "",
                    metadata=score.get("metadata") or {},
                )
            else:
                self._client.create_score(
                    name=score["name"],
                    value=score["value"],
                    trace_id=self._trace_id,
                    data_type=score["data_type"],
                    comment=score.get("comment") or "",
                    metadata=score.get("metadata") or {},
                )
        except Exception:  # noqa: BLE001
            log.debug(
                "langfuse: create_score failed for %s", score.get("name"),
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Process-wide singleton registry (one emitter per session_dir).
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, LangfuseEmitter] = {}
_REGISTRY_LOCK = threading.Lock()


def get_emitter(session_dir: Path) -> LangfuseEmitter:
    """Return the per-session emitter, building it once (cached by session)."""
    key = str(Path(session_dir).resolve())
    with _REGISTRY_LOCK:
        emitter = _REGISTRY.get(key)
        if emitter is None:
            emitter = LangfuseEmitter(Path(session_dir))
            _REGISTRY[key] = emitter
        return emitter


def flush_session(session_dir: Path) -> None:
    """Module-level convenience: flush the emitter for ``session_dir``."""
    get_emitter(session_dir).flush_session()


__all__ = ["LangfuseEmitter", "flush_session", "get_emitter"]
