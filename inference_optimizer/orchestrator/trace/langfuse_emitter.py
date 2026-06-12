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
    trace_dir,
    trace_ext_dir,
)
from . import langfuse_mapping as lfmap
from .trace_env import (
    ENV_LANGFUSE_HOST,
    ENV_LANGFUSE_PUBLIC_KEY,
    ENV_LANGFUSE_SECRET_KEY,
    langfuse_credentials,
    langfuse_credentials_complete,
    langfuse_live_enabled,
)

log = logging.getLogger(__name__)


def _manifest_path(session_dir: Path) -> Path:
    """Return the path to a session's ``manifest.json``.

    Args:
        session_dir: Session directory.

    Returns:
        The manifest file path.
    """
    return session_dir / "manifest.json"


def _receipt_path(session_dir: Path) -> Path:
    """Return the path to a session's Langfuse receipt file.

    Args:
        session_dir: Session directory.

    Returns:
        The ``langfuse_receipt.json`` path under the trace directory.
    """
    return trace_dir(session_dir) / "langfuse_receipt.json"


def _sdk_available() -> bool:
    """Whether the optional ``langfuse`` SDK can be imported (no side effects)."""
    import importlib.util

    try:
        return importlib.util.find_spec("langfuse") is not None
    except Exception:  # noqa: BLE001
        return False


def _to_ns(dt: Any) -> int | None:
    """Datetime -> integer nanoseconds since epoch (langfuse v4 ``end_time``).

    v4's OTEL-based SDK wants integer ns for ``end_time``; v2/v3 accepted a
    ``datetime``. Returns None for a None/zero input so callers can omit the
    kwarg entirely. Best-effort: an unparseable value yields None.
    """
    if dt is None:
        return None
    try:
        from datetime import datetime

        if isinstance(dt, datetime):
            return int(dt.timestamp() * 1_000_000_000)
    except Exception:  # noqa: BLE001
        return None
    return None


def _start_obs(parent: Any, **kwargs: Any) -> Any:
    """Create a child/root observation, tolerant of v2/v3 vs v4 signatures.

    v2/v3 accepted ``start_time=<datetime>`` on ``start_observation``; v4
    removed it (the start is auto-stamped at creation). We try with the
    caller's kwargs first and, if the SDK rejects ``start_time`` with a
    TypeError, retry without it. This keeps backdated timestamps where the
    SDK supports them and degrades to "start = now" only on the SDKs that
    require it, instead of unconditionally dropping the timestamp.
    """
    try:
        return parent.start_observation(**kwargs)
    except TypeError:
        kwargs.pop("start_time", None)
        return parent.start_observation(**kwargs)


def _end_time_wants_int(obs: Any) -> bool:
    """Whether this SDK's ``end(end_time=...)`` wants integer ns (v4) vs a
    datetime (v2/v3), decided by inspecting the parameter annotation.

    We must get the type right on the FIRST call: v4's ``end(datetime)``
    raises a TypeError only *after* it has already ended the underlying
    OTEL span, so a "try datetime then retry int" pattern double-ends the
    span and emits a noisy "Calling end() on an ended span" warning.
    Falls back to datetime (False) when the annotation can't be read.
    """
    try:
        import inspect

        sig = inspect.signature(obs.end)
        ann = sig.parameters.get("end_time")
        ann_str = "" if ann is None else str(ann.annotation)
    except (TypeError, ValueError):
        return False
    return "int" in ann_str.lower()


def _end_obs(obs: Any, end_dt: Any) -> None:
    """End an observation, tolerant of v2/v3 (datetime) vs v4 (int ns).

    Picks the right ``end_time`` type up front (see :func:`_end_time_wants_int`)
    so the span is never ended twice. Falls back to a bare ``end()`` if the
    typed call is rejected, so a signature change can't strand an open span.
    """
    if obs is None:
        return
    if end_dt is None:
        try:
            obs.end()
        except Exception:  # noqa: BLE001
            pass
        return
    end_time = _to_ns(end_dt) if _end_time_wants_int(obs) else end_dt
    try:
        obs.end(end_time=end_time)
    except (TypeError, ValueError):
        try:
            obs.end()
        except Exception:  # noqa: BLE001
            pass


def _otel_attr_value(v: Any) -> Any:
    """Coerce a metadata value into an OTEL-acceptable attribute, or None to
    skip it.

    OTEL span attributes accept only str/bool/int/float (and homogeneous
    sequences of those). Passing ``None`` or a ``dict``/nested value makes
    the SDK log ``Invalid type ... for attribute`` and drop it. So: skip
    ``None``, pass scalars through, and JSON-stringify everything else
    (e.g. the ``workload`` dict) so it still lands on the trace as text.
    """
    if v is None:
        return None
    if isinstance(v, (str, bool, int, float)):
        return v
    try:
        import json

        return json.dumps(v, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(v)


def _set_trace_attrs(
    span: Any,
    *,
    name: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Stamp trace-level name/session_id/metadata, tolerant of v2/v3 vs v4.

    v2/v3 exposed ``span.update_trace(...)``. v4 removed it (decomposed the
    method), so we fall back to writing the documented v4 OTEL trace
    attributes directly on the underlying span (``langfuse.trace.name``,
    ``session.id``, ``langfuse.trace.metadata.*``). Best-effort: a missing
    API on either side just means the trace label isn't set, never a raise.
    """
    try:
        span.update_trace(name=name, session_id=session_id, metadata=metadata)
        return
    except AttributeError:
        pass  # v4: no update_trace — fall through to OTEL attributes
    except Exception:  # noqa: BLE001
        log.debug("langfuse: update_trace failed", exc_info=True)
        return
    otel = getattr(span, "_otel_span", None)
    if otel is None or not hasattr(otel, "set_attribute"):
        log.debug("langfuse: no OTEL span to set trace attrs on")
        return
    try:
        if name is not None:
            otel.set_attribute("langfuse.trace.name", name)
        if session_id is not None:
            otel.set_attribute("session.id", session_id)
        for k, v in (metadata or {}).items():
            clean = _otel_attr_value(v)
            if clean is None:
                continue  # OTEL rejects None / drops it with a warning
            try:
                otel.set_attribute(f"langfuse.trace.metadata.{k}", clean)
            except Exception:  # noqa: BLE001 — skip unserialisable values
                continue
    except Exception:  # noqa: BLE001
        log.debug("langfuse: setting OTEL trace attrs failed", exc_info=True)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dict records.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        The dict records; missing files, unreadable files, and malformed lines
        yield (or are skipped to) an empty/partial list.
    """
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
    """Load a JSON object file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed object, or ``{}`` when the file is missing, unreadable, or
        not a JSON object.
    """
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
        """Initialize the per-session emitter.

        Resolves the manifest and correlation labels unconditionally (even when
        the live push is disabled) so the receipt always reports the right trace
        and correlation ids.

        Args:
            session_dir: Session directory whose traces are emitted.
        """
        self.session_dir = Path(session_dir)
        self._lock = threading.Lock()
        # pair_key -> partial generation parts ({"llm": row} / {"conv": row}).
        self._pending: dict[tuple, dict[str, dict[str, Any]]] = {}
        self._client: Any = None
        # Manifest + correlation are resolved unconditionally (even when the
        # push is disabled) so the receipt always reports the right trace id /
        # correlation key. Live push, offline backfill, and any claw-side
        # upload correlate on claw_session_id (fallback internal session id).
        self._manifest: dict[str, Any] = _load_json(_manifest_path(self.session_dir))
        self._session_label: str | None = lfmap.langfuse_session_id(
            self._manifest, self.session_dir.name,
        )
        self._trace_id: str | None = lfmap.derive_trace_id(
            lfmap.correlation_seed(self._manifest, self.session_dir.name),
        )
        # Span hierarchy caches (lazy): root trace span; one span per phase;
        # one span per (phase, agent). Each Generation nests in its agent span.
        self._root_span: Any = None
        self._phase_spans: dict[str, Any] = {}
        self._agent_spans: dict[tuple[str, str], Any] = {}
        self._trace_attrs_set = False
        # Receipt counters (for the session_breakdown ``langfuse`` section).
        # ``disabled_reason`` is the gate that tripped when not enabled.
        self._disabled_reason: str | None = None
        self._counts: dict[str, int] = {
            "generations_sent": 0,      # Generations successfully started
            "generations_paired": 0,    # of which had both token + text halves
            "generations_text_only": 0,
            "generations_token_only": 0,
            "scores_sent": 0,           # decision Scores created (span + trace)
            "spans_opened": 0,          # phase + agent spans created
            "ext_shards_read": 0,       # out-of-process ext/*.jsonl files swept
            "breakdown_recorded": 0,    # 1 once the full SBD JSON was attached
            "errors": 0,                # swallowed send failures
        }
        self._flushed = False
        self._enabled = self._init_client()

    # -- gating / client setup ------------------------------------------
    def _init_client(self) -> bool:
        """Resolve the three gates and build the SDK client; False -> no-op.

        Records ``_disabled_reason`` (``disabled`` / ``no_credentials`` /
        ``sdk_missing`` / ``init_failed``) so the receipt can explain *why*
        nothing was pushed. Correlation is already resolved in ``__init__``.
        """
        if not langfuse_live_enabled():
            self._disabled_reason = "disabled"
            return False
        if not langfuse_credentials_complete():
            self._disabled_reason = "no_credentials"
            log.warning(
                "langfuse: HYPERLOOM_LANGFUSE_ENABLE is on but LANGFUSE_HOST/"
                "PUBLIC_KEY/SECRET_KEY are not all set; live push disabled.",
            )
            return False
        try:
            from langfuse import get_client  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._disabled_reason = "sdk_missing"
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
            log.info(
                "langfuse: live push enabled (host=%s, session=%s, trace_id=%s)",
                creds.get("LANGFUSE_HOST"), self._session_label, self._trace_id,
            )
            return True
        except Exception:  # noqa: BLE001
            self._disabled_reason = "init_failed"
            log.warning("langfuse: client init failed; live push disabled.", exc_info=True)
            return False

    @property
    def enabled(self) -> bool:
        """Whether live push to Langfuse is enabled for this session."""
        return self._enabled

    # -- span hierarchy (trace -> phase -> agent -> generation) ---------
    def _trace_name(self) -> str:
        """Return the human-readable trace name.

        Returns:
            The model name, else the session label, else ``"hyperloom"``.
        """
        return str(self._manifest.get("model_name") or self._session_label or "hyperloom")

    def _ensure_root(self, start: Any) -> Any:
        """Lazily open the root span and stamp trace-level attrs once."""
        if self._root_span is None:
            self._root_span = _start_obs(
                self._client,
                name=self._trace_name(),
                as_type="span",
                start_time=start,
                trace_context={"trace_id": self._trace_id},
                metadata=lfmap.trace_metadata(self._manifest),
            )
            if not self._trace_attrs_set:
                _set_trace_attrs(
                    self._root_span,
                    name=self._trace_name(),
                    session_id=self._session_label,
                    metadata=lfmap.trace_metadata(self._manifest),
                )
                self._trace_attrs_set = True
        return self._root_span

    def _ensure_phase_span(self, phase: str, start: Any) -> Any:
        """Get-or-create the span for a phase under the trace root.

        Args:
            phase: Phase name.
            start: Span start time.

        Returns:
            The cached or newly opened phase span.
        """
        span = self._phase_spans.get(phase)
        if span is None:
            root = self._ensure_root(start)
            span = _start_obs(
                root,
                name=f"phase:{phase}", as_type="span", start_time=start,
                metadata={"phase": phase},
            )
            self._phase_spans[phase] = span
            self._counts["spans_opened"] += 1
        return span

    def _ensure_agent_span(self, phase: str, agent: str, start: Any) -> Any:
        """Get-or-create the per-(phase, agent) span. This is the 'which agent
        did what' layer; Generations and decision Scores attach here."""
        key = (phase, agent)
        span = self._agent_spans.get(key)
        if span is None:
            phase_span = self._ensure_phase_span(phase, start)
            span = _start_obs(
                phase_span,
                name=f"agent:{agent}", as_type="span", start_time=start,
                metadata={"phase": phase, "agent": agent},
            )
            self._agent_spans[key] = span
            self._counts["spans_opened"] += 1
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
        """Buffer one half of a generation and emit once both halves arrive.

        Args:
            row: The token (``llm``) or conversation (``conv``) row.
            half: Which half this row represents (``"llm"`` or ``"conv"``).
        """
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
            gen = _start_obs(
                parent,
                name=lfmap.generation_name(base),
                as_type="generation",
                start_time=start,
                model=base.get("model"),
                input=(conv_row or {}).get("prompt"),
                output=(conv_row or {}).get("response"),
                metadata=lfmap.generation_metadata(base, phase=phase, has_text=has_text),
                usage_details=lfmap.usage_details(token_row or {}),
            )
            _end_obs(gen, start)
            self._counts["generations_sent"] += 1
            if token_row is not None and conv_row is not None:
                self._counts["generations_paired"] += 1
            elif conv_row is not None:
                self._counts["generations_text_only"] += 1
            else:
                self._counts["generations_token_only"] += 1
        except Exception:  # noqa: BLE001
            self._counts["errors"] += 1
            log.debug("langfuse: emit generation failed", exc_info=True)

    # -- session-end reconcile ------------------------------------------
    def flush_session(self) -> None:
        """Emit leftovers + out-of-process Generations + decision Scores, then flush.

        Run once at session end (from the Coordinator/CLI). Safe to call when
        disabled (no-op). **Idempotent**: a second call is a no-op for the
        push side -- it only re-writes the receipt. Without this guard a
        re-run would re-scan ``ext/*.jsonl`` + ``decision_trace`` and re-emit
        the same out-of-process Generations / Scores, producing duplicates in
        Langfuse (the derived trace_id keeps re-runs on one trace, but the
        children would still double up).
        """
        if not self._enabled:
            # Still drop a receipt so the breakdown can report *why* nothing
            # was pushed (disabled / no_credentials / sdk_missing).
            self._write_receipt()
            return
        if self._flushed:
            log.debug("langfuse: flush_session already ran; skipping re-emit")
            self._write_receipt()
            return
        try:
            self._flush_pending_halves()
            self._flush_ext_shards()
            self._flush_decision_scores()
            self._close_spans()
        except Exception:  # noqa: BLE001
            self._counts["errors"] += 1
            log.debug("langfuse: flush_session reconcile failed", exc_info=True)
        finally:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                self._counts["errors"] += 1
                log.debug("langfuse: client.flush failed", exc_info=True)
            self._flushed = True
            self._write_receipt()

    def record_session_breakdown(self, breakdown: dict[str, Any]) -> None:
        """Attach the complete ``session_breakdown.json`` document to the trace.

        Emitted as one ``session_breakdown`` observation whose ``output`` is the
        full JSON, attached to this session's ``trace_id`` so it lands on the
        same trace even when called after :meth:`flush_session` has closed the
        live spans (the normal order: write file -> flush -> patch langfuse ->
        record here). Idempotent (a second call is a no-op) and best-effort:
        any send failure is swallowed and never breaks shutdown.
        """
        if not self._enabled or not isinstance(breakdown, dict) or not breakdown:
            return
        if self._counts.get("breakdown_recorded"):
            return
        # Cross-process guard: a prior process (the original run, or an earlier
        # `recover-session`) may have already attached the document. The
        # persisted receipt is the only state shared across processes, so an
        # offline recovery in a fresh process stays idempotent.
        persisted = read_receipt(self.session_dir) or {}
        if (persisted.get("counts") or {}).get("breakdown_recorded"):
            self._counts["breakdown_recorded"] = 1
            return
        try:
            obs = _start_obs(
                self._client,
                name="session_breakdown",
                as_type="span",
                trace_context={"trace_id": self._trace_id},
                input=None,
                output=breakdown,
                metadata={
                    "schema_version": breakdown.get("schema_version"),
                    "exporter_version": breakdown.get("exporter_version"),
                    "stop_reason": (breakdown.get("session") or {}).get("stop_reason"),
                },
            )
            # Stamp trace name/session_id too, so a session whose only trace
            # artifact is the breakdown (e.g. no LLM calls) is still grouped.
            _set_trace_attrs(
                obs, name=self._trace_name(), session_id=self._session_label,
            )
            _end_obs(obs, None)
            self._counts["breakdown_recorded"] = 1
        except Exception:  # noqa: BLE001
            self._counts["errors"] += 1
            log.debug("langfuse: record_session_breakdown failed", exc_info=True)
        finally:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                self._counts["errors"] += 1
                log.debug("langfuse: flush after breakdown failed", exc_info=True)
            # Persist the breakdown_recorded flag so a later process (recovery)
            # sees it and skips re-attaching the document.
            self._write_receipt()

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
        """End a span, swallowing any errors.

        Args:
            span: The span object to end.
        """
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
            self._counts["ext_shards_read"] += 1
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
        """Attach a Langfuse Score to the owning agent span or the trace.

        Args:
            score: Score payload (``name``, ``value``, ``data_type``, ...).
            phase: Phase that owns the decision, used to locate the span.
            agent: Agent/component that owns the decision.
        """
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
            self._counts["scores_sent"] += 1
        except Exception:  # noqa: BLE001
            self._counts["errors"] += 1
            log.debug(
                "langfuse: create_score failed for %s", score.get("name"),
                exc_info=True,
            )

    # -- receipt (session_breakdown ``langfuse`` section) ---------------
    def receipt(self) -> dict[str, Any]:
        """A redacted record of whether/where/how much was pushed.

        Shape mirrors the ``langfuse`` section of ``session_breakdown.json``.
        Credentials are never included verbatim -- only the host URL (not a
        secret) and booleans noting that the keys were present. ``counts_final``
        is True once :meth:`flush_session` has run (so the out-of-process ext
        shards and decision scores are reflected); before that it reports the
        in-process running totals.
        """
        creds = langfuse_credentials()
        config = {
            "enable_flag": langfuse_live_enabled(),
            "host": creds.get(ENV_LANGFUSE_HOST),
            "public_key_set": ENV_LANGFUSE_PUBLIC_KEY in creds,
            "secret_key_set": ENV_LANGFUSE_SECRET_KEY in creds,
            "sdk_available": _sdk_available(),
        }
        return {
            "enabled": self._enabled,
            "disabled_reason": self._disabled_reason,
            "config": config,
            "trace_id": self._trace_id,
            "session_id": self._session_label,
            "correlated_on": (
                "claw_session_id"
                if str(self._manifest.get("claw_session_id") or "").strip()
                else "internal_session_id"
            ),
            "counts": dict(self._counts),
            "counts_final": self._flushed,
        }

    def _write_receipt(self) -> None:
        """Persist :meth:`receipt` to ``reports/trace/langfuse_receipt.json``.

        Best-effort: a failed receipt write must never break shutdown. The
        breakdown collector prefers this file (it reflects the post-flush
        final counts) over a live read of the emitter singleton.
        """
        import json

        try:
            path = _receipt_path(self.session_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.receipt(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            log.debug("langfuse: receipt write failed", exc_info=True)


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


def _read_breakdown_file(session_dir: Path) -> dict[str, Any]:
    """Load the written ``session_breakdown.json`` for ``session_dir`` ({} if absent)."""
    import json

    from ...breakdown import BREAKDOWN_FILENAME

    path = Path(session_dir) / BREAKDOWN_FILENAME
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def record_session_breakdown(
    session_dir: Path,
    breakdown: dict[str, Any] | None = None,
) -> None:
    """Attach the final ``session_breakdown.json`` to the session's trace.

    Call after the breakdown is written and the langfuse section patched (so
    the attached document is the complete, post-flush form). Reads the file
    from disk when ``breakdown`` is not supplied. No-op when live push is
    disabled; best-effort (never raises).
    """
    if breakdown is None:
        breakdown = _read_breakdown_file(Path(session_dir))
    get_emitter(session_dir).record_session_breakdown(breakdown)


def read_receipt(session_dir: Path) -> dict[str, Any] | None:
    """Read the persisted ``langfuse_receipt.json`` for ``session_dir``.

    Returns the post-flush receipt dict (preferred by the breakdown
    collector, since its counts are final) or ``None`` if no receipt was
    written -- e.g. the breakdown is being assembled before ``flush_session``
    ran, or live push never happened.
    """
    import json

    path = _receipt_path(session_dir)
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


__all__ = [
    "LangfuseEmitter",
    "flush_session",
    "get_emitter",
    "read_receipt",
    "record_session_breakdown",
]
