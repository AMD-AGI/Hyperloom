# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Live Langfuse push for the trace subsystem (opt-in, best-effort).

The second of two parallel trace sinks (the local ``reports/trace/*.jsonl``
ledger is always written). This module mirrors each LLM call into Langfuse as a
Generation while the run is live, plus a session-end ``flush_session`` that
backfills the recipe-KB / specialist-intel audit spans and the KEEP/REVERT
decision Scores.

Three gates decide whether anything is sent (all must pass, else no-op):

1. master switch ``HYPERLOOM_LANGFUSE_ENABLE`` is on (default off);
2. the three ``LANGFUSE_*`` connection vars are all set;
3. the ``langfuse`` SDK is importable.

Every send is best-effort and any exception is logged-and-swallowed. A Generation
needs model + usage + prompt/response together, but those arrive as two separate
calls (tokens, text) a few ms apart; the emitter buffers by
:func:`..langfuse_mapping.pair_key` and emits once both halves are in (or at
``flush_session`` for whichever half never paired).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session.session_paths import (
    decision_trace_path,
    forge_steps_path,
    gemm_tuning_steps_path,
    recipe_snapshot_audit_jsonl,
    specialist_intel_path,
    trace_dir,
    trace_ext_dir,
)
from . import langfuse_mapping as lfmap
from .trace_env import (
    ENV_LANGFUSE_HOST,
    ENV_LANGFUSE_PUBLIC_KEY,
    ENV_LANGFUSE_SECRET_KEY,
    apply_flush_defaults,
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
    """Whether the optional ``langfuse`` SDK can be imported (no side effects).

    Returns:
        True when the ``langfuse`` SDK can be located, False otherwise.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("langfuse") is not None
    except Exception:  # noqa: BLE001
        return False


def _to_ns(dt: Any) -> int | None:
    """Datetime -> integer nanoseconds since epoch (langfuse v4 ``end_time``).

    v4's OTEL-based SDK wants integer ns for ``end_time``; v2/v3 accepted a
    ``datetime``. Returns None for a None input; best-effort.

    Args:
        dt: the datetime to convert (any other type yields ``None``).

    Returns:
        Integer nanoseconds since the epoch, or ``None`` for a ``None`` /
        non-datetime / unparseable input.
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

    Tries the caller's kwargs first and, if the SDK rejects ``start_time`` with a
    TypeError (v4 removed it), retries without it — keeping backdated timestamps
    where supported and degrading to "start = now" only where required.

    Args:
        parent: the parent observation (or client) to create the child on.
        **kwargs: the keyword arguments forwarded to ``start_observation``;
            ``start_time`` is dropped on a retry if the SDK rejects it.

    Returns:
        The newly started observation.
    """
    try:
        return parent.start_observation(**kwargs)
    except TypeError:
        kwargs.pop("start_time", None)
        return parent.start_observation(**kwargs)


def _end_time_wants_int(obs: Any) -> bool:
    """Whether this SDK's ``end(end_time=...)`` wants integer ns (v4) vs a
    datetime (v2/v3), decided by inspecting the parameter annotation.

    The type must be right on the first call: v4's ``end(datetime)`` raises only
    after already ending the span, so a try-then-retry pattern double-ends it.
    Falls back to datetime (False) when the annotation can't be read.

    Args:
        obs: the observation whose ``end`` signature is inspected.

    Returns:
        True when the SDK's ``end(end_time=...)`` wants integer ns (v4), False
        when it wants a datetime (v2/v3) or the annotation can't be read.
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

    Args:
        obs: the observation to end (``None`` is a no-op).
        end_dt: the end time as a datetime; ``None`` ends with no explicit time.
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

    OTEL span attributes accept only str/bool/int/float. Skip ``None``, pass
    scalars through, and JSON-stringify everything else so it still lands as text.

    Args:
        v: the metadata value to coerce.

    Returns:
        The value unchanged when it is a str/bool/int/float, a JSON string for
        any other non-``None`` value, or ``None`` to skip ``None`` inputs.
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

    Uses ``span.update_trace(...)`` (v2/v3); when absent (v4) falls back to
    writing the v4 OTEL trace attributes directly on the underlying span.
    Best-effort: a missing API just means the label isn't set, never a raise.

    Args:
        span: the span/observation whose trace-level attributes are stamped.
        name: optional trace name.
        session_id: optional session id to group the trace.
        metadata: optional trace-level metadata mapping.
    """
    try:
        span.update_trace(name=name, session_id=session_id, metadata=metadata)
        return
    except AttributeError:
        pass  # v4: no update_trace
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
                continue  # OTEL rejects None
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
        The dict records; missing files, unreadable files, and malformed or
        non-object lines are skipped to an empty/partial list.
    """
    from hyperloom.common.jsonio import read_jsonl

    return read_jsonl(path, require_dict=True, skip_malformed=True, skip_non_dict=True)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed object, or ``{}`` when the file is missing, unreadable, or
        not a JSON object.
    """
    from hyperloom.common.jsonio import read_json

    return read_json(path, default={}, require_dict=True)


class LangfuseEmitter:
    """Per-session live emitter. No-op unless all three gates pass.

    One instance is created lazily per ``session_dir`` via
    :func:`get_emitter`. Thread-safe: the trace appenders may be called from
    the Coordinator loop and from worker threads, so the pairing buffer is
    guarded by a lock.
    """

    def __init__(self, session_dir: Path) -> None:
        """Initialize the per-session emitter.

        Resolves the manifest and correlation labels unconditionally so the
        receipt always reports the right trace and correlation ids.

        Args:
            session_dir: Session directory whose traces are emitted.
        """
        self.session_dir = Path(session_dir)
        self._lock = threading.Lock()
        # pair_key -> partial generation parts ({"llm": row} / {"conv": row}).
        self._pending: dict[tuple, dict[str, dict[str, Any]]] = {}
        self._client: Any = None
        # Manifest + correlation resolved unconditionally so the receipt always
        # reports the right ids. Push / backfill / claw upload correlate on
        # claw_session_id (fallback internal session id).
        self._manifest: dict[str, Any] = _load_json(_manifest_path(self.session_dir))
        self._session_label: str | None = lfmap.langfuse_session_id(
            self._manifest,
            self.session_dir.name,
        )
        self._trace_id: str | None = lfmap.derive_trace_id(
            lfmap.correlation_seed(self._manifest, self.session_dir.name),
        )
        # Span hierarchy caches (lazy): root; one per phase; one per (phase, agent).
        self._root_span: Any = None
        self._phase_spans: dict[str, Any] = {}
        self._agent_spans: dict[tuple[str, str], Any] = {}
        self._trace_attrs_set = False
        # Receipt counters (for the session_breakdown ``langfuse`` section).
        self._disabled_reason: str | None = None
        self._counts: dict[str, int] = {
            "generations_sent": 0,  # Generations successfully started
            "generations_paired": 0,  # of which had both token + text halves
            "generations_text_only": 0,
            "generations_token_only": 0,
            "session_start_recorded": 0,  # 1 once the startup marker was sent
            "status_updates_sent": 0,  # live state.json status snapshots mirrored
            "scores_sent": 0,  # decision Scores created (span + trace)
            "spans_opened": 0,  # phase + agent spans created
            "ext_shards_read": 0,  # out-of-process ext/*.jsonl files swept
            "breakdown_recorded": 0,  # 1 once the full SBD JSON was attached
            "kb_spans_sent": 0,  # KB trace spans (assess/priors/recipe)
            "recipe_audit_read": 0,  # recipe_snapshot/.audit.jsonl rows swept
            "specialist_intel_read": 0,  # specialist_intel.jsonl rows swept
            "forge_steps_read": 0,  # forge_steps.jsonl rows swept
            "gemm_tuning_read": 0,  # gemm_tuning.jsonl rows swept
            "errors": 0,  # swallowed send failures
        }
        self._flushed = False
        # Live-status mirror throttle: last pushed signature + monotonic ts, so a
        # snapshot is sent only on-change or after a slow refresh interval.
        self._last_status_sig: tuple | None = None
        self._last_status_ts: float = 0.0
        self._enabled = self._init_client()

    # -- gating / client setup ------------------------------------------
    def _init_client(self) -> bool:
        """Resolve the three gates and build the SDK client; False -> no-op.

        Records ``_disabled_reason`` (``disabled`` / ``no_credentials`` /
        ``sdk_missing`` / ``init_failed``) so the receipt can explain *why*
        nothing was pushed. Correlation is already resolved in ``__init__``.

        Returns:
            bool: True when all three gates pass and the SDK client was built;
                False (a no-op emitter) otherwise.
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
                type(exc).__name__,
                exc,
            )
            return False
        try:
            # Tighten the SDK auto-flush cadence before the singleton is built so
            # a session killed early still lands its latest observations.
            apply_flush_defaults()
            creds = langfuse_credentials()
            self._client = get_client()
            log.info(
                "langfuse: live push enabled (host=%s, session=%s, trace_id=%s)",
                creds.get("LANGFUSE_HOST"),
                self._session_label,
                self._trace_id,
            )
            return True
        except Exception:  # noqa: BLE001
            self._disabled_reason = "init_failed"
            log.warning("langfuse: client init failed; live push disabled.", exc_info=True)
            return False

    @property
    def enabled(self) -> bool:
        """Whether live push to Langfuse is enabled for this session.

        Returns:
            bool: True when live push is enabled for this session.
        """
        return self._enabled

    # -- span hierarchy (trace -> phase -> agent -> generation) ---------
    def _trace_name(self) -> str:
        """Return the human-readable trace name.

        Returns:
            The model name, else the session label, else ``"hyperloom"``.
        """
        return str(self._manifest.get("model_name") or self._session_label or "hyperloom")

    def _ensure_root(self, start: Any) -> Any:
        """Lazily open the root span and stamp trace-level attrs once.

        Args:
            start: the start time for the root span.

        Returns:
            The cached or newly opened root span.
        """
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
                name=f"phase:{phase}",
                as_type="span",
                start_time=start,
                metadata={"phase": phase},
            )
            self._phase_spans[phase] = span
            self._counts["spans_opened"] += 1
        return span

    def _ensure_agent_span(self, phase: str, agent: str, start: Any) -> Any:
        """Get-or-create the per-(phase, agent) span. This is the 'which agent
        did what' layer; Generations and decision Scores attach here.

        Args:
            phase: Phase name.
            agent: Agent name.
            start: Span start time.

        Returns:
            The cached or newly opened (phase, agent) span.
        """
        key = (phase, agent)
        span = self._agent_spans.get(key)
        if span is None:
            phase_span = self._ensure_phase_span(phase, start)
            span = _start_obs(
                phase_span,
                name=f"agent:{agent}",
                as_type="span",
                start_time=start,
                metadata={"phase": phase, "agent": agent},
            )
            self._agent_spans[key] = span
            self._counts["spans_opened"] += 1
        return span

    # -- live ingest ----------------------------------------------------
    def record_llm_call(self, row: dict[str, Any]) -> None:
        """Buffer a token row; emit the Generation if its text half is in.

        Args:
            row: the token (``llm``) row to buffer.
        """
        if not self._enabled:
            return
        try:
            self._buffer(row, half="llm")
        except Exception:  # noqa: BLE001 — trace must never break the loop
            log.debug("langfuse: record_llm_call failed", exc_info=True)

    def record_conversation(self, row: dict[str, Any]) -> None:
        """Buffer a conversation row; emit the Generation if its tokens are in.

        Args:
            row: the conversation (``conv``) row to buffer.
        """
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

    def record_kb_span(
        self,
        *,
        name: str,
        agent: str,
        output: Any,
        phase: str = lfmap.UNPHASED,
        metadata: dict[str, Any] | None = None,
        ts: str | None = None,
    ) -> None:
        """Emit one non-LLM KB trace as a span nested under its agent span.

        Used for the KB integration trace so KB-usage evidence lands on the same
        trace as the LLM generations. The full trace dict goes in ``output``; a
        scalar summary goes in ``metadata`` for filtering. Best-effort and a no-op
        unless live push is enabled.

        Args:
            name (str): Span name (e.g. ``"kb_assess:iter_3"``).
            agent (str): Owning agent (``"critic"`` / ``"recipe_kb"``).
            output (Any): The trace payload attached as the span output.
            phase (str): Phase bucket; defaults to ``(unphased)``.
            metadata (dict[str, Any] | None): Scalar summary for filtering.
            ts (str | None): ISO timestamp for span start, if known.
        """
        if not self._enabled:
            return
        try:
            start = lfmap.parse_ts(ts)
            parent = self._ensure_agent_span(phase, agent, start)
            obs = _start_obs(
                parent,
                name=name,
                as_type="span",
                start_time=start,
                input=None,
                output=output,
                metadata=metadata or {},
            )
            _end_obs(obs, start)
            self._counts["kb_spans_sent"] += 1
        except Exception:  # noqa: BLE001 — trace must never break the loop
            self._counts["errors"] += 1
            log.debug("langfuse: record_kb_span failed", exc_info=True)

    def _emit_generation(
        self,
        *,
        token_row: dict[str, Any] | None,
        conv_row: dict[str, Any] | None,
    ) -> None:
        """Emit one Generation, nested under its phase -> agent span.

        Args:
            token_row: the token-half row, or ``None`` when only text is in.
            conv_row: the conversation-half row, or ``None`` when only tokens
                are in.
        """
        base = token_row or conv_row or {}
        phase = lfmap.phase_of(base)
        agent = lfmap.agent_of(base)
        # ``ts`` approximates the call END. With a measured ``latency_ms`` the
        # start is backdated to ``ts - latency`` so the leaf shows a real
        # duration; otherwise start == end == ts.
        end = lfmap.parse_ts(base.get("ts"))
        start = lfmap.generation_start(end, (token_row or {}).get("latency_ms"))
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
            _end_obs(gen, end)
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
        """Emit leftover halves + audit spans + decision Scores, then flush.

        Run once at session end. Safe to call when disabled (no-op) and
        idempotent: a second call only re-writes the receipt, avoiding duplicate
        audit spans / Scores in Langfuse.
        """
        if not self._enabled:
            # Still drop a receipt so the breakdown can report why nothing was pushed.
            self._write_receipt()
            return
        if self._flushed:
            log.debug("langfuse: flush_session already ran; skipping re-emit")
            self._write_receipt()
            return
        try:
            self._flush_pending_halves()
            self._flush_ext_shards()
            self._flush_recipe_kb_audit()
            self._flush_specialist_intel()
            self._flush_forge_steps()
            self._flush_gemm_tuning()
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

    def record_session_start(self) -> None:
        """Emit a one-shot ``session_start`` marker the moment a session begins.

        Attached directly to the session's ``trace_id`` so a run aborted in
        pre-flight still leaves a Langfuse trace tying the session dir to its
        ``code_revision`` and dependency commits. Idempotent (cross-process via
        the persisted receipt) and best-effort.
        """
        if not self._enabled:
            return
        if self._counts.get("session_start_recorded"):
            return
        # Cross-process guard: a prior process may have already marked start.
        persisted = read_receipt(self.session_dir) or {}
        if (persisted.get("counts") or {}).get("session_start_recorded"):
            self._counts["session_start_recorded"] = 1
            return
        import os

        payload = lfmap.session_start_payload(
            self._manifest,
            user_data_path=(os.environ.get("USER_DATA_PATH") or "").strip() or None,
            env=os.environ,
        )
        try:
            obs = _start_obs(
                self._client,
                name="session_start",
                as_type="span",
                trace_context={"trace_id": self._trace_id},
                input=None,
                output=payload,
                metadata={
                    "claw_session_id": payload.get("claw_session_id"),
                    "sandbox_user_id": payload.get("sandbox_user_id"),
                    "code_revision": payload.get("code_revision"),
                    "session_dir": payload.get("session_dir"),
                    "user_data_path": payload.get("user_data_path"),
                    "host": payload.get("host"),
                    "image": payload.get("image"),
                },
            )
            # Stamp trace name/session_id so the trace is grouped from the first observation.
            _set_trace_attrs(
                obs,
                name=self._trace_name(),
                session_id=self._session_label,
                metadata=lfmap.trace_metadata(self._manifest),
            )
            _end_obs(obs, None)
            self._counts["session_start_recorded"] = 1
        except Exception:  # noqa: BLE001
            self._counts["errors"] += 1
            log.debug("langfuse: record_session_start failed", exc_info=True)
        finally:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                self._counts["errors"] += 1
                log.debug("langfuse: flush after session_start failed", exc_info=True)
            # Persist the flag so a later process skips re-emitting.
            self._write_receipt()

    def record_session_breakdown(self, breakdown: dict[str, Any]) -> None:
        """Attach the complete ``session_breakdown.json`` document to the trace.

        Emitted as one ``session_breakdown`` observation attached to this
        session's ``trace_id`` so it lands on the same trace even when called
        after :meth:`flush_session` closed the live spans. Idempotent and
        best-effort: any send failure is swallowed.

        Args:
            breakdown: the complete ``session_breakdown.json`` document to
                attach; a non-dict or empty value is a no-op.
        """
        if not self._enabled or not isinstance(breakdown, dict) or not breakdown:
            return
        if self._counts.get("breakdown_recorded"):
            return
        # Cross-process guard via the persisted receipt (the only shared state).
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
            # Stamp trace name/session_id so a breakdown-only session is still grouped.
            _set_trace_attrs(
                obs,
                name=self._trace_name(),
                session_id=self._session_label,
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
            # Persist the flag so a later process skips re-attaching the document.
            self._write_receipt()

    def record_status(
        self,
        status: dict[str, Any],
        *,
        min_refresh_sec: float = 300.0,
    ) -> None:
        """Mirror a live ``state.json`` status snapshot onto the session's trace.

        Two effects, both keyed to this session's ``trace_id``: trace-level
        **metadata** is upserted from ``status`` (always the current snapshot),
        and a lightweight ``session_status`` **observation** is appended so the
        status timeline is queryable.

        Throttled to avoid flooding the trace: a snapshot is sent only when its
        signature changed or after ``min_refresh_sec`` elapsed. Never flushes the
        client; best-effort and a no-op unless live push is enabled.

        Args:
            status (dict[str, Any]): Flat scalar status summary (str/bool/int/
                float values); non-scalars are coerced by ``_otel_attr_value``.
            min_refresh_sec (float): Minimum seconds between unchanged pushes.
        """
        if not self._enabled:
            return
        if not isinstance(status, dict) or not status:
            return
        try:
            import time

            sig = tuple(sorted((str(k), status[k]) for k in status))
            now = time.monotonic()
            if (
                self._last_status_sig is not None
                and sig == self._last_status_sig
                and (now - self._last_status_ts) < min_refresh_sec
            ):
                return
            obs = _start_obs(
                self._client,
                name="session_status",
                as_type="span",
                trace_context={"trace_id": self._trace_id},
                input=None,
                output=status,
                metadata=status,
            )
            # Upsert the trace-level snapshot so consumers can read status off the trace row.
            _set_trace_attrs(
                obs,
                name=self._trace_name(),
                session_id=self._session_label,
                metadata=status,
            )
            _end_obs(obs, None)
            self._counts["status_updates_sent"] += 1
            self._last_status_sig = sig
            self._last_status_ts = now
        except Exception:  # noqa: BLE001 — status mirror must never break the loop
            self._counts["errors"] += 1
            log.debug("langfuse: record_status failed", exc_info=True)

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

        Children (geak / forge / robustness / specialist subprocess) never
        connect to Langfuse; their tokens land in ``ext/<component>-<pid>.jsonl``.
        Emitted as text-less Generations so the trace still accounts their spend.
        """
        ext_dir = trace_ext_dir(self.session_dir)
        if not ext_dir.is_dir():
            return
        for shard in sorted(ext_dir.glob("*.jsonl")):
            self._counts["ext_shards_read"] += 1
            for row in _load_jsonl(shard):
                self._emit_generation(token_row=row, conv_row=None)

    def _flush_recipe_kb_audit(self) -> None:
        """Backfill recipe-snapshot / gbrain remote reads from the audit log.

        The recipe KB dispatcher appends one row per remote read to
        ``runtime/recipe_snapshot/.audit.jsonl``. Each row becomes a
        ``kb:recipe_snapshot:<method>`` span under the ``recipe_kb`` agent. Read
        out-of-band at session end; idempotent via the ``flush_session`` guard.
        """
        rows = _load_jsonl(recipe_snapshot_audit_jsonl(self.session_dir))
        for row in rows:
            self._counts["recipe_audit_read"] += 1
            method = str(row.get("method") or "read")
            self.record_kb_span(
                name=f"kb:recipe_snapshot:{method}",
                agent="recipe_kb",
                output=row,
                metadata={
                    "kind": "recipe_snapshot",
                    "method": method,
                    "remote": row.get("remote"),
                    "resolution": row.get("resolution"),
                    "hit": bool(row.get("hit")),
                },
                ts=row.get("ts"),
            )

    def _flush_specialist_intel(self) -> None:
        """Backfill specialist intel/tool calls as per-call ``intel:<tool>`` spans.

        The specialist runner appends one row per recovered tool call to
        ``reports/trace/specialist_intel.jsonl``. Each row becomes an
        ``intel:<tool>`` span under the ``specialist`` agent so the trace shows
        what a specialist actually read. Read out-of-band at session end;
        idempotent via the ``flush_session`` guard.
        """
        rows = _load_jsonl(specialist_intel_path(self.session_dir))
        for row in rows:
            self._counts["specialist_intel_read"] += 1
            tool = str(row.get("tool") or "tool")
            self.record_kb_span(
                name=f"intel:{tool}",
                agent="specialist",
                output=row,
                metadata={
                    "kind": "specialist_intel",
                    "tool": tool,
                    "task_id": row.get("task_id"),
                    "turn": row.get("turn"),
                    "query": row.get("query"),
                },
                ts=row.get("ts"),
            )

    def _flush_forge_steps(self) -> None:
        """Backfill the Kernel-Forge loop's key steps as ``forge:*`` spans.

        ``kernel_request_handlers`` records each forge attempt's per-iteration
        steps and a run summary to ``reports/trace/forge_steps.jsonl``. Each row
        becomes a ``forge:iter:<n>`` (or ``forge:summary``) span under the
        ``forge`` agent. Read out-of-band at session end; idempotent via the
        ``flush_session`` guard.
        """
        for row in _load_jsonl(forge_steps_path(self.session_dir)):
            self._counts["forge_steps_read"] += 1
            kind = str(row.get("kind") or "iteration")
            if kind == "summary":
                name = "forge:summary"
                metadata = {
                    "kind": "forge_summary",
                    "kernel_id": row.get("kernel_id"),
                    "iterations": row.get("iterations"),
                    "kept": row.get("kept"),
                    "speedup": row.get("speedup"),
                    "improved": row.get("improved"),
                    "termination_reason": row.get("termination_reason"),
                }
            else:
                name = f"forge:iter:{row.get('iteration')}"
                metadata = {
                    "kind": "forge_iteration",
                    "kernel_id": row.get("kernel_id"),
                    "iteration": row.get("iteration"),
                    "decision": row.get("decision"),
                    "wall_ms": row.get("wall_ms"),
                    "snr_db": row.get("snr_db"),
                    "validation_passed": row.get("validation_passed"),
                    "pmc_diagnosis": row.get("pmc_diagnosis"),
                }
            self.record_kb_span(
                name=name, agent="forge", output=row,
                metadata=metadata, ts=row.get("ts"),
            )

    def _flush_gemm_tuning(self) -> None:
        """Backfill each deterministic GEMM-tuning run as a ``gemm_tuning:*`` span.

        ``run_gemm_tuning_handler`` appends one row per run to
        ``reports/trace/gemm_tuning.jsonl``. Each row becomes a
        ``gemm_tuning:<engine>`` span under the ``gemm_tuning`` agent so a trace
        attributes the tuner as its own source. Read out-of-band at session end;
        idempotent via the ``flush_session`` guard.
        """
        for row in _load_jsonl(gemm_tuning_steps_path(self.session_dir)):
            self._counts["gemm_tuning_read"] += 1
            engine = str(row.get("engine") or row.get("backend") or "unknown")
            self.record_kb_span(
                name=f"gemm_tuning:{engine}",
                agent="gemm_tuning",
                output=row,
                metadata={
                    "kind": "gemm_tuning",
                    "engine": engine,
                    "backend": row.get("backend"),
                    "decision": row.get("decision"),
                    "micro_decision": row.get("micro_decision"),
                    "best_speedup": row.get("best_speedup"),
                    "precision": row.get("precision"),
                    "framework": row.get("framework"),
                    "tuned_file": row.get("tuned_file"),
                },
                ts=row.get("ts"),
            )

    def _flush_decision_scores(self) -> None:
        """Convert each decision_trace row into Langfuse Score(s).

        Each score targets the agent span that owns the decision (phase,
        component from the decision metadata). Falls back to a trace-level score
        when no matching span exists.
        """
        for drow in _load_jsonl(decision_trace_path(self.session_dir)):
            scores = lfmap.decision_to_scores(drow)
            if not scores:
                continue
            meta0 = scores[0].get("metadata") or {}
            phase = str(meta0.get("phase") or lfmap.UNPHASED)
            agent = lfmap.span_agent_for(str(meta0.get("component") or ""))
            # Per-decision span carrying ``operation_kind`` so the trace can be
            # filtered by step. Scores attach here when it opens, else to the agent span.
            step_span = self._open_decision_span(drow, phase, agent)
            for score in scores:
                self._create_score(
                    score,
                    phase=phase,
                    agent=agent,
                    span=step_span,
                )
            if step_span is not None:
                self._safe_end(step_span)

    def _open_decision_span(
        self,
        drow: dict[str, Any],
        phase: str,
        agent: str,
    ) -> Any:
        """Open an ``optimization_step:<operation_kind>`` span for one decision.

        Parented to the owning agent span (then phase span, then root). Carries
        operation_kind + proposer + effect in metadata for step filtering.
        Best-effort; returns ``None`` when no parent is open or the SDK rejects it.

        Args:
            drow: the decision_trace row carrying the ``decision`` payload.
            phase: the phase that owns the decision (used to locate the parent).
            agent: the agent that owns the decision (used to locate the parent).

        Returns:
            The opened ``optimization_step`` span, or ``None`` when no parent is
            open or the SDK rejects the call.
        """
        parent = self._agent_spans.get((phase, agent)) or self._phase_spans.get(phase) or self._root_span
        if parent is None:
            return None
        dec = drow.get("decision") or {}
        op_kind = str(dec.get("operation_kind") or "decision")
        # Per-decision token cost so a trace can rank decisions by cost.
        tokens = drow.get("tokens") if isinstance(drow.get("tokens"), dict) else {}
        cost_total = None
        try:
            cost_total = (
                int(tokens.get("total_in", 0) or 0)
                + int(tokens.get("total_out", 0) or 0)
                + int(tokens.get("total_cache", 0) or 0)
            ) or None
        except (TypeError, ValueError):
            cost_total = None
        md = {
            "operation_kind": op_kind,
            "proposer": dec.get("component"),
            "provenance": dec.get("provenance"),
            "scope": dec.get("scope"),
            "change": dec.get("change"),
            "outcome": dec.get("outcome"),
            "gain_pct": dec.get("gain_pct"),
            "variant_name": dec.get("variant_name"),
            "fingerprint": dec.get("fingerprint"),
            "task_id": dec.get("task_id"),
            "phase": phase,
            "tick": drow.get("tick"),
            "metrics": dec.get("metrics"),
            "proposal_scores": dec.get("proposal_scores"),
            "cost_tokens_total": cost_total,
            "cost_calls": (tokens.get("calls") or None),
        }
        md = {k: v for k, v in md.items() if v is not None}
        try:
            return _start_obs(
                parent,
                name=f"optimization_step:{op_kind}",
                as_type="span",
                metadata=md,
            )
        except Exception:  # noqa: BLE001
            log.debug("langfuse: open decision span failed", exc_info=True)
            return None

    def _create_score(
        self,
        score: dict[str, Any],
        *,
        phase: str,
        agent: str,
        span: Any = None,
    ) -> None:
        """Attach a Langfuse Score to a step span / agent span / the trace.

        Args:
            score: Score payload (``name``, ``value``, ``data_type``, ...).
            phase: Phase that owns the decision, used to locate the span.
            agent: Agent/component that owns the decision.
            span: Optional pre-opened step span; preferred over the agent span.
        """
        if span is None:
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
                "langfuse: create_score failed for %s",
                score.get("name"),
                exc_info=True,
            )

    # -- receipt (session_breakdown ``langfuse`` section) ---------------
    def receipt(self) -> dict[str, Any]:
        """A redacted record of whether/where/how much was pushed.

        Shape mirrors the ``langfuse`` section of ``session_breakdown.json``.
        Credentials are never included verbatim (only the host URL and
        key-presence booleans). ``counts_final`` is True once
        :meth:`flush_session` has run, else it reports in-process running totals.

        Returns:
            dict[str, Any]: a redacted receipt dict (enabled flag, disabled
                reason, config, trace/session ids, correlation key, and push
                counts) mirroring the ``langfuse`` breakdown section.
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
                "claw_session_id" if str(self._manifest.get("claw_session_id") or "").strip() else "internal_session_id"
            ),
            "counts": dict(self._counts),
            "counts_final": self._flushed,
        }

    def _write_receipt(self) -> None:
        """Persist :meth:`receipt` to ``reports/trace/langfuse_receipt.json``.

        Best-effort: a failed receipt write must never break shutdown. The
        breakdown collector prefers this file over a live read of the singleton.
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


# Process-wide singleton registry (one emitter per session_dir).
_REGISTRY: dict[str, LangfuseEmitter] = {}
_REGISTRY_LOCK = threading.Lock()


def get_emitter(session_dir: Path) -> LangfuseEmitter:
    """Return the per-session emitter, building it once (cached by session).

    Args:
        session_dir: Session directory the emitter is keyed by.

    Returns:
        LangfuseEmitter: the cached or newly built emitter for the session.
    """
    key = str(Path(session_dir).resolve())
    with _REGISTRY_LOCK:
        emitter = _REGISTRY.get(key)
        if emitter is None:
            emitter = LangfuseEmitter(Path(session_dir))
            _REGISTRY[key] = emitter
        return emitter


def record_session_start(session_dir: Path) -> None:
    """Module-level convenience: emit the startup marker for ``session_dir``.

    Call once right after ``manifest.json`` is written so the session's Langfuse
    trace exists from the start. No-op when disabled; best-effort.

    Args:
        session_dir: Session directory whose startup marker is emitted.
    """
    get_emitter(session_dir).record_session_start()


def flush_session(session_dir: Path) -> None:
    """Module-level convenience: flush the emitter for ``session_dir``.

    Args:
        session_dir: Session directory whose emitter is flushed.
    """
    get_emitter(session_dir).flush_session()


def _read_breakdown_file(session_dir: Path) -> dict[str, Any]:
    """Load the written ``session_breakdown.json`` for ``session_dir`` ({} if absent).

    Args:
        session_dir: Session directory whose breakdown file is read.

    Returns:
        dict[str, Any]: the parsed breakdown object, or ``{}`` when the file is
            missing, unreadable, or not a JSON object.
    """
    from hyperloom.common.jsonio import read_json
    from hyperloom.inference_optimizer.breakdown import BREAKDOWN_FILENAME

    return read_json(Path(session_dir) / BREAKDOWN_FILENAME, default={}, require_dict=True)


def record_session_breakdown(
    session_dir: Path,
    breakdown: dict[str, Any] | None = None,
) -> None:
    """Attach the final ``session_breakdown.json`` to the session's trace.

    Call after the breakdown is written and the langfuse section patched. Reads
    the file from disk when ``breakdown`` is not supplied. No-op when disabled;
    best-effort.

    Args:
        session_dir: Session directory whose trace the breakdown attaches to.
        breakdown: the breakdown document; read from disk when ``None``.
    """
    if breakdown is None:
        breakdown = _read_breakdown_file(Path(session_dir))
    get_emitter(session_dir).record_session_breakdown(breakdown)


def record_status(
    session_dir: Path,
    status: dict[str, Any],
    *,
    min_refresh_sec: float = 300.0,
) -> None:
    """Module-level convenience: mirror a status snapshot for ``session_dir``.

    Reuses the per-session emitter singleton so its throttle state persists. No-op
    when disabled; best-effort.

    Args:
        session_dir: Session directory whose trace the status attaches to.
        status: Flat scalar status summary to mirror.
        min_refresh_sec: Minimum seconds between unchanged pushes.
    """
    get_emitter(session_dir).record_status(status, min_refresh_sec=min_refresh_sec)


def read_receipt(session_dir: Path) -> dict[str, Any] | None:
    """Read the persisted ``langfuse_receipt.json`` for ``session_dir``.

    Returns the post-flush receipt dict (preferred by the breakdown collector
    since its counts are final) or ``None`` if no receipt was written.

    Args:
        session_dir: Session directory whose persisted receipt is read.

    Returns:
        dict[str, Any] | None: the post-flush receipt dict, or ``None`` when no
            receipt was written or it is unreadable.
    """
    from hyperloom.common.jsonio import read_json

    return read_json(_receipt_path(session_dir), default=None, require_dict=True)


__all__ = [
    "LangfuseEmitter",
    "flush_session",
    "get_emitter",
    "read_receipt",
    "record_session_breakdown",
    "record_status",
]
