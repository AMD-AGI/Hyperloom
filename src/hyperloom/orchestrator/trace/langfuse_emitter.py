# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Live Langfuse push for the trace subsystem (opt-in, best-effort)."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json
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

_STATUS_CLOCK_BUCKET_SEC = 60
_STATUS_CLOCK_KEYS = frozenset({"explore_elapsed_s", "session_elapsed_s"})
_STATUS_CLOCK_DERIVED_KEYS = frozenset({"explore_ratio"})


def _status_signature(status: dict[str, Any]) -> tuple:
    """Return a throttle signature with volatile runtime clocks minute-bucketed."""
    items: list[tuple[str, Any]] = []
    for raw_key, raw_value in status.items():
        key = str(raw_key)
        if key in _STATUS_CLOCK_DERIVED_KEYS:
            # Derived from the two elapsed clocks; it will be refreshed whenever either clock crosses a bucket or
            # another semantic field changes.
            continue
        value = raw_value
        if key in _STATUS_CLOCK_KEYS:
            try:
                value = int(float(raw_value) // _STATUS_CLOCK_BUCKET_SEC)
            except (TypeError, ValueError):
                pass
        items.append((key, value))
    return tuple(sorted(items))


def _manifest_path(session_dir: Path) -> Path:
    """Return the path to a session's ``manifest.json``."""
    return session_dir / "manifest.json"


#: Session-end reconcile steps, in run order. The names are persisted in the
#: receipt (``flush_steps_done``), so a restart resumes instead of replaying.
_FLUSH_STEP_NAMES: tuple[str, ...] = (
    "pending_halves",
    "ext_shards",
    "recipe_kb_audit",
    "specialist_intel",
    "forge_steps",
    "gemm_tuning",
    "decision_scores",
    "close_spans",
    "client_flush",
)


def _persisted_ext_cursors(session_dir: Path) -> dict[str, int]:
    """Return how far each ext/ shard was drained by a previous process."""
    persisted = (read_receipt(session_dir) or {}).get("ext_rows_sent")
    if not isinstance(persisted, dict):
        return {}
    cursors: dict[str, int] = {}
    for name, count in persisted.items():
        try:
            cursors[str(name)] = max(0, int(count))
        except (TypeError, ValueError):
            continue
    return cursors


def _receipt_path(session_dir: Path) -> Path:
    """Return the path to a session's Langfuse receipt file."""
    return trace_dir(session_dir) / "langfuse_receipt.json"


def _sdk_available() -> bool:
    """Whether the optional ``langfuse`` SDK can be imported (no side effects)."""
    import importlib.util

    try:
        return importlib.util.find_spec("langfuse") is not None
    except Exception:  # noqa: BLE001
        return False


def _to_ns(dt: Any) -> int | None:
    """Datetime -> integer nanoseconds since epoch (langfuse v4 ``end_time``)."""
    if dt is None:
        return None
    try:
        from datetime import datetime

        if isinstance(dt, datetime):
            return int(dt.timestamp() * 1_000_000_000)
    except Exception:  # noqa: BLE001
        return None
    return None


# Optional kwargs dropped, in order, when the installed SDK rejects them.
_OBS_KWARG_LADDER: tuple[tuple[str, ...], ...] = (
    (),
    ("start_time",),
    ("start_time", "level", "status_message"),
)


def _start_obs(parent: Any, **kwargs: Any) -> Any:
    """Create a child/root observation, tolerant of v2/v3 vs v4 signatures."""
    seen: set[frozenset[str]] = set()
    attempts: list[dict[str, Any]] = []
    for drop in _OBS_KWARG_LADDER:
        attempt = {k: v for k, v in kwargs.items() if k not in drop}
        signature = frozenset(attempt)
        if signature in seen:
            continue
        seen.add(signature)
        attempts.append(attempt)
    for attempt in attempts[:-1]:
        try:
            return parent.start_observation(**attempt)
        except TypeError:
            continue
    # The final rung is not guarded, so a still-rejected signature surfaces its own TypeError instead of one
    # synthesized from a saved exception.
    return parent.start_observation(**attempts[-1])


def _end_time_wants_int(obs: Any) -> bool:
    """Whether this SDK's ``end(end_time=...)`` wants integer ns (v4) vs a datetime (v2/v3), decided by inspecting the parameter annotation."""
    try:
        import inspect

        sig = inspect.signature(obs.end)
        ann = sig.parameters.get("end_time")
        ann_str = "" if ann is None else str(ann.annotation)
    except (TypeError, ValueError):
        return False
    return "int" in ann_str.lower()


def _end_obs(obs: Any, end_dt: Any) -> None:
    """End an observation, tolerant of v2/v3 (datetime) vs v4 (int ns)."""
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
    """Coerce a metadata value into an OTEL-acceptable attribute, or None to skip it."""
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
    """Stamp trace-level name/session_id/metadata, tolerant of v2/v3 vs v4."""
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
    """Load a JSONL file into a list of dict records."""
    from hyperloom.common.jsonio import read_jsonl

    return read_jsonl(path, require_dict=True, skip_malformed=True, skip_non_dict=True)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object file."""
    from hyperloom.common.jsonio import read_json

    return read_json(path, default={}, require_dict=True)


class LangfuseEmitter:
    """Per-session live emitter. No-op unless all three gates pass."""

    def __init__(self, session_dir: Path) -> None:
        """Initialize the per-session emitter."""
        self.session_dir = Path(session_dir)
        self._lock = threading.Lock()
        # pair_key -> partial generation parts ({"llm": row} / {"conv": row}).
        self._pending: dict[tuple, dict[str, dict[str, Any]]] = {}
        self._client: Any = None
        # Manifest + correlation resolved unconditionally so the receipt always reports the right ids.
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
            "generations_failed": 0,  # of which recorded a failed LLM call (level=ERROR)
            "session_start_recorded": 0,  # 1 once the startup marker was sent
            "status_updates_sent": 0,  # live state.json status snapshots mirrored
            "scores_sent": 0,  # decision Scores created (span + trace)
            "spans_opened": 0,  # phase + agent spans created
            "ext_shards_read": 0,  # out-of-process ext/*.jsonl files swept
            "breakdown_recorded": 0,  # 1 once the full SBD JSON was attached
            "kb_spans_sent": 0,  # non-LLM spans (recipe-KB / critic priors / specialist intel / forge / gemm)
            "recipe_audit_read": 0,  # recipe_snapshot/.audit.jsonl read rows swept
            "recipe_write_audit_read": 0,  # of which were recipe-KB writes
            "specialist_intel_read": 0,  # specialist_intel.jsonl rows swept
            "forge_steps_read": 0,  # forge_steps.jsonl rows swept
            "gemm_tuning_read": 0,  # gemm_tuning.jsonl rows swept
            "errors": 0,  # swallowed send failures
        }
        # Reconcile steps that already succeeded in *this* process, so a retry after a partial flush neither re-emits
        # them nor loses the ones still owed.
        self._flush_steps_done: set[str] = set()
        self._flushed = False
        # How many rows of each ext/ shard have been sent, restored from the receipt.
        self._ext_rows_sent: dict[str, int] = _persisted_ext_cursors(self.session_dir)
        # Live-status mirror throttle: last pushed signature + monotonic ts, so a snapshot is sent only on-change or
        # after a slow refresh interval.
        self._last_status_sig: tuple | None = None
        self._last_status_ts: float = 0.0
        self._enabled = self._init_client()

    # -- gating / client setup ------------------------------------------
    def _init_client(self) -> bool:
        """Resolve the three gates and build the SDK client; False -> no-op."""
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
            # Tighten the SDK auto-flush cadence before the singleton is built so a session killed early still lands
            # its latest observations.
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
        """Whether live push to Langfuse is enabled for this session."""
        return self._enabled

    # -- span hierarchy (trace -> phase -> agent -> generation) ---------
    def _trace_name(self) -> str:
        """Return the human-readable trace name."""
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
        """Get-or-create the span for a phase under the trace root."""
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
        """Get-or-create the per-(phase, agent) span."""
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
        """Buffer one half of a generation and emit once both halves arrive."""
        if half == "llm" and lfmap.generation_level(row) == lfmap.LEVEL_ERROR:
            if not self._emit_generation(token_row=row, conv_row=None):
                self._requeue_parts(lfmap.pair_key(row), {"llm": row})
            return
        key = lfmap.pair_key(row)
        emit_parts: dict[str, dict[str, Any]] | None = None
        with self._lock:
            parts = self._pending.setdefault(key, {})
            parts[half] = row
            if "llm" in parts and "conv" in parts:
                emit_parts = self._pending.pop(key)
        if emit_parts is not None and not self._emit_generation(
            token_row=emit_parts.get("llm"),
            conv_row=emit_parts.get("conv"),
        ):
            # The send failed and was swallowed; keep the halves so session-end reconcile can retry them rather than
            # losing the call.
            self._requeue_parts(key, emit_parts)

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
        """Emit one non-LLM KB trace as a span nested under its agent span."""
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
    ) -> bool:
        """Emit one Generation, nested under its phase -> agent span."""
        base = token_row or conv_row or {}
        phase = lfmap.phase_of(base)
        agent = lfmap.agent_of(base)
        # ``ts`` approximates the call END.
        end = lfmap.parse_ts(base.get("ts"))
        start = lfmap.generation_start(end, (token_row or {}).get("latency_ms"))
        has_text = conv_row is not None
        level = lfmap.generation_level(base)
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
                level=level,
                status_message=lfmap.generation_status_message(base),
            )
            _end_obs(gen, end)
            self._counts["generations_sent"] += 1
            if level == lfmap.LEVEL_ERROR:
                self._counts["generations_failed"] += 1
            if token_row is not None and conv_row is not None:
                self._counts["generations_paired"] += 1
            elif conv_row is not None:
                self._counts["generations_text_only"] += 1
            else:
                self._counts["generations_token_only"] += 1
            return True
        except Exception:  # noqa: BLE001
            self._counts["errors"] += 1
            log.debug("langfuse: emit generation failed", exc_info=True)
            return False

    # -- session-end reconcile ------------------------------------------
    def flush_session(self) -> None:
        """Emit leftover halves + audit spans + decision Scores, then flush."""
        if not self._enabled:
            # Still drop a receipt so the breakdown can report why nothing was pushed.
            self._write_receipt()
            return
        if self._flushed:
            log.debug("langfuse: flush_session already ran; skipping re-emit")
            self._write_receipt()
            return
        # ``client_flush`` is last and is a step like any other: everything before it only hands observations to the
        # SDK's buffer, so a failed final flush means nothing reached Langfuse and has to be retried.
        steps: dict[str, Any] = {
            "pending_halves": self._flush_pending_halves,
            "ext_shards": self._flush_ext_shards,
            "recipe_kb_audit": self._flush_recipe_kb_audit,
            "specialist_intel": self._flush_specialist_intel,
            "forge_steps": self._flush_forge_steps,
            "gemm_tuning": self._flush_gemm_tuning,
            "decision_scores": self._flush_decision_scores,
            "close_spans": self._close_spans,
            "client_flush": self._flush_client,
        }
        for name in _FLUSH_STEP_NAMES:
            if name in self._flush_steps_done:
                continue
            try:
                steps[name]()
            except Exception:  # noqa: BLE001 — one failed step must not skip the rest
                self._counts["errors"] += 1
                log.debug("langfuse: flush step %s failed", name, exc_info=True)
                continue
            self._flush_steps_done.add(name)
        self._flushed = self._flush_steps_done.issuperset(_FLUSH_STEP_NAMES)
        self._write_receipt()

    def _flush_client(self) -> None:
        """Hand the SDK's buffered observations to the network."""
        self._client.flush()

    def record_session_start(self) -> None:
        """Emit a one-shot ``session_start`` marker the moment a session begins."""
        if not self._enabled:
            return
        if self._counts.get("session_start_recorded"):
            return
        # Cross-process guard: a prior process may have already marked start.
        persisted = read_receipt(self.session_dir) or {}
        if (persisted.get("counts") or {}).get("session_start_recorded"):
            self._counts["session_start_recorded"] = 1
            return
        if not self._claim_one_shot("session_start"):
            self._counts["session_start_recorded"] = 1
            return

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
        """Attach the complete ``session_breakdown.json`` document to the trace."""
        if not self._enabled or not isinstance(breakdown, dict) or not breakdown:
            return
        if self._counts.get("breakdown_recorded"):
            return
        # Cross-process guard via the persisted receipt (the only shared state).
        persisted = read_receipt(self.session_dir) or {}
        if (persisted.get("counts") or {}).get("breakdown_recorded"):
            self._counts["breakdown_recorded"] = 1
            return
        if not self._claim_one_shot("session_breakdown"):
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
        """Mirror a live ``state.json`` status snapshot onto the session's trace."""
        if not self._enabled:
            return
        if not isinstance(status, dict) or not status:
            return
        try:
            import time

            sig = _status_signature(status)
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
        """End a span, swallowing any errors."""
        try:
            span.end()
        except Exception:  # noqa: BLE001
            log.debug("langfuse: span end failed", exc_info=True)

    def _flush_pending_halves(self) -> None:
        """Emit any buffered call that only ever got one half (token XOR text)."""
        with self._lock:
            leftovers = list(self._pending.items())
            self._pending.clear()
        requeued = 0
        for key, parts in leftovers:
            if self._emit_generation(
                token_row=parts.get("llm"),
                conv_row=parts.get("conv"),
            ):
                continue
            requeued += 1
            self._requeue_parts(key, parts)
        if requeued:
            raise RuntimeError(f"{requeued} buffered generation(s) could not be sent")

    def _requeue_parts(self, key: tuple, parts: dict[str, dict[str, Any]]) -> None:
        """Put unsent generation halves back on the pending map."""
        with self._lock:
            current = self._pending.setdefault(key, {})
            for half, row in parts.items():
                current.setdefault(half, row)

    def _flush_ext_shards(self) -> None:
        """Backfill out-of-process children's token rows from ext/*.jsonl."""
        ext_dir = trace_ext_dir(self.session_dir)
        if not ext_dir.is_dir():
            return
        unsent = 0
        for shard in sorted(ext_dir.glob("*.jsonl")):
            sent = self._ext_rows_sent.get(shard.name, 0)
            rows = _load_jsonl(shard)
            if sent == 0 and rows:
                self._counts["ext_shards_read"] += 1
            for index in range(sent, len(rows)):
                if not self._emit_generation(token_row=rows[index], conv_row=None):
                    unsent += 1
                    break
                self._ext_rows_sent[shard.name] = index + 1
        if unsent:
            raise RuntimeError(f"{unsent} ext-shard row(s) could not be sent")

    def _flush_recipe_kb_audit(self) -> None:
        """Backfill recipe-KB reads and writes from the audit log."""
        rows = _load_jsonl(recipe_snapshot_audit_jsonl(self.session_dir))
        for row in rows:
            self._counts["recipe_audit_read"] += 1
            if lfmap.recipe_audit_is_write(row):
                self._counts["recipe_write_audit_read"] += 1
                name, metadata = lfmap.recipe_write_span(row)
            else:
                name, metadata = lfmap.recipe_read_span(row)
            self.record_kb_span(
                name=name,
                agent="recipe_kb",
                output=row,
                metadata=metadata,
                ts=row.get("ts"),
            )

    def _flush_specialist_intel(self) -> None:
        """Backfill specialist intel/tool calls as per-call ``intel:<tool>`` spans."""
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
        """Backfill the Kernel-Forge loop's key steps as ``forge:*`` spans."""
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
                name=name,
                agent="forge",
                output=row,
                metadata=metadata,
                ts=row.get("ts"),
            )

    def _flush_gemm_tuning(self) -> None:
        """Backfill each deterministic GEMM-tuning run as a ``gemm_tuning:*`` span."""
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
        """Convert each decision_trace row into Langfuse Score(s)."""
        for drow in _load_jsonl(decision_trace_path(self.session_dir)):
            scores = lfmap.decision_to_scores(drow)
            if not scores:
                continue
            meta0 = scores[0].get("metadata") or {}
            phase = str(meta0.get("phase") or lfmap.UNPHASED)
            agent = lfmap.span_agent_for(str(meta0.get("component") or ""))
            # Per-decision span carrying ``operation_kind`` so the trace can be filtered by step.
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
        """Open an ``optimization_step:<operation_kind>`` span for one decision."""
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
        """Attach a Langfuse Score to a step span / agent span / the trace."""
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
        """A redacted record of whether/where/how much was pushed."""
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
            # Which reconcile steps have completed, so a receipt written after a partial flush says what is still owed
            # instead of reading as final.
            "flush_steps_done": sorted(self._flush_steps_done),
            "ext_rows_sent": dict(self._ext_rows_sent),
        }

    def _claim_one_shot(self, marker: str) -> bool:
        """Take the cross-process claim for a once-per-session push."""
        path = trace_dir(self.session_dir) / f".{marker}.claim"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            log.debug("langfuse: %s already claimed by another process", marker)
            return False
        except OSError:
            log.debug("langfuse: could not write the %s claim", marker, exc_info=True)
            return True
        try:
            os.write(fd, f"{os.getpid()}\n".encode())
        except OSError:
            pass
        finally:
            os.close(fd)
        return True

    def _write_receipt(self) -> None:
        """Persist :meth:`receipt` to ``reports/trace/langfuse_receipt.json``."""
        try:
            atomic_write_json(
                _receipt_path(self.session_dir),
                _stamp_receipt_hash(self.receipt()),
                indent=2,
                sort_keys=True,
                make_parents=True,
                fsync=True,
                fsync_dir=True,
            )
        except Exception:  # noqa: BLE001
            log.debug("langfuse: receipt write failed", exc_info=True)


# Process-wide singleton registry (one emitter per session_dir).
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


def record_session_start(session_dir: Path) -> None:
    """Module-level convenience: emit the startup marker for ``session_dir``."""
    get_emitter(session_dir).record_session_start()


def flush_session(session_dir: Path) -> None:
    """Module-level convenience: flush the emitter for ``session_dir``."""
    get_emitter(session_dir).flush_session()


def record_session_breakdown(
    session_dir: Path,
    breakdown: dict[str, Any] | None = None,
) -> None:
    """Attach the final ``session_breakdown.json`` to the session's trace."""
    if breakdown is None:
        from hyperloom.common.jsonio import read_json
        from hyperloom.inference_optimizer.breakdown import BREAKDOWN_FILENAME

        breakdown = read_json(Path(session_dir) / BREAKDOWN_FILENAME, default={}, require_dict=True)
    get_emitter(session_dir).record_session_breakdown(breakdown)


def record_status(
    session_dir: Path,
    status: dict[str, Any],
    *,
    min_refresh_sec: float = 300.0,
) -> None:
    """Module-level convenience: mirror a status snapshot for ``session_dir``."""
    get_emitter(session_dir).record_status(status, min_refresh_sec=min_refresh_sec)


#: Receipt key holding the SHA-256 of the receipt body (excluding itself).
_RECEIPT_HASH_KEY = "payload_sha256"


def _receipt_body_hash(payload: dict[str, Any]) -> str:
    """Return the SHA-256 of a receipt payload, excluding the hash field itself."""
    import hashlib
    import json

    body = {k: v for k, v in payload.items() if k != _RECEIPT_HASH_KEY}
    return hashlib.sha256(json.dumps(body, indent=2, sort_keys=True).encode("utf-8")).hexdigest()


def _stamp_receipt_hash(payload: dict[str, Any]) -> dict[str, Any]:
    """Return ``payload`` with its body hash stamped in."""
    payload[_RECEIPT_HASH_KEY] = _receipt_body_hash(payload)
    return payload


def read_receipt(session_dir: Path) -> dict[str, Any] | None:
    """Read the persisted ``langfuse_receipt.json`` for ``session_dir``."""
    from hyperloom.common.jsonio import read_json

    payload = read_json(_receipt_path(session_dir), default=None, require_dict=True)
    if payload is None:
        return None
    stamped = payload.get(_RECEIPT_HASH_KEY)
    if stamped is not None and stamped != _receipt_body_hash(payload):
        log.warning(
            "langfuse: ignoring receipt at %s — payload hash mismatch",
            _receipt_path(session_dir),
        )
        return None
    return payload


__all__ = [
    "LangfuseEmitter",
    "flush_session",
    "get_emitter",
    "read_receipt",
    "record_session_breakdown",
    "record_status",
]
