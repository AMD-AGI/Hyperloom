"""Author-time recording of the SBD v6 ``metadata`` section.

Every metadata fact is written the moment it is decided instead of being
re-derived when the breakdown is exported: session identity and the container
image when the manifest is stamped, the structural model summary when the
model's own config is parsed, the launch configuration and session lifecycle
on every state save, and the Langfuse entrypoint when the emitter settles.

All writes go through the recorder's ``metadata`` singleton and are
deep-merged, so each producer contributes only the keys it owns and a later
partial update never erases an earlier one. Recording is best-effort: a
failure here degrades the exported section to its collector fallback and never
propagates to the caller.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.timeutil import iso_z

from .recorder import Recorder, recorder_for
from .trace import trace_skip

log = logging.getLogger(__name__)

SECTION = "metadata"
PRODUCER_COORDINATOR = "coordinator"

# Structural model fields carried verbatim from ``summarize_model_config``.
# ``model_class`` is derived (see :func:`_architecture`) and is not on this
# list; everything else is a straight lift so the exported architecture block
# is the parsed config rather than a five-field digest of it.
_ARCHITECTURE_FIELDS = (
    "model_family",
    "model_type",
    "architectures",
    "attention_type",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_size",
    "intermediate_size",
    "max_position_embeddings",
    "vocab_size",
    "torch_dtype",
    "kv_cache_dtype",
    "quantization",
    "is_moe",
    "num_experts",
    "num_experts_per_tok",
    "has_shared_expert",
    "num_shared_experts",
)

_LANGFUSE_FIELDS = ("enabled", "disabled_reason", "trace_id", "session_id", "trace_url")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _image_id(image: str) -> str:
    """The image's bare tag, i.e. its reference without the registry path."""
    return image.split("/")[-1] if image else ""


def _write(session_dir: Path | str | None, payload: Mapping[str, Any], *, producer: str) -> None:
    """Deep-merge ``payload`` into the ``metadata`` singleton. Never raises."""
    if not session_dir:
        trace_skip(reason="no session_dir", section=SECTION)
        return
    if not payload:
        trace_skip(reason="empty payload", section=SECTION)
        return
    try:
        recorder_for(session_dir, producer=producer).record_upsert_singleton(SECTION, dict(payload))
    except Exception as exc:  # noqa: BLE001
        log.debug("record metadata failed", exc_info=True)
        trace_skip(reason="writer raised", section=SECTION, error=exc)


def record_metadata_identity(
    session_dir: Path | str | None,
    manifest: Mapping[str, Any],
    *,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record session identity and the container image from a fresh manifest.

    Called where the manifest is stamped, which is the only point that knows
    the spawn-time image, host and pid without probing the environment a
    second time at export.

    Args:
        session_dir: The session directory; a falsy value is a no-op.
        manifest: The manifest mapping just built for this session.
        producer: The breakdown producer label that owns the write.
    """
    if not isinstance(manifest, Mapping) or not manifest:
        trace_skip(reason="empty payload", section=SECTION)
        return
    image = _text(manifest.get("image"))
    session = {
        "session_id": _text(manifest.get("session_id")),
        "claw_session_id": manifest.get("claw_session_id") or None,
        "sandbox_user_id": manifest.get("sandbox_user_id") or None,
        "created_at_utc": _text(manifest.get("created_at_utc")),
        "host": _text(manifest.get("host")),
        "session_dir": _text(manifest.get("session_dir")),
        "user_data_path": _text(manifest.get("user_data_path")),
        "code_revision": _text(manifest.get("code_revision")),
        "pid": int(manifest.get("pid") or 0),
        "image": image or None,
        "image_id": _image_id(image) or None,
        "max_minutes": int(manifest.get("max_minutes") or 0),
    }
    workload = manifest.get("workload") if isinstance(manifest.get("workload"), Mapping) else {}
    task_config = {
        "model_name": _text(manifest.get("model_name")),
        "model_path": _text(manifest.get("model_path")),
        "framework_name": _text(manifest.get("framework")),
        "gpu_type": _text(manifest.get("gpu_type")),
        "tp": manifest.get("tp"),
        "conc": workload.get("conc"),
        "isl": workload.get("isl"),
        "osl": workload.get("osl"),
        "precision": _text(workload.get("precision")),
        "max_model_len": workload.get("max_model_len"),
        "objective": dict(manifest.get("objective") or {}),
    }
    _write(session_dir, {"session": session, "task_config": task_config}, producer=producer)


def record_metadata_langfuse(
    session_dir: Path | str | None,
    receipt: Mapping[str, Any],
    *,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record the Langfuse entrypoint and push counts from an emitter receipt.

    Called both when the emitter decides whether it is enabled and after the
    final flush, so a session that pushed nothing still explains why.

    Args:
        session_dir: The session directory; a falsy value is a no-op.
        receipt: The emitter receipt mapping.
        producer: The breakdown producer label that owns the write.
    """
    if not isinstance(receipt, Mapping) or not receipt:
        trace_skip(reason="empty payload", section=SECTION)
        return
    _write(session_dir, {"langfuse": _langfuse(receipt)}, producer=producer)


def snapshot_metadata(rec: Recorder, state: Any) -> None:
    """Snapshot session lifecycle and launch config from a live ``SharedState``.

    Covers the facts that only exist in memory while the run is going: the
    budget anchor and its end, the tick count, the crash/resume history, and
    the operator-supplied launch overrides. Called on every state save, so the
    last write before the session stops is the one the export reads.

    Args:
        rec: The recorder writing on the Coordinator's behalf.
        state: The live ``SharedState`` to snapshot.
    """
    session_id = _text(getattr(state, "session_id", ""))
    if not session_id:
        return
    stop_reason = _text(getattr(state, "stop_reason", ""))
    session = {
        "session_id": session_id,
        "start_ts": _text(getattr(state, "start_ts", "")),
        # A resumed run clears its reason but not necessarily the stale
        # timestamp, so the pair is only ever emitted together.
        "ended_at_utc": iso_z(getattr(state, "stop_ts", "")) if stop_reason else "",
        "max_minutes": int(getattr(state, "max_minutes", 0) or 0),
        "tick_count": int(getattr(state, "tick", 0) or 0),
        "recovery": _recovery(state),
    }
    payload: dict[str, Any] = {"session": session, "task_config": _launch_config(state)}
    architecture = _architecture(
        getattr(state, "model_info", None) or {},
        model_class=_text(getattr(state, "model_class", "")),
    )
    if architecture:
        payload["task_config"]["architecture"] = architecture
    rec.record_upsert_singleton(SECTION, payload)


def _architecture(model_info: Any, *, model_class: str = "") -> dict[str, Any]:
    """The structural model block, or ``{}`` when nothing is known."""
    info = dict(model_info or {}) if isinstance(model_info, Mapping) else {}
    resolved_class = _text(model_class)
    if not resolved_class and info:
        resolved_class = "moe" if bool(info.get("is_moe")) else "dense"
    if not info and not resolved_class:
        return {}
    architecture: dict[str, Any] = {"model_class": resolved_class}
    for field in _ARCHITECTURE_FIELDS:
        if field in info:
            architecture[field] = info[field]
    return architecture


def _launch_config(state: Any) -> dict[str, Any]:
    """Workload shape and operator-supplied launch overrides from live state."""
    server_args = getattr(state, "operator_server_args", "") or getattr(state, "server_args", "")
    return {
        "model_name": _text(getattr(state, "model_name", "")),
        "model_path": _text(getattr(state, "model_path", "")),
        "framework_name": _text(getattr(state, "framework", "")),
        "gpu_type": _text(getattr(state, "gpu_type", "")),
        "tp": getattr(state, "tp", None),
        "conc": getattr(state, "conc", None),
        "isl": getattr(state, "isl", None),
        "osl": getattr(state, "osl", None),
        "precision": _text(getattr(state, "precision", "")),
        "max_model_len": getattr(state, "max_model_len", None),
        "launch_env": dict(getattr(state, "operator_extra_env", None) or {}),
        "launch_server_args": _text(server_args),
    }


def _recovery(state: Any) -> dict[str, Any]:
    """Crash / interruption / resume history from live state.

    Crash timestamps are stored as epoch seconds and exported as ISO, so the
    conversion happens here rather than being repeated by every reader.
    """
    crash_count = int(getattr(state, "crash_count", 0) or 0)
    crash_timestamps: list[str] = []
    for raw in getattr(state, "crash_timestamps", None) or []:
        try:
            crash_timestamps.append(datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat())
        except (TypeError, ValueError, OSError, OverflowError):
            continue
    last_exception: dict[str, Any] | None = None
    raw_exception = getattr(state, "last_tick_exception", None)
    if isinstance(raw_exception, Mapping) and raw_exception:
        # Drop the large traceback; keep the compact postmortem header.
        last_exception = {
            "tick": raw_exception.get("tick"),
            "ts": raw_exception.get("ts"),
            "stage": raw_exception.get("stage"),
            "agent": raw_exception.get("agent"),
            "type": raw_exception.get("type"),
            "message": (str(raw_exception.get("message") or "")[:500] or None),
        }
    resume_pending = bool(getattr(state, "resume_pending_revalidation", False))
    return {
        "recovered": bool(crash_count > 0 or crash_timestamps or resume_pending or last_exception),
        "crash_count": crash_count,
        "crash_timestamps": crash_timestamps,
        "degraded_mode": bool(getattr(state, "degraded_mode", False)),
        "resume_pending_revalidation": resume_pending,
        "last_tick_exception": last_exception,
    }


def _langfuse(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The Langfuse block, resolving ``trace_url`` from host + trace id."""
    block: dict[str, Any] = {}
    for field in _LANGFUSE_FIELDS:
        if field in receipt:
            block[field] = receipt[field]
    block["enabled"] = bool(receipt.get("enabled"))
    if not block.get("trace_url"):
        config = receipt.get("config") if isinstance(receipt.get("config"), Mapping) else {}
        host = _text(config.get("host")).rstrip("/")
        trace_id = _text(receipt.get("trace_id"))
        block["trace_url"] = f"{host}/trace/{trace_id}" if host and trace_id else None
    counts = receipt.get("counts")
    if isinstance(counts, Mapping):
        block["counts"] = {str(k): int(v or 0) for k, v in counts.items()}
    return block


__all__ = [
    "record_metadata_identity",
    "record_metadata_langfuse",
    "snapshot_metadata",
]
