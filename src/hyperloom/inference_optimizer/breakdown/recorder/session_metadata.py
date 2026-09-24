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

import time
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.coerce import to_unix
from hyperloom.common.timeutil import iso_z

from ..session_facts import architecture_block, grading_block, recovery_block, workload_signature
from .recorder import Recorder, recorder_for
from .trace import trace_skip

SECTION = "metadata"
PRODUCER_COORDINATOR = "coordinator"

_LANGFUSE_FIELDS = ("enabled", "disabled_reason", "trace_id", "session_id", "trace_url")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _image_id(image: str) -> str:
    """The image's bare tag, i.e. its reference without the registry path."""
    return image.split("/")[-1] if image else ""


def _write(session_dir: Path | str | None, payload: Mapping[str, Any], *, producer: str) -> None:
    """Deep-merge ``payload`` into the ``metadata`` singleton.

    Spool failures are parked by :class:`Recorder`. Projection bugs raise.
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section=SECTION)
        return
    if not payload:
        trace_skip(reason="empty payload", section=SECTION)
        return
    recorder_for(session_dir, producer=producer).record_upsert_singleton(SECTION, dict(payload))


def record_metadata_identity(
    session_dir: Path | str | None,
    manifest: Mapping[str, Any],
    *,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record session identity and the container image from a fresh manifest.

    Called where the manifest is stamped, which is the only point that knows
    the spawn-time image, host and pid without probing the environment a
    second time at export. A falsy ``session_dir`` is a no-op.
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
        "framework_version": _text(manifest.get("framework_version")),
        "gpu_type": _text(manifest.get("gpu_type")),
        "tp": manifest.get("tp"),
        "conc": workload.get("conc"),
        "isl": workload.get("isl"),
        "osl": workload.get("osl"),
        "precision": _text(workload.get("precision")),
        "max_model_len": workload.get("max_model_len"),
        "objective": dict(manifest.get("objective") or {}),
    }
    signature = workload_signature(task_config)
    if signature:
        task_config["workload_signature"] = signature
    _write(session_dir, {"session": session, "task_config": task_config}, producer=producer)


def record_metadata_langfuse(
    session_dir: Path | str | None,
    receipt: Mapping[str, Any],
    *,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record the Langfuse entrypoint and push counts from an emitter receipt.

    Called both when the emitter decides whether it is enabled and after the
    final flush, so a session that pushed nothing still explains why. A falsy
    ``session_dir`` is a no-op.
    """
    if not isinstance(receipt, Mapping) or not receipt:
        trace_skip(reason="empty payload", section=SECTION)
        return
    _write(session_dir, {"langfuse": _langfuse(receipt)}, producer=producer)


def snapshot_metadata(rec: Recorder, state: Any) -> None:
    """Snapshot session lifecycle and launch config from a live ``SharedState``.

    Covers the facts that only exist in memory while the run is going: the
    budget anchor and its end, how long the run has been going, the tick count,
    the crash/resume history, and the operator-supplied launch overrides.
    Called on every state save, so the last write before the session stops is
    the one the export reads -- which is what freezes the elapsed time at the
    end of the run instead of letting a re-export stretch it.
    """
    session_id = _text(getattr(state, "session_id", ""))
    if not session_id:
        return
    stop_reason = _text(getattr(state, "stop_reason", ""))
    leg_seconds, total_seconds = _elapsed_seconds(state)
    session = {
        "session_id": session_id,
        "start_ts": _text(getattr(state, "start_ts", "")),
        # A resumed run clears its reason but not necessarily the stale
        # timestamp, so the pair is only ever emitted together.
        "ended_at_utc": iso_z(getattr(state, "stop_ts", "")) if stop_reason else "",
        "max_minutes": int(getattr(state, "max_minutes", 0) or 0),
        "elapsed_minutes": round(leg_seconds / 60.0, 2),
        "total_elapsed_minutes": round(total_seconds / 60.0, 2),
        "tick_count": int(getattr(state, "tick", 0) or 0),
        "recovery": recovery_block(state),
    }
    payload: dict[str, Any] = {
        "session": session,
        "task_config": _launch_config(state),
        "grading": grading_block(state),
    }
    architecture = architecture_block(
        getattr(state, "model_info", None) or {},
        model_class=_text(getattr(state, "model_class", "")),
    )
    if architecture:
        payload["task_config"]["architecture"] = architecture
    rec.record_upsert_singleton(SECTION, payload)


def _launch_config(state: Any) -> dict[str, Any]:
    """Workload shape and operator-supplied launch overrides from live state."""
    server_args = getattr(state, "operator_server_args", "") or getattr(state, "server_args", "")
    config: dict[str, Any] = {
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
    # The singleton merges leaf-by-leaf with no notion of an empty value, so
    # the version detected at launch has to be left alone rather than
    # overwritten every save by a state field that stays empty until (and
    # unless) the framework reports one.
    framework_version = _text(getattr(state, "framework_version", ""))
    if framework_version:
        config["framework_version"] = framework_version
    signature = workload_signature(config)
    if signature:
        config["workload_signature"] = signature
    return config


def _elapsed_seconds(state: Any) -> tuple[float, float]:
    """Seconds this run leg has been running, and the total across all legs.

    The leg starts at ``resumed_ts``, falling back to ``start_ts`` only for a
    session that has run once: a resume after a clean stop keeps the original
    anchor so the wall-clock budget still counts from there, and measuring the
    leg from it would charge the leg with the gap between the two. It ends at
    ``stop_ts``, which is evidence of an end only while a ``stop_reason``
    stands, and any stamp that does not postdate the leg's start is a stale one
    from the previous leg. The total is the session's own charged budget
    (``elapsed_charged_sec`` plus what the live leg has run since the last
    charge), read rather than recomputed because the budget that stops the run
    is the one a report has to agree with.
    """
    started = to_unix(_text(getattr(state, "resumed_ts", "")) or _text(getattr(state, "start_ts", "")), 0.0) or 0.0
    ended = 0.0
    if _text(getattr(state, "stop_reason", "")):
        ended = to_unix(_text(getattr(state, "stop_ts", "")), 0.0) or 0.0
    if ended <= started:
        ended = time.time()
    leg = max(0.0, ended - started) if started > 0.0 else 0.0
    charged = max(0.0, float(getattr(state, "elapsed_charged_sec", 0.0) or 0.0))
    anchor = float(getattr(state, "leg_anchor_unix", 0.0) or 0.0)
    live = max(0.0, time.time() - anchor) if anchor > 0.0 else 0.0
    total = charged + live
    return leg, total or leg


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
