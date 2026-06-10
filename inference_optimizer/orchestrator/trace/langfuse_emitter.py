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
            self._trace_id = lfmap.derive_trace_id(self.session_dir.name)
            log.info(
                "langfuse: live push enabled (host=%s, trace_id=%s)",
                creds.get("LANGFUSE_HOST"), self._trace_id,
            )
            return True
        except Exception:  # noqa: BLE001
            log.warning("langfuse: client init failed; live push disabled.", exc_info=True)
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

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
        """Send one Generation built from a token row and/or conversation row."""
        base = token_row or conv_row or {}
        phase = str(base.get("phase") or lfmap.UNPHASED)
        start = lfmap.parse_ts(base.get("ts"))
        has_text = conv_row is not None
        try:
            gen = self._client.start_observation(
                name=lfmap.generation_name(base),
                as_type="generation",
                start_time=start,
                trace_context={"trace_id": self._trace_id},
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
        except Exception:  # noqa: BLE001
            log.debug("langfuse: flush_session reconcile failed", exc_info=True)
        finally:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                log.debug("langfuse: client.flush failed", exc_info=True)

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
        """Convert each decision_trace row into Langfuse Score(s) on the trace."""
        for drow in _load_jsonl(decision_trace_path(self.session_dir)):
            for score in lfmap.decision_to_scores(drow):
                try:
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
