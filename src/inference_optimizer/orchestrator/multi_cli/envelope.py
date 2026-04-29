"""A2A v0 envelope — single JSONL line shape used for both inbox and outbox.

Each line in ``$SESSION_DIR/agents/<name>/{inbox,outbox}.jsonl`` is one
:class:`Envelope` JSON-encoded. The envelope is intentionally **a
superset** of the existing :class:`~inference_optimizer.orchestrator.message_bus.Message`
shape, so the Router can mirror bus events into JSONL with zero loss.

Design notes
------------

* We carry a ``kind`` discriminator (``"intent"`` / ``"message"``) instead
  of two dataclasses because:

  - reading agents only need to follow `inbox.jsonl` line by line;
    one shape avoids a parse-time branch;
  - the Conductor can apply the same `seq` / `msg_id` invariants to
    both flavours without inheritance / unions.

* ``seq`` semantics:

  - Inbox: matches the SQLite ``events.seq`` (global monotonic) so an
    agent that resumes can use ``cursors.last_processed_seq`` to skip
    already-seen lines verbatim.
  - Outbox: per-file monotonic counter assigned by whoever writes the
    line (the agent or :func:`write_envelope`); used only as an
    idempotency / dedup key — the *real* seq comes from the bus once
    the Router accepts the intent.

* The on-disk format is **append-only JSONL** with one envelope per
  line. We rely on POSIX atomic-append semantics for the writer; readers
  use a `_cursor.txt` file to remember the byte offset reached so they
  don't reparse on every poll.

* Writes go through :func:`write_envelope` rather than ``open(...).write``
  so a future swap to inotify / fsync batching stays a one-line change.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator


# ---------------------------------------------------------------------------
class EnvelopeError(ValueError):
    """Raised for malformed envelope input or invalid file lines."""


class EnvelopeKind(str, Enum):
    """Discriminator on every JSONL line.

    INTENT
        Agent → Router. Carries one ``intent_type`` / ``payload`` pair,
        validated against ``IntentType`` downstream.
    MESSAGE
        Router → agent (or fan-out for ``to_agent="*"``). Mirrors a bus
        :class:`Message` row by row: ``topic`` + ``payload``.
    """

    INTENT = "intent"
    MESSAGE = "message"


# ---------------------------------------------------------------------------
# Per-process serial number for outbox writes — small helper so callers
# don't need to track seq themselves when the file's existing tail is
# unknown. The Router uses the SQLite seq for inbox; only outbox writes
# (mostly tests + agent stubs) need this.
# ---------------------------------------------------------------------------
class _LocalSeqAllocator:
    """Per-file monotonic seq counter, scoped to one process.

    The first write inspects the existing file (if any) to bootstrap from
    its last seq, then keeps an in-memory counter. This is *not* safe
    across processes — but multi-cli writers are 1:1 with files (each
    agent owns its outbox), so this is sufficient.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[Path, int] = {}

    def next_seq(self, path: Path) -> int:
        with self._lock:
            cur = self._counters.get(path)
            if cur is None:
                cur = _read_max_seq(path)
                self._counters[path] = cur
            cur += 1
            self._counters[path] = cur
            return cur

    def reset(self, path: Path | None = None) -> None:
        with self._lock:
            if path is None:
                self._counters.clear()
            else:
                self._counters.pop(path, None)


_SEQ_ALLOCATOR = _LocalSeqAllocator()


def _read_max_seq(path: Path) -> int:
    """Return the largest ``seq`` present in ``path`` (0 when absent).

    Used to bootstrap the per-file counter when the writer process
    starts after an existing run wrote some lines.
    """
    if not path.is_file():
        return 0
    max_seq = 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seq = obj.get("seq")
                if isinstance(seq, int) and seq > max_seq:
                    max_seq = seq
    except OSError:
        return max_seq
    return max_seq


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
@dataclass
class Envelope:
    """One JSONL line.

    Fields with ``None`` are omitted on serialise so the JSONL stays
    human-grep-friendly; round-trip equality is preserved by
    :meth:`from_json` reinjecting them as ``None``.
    """

    kind: EnvelopeKind
    msg_id: str
    seq: int
    ts: str
    from_agent: str
    to_agent: str
    payload: dict[str, Any] = field(default_factory=dict)
    # MESSAGE-only:
    topic: str | None = None
    priority: int = 1
    in_reply_to: str | None = None
    # INTENT-only:
    intent_type: str | None = None

    # ------------------------------------------------------------------
    @classmethod
    def message(
        cls,
        *,
        msg_id: str,
        seq: int,
        from_agent: str,
        to_agent: str,
        topic: str,
        payload: dict[str, Any] | None = None,
        priority: int = 1,
        in_reply_to: str | None = None,
        ts: str | None = None,
    ) -> "Envelope":
        return cls(
            kind=EnvelopeKind.MESSAGE,
            msg_id=msg_id,
            seq=seq,
            ts=ts or _now_iso(),
            from_agent=from_agent,
            to_agent=to_agent,
            topic=topic,
            payload=dict(payload or {}),
            priority=priority,
            in_reply_to=in_reply_to,
            intent_type=None,
        )

    @classmethod
    def intent(
        cls,
        *,
        from_agent: str,
        intent_type: str,
        payload: dict[str, Any] | None = None,
        seq: int | None = None,
        msg_id: str | None = None,
        in_reply_to: str | None = None,
        to_agent: str = "conductor",
        ts: str | None = None,
    ) -> "Envelope":
        return cls(
            kind=EnvelopeKind.INTENT,
            msg_id=msg_id or uuid.uuid4().hex,
            seq=seq if seq is not None else 0,
            ts=ts or _now_iso(),
            from_agent=from_agent,
            to_agent=to_agent,
            topic=None,
            payload=dict(payload or {}),
            priority=1,
            in_reply_to=in_reply_to,
            intent_type=intent_type,
        )

    # ------------------------------------------------------------------
    def to_json(self) -> str:
        d = asdict(self)
        d["kind"] = self.kind.value
        # Drop fields that don't apply to this envelope kind so the JSONL
        # line stays compact + grep-friendly.
        if self.kind is EnvelopeKind.INTENT:
            d.pop("topic", None)
            d.pop("priority", None)
        else:
            d.pop("intent_type", None)
        # Always drop None fields to keep lines small.
        d = {k: v for k, v in d.items() if v is not None}
        return json.dumps(d, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "Envelope":
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EnvelopeError(f"invalid JSONL line: {exc}") from exc
        if not isinstance(obj, dict):
            raise EnvelopeError(f"envelope must be a JSON object, got {type(obj).__name__}")
        try:
            kind = EnvelopeKind(obj["kind"])
        except KeyError as exc:
            raise EnvelopeError("envelope missing 'kind'") from exc
        except ValueError as exc:
            raise EnvelopeError(f"envelope kind {obj['kind']!r} is not recognised") from exc

        for required in ("msg_id", "seq", "ts", "from_agent", "to_agent"):
            if required not in obj:
                raise EnvelopeError(f"envelope missing required field {required!r}")

        try:
            return cls(
                kind=kind,
                msg_id=str(obj["msg_id"]),
                seq=int(obj["seq"]),
                ts=str(obj["ts"]),
                from_agent=str(obj["from_agent"]),
                to_agent=str(obj["to_agent"]),
                payload=dict(obj.get("payload", {}) or {}),
                topic=obj.get("topic"),
                priority=int(obj.get("priority", 1)),
                in_reply_to=obj.get("in_reply_to"),
                intent_type=obj.get("intent_type"),
            )
        except (TypeError, ValueError) as exc:
            raise EnvelopeError(f"envelope failed to parse: {exc}") from exc


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def write_envelope(path: Path, env: Envelope) -> int:
    """Append one envelope to ``path`` (creating parents). Returns the
    written ``seq`` (auto-allocated when ``env.seq <= 0``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if env.seq <= 0:
        env.seq = _SEQ_ALLOCATOR.next_seq(path)
    line = env.to_json()
    # POSIX append is atomic for writes < PIPE_BUF (typically 4 KiB) and
    # JSONL lines stay well under that for any sane payload. We still
    # serialise local writers via the seq allocator's lock above.
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return env.seq


def read_envelopes(path: Path) -> list[Envelope]:
    """Read every envelope in ``path`` from disk (best-effort — bad lines
    are skipped). Used by tests + Router cold-start replay.
    """
    out: list[Envelope] = []
    if not Path(path).is_file():
        return out
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                out.append(Envelope.from_json(line))
            except EnvelopeError:
                continue
    return out


def iter_new_envelopes(path: Path, *, after_offset: int) -> Iterator[tuple[int, Envelope]]:
    """Yield ``(new_offset, envelope)`` pairs from ``path`` starting at
    byte ``after_offset``.

    ``new_offset`` is the byte position *after* the yielded envelope's
    line. Callers persist the latest one to a cursor file so they only
    re-read fresh tail bytes on subsequent polls.

    Malformed lines are skipped silently; their bytes are still consumed
    and counted in ``new_offset`` so the reader keeps moving forward.
    """
    p = Path(path)
    if not p.is_file():
        return
    try:
        with p.open("rb") as fh:
            fh.seek(after_offset)
            while True:
                raw = fh.readline()
                if not raw:
                    return
                if not raw.endswith(b"\n"):
                    # Partial line — writer mid-flush. Don't advance offset
                    # so we re-read it fully next time.
                    return
                offset_after = fh.tell()
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    env = Envelope.from_json(line)
                except EnvelopeError:
                    continue
                yield (offset_after, env)
    except OSError:
        return


def envelopes_since_seq(path: Path, *, after_seq: int) -> list[Envelope]:
    """Return all envelopes in ``path`` with ``seq > after_seq``."""
    return [e for e in read_envelopes(path) if e.seq > after_seq]


# ---------------------------------------------------------------------------
# Cursor file — tracks the byte offset a follower has consumed
# ---------------------------------------------------------------------------
def read_cursor(path: Path) -> int:
    """Return the byte offset stored in ``path`` (0 when missing/invalid)."""
    p = Path(path)
    if not p.is_file():
        return 0
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError:
        return 0
    try:
        return max(0, int(text))
    except ValueError:
        return 0


def write_cursor(path: Path, offset: int) -> None:
    """Atomically persist ``offset`` to ``path`` via tmp + rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(str(int(offset)), encoding="utf-8")
    os.replace(tmp, p)


__all__ = [
    "Envelope",
    "EnvelopeError",
    "EnvelopeKind",
    "envelopes_since_seq",
    "iter_new_envelopes",
    "read_cursor",
    "read_envelopes",
    "write_cursor",
    "write_envelope",
]
