"""MultiCLIRouter — JSONL ↔ SQLite bridge for ``--mode multi-cli``.

Two flows
---------

**Outbox → bus** (Phase 2+: cross-process intent dispatch)

    Each agent CLI writes :class:`Envelope` lines with
    ``kind=intent`` to ``$SESSION_DIR/agents/<name>/outbox.jsonl``. The
    Router watches every active agent's outbox via
    :func:`iter_new_envelopes`, reconstructs an :class:`Intent`, runs it
    through :class:`PolicyGate`, and (when accepted) hands it to a
    caller-supplied ``intent_handler`` — typically ``Conductor._handle_intent``.
    Denied intents become ``policy_denied`` observations on the bus exactly
    like in the in-process reactor path.

**Bus → inbox** (Phase 1: passive mirror)

    Whenever a new event lands in the SQLite ``events`` table the Router
    fans it out as a ``kind=message`` envelope to every relevant
    inbox.jsonl (the destination's box, plus every agent's box when
    ``to_agent="*"``). Each agent's CLI consumes its inbox by following
    the file. This is also what lets a cold-started CLI use a byte-cursor
    instead of replaying SQLite.

Phase plan (mirrors the project plan):

    Phase 1 — wire :meth:`mirror_bus_tick` into Conductor as a passive
              writer alongside the existing reactor model. Zero behaviour
              change; we just gain JSONL output.
    Phase 2 — add :meth:`drain_outbox_tick` per active agent and skip the
              corresponding in-process reactor; verify cross-process
              PolicyGate enforcement.
    Phase 3 — every reactor swapped for a CLI; in-process reactors become
              the test-only fallback.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, TYPE_CHECKING

from ..intent_parser import Intent, IntentType
from ..message_bus import Message, MessageBus
from .envelope import (
    Envelope,
    EnvelopeError,
    EnvelopeKind,
    iter_new_envelopes,
    read_cursor,
    write_cursor,
    write_envelope,
)

if TYPE_CHECKING:  # pragma: no cover - type-only
    from ..policy import PolicyGate
    from .agent_card import AgentCard


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
class RouterError(RuntimeError):
    """Raised when the Router cannot honour a routing request."""


# ---------------------------------------------------------------------------
# Path conventions for the per-agent session directory
# ---------------------------------------------------------------------------
def agent_session_dir(session_dir: Path, agent_name: str) -> Path:
    """``$SESSION_DIR/agents/<name>/`` — created on demand."""
    p = Path(session_dir) / "agents" / agent_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def agent_inbox_path(
    session_dir: Path, agent_name: str, *, filename: str = "inbox.jsonl"
) -> Path:
    return agent_session_dir(session_dir, agent_name) / filename


def agent_outbox_path(
    session_dir: Path, agent_name: str, *, filename: str = "outbox.jsonl"
) -> Path:
    return agent_session_dir(session_dir, agent_name) / filename


def _outbox_cursor_path(outbox: Path) -> Path:
    """Co-located ``<file>.cursor`` byte offset record."""
    return outbox.with_suffix(outbox.suffix + ".cursor")


def _inbox_seq_cursor_path(inbox: Path) -> Path:
    """Router-private cursor: last bus seq we mirrored into ``inbox``.

    NOTE: this is intentionally distinct from the agent-side cursor
    (``inbox.jsonl.seq`` — written by the agent itself to track which
    envelopes it has processed). The Router writes ``inbox.jsonl.mirrored``
    so the two cursors do not trample each other.
    """
    return inbox.with_suffix(inbox.suffix + ".mirrored")


# ---------------------------------------------------------------------------
# Intent reconstruction (envelope → Intent)
# ---------------------------------------------------------------------------
def envelope_to_intent(env: Envelope) -> Intent:
    """Map an outbox INTENT envelope back into an :class:`Intent`.

    Raises :class:`EnvelopeError` if the envelope is the wrong kind or its
    ``intent_type`` is unknown — callers should record those as
    ``policy_denied`` observations (rule="payload") rather than crashing.
    """
    if env.kind is not EnvelopeKind.INTENT:
        raise EnvelopeError(
            f"envelope kind={env.kind.value!r} is not 'intent'"
        )
    if not env.intent_type:
        raise EnvelopeError("intent envelope missing 'intent_type'")
    try:
        itype = IntentType(env.intent_type)
    except ValueError as exc:
        raise EnvelopeError(
            f"unknown intent_type {env.intent_type!r}"
        ) from exc
    return Intent(type=itype, payload=dict(env.payload or {}))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@dataclass
class _MirrorState:
    """Tracks last bus seq mirrored into one inbox file."""

    last_seq: int = 0
    cursor_path: Path | None = None


@dataclass
class _DrainState:
    """Tracks byte offset reached in one outbox file."""

    offset: int = 0
    cursor_path: Path | None = None


# Type alias for the intent handler callback.
IntentHandler = Callable[[str, Intent], Awaitable[None]]
DenyRecorder = Callable[[str, Intent, str, str], Awaitable[None]]


class MultiCLIRouter:
    """JSONL ↔ SQLite bridge for one session.

    Construction takes the Conductor's existing wiring (bus, policy) plus
    callbacks for "what to do when an intent is accepted / denied". The
    Router is intentionally **dumb**: it doesn't own state machines, it
    doesn't call ``backend.run``. It only:

    * mirrors new bus events into agent inboxes (Phase 1);
    * reads agent outboxes and forwards intents to the supplied handlers
      (Phase 2+).

    Both directions are pulled by ``tick`` methods so callers stay in
    control of cadence; an :meth:`run` convenience wraps them in a loop.
    """

    DEFAULT_TICK_S = 0.5

    def __init__(
        self,
        *,
        session_dir: Path,
        bus: MessageBus,
        policy: "PolicyGate | None" = None,
        agents: Iterable["AgentCard"] | None = None,
        intent_handler: IntentHandler | None = None,
        deny_recorder: DenyRecorder | None = None,
        tick_s: float | None = None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.bus = bus
        self.policy = policy
        self.intent_handler = intent_handler
        self.deny_recorder = deny_recorder
        self.tick_s = tick_s or self.DEFAULT_TICK_S
        self._agents: dict[str, "AgentCard"] = {a.name: a for a in (agents or ())}
        self._mirror_state: dict[str, _MirrorState] = {}
        self._drain_state: dict[str, _DrainState] = {}
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Agent registry mutators (dynamic enable/disable)
    # ------------------------------------------------------------------
    def register(self, card: "AgentCard") -> None:
        """Add or replace one agent card. Reuses existing cursor state."""
        self._agents[card.name] = card

    def unregister(self, agent_name: str) -> None:
        self._agents.pop(agent_name, None)
        self._mirror_state.pop(agent_name, None)
        self._drain_state.pop(agent_name, None)

    @property
    def agents(self) -> Mapping[str, "AgentCard"]:
        return dict(self._agents)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def inbox_path(self, agent_name: str) -> Path:
        card = self._agents.get(agent_name)
        filename = card.inbox_filename if card else "inbox.jsonl"
        return agent_inbox_path(self.session_dir, agent_name, filename=filename)

    def outbox_path(self, agent_name: str) -> Path:
        card = self._agents.get(agent_name)
        filename = card.outbox_filename if card else "outbox.jsonl"
        return agent_outbox_path(self.session_dir, agent_name, filename=filename)

    # ------------------------------------------------------------------
    # Bus → inbox mirror
    # ------------------------------------------------------------------
    def _mirror_state_for(self, agent_name: str) -> _MirrorState:
        st = self._mirror_state.get(agent_name)
        if st is not None:
            return st
        cursor_file = _inbox_seq_cursor_path(self.inbox_path(agent_name))
        last_seq = 0
        if cursor_file.is_file():
            try:
                last_seq = int(cursor_file.read_text(encoding="utf-8").strip() or 0)
            except (OSError, ValueError):
                last_seq = 0
        st = _MirrorState(last_seq=last_seq, cursor_path=cursor_file)
        self._mirror_state[agent_name] = st
        return st

    async def mirror_bus_tick(self) -> int:
        """Pull new events from the bus and append them to recipient inboxes.

        Returns the total number of envelope lines written (across all
        agents). Safe to call from any cadence — replay is idempotent
        thanks to per-inbox seq cursors.
        """
        if not self._agents:
            return 0
        written = 0
        for name, card in self._agents.items():
            if not card.enabled:
                continue
            written += await self._mirror_for_agent(name)
        return written

    async def _mirror_for_agent(self, agent_name: str) -> int:
        st = self._mirror_state_for(agent_name)
        msgs: list[Message] = await self.bus.replay_for(
            agent_name, after_seq=st.last_seq
        )
        if not msgs:
            return 0
        inbox = self.inbox_path(agent_name)
        wrote = 0
        max_seq = st.last_seq
        for msg in msgs:
            # Don't echo the agent's own message back to itself.
            if msg.from_agent == agent_name:
                if msg.seq is not None:
                    max_seq = max(max_seq, msg.seq)
                continue
            env = Envelope.message(
                msg_id=msg.msg_id,
                seq=msg.seq if msg.seq is not None else 0,
                from_agent=msg.from_agent,
                to_agent=msg.to_agent,
                topic=msg.topic,
                payload=dict(msg.payload),
                priority=msg.priority,
                in_reply_to=msg.in_reply_to,
                ts=msg.ts,
            )
            try:
                write_envelope(inbox, env)
                wrote += 1
            except OSError:
                log.exception("router: failed to mirror msg=%s into %s",
                              msg.msg_id, inbox)
                # Stop here so the seq cursor isn't advanced past the
                # un-mirrored message; we'll retry next tick.
                break
            if msg.seq is not None:
                max_seq = max(max_seq, msg.seq)
        if max_seq > st.last_seq:
            st.last_seq = max_seq
            if st.cursor_path is not None:
                try:
                    st.cursor_path.parent.mkdir(parents=True, exist_ok=True)
                    st.cursor_path.write_text(str(max_seq), encoding="utf-8")
                except OSError:
                    log.exception("router: failed to persist mirror cursor for %s",
                                  agent_name)
        return wrote

    # ------------------------------------------------------------------
    # Outbox → intent_handler drain
    # ------------------------------------------------------------------
    def _drain_state_for(self, agent_name: str) -> _DrainState:
        st = self._drain_state.get(agent_name)
        if st is not None:
            return st
        outbox = self.outbox_path(agent_name)
        cursor_file = _outbox_cursor_path(outbox)
        st = _DrainState(offset=read_cursor(cursor_file), cursor_path=cursor_file)
        self._drain_state[agent_name] = st
        return st

    async def drain_outbox_tick(self) -> int:
        """Drain every agent's outbox once. Returns intents processed.

        For each new envelope:
            * decode → :class:`Intent`
            * PolicyGate check
            * if accepted: hand to ``intent_handler``
            * if denied: hand to ``deny_recorder`` (typically logs a
              ``policy_denied`` observation on the bus)
        """
        if not self._agents:
            return 0
        processed = 0
        for name, card in self._agents.items():
            if not card.enabled:
                continue
            processed += await self._drain_for_agent(name)
        return processed

    async def _drain_for_agent(self, agent_name: str) -> int:
        st = self._drain_state_for(agent_name)
        outbox = self.outbox_path(agent_name)
        new_envelopes: list[tuple[int, Envelope]] = list(
            iter_new_envelopes(outbox, after_offset=st.offset)
        )
        if not new_envelopes:
            return 0
        processed = 0
        for offset_after, env in new_envelopes:
            await self._handle_outbox_envelope(agent_name, env)
            processed += 1
            st.offset = offset_after
            if st.cursor_path is not None:
                try:
                    write_cursor(st.cursor_path, st.offset)
                except OSError:
                    log.exception("router: failed to persist drain cursor for %s",
                                  agent_name)
        return processed

    async def _handle_outbox_envelope(
        self, agent_name: str, env: Envelope
    ) -> None:
        if env.kind is not EnvelopeKind.INTENT:
            log.debug(
                "router: skipping non-intent envelope from %s (kind=%s msg_id=%s)",
                agent_name, env.kind.value, env.msg_id,
            )
            return
        try:
            intent = envelope_to_intent(env)
        except EnvelopeError as exc:
            log.info(
                "router: malformed intent from %s msg_id=%s: %s",
                agent_name, env.msg_id, exc,
            )
            if self.deny_recorder is not None:
                # Use a placeholder intent so the deny path can record an
                # observation; the recorder is allowed to ignore type=None.
                placeholder = Intent(
                    type=IntentType.SEND_MESSAGE, payload={"original": env.to_json()}
                )
                await self.deny_recorder(
                    agent_name, placeholder, "payload", str(exc)
                )
            return
        # PolicyGate (when wired). Conductor's gate consults the role
        # registry; we trust the agent_name in the envelope is the
        # producing agent (filenames already namespace by agent).
        if self.policy is not None:
            from ..policy import PolicyDenied  # local import to avoid cycles
            try:
                self.policy.validate_intent(agent_name, intent)
            except PolicyDenied as exc:
                log.info(
                    "router: policy denied agent=%s intent=%s rule=%s: %s",
                    agent_name, intent.type.value, exc.rule, exc,
                )
                if self.deny_recorder is not None:
                    await self.deny_recorder(
                        agent_name, intent, exc.rule or "unknown", str(exc)
                    )
                return
        if self.intent_handler is not None:
            try:
                await self.intent_handler(agent_name, intent)
            except Exception:  # noqa: BLE001 — log + swallow to keep the loop alive
                log.exception(
                    "router: intent_handler raised for agent=%s intent=%s",
                    agent_name, intent.type.value,
                )

    # ------------------------------------------------------------------
    # Cold start — replay every agent's outbox tail before live ticks
    # ------------------------------------------------------------------
    async def replay_outboxes(self) -> int:
        """Process every existing outbox line not yet covered by its cursor.

        Useful after a Conductor restart to pick up intents written by an
        agent CLI while we were offline.
        """
        return await self.drain_outbox_tick()

    # ------------------------------------------------------------------
    # Convenience loop
    # ------------------------------------------------------------------
    def request_stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        """Drive ``drain_outbox_tick`` + ``mirror_bus_tick`` at ``tick_s``.

        Stops when :meth:`request_stop` is invoked. Errors in either tick
        log + continue — the Router never tears the Conductor down.
        """
        while not self._stop_event.is_set():
            try:
                await self.drain_outbox_tick()
            except Exception:  # noqa: BLE001
                log.exception("router: drain_outbox_tick crashed")
            try:
                await self.mirror_bus_tick()
            except Exception:  # noqa: BLE001
                log.exception("router: mirror_bus_tick crashed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.tick_s)
            except asyncio.TimeoutError:
                continue


__all__ = [
    "DenyRecorder",
    "IntentHandler",
    "MultiCLIRouter",
    "RouterError",
    "agent_inbox_path",
    "agent_outbox_path",
    "agent_session_dir",
    "envelope_to_intent",
]
