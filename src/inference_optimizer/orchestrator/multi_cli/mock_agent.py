"""Mock CLI agent — Python module that follows the multi-CLI A2A protocol.

Why
---

The launcher's ``backend=mock`` pane writes one heartbeat then exits;
that's not enough to validate the *full* loop (Conductor -> launcher ->
real subprocess -> Router -> bus -> back to inbox).

This module is a "real" CLI agent that any test or operator can swap in
for ``claude`` / ``codex``. It implements the protocol the production
agent prompts describe:

1. Read ``$AGENT_DIR/inbox.jsonl.seq`` to discover the last bus seq it
   processed (``0`` when missing).
2. Read every line from ``$AGENT_DIR/inbox.jsonl`` whose envelope
   ``seq > last_seq``.
3. For each new envelope, optionally emit a response intent appended to
   ``$AGENT_DIR/outbox.jsonl`` using the canonical envelope schema
   (per-file monotonic ``seq``).
4. Persist the new last_seq to ``$AGENT_DIR/inbox.jsonl.seq``.
5. Sleep ``--poll-s`` seconds, repeat.
6. Exit cleanly when ``$SESSION_DIR/STOP_AGENT_<name>`` exists.

Behaviour is configurable via flags so the same module can stand in for
*any* agent role:

  --emit-on-event TOPIC=INTENT_TYPE   send an intent every time we see
                                      a message whose topic matches
                                      TOPIC. May repeat.
  --emit-every N                      additionally emit one heartbeat
                                      every N polls regardless of inbox.
  --baseline-on-start                 once at startup, emit a single
                                      ``delegate(action_name=baseline)``
                                      intent (executor-only — used by
                                      the e2e smoke).
  --max-iterations N                  bail after N polls (test hard cap).

Run via::

    python -m inference_optimizer.orchestrator.multi_cli.mock_agent \\
        --agent-name executor --session-dir /tmp/io/<sid>

The launcher writes a pane script that invokes exactly this module when
``AgentCard.backend == "mock-cli"``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Imported as part of the package so the module participates in the
# normal logging / version sharing.
from .envelope import (
    Envelope,
    EnvelopeError,
    write_envelope,
    read_cursor as _read_cursor,
    write_cursor as _write_cursor,
)


log = logging.getLogger("inference_optimizer.mock_agent")


# Default response map: topic -> intent_type to emit when the agent sees
# a matching inbox message. Conservative; tests override via CLI.
_DEFAULT_RESPONSES: dict[str, str] = {
    # Reflective heartbeats — every agent acknowledges the clock so
    # tests can prove "the agent saw the event".
    "reflection_tick": "send_message",
    "event": "send_message",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _agent_dir(session_dir: Path, agent_name: str) -> Path:
    p = Path(session_dir) / "agents" / agent_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _inbox_path(session_dir: Path, agent_name: str) -> Path:
    return _agent_dir(session_dir, agent_name) / "inbox.jsonl"


def _outbox_path(session_dir: Path, agent_name: str) -> Path:
    return _agent_dir(session_dir, agent_name) / "outbox.jsonl"


def _seq_cursor_path(inbox: Path) -> Path:
    return inbox.with_suffix(inbox.suffix + ".seq")


def _stop_path(session_dir: Path, agent_name: str) -> Path:
    return Path(session_dir) / f"STOP_AGENT_{agent_name}"


# ---------------------------------------------------------------------------
def _read_inbox_after(inbox: Path, *, after_seq: int) -> list[Envelope]:
    """Best-effort tail-read of new inbox envelopes.

    Equivalent to the bash one-liner the system_prompt instructs a real
    claude CLI to run. We re-implement it in Python here so the tests
    don't depend on awk/jq.
    """
    if not inbox.is_file():
        return []
    out: list[Envelope] = []
    try:
        with inbox.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    env = Envelope.from_json(line)
                except EnvelopeError:
                    continue
                if env.seq > after_seq:
                    out.append(env)
    except OSError:
        return out
    return out


def _emit_intent(
    outbox: Path,
    *,
    from_agent: str,
    intent_type: str,
    payload: dict,
    in_reply_to: str | None = None,
) -> int:
    env = Envelope.intent(
        from_agent=from_agent,
        intent_type=intent_type,
        payload=payload,
        msg_id=uuid.uuid4().hex,
        in_reply_to=in_reply_to,
        ts=_now_iso(),
    )
    return write_envelope(outbox, env)


# ---------------------------------------------------------------------------
def _parse_response_map(spec_list: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = dict(_DEFAULT_RESPONSES)
    for spec in spec_list or ():
        if "=" not in spec:
            raise SystemExit(
                f"--emit-on-event needs TOPIC=INTENT_TYPE, got {spec!r}"
            )
        topic, intent_type = spec.split("=", 1)
        out[topic.strip()] = intent_type.strip()
    return out


# ---------------------------------------------------------------------------
def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m inference_optimizer.orchestrator.multi_cli.mock_agent",
        description="Mock CLI agent following the multi-cli A2A protocol.",
    )
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--poll-s", type=float, default=0.2)
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="hard cap on poll loops (0 = unlimited)")
    parser.add_argument("--emit-every", type=int, default=0,
                        help="emit a heartbeat every N polls (0 = disabled)")
    parser.add_argument("--emit-on-event", action="append", default=[],
                        help="TOPIC=INTENT_TYPE; may repeat. "
                             "Default: reflection_tick=send_message, event=send_message")
    parser.add_argument("--baseline-on-start", action="store_true",
                        help="executor-only: emit one delegate(baseline) at startup")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s mock_agent[%(name)s] :: %(message)s",
    )
    log_local = logging.getLogger(args.agent_name)

    inbox = _inbox_path(args.session_dir, args.agent_name)
    outbox = _outbox_path(args.session_dir, args.agent_name)
    cursor = _seq_cursor_path(inbox)
    stop = _stop_path(args.session_dir, args.agent_name)
    response_map = _parse_response_map(args.emit_on_event)

    log_local.info(
        "starting mock_agent agent=%s session=%s inbox=%s outbox=%s",
        args.agent_name, args.session_dir, inbox, outbox,
    )
    log_local.info("response_map=%s emit_every=%s baseline_on_start=%s",
                   response_map, args.emit_every, args.baseline_on_start)

    if args.baseline_on_start:
        seq = _emit_intent(
            outbox,
            from_agent=args.agent_name,
            intent_type="delegate",
            payload={
                "action_name": "baseline",
                "params": {},
                "predicted_gain_pct": 0.0,
                "reason": "first measurement (mock agent baseline-on-start)",
            },
        )
        log_local.info("baseline-on-start: emitted delegate(baseline) seq=%d", seq)

    iteration = 0
    while True:
        if stop.exists():
            log_local.info("STOP file present (%s); exiting cleanly", stop)
            return 0
        iteration += 1
        if args.max_iterations and iteration > args.max_iterations:
            log_local.info("max_iterations=%d reached; exiting", args.max_iterations)
            return 0

        last_seq = _read_cursor(cursor)
        new_envs = _read_inbox_after(inbox, after_seq=last_seq)
        if new_envs:
            log_local.info(
                "iter=%d picked up %d new inbox envelopes (last_seq=%d)",
                iteration, len(new_envs), last_seq,
            )
        for env in new_envs:
            topic = env.topic or "(no-topic)"
            intent_type = response_map.get(topic)
            if intent_type:
                seq = _emit_intent(
                    outbox,
                    from_agent=args.agent_name,
                    intent_type=intent_type,
                    payload={
                        "topic": "heartbeat",
                        "body_md": (
                            f"mock_agent {args.agent_name} acked "
                            f"topic={topic} from={env.from_agent}"
                        ),
                    },
                    in_reply_to=env.msg_id,
                )
                log_local.debug(
                    "iter=%d emitted %s in reply to %s seq=%d",
                    iteration, intent_type, env.msg_id, seq,
                )
        if new_envs:
            _write_cursor(cursor, new_envs[-1].seq)

        if args.emit_every and iteration % args.emit_every == 0:
            _emit_intent(
                outbox,
                from_agent=args.agent_name,
                intent_type="send_message",
                payload={
                    "topic": "heartbeat",
                    "body_md": f"mock_agent {args.agent_name} iter={iteration}",
                },
            )
            log_local.debug("iter=%d periodic heartbeat", iteration)

        time.sleep(args.poll_s)


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
