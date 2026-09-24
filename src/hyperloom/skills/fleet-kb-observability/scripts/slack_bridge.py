#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Relay Fleet KB operation events into Slack job threads."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any


class BridgeError(RuntimeError):
    pass


def _request_json(
    request: urllib.request.Request,
    *,
    timeout: float = 30,
) -> dict[str, Any]:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise BridgeError("response is not a JSON object")
    return value


def _signal_summary(value: Any) -> str:
    signals = value if isinstance(value, list) else []
    rows: list[str] = []
    for item in signals[:8]:
        if not isinstance(item, dict):
            continue
        target = str(item.get("field") or "free_text")
        raw = item["value"] if "value" in item else item.get("text", "")
        query = str(raw).replace("\n", " ").strip()
        query = f"{query[:93]}..." if len(query) > 96 else query
        try:
            weight = float(item.get("weight") or 0)
        except (TypeError, ValueError):
            weight = 0
        rows.append(f"`{target}`={query!r} ({weight:.2f})")
    return "; ".join(rows) or "none"


class State:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS delivered (
                event_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    def cursor(self) -> int:
        row = self.connection.execute("SELECT value FROM metadata WHERE key = 'cursor'").fetchone()
        return int(row[0]) if row else 0

    def was_delivered(self, event_id: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM delivered WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            is not None
        )

    def commit(self, event_id: str, sequence: int) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO delivered(event_id, sequence)
                VALUES (?, ?)
                """,
                (event_id, sequence),
            )
            self.connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES ('cursor', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(sequence),),
            )


def _format_event(event: dict[str, Any]) -> str | None:
    event_type = str(event.get("event_type") or "")
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    correlation = event.get("correlation") if isinstance(event.get("correlation"), dict) else {}
    worker = str(correlation.get("worker_id") or "?")
    run_id = str(correlation.get("run_id") or "?")
    scope_id = str(correlation.get("scope_id") or payload.get("scope_id") or "?")
    if event_type == "kb.read.started":
        return None
    if event_type == "kb.read.completed":
        refs = [
            str(item.get("id") or "")
            for item in payload.get("rendered_refs", [])
            if isinstance(item, dict) and item.get("id")
        ]
        omitted = [str(item) for item in payload.get("omitted_experience_ids", [])]
        warnings = [str(item) for item in payload.get("warnings", [])]
        return "\n".join(
            (
                (f":books: *KB read completed* · worker `{worker}` · run `{run_id}` · scope `{scope_id}`"),
                f"Signals: {_signal_summary(payload.get('signals'))}",
                f"Eligible: {int(payload.get('eligible_count') or 0)} Experiences",
                f"Found: {int(payload.get('group_count') or 0)} Repeat Groups",
                (f"Injected: {', '.join(f'`{item}`' for item in refs) or 'none'}"),
                f"Omitted: {', '.join(omitted) or 'none'}",
                f"Latency: {float(payload.get('latency_ms') or 0):.0f} ms",
                f"Warnings: {', '.join(warnings) or 'none'}",
            )
        )
    if event_type.startswith("kb.read."):
        return (
            f":warning: *KB read {event_type.rsplit('.', 1)[-1]}* · "
            f"worker `{worker}` · run `{run_id}`\n"
            "Hyperloom continued without injected KB evidence."
        )
    if event_type == "kb.discovery.completed":
        candidates = [item for item in payload.get("candidates", []) if isinstance(item, dict)]
        lines = [
            (
                f"`{item.get('experience_id') or '?'}` · "
                f"{item.get('trust_state') or '?'} · "
                f"source `{item.get('source_scope_id') or '?'}` · "
                f"{item.get('decision') or '?'} · "
                f"{item.get('baseline_value')} → {item.get('outcome_value')} · "
                f"{item.get('change_summary') or '(no change summary)'}"
            )
            for item in candidates[:10]
        ]
        return "\n".join(
            (
                (f":mag: *KB discovery* · target scope `{scope_id}` · {len(candidates)} candidate(s)"),
                f"Question: {payload.get('message') or '(none)'}",
                *(lines or ["No related Experiences found."]),
                ("These are unverified and unavailable to Hyperloom until you explicitly verify them for this run."),
            )
        )
    if event_type in {
        "kb.experiences.selected",
        "kb.experiences.verified_for_scope",
    }:
        rows = payload.get(
            "verifications",
            payload.get("selections", []),
        )
        ids = [
            str(item.get("experience_id") or "")
            for item in rows
            if isinstance(item, dict) and item.get("experience_id")
        ]
        return "\n".join(
            (
                (f":white_check_mark: *Experiences verified for Run* · scope `{scope_id}`"),
                f"Verified by: `{payload.get('actor_id') or '?'}`",
                (f"Experiences: {', '.join(f'`{item}`' for item in ids) or 'none'}"),
                ("Catalog state remains `unverified`; this is a `verified_for_scope` human decision."),
            )
        )
    if event_type == "kb.experience.cataloged":
        return "\n".join(
            (
                (f":floppy_disk: *KB Experience {payload.get('status', 'cataloged')}*"),
                f"Change: {payload.get('change_summary') or '(none)'}",
                f"Decision: `{payload.get('decision') or '?'}`",
                (f"Measurement: {payload.get('baseline_value')} → {payload.get('outcome_value')}"),
                f"Experience: `{payload.get('experience_id') or '?'}`",
                "State: `unverified` · unavailable until scope verification",
            )
        )
    return None


class Bridge:
    def __init__(self, state: State) -> None:
        self.state = state
        self.fleet_url = os.environ["HYPERLOOM_FLEET_KB_URL"].rstrip("/")
        self.fleet_token = os.environ["HYPERLOOM_FLEET_KB_BOT_TOKEN"]
        self.fleet_id = os.environ.get(
            "HYPERLOOM_FLEET_KB_ID",
            "customer-demo",
        )
        self.slack_token = os.environ["SLACK_BOT_TOKEN"]
        self.slack_channel = os.environ["SLACK_CHANNEL_ID"]

    def _events(self) -> list[dict[str, Any]]:
        request = urllib.request.Request(
            (f"{self.fleet_url}/v1/events?after={self.state.cursor()}&limit=100"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.fleet_token}",
                "X-Hyperloom-Fleet-ID": self.fleet_id,
            },
        )
        value = _request_json(request)
        events = value.get("events")
        if not isinstance(events, list):
            raise BridgeError("Fleet KB events response has no events list")
        return [item for item in events if isinstance(item, dict)]

    def _send(self, event: dict[str, Any], text: str) -> None:
        correlation = event.get("correlation") if isinstance(event.get("correlation"), dict) else {}
        body: dict[str, Any] = {
            "channel": self.slack_channel,
            "text": text,
            "client_msg_id": str(event["event_id"]),
            "unfurl_links": False,
            "unfurl_media": False,
        }
        thread_id = str(correlation.get("thread_id") or "")
        if thread_id:
            body["thread_ts"] = thread_id
        response = _request_json(
            urllib.request.Request(
                "https://slack.com/api/chat.postMessage",
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": f"Bearer {self.slack_token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
        )
        if response.get("ok") is not True:
            raise BridgeError(f"Slack rejected event {event['event_id']}: {response.get('error')}")

    def run_once(self) -> int:
        delivered = 0
        for event in self._events():
            event_id = str(event.get("event_id") or "")
            sequence = int(event.get("sequence") or 0)
            if not event_id or sequence <= 0:
                raise BridgeError("Fleet KB event identity is invalid")
            if self.state.was_delivered(event_id):
                self.state.commit(event_id, sequence)
                continue
            text = _format_event(event)
            if text is not None:
                self._send(event, text)
                delivered += 1
            self.state.commit(event_id, sequence)
        return delivered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("~/.local/state/hyperloom/fleet-kb-slack.sqlite3").expanduser(),
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    bridge = Bridge(State(args.state))
    while True:
        bridge.run_once()
        if args.once:
            return 0
        time.sleep(max(0.2, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
