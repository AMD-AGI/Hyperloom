# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Persistent round state: one record per step in ``<round_dir>/state.json``, written atomically.

A record moves pending -> running -> done | skipped | failed. A driver that dies mid-step
leaves the record ``running``; the next run treats it as not done and runs the step again.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = "state.json"
LOCK_FILE = ".rsi.lock"
FINISHED = ("done", "skipped")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class StepRecord:
    status: str = "pending"
    started: str = ""
    finished: str = ""
    attempts: int = 0
    error: str = ""
    outputs: dict = field(default_factory=dict)

    def start(self) -> None:
        self.status, self.started, self.finished, self.error = "running", utc_now(), "", ""
        self.attempts += 1

    def finish(self, outputs: dict) -> None:
        self.status = "skipped" if outputs.get("skipped") else "done"
        self.finished, self.outputs = utc_now(), outputs

    def fail(self, error: str) -> None:
        self.status, self.finished, self.error = "failed", utc_now(), error


class RoundState:
    """Step records, the agent spend so far, and free-form data a long step keeps across restarts."""

    def __init__(self, path: Path, steps: dict[str, StepRecord], agent_cost_usd: float = 0.0, data: dict | None = None):
        self.path = path
        self.steps = steps
        self.agent_cost_usd = agent_cost_usd
        self.data = data or {}

    @classmethod
    def load(cls, round_dir: Path) -> RoundState:
        path = round_dir / STATE_FILE
        if not path.exists():
            return cls(path, {})
        raw = json.loads(path.read_text())
        steps = {name: StepRecord(**rec) for name, rec in raw.get("steps", {}).items()}
        return cls(path, steps, float(raw.get("agent_cost_usd", 0.0)), raw.get("data") or {})

    def save(self) -> None:
        payload = {
            "steps": {name: asdict(rec) for name, rec in self.steps.items()},
            "agent_cost_usd": round(self.agent_cost_usd, 6),
            "data": self.data,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    def step(self, name: str) -> StepRecord:
        return self.steps.setdefault(name, StepRecord())


@contextmanager
def round_lock(round_dir: Path) -> Iterator[None]:
    """Hold the round's lock for the duration; exits if another driver already holds it."""
    round_dir.mkdir(parents=True, exist_ok=True)
    path = round_dir / LOCK_FILE
    with open(path, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another rsi driver is running this round (lock {path})") from None
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
