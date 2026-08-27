"""Durable journal for one commit-bound Analysis Agent session."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from kernelforge.durable_io import atomic_write_text


SESSION_SCHEMA_VERSION = 2
MAX_ANALYSIS_SESSION_ATTEMPTS = 2


class AnalysisAttemptLimitError(RuntimeError):
    """Raised when one commit has exhausted its Analysis session attempts."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AnalysisSessionJournal:
    """Persist one Analysis session attempt and its validated outputs."""

    def __init__(
        self,
        root: Path,
        *,
        analysis_commit: str,
        driver_digest: str,
        source_digest: str,
    ) -> None:
        self.root = root.resolve()
        self.path = self.root / "workflow.json"
        self.events_path = self.root / "workflow_events.jsonl"
        self.analysis_commit = analysis_commit
        self.driver_digest = driver_digest
        self.source_digest = source_digest
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_initialize()
        self._recover_interrupted_session()

    @property
    def status(self) -> str:
        return str(self.state["session"]["status"])

    @property
    def attempts(self) -> int:
        return int(self.state["session"].get("attempts", 0))

    @staticmethod
    def _new_session() -> dict[str, Any]:
        return {
            "status": "PENDING",
            "attempts": 0,
            "started_at": "",
            "completed_at": "",
            "outputs": [],
            "output_digests": {},
            "error": "",
        }

    def _load_or_initialize(self) -> dict[str, Any]:
        if self.path.is_file():
            state = json.loads(self.path.read_text())
            expected = (
                state.get("schema_version") == SESSION_SCHEMA_VERSION,
                state.get("analysis_commit") == self.analysis_commit,
                state.get("driver_digest") == self.driver_digest,
                state.get("source_digest") == self.source_digest,
                isinstance(state.get("session"), dict),
            )
            if not all(expected):
                raise ValueError("analysis session inputs do not match durable checkpoint")
            return state

        state = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "analysis_commit": self.analysis_commit,
            "driver_digest": self.driver_digest,
            "source_digest": self.source_digest,
            "status": "RUNNING",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "session": self._new_session(),
        }
        self._write_state(state)
        self._append_event("analysis_session_initialized")
        return state

    def _write_state(self, state: dict[str, Any] | None = None) -> None:
        payload = self.state if state is None else state
        payload["updated_at"] = _utc_now()
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _append_event(self, event_type: str, **fields: Any) -> None:
        event = {
            "ts": _utc_now(),
            "type": event_type,
            **fields,
        }
        with self.events_path.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _recover_interrupted_session(self) -> None:
        session = self.state["session"]
        if session.get("status") != "RUNNING":
            return
        session["status"] = "PENDING"
        session["error"] = "interrupted before completion"
        self._write_state()
        self._append_event("analysis_session_interrupted")

    def begin(self) -> None:
        session = self.state["session"]
        if session["status"] == "COMPLETE":
            return
        if self.attempts >= MAX_ANALYSIS_SESSION_ATTEMPTS:
            raise AnalysisAttemptLimitError(
                "Analysis attempt limit reached for "
                f"{self.analysis_commit}: {self.attempts}/"
                f"{MAX_ANALYSIS_SESSION_ATTEMPTS}"
            )
        session.update(
            {
                "status": "RUNNING",
                "attempts": int(session.get("attempts", 0)) + 1,
                "started_at": _utc_now(),
                "completed_at": "",
                "outputs": [],
                "output_digests": {},
                "error": "",
            }
        )
        self._write_state()
        self._append_event(
            "analysis_session_started",
            attempt=session["attempts"],
        )

    def reopen(self) -> None:
        """Reopen a published PARTIAL session for its remaining attempt."""
        if self.attempts >= MAX_ANALYSIS_SESSION_ATTEMPTS:
            raise AnalysisAttemptLimitError(
                "Analysis attempt limit reached for "
                f"{self.analysis_commit}: {self.attempts}/"
                f"{MAX_ANALYSIS_SESSION_ATTEMPTS}"
            )
        session = self.state["session"]
        session["status"] = "PENDING"
        session["completed_at"] = ""
        session["error"] = ""
        self.state["status"] = "RUNNING"
        self._write_state()
        self._append_event(
            "analysis_session_reopened",
            attempts=session["attempts"],
        )

    def complete(self, outputs: tuple[Path, ...]) -> None:
        relative_outputs = []
        output_digests = {}
        for path in outputs:
            resolved = path.resolve()
            relative = str(resolved.relative_to(self.root))
            relative_outputs.append(relative)
            if resolved.is_file():
                output_digests[relative] = _sha256(resolved)
        session = self.state["session"]
        session.update(
            {
                "status": "COMPLETE",
                "completed_at": _utc_now(),
                "outputs": relative_outputs,
                "output_digests": output_digests,
                "error": "",
            }
        )
        self._write_state()
        self._append_event(
            "analysis_session_completed",
            outputs=relative_outputs,
        )

    def fail(self, error: str) -> None:
        session = self.state["session"]
        session.update(
            {
                "status": "FAILED",
                "completed_at": _utc_now(),
                "error": str(error)[:2000],
            }
        )
        self._write_state()
        self._append_event(
            "analysis_session_failed",
            error=session["error"],
        )

    def finalize(self, status: str) -> None:
        if status not in {"READY", "PARTIAL", "FAILED"}:
            raise ValueError(f"invalid analysis session status: {status}")
        self.state["status"] = status
        self.state["completed_at"] = _utc_now()
        self._write_state()
        self._append_event(
            "analysis_session_finalized",
            status=status,
        )
