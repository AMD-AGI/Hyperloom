# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``breakdown.claw_mirror.mirror_breakdown_to_claw_storage``.

The hyperloom CLI writes ``session_breakdown.json`` into ``session_dir``
(which under the default ``per_model_ts`` layout is
``<workspace>/<model>/<ts>/``). The claw sandbox checkpoint sync only
watches the ``hyperloom/`` subtree under ``$USER_DATA_PATH``; without
the mirror tested here the canonical breakdown is lost when the sandbox
is reaped, breaking ``claw-stats-service`` historical lookups.

The mirror lives in its own module (rather than as a private helper on
``cli.py``) precisely so this test file can import it without dragging
in cli's ``orchestrator/action_executors`` graph (which imports
``fcntl`` and other POSIX-only modules).

These tests exercise the mirror helper in isolation:

* happy-path mirror lands at ``<workspace_root>/hyperloom/<sid>/...``;
* empty ``session_id`` is a no-op (warns, returns ``None``);
* unreadable source / unwritable dest is a no-op (logs, returns ``None``)
  — the canonical write path is the source of truth and MUST NOT be
  broken by mirror failures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown.claw_mirror import (
    mirror_breakdown_to_claw_storage,
)
from inference_optimizer.paths import ENV_USER_DATA_PATH


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin ``$USER_DATA_PATH`` at ``tmp_path`` so ``workspace_root()``
    returns a writable scratch dir for the duration of the test."""
    monkeypatch.setenv(ENV_USER_DATA_PATH, str(tmp_path))
    return tmp_path


def _make_breakdown(session_dir: Path, payload: dict | None = None) -> Path:
    """Drop a minimal ``session_breakdown.json`` under a per-model-ts
    subdir of the workspace, mimicking hyperloom's canonical layout."""
    session_dir.mkdir(parents=True, exist_ok=True)
    bp = session_dir / "session_breakdown.json"
    bp.write_text(
        json.dumps(payload or {"schema_version": "1.1.0", "session": {}}),
        encoding="utf-8",
    )
    return bp


def test_mirrors_to_workspace_hyperloom_subtree(workspace: Path) -> None:
    """Happy path: copy lands at ``hyperloom/<sid>/session_breakdown.json``
    with identical bytes — that's the path the claw sync watches."""
    sid = "abc123"
    canonical = _make_breakdown(
        workspace / "Llama-3.1-8B" / "20260525T040000Z",
        payload={"schema_version": "1.1.0", "session": {"session_id": sid}},
    )

    mirror = mirror_breakdown_to_claw_storage(canonical, session_id=sid)

    expected = workspace / "hyperloom" / sid / "session_breakdown.json"
    assert mirror == expected
    assert mirror.is_file()
    assert mirror.read_bytes() == canonical.read_bytes()


def test_empty_session_id_is_noop(
    workspace: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty/whitespace ``session_id`` would collapse the mirror dir into
    ``hyperloom/`` itself, which is wrong; the helper must bail out and
    leave the canonical file untouched."""
    canonical = _make_breakdown(workspace / "model" / "ts")

    with caplog.at_level("WARNING"):
        result = mirror_breakdown_to_claw_storage(canonical, session_id="")

    assert result is None
    assert not (workspace / "hyperloom").exists()
    assert any("empty session_id" in r.message for r in caplog.records)


def test_missing_source_is_swallowed(
    workspace: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the canonical write didn't actually produce a file (caller
    bug, disk full, etc.) the mirror must log + return ``None`` rather
    than raise — the finally block in ``_run_optimize`` relies on this
    to keep ``stop_reason`` propagation intact."""
    bogus = workspace / "model" / "ts" / "session_breakdown.json"

    with caplog.at_level("ERROR"):
        result = mirror_breakdown_to_claw_storage(bogus, session_id="sid42")

    assert result is None
    assert any("claw-mirror failed" in r.message for r in caplog.records)


def test_overwrites_existing_mirror(workspace: Path) -> None:
    """Repeat runs with the same session_id (e.g. ``--resume``) must
    overwrite the previous mirror — otherwise downstream consumers
    would see a stale early snapshot."""
    sid = "resumed-sid"
    canonical = _make_breakdown(
        workspace / "m" / "ts1",
        payload={"schema_version": "1.1.0", "session": {"tick": 1}},
    )
    mirror_breakdown_to_claw_storage(canonical, session_id=sid)

    # Second run, same sid, fresher payload.
    canonical2 = _make_breakdown(
        workspace / "m" / "ts2",
        payload={"schema_version": "1.1.0", "session": {"tick": 99}},
    )
    mirror = mirror_breakdown_to_claw_storage(canonical2, session_id=sid)

    assert mirror is not None
    payload = json.loads(mirror.read_text(encoding="utf-8"))
    assert payload["session"]["tick"] == 99
