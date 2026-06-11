# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``breakdown.claw_mirror.mirror_breakdown_to_claw_storage``."""

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
    """Pin ``$USER_DATA_PATH`` at ``tmp_path`` so ``workspace_root()`` is writable."""
    monkeypatch.setenv(ENV_USER_DATA_PATH, str(tmp_path))
    return tmp_path


def _make_breakdown(session_dir: Path, payload: dict | None = None) -> Path:
    """Drop a minimal ``session_breakdown.json`` mimicking hyperloom's layout."""
    session_dir.mkdir(parents=True, exist_ok=True)
    bp = session_dir / "session_breakdown.json"
    bp.write_text(
        json.dumps(payload or {"schema_version": "1.1.0", "session": {}}),
        encoding="utf-8",
    )
    return bp


def test_mirrors_to_workspace_hyperloom_subtree(workspace: Path) -> None:
    """Happy path: copy lands at ``hyperloom/<sid>/session_breakdown.json``."""
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
    """Empty/whitespace ``session_id`` must bail out, leaving the canonical file."""
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
    """A missing source must log + return ``None`` rather than raise."""
    bogus = workspace / "model" / "ts" / "session_breakdown.json"

    with caplog.at_level("ERROR"):
        result = mirror_breakdown_to_claw_storage(bogus, session_id="sid42")

    assert result is None
    assert any("claw-mirror failed" in r.message for r in caplog.records)


def test_overwrites_existing_mirror(workspace: Path) -> None:
    """Repeat runs with the same session_id must overwrite the previous mirror."""
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
