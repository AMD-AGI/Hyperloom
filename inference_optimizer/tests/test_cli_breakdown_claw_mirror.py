"""Unit tests for ``breakdown.claw_mirror``.

Hyperloom writes ``session_breakdown.json`` + ``manifest.json`` into
``session_dir`` (which under the default ``per_model_ts`` layout is
``<USER_DATA_PATH>/<model>/<ts>/``). Two best-effort mirrors keep the
claw pipeline whole:

1. :func:`mirror_breakdown_to_claw_storage` copies the breakdown to
   ``<workspace_root>/hyperloom/<sid>/`` (the original
   ``$USER_DATA_PATH``-relative checkpoint-synced subtree).
2. :func:`mirror_session_artifacts_to_claw_storage` ADDITIONALLY copies
   ``manifest.json`` + ``session_breakdown.json`` to the claw S3-sync ROOT
   (``/workspace``), so primus-claw uploads them to the deterministic root
   key the collector reads
   (``users/<uid>/sessions/<csid>/{manifest,session_breakdown}.json``).

The mirrors live in their own module so this file needn't drag in cli's
``orchestrator/action_executors`` graph (which imports POSIX-only modules).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown.claw_mirror import (
    ENV_CLAW_SESSION_ID,
    ENV_CLAW_WORKSPACE,
    claw_sync_root,
    mirror_breakdown_to_claw_storage,
    mirror_session_artifacts_to_claw_storage,
)
from inference_optimizer.paths import ENV_USER_DATA_PATH


# ---------------------------------------------------------------------------
# 1. Original USER_DATA_PATH/hyperloom/<sid> mirror
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin ``$USER_DATA_PATH`` at ``tmp_path`` so ``workspace_root()``
    returns a writable scratch dir for the duration of the test."""
    monkeypatch.setenv(ENV_USER_DATA_PATH, str(tmp_path))
    return tmp_path


def _make_breakdown(session_dir: Path, payload: dict | None = None) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    bp = session_dir / "session_breakdown.json"
    bp.write_text(
        json.dumps(payload or {"schema_version": "1.1.0", "session": {}}),
        encoding="utf-8",
    )
    return bp


def test_mirrors_to_workspace_hyperloom_subtree(workspace: Path) -> None:
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


def test_empty_session_id_is_noop(workspace: Path, caplog) -> None:
    canonical = _make_breakdown(workspace / "model" / "ts")
    with caplog.at_level("WARNING"):
        result = mirror_breakdown_to_claw_storage(canonical, session_id="")
    assert result is None
    assert not (workspace / "hyperloom").exists()
    assert any("empty session_id" in r.message for r in caplog.records)


def test_missing_source_is_swallowed(workspace: Path, caplog) -> None:
    bogus = workspace / "model" / "ts" / "session_breakdown.json"
    with caplog.at_level("ERROR"):
        result = mirror_breakdown_to_claw_storage(bogus, session_id="sid42")
    assert result is None
    assert any("claw-mirror failed" in r.message for r in caplog.records)


def test_overwrites_existing_mirror(workspace: Path) -> None:
    sid = "resumed-sid"
    canonical = _make_breakdown(
        workspace / "m" / "ts1",
        payload={"schema_version": "1.1.0", "session": {"tick": 1}},
    )
    mirror_breakdown_to_claw_storage(canonical, session_id=sid)
    canonical2 = _make_breakdown(
        workspace / "m" / "ts2",
        payload={"schema_version": "1.1.0", "session": {"tick": 99}},
    )
    mirror = mirror_breakdown_to_claw_storage(canonical2, session_id=sid)
    assert mirror is not None
    payload = json.loads(mirror.read_text(encoding="utf-8"))
    assert payload["session"]["tick"] == 99


# ---------------------------------------------------------------------------
# 2. Redundant claw S3-sync ROOT mirror (manifest + breakdown)
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Simulate a claw sandbox: a ``/workspace`` sync root + a session dir
    OUTSIDE it (mimicking USER_DATA_PATH on a separate wekafs mount)."""
    sync_root = tmp_path / "workspace"
    sync_root.mkdir()
    session_dir = tmp_path / "wekafs" / "users" / "u1" / "model" / "ts"
    session_dir.mkdir(parents=True)
    monkeypatch.setenv(ENV_CLAW_SESSION_ID, "csid-1")
    monkeypatch.setenv(ENV_CLAW_WORKSPACE, str(sync_root))
    return {"sync_root": sync_root, "session_dir": session_dir}


def _seed(session_dir: Path, *, breakdown: dict | None = None, manifest: dict | None = None) -> None:
    if breakdown is not None:
        (session_dir / "session_breakdown.json").write_text(
            json.dumps(breakdown), encoding="utf-8"
        )
    if manifest is not None:
        (session_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )


def test_root_mirror_breakdown_and_manifest(sandbox) -> None:
    sd, root = sandbox["session_dir"], sandbox["sync_root"]
    _seed(
        sd,
        breakdown={"schema_version": "1.1.0", "session": {"session_id": "s"}},
        manifest={"claw_session_id": "csid-1", "session_id": "s"},
    )
    written = mirror_session_artifacts_to_claw_storage(sd)
    assert sorted(p.name for p in written) == ["manifest.json", "session_breakdown.json"]
    assert (root / "session_breakdown.json").is_file()
    assert (root / "manifest.json").is_file()
    assert (root / "session_breakdown.json").read_bytes() == (
        sd / "session_breakdown.json"
    ).read_bytes()


def test_root_mirror_noop_outside_claw_sandbox(sandbox, monkeypatch) -> None:
    monkeypatch.delenv(ENV_CLAW_SESSION_ID, raising=False)
    _seed(sandbox["session_dir"], breakdown={"x": 1}, manifest={"y": 2})
    assert claw_sync_root() is None
    assert mirror_session_artifacts_to_claw_storage(sandbox["session_dir"]) == []
    assert not (sandbox["sync_root"] / "session_breakdown.json").exists()


def test_root_mirror_noop_when_sync_root_missing(sandbox, monkeypatch) -> None:
    monkeypatch.setenv(ENV_CLAW_WORKSPACE, str(sandbox["sync_root"] / "nope"))
    _seed(sandbox["session_dir"], breakdown={"x": 1})
    assert claw_sync_root() is None
    assert mirror_session_artifacts_to_claw_storage(sandbox["session_dir"]) == []


def test_root_mirror_skips_missing_source_files(sandbox) -> None:
    _seed(sandbox["session_dir"], breakdown={"only": "breakdown"})
    written = mirror_session_artifacts_to_claw_storage(sandbox["session_dir"])
    assert [p.name for p in written] == ["session_breakdown.json"]


def test_root_mirror_nothing_to_mirror_returns_empty(sandbox) -> None:
    assert mirror_session_artifacts_to_claw_storage(sandbox["session_dir"]) == []


def test_root_mirror_overwrites_with_freshest(sandbox) -> None:
    sd, root = sandbox["session_dir"], sandbox["sync_root"]
    _seed(sd, breakdown={"tick": 1})
    mirror_session_artifacts_to_claw_storage(sd)
    _seed(sd, breakdown={"tick": 99})
    mirror_session_artifacts_to_claw_storage(sd)
    payload = json.loads((root / "session_breakdown.json").read_text(encoding="utf-8"))
    assert payload["tick"] == 99
