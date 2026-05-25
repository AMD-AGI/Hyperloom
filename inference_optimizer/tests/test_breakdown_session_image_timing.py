"""Tests for session.image / session.image_id / session.image_digest and
session_started_at_utc / session_ended_at_utc / session_duration_seconds.

Covers the multi-source image probe (state.json → manifest.json →
runs/baseline yamls → env) and the real-session lifecycle timestamps
(state.start_ts → manifest.created_at_utc → phase_timeline derive;
state.closing_started_unix → state.stopped_at → phase_timeline derive).

Also verifies:

* The closing phase event picks up a non-None ``duration_seconds`` once
  ``session_ended_at_utc`` is known.
* ``data_provenance.session`` enumerates every image / timing candidate
  source it consulted (including the four image probes and three env
  vars) so an operator can see exactly which paths were tried.
* The provenance ``notes`` field tells the consumer when image / timing
  was missing everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown import build
from inference_optimizer.breakdown.collectors import (
    _extract_image_info,
    _extract_session_timing,
    collect_session,
    enrich_session_and_timeline,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _entries_by_section(prov: list[dict]) -> dict[str, dict]:
    return {e["section"]: e for e in prov if isinstance(e, dict)}


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------
def test_session_image_from_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``state.json`` carries ``image``, it must take precedence
    over every other source (including the env-var fallback)."""
    for name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(name, raising=False)
    sd = tmp_path / "sess"
    _write_json(sd / "manifest.json", {
        "schema_version": 1,
        "session_id": "img-state",
        "image": "should-not-win",
    })
    _write_json(sd / "state.json", {
        "session_id": "img-state",
        "image": "rocm/sglang:6.4",
        "image_id": "img-abc123",
        "image_digest": "sha256:deadbeef" + "0" * 56,
        "start_ts": "2026-05-21T16:01:54+00:00",
    })

    b = build(sd)
    sess = b["session"]
    assert sess["image"] == "rocm/sglang:6.4"
    assert sess["image_id"] == "img-abc123"
    assert sess["image_digest"] == "sha256:deadbeef" + "0" * 56


def test_session_image_from_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When state has no image but manifest does, manifest wins."""
    for name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(name, raising=False)
    sd = tmp_path / "sess"
    _write_json(sd / "manifest.json", {
        "schema_version": 1,
        "session_id": "img-mfst",
        "image": "rocm/vllm:7.0",
        "image_digest": "sha256:beadface" + "0" * 56,
    })
    _write_json(sd / "state.json", {
        "session_id": "img-mfst",
        "start_ts": "2026-05-21T16:01:54+00:00",
    })

    b = build(sd)
    sess = b["session"]
    assert sess["image"] == "rocm/vllm:7.0"
    assert sess["image_digest"] == "sha256:beadface" + "0" * 56
    assert sess["image_id"] is None


def test_session_image_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When neither state nor manifest carry an image, fall through to
    ``HYPERLOOM_IMAGE``."""
    monkeypatch.setenv("HYPERLOOM_IMAGE", "registry.example/sgl:env-only")
    monkeypatch.delenv("CONTAINER_IMAGE", raising=False)
    monkeypatch.delenv("IMAGE", raising=False)
    sd = tmp_path / "sess"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "img-env"})
    _write_json(sd / "state.json", {
        "session_id": "img-env",
        "start_ts": "2026-05-21T16:01:54+00:00",
    })

    b = build(sd)
    sess = b["session"]
    assert sess["image"] == "registry.example/sgl:env-only"


def test_session_image_from_baseline_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When all higher-priority sources are empty but the baseline yaml
    carries ``docker_image`` (the magpie benchmark config alias for
    image), the helper must pick it up."""
    for name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(name, raising=False)
    sd = tmp_path / "sess"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "img-yaml"})
    _write_json(sd / "state.json", {
        "session_id": "img-yaml",
        "start_ts": "2026-05-21T16:01:54+00:00",
    })
    yaml = sd / "runs/baseline/abc/baseline_config.with_envs.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml.write_text(
        "benchmark:\n"
        "  framework: sglang\n"
        "docker_image: rocm/sglang:6.5-yaml\n",
        encoding="utf-8",
    )

    info = _extract_image_info({}, {}, sd)
    assert info["image"] == "rocm/sglang:6.5-yaml"

    b = build(sd)
    assert b["session"]["image"] == "rocm/sglang:6.5-yaml"


def test_session_image_missing_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the image isn't in any source, ``session.image`` is None,
    a single warning is appended, and ``data_provenance.session.notes``
    explains the absence + ``sources`` lists every candidate we probed."""
    for name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(name, raising=False)
    sd = tmp_path / "sess"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "img-none"})
    _write_json(sd / "state.json", {
        "session_id": "img-none",
        "start_ts": "2026-05-21T16:01:54+00:00",
    })

    b = build(sd)
    sess = b["session"]
    assert sess["image"] is None
    assert sess["image_id"] is None
    assert sess["image_digest"] is None
    assert any("image: not configured" in w for w in b["warnings"]), b["warnings"]

    prov_sess = _entries_by_section(b["data_provenance"])["session"]
    # All four image candidate probes + 3 env probes must be present
    # so the operator sees exactly which sources we tried.
    src_roles = [s.get("role") for s in prov_sess["sources"]]
    assert any("state.json: image" in (r or "") for r in src_roles), src_roles
    assert any("manifest.json: image" in (r or "") for r in src_roles), src_roles
    assert any("baseline_config.with_envs.yaml" in (r or "") for r in src_roles), src_roles
    assert any("benchmark config.yaml" in (r or "") for r in src_roles), src_roles
    for env_name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        assert any(env_name in (r or "") for r in src_roles), (env_name, src_roles)
    assert any("no image metadata found" in n for n in prov_sess["notes"]), prov_sess["notes"]


# ---------------------------------------------------------------------------
# Session timing extraction
# ---------------------------------------------------------------------------
def test_session_timing_from_state_started_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When state has start_ts + stopped_at (or closing_started_unix),
    session_started/ended/duration must be populated and consistent."""
    for name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(name, raising=False)
    sd = tmp_path / "sess"
    _write_json(sd / "manifest.json", {
        "schema_version": 1, "session_id": "t1",
        "created_at_utc": "2026-05-21T16:01:54+00:00",
    })
    _write_json(sd / "state.json", {
        "session_id": "t1",
        "start_ts":   "2026-05-21T16:01:54+00:00",
        "stopped_at": "2026-05-21T19:12:03+00:00",
    })

    b = build(sd)
    sess = b["session"]
    assert sess["session_started_at_utc"] == "2026-05-21T16:01:54+00:00"
    assert sess["session_ended_at_utc"]   == "2026-05-21T19:12:03+00:00"
    # 3h10m9s = 11409s
    assert sess["session_duration_seconds"] == pytest.approx(11409.0, abs=1.0)
    # elapsed_minutes mirrors the real duration
    assert sess["elapsed_minutes"] == pytest.approx(11409.0 / 60.0, abs=0.05)


def test_session_timing_closing_started_unix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``closing_started_unix`` (an epoch float) is the most common
    real-session shutdown signal and must populate session_ended."""
    for name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(name, raising=False)
    sd = tmp_path / "sess"
    _write_json(sd / "manifest.json", {
        "schema_version": 1, "session_id": "t2",
        "created_at_utc": "2026-05-21T16:01:54+00:00",
    })
    _write_json(sd / "state.json", {
        "session_id": "t2",
        "start_ts": "2026-05-21T16:01:54+00:00",
        "closing_started_unix": 1779388323.0,  # = 2026-05-21T19:12:03Z
    })

    b = build(sd)
    sess = b["session"]
    assert sess["session_started_at_utc"] == "2026-05-21T16:01:54+00:00"
    assert sess["session_ended_at_utc"] is not None
    assert sess["session_duration_seconds"] is not None
    assert sess["session_duration_seconds"] > 0


def test_session_timing_derived_from_phase_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When state lacks any direct timestamp but phase_timeline carries
    events, the cross-section reconciler fills started_at / ended_at
    from the first / last event timestamp."""
    for name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(name, raising=False)

    # Simulate phase_timeline manually since wiring it through build()
    # for a fully-empty state.json is awkward.
    session_meta = {
        "session_started_at_utc": None,
        "session_ended_at_utc":   None,
        "session_duration_seconds": None,
        "elapsed_minutes": 0.0,
    }
    phase_timeline = [
        {"ts": "2026-05-22T10:00:00+00:00", "action": "baseline",
         "duration_seconds": 60.0,
         "ended_ts_utc": "2026-05-22T10:01:00+00:00"},
        {"ts": "2026-05-22T11:30:00+00:00", "action": "validate_stack",
         "duration_seconds": 120.0,
         "ended_ts_utc": "2026-05-22T11:32:00+00:00"},
    ]
    enrich_session_and_timeline(session_meta, phase_timeline, state={})

    assert session_meta["session_started_at_utc"] == "2026-05-22T10:00:00+00:00"
    assert session_meta["session_ended_at_utc"]   == "2026-05-22T11:32:00+00:00"
    # 1h32m = 5520s
    assert session_meta["session_duration_seconds"] == pytest.approx(5520.0, abs=1.0)


def test_session_timing_closing_duration_filled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a closing event has no duration_seconds but
    ``session_ended_at_utc`` is known, the post-processor fills it
    with ``session_end - closing.ts``."""
    for name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(name, raising=False)

    session_meta = {
        "session_started_at_utc": "2026-05-21T16:01:54+00:00",
        "session_ended_at_utc":   "2026-05-21T19:12:03+00:00",
        "session_duration_seconds": 11409.0,
        "elapsed_minutes": 190.15,
    }
    phase_timeline = [
        {"ts": "2026-05-21T16:01:54+00:00", "action": "baseline",
         "duration_seconds": 60.0},
        {"ts": "2026-05-21T19:00:00+00:00", "action": "closing",
         "duration_seconds": None, "ended_ts_utc": None},
    ]
    enrich_session_and_timeline(session_meta, phase_timeline, state={})

    closing = phase_timeline[-1]
    # 12m3s = 723s
    assert closing["duration_seconds"] == pytest.approx(723.0, abs=1.0)
    assert closing["ended_ts_utc"] == "2026-05-21T19:12:03+00:00"


def test_session_timing_provenance_notes_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither start nor end timestamp can be resolved,
    ``data_provenance.session.notes`` must say so explicitly."""
    for name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(name, raising=False)
    sd = tmp_path / "sess"
    # Manifest with no created_at_utc, state with no start_ts.
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "t3"})
    _write_json(sd / "state.json", {"session_id": "t3"})

    b = build(sd)
    sess = b["session"]
    # Neither endpoint resolved → both None.
    assert sess["session_started_at_utc"] is None
    assert sess["session_ended_at_utc"]   is None
    assert sess["session_duration_seconds"] is None

    prov_sess = _entries_by_section(b["data_provenance"])["session"]
    assert any(
        "session timing: no startup/shutdown timestamps" in n
        for n in prov_sess["notes"]
    ), prov_sess["notes"]


# ---------------------------------------------------------------------------
# Helper-level unit checks
# ---------------------------------------------------------------------------
def test_extract_image_info_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """state > manifest > baseline yaml > env, demonstrated explicitly."""
    monkeypatch.setenv("HYPERLOOM_IMAGE", "env-img")
    sd = tmp_path / "x"
    sd.mkdir()
    yaml = sd / "runs/baseline/k/baseline_config.with_envs.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml.write_text("docker_image: yaml-img\n", encoding="utf-8")

    # env-only.
    assert _extract_image_info({}, {}, None)["image"] == "env-img"
    # yaml wins over env.
    assert _extract_image_info({}, {}, sd)["image"] == "yaml-img"
    # manifest wins over yaml + env.
    assert _extract_image_info({}, {"image": "mfst-img"}, sd)["image"] == "mfst-img"
    # state wins over manifest + yaml + env.
    out = _extract_image_info({"image": "state-img"}, {"image": "mfst-img"}, sd)
    assert out["image"] == "state-img"


def test_extract_session_timing_helper() -> None:
    """Direct helper test with a populated state.json shape."""
    state = {
        "start_ts":   "2026-05-22T10:00:00+00:00",
        "stopped_at": "2026-05-22T12:00:00+00:00",
    }
    out = _extract_session_timing(state, {}, phase_timeline=None)
    assert out["session_started_at_utc"] == "2026-05-22T10:00:00+00:00"
    assert out["session_ended_at_utc"]   == "2026-05-22T12:00:00+00:00"
    assert out["session_duration_seconds"] == pytest.approx(7200.0, abs=1.0)


def test_collect_session_emits_all_new_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``collect_session`` must always return the five new keys, even
    when every underlying source is empty (we want explicit None
    rather than missing keys so consumers don't have to do
    ``.get(...)`` everywhere)."""
    for name in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        monkeypatch.delenv(name, raising=False)
    sd = tmp_path / "x"
    sd.mkdir()
    warnings: list[str] = []
    out = collect_session(sd, state={}, manifest={}, warnings=warnings)
    for key in ("image", "image_id", "image_digest",
                "session_started_at_utc", "session_ended_at_utc",
                "session_duration_seconds"):
        assert key in out, key
    assert out["image"] is None
    assert out["image_id"] is None
    assert out["image_digest"] is None
    assert out["session_started_at_utc"] is None
    assert out["session_ended_at_utc"] is None
    assert out["session_duration_seconds"] is None
