"""Tests for the ``coverage`` field on ``session_breakdown.json``.

The exporter classifies every build into one of three coverage levels
based on whether ``state.json`` and ``manifest.json`` were both present
at build time:

* ``full``       — both present
* ``partial``    — exactly one present
* ``shell_only`` — neither present (post-orchestrator output dirs)

For ``shell_only`` builds the exporter additionally collapses all
collector-emitted warnings into a single ``coverage:`` marker line so the
JSON stays low-noise. These tests pin that contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.breakdown import build


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_coverage_full_when_state_and_manifest_present(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "s1"})
    _write_json(sd / "state.json", {"session_id": "s1", "baseline_tput": 100.0})

    b = build(sd)

    assert b["coverage"] == "full"
    assert not any("state.json missing" in w for w in b["warnings"])
    assert not any("manifest.json missing" in w for w in b["warnings"])
    assert not any(w.startswith("coverage:") for w in b["warnings"])


def test_coverage_partial_when_only_state_present(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _write_json(sd / "state.json", {"session_id": "s2", "baseline_tput": 50.0})

    b = build(sd)

    assert b["coverage"] == "partial"
    assert any("manifest.json missing" in w for w in b["warnings"])
    assert not any("state.json missing" in w for w in b["warnings"])


def test_coverage_partial_when_only_manifest_present(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "s3"})

    b = build(sd)

    assert b["coverage"] == "partial"
    assert any("state.json missing" in w for w in b["warnings"])
    assert not any("manifest.json missing" in w for w in b["warnings"])


def test_coverage_shell_only_when_neither_present(tmp_path: Path) -> None:
    """For post-orchestrator output directories (no session state) the
    exporter must emit ``coverage: shell_only`` and collapse the
    cascade of follow-on warnings into a single marker line so the JSON
    stays scannable. Without this consolidation each shell-only dir
    would emit ~4 unrelated warnings (state missing, manifest missing,
    image not configured, framework_args extraction failed)."""
    sd = tmp_path / "session"
    sd.mkdir()
    (sd / "orchestrator_final.json").write_text("{}", encoding="utf-8")

    b = build(sd)

    assert b["coverage"] == "shell_only"
    # Exactly one warning, and it MUST be the coverage marker.
    assert len(b["warnings"]) == 1, b["warnings"]
    assert b["warnings"][0].startswith("coverage: shell_only")
    # And specifically — the two missing-file warnings that previously
    # showed up here MUST have been consolidated away.
    assert not any("state.json missing" in w for w in b["warnings"])
    assert not any("manifest.json missing" in w for w in b["warnings"])
