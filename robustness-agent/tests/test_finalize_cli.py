"""Tests for the ``finalize`` subcommand of robustness_agent.runtime.cli."""

from __future__ import annotations

import json
from pathlib import Path


from robustness_agent.runtime.cli import main


def test_finalize_creates_postmortem(tmp_path: Path, capsys):
    sd = tmp_path / "sess-123"
    sd.mkdir()
    # Seed a finding so the postmortem has content.
    findings_dir = sd / "agents" / "robustness" / "findings"
    findings_dir.mkdir(parents=True)
    (findings_dir / "sess-123.jsonl").write_text(
        json.dumps({
            "tick_index": 1, "timestamp_unix": 1.0,
            "symptom_name": "x", "severity": "high",
            "summary": "s", "intents": [], "evidence": {},
            "rca_text": "",
        }) + "\n",
        encoding="utf-8",
    )

    rc = main([
        "finalize",
        "--session-dir", str(sd),
        "--stop-reason", "manual_test",
        "--out", "-",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["wrote_new_files"] is True
    assert payload["stop_reason"] == "manual_test"
    assert payload["session_id"] == "sess-123"
    assert (sd / "reports" / "robustness_postmortem.md").is_file()
    assert (sd / "reports" / "decision_trace.json").is_file()
    assert (sd / "reports" / ".robustness_finalized").is_file()


def test_finalize_is_idempotent(tmp_path: Path, capsys):
    sd = tmp_path / "sess-abc"
    sd.mkdir()
    rc1 = main([
        "finalize",
        "--session-dir", str(sd),
        "--out", "-",
    ])
    assert rc1 == 0
    capsys.readouterr()  # drain
    # Second call must not overwrite; ``wrote_new_files=False``.
    rc2 = main([
        "finalize",
        "--session-dir", str(sd),
        "--out", "-",
    ])
    assert rc2 == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["wrote_new_files"] is False


def test_finalize_rejects_missing_session_dir(tmp_path: Path, capsys):
    rc = main([
        "finalize",
        "--session-dir", str(tmp_path / "does_not_exist"),
        "--out", "-",
    ])
    assert rc == 2
    # Error message goes to stderr (RuntimeAdapterError); session_dir
    # check is reported as a clean exit-2.
    captured = capsys.readouterr()
    assert "does not point to a directory" in captured.err


def test_finalize_defaults_session_id_to_dirname(tmp_path: Path, capsys):
    sd = tmp_path / "auto-named"
    sd.mkdir()
    rc = main([
        "finalize",
        "--session-dir", str(sd),
        "--out", "-",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["session_id"] == "auto-named"
    assert payload["stop_reason"] == "manual_finalize"
