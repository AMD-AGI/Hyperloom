"""CLI tests for dump_session_breakdown --detail-level."""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.scripts.dump_session_breakdown import main


def test_dump_session_breakdown_detail_level_verbose(
    tmp_path: Path, capsys,
) -> None:
    sd = tmp_path / "session"
    sd.mkdir()
    (sd / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "session_id": "cli-dj"}),
        encoding="utf-8",
    )
    (sd / "state.json").write_text(
        json.dumps({"session_id": "cli-dj", "baseline_tput": 100.0}),
        encoding="utf-8",
    )
    rc = main([
        "--session-dir", str(sd),
        "--dry-run",
        "--detail-level", "verbose",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "detail_level=verbose" in out
