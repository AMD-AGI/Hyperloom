# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ci/progress.py (promote / list-remaining / stats)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import progress  # noqa: E402


# ── _load_json ──


def test_load_json_missing_returns_empty(tmp_path: Path):
    assert progress._load_json(tmp_path / "nope.json") == {}


def test_load_json_valid(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert progress._load_json(p) == {"x": 1}


def test_load_json_invalid_returns_empty(tmp_path: Path, capsys):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert progress._load_json(p) == {}
    assert "could not parse" in capsys.readouterr().err


# ── _summary_rows ──


def test_summary_rows_bare_list(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"model": "m"}]), encoding="utf-8")
    assert progress._summary_rows(p) == [{"model": "m"}]


def test_summary_rows_rows_key(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"rows": [{"model": "a"}]}), encoding="utf-8")
    assert progress._summary_rows(p) == [{"model": "a"}]


def test_summary_rows_models_key(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"models": [{"model": "b"}]}), encoding="utf-8")
    assert progress._summary_rows(p) == [{"model": "b"}]


# ── _classify_status ──


def test_classify_completed():
    row = {"baseline_tok_per_gpu": 1, "optimized_tok_per_gpu": 2, "gain_pct": 10}
    assert progress._classify_status(row) == ("completed", None)


def test_classify_partial():
    status, reason = progress._classify_status({"optimized_tok_per_gpu": 2})
    assert status == "partial"
    assert "single-data point" in reason


def test_classify_failed_final_status():
    status, reason = progress._classify_status({"final_status": "Failed"})
    assert status == "failed"
    assert "final_status=Failed" in reason


def test_classify_failed_submit_status():
    status, reason = progress._classify_status({"submit_status": "error"})
    assert status == "failed"
    assert "submit_status=error" in reason


def test_classify_failed_no_data():
    assert progress._classify_status({}) == ("failed", "no data produced")


# ── _row_to_entry ──


def test_row_to_entry_full():
    row = {
        "model": "org/m",
        "framework": "sglang",
        "precision": "fp8",
        "tp": 8,
        "params_b": 7,
        "gain_pct": 12.345,
        "vs_inferenceX_pct": 3.21,
        "task_id": "t1",
        "baseline_tok_per_gpu": 1,
        "optimized_tok_per_gpu": 2,
    }
    e = progress._row_to_entry(row)
    assert e["repo_id"] == "org/m"
    assert e["status"] == "completed"
    assert e["gain_pct"] == 12.35
    assert e["vs_infx_pct"] == 3.21
    assert e["task_id"] == "t1"


def test_row_to_entry_failed_has_reason():
    e = progress._row_to_entry({"model": "m"})
    assert e["status"] == "failed"
    assert "reason" in e


# ── cmd_promote ──


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_promote_empty_summary(tmp_path: Path):
    s = tmp_path / "sum.json"
    s.write_text(json.dumps([]), encoding="utf-8")
    rc = progress.cmd_promote(_ns(summary=str(s), already_done=str(tmp_path / "ad.json"), write=False))
    assert rc == 1


def test_promote_dry_run(tmp_path: Path, capsys):
    s = tmp_path / "sum.json"
    s.write_text(
        json.dumps(
            [
                {"model": "org/m1", "baseline_tok_per_gpu": 1, "optimized_tok_per_gpu": 2, "gain_pct": 5},
            ]
        ),
        encoding="utf-8",
    )
    ad = tmp_path / "ad.json"
    rc = progress.cmd_promote(_ns(summary=str(s), already_done=str(ad), write=False))
    assert rc == 0
    assert "dry-run" in capsys.readouterr().out
    assert not ad.exists()


def test_promote_write_creates_file(tmp_path: Path):
    s = tmp_path / "sum.json"
    s.write_text(
        json.dumps(
            [
                {"model": "org/m1", "baseline_tok_per_gpu": 1, "optimized_tok_per_gpu": 2, "gain_pct": 5},
                {"model": "org/m2"},
            ]
        ),
        encoding="utf-8",
    )
    ad = tmp_path / "sub" / "ad.json"
    rc = progress.cmd_promote(_ns(summary=str(s), already_done=str(ad), write=True))
    assert rc == 0
    data = json.loads(ad.read_text(encoding="utf-8"))
    assert data["_meta"]["count"] == 2
    repos = {m["repo_id"] for m in data["models"]}
    assert repos == {"org/m1", "org/m2"}


def test_promote_upgrades_not_downgrades(tmp_path: Path):
    ad = tmp_path / "ad.json"
    ad.write_text(json.dumps({"models": [{"repo_id": "org/m", "status": "failed"}]}), encoding="utf-8")
    s = tmp_path / "sum.json"
    s.write_text(
        json.dumps(
            [
                {"model": "org/m", "baseline_tok_per_gpu": 1, "optimized_tok_per_gpu": 2, "gain_pct": 5},
            ]
        ),
        encoding="utf-8",
    )
    rc = progress.cmd_promote(_ns(summary=str(s), already_done=str(ad), write=True))
    assert rc == 0
    data = json.loads(ad.read_text(encoding="utf-8"))
    assert data["models"][0]["status"] == "completed"


def test_promote_nothing_to_do(tmp_path: Path, capsys):
    ad = tmp_path / "ad.json"
    ad.write_text(json.dumps({"models": [{"repo_id": "org/m", "status": "completed"}]}), encoding="utf-8")
    s = tmp_path / "sum.json"
    # Same completed status -> no upgrade, no new entry.
    s.write_text(
        json.dumps(
            [
                {"model": "org/m", "baseline_tok_per_gpu": 1, "optimized_tok_per_gpu": 2, "gain_pct": 5},
            ]
        ),
        encoding="utf-8",
    )
    rc = progress.cmd_promote(_ns(summary=str(s), already_done=str(ad), write=True))
    assert rc == 0
    assert "nothing to promote" in capsys.readouterr().out


def test_promote_many_new_entries_truncated(tmp_path: Path, capsys):
    rows = [
        {"model": f"org/m{i}", "baseline_tok_per_gpu": 1, "optimized_tok_per_gpu": 2, "gain_pct": 5} for i in range(20)
    ]
    s = tmp_path / "sum.json"
    s.write_text(json.dumps(rows), encoding="utf-8")
    rc = progress.cmd_promote(_ns(summary=str(s), already_done=str(tmp_path / "ad.json"), write=False))
    assert rc == 0
    assert "and 5 more" in capsys.readouterr().out


# ── cmd_list_remaining ──


def test_list_remaining_empty_candidates(tmp_path: Path):
    c = tmp_path / "c.json"
    c.write_text(json.dumps({"candidates": []}), encoding="utf-8")
    rc = progress.cmd_list_remaining(_ns(candidates=str(c), already_done=str(tmp_path / "ad.json")))
    assert rc == 1


def test_list_remaining_prints_unrun(tmp_path: Path, capsys):
    c = tmp_path / "c.json"
    c.write_text(json.dumps({"candidates": [{"repo_id": "a"}, {"repo_id": "b"}]}), encoding="utf-8")
    ad = tmp_path / "ad.json"
    ad.write_text(json.dumps({"models": [{"repo_id": "a"}]}), encoding="utf-8")
    rc = progress.cmd_list_remaining(_ns(candidates=str(c), already_done=str(ad)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 remaining of 2" in out
    assert "b" in out


# ── cmd_stats ──


def test_stats_basic(tmp_path: Path, capsys):
    ad = tmp_path / "ad.json"
    ad.write_text(
        json.dumps(
            {
                "models": [
                    {"repo_id": "a", "status": "completed", "framework": "sglang", "precision": "fp8", "gain_pct": 10},
                    {"repo_id": "b", "status": "failed"},
                ]
            }
        ),
        encoding="utf-8",
    )
    rc = progress.cmd_stats(_ns(already_done=str(ad), candidates=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Total models: 2" in out
    assert "With gain%" in out


def test_stats_with_candidates(tmp_path: Path, capsys):
    ad = tmp_path / "ad.json"
    ad.write_text(json.dumps({"models": [{"repo_id": "a", "status": "completed"}]}), encoding="utf-8")
    c = tmp_path / "c.json"
    c.write_text(json.dumps({"candidates": [{"repo_id": "a"}, {"repo_id": "b"}]}), encoding="utf-8")
    rc = progress.cmd_stats(_ns(already_done=str(ad), candidates=str(c)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Pool size:    2" in out
    assert "Remaining:    1" in out


# ── main dispatch ──


def test_main_promote(tmp_path: Path, monkeypatch):
    s = tmp_path / "sum.json"
    s.write_text(json.dumps([{"model": "org/m"}]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["progress.py", "promote", str(s), "--already-done", str(tmp_path / "ad.json")])
    assert progress.main() == 0


def test_main_stats(tmp_path: Path, monkeypatch):
    ad = tmp_path / "ad.json"
    ad.write_text(json.dumps({"models": []}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["progress.py", "stats", "--already-done", str(ad)])
    assert progress.main() == 0


def test_main_list_remaining(tmp_path: Path, monkeypatch):
    c = tmp_path / "c.json"
    c.write_text(json.dumps({"candidates": [{"repo_id": "a"}]}), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["progress.py", "list-remaining", str(c), "--already-done", str(tmp_path / "ad.json")]
    )
    assert progress.main() == 0
