# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ci/build_summary.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import build_summary as bs  # noqa: E402


# ── formatting helpers ──


def test_fmt_num():
    assert bs.fmt_num(None) == "—"
    assert bs.fmt_num(1.234) == "1.2"
    assert bs.fmt_num("notnum") == "notnum"


def test_fmt_pct():
    assert bs.fmt_pct(None) == "—"
    assert bs.fmt_pct(5.0) == "+5.00%"
    assert bs.fmt_pct("x") == "x"


def test_gain_medal():
    assert bs.gain_medal(None) == ""
    assert bs.gain_medal(60) == "🥇🥇🥇🥇"
    assert bs.gain_medal(30) == "🥇🥇🥇"
    assert bs.gain_medal(15) == "🥇🥇"
    assert bs.gain_medal(5) == "🥇"
    assert bs.gain_medal(0.5) == "🟢"
    assert bs.gain_medal(0) == "➖"
    assert bs.gain_medal(-1) == ""


def test_status_icon():
    assert bs.status_icon({"ci_success": True}) == "✅"
    assert bs.status_icon({"final_status": "Failed"}) == "❌"
    assert bs.status_icon({"baseline_tok_per_gpu": 1, "optimized_tok_per_gpu": 2}) == "✅"
    assert bs.status_icon({"baseline_tok_per_gpu": 1}) == "🟡"
    assert bs.status_icon({}) == "❌"


def test_derive_params():
    assert bs.derive_params("org/Model-14B-Instruct") == "14B"
    assert bs.derive_params("org/Qwen-1.5B") == "1.5B"
    assert bs.derive_params("org/no-params") is None
    assert bs.derive_params(None) is None


def test_short_model_name():
    assert bs.short_model_name("org/Model") == "Model"
    assert bs.short_model_name(None) == "—"


def test_gain_sort_key():
    assert bs.gain_sort_key({"ci_success": True, "gain_pct": 10}) == (0, -10.0)
    assert bs.gain_sort_key({"ci_success": False}) == (1, 1.0)
    assert bs.gain_sort_key({"ci_success": True, "gain_pct": "bad"}) == (0, 1.0)


# ── build_run_metadata ──


def test_build_run_metadata(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    md = bs.build_run_metadata()
    assert md["source"] == "hyperloom-ci"
    assert md["github_run_id"] == "42"


# ── collect_rows / render_markdown / main (end-to-end) ──


def _make_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    artifacts = tmp_path / "artifacts"
    manifests = tmp_path / "manifests"
    (artifacts / "task1").mkdir(parents=True)
    (artifacts / "task1" / "ci_metrics.json").write_text(
        json.dumps(
            {
                "baseline_throughput": 100,
                "optimized_throughput": 130,
                "model": "org/m-7B",
                "framework": "sglang",
                "tp": 8,
            }
        ),
        encoding="utf-8",
    )
    manifests.mkdir()
    (manifests / "submission_manifest.json").write_text(
        json.dumps({"records": [{"task_id": "task1", "model": "org/m-7B", "final_status": "Succeeded"}]}),
        encoding="utf-8",
    )
    return artifacts, manifests


def test_collect_rows(tmp_path: Path):
    artifacts, manifests = _make_artifacts(tmp_path)
    rows, normalized = bs.collect_rows(artifacts, manifests)
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "org/m-7B"
    assert row["gain_pct"] == 30.0
    assert row["ci_success"] is True


def test_render_markdown(tmp_path: Path):
    artifacts, manifests = _make_artifacts(tmp_path)
    rows, _ = bs.collect_rows(artifacts, manifests)
    md = bs.render_markdown(rows, 1024, 1024)
    assert "# Hyperloom CI Summary" in md
    assert "`m-7B`" in md


def test_main(tmp_path: Path, monkeypatch):
    artifacts, manifests = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_summary.py",
            "--artifacts-dir",
            str(artifacts),
            "--manifests-dir",
            str(manifests),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert bs.main() == 0
    assert (out_dir / "ci_summary.md").exists()
    assert (out_dir / "ci_summary.json").exists()
    data = json.loads((out_dir / "ci_summary.json").read_text(encoding="utf-8"))
    assert data["rows"]
