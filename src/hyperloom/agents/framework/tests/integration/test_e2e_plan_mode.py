# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration test: ``fa explore`` (plan mode) e2e — runs the CLI as a subprocess (argparse + JSON-IO + explorer) against the in-process ``fake_pr_monitor`` fixture (no internet)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_request(
    tmp_path: Path,
    *,
    pr_monitor_base_url: str,
    work_dir: Path,
) -> Path:
    """Materialise an ExploreRequest JSON for plan-mode integration."""
    req = {
        "framework": "sglang",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": str(work_dir),
        "baseline": {"throughput": 1.0, "accuracy": 0.9, "completed": "1/1"},
        "search_perf_prs": True,
        "max_search_candidates": 1,
        "pr_monitor": {"base_url": pr_monitor_base_url, "timeout_sec": 5.0},
        "search_modes": ["pr_monitor"],
        "gap_description": "improve sglang fp8 MoE on MI300X",
        "prepare_candidate_env": False,
        "commands": {},
    }
    path = tmp_path / "req.json"
    path.write_text(json.dumps(req), encoding="utf-8")
    return path


def test_explore_plan_mode_e2e_against_fake_pr_monitor(tmp_path: Path, fake_pr_monitor: str) -> None:
    """``fa explore`` (plan) writes pr.patches + pr_files.json + a sane summary."""
    work_dir = tmp_path / "work"
    req_path = _write_request(tmp_path, pr_monitor_base_url=fake_pr_monitor, work_dir=work_dir)
    summary_path = tmp_path / "summary.json"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hyperloom.agents.framework.runtime.cli",
            "explore",
            "--request",
            str(req_path),
            "--out",
            str(summary_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "plan"
    assert summary["promotion_policy"] == "manual_only"
    assert summary["audit_materials"]["patch_files_present"] == 1
    assert summary["audit_materials"]["files_json_present"] == 1
    assert len(summary["candidates"]) == 1
    c = summary["candidates"][0]
    assert c["candidate"]["ref"] == "PR:1"
    assert c["candidate"]["author"] == "fakebot"
    assert c["candidate"]["head_sha"].startswith("deadbeef")
    assert c["status"] == "planned"

    cand_dir = work_dir / "candidates" / "01_pr-1"
    patches = (cand_dir / "pr.patches").read_text(encoding="utf-8")
    assert "diff --git a/python/x/foo.py" in patches
    assert "+b" in patches
    files_json = json.loads((cand_dir / "pr_files.json").read_text(encoding="utf-8"))
    assert files_json["number"] == 1
    assert files_json["files"][0]["file_path"] == "python/x/foo.py"
