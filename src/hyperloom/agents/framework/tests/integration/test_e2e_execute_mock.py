# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration test: ``fa explore --execute`` against fake PR Monitor + echo build/bench; asserts deterministic winner gate and KB auto-append when ``kb_domain`` is set."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_request(
    tmp_path: Path,
    *,
    pr_monitor_base_url: str,
    work_dir: Path,
) -> Path:
    """Materialise an ExploreRequest JSON for execute-mode integration."""
    # Inline shell commands writing JSON to per-candidate paths.
    bench_cmd = (
        'python3 -c "import json,sys; '
        "open(sys.argv[1],'w').write(json.dumps({'throughput': 200.0, 'completed': '1/1'}))\" "
        "{candidate_dir}/benchmark.json"
    )
    acc_cmd = (
        'python3 -c "import json,sys; '
        "open(sys.argv[1],'w').write(json.dumps({'accuracy': 0.95}))\" "
        "{candidate_dir}/accuracy.json"
    )
    req = {
        "framework": "sglang",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": str(work_dir),
        "baseline": {"throughput": 100.0, "accuracy": 0.9, "completed": "1/1"},
        "thresholds": {"min_throughput_ratio": 1.05, "max_accuracy_drop": 0.05},
        "search_perf_prs": True,
        "max_search_candidates": 1,
        "pr_monitor": {"base_url": pr_monitor_base_url, "timeout_sec": 5.0},
        "search_modes": ["pr_monitor"],
        "prepare_candidate_env": False,
        "kb_domain": "framework",
        "commands": {
            "benchmark": {"command": bench_cmd, "timeout_sec": 30, "required": True},
            "accuracy": {"command": acc_cmd, "timeout_sec": 30, "required": True},
        },
        "outputs": {
            "benchmark_json": "{candidate_dir}/benchmark.json",
            "accuracy_json": "{candidate_dir}/accuracy.json",
        },
    }
    path = tmp_path / "req.json"
    path.write_text(json.dumps(req), encoding="utf-8")
    return path


def test_explore_execute_mock_winner_and_kb_append(tmp_path: Path, fake_pr_monitor: str) -> None:
    """``fa explore --execute`` picks the winner and auto-appends KB."""
    work_dir = tmp_path / "work"
    kb_root = tmp_path / "kb"
    req_path = _write_request(tmp_path, pr_monitor_base_url=fake_pr_monitor, work_dir=work_dir)
    summary_path = tmp_path / "summary.json"

    env = dict(os.environ)
    env["INFERENCE_OPTIMIZER_FA_KB_PATH"] = str(kb_root)
    env.pop("FRAMEWORK_AGENT_KB_DIR", None)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hyperloom.agents.framework.runtime.cli",
            "explore",
            "--execute",
            "--request",
            str(req_path),
            "--out",
            str(summary_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "execute"
    assert summary["winner_ref"] == "PR:1"
    assert summary["promotion_policy"] == "manual_only"
    assert summary["kb_contribution"]["status"] == "appended"
    assert summary["kb_contribution"]["domain"] == "framework"

    cand = summary["candidates"][0]
    assert cand["status"] == "succeeded"
    assert cand["winner"] is True
    assert cand["throughput"] == 200.0
    assert cand["accuracy"] == 0.95
    assert [c["name"] for c in cand["commands"]] == ["benchmark", "accuracy"]
    assert all(c["returncode"] == 0 for c in cand["commands"])

    kb_file = kb_root / "framework" / "empirical_kb.md"
    assert kb_file.is_file()
    text = kb_file.read_text(encoding="utf-8")
    assert "sglang winner PR:1" in text
    assert "throughput_ratio" in text
    assert "source=`fa explore --execute`" in text
