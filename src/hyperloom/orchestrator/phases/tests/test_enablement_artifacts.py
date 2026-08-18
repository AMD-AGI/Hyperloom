# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the enablement round artifact snapshot."""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.phases._enablement_artifacts import (
    _FILE_SIZE_LIMIT,
    snapshot_round,
    write_setting_script,
)
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound


def _res(**kw):
    base = {
        "status": "kept",
        "specialist_task_id": "abc123",
        "patches_applied": [],
        "config_changes_applied": {},
        "extra_envs_applied": {},
        "extra_server_args_applied": "",
        "setup_commands_applied": [],
        "after_signature": {},
        "enablement_accepted_config_path": "",
        "enablement_effective_config": {},
        "enablement_launch_log": "",
    }
    base.update(kw)
    return base


def test_round_json_records_the_result(tmp_path):
    snapshot_round(
        tmp_path,
        _res(extra_envs_applied={"A": "1"}, extra_server_args_applied="--flag"),
    )
    data = json.loads((tmp_path / "reports" / "enablement" / "abc123" / "round.json").read_text())
    assert data["status"] == "kept"
    assert data["extra_envs_applied"] == {"A": "1"}
    assert data["extra_server_args_applied"] == "--flag"


def test_applied_patch_is_copied(tmp_path):
    src = tmp_path / "runs" / "specialist" / "abc123" / "patches"
    src.mkdir(parents=True)
    (src / "001_fix.patch").write_text("diff --git a/f b/f\n", encoding="utf-8")
    snapshot_round(tmp_path, _res(patches_applied=[str(src / "001_fix.patch")]))
    dest = tmp_path / "reports" / "enablement" / "abc123" / "patches" / "001_fix.patch"
    assert dest.read_text() == "diff --git a/f b/f\n"


def test_unapplied_workspace_patch_is_still_copied(tmp_path):
    """A reverted attempt still explains what was tried."""
    src = tmp_path / "runs" / "specialist" / "abc123" / "worktree" / "patches"
    src.mkdir(parents=True)
    (src / "002_try.diff").write_text("diff\n", encoding="utf-8")
    snapshot_round(tmp_path, _res())
    assert (tmp_path / "reports" / "enablement" / "abc123" / "patches" / "002_try.diff").is_file()


def test_specialist_result_and_prompt_are_copied(tmp_path):
    ws = tmp_path / "runs" / "specialist" / "abc123"
    ws.mkdir(parents=True)
    (ws / "specialist_done.json").write_text('{"summary": "ok"}', encoding="utf-8")
    (ws / "prompt.md").write_text("# prompt", encoding="utf-8")
    snapshot_round(tmp_path, _res())
    out = tmp_path / "reports" / "enablement" / "abc123"
    assert (out / "specialist_done.json").is_file()
    assert (out / "prompt.md").is_file()


def test_launch_config_is_copied(tmp_path):
    cfg = tmp_path / "runs" / "integrate_patch" / "t1" / "integrate_patch.with_envs.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("tp: 8\n", encoding="utf-8")
    snapshot_round(tmp_path, _res(enablement_accepted_config_path=str(cfg)))
    assert (tmp_path / "reports" / "enablement" / "abc123" / "launch_config.yaml").is_file()


def test_oversized_artifact_is_skipped(tmp_path):
    src = tmp_path / "runs" / "specialist" / "abc123" / "patches"
    src.mkdir(parents=True)
    big = src / "003_big.patch"
    big.write_bytes(b"x" * (_FILE_SIZE_LIMIT + 1))
    snapshot_round(tmp_path, _res(patches_applied=[str(big)]))
    assert not (tmp_path / "reports" / "enablement" / "abc123" / "patches" / "003_big.patch").exists()


def test_launch_log_excerpt_is_bounded(tmp_path):
    snapshot_round(tmp_path, _res(enablement_launch_log="E" * 5000))
    data = json.loads((tmp_path / "reports" / "enablement" / "abc123" / "round.json").read_text())
    assert len(data["launch_log_excerpt"]) == 1200


def test_unsafe_task_id_is_refused(tmp_path):
    with pytest.raises(ValueError):
        snapshot_round(tmp_path, _res(specialist_task_id="../evil"))


def test_demoted_switch_gate_is_recorded(tmp_path):
    snapshot_round(tmp_path, _res(framework_switch_problems=["patch gates on undeclared HL_X"]))
    data = json.loads((tmp_path / "reports" / "enablement" / "abc123" / "round.json").read_text())
    assert data["framework_switch_problems"] == ["patch gates on undeclared HL_X"]


def test_round_without_a_specialist_is_skipped(tmp_path):
    """Phase-synthesised rounds carry no task id and would all collide."""
    snapshot_round(tmp_path, {"enablement": True, "status": "reverted", "reason": "artifact_unreadable"})
    assert not (tmp_path / "reports" / "enablement").exists()


def test_write_setting_script_produces_executable(tmp_path):
    en = EnablementRound()
    en.setup_commands = ["pip install vllm==0.24"]
    en.accepted_config = {"extra_envs": {"VLLM_ROCM_USE_AITER": "1"}, "extra_server_args": "--tp 4"}

    patch_src = tmp_path / "runs" / "specialist" / "s1" / "001.patch"
    patch_src.parent.mkdir(parents=True)
    patch_src.write_text("diff --git a/f b/f\n", encoding="utf-8")
    en.kept_patches = [str(patch_src)]

    rel = write_setting_script(tmp_path, en, {"framework_root": "/sgl-workspace/sglang"}, "sglang")
    out = tmp_path / rel
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "pip install vllm==0.24" in text
    assert "export VLLM_ROCM_USE_AITER=1" in text
    assert "apply_patch" in text
    assert "patches/001.patch" in text
    assert "export FRAMEWORK_ROOT=/sgl-workspace/sglang" in text
    assert "sglang.launch_server" in text
    assert "--tp 4" in text
    assert (out.stat().st_mode & 0o755) == 0o755


def test_write_setting_script_copies_patches(tmp_path):
    en = EnablementRound()
    patch_src = tmp_path / "fix.patch"
    patch_src.write_text("diff\n", encoding="utf-8")
    en.kept_patches = [str(patch_src)]

    write_setting_script(tmp_path, en, {}, "sglang")
    assert (tmp_path / "reports" / "enablement" / "patches" / "fix.patch").is_file()


def test_write_setting_script_runtime_note(tmp_path):
    en = EnablementRound()
    en.active_runtime = {"venv_root": "/session/enablement/stacks/sglang/s1/venv"}

    write_setting_script(tmp_path, en, {}, "sglang")
    text = (tmp_path / "reports" / "enablement" / "enablement_setting.sh").read_text()
    assert "isolated attempt venv" in text


def test_write_setting_script_minimal_no_enablement_params(tmp_path):
    """Without patches/setup, a basic launch line is still emitted."""
    en = EnablementRound()
    en.accepted_config = {"extra_server_args": "--block-size 128", "extra_envs": {}}

    write_setting_script(tmp_path, en, {}, "vllm", model="/models/M", tp=8)
    text = (tmp_path / "reports" / "enablement" / "enablement_setting.sh").read_text()
    assert "vllm serve $MODEL" in text
    assert "export MODEL=/models/M" in text
    assert "export TP=8" in text
