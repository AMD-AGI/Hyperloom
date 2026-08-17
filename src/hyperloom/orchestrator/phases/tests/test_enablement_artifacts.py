# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the enablement round artifact snapshot."""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.phases._enablement_artifacts import (
    _FILE_SIZE_LIMIT,
    snapshot_round,
)


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
