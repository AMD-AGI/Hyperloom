# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the enablement round artifact snapshot helper."""

from __future__ import annotations

import json


from hyperloom.orchestrator.phases._enablement_artifacts import snapshot_round


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


def test_snapshot_creates_round_json(tmp_path):
    snapshot_round(tmp_path, _res())
    rj = tmp_path / "reports" / "enablement" / "abc123" / "round.json"
    assert rj.is_file()
    data = json.loads(rj.read_text())
    assert data["status"] == "kept"
    assert data["specialist_task_id"] == "abc123"


def test_snapshot_copies_existing_patch(tmp_path):
    patch_src = tmp_path / "runs" / "specialist" / "abc123" / "patches"
    patch_src.mkdir(parents=True)
    (patch_src / "001_fix.patch").write_text("diff --git a/f b/f\n+fix", encoding="utf-8")
    snapshot_round(tmp_path, _res())
    dest = tmp_path / "reports" / "enablement" / "abc123" / "patches" / "001_fix.patch"
    assert dest.is_file()


def test_snapshot_noop_on_invalid_input(tmp_path):
    # None session_dir
    snapshot_round(None, _res())
    # non-dict res
    snapshot_round(tmp_path, "bad")
    assert not (tmp_path / "reports" / "enablement").exists()


def test_snapshot_handles_missing_workspace_gracefully(tmp_path):
    """workspace not found → no crash, round.json still written."""
    snapshot_round(tmp_path, _res(specialist_task_id="no_ws_tid"))
    rj = tmp_path / "reports" / "enablement" / "no_ws_tid" / "round.json"
    assert rj.is_file()


def test_snapshot_sanitises_task_id_path_traversal(tmp_path):
    snapshot_round(tmp_path, _res(specialist_task_id="../evil"))
    round_dirs = list((tmp_path / "reports" / "enablement").iterdir())
    assert len(round_dirs) == 1
    assert ".." not in str(round_dirs[0])


def test_snapshot_copies_specialist_done_and_prompt(tmp_path):
    ws = tmp_path / "runs" / "specialist" / "tid_x"
    ws.mkdir(parents=True)
    (ws / "specialist_done.json").write_text('{"summary": "ok"}', encoding="utf-8")
    (ws / "prompt.md").write_text("# prompt", encoding="utf-8")
    snapshot_round(tmp_path, _res(specialist_task_id="tid_x"))
    out = tmp_path / "reports" / "enablement" / "tid_x"
    assert (out / "specialist_done.json").is_file()
    assert (out / "prompt.md").is_file()


def test_snapshot_launch_config_copied(tmp_path):
    cfg = tmp_path / "runs" / "integrate_patch" / "t1" / "launch.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("tp: 8\n", encoding="utf-8")
    snapshot_round(
        tmp_path,
        _res(specialist_task_id="t1", enablement_accepted_config_path=str(cfg)),
    )
    assert (tmp_path / "reports" / "enablement" / "t1" / "launch_config.yaml").is_file()
