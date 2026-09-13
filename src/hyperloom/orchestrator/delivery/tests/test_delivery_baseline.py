# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pre-image is on record before the tree moves, and the record is what is checked."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.bringup import trees
from hyperloom.orchestrator.delivery import parse_deliverable
from hyperloom.orchestrator.delivery import ledger


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture()
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("base\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def test_a_wheel_under_an_unrelated_checkout_is_not_that_checkout(checkout: Path) -> None:
    nested = checkout / "site-packages" / "sglang"
    nested.mkdir(parents=True)
    (nested / "__init__.py").write_text("x\n", encoding="utf-8")

    assert trees.tree_kind(nested) == trees.VCS_NONE
    pinned = trees.resolve_trees([str(nested)])
    assert pinned[0].vcs == trees.VCS_NONE
    # Borrowing the enclosing repo's commit would offer a diff base that never
    # contained these files.
    assert pinned[0].head_commit == ""


def test_a_checkout_records_the_commit_it_was_pinned_at(checkout: Path) -> None:
    pinned = trees.resolve_trees([str(checkout)])[0]
    assert pinned.vcs == trees.VCS_GIT
    assert len(pinned.head_commit) == 40


def test_legacy_keys_are_read_on_input_and_never_emitted() -> None:
    parsed = parse_deliverable(
        {
            "patches_written": ["/w/patches/one.patch"],
            "artifacts_written": [{"source": "/w/cfg.json", "target": "pkg/cfg.json"}],
            "extra_envs": {"A": "1"},
            "extra_server_args": "--flag",
            "setup_commands": ["pip install x"],
        },
        default_tree_id="tree-1",
    )

    assert parsed.patches == ("/w/patches/one.patch",)
    assert parsed.envs == {"A": "1"}
    assert parsed.server_args == "--flag"
    assert parsed.setup_commands == ("pip install x",)
    emitted = parsed.to_dict()
    for legacy in ("patches_written", "artifacts_written", "extra_envs", "extra_server_args"):
        assert legacy not in emitted


def test_the_backup_ledger_outlives_the_process_that_wrote_it(tmp_path: Path) -> None:
    backup_root = tmp_path / "backups"
    record = {"target": str(tmp_path / "a.py"), "backup_path": str(backup_root / "a.bak"), "revert_action": "restore"}
    assert ledger.append_record(backup_root, record)

    # A later process holds none of the records the apply took.
    assert ledger.load_records(backup_root) == [record]
    assert ledger.merge_records([], backup_root) == [record]
    # The same record held in memory is not reverted twice.
    assert ledger.merge_records([record], backup_root) == [record]
