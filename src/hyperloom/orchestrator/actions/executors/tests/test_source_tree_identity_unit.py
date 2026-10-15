# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tree identity of an integrate attempt and what a replaced tree still owes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors.source_tree_identity import (
    ensure_tree_instance,
    new_recovery_record,
    read_tree_instance,
    settle_replaced_tree,
    tree_replaced,
)
from hyperloom.orchestrator.delivery.ledger import append_record, file_digest, mark_prepared
from hyperloom.orchestrator.tests._helpers import init_git_repo


def test_identity_is_created_once_and_survives_in_the_same_checkout(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    init_git_repo(root)
    assert read_tree_instance(root) == ""

    first = ensure_tree_instance(root)

    assert first and ensure_tree_instance(root) == first == read_tree_instance(root)
    assert not [p for p in (root / ".git").iterdir() if p.name.startswith("tmp")]
    shutil.rmtree(root)
    init_git_repo(root)
    assert read_tree_instance(root) == ""


def test_a_fresh_clone_at_the_same_path_reads_as_replaced(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    init_git_repo(root)
    recovery = new_recovery_record(root, git_head="abc", recovery_root=tmp_path / "recovery")
    pending = {"framework_source_root": str(root), "recovery": {**recovery, "phase": "ready"}}
    assert not tree_replaced(pending, "")

    shutil.rmtree(root)
    init_git_repo(root)

    assert tree_replaced(pending, "")


@pytest.mark.parametrize("phase", ["before_stash", "ready", "applied", "applied_with_restored_stash", "files_restored"])
def test_every_undischarged_phase_reads_a_new_identity_as_replaced(tmp_path: Path, phase: str) -> None:
    root = tmp_path / "tree"
    init_git_repo(root)
    recovery = {"version": 1, "phase": phase, "tree_instance": "recorded-elsewhere", "stash_oid": ""}

    assert tree_replaced({"framework_source_root": str(root), "recovery": recovery}, None)


@pytest.mark.parametrize("phase", ["restored", "accepted"])
def test_a_discharged_window_never_reads_as_replaced(tmp_path: Path, phase: str) -> None:
    root = tmp_path / "tree"
    init_git_repo(root)
    recovery = {"version": 1, "phase": phase, "tree_instance": "recorded-elsewhere", "stash_oid": ""}

    assert not tree_replaced({"framework_source_root": str(root), "recovery": recovery}, "lost-container")


def test_a_linked_worktree_recreated_at_the_same_path_reads_as_replaced(tmp_path: Path) -> None:
    """Linked worktrees share one object store and stash list, but not the identity."""
    main = tmp_path / "main"
    init_git_repo(main)
    linked = tmp_path / "linked"
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-q", "--detach", str(linked)], check=True)
    recovery = new_recovery_record(linked, git_head="abc", recovery_root=tmp_path / "recovery")
    pending = {"framework_source_root": str(linked), "recovery": {**recovery, "phase": "ready"}}
    assert read_tree_instance(main) == ""
    assert not tree_replaced(pending, None)

    subprocess.run(["git", "-C", str(main), "worktree", "remove", "--force", str(linked)], check=True)
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-q", "--detach", str(linked)], check=True)

    assert tree_replaced(pending, None)


def test_an_unreadable_tree_is_not_called_replaced(tmp_path: Path) -> None:
    pending = {
        "framework_source_root": str(tmp_path / "missing"),
        "recovery": {"version": 1, "phase": "ready", "tree_instance": "deadbeef", "stash_oid": "0" * 40},
    }

    assert not tree_replaced(pending, "lost-container")


def test_a_non_git_attempt_records_no_identity(tmp_path: Path) -> None:
    root = tmp_path / "plain"
    root.mkdir()

    assert "tree_instance" not in new_recovery_record(root, git_head="", recovery_root=tmp_path / "recovery")


def test_a_replaced_tree_restores_only_targets_outside_it(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    init_git_repo(root)
    recovery_root = tmp_path / "recovery"
    backups = recovery_root / "artifact_backups"
    backups.mkdir(parents=True)
    outside = tmp_path / "persistent" / "tuned.csv"
    outside.parent.mkdir()
    outside.write_text("before\n", encoding="utf-8")
    shutil.copy2(outside, backups / "000_tuned.csv.bak")
    inside = root / "configs" / "tuned.json"
    for target, existed in ((outside, True), (inside, False)):
        assert append_record(
            backups,
            {
                "target": str(target),
                "rel_target": target.name,
                "root": str(target.parent),
                "kind": "file",
                "existed": existed,
                "backup": str(backups / "000_tuned.csv.bak") if existed else None,
                "source": str(tmp_path / "candidate"),
                "pre_image_sha256": file_digest(backups / "000_tuned.csv.bak") if existed else "",
                "mode": None,
            },
        )
    assert mark_prepared(backups)
    outside.write_text("candidate\n", encoding="utf-8")
    pending = {
        "framework_source_root": str(root),
        "artifacts": [{"target": str(outside)}, {"target": str(inside)}],
        "recovery": {
            "version": 1,
            "phase": "ready",
            "root": str(recovery_root),
            "artifacts_prepared": True,
            "stash_oid": "",
        },
    }

    summary = settle_replaced_tree(pending)

    assert summary["failed"] == []
    assert summary["tree_replaced"] is True
    assert summary["artifacts_reverted"] == ["tuned.csv"]
    assert outside.read_text(encoding="utf-8") == "before\n"
    assert pending["recovery"]["phase"] == "restored"


def _legacy_pending(root: Path, tmp_path: Path, **extra: object) -> dict:
    """A sentinel from before tree identities, whose stash object no tree here holds."""
    recovery = {"version": 1, "phase": "ready", "stash_oid": "1" * 40, "root": str(tmp_path / "recovery"), **extra}
    return {"framework_source_root": str(root), "recovery": recovery}


def _patch(tmp_path: Path, name: str, before: str, after: str) -> str:
    patch = tmp_path / name
    patch.write_text(
        f"diff --git a/cfg.json b/cfg.json\n--- a/cfg.json\n+++ b/cfg.json\n@@ -1 +1 @@\n-{before}\n+{after}\n"
    )
    return str(patch)


@pytest.mark.parametrize(
    ("tree_text", "replaced"),
    [
        ("A\n", True),  # a fresh checkout: the whole candidate still applies forward
        ("B\n", False),  # the first patch is applied here
        ("C\n", False),  # the whole candidate is applied here
        ("C-edited\n", False),  # applied, then edited over: neither direction applies
    ],
)
def test_a_legacy_sentinel_reads_as_replaced_only_when_the_tree_holds_the_candidates_preimage(
    tmp_path: Path, tree_text: str, replaced: bool
) -> None:
    root = tmp_path / "tree"
    init_git_repo(root, seed_file="cfg.json", seed_text="A\n")
    (root / "cfg.json").write_text(tree_text, encoding="utf-8")
    pending = _legacy_pending(root, tmp_path)
    pending["patches"] = [_patch(tmp_path, "p1.diff", "A", "B"), _patch(tmp_path, "p2.diff", "B", "C")]

    assert tree_replaced(pending, "") is replaced
    assert not tree_replaced(pending, None)


@pytest.mark.parametrize(("tree_text", "replaced"), [("before\n", True), ("candidate\n", False)])
def test_a_legacy_sentinel_with_an_artifact_inside_the_tree_needs_its_preimage_there(
    tmp_path: Path, tree_text: str, replaced: bool
) -> None:
    root = tmp_path / "tree"
    init_git_repo(root, seed_file="cfg.json", seed_text="before\n")
    pending = _legacy_pending(root, tmp_path, artifacts_prepared=True)
    backups = Path(pending["recovery"]["root"]) / "artifact_backups"
    backups.mkdir(parents=True)
    target = root / "cfg.json"
    shutil.copy2(target, backups / "000_cfg.json.bak")
    assert append_record(
        backups,
        {
            "target": str(target),
            "rel_target": "cfg.json",
            "root": str(root),
            "kind": "file",
            "existed": True,
            "backup": str(backups / "000_cfg.json.bak"),
            "source": str(tmp_path / "candidate"),
            "pre_image_sha256": file_digest(backups / "000_cfg.json.bak"),
            "mode": None,
        },
    )
    assert mark_prepared(backups)
    target.write_text(tree_text, encoding="utf-8")

    assert tree_replaced(pending, "lost-container") is replaced
