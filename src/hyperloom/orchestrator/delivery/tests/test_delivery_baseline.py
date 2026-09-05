# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pre-image is on record before the tree moves, and the record is what is checked."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.bringup import trees
from hyperloom.orchestrator.delivery import (
    NO_PRE_IMAGE,
    Artifact,
    Deliverable,
    DeliverableRefused,
    capture_baseline,
    drifted_paths,
    file_digest,
    freeze_digests,
    mismatched_recorded_artifacts,
    parse_deliverable,
)
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


def test_a_non_git_baseline_records_hash_mode_and_existence(tmp_path: Path) -> None:
    root = tmp_path / "wheel"
    root.mkdir()
    (root / "present.py").write_text("one\n", encoding="utf-8")
    (root / "present.py").chmod(0o644)

    baseline = capture_baseline(
        tree_id="wheel-1",
        root=root,
        kind=trees.VCS_NONE,
        targets=["present.py", "hook.py"],
    )

    present = baseline.entry("present.py")
    assert present is not None and present.existed and present.mode == 0o644 and present.sha256
    absent = baseline.entry("hook.py")
    assert absent is not None and not absent.existed and absent.sha256 == ""


def test_drift_is_kind_dispatched(tmp_path: Path, checkout: Path) -> None:
    wheel = tmp_path / "wheel"
    wheel.mkdir()
    (wheel / "a.py").write_text("one\n", encoding="utf-8")
    wheel_baseline = capture_baseline(tree_id="w", root=wheel, kind=trees.VCS_NONE, targets=["a.py"])
    git_baseline = capture_baseline(tree_id="g", root=checkout, kind=trees.VCS_GIT)

    assert drifted_paths(wheel_baseline) == ()
    assert drifted_paths(git_baseline) == ()

    (wheel / "a.py").write_text("two\n", encoding="utf-8")
    (checkout / "pkg" / "mod.py").write_text("two\n", encoding="utf-8")

    assert drifted_paths(wheel_baseline) == ("a.py",)
    assert drifted_paths(git_baseline) == ("pkg/mod.py",)


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


def test_a_digest_the_specialist_supplied_is_discarded() -> None:
    # The declaration is not a route for a digest: the install resolves the
    # artifacts it will copy, and freezing is the only writer of a digest.
    parsed = parse_deliverable(
        {
            "deliverable": {
                "tree_id": "t",
                "artifacts": [
                    {
                        "source": "/w/cfg.json",
                        "target": "pkg/cfg.json",
                        "source_sha256": "f" * 64,
                        "pre_image_sha256": "e" * 64,
                    }
                ],
            }
        },
        default_tree_id="t",
    )
    assert parsed.artifacts == ()


def test_the_frozen_pre_image_is_the_baseline_not_the_live_tree(tmp_path: Path) -> None:
    root = tmp_path / "wheel"
    root.mkdir()
    target = root / "cfg.json"
    target.write_text("before\n", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = workspace / "cfg.json"
    source.write_text("after\n", encoding="utf-8")

    baseline = capture_baseline(tree_id="w", root=root, kind=trees.VCS_NONE, targets=["cfg.json"])
    pre_image = baseline.entry("cfg.json").sha256  # type: ignore[union-attr]

    # The apply lands before anything downstream looks at the digests.
    target.write_text("after\n", encoding="utf-8")

    frozen = freeze_digests(
        Deliverable(tree_id="w", artifacts=(Artifact(target="cfg.json", tree_id="w", source=str(source)),)),
        baselines={"w": baseline},
        validated_roots=[workspace],
    )
    artifact = frozen.artifacts[0]
    assert artifact.pre_image_sha256 == pre_image
    # Recomputing the pre-image in the live tree would have compared the tree
    # to itself: the target now holds exactly what the source does.
    assert artifact.source_sha256 != pre_image
    assert artifact.source_sha256 == file_digest(target)


def test_a_target_that_did_not_exist_is_marked_rather_than_left_blank(tmp_path: Path) -> None:
    root = tmp_path / "wheel"
    root.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "hook.py").write_text("hook\n", encoding="utf-8")
    baseline = capture_baseline(tree_id="w", root=root, kind=trees.VCS_NONE, targets=["hook.py"])

    frozen = freeze_digests(
        Deliverable(
            tree_id="w",
            artifacts=(Artifact(target="hook.py", tree_id="w", source=str(workspace / "hook.py")),),
        ),
        baselines={"w": baseline},
        validated_roots=[workspace],
    )
    assert frozen.artifacts[0].pre_image_sha256 == NO_PRE_IMAGE


def test_in_place_authoring_with_no_pre_image_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "wheel"
    root.mkdir()
    live = root / "edited.py"
    live.write_text("mutated in place\n", encoding="utf-8")
    baseline = capture_baseline(tree_id="w", root=root, kind=trees.VCS_NONE, targets=[])

    with pytest.raises(DeliverableRefused):
        freeze_digests(
            Deliverable(
                tree_id="w",
                artifacts=(Artifact(target="edited.py", tree_id="w", source=str(live)),),
            ),
            baselines={"w": baseline},
            validated_roots=[tmp_path / "ws"],
        )


def test_a_base_artifact_that_moved_since_its_round_is_named(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = workspace / "cfg.json"
    source.write_text("validated\n", encoding="utf-8")
    root = tmp_path / "wheel"
    root.mkdir()
    baseline = capture_baseline(tree_id="w", root=root, kind=trees.VCS_NONE, targets=["cfg.json"])

    frozen = freeze_digests(
        Deliverable(tree_id="w", artifacts=(Artifact(target="cfg.json", tree_id="w", source=str(source)),)),
        baselines={"w": baseline},
        validated_roots=[workspace],
    )
    recorded = [
        {
            "target": "cfg.json",
            "source": str(source),
            "source_sha256": frozen.artifacts[0].source_sha256,
        }
    ]
    assert mismatched_recorded_artifacts(recorded) == ()

    source.write_text("something else\n", encoding="utf-8")
    assert mismatched_recorded_artifacts(recorded) == ("cfg.json",)


def test_a_record_with_no_frozen_digest_does_not_block_the_base_replay() -> None:
    assert mismatched_recorded_artifacts([{"target": "cfg.json", "source": "/nope"}]) == ()


def test_the_backup_ledger_outlives_the_process_that_wrote_it(tmp_path: Path) -> None:
    backup_root = tmp_path / "backups"
    record = {"target": str(tmp_path / "a.py"), "backup_path": str(backup_root / "a.bak"), "revert_action": "restore"}
    assert ledger.append_record(backup_root, record)

    # A later process holds none of the records the apply took.
    assert ledger.load_records(backup_root) == [record]
    assert ledger.merge_records([], backup_root) == [record]
    # The same record held in memory is not reverted twice.
    assert ledger.merge_records([record], backup_root) == [record]
