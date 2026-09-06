# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-root identity, base_sha capture point, and content capture at the KEEP."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._patch_snapshot import _git_commit_kept
from hyperloom.orchestrator.actions.executors.integrate_patch import _git_head_sha
from hyperloom.orchestrator.enablement.recipe.keep_records import (
    build_root_records,
    capture_root_snapshots,
    classify_root,
    collect_contributions,
    declared_targets,
)
from hyperloom.orchestrator.enablement.recipe.keep_probe import resolve_keep_interpreter

BASE_TEXT = "value = 1\n"
PATCHED_TEXT = "value = 2\n"
TARGET = "srt/module.py"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "framework"
    (root / "srt").mkdir(parents=True)
    (root / TARGET).write_text(BASE_TEXT, encoding="utf-8")
    _git(root.parent, "init", "-q", str(root))
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def test_base_sha_is_the_tree_the_patches_apply_to(repo: Path):
    """The recorded sha must predate the KEEP commit for that root.

    Reading HEAD after the KEEP commit records a tree that already contains the
    patch, so a replay applying the recorded patch step to the recorded base
    would apply it a second time.
    """
    recorded = _git_head_sha(repo)

    (repo / TARGET).write_text(PATCHED_TEXT, encoding="utf-8")
    ok, _note = _git_commit_kept(repo, "hyperloom KEEP", [TARGET])
    assert ok

    post_keep = _git_head_sha(repo)
    assert recorded != post_keep
    assert recorded == _git(repo, "rev-parse", "HEAD~1")
    # The patch applies exactly once against the recorded base: that tree still
    # holds the pre-patch content.
    assert _git(repo, "show", f"{recorded}:{TARGET}") == BASE_TEXT.strip()
    assert _git(repo, "show", f"{post_keep}:{TARGET}") == PATCHED_TEXT.strip()


def test_root_record_carries_the_pre_keep_base_sha(repo: Path):
    recorded = _git_head_sha(repo)
    (repo / TARGET).write_text(PATCHED_TEXT, encoding="utf-8")
    _git_commit_kept(repo, "hyperloom KEEP", [TARGET])

    records = build_root_records(
        contributions={str(repo): {"patch_apply"}},
        base_sha_by_root={str(repo): recorded},
        git_roots=[str(repo)],
        session_framework_root=str(repo),
    )
    assert records[0]["base_sha"] == recorded
    assert records[0]["base_sha"] != _git_head_sha(repo)


def test_non_git_root_has_a_null_base_sha_beside_its_is_git_flag():
    records = build_root_records(
        contributions={"/plain/tree": {"artifact_install"}},
        base_sha_by_root={},
        git_roots=[],
        session_framework_root="/fr",
    )
    assert records[0]["is_git"] is False and records[0]["base_sha"] == ""


def test_root_kinds_and_replay_anchors():
    assert classify_root("/fr", session_framework_root="/fr") == (
        "framework_checkout",
        {"anchor": "framework_root", "rel": ""},
    )
    kind, target = classify_root("/opt/venv/lib/python3.10/site-packages/aiter", session_framework_root="/fr")
    assert kind == "site_packages" and target == {"anchor": "site_packages", "rel": "aiter"}
    assert classify_root("/elsewhere", session_framework_root="/fr")[1]["anchor"] == "unmappable"


def test_contributions_split_inputs_from_output_targets():
    contributions = collect_contributions(
        framework_root="/fr",
        patch_roots={"/p/1.patch": "/fr"},
        artifacts=[{"target": "/pkg/a.py", "rel_target": "a.py", "root": "/pkg"}],
    )
    assert contributions["/fr"] == {"patch_apply"}
    assert contributions["/pkg"] == {"artifact_install"}


def test_declared_targets_separate_upserts_from_deletions():
    targets = declared_targets(
        framework_root="/fr",
        upserted=["srt/a.py"],
        deleted=["srt/gone.py"],
        artifacts=[{"rel_target": "srt/art.py", "root": "/fr"}],
    )
    assert targets["/fr"] == {"srt/a.py": "upsert", "srt/gone.py": "delete", "srt/art.py": "upsert"}


def test_snapshot_capture_is_portable_and_records_declared_ops(repo: Path, tmp_path: Path):
    (repo / TARGET).write_text(PATCHED_TEXT, encoding="utf-8")
    records = build_root_records(
        contributions={str(repo): {"patch_apply"}},
        base_sha_by_root={str(repo): "a" * 40},
        git_roots=[str(repo)],
        session_framework_root=str(repo),
    )
    session_dir = tmp_path / "session"
    manifests = capture_root_snapshots(
        records=records,
        targets={str(repo): {TARGET: "upsert", "srt/gone.py": "delete"}},
        dest_root=session_dir / "optimization_stack" / "enablement",
        session_dir=session_dir,
    )
    manifest = manifests[0]
    assert manifest["complete"] is True
    assert {f["rel"]: f["op"] for f in manifest["files"]} == {TARGET: "upsert", "srt/gone.py": "delete"}
    assert "framework_root" not in manifest and "snapshot_dir" not in manifest
    assert manifest["snapshot_ref"] == f"optimization_stack/enablement/{records[0]['id']}"
    assert (session_dir / manifest["snapshot_ref"] / "files" / TARGET).read_text() == PATCHED_TEXT


def test_undeclared_absent_target_is_recorded_missing_and_incomplete(repo: Path, tmp_path: Path):
    records = build_root_records(
        contributions={str(repo): {"patch_apply"}},
        base_sha_by_root={},
        git_roots=[str(repo)],
        session_framework_root=str(repo),
    )
    manifests = capture_root_snapshots(
        records=records,
        targets={str(repo): {"srt/never.py": "upsert"}},
        dest_root=tmp_path / "dest",
        session_dir=tmp_path,
    )
    assert manifests[0]["complete"] is False
    assert manifests[0]["files"][0]["op"] == "missing"


def test_keep_interpreter_prefers_the_override_then_the_bypass_backend():
    assert (
        resolve_keep_interpreter({"runtime_python_exe": "/a/py", "framework_python": "/b/py"}, backend_name="bypass")
        == "/a/py"
    )
    assert resolve_keep_interpreter({"framework_python": "/b/py"}, backend_name="magpie") == "/b/py"
    assert resolve_keep_interpreter({}, backend_name="bypass", bypass_interpreter="/c/py") == "/c/py"
    # Under any other backend the launching interpreter is not resolvable, and
    # naming a plausible one would reproduce the defect this closes.
    assert resolve_keep_interpreter({}, backend_name="magpie", bypass_interpreter="/c/py") == ""
