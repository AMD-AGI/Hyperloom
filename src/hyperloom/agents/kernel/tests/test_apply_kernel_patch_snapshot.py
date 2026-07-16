# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Snapshot (content-addressed) deploy: land the whole patch byte-for-byte or fail.

Covers the hard requirement that the entire backend patch is applied exactly as
given — every addition, deletion, and substitution — across all files
atomically, with no fuzzy/partial application and a hard, repo-restoring failure
otherwise.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


_APPLY_TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "apply_kernel_patch.py"


@pytest.fixture(scope="module")
def akp() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_akp_snapshot_under_test", _APPLY_TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# manifest parser

def test_parse_manifest_modify_add_delete(akp):
    patch = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/n.py b/n.py\nnew file mode 100644\n--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n+x\n"
        "diff --git a/d.py b/d.py\ndeleted file mode 100644\n--- a/d.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-y\n"
    )
    descs = akp.parse_patch_manifest(patch)
    ops = {d["path"]: d["op"] for d in descs}
    assert ops == {"m.py": "write", "n.py": "write", "d.py": "delete"}


def test_parse_manifest_rename_maps_to_delete_plus_write(akp):
    patch = (
        "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n"
    )
    descs = akp.parse_patch_manifest(patch)
    assert {"op": "delete", "path": "old.py", "mode": "", "binary": False} in descs
    assert any(d["op"] == "write" and d["path"] == "new.py" for d in descs)


def test_parse_manifest_chmod_and_binary_and_mode(akp):
    chmod = akp.parse_patch_manifest("diff --git a/s b/s\nold mode 100644\nnew mode 100755\n")
    assert chmod == [{"op": "write", "path": "s", "mode": "0755", "binary": False}]
    binary = akp.parse_patch_manifest(
        "diff --git a/b.bin b/b.bin\nnew file mode 100644\nindex 0..1\nGIT binary patch\nliteral 0\n"
    )
    assert binary[0]["binary"] is True and binary[0]["mode"] == "0644"


def test_parse_manifest_empty_raises(akp):
    with pytest.raises(ValueError):
        akp.parse_patch_manifest("   \n")


# apply_snapshot core

def _mk(tmp_path):
    repo = tmp_path / "repo"
    snap = tmp_path / "snap"
    repo.mkdir()
    snap.mkdir()
    return repo, snap, tmp_path / "backup"


def test_apply_snapshot_modify_add_delete_atomic(tmp_path, akp):
    repo, snap, bk = _mk(tmp_path)
    (repo / "mod.py").write_text("OLD\n")
    (repo / "del.py").write_text("bye\n")
    (snap / "mod.py").write_text("NEW\n")
    (snap / "add.py").write_text("ADD\n")
    descs = [
        {"op": "write", "path": "mod.py", "mode": "", "binary": False},
        {"op": "write", "path": "add.py", "mode": "0755", "binary": False},
        {"op": "delete", "path": "del.py", "mode": "", "binary": False},
    ]
    r = akp.apply_snapshot(descriptors=descs, snapshot_dir=snap, repo_root=repo, backup_dir=bk)
    assert r["status"] == "ok"
    assert (repo / "mod.py").read_text() == "NEW\n"
    assert (repo / "add.py").read_text() == "ADD\n"
    assert oct((repo / "add.py").stat().st_mode)[-3:] == "755"
    assert not (repo / "del.py").exists()


def test_apply_snapshot_preflight_missing_content_writes_nothing(tmp_path, akp):
    repo, snap, bk = _mk(tmp_path)
    (repo / "a.py").write_text("A0\n")
    (snap / "a.py").write_text("A1\n")
    descs = [
        {"op": "write", "path": "a.py", "mode": "", "binary": False},
        {"op": "write", "path": "missing.py", "mode": "", "binary": False},
    ]
    r = akp.apply_snapshot(descriptors=descs, snapshot_dir=snap, repo_root=repo, backup_dir=bk)
    assert r["status"] == "failed"
    assert (repo / "a.py").read_text() == "A0\n"  # untouched


@pytest.mark.parametrize("bad", ["../victim.py", "/etc/evil.py", "sub/../../victim.py"])
def test_apply_snapshot_rejects_path_escape(tmp_path, akp, bad):
    repo, snap, bk = _mk(tmp_path)
    victim = tmp_path / "victim.py"
    victim.write_text("safe\n")
    (snap / "x.py").write_text("x\n")
    descs = [{"op": "write", "path": bad, "mode": "", "binary": False}]
    r = akp.apply_snapshot(descriptors=descs, snapshot_dir=snap, repo_root=repo, backup_dir=bk)
    assert r["status"] == "failed"
    assert victim.read_text() == "safe\n"


def test_apply_snapshot_symlink_reproduced_not_dereferenced(tmp_path, akp):
    repo, snap, bk = _mk(tmp_path)
    os.symlink("/etc/hostname", snap / "link.py")
    descs = [{"op": "write", "path": "link.py", "mode": "", "binary": False}]
    r = akp.apply_snapshot(descriptors=descs, snapshot_dir=snap, repo_root=repo, backup_dir=bk)
    assert r["status"] == "ok"
    assert (repo / "link.py").is_symlink()


def test_apply_snapshot_midapply_failure_restores_all(tmp_path, akp, monkeypatch):
    repo, snap, bk = _mk(tmp_path)
    (repo / "first.py").write_text("F0\n")
    (snap / "first.py").write_text("F1\n")
    (snap / "second.py").write_text("S1\n")
    descs = [
        {"op": "write", "path": "first.py", "mode": "", "binary": False},
        {"op": "write", "path": "second.py", "mode": "", "binary": False},
    ]
    # Inject a write failure on the second file's apply copy after the first
    # succeeded.
    real_copy2 = akp.shutil.copy2

    def flaky(src, dst, *a, **k):
        if Path(dst).parent == repo and Path(dst).name == "second.py":
            raise OSError("injected write failure")
        return real_copy2(src, dst, *a, **k)

    monkeypatch.setattr(akp.shutil, "copy2", flaky)
    r = akp.apply_snapshot(descriptors=descs, snapshot_dir=snap, repo_root=repo, backup_dir=bk)
    assert r["status"] == "failed"
    assert (repo / "first.py").read_text() == "F0\n"  # rolled back
    assert not (repo / "second.py").exists()


# end-to-end apply_kernel_patch snapshot mode + revert

def _patch_text():
    return (
        "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n@@ -1,2 +1,2 @@\n import triton\n-OLD\n+NEW\n"
        "diff --git a/add.py b/add.py\nnew file mode 100644\n--- /dev/null\n+++ b/add.py\n@@ -0,0 +1 @@\n+ADD\n"
        "diff --git a/del.py b/del.py\ndeleted file mode 100644\n--- a/del.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n"
    )


def test_snapshot_mode_apply_then_revert_roundtrip(tmp_path, akp):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("import triton\nOLD\n")
    (repo / "del.py").write_text("bye\n")
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "mod.py").write_text("import triton\nNEW\n")
    (snap / "add.py").write_text("ADD\n")
    patch = tmp_path / "p.patch"
    patch.write_text(_patch_text())

    r = akp.apply_kernel_patch(
        patch_path=patch, target_file=str(repo / "mod.py"), backup_root=tmp_path / "bk",
        snapshot_dir=snap, allow_unknown_target=True, skip_rebuild=True, repo_root=str(repo),
    )
    assert r["status"] == "ok"
    assert (repo / "mod.py").read_text() == "import triton\nNEW\n"
    assert (repo / "add.py").exists()
    assert not (repo / "del.py").exists()

    rv = akp.revert_kernel_patch(r["manifest_path"])
    assert rv["status"] == "ok"
    assert (repo / "mod.py").read_text() == "import triton\nOLD\n"  # modified restored
    assert not (repo / "add.py").exists()  # added unlinked
    assert (repo / "del.py").read_text() == "bye\n"  # deleted recreated


def test_snapshot_mode_post_verify_mismatch_fails_and_restores(tmp_path, akp, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "k.py").write_text("V0\n")
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "k.py").write_text("V1\n")
    patch = tmp_path / "p.patch"
    patch.write_text("diff --git a/k.py b/k.py\n--- a/k.py\n+++ b/k.py\n@@ -1 +1 @@\n-V0\n+V1\n")

    # Corrupt the applied file once during apply so post-verify detects a
    # content mismatch vs the snapshot and must restore.
    real_copy2 = akp.shutil.copy2
    state = {"poisoned": False}

    def poisoned(src, dst, *a, **k):
        real_copy2(src, dst, *a, **k)
        if not state["poisoned"] and Path(dst).name == "k.py" and Path(dst).parent == repo:
            Path(dst).write_text("CORRUPT\n")
            state["poisoned"] = True

    monkeypatch.setattr(akp.shutil, "copy2", poisoned)
    r = akp.apply_kernel_patch(
        patch_path=patch, target_file=str(repo / "k.py"), backup_root=tmp_path / "bk",
        snapshot_dir=snap, allow_unknown_target=True, skip_rebuild=True, repo_root=str(repo),
    )
    assert r["status"] == "failed"
    monkeypatch.undo()
    assert (repo / "k.py").read_text() == "V0\n"  # restored to original


def test_snapshot_mode_unparseable_patch_hard_fails(tmp_path, akp):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "k.py").write_text("V0\n")
    snap = tmp_path / "snap"
    snap.mkdir()
    patch = tmp_path / "p.patch"
    patch.write_text("")  # empty -> nothing to apply
    r = akp.apply_kernel_patch(
        patch_path=patch, target_file=str(repo / "k.py"), backup_root=tmp_path / "bk",
        snapshot_dir=snap, allow_unknown_target=True, skip_rebuild=True, repo_root=str(repo),
    )
    assert r["status"] == "failed"
    assert (repo / "k.py").read_text() == "V0\n"


# _is_multi_node is env-only ($INFERENCE_OPTIMIZER_NODES): a co-tenant cannot
# force multi-node fan-out by planting a world-writable state file.


def test_is_multi_node_true_when_env_ge_2(akp, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    assert akp._is_multi_node() is True


def test_is_multi_node_false_when_unset_or_single(akp, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    assert akp._is_multi_node() is False
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    assert akp._is_multi_node() is False


def test_is_multi_node_false_on_non_numeric(akp, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "not-an-int")
    assert akp._is_multi_node() is False


def test_is_multi_node_ignores_planted_state_file(akp, monkeypatch, tmp_path):
    planted = tmp_path / "multi_node_state.json"
    planted.write_text('{"nodes": 8}', encoding="utf-8")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(planted))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    assert akp._is_multi_node() is False
