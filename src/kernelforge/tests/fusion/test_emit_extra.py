# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover emit helper branches: empty repo, classify, export failure, restore prune."""

from __future__ import annotations

import subprocess

from kernelforge.fusion import emit
from kernelforge.fusion.emit import (
    _classify,
    _is_fused_module_name,
    _tracked_paths,
    export_artifacts,
    restore_exported_changes,
)
from kernelforge.fusion.models import FusionArtifacts


def _init_repo(repo):
    for args in (
        ["init", "-q"],
        ["-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "--allow-empty", "-qm", "base"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def test_tracked_paths_empty_returns_set():
    assert _tracked_paths("/repo", []) == set()


def test_classify_new_kernel_and_wiring():
    assert _classify("dir/foo_fused.py", "/m/lfm2.py") == "new_kernel"
    assert _classify("dir/fusion_helper.py", "/m/lfm2.py") == "new_kernel"
    assert _classify("dir/lfm2.py", "/m/lfm2.py") == "framework_wiring_edit"
    assert _classify("dir/other.py", "/m/lfm2.py") == "framework_wiring_edit"


def test_is_fused_module_name_excludes_lookalikes():
    # real fused-kernel modules
    assert _is_fused_module_name("qwen3_fused.py")
    assert _is_fused_module_name("fused_moe.py")
    assert _is_fused_module_name("fusion_helper.py")
    assert _is_fused_module_name("attn_qk_fusion.py")
    # lookalikes that merely contain "fusion" mid-word must NOT match
    assert not _is_fused_module_name("diffusion.py")
    assert not _is_fused_module_name("confusion.py")
    assert not _is_fused_module_name("diffusion_gemma.py")


def test_export_nongit_patch_is_git_apply_compatible(tmp_path):
    """The difflib-generated patch must apply cleanly with `git apply` (what
    Hyperloom uses at integrate)."""
    repo, src, out = _nongit_pkg(tmp_path)
    pristine_text = src.read_text()
    src.write_text(
        "import os\nFUSED = os.environ.get('QWEN3_FUSED', '0') == '1'\ndef forward(x):\n    return x\n",
        encoding="utf-8",
    )
    arts = export_artifacts(str(repo), str(src), out, pristine_dir=str(out / ".pristine"))
    assert arts.patch is not None
    # apply the patch against a fresh checkout of the pristine tree
    apply_root = tmp_path / "apply_here"
    (apply_root / "models").mkdir(parents=True)
    (apply_root / "models" / "qwen3.py").write_text(pristine_text, encoding="utf-8")
    subprocess.run(["git", "-C", str(apply_root), "init", "-q"], check=True, capture_output=True, text=True)
    chk = subprocess.run(["git", "-C", str(apply_root), "apply", "--check", arts.patch], capture_output=True, text=True)
    assert chk.returncode == 0, f"git apply --check failed: {chk.stderr}"
    subprocess.run(["git", "-C", str(apply_root), "apply", arts.patch], check=True, capture_output=True, text=True)
    assert "FUSED = os.environ" in (apply_root / "models" / "qwen3.py").read_text()


def test_export_nongit_sets_repo_root(tmp_path):
    """The manifest must carry the repo_root the patch paths are relative to, so
    Hyperloom applies against the SAME root (site-packages, not a git toplevel)."""
    repo, src, out = _nongit_pkg(tmp_path)
    src.write_text("FUSED = 1\ndef forward(x):\n    return x\n", encoding="utf-8")
    arts = export_artifacts(str(repo), str(src), out, pristine_dir=str(out / ".pristine"))
    assert arts.patch is not None
    assert arts.repo_root == str(repo.resolve())
    assert arts.to_dict()["repo_root"] == str(repo.resolve())


def test_snapshot_returns_empty_when_main_source_fails(tmp_path, monkeypatch):
    """#8: if the MAIN source snapshot fails, return "" so export does not treat the
    edited source as a brand-new file."""
    from kernelforge.fusion import command as cli

    repo, src, out = _nongit_pkg(tmp_path)

    real_copy = cli.shutil.copy2

    def flaky_copy(s, d, *a, **k):
        if str(s) == str(src):
            raise OSError("disk full")
        return real_copy(s, d, *a, **k)

    monkeypatch.setattr(cli.shutil, "copy2", flaky_copy)
    assert cli._snapshot_fusion_source(str(repo), str(src), out) == ""


def test_export_empty_repo_root(tmp_path):
    arts = export_artifacts("", "/src.py", tmp_path / "out")
    assert isinstance(arts, FusionArtifacts)
    assert arts.patch is None and arts.changes == []


def _nongit_pkg(tmp_path):
    """A non-git 'pip install' style framework dir + a pristine snapshot of its src."""
    repo = tmp_path / "vllm_pkg"
    (repo / "models").mkdir(parents=True)
    src = repo / "models" / "qwen3.py"
    pristine_text = "def forward(x):\n    return x\n"
    src.write_text(pristine_text, encoding="utf-8")
    pristine = tmp_path / "out" / ".pristine" / "models"
    pristine.mkdir(parents=True)
    (pristine / "qwen3.py").write_text(pristine_text, encoding="utf-8")
    return repo, src, tmp_path / "out"


def test_export_nongit_uses_pristine_snapshot(tmp_path):
    """Repro: a non-git framework (pip install) must STILL produce a patch.

    git diff is empty in a non-git dir, so the KEPT fusion previously shipped
    patch=null and integrate skipped it. With a pre-authoring pristine snapshot the
    edit is captured as a unified diff.
    """
    repo, src, out = _nongit_pkg(tmp_path)
    # author edits the source in place (env-gated fusion)
    src.write_text(
        "import os\nFUSED = os.environ.get('QWEN3_FUSED', '0') == '1'\ndef forward(x):\n    return x\n",
        encoding="utf-8",
    )
    arts = export_artifacts(str(repo), str(src), out, pristine_dir=str(out / ".pristine"))
    assert arts.patch is not None, "non-git export must still produce a patch"
    patch_text = (out / "fusion.patch").read_text()
    assert "diff --git a/models/qwen3.py b/models/qwen3.py" in patch_text
    assert "+FUSED = os.environ" in patch_text
    assert any(c["path"] == "models/qwen3.py" for c in arts.changes)


def test_export_nongit_without_snapshot_returns_empty(tmp_path):
    """No pristine snapshot -> nothing to diff against -> empty (documents the need)."""
    repo, src, out = _nongit_pkg(tmp_path)
    src.write_text("x = 1\n", encoding="utf-8")
    arts = export_artifacts(str(repo), str(src), out)  # no pristine_dir
    assert arts.patch is None and arts.changes == []


def test_export_nongit_ignores_unchanged_preexisting_fused_sibling(tmp_path):
    """A pre-existing framework file matching *fusion*/*_fused* (snapshotted, unchanged)
    must NOT be emitted as a new file nor deleted by restore."""
    repo, src, out = _nongit_pkg(tmp_path)
    sibling = repo / "models" / "other_fusion.py"
    sibling_text = "PRE_EXISTING = 1\n"
    sibling.write_text(sibling_text, encoding="utf-8")
    # _snapshot_fusion_source also snapshots existing fused siblings.
    (out / ".pristine" / "models" / "other_fusion.py").write_text(sibling_text, encoding="utf-8")
    # author edits only the main source
    src.write_text("FUSED = 1\ndef forward(x):\n    return x\n", encoding="utf-8")
    arts = export_artifacts(str(repo), str(src), out, pristine_dir=str(out / ".pristine"))
    assert arts.patch is not None
    assert all(c["path"] != "models/other_fusion.py" for c in arts.changes), (
        "unchanged pre-existing fused sibling must not be reported as a change"
    )
    restore_exported_changes(str(repo), arts, pristine_dir=str(out / ".pristine"))
    assert sibling.is_file() and sibling.read_text() == sibling_text, (
        "restore must not delete an unrelated pre-existing framework file"
    )


def test_export_nongit_emits_new_author_module(tmp_path):
    """An author-created fused module (no pristine snapshot) IS emitted as a new file."""
    repo, src, out = _nongit_pkg(tmp_path)
    new_mod = repo / "models" / "qwen3_fused_kernel.py"
    new_mod.write_text("def fused(): return 1\n", encoding="utf-8")  # no snapshot => new
    arts = export_artifacts(str(repo), str(src), out, pristine_dir=str(out / ".pristine"))
    assert any(c["path"] == "models/qwen3_fused_kernel.py" for c in arts.changes)
    assert "b/models/qwen3_fused_kernel.py" in (out / "fusion.patch").read_text()


def test_restore_nongit_reverts_from_snapshot(tmp_path):
    """Non-git restore rewrites the live source back to the pristine snapshot."""
    repo, src, out = _nongit_pkg(tmp_path)
    src.write_text("EDITED = True\n", encoding="utf-8")
    arts = FusionArtifacts()
    arts.patch = str(out / "fusion.patch")
    arts.changes = [{"path": "models/qwen3.py", "kind": "framework_wiring_edit"}]
    restore_exported_changes(str(repo), arts, pristine_dir=str(out / ".pristine"))
    assert src.read_text() == "def forward(x):\n    return x\n"


def test_export_no_scoped_paths(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _init_repo(repo)
    # source_file empty and no fused-marked untracked files -> nothing scoped.
    arts = export_artifacts(str(repo), "", tmp_path / "out")
    assert arts.patch is None and arts.changes == []


def test_export_handles_subprocess_error(tmp_path, monkeypatch):
    repo = tmp_path / "r"
    repo.mkdir()
    _init_repo(repo)

    def boom(*a, **k):
        raise OSError("git gone")

    monkeypatch.setattr(emit, "_fusion_scoped_paths", boom)
    arts = export_artifacts(str(repo), "/src.py", tmp_path / "out")
    assert arts.patch is None and arts.changes == []


def test_restore_noop_without_patch():
    arts = FusionArtifacts()
    # No patch -> returns immediately (no git calls).
    restore_exported_changes("/repo", arts)


def test_restore_skips_empty_path(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _init_repo(repo)
    arts = FusionArtifacts(changes=[{"path": ""}], patch="/tmp/x.patch")
    restore_exported_changes(str(repo), arts)  # empty path is skipped silently


def test_restore_removes_untracked_and_prunes_dirs(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _init_repo(repo)
    sub = repo / "a" / "b"
    sub.mkdir(parents=True)
    f = sub / "foo_fused.py"
    f.write_text("# kernel\n")
    arts = FusionArtifacts(changes=[{"path": "a/b/foo_fused.py"}], patch="/tmp/x.patch")
    restore_exported_changes(str(repo), arts)
    assert not f.exists()
    # empty parent dirs pruned back toward repo root
    assert not (repo / "a").exists()


def test_export_nongit_honors_custom_patch_name(tmp_path):
    """A per-sibling ``patch_name`` writes ``out/<patch_name>`` and NOT the legacy
    ``fusion.patch``, so N keepers exported into one dir do not clobber each other."""
    repo, src, out = _nongit_pkg(tmp_path)
    src.write_text("FUSED = 1\ndef forward(x):\n    return x\n", encoding="utf-8")
    arts = export_artifacts(
        str(repo), str(src), out, pristine_dir=str(out / ".pristine"), patch_name="fusion_0.patch"
    )
    assert arts.patch == str(out / "fusion_0.patch")
    assert (out / "fusion_0.patch").is_file()
    assert not (out / "fusion.patch").exists()


def test_export_nongit_two_siblings_do_not_overwrite(tmp_path):
    """Exporting two recipes into the same out dir under distinct names keeps both
    patch files intact (the multi-patch nomination invariant)."""
    repo, src, out = _nongit_pkg(tmp_path)
    # sibling 0 edits the main source
    src.write_text("A = 1\ndef forward(x):\n    return x\n", encoding="utf-8")
    a0 = export_artifacts(
        str(repo), str(src), out, pristine_dir=str(out / ".pristine"), patch_name="fusion_0.patch"
    )
    text0 = (out / "fusion_0.patch").read_text()
    # sibling 1 makes a different edit to the same source; exported under a new name
    src.write_text("B = 2\ndef forward(x):\n    return x\n", encoding="utf-8")
    a1 = export_artifacts(
        str(repo), str(src), out, pristine_dir=str(out / ".pristine"), patch_name="fusion_1.patch"
    )
    assert a0.patch != a1.patch
    # sibling 0's file is untouched by sibling 1's export
    assert (out / "fusion_0.patch").read_text() == text0
    assert "+A = 1" in (out / "fusion_0.patch").read_text()
    assert "+B = 2" in (out / "fusion_1.patch").read_text()


def test_restore_checks_out_tracked_file(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True, text=True)
    f = repo / "wired.py"
    f.write_text("original\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-qm", "base"],
        check=True,
        capture_output=True,
        text=True,
    )
    f.write_text("modified\n")
    arts = FusionArtifacts(changes=[{"path": "wired.py"}], patch="/tmp/x.patch")
    restore_exported_changes(str(repo), arts)
    assert f.read_text() == "original\n"  # tracked file checked out
