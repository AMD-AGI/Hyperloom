# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch coverage for integrate_patch helper functions: framework-root
resolution, git apply / reverse / checkout spawn-failure handling, patch-path
resolution, and the best-effort revert fallback chain."""

from __future__ import annotations

import types
from pathlib import Path

from hyperloom.orchestrator.actions.executors import _git as gitmod
from hyperloom.orchestrator.actions.executors import integrate_patch as ip


class _CP:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


def test_now_iso():
    assert "T" in ip._now_iso()


def test_resolve_framework_root_explicit_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(tmp_path)])
    assert ip._resolve_framework_root(str(tmp_path)) == tmp_path


def test_resolve_framework_root_create_requires_explicit_root(tmp_path, monkeypatch):
    root = tmp_path / "framework"
    root.mkdir()
    create = "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+new\n"
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(root)])
    monkeypatch.setattr(ip, "resolve_session_framework_root", lambda: "")

    assert ip._resolve_framework_root(None, patch_texts=[create]) is None
    assert ip._resolve_framework_root(str(root), patch_texts=[create]) == root


def test_resolve_framework_root_rejects_ambiguous_matches(tmp_path, monkeypatch):
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        root.mkdir()
        (root / "file.py").write_text("old\n", encoding="utf-8")
    patch = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n"
    monkeypatch.setattr(
        ip,
        "resolve_source_file_allowlist",
        lambda: [str(root) for root in roots],
    )
    monkeypatch.setattr(ip, "resolve_session_framework_root", lambda: "")

    assert ip._resolve_framework_root(None, patch_texts=[patch]) is None


def test_resolve_framework_root_explicit_outside_allowlist_rejected(tmp_path, monkeypatch):
    """An explicit override outside the source allowlist must not be honoured."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(allowed)])
    assert ip._resolve_framework_root(str(outside)) is None


def test_resolve_framework_root_explicit_nested_under_allowlist(tmp_path, monkeypatch):
    """A subdirectory of an allowlisted root may be selected explicitly."""
    fw = tmp_path / "fw"
    nested = fw / "pkg"
    nested.mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])
    assert ip._resolve_framework_root(str(nested)) == nested


def test_resolve_framework_root_accepts_non_git_installed_package(tmp_path, monkeypatch):
    packages = tmp_path / "lib" / "python3.12" / "site-packages"
    package = packages / "unrelated_package"
    package.mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(packages)])
    assert ip._resolve_framework_root(str(package)) == package


def test_resolve_framework_root_slash_override_rejected(tmp_path, monkeypatch):
    """An explicit ``/`` override must never be returned as the framework root."""
    fw = tmp_path / "fw"
    fw.mkdir()
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])
    assert ip._resolve_framework_root("/") is None


def test_resolve_framework_root_unresolvable_explicit_rejected(tmp_path, monkeypatch):
    """Broken symlinks for explicit overrides are rejected without raising."""
    fw = tmp_path / "fw"
    fw.mkdir()
    broken = tmp_path / "broken-link"
    broken.symlink_to(tmp_path / "missing-target")
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(fw)])
    assert ip._resolve_framework_root(str(broken)) is None


def test_resolve_framework_root_explicit_missing_rejected(tmp_path, monkeypatch):
    gitroot = tmp_path / "fw"
    (gitroot / ".git").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(gitroot)])
    assert ip._resolve_framework_root("/no/such/dir") is None


def test_resolve_framework_root_non_git_fallback(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [str(plain)])
    assert ip._resolve_framework_root(None) == plain


def test_resolve_framework_root_none(monkeypatch):
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [])
    assert ip._resolve_framework_root(None) is None


def test_run_git_apply_spawn_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(gitmod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    ok, err = ip._run_git_apply(tmp_path, tmp_path / "p.patch", p_level=1, three_way=False, check_only=True)
    assert ok is False
    assert "spawn failed" in err


def test_run_git_apply_success(tmp_path, monkeypatch):
    monkeypatch.setattr(gitmod.subprocess, "run", lambda *a, **k: _CP(0, ""))
    ok, err = ip._run_git_apply(tmp_path, tmp_path / "p.patch", p_level=1, three_way=True, check_only=True)
    assert ok is True


def test_preflight_missing_targets_read_error(tmp_path):
    # A directory path -> read_text raises OSError -> skipped
    records = ip._preflight_missing_targets(tmp_path, [tmp_path])
    assert records == []


def test_preflight_missing_targets_records(tmp_path, monkeypatch):
    patch = tmp_path / "p.patch"
    patch.write_text("--- a/ghost.py\n+++ b/ghost.py\n", encoding="utf-8")
    monkeypatch.setattr(ip, "patch_targets_missing", lambda text, root: ["a/ghost.py"])
    records = ip._preflight_missing_targets(tmp_path, [patch])
    assert records[0]["missing_targets"] == ["a/ghost.py"]


def test_git_apply_check_only_after_detect(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "_detect_p_level", lambda *a, **k: 2)
    ok, err = ip._git_apply(tmp_path, tmp_path / "p.patch", check_only=True)
    assert ok is True and err == ""


def test_git_apply_no_level(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "_detect_p_level", lambda *a, **k: None)
    monkeypatch.setattr(ip, "_run_git_apply", lambda *a, **k: (False, "no apply"))
    ok, err = ip._git_apply(tmp_path, tmp_path / "p.patch")
    assert ok is False


def test_git_apply_real_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "_detect_p_level", lambda *a, **k: 1)
    monkeypatch.setattr(ip, "_run_git_apply", lambda *a, **k: (True, ""))
    ok, _ = ip._git_apply(tmp_path, tmp_path / "p.patch", check_only=False)
    assert ok is True


def test_git_apply_reverse_success(tmp_path, monkeypatch):
    # check passes at level 1, then real reverse-apply succeeds
    monkeypatch.setattr(gitmod.subprocess, "run", lambda *a, **k: _CP(0, ""))
    ok, err = ip._git_apply_reverse(tmp_path, tmp_path / "p.patch")
    assert ok is True


def test_git_apply_reverse_check_spawn_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(gitmod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    ok, err = ip._git_apply_reverse(tmp_path, tmp_path / "p.patch")
    assert ok is False and "spawn failed" in err


def test_git_apply_reverse_real_fails(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _run(*a, **k):
        calls["n"] += 1
        # first call (--check) ok, second (real) fails
        return _CP(0, "") if calls["n"] == 1 else _CP(1, "reverse failed")

    monkeypatch.setattr(gitmod.subprocess, "run", _run)
    ok, err = ip._git_apply_reverse(tmp_path, tmp_path / "p.patch")
    assert ok is False and "reverse failed" in err


def test_git_apply_reverse_real_spawn_fail(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _run(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _CP(0, "")  # --check passes
        raise FileNotFoundError("git")  # real reverse-apply spawn fails

    monkeypatch.setattr(gitmod.subprocess, "run", _run)
    ok, err = ip._git_apply_reverse(tmp_path, tmp_path / "p.patch")
    assert ok is False and "spawn failed" in err


def test_git_apply_reverse_no_level(tmp_path, monkeypatch):
    monkeypatch.setattr(gitmod.subprocess, "run", lambda *a, **k: _CP(1, "no"))
    ok, err = ip._git_apply_reverse(tmp_path, tmp_path / "p.patch")
    assert ok is False and "no matching -p level" in err


_DIFF = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1 +1 @@\n-old\n+new\n"


def test_patch_touched_paths_returns_only_patched_file(tmp_path):
    # Patch-targeted file exists; an unrelated dirty file does not.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("new\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("dirty\n", encoding="utf-8")
    patch = tmp_path / "p.patch"
    patch.write_text(_DIFF, encoding="utf-8")

    touched = ip._patch_touched_paths(tmp_path, [patch])
    assert touched == ["pkg/mod.py"]
    assert "unrelated.py" not in touched


def test_patch_touched_paths_skips_unresolvable_and_creations(tmp_path):
    # Creation patch: new file present (strip level 1), old is /dev/null.
    (tmp_path / "new.py").write_text("content\n", encoding="utf-8")
    create = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+content\n"
    patch = tmp_path / "c.patch"
    patch.write_text(create, encoding="utf-8")
    assert ip._patch_touched_paths(tmp_path, [patch]) == ["new.py"]


def test_patch_touched_paths_emits_deleted_path(tmp_path):
    """A pure-deletion patch emits the OLD path so git add -A stages the removal.

    Post-apply the file is gone (new == /dev/null); the old path must still be
    returned, else the KEEP commits nothing and a later REVERT resurrects it.
    """
    delete = "--- a/pkg/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-content\n"
    patch = tmp_path / "d.patch"
    patch.write_text(delete, encoding="utf-8")
    assert ip._patch_touched_paths(tmp_path, [patch]) == ["pkg/gone.py"]


def test_patch_touched_paths_mixed_create_and_delete(tmp_path):
    """A patch that creates one file and deletes another emits both paths."""
    (tmp_path / "kept.py").write_text("hi\n", encoding="utf-8")
    mixed = "--- /dev/null\n+++ b/kept.py\n@@ -0,0 +1 @@\n+hi\n--- a/dropped.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n"
    patch = tmp_path / "m.patch"
    patch.write_text(mixed, encoding="utf-8")
    assert sorted(ip._patch_touched_paths(tmp_path, [patch])) == [
        "dropped.py",
        "kept.py",
    ]


def test_git_commit_kept_no_paths_is_benign_noop(tmp_path, monkeypatch):
    # Empty path set must never shell out and must report success (no-op).
    called = {"n": 0}

    def _run(*a, **k):
        called["n"] += 1
        return _CP(0, "")

    monkeypatch.setattr(gitmod.subprocess, "run", _run)
    ok, note = ip._git_commit_kept(tmp_path, "msg", [])
    assert ok is True
    assert called["n"] == 0  # never invoked git


def test_git_commit_kept_scopes_add_to_paths(tmp_path, monkeypatch):
    captured = {}

    def _run(cmd, *a, **k):
        captured.setdefault("cmds", []).append(cmd)
        return _CP(0, "")

    monkeypatch.setattr(gitmod.subprocess, "run", _run)
    ok, _ = ip._git_commit_kept(tmp_path, "msg", ["pkg/mod.py"])
    assert ok is True
    add_cmd = captured["cmds"][0]
    # The add must be pathspec-scoped, never a blanket "git add -A" of the tree.
    assert add_cmd[-3:] == ["-A", "--", "pkg/mod.py"]


def test_git_commit_kept_note_is_empty_only_on_a_real_commit(tmp_path):
    """The realized-diff harvest gates on this note: '' means HEAD advanced.

    A no-op commit must report a non-empty note so the caller does not harvest
    the previous KEEP's diff as this KEEP's realized change. This uses real git
    to lock the exact contract the gate depends on.
    """
    import subprocess

    def _git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
        )

    _git("init", "-q")
    _git("config", "user.email", "t@t")
    _git("config", "user.name", "t")
    target = tmp_path / "pkg" / "mod.py"
    target.parent.mkdir(parents=True)
    target.write_text("x = 1\n", encoding="utf-8")

    # A real tree change commits and reports an empty note (HEAD advances).
    ok, note = ip._git_commit_kept(tmp_path, "keep-1", ["pkg/mod.py"])
    assert ok is True
    assert note == ""

    # Re-committing the same, unchanged path is a benign no-op: HEAD does not
    # advance, so the note must be non-empty and the harvest must be skipped.
    ok, note = ip._git_commit_kept(tmp_path, "keep-2", ["pkg/mod.py"])
    assert ok is True
    assert note == "nothing to commit"


def test_git_checkout_clean_spawn_fail(tmp_path, monkeypatch):
    """git checkout spawn failure is reported directly."""
    monkeypatch.setattr(gitmod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    ok, err = ip._git_checkout_clean(tmp_path)
    assert ok is False and "checkout spawn failed" in err


def test_stash_if_dirty_clean_tree(tmp_path, monkeypatch):
    """Clean working tree → returns 'clean'."""
    monkeypatch.setattr(
        gitmod.subprocess, "run", lambda *a, **k: type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    )
    state, note = ip._git_stash_if_dirty(tmp_path)
    assert state == "clean"


def test_stash_if_dirty_stash_success(tmp_path, monkeypatch):
    """Dirty tree + stash succeeds → returns 'stashed'."""
    calls = []

    def _run(cmd, *a, **k):
        calls.append(cmd)
        if "status" in cmd:
            return type("CP", (), {"returncode": 0, "stdout": "M foo.py\n", "stderr": ""})()
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(gitmod.subprocess, "run", _run)
    state, note = ip._git_stash_if_dirty(tmp_path)
    assert state == "stashed"
    assert any("stash" in c for c in calls[-1])


def test_stash_if_dirty_stash_fails(tmp_path, monkeypatch):
    """Dirty tree + stash push fails → returns 'failed'."""

    def _run(cmd, *a, **k):
        if "status" in cmd:
            return type("CP", (), {"returncode": 0, "stdout": "M foo.py\n", "stderr": ""})()
        return type("CP", (), {"returncode": 1, "stdout": "", "stderr": "cannot stash"})()

    monkeypatch.setattr(gitmod.subprocess, "run", _run)
    state, note = ip._git_stash_if_dirty(tmp_path)
    assert state == "failed"
    assert "cannot stash" in note


def test_stash_if_dirty_status_exception(tmp_path, monkeypatch):
    """git status throws → returns 'failed'."""
    monkeypatch.setattr(gitmod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    state, note = ip._git_stash_if_dirty(tmp_path)
    assert state == "failed"


def test_checkout_clean_discards_candidate_dirty_without_stashing(tmp_path):
    """Checkout fallback cleans candidate-owned dirty state without creating a stash."""
    import subprocess as sp

    sp.run(["git", "init", str(tmp_path)], capture_output=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], capture_output=True)
    sp.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], capture_output=True)
    (tmp_path / "f.py").write_text("orig\n")
    sp.run(["git", "-C", str(tmp_path), "add", "-A"], capture_output=True)
    sp.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True)
    (tmp_path / "f.py").write_text("dirty\n")
    (tmp_path / "new_candidate.py").write_text("candidate\n")
    ok, err = ip._git_checkout_clean(tmp_path)
    assert ok is True
    assert (tmp_path / "f.py").read_text() == "orig\n"
    assert not (tmp_path / "new_candidate.py").exists()
    cp = sp.run(["git", "-C", str(tmp_path), "stash", "list"], capture_output=True, text=True)
    assert "hyperloom-auto-stash" not in cp.stdout


def test_restore_stash_if_needed_clean_noop(tmp_path):
    assert ip._git_restore_stash_if_needed(tmp_path, "clean", "") == ""


def test_with_stash_restore_adds_error_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "_git_restore_stash_if_needed", lambda *a, **k: "boom")
    out = ip._with_stash_restore(tmp_path, "stashed", "stash@{0}", {"status": "kept"})
    assert out["status"] == "kept"
    assert out["stash_restore_error"] == "boom"


def test_resolve_patch_paths_scan(tmp_path):
    base = tmp_path / "patches"
    base.mkdir()
    (base / "a.patch").write_text("x", encoding="utf-8")
    (base / "b.diff").write_text("y", encoding="utf-8")
    out = ip._resolve_patch_paths(specialist_workspace=tmp_path, explicit_patches=None, done_payload=None)
    names = sorted(p.name for p in out)
    assert names == ["a.patch", "b.diff"]


def test_resolve_patch_paths_missing_logged(tmp_path):
    out = ip._resolve_patch_paths(
        specialist_workspace=tmp_path, explicit_patches=["/no/such/file.patch"], done_payload=None
    )
    assert out == []


def test_resolve_patch_paths_from_done_payload(tmp_path):
    p = tmp_path / "x.patch"
    p.write_text("z", encoding="utf-8")
    out = ip._resolve_patch_paths(
        specialist_workspace=tmp_path, explicit_patches=None, done_payload={"patches_written": [str(p)]}
    )
    assert out[0].name == "x.patch"


def test_resolve_patch_paths_drops_outside_workspace(tmp_path):
    """An absolute patch path outside the specialist workspace is dropped."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # A real, existing file that lives OUTSIDE the workspace.
    outside = tmp_path / "evil.patch"
    outside.write_text("z", encoding="utf-8")
    out = ip._resolve_patch_paths(specialist_workspace=workspace, explicit_patches=[str(outside)], done_payload=None)
    assert out == []


def test_resolve_patch_paths_accepts_inside_workspace(tmp_path):
    """A patch inside the workspace (or its worktree) is accepted."""
    workspace = tmp_path / "ws"
    (workspace / "worktree" / "patches").mkdir(parents=True)
    good = workspace / "worktree" / "patches" / "ok.patch"
    good.write_text("z", encoding="utf-8")
    out = ip._resolve_patch_paths(specialist_workspace=workspace, explicit_patches=[str(good)], done_payload=None)
    assert [p.name for p in out] == ["ok.patch"]


def test_resolve_patch_paths_containment_survives_symlinked_workspace(tmp_path):
    """A symlinked workspace root still matches (both sides resolved)."""
    real = tmp_path / "real_ws"
    (real / "patches").mkdir(parents=True)
    good = real / "patches" / "ok.patch"
    good.write_text("z", encoding="utf-8")
    link = tmp_path / "link_ws"
    link.symlink_to(real)
    out = ip._resolve_patch_paths(
        specialist_workspace=link, explicit_patches=[str(link / "patches" / "ok.patch")], done_payload=None
    )
    assert [p.name for p in out] == ["ok.patch"]


def test_read_done_payload(tmp_path):
    assert ip._read_done_payload(tmp_path) is None
    (tmp_path / "specialist_done.json").write_text("{bad", encoding="utf-8")
    assert ip._read_done_payload(tmp_path) is None
    (tmp_path / "specialist_done.json").write_text('{"ok": 1}', encoding="utf-8")
    assert ip._read_done_payload(tmp_path) == {"ok": 1}


def _executor():
    return ip.IntegratePatchExecutor(session_dir=None)


def test_revert_patches_none_root():
    ex = _executor()
    assert ex._revert_patches(None, [Path("/x")]) == []


def test_revert_patches_reverse_ok(tmp_path, monkeypatch):
    ex = _executor()
    monkeypatch.setattr(ip, "_git_apply_reverse", lambda r, p: (True, ""))
    applied = [tmp_path / "a.patch", tmp_path / "b.patch"]
    reverted = ex._revert_patches(tmp_path, applied)
    assert set(reverted) == set(applied)


def test_revert_patches_checkout_fallback(tmp_path, monkeypatch):
    ex = _executor()
    monkeypatch.setattr(ip, "_git_apply_reverse", lambda r, p: (False, "boom"))
    monkeypatch.setattr(ip, "_git_checkout_clean", lambda r: (True, ""))
    applied = [tmp_path / "a.patch", tmp_path / "b.patch"]
    reverted = ex._revert_patches(tmp_path, applied)
    assert set(reverted) == set(applied)  # checkout reverts all


def test_revert_patches_checkout_fails(tmp_path, monkeypatch):
    ex = _executor()
    monkeypatch.setattr(ip, "_git_apply_reverse", lambda r, p: (False, "boom"))
    monkeypatch.setattr(ip, "_git_checkout_clean", lambda r: (False, "denied"))
    applied = [tmp_path / "a.patch"]
    reverted = ex._revert_patches(tmp_path, applied)
    assert reverted == []


class _Verdict:
    def __init__(self, verdict: str):
        self._v = verdict

    def get_specialist_patch_verdict(self, tid: str) -> str:
        return self._v


def test_enforce_critic_gate_noop_when_no_shared_state():
    assert ip._enforce_critic_gate(None, "spec") is None


def test_enforce_critic_gate_passes_on_permissive_verdict():
    assert ip._enforce_critic_gate(_Verdict("approve"), "spec") is None
    assert ip._enforce_critic_gate(_Verdict("advise"), "spec") is None


def test_enforce_critic_gate_rejects_on_non_permissive_verdict():
    out = ip._enforce_critic_gate(_Verdict("reject"), "spec-1")
    assert out is not None
    assert out["status"] == "rejected_by_critic"
    assert out["specialist_task_id"] == "spec-1"
    assert out["patches_applied"] == []
    assert "reject" in out["reason"]


def test_enforce_critic_gate_rejects_when_no_verdict_on_record():
    out = ip._enforce_critic_gate(_Verdict(""), "spec-2")
    assert out is not None
    assert out["status"] == "rejected_by_critic"
    assert "no Critic verdict on record" in out["reason"]


def test_enforce_critic_gate_handles_state_without_verdict_method():
    class _NoMethod:
        pass

    # AttributeError on get_specialist_patch_verdict is treated as "no verdict".
    out = ip._enforce_critic_gate(_NoMethod(), "spec-4")
    assert out is not None
    assert out["status"] == "rejected_by_critic"


def test_upstream_pr_lane_refuses_an_unreviewed_candidate(tmp_path: Path) -> None:
    """The lane fetches a diff from a remote and applies it to the live tree.

    PolicyGate does not re-validate a queued or resume-dispatched row, which is
    why the specialist lane gates again in the executor; this lane ran the
    fetch and the apply with no verdict check of its own.
    """
    ex = ip.IntegratePatchExecutor(session_dir=tmp_path)
    ctx = types.SimpleNamespace(task=types.SimpleNamespace(task_id="t-cand"))
    params = {
        "candidate": {"repo": "vllm-project/vllm", "pr_number": 1015},
        "framework_agent_candidate_id": "vllm-project/vllm#1015",
    }

    out = ex._stage_resolve_upstream_pr(ctx, params, _Verdict("reject"))

    assert out is not None
    assert out["status"] == "rejected_by_critic"
    assert "patches" not in params


def _git_tree(root: Path) -> None:
    """Initialise a git checkout with one committed file."""
    import subprocess as sp

    sp.run(["git", "init", str(root)], capture_output=True)
    sp.run(["git", "-C", str(root), "config", "user.email", "t@t"], capture_output=True)
    sp.run(["git", "-C", str(root), "config", "user.name", "T"], capture_output=True)
    (root / "keep.py").write_text("original\n", encoding="utf-8")
    sp.run(["git", "-C", str(root), "add", "-A"], capture_output=True)
    sp.run(["git", "-C", str(root), "commit", "-q", "-m", "base"], capture_output=True)


def test_harvest_realized_diff_reads_the_keep_commit(tmp_path):
    """The KEEP is already committed, so its own commit is the realized change."""
    import subprocess as sp

    from hyperloom.orchestrator.actions.executors._patch_snapshot import harvest_realized_diff

    root = tmp_path / "framework"
    root.mkdir()
    _git_tree(root)
    (root / "keep.py").write_text("patched\n", encoding="utf-8")
    (root / "generated.py").write_text("side effect\n", encoding="utf-8")
    sp.run(["git", "-C", str(root), "add", "-A", "--", "keep.py", "generated.py"], capture_output=True)
    sp.run(["git", "-C", str(root), "commit", "-q", "-m", "keep"], capture_output=True)

    written = harvest_realized_diff(root, ["keep.py", "generated.py"], tmp_path / "out" / "realized.patch")

    assert written == str(tmp_path / "out" / "realized.patch")
    text = Path(written).read_text(encoding="utf-8")
    assert "-original" in text and "+patched" in text
    # A file the delivered patch never named still travels.
    assert "generated.py" in text


def test_harvest_realized_diff_returns_empty_without_a_change(tmp_path):
    from hyperloom.orchestrator.actions.executors._patch_snapshot import harvest_realized_diff

    root = tmp_path / "framework"
    root.mkdir()
    _git_tree(root)

    assert harvest_realized_diff(root, ["keep.py"], tmp_path / "realized.patch") == ""
    assert not (tmp_path / "realized.patch").exists()


def test_harvest_realized_diff_returns_empty_outside_git(tmp_path):
    from hyperloom.orchestrator.actions.executors._patch_snapshot import harvest_realized_diff

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "keep.py").write_text("x\n", encoding="utf-8")

    assert harvest_realized_diff(plain, ["keep.py"], tmp_path / "realized.patch") == ""


def test_harvest_realized_diff_refuses_an_empty_path_set(tmp_path):
    from hyperloom.orchestrator.actions.executors._patch_snapshot import harvest_realized_diff

    assert harvest_realized_diff(tmp_path, [], tmp_path / "realized.patch") == ""
