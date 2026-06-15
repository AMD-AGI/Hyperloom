# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Branch coverage for integrate_patch helper functions: framework-root
resolution, git apply / reverse / checkout spawn-failure handling, patch-path
resolution, and the best-effort revert fallback chain."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from inference_optimizer.orchestrator.action_executors import integrate_patch as ip


class _CP:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


# ---- _now_iso -------------------------------------------------------------
def test_now_iso():
    assert "T" in ip._now_iso()


# ---- _resolve_framework_root ----------------------------------------------
def test_resolve_framework_root_explicit_dir(tmp_path):
    assert ip._resolve_framework_root(str(tmp_path)) == tmp_path


def test_resolve_framework_root_explicit_missing_then_git(tmp_path, monkeypatch):
    gitroot = tmp_path / "fw"
    (gitroot / ".git").mkdir(parents=True)
    monkeypatch.setattr(ip, "resolve_source_file_allowlist",
                        lambda: [str(gitroot)])
    # explicit doesn't exist -> warn + fall back to git allowlist entry
    assert ip._resolve_framework_root("/no/such/dir") == gitroot


def test_resolve_framework_root_non_git_fallback(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setattr(ip, "resolve_source_file_allowlist",
                        lambda: [str(plain)])
    assert ip._resolve_framework_root(None) == plain


def test_resolve_framework_root_none(monkeypatch):
    monkeypatch.setattr(ip, "resolve_source_file_allowlist", lambda: [])
    assert ip._resolve_framework_root(None) is None


# ---- _run_git_apply spawn failure -----------------------------------------
def test_run_git_apply_spawn_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(ip.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    ok, err = ip._run_git_apply(tmp_path, tmp_path / "p.patch",
                                p_level=1, three_way=False, check_only=True)
    assert ok is False
    assert "spawn failed" in err


def test_run_git_apply_success(tmp_path, monkeypatch):
    monkeypatch.setattr(ip.subprocess, "run", lambda *a, **k: _CP(0, ""))
    ok, err = ip._run_git_apply(tmp_path, tmp_path / "p.patch",
                                p_level=1, three_way=True, check_only=True)
    assert ok is True


# ---- _preflight_missing_targets read error --------------------------------
def test_preflight_missing_targets_read_error(tmp_path):
    # A directory path -> read_text raises OSError -> skipped
    records = ip._preflight_missing_targets(tmp_path, [tmp_path])
    assert records == []


def test_preflight_missing_targets_records(tmp_path, monkeypatch):
    patch = tmp_path / "p.patch"
    patch.write_text("--- a/ghost.py\n+++ b/ghost.py\n", encoding="utf-8")
    monkeypatch.setattr(ip, "patch_targets_missing",
                        lambda text, root: ["a/ghost.py"])
    records = ip._preflight_missing_targets(tmp_path, [patch])
    assert records[0]["missing_targets"] == ["a/ghost.py"]


# ---- _git_apply check_only after detect -----------------------------------
def test_git_apply_check_only_after_detect(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "_detect_p_level", lambda *a, **k: 2)
    ok, err = ip._git_apply(tmp_path, tmp_path / "p.patch", check_only=True)
    assert ok is True and err == ""


def test_git_apply_no_level(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "_detect_p_level", lambda *a, **k: None)
    monkeypatch.setattr(ip, "_run_git_apply",
                        lambda *a, **k: (False, "no apply"))
    ok, err = ip._git_apply(tmp_path, tmp_path / "p.patch")
    assert ok is False


def test_git_apply_real_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "_detect_p_level", lambda *a, **k: 1)
    monkeypatch.setattr(ip, "_run_git_apply", lambda *a, **k: (True, ""))
    ok, _ = ip._git_apply(tmp_path, tmp_path / "p.patch", check_only=False)
    assert ok is True


# ---- _git_apply_reverse ---------------------------------------------------
def test_git_apply_reverse_success(tmp_path, monkeypatch):
    # check passes at level 1, then real reverse-apply succeeds
    monkeypatch.setattr(ip.subprocess, "run", lambda *a, **k: _CP(0, ""))
    ok, err = ip._git_apply_reverse(tmp_path, tmp_path / "p.patch")
    assert ok is True


def test_git_apply_reverse_check_spawn_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(ip.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    ok, err = ip._git_apply_reverse(tmp_path, tmp_path / "p.patch")
    assert ok is False and "spawn failed" in err


def test_git_apply_reverse_real_fails(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _run(*a, **k):
        calls["n"] += 1
        # first call (--check) ok, second (real) fails
        return _CP(0, "") if calls["n"] == 1 else _CP(1, "reverse failed")

    monkeypatch.setattr(ip.subprocess, "run", _run)
    ok, err = ip._git_apply_reverse(tmp_path, tmp_path / "p.patch")
    assert ok is False and "reverse failed" in err


def test_git_apply_reverse_real_spawn_fail(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _run(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _CP(0, "")  # --check passes
        raise FileNotFoundError("git")  # real reverse-apply spawn fails

    monkeypatch.setattr(ip.subprocess, "run", _run)
    ok, err = ip._git_apply_reverse(tmp_path, tmp_path / "p.patch")
    assert ok is False and "spawn failed" in err


def test_git_apply_reverse_no_level(tmp_path, monkeypatch):
    monkeypatch.setattr(ip.subprocess, "run", lambda *a, **k: _CP(1, "no"))
    ok, err = ip._git_apply_reverse(tmp_path, tmp_path / "p.patch")
    assert ok is False and "no matching -p level" in err


# ---- _patch_touched_paths / commit scoping --------------------------------
_DIFF = (
    "--- a/pkg/mod.py\n"
    "+++ b/pkg/mod.py\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)


def test_patch_touched_paths_returns_only_patched_file(tmp_path):
    # Source file the patch targets exists; an unrelated dirty file does NOT.
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
    create = (
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1 @@\n"
        "+content\n"
    )
    patch = tmp_path / "c.patch"
    patch.write_text(create, encoding="utf-8")
    assert ip._patch_touched_paths(tmp_path, [patch]) == ["new.py"]


def test_git_commit_kept_no_paths_is_benign_noop(tmp_path, monkeypatch):
    # Empty path set must never shell out and must report success (no-op).
    called = {"n": 0}

    def _run(*a, **k):
        called["n"] += 1
        return _CP(0, "")

    monkeypatch.setattr(ip.subprocess, "run", _run)
    ok, note = ip._git_commit_kept(tmp_path, "msg", [])
    assert ok is True
    assert called["n"] == 0  # never invoked git


def test_git_commit_kept_scopes_add_to_paths(tmp_path, monkeypatch):
    captured = {}

    def _run(cmd, *a, **k):
        captured.setdefault("cmds", []).append(cmd)
        return _CP(0, "")

    monkeypatch.setattr(ip.subprocess, "run", _run)
    ok, _ = ip._git_commit_kept(tmp_path, "msg", ["pkg/mod.py"])
    assert ok is True
    add_cmd = captured["cmds"][0]
    # The add must be pathspec-scoped, never a blanket "git add -A" of the tree.
    assert add_cmd[-3:] == ["-A", "--", "pkg/mod.py"]


# ---- _git_checkout_clean spawn failure ------------------------------------
def test_git_checkout_clean_spawn_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(ip.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    ok, err = ip._git_checkout_clean(tmp_path)
    assert ok is False and "spawn failed" in err


# ---- _resolve_patch_paths -------------------------------------------------
def test_resolve_patch_paths_scan(tmp_path):
    base = tmp_path / "patches"
    base.mkdir()
    (base / "a.patch").write_text("x", encoding="utf-8")
    (base / "b.diff").write_text("y", encoding="utf-8")
    out = ip._resolve_patch_paths(
        specialist_workspace=tmp_path, explicit_patches=None,
        done_payload=None)
    names = sorted(p.name for p in out)
    assert names == ["a.patch", "b.diff"]


def test_resolve_patch_paths_missing_logged(tmp_path):
    out = ip._resolve_patch_paths(
        specialist_workspace=tmp_path,
        explicit_patches=["/no/such/file.patch"],
        done_payload=None)
    assert out == []


def test_resolve_patch_paths_from_done_payload(tmp_path):
    p = tmp_path / "x.patch"
    p.write_text("z", encoding="utf-8")
    out = ip._resolve_patch_paths(
        specialist_workspace=tmp_path, explicit_patches=None,
        done_payload={"patches_written": [str(p)]})
    assert out[0].name == "x.patch"


# ---- _read_done_payload ---------------------------------------------------
def test_read_done_payload(tmp_path):
    assert ip._read_done_payload(tmp_path) is None
    (tmp_path / "specialist_done.json").write_text("{bad", encoding="utf-8")
    assert ip._read_done_payload(tmp_path) is None
    (tmp_path / "specialist_done.json").write_text('{"ok": 1}', encoding="utf-8")
    assert ip._read_done_payload(tmp_path) == {"ok": 1}


# ---- _revert_patches fallback chain ---------------------------------------
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
