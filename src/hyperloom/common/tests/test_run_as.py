# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``common/run_as``: resolve a specialist user and start a child as it."""

from __future__ import annotations

import contextlib
import os
import pwd
import stat
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.common import run_as as run_as_mod
from hyperloom.common.run_as import RunAsError, resolve_run_as, specialist_run_as

UNPRIVILEGED = "nobody"


def _account_exists(name: str) -> bool:
    try:
        pwd.getpwnam(name)
    except KeyError:
        return False
    return True


needs_root = pytest.mark.skipif(
    os.geteuid() != 0 or not _account_exists(UNPRIVILEGED),
    reason=f"demoting needs root (setuid) and an existing {UNPRIVILEGED!r} account; creates no users",
)


@pytest.fixture
def shared_tmp() -> Iterator[Path]:
    """A temp dir the demoted account can traverse (pytest's tmp_path is 0700)."""
    with tempfile.TemporaryDirectory(prefix="run_as.") as raw:
        path = Path(raw)
        path.chmod(0o755)
        yield path


def _run_as_unprivileged(script: str) -> subprocess.CompletedProcess[str]:
    """Run a ``/bin/sh`` script as the unprivileged account (the venv's python is not reachable by it)."""
    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    return subprocess.run(["/bin/sh", "-c", script], capture_output=True, text=True, check=False, **spec.popen_kwargs())


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_blank_means_no_demote(value, monkeypatch):
    assert resolve_run_as(value) is None
    if value is None:
        monkeypatch.delenv(run_as_mod.SPECIALIST_USER_ENV, raising=False)
    else:
        monkeypatch.setenv(run_as_mod.SPECIALIST_USER_ENV, value)
    assert specialist_run_as() is None


def test_unknown_user_is_refused():
    with pytest.raises(RunAsError, match="no such user"):
        resolve_run_as("no-such-user-d16")


def test_root_is_refused(monkeypatch):
    monkeypatch.setenv(run_as_mod.SPECIALIST_USER_ENV, "root")
    with pytest.raises(RunAsError, match="uid 0"):
        specialist_run_as()


@pytest.mark.skipif(not _account_exists(UNPRIVILEGED), reason=f"no {UNPRIVILEGED!r} account")
def test_the_apps_own_user_is_refused(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: pwd.getpwnam(UNPRIVILEGED).pw_uid)
    with pytest.raises(RunAsError, match="app's own user"):
        resolve_run_as(UNPRIVILEGED)


@pytest.mark.skipif(not _account_exists(UNPRIVILEGED), reason=f"no {UNPRIVILEGED!r} account")
def test_membership_of_group_zero_is_refused(monkeypatch):
    monkeypatch.setattr(os, "getgrouplist", lambda _name, gid: [gid, 0])
    with pytest.raises(RunAsError, match="gid 0"):
        resolve_run_as(UNPRIVILEGED)


@pytest.mark.skipif(not _account_exists(UNPRIVILEGED), reason=f"no {UNPRIVILEGED!r} account")
def test_resolution_names_the_account_and_its_groups():
    spec = resolve_run_as(UNPRIVILEGED)
    entry = pwd.getpwnam(UNPRIVILEGED)
    assert spec is not None
    assert (spec.uid, spec.gid, spec.home) == (entry.pw_uid, entry.pw_gid, entry.pw_dir)
    kwargs = spec.popen_kwargs()
    assert (kwargs["user"], kwargs["group"]) == (entry.pw_uid, entry.pw_gid)
    assert entry.pw_gid in kwargs["extra_groups"]


@pytest.mark.skipif(not _account_exists(UNPRIVILEGED), reason=f"no {UNPRIVILEGED!r} account")
def test_apply_env_sets_identity_and_marks_git_safe_directories():
    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    env = {"HOME": "/app-home", "USER": "app", "USERNAME": "app"}
    spec.apply_env(env, safe_dirs=[Path("/w/one"), Path("/w/two")])
    assert env["HOME"] == spec.home
    assert env["USER"] == env["LOGNAME"] == env["USERNAME"] == UNPRIVILEGED
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert (env["GIT_CONFIG_KEY_1"], env["GIT_CONFIG_VALUE_1"]) == ("safe.directory", "/w/two")


@needs_root
def test_child_runs_as_the_named_uid_and_group():
    out = _run_as_unprivileged("id -u; id -g; id -G")
    entry = pwd.getpwnam(UNPRIVILEGED)
    assert out.returncode == 0, out.stderr
    uid, gid, groups = out.stdout.split("\n")[:3]
    assert (uid, gid) == (str(entry.pw_uid), str(entry.pw_gid))
    assert "0" not in groups.split()


@needs_root
def test_child_cannot_read_an_app_owned_private_file(shared_tmp):
    private_file = shared_tmp / "private_file"
    private_file.write_text("marker", encoding="utf-8")
    private_file.chmod(0o600)
    out = _run_as_unprivileged(f"cat {private_file}")
    assert out.returncode != 0
    assert "Permission denied" in out.stderr
    assert "marker" not in out.stdout


@needs_root
def test_child_cannot_read_the_apps_environ():
    out = _run_as_unprivileged(f"cat /proc/{os.getpid()}/environ")
    assert out.returncode != 0
    assert "Permission denied" in out.stderr


@needs_root
def test_shared_tree_is_writable_by_the_child_and_still_the_apps(shared_tmp):
    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    tree = shared_tmp / "wt"
    (tree / "pkg").mkdir(parents=True)
    (tree / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    outside = shared_tmp / "outside"
    outside.write_text("untouched", encoding="utf-8")
    outside.chmod(0o600)
    (tree / "link").symlink_to(outside)
    spec.share_tree(tree)
    out = _run_as_unprivileged(
        f"cd {tree} && echo x = 2 > pkg/mod.py && echo y = 1 > pkg/new.py && mkdir made && echo z > made/f"
    )
    assert out.returncode == 0, out.stderr
    assert (tree / "pkg" / "mod.py").read_text(encoding="utf-8") == "x = 2\n"
    assert stat.S_IMODE((tree / "pkg").stat().st_mode) & stat.S_ISGID
    # What the child made carries the shared group, so the app can rewrite it.
    assert (tree / "made" / "f").stat().st_gid == spec.gid
    assert (tree / "made" / "f").stat().st_mode & stat.S_IWGRP
    # The tree is still the app's, and a symlink's target was not followed.
    assert (tree / "pkg").stat().st_uid == os.geteuid()
    assert outside.stat().st_gid != spec.gid
    assert stat.S_IMODE(outside.stat().st_mode) == 0o600


@needs_root
def test_sticky_workspace_keeps_the_apps_files_from_the_child(shared_tmp):
    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    ws = shared_tmp / "ws"
    ws.mkdir()
    (ws / "prompt.md").write_text("p", encoding="utf-8")
    spec.share_tree(ws, sticky=True)
    out = _run_as_unprivileged(f"rm -f {ws}/prompt.md")
    assert out.returncode != 0
    assert (ws / "prompt.md").exists()


@pytest.mark.skipif(not _account_exists(UNPRIVILEGED), reason=f"no {UNPRIVILEGED!r} account")
def test_a_symlink_as_the_root_of_the_shared_tree_is_refused(tmp_path):
    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    target = tmp_path / "app_owned"
    target.mkdir()
    (target / "key").write_text("k", encoding="utf-8")
    (target / "key").chmod(0o600)
    link = tmp_path / "wt"
    link.symlink_to(target)
    with pytest.raises(OSError, match="symlink"):
        spec.share_tree(link)
    assert stat.S_IMODE((target / "key").stat().st_mode) == 0o600


@needs_root
def test_share_tree_leaves_git_alone(shared_tmp):
    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    (shared_tmp / ".git").write_text("gitdir: /elsewhere", encoding="utf-8")
    (shared_tmp / "sub" / ".git").mkdir(parents=True)
    (shared_tmp / "sub" / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    (shared_tmp / "sub" / "f.py").write_text("x", encoding="utf-8")
    before = {
        p: (p.stat().st_gid, stat.S_IMODE(p.stat().st_mode))
        for p in (shared_tmp / ".git", shared_tmp / "sub" / ".git" / "HEAD")
    }
    spec.share_tree(shared_tmp)
    assert {p: (p.stat().st_gid, stat.S_IMODE(p.stat().st_mode)) for p in before} == before
    assert (shared_tmp / "sub" / "f.py").stat().st_gid == spec.gid


@needs_root
def test_demoted_child_files_are_group_writable_and_a_plain_launch_keeps_the_umask(shared_tmp):
    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    spec.share_tree(shared_tmp)
    assert spec.popen_kwargs()["umask"] == 0o007
    out = _run_as_unprivileged(f"umask; cd {shared_tmp} && touch made")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0007"
    assert (shared_tmp / "made").stat().st_mode & stat.S_IWGRP
    # Without a demote nothing sets a umask, so a child inherits the app's own.
    plain = subprocess.run(["/bin/sh", "-c", "umask"], capture_output=True, text=True, check=True)
    app_umask = os.umask(0)
    os.umask(app_umask)
    assert plain.stdout.strip() == f"{app_umask:04o}"


@pytest.fixture
def restore_umask() -> Iterator[None]:
    saved = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(saved)


def test_resolving_the_user_does_not_touch_the_umask(monkeypatch, restore_umask):
    monkeypatch.setenv(run_as_mod.SPECIALIST_USER_ENV, UNPRIVILEGED if _account_exists(UNPRIVILEGED) else "root")
    # Whether the account resolves (non-root runs may be the account) or is refused, the umask must not move.
    with contextlib.suppress(RunAsError):
        specialist_run_as()
    assert os.umask(0o022) == 0o022


@needs_root
def test_read_only_share_gives_the_group_read_and_search_but_not_write(shared_tmp):
    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    (shared_tmp / "d").mkdir()
    (shared_tmp / "d" / "f").write_text("x", encoding="utf-8")
    for p in (shared_tmp / "d", shared_tmp / "d" / "f"):
        p.chmod(0o700 if p.is_dir() else 0o600)
    spec.share_tree(shared_tmp / "d", write=False)
    assert stat.S_IMODE((shared_tmp / "d").stat().st_mode) == 0o2750
    assert stat.S_IMODE((shared_tmp / "d" / "f").stat().st_mode) == 0o640
    assert _run_as_unprivileged(f"cat {shared_tmp}/d/f").stdout == "x"
    assert _run_as_unprivileged(f"echo y > {shared_tmp}/d/f").returncode != 0
    assert _run_as_unprivileged(f"touch {shared_tmp}/d/new").returncode != 0


@needs_root
def test_writable_share_modes(shared_tmp):
    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    (shared_tmp / "d").mkdir()
    (shared_tmp / "d" / "f").write_text("x", encoding="utf-8")
    (shared_tmp / "d").chmod(0o700)
    (shared_tmp / "d" / "f").chmod(0o600)
    spec.share_tree(shared_tmp / "d")
    assert stat.S_IMODE((shared_tmp / "d").stat().st_mode) == 0o2770
    assert stat.S_IMODE((shared_tmp / "d" / "f").stat().st_mode) == 0o660


# ---- review r2: a demoted child cannot make the app write elsewhere, and holds no capability ----
@needs_root
def test_a_child_planting_the_redaction_temp_name_cannot_make_the_app_overwrite_a_file(shared_tmp):
    from hyperloom.common.env_safety import redact_file_in_place

    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    workspace = shared_tmp / "ws"
    workspace.mkdir()
    log = workspace / "process.log"
    log.write_text("token ghp_abcdef123456\n", encoding="utf-8")
    victim = shared_tmp / "victim"  # root's file, which the child can neither read nor write
    victim.write_text("keep me", encoding="utf-8")
    victim.chmod(0o600)
    spec.share_tree(workspace, sticky=True)
    log.chmod(0o600)
    planted = _run_as_unprivileged(f"ln -s {victim} {workspace}/process.log.redacting")
    assert planted.returncode == 0, planted.stderr
    redact_file_in_place(log)
    assert victim.read_text(encoding="utf-8") == "keep me"
    assert "ghp_abcdef" not in log.read_text(encoding="utf-8")
    assert not (workspace / "process.log.redacting").exists()


_CAP_PROBE = (
    "grep -E '^(CapInh|CapPrm|CapEff|CapAmb|NoNewPrivs):' /proc/self/status; "
    "setpriv --reuid=12345 true 2>/dev/null; echo setuid_back_rc=$?"
)
_CAP_ONLY_APP = """
import json, subprocess, sys
a = json.loads(sys.argv[1])
r = subprocess.run(a["cmd"], user=a["uid"], group=a["gid"], extra_groups=a["groups"], capture_output=True, text=True)
print(r.stdout, end="")
"""


def _probe_fields(text: str) -> dict[str, str]:
    return dict(
        line.replace(":", "=").replace("\t", "").split("=", 1)
        for line in text.splitlines()
        if "=" in line or ":" in line
    )


@needs_root
def test_a_demoted_child_of_a_root_app_holds_no_capability_and_cannot_setuid_back():
    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    done = subprocess.run(
        spec.command(["/bin/sh", "-c", _CAP_PROBE]), capture_output=True, text=True, check=False, **spec.popen_kwargs()
    )
    fields = _probe_fields(done.stdout)
    assert {k: int(fields[k], 16) for k in ("CapInh", "CapPrm", "CapEff", "CapAmb")} == dict.fromkeys(
        ("CapInh", "CapPrm", "CapEff", "CapAmb"), 0
    ), done.stdout
    assert fields["NoNewPrivs"] == "1"
    assert fields["setuid_back_rc"] != "0"


@needs_root
def test_a_demoted_child_of_a_capability_only_app_holds_no_capability_and_cannot_setuid_back():
    """The documented non-root setup: the app has CAP_SETUID/SETGID/KILL as ambient capabilities, not uid 0."""
    import json

    python = "/usr/bin/python3"
    if not os.access(python, os.X_OK):
        pytest.skip("needs a system python the unprivileged app uid can run")
    spec = resolve_run_as(UNPRIVILEGED)
    assert spec is not None
    caps = "+setuid,+setgid,+kill"
    app = [
        "setpriv", "--reuid=12345", "--regid=12345", "--clear-groups",
        f"--inh-caps={caps}", f"--ambient-caps={caps}", "--",
        python, "-I", "-c", _CAP_ONLY_APP,
    ]  # fmt: skip
    arguments = {
        "cmd": spec.command(["/bin/sh", "-c", _CAP_PROBE]),
        "uid": spec.uid,
        "gid": spec.gid,
        "groups": list(spec.groups),
    }
    done = subprocess.run([*app, json.dumps(arguments)], capture_output=True, text=True, check=False)
    fields = _probe_fields(done.stdout)
    assert done.returncode == 0, done.stderr
    assert [int(fields[k], 16) for k in ("CapInh", "CapPrm", "CapEff", "CapAmb")] == [0, 0, 0, 0], done.stdout
    assert fields["NoNewPrivs"] == "1"
    assert fields["setuid_back_rc"] != "0"
    # Control: the same app without the drop hands the capabilities over, so the probe can tell.
    bare = {**arguments, "cmd": ["/bin/sh", "-c", _CAP_PROBE]}
    leaked = subprocess.run([*app, json.dumps(bare)], capture_output=True, text=True, check=False)
    assert int(_probe_fields(leaked.stdout)["CapAmb"], 16) != 0, leaked.stdout
    assert _probe_fields(leaked.stdout)["setuid_back_rc"] == "0"


def test_resolving_a_user_without_the_drop_tool_is_refused(monkeypatch):
    monkeypatch.setattr(run_as_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RunAsError, match="setpriv"):
        resolve_run_as(UNPRIVILEGED if _account_exists(UNPRIVILEGED) else "daemon")
