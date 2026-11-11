# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom_common.git_safety``: no program a worktree or its repository's config names runs.

Each case plants a program, runs the command that reaches it through ``app_git`` and finds no trace of it, then plants
the same thing in a second tree and runs the same command with plain git, which must run it: a plant that plain git
does not run would prove nothing.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import http.server
import os
import pty
import shlex
import shutil
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import NoReturn
from dataclasses import dataclass
from pathlib import Path

import pytest

from hyperloom_common import git_safety
from hyperloom_common.git_safety import NO_PROGRAMS, app_git, app_git_diff
from hyperloom_common.shared_storage import current_umask


def _plain_env() -> dict[str, str]:
    """Git with nothing but the repository's config: no ``GIT_*`` of the caller's, no system or global file, no proxy."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and not k.lower().endswith("_proxy")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_TERMINAL_PROMPT="0")
    return env


def _git(repo: Path, *args: str, stdin: str | None = None) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], input=stdin, env=_plain_env(), check=True, capture_output=True, text=True
    )
    return done.stdout


def _commit(repo: Path, message: str = "c") -> None:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t.local", "-c", "user.name=t", "commit", "-qm", message)


@dataclass(frozen=True)
class _Tree:
    """A repository ``base`` with the linked worktree ``worktree``, and two programs that leave ``canary`` behind."""

    root: Path
    base: Path
    worktree: Path
    program: Path  # exits 0
    filter: Path  # passes its input through

    @property
    def canary(self) -> Path:
        return self.root / "ran"

    def config(self, *settings: tuple[str, str]) -> None:
        for key, value in settings:
            _git(self.base, "config", key, value)

    def attributes(self, line: str) -> None:
        (self.worktree / ".gitattributes").write_text(f"{line}\n", encoding="utf-8")

    def change(self) -> None:
        # The committed size, so git must read the file to see the change, and a timestamp a second past the checkout's:
        # a rewrite within the same timestamp tick has the index entry's stat and looks unchanged, read or not.
        changed = self.worktree / "mod.py"
        changed.write_text("new = 2\n", encoding="utf-8")
        later = changed.stat().st_mtime_ns + 1_000_000_000
        os.utime(changed, ns=(later, later))
        (self.worktree / "added.py").write_text("fresh = 3\n", encoding="utf-8")


def _tree(root: Path) -> _Tree:
    root.mkdir()
    tree = _Tree(root, root / "base", root / "worktree", root / "program", root / "filter")
    for program, then in ((tree.program, "exit 0"), (tree.filter, "exec cat")):
        program.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(tree.canary))}\n{then}\n", encoding="utf-8")
        program.chmod(0o755)
    tree.base.mkdir()
    (tree.base / "mod.py").write_text("old = 1\n", encoding="utf-8")
    _git(root, "init", "-q", str(tree.base))
    tree.config(("user.email", "t@t.local"), ("user.name", "t"))
    _commit(tree.base, "base")
    _git(tree.base, "worktree", "add", "-q", "-b", "task", str(tree.worktree))
    return tree


_Plant = Callable[[_Tree], Sequence[str]]


def _proves(tmp_path: Path, plant: _Plant) -> subprocess.CompletedProcess[str]:
    """Run ``plant``'s command by ``app_git`` (nothing runs), then by plain git in a second tree (the plant runs)."""
    app = _tree(tmp_path / "app")
    done = app_git(app.worktree, *plant(app))
    assert not app.canary.exists(), done.stderr
    plain = _tree(tmp_path / "plain")
    proof = subprocess.run(
        ["git", "-C", str(plain.worktree), *plant(plain)],
        stdin=subprocess.DEVNULL,
        env=_plain_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert plain.canary.exists(), f"plain git did not run the plant, so it proves nothing: {proof.stderr}"
    return done


def _filter(program: str, *command: str) -> _Plant:
    def plant(tree: _Tree) -> Sequence[str]:
        tree.config((f"filter.evil.{program}", str(tree.filter)))
        tree.attributes("* filter=evil")
        tree.change()
        return command

    return plant


def _dotted_filter(tree: _Tree) -> Sequence[str]:
    tree.config(("filter.Evil.Name.clean", str(tree.filter)))
    tree.attributes("* filter=Evil.Name")
    tree.change()
    return ("diff", "--no-ext-diff", "HEAD")


def _included_filter(key: str) -> _Plant:
    """A driver the repository's config defines only through a file it includes."""

    def plant(tree: _Tree) -> Sequence[str]:
        included = tree.root / "included"
        included.write_text(f'[filter "evil"]\n\tclean = {tree.filter}\n', encoding="utf-8")
        tree.config((key, str(included)))
        tree.attributes("* filter=evil")
        tree.change()
        return ("diff", "--no-ext-diff", "HEAD")

    return plant


def _diff_program(key: str, *command: str) -> _Plant:
    def plant(tree: _Tree) -> Sequence[str]:
        tree.config((key, str(tree.program)))
        tree.attributes("* diff=evil")
        tree.change()
        return command

    return plant


def _merge_driver(tree: _Tree) -> Sequence[str]:
    tree.config(("merge.evil.driver", f"{tree.program} %O %A %B"))
    _git(tree.worktree, "checkout", "-q", "-b", "other")
    (tree.worktree / "mod.py").write_text("theirs = 1\n", encoding="utf-8")
    _commit(tree.worktree)
    _git(tree.worktree, "checkout", "-q", "task")
    (tree.worktree / "mod.py").write_text("ours = 1\n", encoding="utf-8")
    tree.attributes("* merge=evil")
    _commit(tree.worktree)
    return ("merge", "--no-edit", "other")


def _trailer(tree: _Tree) -> Sequence[str]:
    tree.config(("trailer.ev.command", str(tree.program)))
    message = tree.root / "message"
    message.write_text("subject\n\nbody\n", encoding="utf-8")
    return ("interpret-trailers", str(message))


def _hook(where: str) -> _Plant:
    """A hook in the repository's hooks directory, or in one ``core.hooksPath`` names."""

    def plant(tree: _Tree) -> Sequence[str]:
        hooks = tree.base / ".git" / "hooks" if where == "hooks-dir" else tree.root / "hooks"
        hooks.mkdir(exist_ok=True)
        (hooks / "post-index-change").symlink_to(tree.program)
        if where == "hooksPath":
            tree.config(("core.hooksPath", str(hooks)))
        tree.change()
        return ("add", "-A", "-N")

    return plant


def _configured_hook(tree: _Tree) -> Sequence[str]:
    """A hook defined in config (git 2.54 on), which ``core.hooksPath`` does not reach."""
    tree.config(("hook.evil.command", str(tree.program)), ("hook.evil.event", "pre-commit"))
    tree.change()
    _git(tree.worktree, "add", "-A")
    return ("commit", "-qm", "c")


_CONFIG_HOOKS = "hook.<friendly-name>.command" in _git(Path.cwd(), "help", "--config")


def _fsmonitor(tree: _Tree) -> Sequence[str]:
    tree.config(("core.fsmonitor", str(tree.program)))
    return ("status", "--porcelain")


def _editor(tree: _Tree) -> Sequence[str]:
    tree.config(("core.editor", str(tree.program)))
    tree.change()
    _git(tree.worktree, "add", "-A")
    return ("commit",)


def _sequence_editor(tree: _Tree) -> Sequence[str]:
    tree.config(("sequence.editor", str(tree.program)))
    tree.change()
    _commit(tree.worktree)
    return ("rebase", "-i", "HEAD~1")


def _alternate_refs(tree: _Tree) -> Sequence[str]:
    other = tree.root / "other"
    _git(tree.root, "init", "-q", str(other))
    (other / "f.txt").write_text("x\n", encoding="utf-8")
    _commit(other)
    (tree.base / ".git" / "objects" / "info" / "alternates").write_text(f"{other}/.git/objects\n", encoding="utf-8")
    tree.config(("core.alternateRefsCommand", str(tree.program)))
    return ("rev-list", "--alternate-refs", "HEAD")


_ARMOR = {"openpgp": "PGP SIGNATURE", "ssh": "SSH SIGNATURE", "x509": "SIGNED MESSAGE"}


def _signed(tree: _Tree, kind: str, ref: str = "HEAD") -> None:
    """Point ``ref`` at a child of HEAD whose (forged) signature is ``kind``: checking it starts that kind's program."""
    head = _git(tree.worktree, "rev-parse", "HEAD").strip()
    content = _git(tree.worktree, "rev-parse", "HEAD^{tree}").strip()
    armor = _ARMOR[kind]
    commit = (
        f"tree {content}\nparent {head}\nauthor t <t@t.local> 1 +0000\ncommitter t <t@t.local> 1 +0000\n"
        f"gpgsig -----BEGIN {armor}-----\n iQ==\n -----END {armor}-----\n\nsigned\n"
    )
    _git(
        tree.worktree,
        "update-ref",
        ref,
        _git(tree.worktree, "hash-object", "-t", "commit", "-w", "--stdin", stdin=commit).strip(),
    )


def _verifies(kind: str, key: str) -> _Plant:
    def plant(tree: _Tree) -> Sequence[str]:
        _signed(tree, kind)
        signers = tree.root / "allowed_signers"
        signers.write_text("", encoding="utf-8")
        tree.config((key, str(tree.program)), ("gpg.ssh.allowedSignersFile", str(signers)))
        return ("log", "-1", "--format=%G?")

    return plant


def _shows_signature(tree: _Tree) -> Sequence[str]:
    _signed(tree, "openpgp")
    tree.config(("log.showSignature", "true"), ("gpg.program", str(tree.program)))
    return ("log", "-1", "--format=%s")


def _signs(key: str, *command: str) -> _Plant:
    """``key`` asks git to sign what ``command`` makes, with the program the config names."""

    def plant(tree: _Tree) -> Sequence[str]:
        tree.config((key, "true"), ("gpg.program", str(tree.program)))
        tree.change()
        _git(tree.worktree, "add", "-A")
        return command

    return plant


def _verifies_a_merge(tree: _Tree) -> Sequence[str]:
    _signed(tree, "openpgp", "refs/heads/other")
    tree.config(("merge.verifySignatures", "true"), ("gpg.program", str(tree.program)))
    return ("merge", "--no-edit", "other")


def _default_signing_key(tree: _Tree) -> Sequence[str]:
    tree.config(("gpg.format", "ssh"), ("gpg.ssh.defaultKeyCommand", str(tree.program)))
    tree.change()
    _git(tree.worktree, "add", "-A")
    return ("commit", "-S", "-qm", "signed")


def _auto_gc(tree: _Tree) -> Sequence[str]:
    """Old unreachable objects enough for ``gc --auto``, which then asks ``gc.recentObjectsHook`` what to keep."""
    objects = tree.base / ".git" / "objects"
    found = (n for n in range(4096) if hashlib.sha1(f"blob {len(str(n)) + 1}\0{n}\n".encode()).hexdigest()[:2] == "17")
    old = time.time() - 7200
    for n in (next(found), next(found)):
        blob = _git(tree.base, "hash-object", "-w", "--stdin", stdin=f"{n}\n").strip()
        os.utime(objects / blob[:2] / blob[2:], (old, old))
    tree.config(
        ("maintenance.strategy", "gc"),
        ("gc.auto", "1"),
        ("gc.autoDetach", "false"),
        ("maintenance.autoDetach", "false"),
        ("gc.pruneExpire", "1.hour.ago"),
        ("gc.recentObjectsHook", str(tree.program)),
    )
    tree.change()
    _git(tree.worktree, "add", "-A")
    return ("commit", "-qm", "c")


def _autocorrect(tree: _Tree) -> Sequence[str]:
    tree.config(("alias.zzevilx", f"!{tree.program}"), ("help.autoCorrect", "immediate"))
    return ("zzevily",)


def _submodule(tree: _Tree, attributes: str, *settings: tuple[str, str]) -> Path:
    """A submodule ``sub`` checked out in the worktree, its repository in the worktree's admin dir, with ``settings``."""
    source = tree.root / "subsource"
    _git(tree.root, "init", "-q", str(source))
    (source / "f.txt").write_text("a\n", encoding="utf-8")
    (source / ".gitattributes").write_text(f"{attributes}\n", encoding="utf-8")
    _commit(source)
    _git(tree.worktree, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(source), "sub")
    _commit(tree.worktree)
    sub = tree.worktree / "sub"
    for key, value in settings:
        _git(sub, "config", key, value)
    return sub


def _recursing_reset(tree: _Tree) -> Sequence[str]:
    sub = _submodule(tree, "* filter=evil", ("filter.evil.smudge", str(tree.filter)), ("filter.evil.clean", "cat"))
    (sub / "f.txt").write_text("changed\n", encoding="utf-8")
    tree.config(("submodule.recurse", "true"))
    return ("reset", "-q", "--hard")


def _inline_submodule_diff(tree: _Tree) -> Sequence[str]:
    sub = _submodule(tree, "* diff=evil", ("diff.evil.textconv", str(tree.program)))
    (sub / "f.txt").write_text("changed\n", encoding="utf-8")
    _commit(sub)
    tree.config(("diff.submodule", "diff"))
    return ("diff", "--no-ext-diff", "HEAD")


def _nested_repository(*command: str) -> _Plant:
    """A repository nested in the worktree, tracked as a gitlink, with local changes its own filter would clean."""

    def plant(tree: _Tree) -> Sequence[str]:
        nested = tree.worktree / "vendored"
        _git(tree.root, "init", "-q", str(nested))
        (nested / "f.txt").write_text("a\n", encoding="utf-8")
        _commit(nested)
        _commit(tree.worktree)
        _git(nested, "config", "filter.inner.clean", str(tree.filter))
        (nested / ".gitattributes").write_text("* filter=inner\n", encoding="utf-8")
        (nested / "f.txt").write_text("b\n", encoding="utf-8")
        return command

    return plant


def _lazy_fetch(tree: _Tree, url: str, *settings: tuple[str, str]) -> Sequence[str]:
    """A diff that needs a blob the repository lacks, which git fetches from its promisor remote at ``url``."""
    (tree.worktree / "big.txt").write_text("x\n", encoding="utf-8")
    _commit(tree.worktree)
    blob = _git(tree.worktree, "rev-parse", "HEAD:big.txt").strip()
    (tree.base / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    (tree.worktree / "big.txt").write_text("y\n", encoding="utf-8")
    tree.config(
        ("core.repositoryformatversion", "1"),
        ("extensions.partialClone", "origin"),
        ("remote.origin.promisor", "true"),
        ("remote.origin.url", url),
        *settings,
    )
    return ("diff", "--no-ext-diff", "HEAD")


def _over_ext(tree: _Tree) -> Sequence[str]:
    return _lazy_fetch(tree, f"ext::{tree.program}", ("protocol.ext.allow", "always"))


def _over_ssh(tree: _Tree) -> Sequence[str]:
    return _lazy_fetch(tree, "ssh://example.invalid/r.git", ("core.sshCommand", str(tree.program)))


def _over_git_proxy(tree: _Tree) -> Sequence[str]:
    return _lazy_fetch(tree, "git://example.invalid/r.git", ("core.gitProxy", str(tree.program)))


def _over_upload_pack(tree: _Tree) -> Sequence[str]:
    return _lazy_fetch(tree, str(tree.base), ("remote.origin.uploadpack", str(tree.program)))


_PLANTS = [
    *(
        pytest.param(_filter(program, *command), "", id=f"filter-{program}-{command[0]}")
        for program, command in (
            ("clean", ("add", "-A")),
            ("clean", ("diff", "--no-ext-diff", "HEAD")),
            ("clean", ("status", "--porcelain")),
            ("smudge", ("checkout", "--", "mod.py")),
            ("process", ("add", "-A")),
            ("process", ("diff", "--no-ext-diff", "HEAD")),
            ("process", ("status", "--porcelain")),
            ("process", ("checkout", "--", "mod.py")),
        )
    ),
    pytest.param(_dotted_filter, "", id="filter-dotted-name"),
    pytest.param(_included_filter("include.path"), "", id="filter-included"),
    pytest.param(_included_filter("includeIf.onbranch:task.path"), "", id="filter-included-if"),
    pytest.param(_diff_program("diff.external", "diff", "HEAD"), None, id="diff-external"),
    pytest.param(_diff_program("diff.evil.command", "diff", "HEAD"), None, id="diff-driver-command"),
    pytest.param(_diff_program("diff.evil.textconv", "diff", "--no-ext-diff", "HEAD"), None, id="diff-driver-textconv"),
    pytest.param(_merge_driver, None, id="merge-driver"),
    pytest.param(_trailer, "", id="trailer-command"),
    pytest.param(_hook("hooks-dir"), "", id="hook-in-hooks-dir"),
    pytest.param(_hook("hooksPath"), "", id="hook-in-hooksPath"),
    pytest.param(
        _configured_hook,
        "",
        id="hook-in-config",
        marks=pytest.mark.skipif(not _CONFIG_HOOKS, reason="this git defines no hooks in config"),
    ),
    pytest.param(_fsmonitor, "", id="fsmonitor"),
    pytest.param(_editor, None, id="editor"),
    pytest.param(_sequence_editor, "", id="sequence-editor"),
    pytest.param(_alternate_refs, "", id="alternate-refs-command"),
    pytest.param(_verifies("openpgp", "gpg.program"), "", id="verify-gpg-program"),
    pytest.param(_verifies("openpgp", "gpg.openpgp.program"), "", id="verify-gpg-openpgp-program"),
    pytest.param(_verifies("ssh", "gpg.ssh.program"), "", id="verify-gpg-ssh-program"),
    pytest.param(_verifies("x509", "gpg.x509.program"), "", id="verify-gpg-x509-program"),
    pytest.param(_shows_signature, "signed", id="log-show-signature"),
    pytest.param(_signs("commit.gpgSign", "commit", "-qm", "c"), "", id="sign-commit"),
    pytest.param(_signs("tag.gpgSign", "tag", "-m", "t", "t1"), "", id="sign-tag"),
    pytest.param(_signs("tag.forceSignAnnotated", "tag", "-m", "t", "t1"), "", id="sign-annotated-tag"),
    pytest.param(_verifies_a_merge, "", id="verify-merge"),
    pytest.param(_default_signing_key, None, id="ssh-default-key-command"),
    pytest.param(_auto_gc, "", id="auto-gc-recent-objects-hook"),
    pytest.param(_autocorrect, None, id="autocorrect-alias"),
    pytest.param(_recursing_reset, "", id="submodule-recurse"),
    pytest.param(_inline_submodule_diff, "+Subproject commit", id="submodule-inline-diff"),
    pytest.param(_nested_repository("status", "--porcelain"), "", id="nested-repository-status"),
    pytest.param(_nested_repository("diff", "HEAD"), "", id="nested-repository-diff"),
    pytest.param(_over_ext, None, id="transport-ext"),
    pytest.param(_over_ssh, None, id="transport-ssh-command"),
    pytest.param(_over_git_proxy, None, id="transport-git-proxy"),
    pytest.param(_over_upload_pack, None, id="transport-upload-pack"),
]


@pytest.mark.parametrize(("plant", "expected"), _PLANTS)
def test_no_program_the_repository_configures_runs(tmp_path, plant, expected):
    """``expected`` is None where the command may fail closed, else what its output holds when it works as it should."""
    done = _proves(tmp_path, plant)

    if expected is not None:
        assert done.returncode == 0, done.stderr
        assert expected in done.stdout


class _Unauthorized(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="r"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def unauthorized() -> Iterator[str]:
    """The URL of a server that asks for credentials, which an http fetch then gets from a program."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Unauthorized)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/r.git"
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.parametrize("key", ["credential.helper", "core.askPass"])
def test_an_http_fetch_starts_no_credential_program(tmp_path, unauthorized, key):
    done = _proves(tmp_path, lambda tree: _lazy_fetch(tree, unauthorized, (key, str(tree.program))))

    assert "not allowed" in done.stderr


def test_a_driver_the_command_line_cannot_blank_runs_nothing(tmp_path):
    """``-c filter.a=b.clean=`` sets ``filter.a``; the attribute ``filter=a=b`` still names the driver ``a=b``."""
    tree = _tree(tmp_path / "t")
    tree.config(("filter.a=b.clean", str(tree.filter)))
    tree.attributes("* filter=a=b")
    tree.change()

    with pytest.raises(OSError, match="holds '='"):
        app_git(tree.worktree, "diff", "--no-ext-diff", "HEAD")

    assert not tree.canary.exists()


def test_a_log_starts_no_signature_program_at_all(tmp_path, monkeypatch):
    """``gpg.program=false`` is what a verification starts; a plain ``log`` starts not even that."""
    shadow = tmp_path / "bin"
    shadow.mkdir()
    started = tmp_path / "started"
    (shadow / "false").write_text(f"#!/bin/sh\ntouch {shlex.quote(str(started))}\nexit 1\n", encoding="utf-8")
    (shadow / "false").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shadow}{os.pathsep}{os.environ['PATH']}")
    tree = _tree(tmp_path / "t")
    _signed(tree, "openpgp")
    tree.config(("log.showSignature", "true"))

    assert app_git(tree.worktree, "log", "-1", "--format=%s").stdout == "signed\n"
    assert not started.exists()
    app_git(tree.worktree, "log", "-1", "--format=%G?")
    assert started.exists()


def _on_a_terminal(*argv: str) -> None:
    primary, secondary = pty.openpty()
    try:
        subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=secondary,
            stderr=secondary,
            env=_plain_env(),
            timeout=60,
            check=False,
        )
    finally:
        os.close(secondary)
        os.close(primary)


def test_no_pager_starts_even_on_a_terminal(tmp_path):
    """``app_git`` captures its output, so git starts no pager; ``NO_PROGRAMS`` holds where output is a terminal."""
    tree = _tree(tmp_path / "t")
    tree.config(("core.pager", str(tree.program)), ("pager.log", str(tree.program)))

    _on_a_terminal("git", *NO_PROGRAMS, "-C", str(tree.worktree), "log")
    assert not tree.canary.exists()
    _on_a_terminal("git", "-C", str(tree.worktree), "log")
    assert tree.canary.exists()


def test_the_diff_does_not_run_git_in_a_repository_the_index_tracks(tmp_path):
    """An ``add`` over a tracked nested repository runs git in it, every pin notwithstanding; ``app_git_diff`` adds none."""
    tree = _tree(tmp_path / "t")
    _nested_repository()(tree)
    (tree.worktree / ".gitmodules").write_text(
        '[submodule "vendored"]\n\tpath = vendored\n\turl = ./vendored\n\tignore = none\n', encoding="utf-8"
    )
    tree.change()

    done = app_git_diff(tree.worktree, "HEAD", ["."])

    assert not tree.canary.exists()
    assert "+new = 2" in done.stdout
    assert "+fresh = 3" in done.stdout
    app_git(tree.worktree, "add", "-A", "-N")
    assert tree.canary.exists()


def test_the_diff_shows_every_untracked_path(tmp_path):
    """More paths than one ``add`` names, and one whose name is not UTF-8, all render as created."""
    tree = _tree(tmp_path / "t")
    for n in range(300):
        (tree.worktree / f"new{n}.txt").write_text(f"{n}\n", encoding="utf-8")
    (tree.worktree / os.fsdecode(b"latin-\xe9.txt")).write_text("e\n", encoding="utf-8")

    done = app_git_diff(tree.worktree, "HEAD", ["."])

    assert done.returncode == 0, done.stderr
    assert done.stdout.count("new file mode") == 301
    assert '"b/latin-\\351.txt"' in done.stdout


def test_a_worktree_of_another_repository_than_the_caller_names_runs_nothing(tmp_path):
    """Ownership and the ``gitdir`` back-pointer hold for a worktree of any repository the app owns; ``common`` names
    the one the caller made it from, and another's admin directory is refused before git runs."""
    mine, other = _tree(tmp_path / "mine"), _tree(tmp_path / "other")
    other.config(("filter.own.clean", str(other.filter)))
    other.attributes("* filter=own")
    other.change()

    with pytest.raises(OSError, match="is not one of the repository"):
        app_git(other.worktree, "add", "-A", admin=git_safety.trusted_admin_dir(other.worktree, mine.base / ".git"))

    assert not other.canary.exists()
    assert (
        app_git(
            mine.worktree,
            "status",
            "--porcelain",
            admin=git_safety.trusted_admin_dir(mine.worktree, mine.base / ".git"),
        ).returncode
        == 0
    )
    assert app_git(other.worktree, "status", "--porcelain").returncode == 0


def _chown_tree(root: Path, uid: int, gid: int) -> None:
    for path in (root, *root.rglob("*")):
        os.lchown(path, uid, gid)


_ROOT_ONLY = pytest.mark.skipif(os.geteuid() != 0, reason="an entry of another owner needs root to chown")
_ANOTHER, _GROUP, _STRANGER = 65534, 65533, 65532
_REFUSED = "git's own state of a shared repository is refused"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


@contextlib.contextmanager
def _umask(mask: int) -> Iterator[None]:
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


def _shared_tree(root: Path, mask: int = 0o022) -> tuple[_Tree, Path]:
    """A tree made as the app makes a repository below shared storage, under the umask ``mask``: the base directory
    ``2755``, git's directory made by ``git init --shared=group`` under :func:`git_safety.git_dir_umask`, then laid
    out by :func:`git_safety.share_objects_and_refs`; the worktree added after."""
    root.mkdir()
    root.chmod(0o2775)
    if not root.stat().st_mode & stat.S_ISGID:
        pytest.skip("this filesystem does not keep the setgid bit for the test user")
    tree = _Tree(root, root / "base", root / "worktree", root / "program", root / "filter")
    for program, then in ((tree.program, "exit 0"), (tree.filter, "exec cat")):
        program.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(tree.canary))}\n{then}\n", encoding="utf-8")
        program.chmod(0o755)
    with _umask(mask):
        tree.base.mkdir(0o755)
        tree.base.chmod(0o2755)
        (tree.base / "mod.py").write_text("old = 1\n", encoding="utf-8")
        subprocess.run(
            ["git", "init", "-q", "--shared=group", str(tree.base)],
            env=_plain_env(),
            umask=git_safety.git_dir_umask(),
            check=True,
            capture_output=True,
        )
        common = tree.base / ".git"
        git_safety.share_objects_and_refs(common)
        _commit(tree.base, "base")
    # As the app adds one in a shared repository: under the shared child umask, so the group writes its admin directory.
    subprocess.run(
        ["git", "-C", str(tree.base), "worktree", "add", "-q", "-b", "task", str(tree.worktree)],
        env=_plain_env(),
        umask=0o002,
        check=True,
        capture_output=True,
    )
    return tree, common


def _shared_entries(common: Path) -> list[Path]:
    return [common / name / entry for name in ("objects", "refs", "logs") for entry in ("", *os.listdir(common / name))]


@pytest.mark.parametrize("mask", [0o002, 0o022], ids=["umask-002", "umask-022"])
def test_a_shared_repository_is_laid_out_so_only_objects_and_refs_are_the_group_s(tmp_path, mask):
    """Whatever the umask: the git directory ``3775``, its config ``0644`` (the next account reads it), ``info`` and
    the hooks writable by their maker alone; objects, refs, reflogs, ``worktrees`` and an admin directory the group's,
    and what the app's git adds there later the group's too."""
    tree, common = _shared_tree(tmp_path / "t", mask)
    admin = common / "worktrees" / tree.worktree.name

    assert _mode(common) == 0o3775 and _mode(tree.base) == 0o2755
    assert _mode(common / "config") == 0o644 and _mode(common / "HEAD") == 0o644
    for name in ("info", "hooks"):
        for path in (common / name, *(common / name).rglob("*")):
            assert not _mode(path) & 0o022, (path, oct(_mode(path)))
    for path in (*_shared_entries(common), common / "worktrees", admin):
        if path.is_dir():
            assert _mode(path) == 0o2775, (path, oct(_mode(path)))
    assert {_mode(path) for path in (common / "logs").rglob("*") if path.is_file()} == {0o664}
    assert git_safety.shared_group(common) == common.stat().st_gid
    (tree.worktree / "new.py").write_text("made = 1\n", encoding="utf-8")
    with _umask(mask):
        assert app_git(tree.worktree, "add", "-A").returncode == 0
        done = app_git(tree.worktree, "-c", "user.name=t", "-c", "user.email=t@t.local", "commit", "-q", "-m", "x")
    assert done.returncode == 0, done.stderr
    assert _mode(common / "refs" / "heads" / "task") & 0o060 == 0o060
    assert {_mode(path) for path in (common / "objects").iterdir() if len(path.name) == 2} == {0o2775}
    assert _mode(admin / "index") & 0o060 == 0o060


def _group_writable(path: Path) -> None:
    path.chmod(_mode(path) | 0o020)


def _linked_config(common: Path) -> None:
    kept = common.parent.parent / "config.kept"
    shutil.copyfile(common / "config", kept)
    (common / "config").unlink()
    (common / "config").symlink_to(kept)


def _planted(name: str) -> Callable[[Path], None]:
    def plant(common: Path) -> None:
        (common / name).write_text("planted\n", encoding="utf-8")

    return plant


_SPOILS: dict[str, Callable[[Path], None]] = {
    "config-0664": lambda common: _group_writable(common / "config"),
    "config-a-link": _linked_config,
    "info-group-writable": lambda common: _group_writable(common / "info"),
    "info-entry-group-writable": lambda common: _group_writable(common / "info" / "exclude"),
    "hooks-group-writable": lambda common: _group_writable(common / "hooks"),
    "git-dir-without-sticky-bit": lambda common: common.chmod(0o2775),
    "base-group-writable": lambda common: common.parent.chmod(0o2775),
    "commondir-in-the-git-dir": _planted("commondir"),
    "config-worktree-in-the-git-dir": _planted("config.worktree"),
}


@pytest.mark.parametrize("spoil", list(_SPOILS))
def test_a_shared_repository_holding_state_another_account_could_write_starts_no_git(tmp_path, monkeypatch, spoil):
    """Refused before any git starts, whichever way the admin directory was found."""
    tree, common = _shared_tree(tmp_path / "t")
    recorded = git_safety.recorded_admin_dir(tree.base, tree.worktree)
    assert app_git(tree.worktree, "status", "--porcelain", admin=recorded).returncode == 0
    _SPOILS[spoil](common)

    def no_git(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError(f"git started: {args}")

    monkeypatch.setattr(git_safety.subprocess, "run", no_git)
    with pytest.raises(OSError, match=_REFUSED):
        app_git(tree.worktree, "status", "--porcelain")
    with pytest.raises(OSError, match=_REFUSED):
        app_git(tree.worktree, "status", "--porcelain", admin=recorded)


def test_a_repository_of_one_account_is_not_checked(tmp_path):
    """Off shared storage nothing changes: a config the group can write still runs, as before."""
    tree = _tree(tmp_path / "t")
    common = tree.base / ".git"
    (common / "config").chmod(0o664)
    common.chmod(0o775)

    assert git_safety.shared_group(common) is None
    assert git_safety.check_shared_repository(common) is None
    assert app_git(tree.worktree, "status", "--porcelain").returncode == 0


def test_a_rewritten_commondir_names_no_other_repository(tmp_path, monkeypatch):
    """An admin directory is the group's to write in a shared repository: a ``commondir`` pointed at a repository of
    another's, whose config names a filter, is refused before git runs, and even unrefused reaches no config of git's,
    which is pinned to the checked repository (``GIT_COMMON_DIR``)."""
    tree, common = _shared_tree(tmp_path / "t")
    admin = common / "worktrees" / tree.worktree.name
    planted = tmp_path / "planted.git"
    shutil.copytree(common, planted, symlinks=True)
    _git(tree.base, "--git-dir", str(planted), "config", "filter.own.clean", str(tree.filter))
    (admin / "commondir").write_text(f"{planted}\n", encoding="utf-8")
    tree.attributes("* filter=own")
    tree.change()

    with pytest.raises(OSError, match="commondir names another repository"):
        app_git(tree.worktree, "add", "-A")
    # Unrefused, and the driver taken as added after the config was read to blank it: the pin alone keeps it out.
    monkeypatch.setattr(git_safety, "_names_its_repository", lambda admin, common: None)
    monkeypatch.setattr(git_safety, "blank_driver_programs", lambda *args: [])
    done = app_git(tree.worktree, "add", "-A")

    assert done.returncode == 0, done.stderr
    assert not tree.canary.exists()
    _plain_add(tree)
    assert tree.canary.exists(), "plain git follows the rewritten commondir to the planted config, so the plant is live"


def _repository_of_another_account(tmp_path: Path, *, shared: bool) -> tuple[_Tree, Path]:
    """A tree another account of ``_GROUP`` made, in a directory only this account writes: laid out for the group when
    ``shared``, else made under a ``022`` umask with nothing the group writes."""
    tmp_path.mkdir()
    if shared:
        tree, common = _shared_tree(tmp_path / "t")
    else:
        with _umask(0o022):
            tree = _tree(tmp_path / "t")
        common = tree.base / ".git"
    tree.root.chmod(0o755)
    _chown_tree(tree.base, _ANOTHER, _GROUP)
    _chown_tree(common, _ANOTHER, _GROUP)
    return tree, common


@_ROOT_ONLY
def test_an_admin_directory_another_account_made_is_trusted_only_in_a_shared_repository_of_its_group(tmp_path):
    """Another account's admin directory is the app's only when ``common`` is a shared repository (its objects and
    refs set up to be shared) and the entry is in their group: the run's legs, accounts of one group, add worktrees in
    turn. Its config is the maker's, which the app's git reads but does not write."""
    tree, common = _repository_of_another_account(tmp_path / "unshared", shared=False)
    with pytest.raises(OSError, match="not a directory owned by the app"):
        git_safety.trusted_admin_dir(tree.worktree, common)

    tree, common = _repository_of_another_account(tmp_path / "shared", shared=True)
    admin = common / "worktrees" / "worktree"
    assert (
        app_git(
            tree.worktree, "status", "--porcelain", admin=git_safety.trusted_admin_dir(tree.worktree, common)
        ).returncode
        == 0
    )
    with pytest.raises(OSError, match="not a directory owned by the app"):
        app_git(tree.worktree, "status", "--porcelain")  # no common: the uid alone
    os.lchown(admin, _ANOTHER, _STRANGER)
    with pytest.raises(OSError, match="not a directory owned by the app"):
        app_git(tree.worktree, "status", "--porcelain", admin=git_safety.trusted_admin_dir(tree.worktree, common))
    os.lchown(admin, _ANOTHER, _GROUP)
    os.lchown(admin / "gitdir", _ANOTHER, _STRANGER)
    with pytest.raises(OSError, match="not an admin dir of a linked worktree"):
        app_git(tree.worktree, "status", "--porcelain", admin=git_safety.trusted_admin_dir(tree.worktree, common))


@_ROOT_ONLY
@pytest.mark.parametrize("objects_mode", [0o2755, 0o2777], ids=oct)
def test_a_repository_whose_objects_are_not_set_up_to_be_shared_trusts_no_entry_of_its_group(tmp_path, objects_mode):
    """Objects the group cannot write (``2755``), or that others can write too (``2777``), are not a shared
    repository's, by the rule that sets git's umask: an entry of the group is refused."""
    tree, common = _repository_of_another_account(tmp_path / "r", shared=True)
    (common / "objects").chmod(objects_mode)

    with pytest.raises(OSError, match="not a directory owned by the app"):
        git_safety.trusted_admin_dir(tree.worktree, common)


@_ROOT_ONLY
@pytest.mark.parametrize("entry", ["config", "info", "hooks", "base", "git-dir"])
def test_state_of_another_account_than_the_maker_is_refused(tmp_path, entry):
    """The account that made the repository owns its top directory, which no account of the group can change: a
    config, ``info``, hooks or git directory of another account is refused, and so is a base directory of another
    account than the git directory's owner."""
    tree, common = _shared_tree(tmp_path / "t")
    target = {"base": tree.base, "git-dir": common}.get(entry, common / entry)
    os.lchown(target, _ANOTHER, -1)

    with pytest.raises(OSError, match=_REFUSED):
        app_git(tree.worktree, "status", "--porcelain", admin=common / "worktrees" / "worktree")


@_ROOT_ONLY
def test_a_repository_another_account_put_in_place_of_the_maker_s_is_refused(tmp_path):
    """Below a session directory the group can write and that the maker owns, another account renames the base away
    and puts a repository of its own in its place: every entry of that one carries the group, and the gitfile still
    names it, but the session directory is not that account's, so the app's git refuses it."""
    tree, common = _shared_tree(tmp_path / "t")
    admin = git_safety.trusted_admin_dir(tree.worktree, common)
    assert app_git(tree.worktree, "status", "--porcelain", admin=admin).returncode == 0
    tree.base.rename(tree.root / "renamed")
    shutil.copytree(tree.root / "renamed", tree.base, symlinks=True)
    _chown_tree(tree.base, _ANOTHER, common.stat().st_gid)

    with pytest.raises(OSError, match="put another repository in place"):
        app_git(tree.worktree, "status", "--porcelain", admin=git_safety.trusted_admin_dir(tree.worktree, common))
    tree.root.chmod(0o3775)  # a sticky session directory: no account renames another's entry
    assert app_git(tree.worktree, "status", "--porcelain", admin=admin).returncode == 0


def test_a_link_at_the_git_directory_s_own_name_is_not_a_shared_repository(tmp_path):
    """``common`` named through a link is refused, not judged by what the link points at."""
    tree, common = _shared_tree(tmp_path / "t")
    link = tmp_path / "link"
    link.symlink_to(common)

    with pytest.raises(OSError) as raised:
        git_safety.trusted_admin_dir(tree.worktree, link)
    assert raised.value.errno == errno.ENOTDIR  # O_NOFOLLOW with O_DIRECTORY on a link


def _blob_directory(data: bytes) -> str:
    """The object directory git stores a blob of ``data`` in."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()[:2]


@pytest.mark.parametrize("layout", ["shared", "git-dir-2775", "plain"])
def test_git_writes_what_the_group_can_write_only_in_a_shared_repository(tmp_path, layout):
    """In a shared repository (:func:`git_safety.shared_group`) the next leg, another account of the group, writes the
    index and the objects after this one: git runs under the shared child umask there, and under this process's
    anywhere else, a git directory that is merely group-writable included."""
    umask = current_umask()
    assert umask & 0o020, "the test needs a umask that keeps group write off what a plain create makes"
    if layout == "shared":
        tree, common = _shared_tree(tmp_path / "t", umask)
    else:
        tree = _tree(tmp_path / "t")
        common = tree.base / ".git"
        if layout == "git-dir-2775":
            common.chmod(0o2775)
    before = {path.name for path in (common / "objects").iterdir() if path.is_dir()}
    # A blob whose object directory git has not made yet (the base commit's objects vary with its timestamp).
    content = next(
        data for data in (f"new = {n}\n".encode() for n in range(1000)) if _blob_directory(data) not in before
    )
    (tree.worktree / "new.py").write_bytes(content)

    done = app_git(tree.worktree, "add", "-A", admin=git_safety.trusted_admin_dir(tree.worktree, common))

    assert done.returncode == 0, done.stderr
    made = [path for path in (common / "objects").iterdir() if path.is_dir() and path.name not in before]
    assert common / "objects" / _blob_directory(content) in made
    index = common / "worktrees" / tree.worktree.name / "index"
    expected = (0o775, 0o664) if layout == "shared" else (0o777 & ~umask, 0o666 & ~umask)
    assert made and {stat.S_IMODE(path.stat().st_mode) & 0o777 for path in made} == {expected[0]}
    assert stat.S_IMODE(index.stat().st_mode) == expected[1]
    (tree.worktree / "mod.py").unlink()
    assert app_git(tree.worktree, "checkout", "--", "mod.py").returncode == 0
    assert _mode(tree.worktree / "mod.py") == expected[1], "a worktree file git writes follows git's umask alone"


def test_the_caller_s_environment_never_outranks_what_keeps_the_config_out(tmp_path):
    """``env`` adds an index file or an identity; a variable that would read another config or allow a transport is
    the app's, whatever the caller passes."""
    tree = _tree(tmp_path / "t")
    planted = tmp_path / "planted.gitconfig"
    planted.write_text(f"[core]\n\tpager = {tree.program}\n[alias]\n\tst = status\n", encoding="utf-8")
    index = tmp_path / "index"

    read = app_git(tree.worktree, "read-tree", "HEAD", env={"GIT_INDEX_FILE": str(index)})
    aliased = app_git(tree.worktree, "st", env={"GIT_CONFIG_GLOBAL": str(planted), "GIT_ALLOW_PROTOCOL": "ext"})

    assert read.returncode == 0 and index.is_file()
    assert aliased.returncode != 0, "the planted global config was read"
    plain = subprocess.run(
        ["git", "-C", str(tree.worktree), "st"],
        env={**_plain_env(), "GIT_CONFIG_GLOBAL": str(planted)},
        capture_output=True,
        check=False,
    )
    assert plain.returncode == 0, "the planted global config defines the alias, so the plant is live"


def _plain_add(tree: _Tree) -> None:
    argv = ["git", "-C", str(tree.worktree), "-c", "safe.directory=*", "add", "-A"]
    subprocess.run(argv, env=_plain_env(), capture_output=True, check=False)


@pytest.mark.parametrize("common", [False, True], ids=["any-repository", "the-caller-s"])
def test_a_symlink_on_the_way_to_the_admin_dir_flipped_after_the_check_reaches_no_git(tmp_path, monkeypatch, common):
    """The gitfile names the admin dir through a symlink the worktree's writer owns, pointing at the real one when it
    is checked and at the writer's own repository when git runs: git is given the path the check resolved."""
    tree = _tree(tmp_path / "t")
    git = tree.base / ".git"
    # The writer's repository: the base's copied, its config naming a filter, its admin dir naming this gitfile back.
    own = tree.worktree / ".writer"
    shutil.copytree(git, own / ".git", symlinks=True)
    _git(own, "config", "filter.own.clean", str(tree.filter))
    link = tree.worktree / "lnk"
    link.symlink_to(tree.base)
    (tree.worktree / ".git").write_text(f"gitdir: {link}/.git/worktrees/worktree\n", encoding="utf-8")
    tree.attributes("* filter=own")
    tree.change()
    checked = git_safety.trusted_admin_dir

    def check_then_flip(worktree: Path, caller_s: Path | None = None) -> Path:
        admin = checked(worktree, caller_s)
        flipped = tmp_path / "flipped"
        flipped.symlink_to(own)
        os.replace(flipped, link)
        return admin

    monkeypatch.setattr(git_safety, "trusted_admin_dir", check_then_flip)

    done = app_git(
        tree.worktree, "add", "-A", admin=git_safety.trusted_admin_dir(tree.worktree, git) if common else None
    )

    assert done.returncode == 0, done.stderr
    assert not tree.canary.exists()
    with pytest.raises(OSError, match="lies in the worktree"):
        checked(tree.worktree, git if common else None)  # flipped before the check, the writer's admin dir is refused
    _plain_add(tree)
    assert tree.canary.exists(), "plain git follows the flipped link to the writer's config, so the plant is live"


@pytest.mark.skipif(os.geteuid() != 0, reason="a directory of another owner needs root to chown")
@pytest.mark.parametrize("level", [0, 1, 2], ids=["admin-dir", "worktrees", "repository"])
def test_an_admin_dir_on_a_path_another_account_owns_runs_nothing(tmp_path, level):
    tree = _tree(tmp_path / "t")
    admin = tree.base / ".git" / "worktrees" / "worktree"
    tree.config(("filter.own.clean", str(tree.filter)))
    tree.attributes("* filter=own")
    tree.change()
    os.lchown((admin, admin.parent, admin.parent.parent)[level], 65534, 65534)

    with pytest.raises(OSError, match="not a directory owned by the app"):
        app_git(tree.worktree, "add", "-A")

    assert not tree.canary.exists()
    _plain_add(tree)
    assert tree.canary.exists(), "plain git runs the filter, so the plant is live"


def test_a_gitfile_that_is_a_symlink_is_not_followed_even_to_a_true_pointer(tmp_path):
    """The app reads no file the writer points ``.git`` at (a pipe would hang it, a device fill its memory)."""
    tree = _tree(tmp_path / "t")
    gitfile = tree.worktree / ".git"
    pointer = tmp_path / "pointer"
    pointer.write_text(gitfile.read_text(encoding="utf-8"), encoding="utf-8")
    gitfile.unlink()
    gitfile.symlink_to(pointer)

    with pytest.raises(OSError) as refused:
        app_git(tree.worktree, "status", "--porcelain")

    assert refused.value.errno == errno.ELOOP
    assert subprocess.run(["git", "-C", str(tree.worktree), "status"], env=_plain_env(), check=False).returncode == 0


def test_an_admin_dir_the_gitfile_names_inside_the_worktree_runs_nothing(tmp_path):
    """Whoever writes the worktree may write files the app's uid owns, so an admin dir there is never the app's."""
    tree = _tree(tmp_path / "t")
    own = tree.worktree / ".writer"
    shutil.copytree(tree.base / ".git", own, symlinks=True)
    with (own / "config").open("a", encoding="utf-8") as config:
        config.write(f'[filter "own"]\n\tclean = {tree.filter}\n')
    (tree.worktree / ".git").write_text(f"gitdir: {own}/worktrees/worktree\n", encoding="utf-8")
    tree.attributes("* filter=own")
    tree.change()

    with pytest.raises(OSError, match="lies in the worktree"):
        app_git(tree.worktree, "add", "-A")

    assert not tree.canary.exists()
    _plain_add(tree)
    assert tree.canary.exists(), "plain git follows the gitfile to the writer's config, so the plant is live"
