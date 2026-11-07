# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Other accounts on local disk, for the tests of what one account leaves another on shared storage.

Two accounts share a group and a third is outside it; code runs as one of them through ``setpriv``, so only root can
run these tests, or in the single-id user namespace a container of one of them is (its root mapped to the account, the
group it holds unmapped). The ids are ones no account or group of this host uses.

The checkout may sit in a home directory only root can enter, so what such an account imports is copied below a
directory every account can reach (``lib``): the ``hyperloom_common`` modules the shared-storage code is made of, which
need nothing but the standard library, and whatever a test adds.
"""

from __future__ import annotations

import grp
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

import hyperloom_common

_COMMON_MODULES = ("__init__.py", "io.py", "shared_storage.py", "timeutil.py")
_FIRST_FREE_ID = 60000
_TIMEOUT_SEC = 60


@dataclass(frozen=True)
class Accounts:
    """``first`` and ``second`` hold ``group``; ``outsider`` does not. ``root`` is a directory every account can reach."""

    root: Path
    python: str
    group: int
    first: int
    second: int
    outsider: int

    @property
    def lib(self) -> Path:
        return self.root / "lib"

    def shared_dir(self, name: str) -> Path:
        """A new directory ``root:group`` ``2775`` below :attr:`root`: a key directory as ops sets one up."""
        path = self.root / name
        path.mkdir()
        os.chown(path, 0, self.group)
        path.chmod(0o2775)
        return path

    def add_module(self, source: Path, name: str) -> None:
        """Make ``source`` importable as ``name`` by the code the accounts run."""
        shutil.copyfile(source, self.lib / f"{name}.py")

    def run(self, uid: int, code: str, *, umask: int = 0o022) -> str:
        """Run ``code`` as ``uid`` (with :attr:`group` unless it is :attr:`outsider`) under ``umask``; its stdout."""
        return _finish(
            subprocess.run(self._command(uid, code, umask), capture_output=True, text=True, timeout=_TIMEOUT_SEC)
        )

    def run_in_single_id_namespace(self, uid: int, code: str, *, umask: int = 0o022) -> str:
        """Run ``code`` as root of a new user namespace that maps only ``0`` to ``uid`` (``setgroups`` denied, so the
        account keeps :attr:`group` without seeing it); its stdout. The maps are written from here, as a container
        runtime writes them."""
        wait_then_run = ["sh", "-c", 'read _ && exec "$@"', "sh", *self._interpreter(code, umask)]
        prefix = self._setpriv(uid)
        child = subprocess.Popen(
            [*prefix, "unshare", "--user", "--", *wait_then_run],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _map_single_id(child.pid, uid)
        except OSError as exc:
            child.kill()
            child.communicate()
            pytest.skip(f"cannot map the ids of a new user namespace: {exc}")
        out, err = child.communicate("go\n", timeout=_TIMEOUT_SEC)
        return _finish(subprocess.CompletedProcess(child.args, child.returncode, out, err))

    def run_as_root_in_single_id_namespace(self, code: str, *, umask: int = 0o022) -> str:
        """Run ``code`` as root of the new user namespace ``unshare --user --map-root-user`` makes for root holding
        :attr:`group`: only ``0`` is mapped, to root (``setgroups`` denied), so the group is held but unmapped, as in a
        container whose root is the host account; its stdout."""
        prefix = ["setpriv", f"--groups={self.group}", "--", "unshare", "--user", "--map-root-user", "--"]
        probe = subprocess.run([*prefix, "true"], capture_output=True, text=True, timeout=_TIMEOUT_SEC, check=False)
        if probe.returncode != 0:
            pytest.skip(f"unshare --user --map-root-user fails here: {probe.stderr.strip()}")
        command = [*prefix, *self._interpreter(code, umask)]
        return _finish(subprocess.run(command, capture_output=True, text=True, timeout=_TIMEOUT_SEC))

    def _command(self, uid: int, code: str, umask: int) -> list[str]:
        return [*self._setpriv(uid), *self._interpreter(code, umask)]

    def _setpriv(self, uid: int) -> list[str]:
        groups = uid if uid == self.outsider else self.group
        return ["setpriv", f"--reuid={uid}", f"--regid={uid}", f"--groups={groups}", "--"]

    def _interpreter(self, code: str, umask: int) -> list[str]:
        preamble = f"import os, sys\nos.umask({umask:#o})\nsys.path.insert(0, {str(self.lib)!r})\n"
        return [self.python, "-I", "-c", preamble + code]


def _finish(done: subprocess.CompletedProcess[str]) -> str:
    if done.returncode != 0:
        raise AssertionError(f"exit {done.returncode}: {done.stderr}")
    return done.stdout


def _map_single_id(pid: int, uid: int) -> None:
    ours = os.stat("/proc/self/ns/user").st_ino
    deadline = time.monotonic() + _TIMEOUT_SEC
    while os.stat(f"/proc/{pid}/ns/user").st_ino == ours:
        if time.monotonic() > deadline:
            raise TimeoutError(f"process {pid} never entered a new user namespace")
        time.sleep(0.01)
    for name, text in (("setgroups", "deny"), ("uid_map", f"0 {uid} 1"), ("gid_map", f"0 {uid} 1")):
        Path(f"/proc/{pid}/{name}").write_text(text)


def _free_ids(count: int) -> list[int]:
    used = {entry.pw_uid for entry in pwd.getpwall()} | {entry.gr_gid for entry in grp.getgrall()}
    used |= {entry.pw_gid for entry in pwd.getpwall()}
    return [candidate for candidate in range(_FIRST_FREE_ID, _FIRST_FREE_ID + 5000) if candidate not in used][:count]


def _runnable_python(uid: int) -> str | None:
    """An interpreter ``uid`` can run: this one, unless it lies where only root can reach, else the system's."""
    for candidate in (sys.executable, shutil.which("python3", path="/usr/bin:/bin")):
        if candidate is None:
            continue
        probe = ["setpriv", f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "--", candidate, "-I", "-c", "pass"]
        if subprocess.run(probe, capture_output=True, timeout=_TIMEOUT_SEC, check=False).returncode == 0:
            return candidate
    return None


#: An account no one on the host is: the owner of a link "another account" put there.
_ANOTHER_ACCOUNT = 60999


def theirs(link: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make ``link`` another account's: chowned to one where this is root, else this account seen as another."""
    if os.geteuid() == 0:
        os.lchown(link, _ANOTHER_ACCOUNT, _ANOTHER_ACCOUNT)
    else:
        euid = os.geteuid()
        monkeypatch.setattr(os, "geteuid", lambda: euid + 1)
    return link


@contextmanager
def other_accounts() -> Iterator[Accounts]:
    """:class:`Accounts` for one test, skipping it where they cannot be had; their directory is removed after it."""
    if os.geteuid() != 0:
        pytest.skip("another account's writes need root, to switch to that account with setpriv")
    if shutil.which("setpriv") is None:
        pytest.skip("setpriv is not installed")
    group, first, second, outsider = _free_ids(4)
    python = _runnable_python(first)
    if python is None:
        pytest.skip("no interpreter that another account can run")
    root = Path(tempfile.mkdtemp(prefix="hl-shared-accounts-"))
    try:
        root.chmod(0o755)
        lib = root / "lib" / "hyperloom_common"
        lib.mkdir(parents=True)
        common = Path(hyperloom_common.__file__).parent
        for name in _COMMON_MODULES:
            shutil.copyfile(common / name, lib / name)
        yield Accounts(root, python, group, first, second, outsider)
    finally:
        shutil.rmtree(root)
