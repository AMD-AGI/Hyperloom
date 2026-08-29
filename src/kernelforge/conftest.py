# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures and guardrails for every KernelForge test tree.

This sits at the package root, not inside ``tests/``, because a conftest only
reaches its own directory and below. The guardrails here are the kind that are
worthless when partially applied -- the site-packages write guard, the isolated
``KERNELFORGE_PROJECT_ROOT``, the child-process ``PYTHONPATH`` -- and forge has
a second test tree under ``gemm_tune/tests/`` that a conftest in ``tests/``
silently skipped, along with any tree added next to it later. Package root is
the only location that covers all of them without a copy per directory that
would drift.

It ships in the wheel as a consequence. That is a few KB of a module pytest
imports during collection and nothing imports at runtime; the packaging lint
covers it as an ordinary module.
"""

from __future__ import annotations

import builtins
import io
import os
import re
from pathlib import Path

import pytest

import kernelforge

#: Root of the installed package. Everything under it is read-only at runtime:
#: it may live in a root-owned site-packages and is replaced wholesale on
#: upgrade, so anything written there is silently lost.
PACKAGE_ROOT = Path(kernelforge.__file__).resolve().parent
_PACKAGE_PREFIX = str(PACKAGE_ROOT) + os.sep

#: The directory ``kernelforge`` is importable from -- ``src/`` in a checkout,
#: ``site-packages`` under a wheel install.
SRC_ROOT = PACKAGE_ROOT.parent


@pytest.fixture(scope="session", autouse=True)
def _src_root_on_child_pythonpath() -> None:
    """Extend pytest's in-process ``pythonpath`` to subprocesses.

    Roughly 250 call sites in this tree spawn ``sys.executable`` and expect to
    import ``kernelforge`` / ``kernelforge.llm`` there. Upstream KernelForge got away
    with it because its CI always ran against ``pip install -e``; run the suite
    from a bare checkout instead -- which the ``pythonpath = ["src", "."]`` ini
    setting makes work for the *parent* -- and every one of those children dies
    with ModuleNotFoundError. Setting it once here is the same statement pytest
    already makes in-process, extended to what the tests fork.

    Session-scoped and deliberately not undone: children are spawned from every
    scope, and under a wheel install this prepends site-packages, which is a
    no-op.
    """
    existing = os.environ.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if str(SRC_ROOT) not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(SRC_ROOT), *parts])


def _find_repo_root() -> Path | None:
    """Walk up for the pyproject.toml; returns None when installed from a wheel."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return None


#: Repository root, or ``None`` under a wheel install. Tests that genuinely need
#: repository metadata must skip when this is ``None`` rather than guess a depth.
REPO_ROOT = _find_repo_root()

requires_repo_root = pytest.mark.skipif(
    REPO_ROOT is None,
    reason="needs the source checkout (pyproject.toml + src/)",
)


@pytest.fixture
def repo_root() -> Path:
    """Repository root; skips the test under a wheel install."""
    if REPO_ROOT is None:
        pytest.skip("needs the source checkout (pyproject.toml + src/)")
    return REPO_ROOT


def _inside_package(target: object) -> bool:
    """Whether an ``open``/``os`` path argument points into the package."""
    if isinstance(target, int):  # already-open file descriptor
        return False
    try:
        path = os.fspath(target)
    except TypeError:
        return False
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    absolute = os.path.abspath(path)
    if not (absolute == str(PACKAGE_ROOT) or absolute.startswith(_PACKAGE_PREFIX)):
        return False
    # Bytecode caching is the interpreter's business, not runtime state.
    return "__pycache__" not in absolute.split(os.sep) and not absolute.endswith((".pyc", ".pyo"))


def _refuse(target: object, how: str) -> None:
    raise AssertionError(
        f"test attempted to {how} inside the installed kernelforge package: {target!r}. "
        "Runtime state belongs under kernelforge.resources.default_project_root(); the "
        "packaged data tree is read-only and is replaced on upgrade."
    )


@pytest.fixture(autouse=True)
def _no_writes_under_site_packages(monkeypatch):
    """Fail any test that writes, creates or deletes inside the package.

    The data trees moved *into* ``kernelforge/data`` when KernelForge was
    vendored into Hyperloom, which put every historical "write next to the
    knowledge base" code path on a collision course with site-packages. A
    writable site-packages makes that silently pollute the installation and
    vanish on upgrade; a read-only one makes it explode halfway through a run.
    This is the long-lived guard against reintroducing either.

    The hooks sit on the lowest-level primitives so ``pathlib``, ``shutil`` and
    ``open`` are all covered without patching each of them.
    """
    real_open = builtins.open
    real_os_open = os.open
    real_mkdir = os.mkdir
    real_makedirs = os.makedirs
    real_remove = os.remove
    real_unlink = os.unlink
    real_rmdir = os.rmdir
    real_rename = os.rename
    real_replace = os.replace

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")) and _inside_package(file):
            _refuse(file, "open for writing")
        return real_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writing and _inside_package(path):
            _refuse(path, "open for writing")
        return real_os_open(path, flags, *args, **kwargs)

    def _guard_one(real, how, index=0):
        def wrapper(*args, **kwargs):
            if len(args) > index and _inside_package(args[index]):
                _refuse(args[index], how)
            return real(*args, **kwargs)

        return wrapper

    def guarded_rename(src, dst, *args, **kwargs):
        if _inside_package(dst) or _inside_package(src):
            _refuse(dst, "rename into or out of")
        return real_rename(src, dst, *args, **kwargs)

    def guarded_replace(src, dst, *args, **kwargs):
        if _inside_package(dst) or _inside_package(src):
            _refuse(dst, "replace into or out of")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(os, "mkdir", _guard_one(real_mkdir, "mkdir"))
    monkeypatch.setattr(os, "makedirs", _guard_one(real_makedirs, "makedirs"))
    monkeypatch.setattr(os, "remove", _guard_one(real_remove, "remove"))
    monkeypatch.setattr(os, "unlink", _guard_one(real_unlink, "unlink"))
    monkeypatch.setattr(os, "rmdir", _guard_one(real_rmdir, "rmdir"))
    monkeypatch.setattr(os, "rename", guarded_rename)
    monkeypatch.setattr(os, "replace", guarded_replace)


@pytest.fixture(scope="session")
def _state_root_base(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("kernelforge-state")


@pytest.fixture(autouse=True)
def _isolated_state_root(request, _state_root_base, monkeypatch):
    """Point the writable-state root at a per-test temporary directory.

    Without this, ``default_project_root()`` falls through to
    ``~/.cache/hyperloom/kernelforge``: the suite would accumulate state in the
    developer's home directory and read back another test's leftovers. The
    directory is not created -- callers mkdir on demand, and a test that never
    touches the state root leaves nothing behind.
    """
    if os.environ.get("KERNELFORGE_PROJECT_ROOT", "").strip():
        return
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid)[-120:]
    monkeypatch.setenv("KERNELFORGE_PROJECT_ROOT", str(_state_root_base / slug))
