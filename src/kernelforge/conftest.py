# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures and guardrails for every KernelForge test tree."""

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
    """Extend pytest's in-process ``pythonpath`` to subprocesses."""
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
    """Fail any test that writes, creates or deletes inside the package."""
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
    """Point the writable-state root at a per-test temporary directory."""
    if os.environ.get("KERNELFORGE_PROJECT_ROOT", "").strip():
        return
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid)[-120:]
    monkeypatch.setenv("KERNELFORGE_PROJECT_ROOT", str(_state_root_base / slug))


def kb_store_run_config(tmp_path: Path, token: str) -> "object":
    """A KB Store run configuration whose credential is a recognizable string.

    Three modules build the same remote-knowledge ``Config`` to assert that the
    token never reaches a persisted error, a log line, or an agent prompt. The
    assertions only mean something while all three agree on the shape, so the
    builder lives here rather than as three copies that can drift apart. The
    imports are deferred: conftest is imported during collection, before the
    guardrails above are installed.
    """
    from kernelforge.config import Config
    from kernelforge.knowledge.experience_store import (
        REMOTE_BACKEND_KB_STORE,
        KnowledgeConfig,
    )

    knowledge = KnowledgeConfig.from_env(
        {},
        mode="remote",
        local_root=tmp_path / "remote-knowledge",
        kb_store_url="http://in-memory",
        kb_store_token=token,
        remote_backend=REMOTE_BACKEND_KB_STORE,
    )
    return Config.from_env(
        workspace=str(tmp_path),
        gpu_target="gfx950",
        gpu_type="mi355x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )


@pytest.fixture
def isolated_provider_registry(monkeypatch):
    """Give the requesting test its own copy of the agent-provider registry.

    ``register_agent_provider`` writes into module-level state that outlives the
    test that called it, and the registry offers no way to unregister, so a fake
    registered by one test stays visible to every later test in the same worker
    process -- which is how these tests came to depend on the order xdist
    happened to shard them in. Discovery runs first so the snapshot already
    holds the built-ins and any installed plugin; the module globals are then
    rebound to copies that monkeypatch drops during teardown.

    Opt in per module with an autouse wrapper rather than making this autouse
    here: discovery is wasted work for the thousands of tests that never touch
    the registry.
    """
    from kernelforge.agent_backends import registry

    registry.discover_agent_providers()
    monkeypatch.setattr(registry, "_providers", dict(registry._providers))
    monkeypatch.setattr(registry, "_plugin_errors", dict(registry._plugin_errors))
