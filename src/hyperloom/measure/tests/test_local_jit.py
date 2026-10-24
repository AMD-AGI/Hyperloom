# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local backend's part of the shared compiled-code cache: seed before the server starts, stage after a success (4.21).

``run_grid`` is the stand-in of ``test_local_backend``; here it is also a compiler that writes a module into the
directory ``AITER_JIT_DIR`` names while the test runs, which is what the backend must have set for it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.measure.jit.builds import BuildLog
from hyperloom.measure.jit_kinds import AITER_JIT
from hyperloom.measure.jit.session import JitCacheSession, KindPlan
from hyperloom.measure.jit.store import DirJitStore
from hyperloom.measure.phases import PhaseRecorder
from hyperloom.measure.tests.test_local_backend import (  # noqa: F401 - the module's fixtures
    _call,
    _Grid,
    _profile_point,
    _run,
    _with_trace,
    magpie_backend,
    server,
)

KEY = "a" * 64
NOTHING = BuildLog({}, frozenset())


class _Compiler(_Grid):
    """What ``run_grid`` does for a server that compiles one module into the directory the environment names."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.directories: list[str | None] = []
        self.present: list[list[str]] = []

    async def __call__(self, **kwargs):
        directory = os.environ.get("AITER_JIT_DIR")
        self.directories.append(directory)
        self.present.append(sorted(p.name for p in Path(directory).iterdir()) if directory else [])
        if directory:
            (Path(directory) / "module_built.so").write_bytes(b"compiled")
        return await super().__call__(**kwargs)


@pytest.fixture
def compiler(monkeypatch, tmp_path) -> _Compiler:
    from hyperloom.measure import local

    fake = _Compiler(tmp_path)
    monkeypatch.setattr(local, "run_grid", fake)
    return fake


def _session(tmp_path: Path) -> JitCacheSession:
    return JitCacheSession(
        DirJitStore(tmp_path / "store"),
        [KindPlan(AITER_JIT, KEY)],
        private_root=tmp_path / "private",
        staging=tmp_path / "staged",
    )


def _staged(tmp_path: Path) -> bool:
    return (tmp_path / "staged" / "aiter_jit" / "manifest.json").is_file()


def test_the_server_is_started_with_the_variable_naming_a_directory_the_test_owns_and_it_is_restored_after(
    compiler, server, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)
    session = _session(tmp_path)

    outcome = _run(_call(tmp_path, jit=session))

    assert outcome.returncode == 0
    assert compiler.directories == [str(tmp_path / "private" / "aiter_jit")]
    assert "AITER_JIT_DIR" not in os.environ, "the process environment is put back, as for every setting of a test"


def test_a_successful_test_stages_what_its_compiler_added(compiler, server, tmp_path) -> None:
    session = _session(tmp_path)

    _run(_call(tmp_path, jit=session))

    (kind,) = session.report(NOTHING).kinds
    assert _staged(tmp_path) and kind.contribution is not None and kind.contribution.files == 1


def test_a_failed_test_stages_nothing(compiler, server, tmp_path) -> None:
    session = _session(tmp_path)
    compiler.answers["test"] = compiler.result("test", status="failed", returncode=1)

    outcome = _run(_call(tmp_path, jit=session))

    assert outcome.returncode != 0 and not _staged(tmp_path)


def test_a_profile_test_is_given_the_cache_but_adds_nothing_to_it(compiler, server, tmp_path) -> None:
    """A profile round runs under the profiler's settings: what it compiled is not what a plain test would have."""
    session = _session(tmp_path)
    _with_trace(compiler)

    outcome = _run(_call(tmp_path, _profile_point(tmp_path), kind="profile", jit=session))

    assert outcome.returncode == 0 and compiler.directories[0] is not None, "it was pointed at its directory"
    assert not _staged(tmp_path)


def test_a_call_without_a_cache_runs_as_it_always_did(compiler, server, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)

    outcome = _run(_call(tmp_path))

    assert outcome.returncode == 0 and compiler.directories == [None]
    assert "jit_cache" not in [p.name for p in outcome.phases]


def test_seeding_is_a_phase_of_its_own_before_the_tables_are_laid_out(compiler, server, tmp_path) -> None:
    recorder = PhaseRecorder()

    outcome = _run(_call(tmp_path, jit=_session(tmp_path), phases=recorder))

    assert outcome.phases[0].name == "jit_cache", "the first thing the backend stamps, ahead of kernel_tables"
