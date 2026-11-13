# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Running the whole process inside bubblewrap: mode resolution, derived binds, and the real launch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.common.reexec import (
    ACTIVE_ENV,
    BWRAP_ENV,
    MODE_BWRAP,
    MODE_OFF,
    READONLY_ENV,
    REEXEC_ENV,
    TMP_ENV,
    WRITABLE_ENV,
    ReexecError,
    ReexecRoots,
    _declare_roots,
    _policy,
    foreign_interpreter_roots,
    is_reexeced,
    package_root,
    reexec_if_required,
    resolve_mode,
    verify_bindings,
)
from hyperloom_common.bwrap import Scratch, bwrap_probe

_REAL_BWRAP = "/usr/bin/bwrap"
_probe = bwrap_probe(_REAL_BWRAP) if Path(_REAL_BWRAP).exists() else None
needs_bwrap = pytest.mark.skipif(_probe is None or not _probe.ok, reason="no usable bubblewrap here")


def test_the_default_is_off() -> None:
    assert resolve_mode({}) == MODE_OFF


def test_a_typo_is_refused_rather_than_ignored() -> None:
    """Silently leaving a run unconfined is the one outcome worse than failing."""
    with pytest.raises(ReexecError, match="is not one of"):
        resolve_mode({REEXEC_ENV: "bwarp"})


def test_the_mode_is_case_and_space_insensitive() -> None:
    assert resolve_mode({REEXEC_ENV: "  BWRAP "}) == MODE_BWRAP


def test_off_returns_without_touching_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REEXEC_ENV, raising=False)
    reexec_if_required(["optimize"], ReexecRoots(writable=(Path("/nonexistent/should-not-be-made"),)))
    assert not Path("/nonexistent/should-not-be-made").exists()


def test_the_inside_process_is_recognised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ACTIVE_ENV, "1")
    assert is_reexeced() is True


def test_declared_roots_merge_the_operators_and_the_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Republished so the in-sandbox check covers the derived binds, not only the typed ones."""
    operator, derived, ro = tmp_path / "op", tmp_path / "derived", tmp_path / "ro"
    ro.mkdir()
    monkeypatch.setenv(WRITABLE_ENV, str(operator))
    monkeypatch.setenv(READONLY_ENV, str(ro))

    writable, readonly = _declare_roots(ReexecRoots(writable=(derived,)))

    assert writable == (operator, derived)
    assert readonly == (ro,)
    assert derived.is_dir(), "an output root the run will write is created before the bind"
    assert os.environ[WRITABLE_ENV] == os.pathsep.join((str(operator), str(derived)))


def test_verify_bindings_names_a_root_that_never_arrived(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """bwrap exits 0 for a mount a shim dropped, so absence is the only evidence."""
    monkeypatch.setenv(READONLY_ENV, str(tmp_path / "missing"))
    monkeypatch.setenv(WRITABLE_ENV, "")
    with pytest.raises(ReexecError, match="which is not inside the sandbox"):
        verify_bindings()


def test_verify_bindings_writes_rather_than_asking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """os.access cannot tell a read-only mount from a permission bit, and lies outright for uid 0."""
    readonly_root = tmp_path / "ro"
    readonly_root.mkdir()
    readonly_root.chmod(0o500)
    monkeypatch.setenv(WRITABLE_ENV, str(readonly_root))
    monkeypatch.setenv(READONLY_ENV, "")
    try:
        if os.access(readonly_root, os.W_OK):
            pytest.skip("running as a uid that writes anywhere; the write test is what this asserts")
        with pytest.raises(ReexecError, match="not writable inside the sandbox"):
            verify_bindings()
    finally:
        readonly_root.chmod(0o700)


def test_a_declared_tmp_that_is_not_a_directory_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REEXEC_ENV, MODE_BWRAP)
    monkeypatch.setenv(TMP_ENV, str(tmp_path / "not-there"))
    monkeypatch.delenv(ACTIVE_ENV, raising=False)
    with pytest.raises(ReexecError, match="not an existing directory"):
        reexec_if_required(["optimize"])


def test_the_package_root_travels_when_it_is_a_source_checkout() -> None:
    """Without it the replacement silently imports whichever Hyperloom the venv holds, or none."""
    root = package_root()
    if root is None:
        pytest.skip("installed as a wheel inside the venv, which is already bound")
    assert (root / "pyproject.toml").is_file()


def test_a_foreign_uv_venv_binds_the_alias_and_its_target() -> None:
    """uv records ``home`` unresolved, naming a version-alias link. Binding only the alias
    leaves its target unbound and the interpreter dead inside the sandbox."""
    roots = foreign_interpreter_roots(Path(sys.prefix) / "bin" / "python")
    if not roots:
        pytest.skip("not running from a venv with a pyvenv.cfg")
    assert Path(sys.prefix) in roots
    resolved = [r for r in roots if r == r.resolve()]
    assert resolved, "the resolved distribution directory is bound alongside the alias"


def test_a_system_interpreter_needs_no_extra_roots(tmp_path: Path) -> None:
    assert foreign_interpreter_roots(tmp_path / "bin" / "python") == ()


@needs_bwrap
def test_the_sandbox_really_binds_what_was_declared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The end-to-end property: declared roots arrive, the writable one writes through to the host."""
    work, readable = tmp_path / "work", tmp_path / "readable"
    readable.mkdir()
    (readable / "input.txt").write_text("hello\n", encoding="utf-8")
    monkeypatch.delenv(WRITABLE_ENV, raising=False)
    monkeypatch.delenv(READONLY_ENV, raising=False)
    writable, readonly = _declare_roots(ReexecRoots(writable=(work,), readonly=(readable,)))
    home = work / ".hyperloom-home"
    home.mkdir(parents=True, exist_ok=True)
    policy = _policy(writable, (*readonly, home.parent), home, Scratch.create())

    body = (
        "import os;"
        "from hyperloom.common.reexec import verify_bindings; verify_bindings();"
        f"open({str(work / 'written')!r}, 'w').write(open({str(readable / 'input.txt')!r}).read());"
        "print(os.environ['HOME'])"
    )
    run = subprocess.run(
        policy.argv(_probe, [sys.executable, "-c", body]),
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])},
    )

    assert run.returncode == 0, run.stderr[-2000:]
    assert (work / "written").read_text(encoding="utf-8") == "hello\n"
    assert run.stdout.strip() == str(home), "HOME is the run's own, not the operator's"


@needs_bwrap
def test_a_specialist_can_still_be_confined_inside_the_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The two tiers nest on purpose: the outer bounds the operator's blast radius, the inner
    bounds the agent. If the inner probe stopped passing, every specialist would silently
    fall back to NoIsolation inside a sandbox that grants write to the framework tree."""
    monkeypatch.delenv(WRITABLE_ENV, raising=False)
    monkeypatch.delenv(READONLY_ENV, raising=False)
    writable, readonly = _declare_roots(ReexecRoots(writable=(tmp_path / "work",)))
    home = (tmp_path / "work") / ".hyperloom-home"
    home.mkdir(parents=True, exist_ok=True)
    policy = _policy(writable, (*readonly, home.parent), home, Scratch.create())

    run = subprocess.run(
        policy.argv(
            _probe,
            [sys.executable, "-c", "from hyperloom_common.bwrap import bwrap_probe; print(bwrap_probe().ok)"],
        ),
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3]), "PATH": "/usr/bin:/bin"},
    )

    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "True", "bubblewrap must still confine a specialist from inside"


def test_a_named_bwrap_is_used_instead_of_the_one_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A site that installs a ``bwrap`` wrapper for something else leaves it ahead of the
    real one, and the probe then judges the wrapper. Naming the real binary is what the
    board run scripts were already doing by hard-coding ``/usr/bin/bwrap``."""
    monkeypatch.setenv(REEXEC_ENV, MODE_BWRAP)
    monkeypatch.setenv(BWRAP_ENV, str(tmp_path / "not-a-program"))
    monkeypatch.delenv(ACTIVE_ENV, raising=False)
    monkeypatch.delenv(TMP_ENV, raising=False)
    with pytest.raises(ReexecError, match="not an executable here"):
        reexec_if_required(["optimize"])


@needs_bwrap
def test_the_named_bwrap_is_the_one_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe must judge what was named, not whatever ``which`` finds first."""
    assert bwrap_probe(_REAL_BWRAP).path == _REAL_BWRAP
