# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sandbox re-exec: mode resolution, bind construction, and failing closed."""

from __future__ import annotations

import subprocess
import sys

import pytest

from hyperloom.common.sandbox import (
    SANDBOX_ACTIVE_ENV,
    SANDBOX_HOME,
    SANDBOX_MODE_ENV,
    SANDBOX_WRITABLE_ENV,
    SandboxError,
    build_bwrap_args,
    interpreter_support_dir,
    is_sandboxed,
    probe_sandbox_capability,
    reexec_if_required,
    resolve_sandbox_mode,
    verify_sandbox_bindings,
    writable_roots_from_env,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Keep the ambient environment out of every case."""
    for key in (SANDBOX_MODE_ENV, SANDBOX_ACTIVE_ENV, SANDBOX_WRITABLE_ENV):
        monkeypatch.delenv(key, raising=False)


def _venv(tmp_path, home: str | None):
    """Build a venv root whose pyvenv.cfg declares ``home``."""
    root = tmp_path / "venv"
    (root / "bin").mkdir(parents=True)
    body = "implementation = CPython\n"
    if home is not None:
        body += f"home = {home}\n"
    (root / "pyvenv.cfg").write_text(body)
    return root


# --- mode resolution -------------------------------------------------------


def test_default_mode_is_off_so_existing_runs_are_unchanged():
    assert resolve_sandbox_mode({}) == "off"


@pytest.mark.parametrize("value", ["bwrap", "BWRAP", "  bwrap  "])
def test_mode_is_normalised(value):
    assert resolve_sandbox_mode({SANDBOX_MODE_ENV: value}) == "bwrap"


def test_unknown_mode_raises_rather_than_degrading():
    """A typo must not quietly mean 'unsandboxed'."""
    with pytest.raises(SandboxError, match="not one of"):
        resolve_sandbox_mode({SANDBOX_MODE_ENV: "bubblewrap"})


def test_is_sandboxed_reads_the_active_marker():
    assert is_sandboxed({SANDBOX_ACTIVE_ENV: "1"}) is True
    assert is_sandboxed({SANDBOX_ACTIVE_ENV: "0"}) is False
    assert is_sandboxed({}) is False


# --- interpreter support dir ----------------------------------------------


def test_external_interpreter_resolves_to_the_distribution_root(tmp_path):
    """The parent of ``home`` is bound: the interpreter needs its sibling lib/."""
    support = tmp_path / "uv" / "cpython-3.12-linux-x86_64-gnu"
    (support / "bin").mkdir(parents=True)
    root = _venv(tmp_path, str(support / "bin"))

    assert interpreter_support_dir(str(root)) == support


def test_the_alias_path_is_used_verbatim_not_resolved(tmp_path):
    """``home`` names a version-alias symlink; resolving it breaks the chain.

    uv points the venv at ``cpython-3.12-…``, itself a symlink to
    ``cpython-3.12.13-…``. Binding the resolved target leaves the alias absent
    and the interpreter unreachable, so the declared value must survive intact.
    """
    real = tmp_path / "uv" / "cpython-3.12.13-linux-x86_64-gnu"
    (real / "bin").mkdir(parents=True)
    alias = tmp_path / "uv" / "cpython-3.12-linux-x86_64-gnu"
    alias.symlink_to(real)
    root = _venv(tmp_path, str(alias / "bin"))

    resolved = interpreter_support_dir(str(root))

    assert resolved == alias
    assert resolved != real


def test_interpreter_inside_the_venv_needs_no_extra_bind(tmp_path):
    root = _venv(tmp_path, None)
    assert interpreter_support_dir(str(root)) is None


def test_system_interpreter_needs_no_extra_bind(tmp_path):
    """A venv on /usr/bin/python is covered by the system roots already bound."""
    root = _venv(tmp_path, "/usr/bin")
    assert interpreter_support_dir(str(root)) is None


def test_missing_pyvenv_cfg_is_not_an_error(tmp_path):
    (tmp_path / "bare").mkdir()
    assert interpreter_support_dir(str(tmp_path / "bare")) is None


# --- bind construction -----------------------------------------------------


def _args(tmp_path, **kw):
    return build_bwrap_args(bwrap="/usr/bin/bwrap", venv=tmp_path / "venv", **kw)


def test_sysfs_is_bound(tmp_path):
    """Without /sys, ROCm reports zero GPUs and raises nothing."""
    args = _args(tmp_path)
    assert args[args.index("/sys") - 1] == "--ro-bind-try"


def test_tmpdir_is_redirected_into_the_sandbox(tmp_path):
    """An inherited TMPDIR naming an unbound path makes ROCm's JIT segfault."""
    args = _args(tmp_path)
    assert args[args.index("TMPDIR") + 1] == "/tmp"


def test_home_does_not_shadow_the_interpreter(tmp_path):
    """A tmpfs over the real home would hide ~/.local/share/uv/python."""
    support = tmp_path / "home" / "user" / ".local" / "share" / "uv" / "python" / "cpy"
    args = _args(tmp_path, support_dir=support)
    assert SANDBOX_HOME not in str(support)
    assert args[args.index("HOME") + 1] == SANDBOX_HOME


def test_gpu_devices_use_dev_bind(tmp_path):
    """A plain bind of a device node does not carry device access."""
    args = _args(tmp_path)
    for device in ("/dev/kfd", "/dev/dri"):
        assert args[args.index(device) - 1] == "--dev-bind-try"


def test_dev_shm_gets_its_own_tmpfs(tmp_path):
    """Ray's object store is not served by the /dev/shm inside bwrap's --dev."""
    args = _args(tmp_path)
    assert args[args.index("/dev/shm") - 1] == "--tmpfs"


def test_the_venv_is_read_only(tmp_path):
    """Installed packages are not a deliverable; writes to them are a mistake."""
    venv = str(tmp_path / "venv")
    args = _args(tmp_path)
    assert args[args.index(venv) - 1] == "--ro-bind"


def test_declared_writable_roots_are_bound_read_write(tmp_path):
    session = tmp_path / "session"
    args = _args(tmp_path, writable_roots=[session])
    assert args[args.index(str(session)) - 1] == "--bind-try"


def test_the_active_marker_is_set_for_the_child(tmp_path):
    """Without it the child would re-exec itself forever."""
    args = _args(tmp_path)
    assert args[args.index(SANDBOX_ACTIVE_ENV) + 1] == "1"


def test_args_end_with_a_separator(tmp_path):
    assert _args(tmp_path)[-1] == "--"


def test_writable_roots_parse_from_the_environment():
    roots = writable_roots_from_env({SANDBOX_WRITABLE_ENV: "/a:/b"})
    assert [str(r) for r in roots] == ["/a", "/b"]
    assert writable_roots_from_env({}) == ()


# --- capability probe and failing closed -----------------------------------


def test_probe_runs_a_real_namespace_operation():
    """Finding the binary proves nothing; the mount must actually be attempted."""
    seen = {}

    def runner(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    assert probe_sandbox_capability("/usr/bin/bwrap", runner=runner) is True
    assert "--unshare-user" in seen["cmd"]


def test_probe_reports_failure_when_the_kernel_refuses():
    def runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1)

    assert probe_sandbox_capability("/usr/bin/bwrap", runner=runner) is False


def test_probe_survives_a_missing_executable():
    def runner(cmd, **kwargs):
        raise OSError("no such file")

    assert probe_sandbox_capability("/nope/bwrap", runner=runner) is False


# --- the binds are verified from inside -------------------------------------


def test_verify_accepts_a_root_that_is_really_writable(tmp_path):
    verify_sandbox_bindings([tmp_path])


def test_verify_rejects_a_root_the_bind_never_created(tmp_path):
    """A dropped bind leaves no trace; bwrap still exits 0."""
    with pytest.raises(SandboxError, match="does not exist inside"):
        verify_sandbox_bindings([tmp_path / "never-bound"])


def test_verify_rejects_a_root_bound_read_only(tmp_path, monkeypatch):
    """Patched rather than chmod'd: uid 0 may write through a mode of 0o500."""

    def refuse(*args, **kwargs):
        raise OSError("Read-only file system")

    monkeypatch.setattr("hyperloom.common.sandbox.tempfile.NamedTemporaryFile", refuse)
    with pytest.raises(SandboxError, match="not writable inside"):
        verify_sandbox_bindings([tmp_path])


def test_verify_leaves_nothing_behind(tmp_path):
    verify_sandbox_bindings([tmp_path])
    assert list(tmp_path.iterdir()) == []


def test_verify_names_the_shim_as_the_likely_cause(tmp_path):
    """The failure this exists to catch is an argument rewriter on PATH."""
    with pytest.raises(SandboxError, match="which -a bwrap"):
        verify_sandbox_bindings([tmp_path / "never-bound"])


def test_being_inside_the_sandbox_triggers_verification(monkeypatch, tmp_path):
    monkeypatch.setenv(SANDBOX_MODE_ENV, "bwrap")
    monkeypatch.setenv(SANDBOX_ACTIVE_ENV, "1")
    monkeypatch.setenv(SANDBOX_WRITABLE_ENV, str(tmp_path / "never-bound"))
    with pytest.raises(SandboxError, match="does not exist inside"):
        reexec_if_required([])


def test_mode_off_skips_verification_entirely(monkeypatch, tmp_path):
    """An unsandboxed run declares nothing and must not be second-guessed."""
    monkeypatch.setenv(SANDBOX_WRITABLE_ENV, str(tmp_path / "never-bound"))
    reexec_if_required([])


def test_mode_off_does_not_reexec(monkeypatch):
    monkeypatch.setattr(
        "hyperloom.common.sandbox.os.execv",
        lambda *a: pytest.fail("must not re-exec when the mode is off"),
    )
    reexec_if_required([])


def test_already_sandboxed_does_not_reexec(monkeypatch):
    """The guard that stops an infinite exec loop."""
    monkeypatch.setenv(SANDBOX_MODE_ENV, "bwrap")
    monkeypatch.setenv(SANDBOX_ACTIVE_ENV, "1")
    monkeypatch.setattr(
        "hyperloom.common.sandbox.os.execv",
        lambda *a: pytest.fail("must not re-exec inside the sandbox"),
    )
    reexec_if_required([])


def test_missing_bwrap_fails_closed(monkeypatch):
    monkeypatch.setenv(SANDBOX_MODE_ENV, "bwrap")
    monkeypatch.setattr("hyperloom.common.sandbox.shutil.which", lambda _: None)
    with pytest.raises(SandboxError, match="requires the bwrap executable"):
        reexec_if_required([])


def test_an_incapable_kernel_fails_closed(monkeypatch):
    """Refusing beats running unsandboxed after the operator asked for isolation."""
    monkeypatch.setenv(SANDBOX_MODE_ENV, "bwrap")
    monkeypatch.setattr("hyperloom.common.sandbox.shutil.which", lambda _: "/usr/bin/bwrap")
    monkeypatch.setattr("hyperloom.common.sandbox.probe_sandbox_capability", lambda *a, **k: False)
    with pytest.raises(SandboxError, match="refuses to create the sandbox"):
        reexec_if_required([])


def test_reexec_passes_the_original_command_through(monkeypatch):
    monkeypatch.setenv(SANDBOX_MODE_ENV, "bwrap")
    monkeypatch.setattr("hyperloom.common.sandbox.shutil.which", lambda _: "/usr/bin/bwrap")
    monkeypatch.setattr("hyperloom.common.sandbox.probe_sandbox_capability", lambda *a, **k: True)
    captured = {}
    monkeypatch.setattr(
        "hyperloom.common.sandbox.os.execv",
        lambda path, argv: captured.update(path=path, argv=argv),
    )

    reexec_if_required(["optimize", "--max-hours", "1"])

    assert captured["path"] == "/usr/bin/bwrap"
    argv = captured["argv"]
    assert argv[0] == "/usr/bin/bwrap"
    # Everything after the separator is the original command, re-run under this
    # interpreter, so the sandboxed process does exactly what was asked of it.
    assert argv[argv.index("--") + 1 :] == [
        sys.executable,
        "-m",
        "hyperloom",
        "optimize",
        "--max-hours",
        "1",
    ]


# --- the importing tree travels with the process ---------------------------


def test_source_checkout_is_bound(tmp_path):
    """A PYTHONPATH checkout is not inside the venv and must be carried in."""
    pkg = tmp_path / "src"
    args = _args(tmp_path, package_dir=pkg)
    assert args[args.index(str(pkg)) - 1] == "--ro-bind"


def test_package_root_returns_none_for_a_wheel_install(monkeypatch, tmp_path):
    """Installed inside the venv, it is already covered by the venv bind."""
    venv = tmp_path / "venv"
    pkg = venv / "lib" / "python3.12" / "site-packages" / "hyperloom"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    monkeypatch.setattr("hyperloom.common.sandbox.sys.prefix", str(venv))

    import hyperloom

    monkeypatch.setattr(hyperloom, "__file__", str(pkg / "__init__.py"))
    from hyperloom.common.sandbox import package_root

    assert package_root() is None
