# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Re-execute Hyperloom inside a bubblewrap sandbox.

Hyperloom spawns subprocesses from more than seventy call sites -- benchmarks,
specialists, the kernel lane, installers -- so wrapping each one is not a
contract anything could hold. Re-executing the process instead puts every
descendant inside one sandbox by construction, and leaves the spawn sites
untouched.

The sandbox exists to keep a run from depending on, or leaking into, host state
the operator never declared: ``$HOME``, credentials, unrelated checkouts,
ambient caches. It is not a security boundary against hostile code -- bubblewrap
without ``--unshare-net`` shares the network, and the GPU devices are bound in.

Mirrors the Codex sandbox contract in :mod:`hyperloom.common.codex_session`:
validate the mode, probe the real capability, and **fail closed**. A run that
asked to be isolated and cannot be is an error, never a silent downgrade.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .claude_cli import resolve_claude_executable

#: Selects the sandbox. ``off`` (default) keeps today's behaviour exactly.
SANDBOX_MODE_ENV = "HYPERLOOM_SANDBOX_MODE"
#: Set inside the sandbox. The re-exec guard, and what :func:`is_sandboxed` reads.
SANDBOX_ACTIVE_ENV = "HYPERLOOM_SANDBOX_ACTIVE"
#: Extra paths to bind read-write, colon-separated. Outputs the operator owns.
SANDBOX_WRITABLE_ENV = "HYPERLOOM_SANDBOX_WRITABLE"
#: Extra paths to bind read-only, colon-separated. Inputs the operator doesn't
#: own and the run must not edit -- model weights, datasets, reference images.
SANDBOX_READONLY_ENV = "HYPERLOOM_SANDBOX_READONLY"

MODE_OFF = "off"
MODE_BWRAP = "bwrap"
SANDBOX_MODES: tuple[str, ...] = (MODE_OFF, MODE_BWRAP)
DEFAULT_SANDBOX_MODE = MODE_OFF

#: Sandbox ``$HOME``. Deliberately not under /home: a tmpfs over the real home
#: would shadow the uv-managed interpreter, which lives at
#: ``~/.local/share/uv/python/``, and the venv would stop resolving.
SANDBOX_HOME = "/hyperloom-home"

#: Prepended to ``$PATH`` inside the sandbox; holds a ``claude`` link to the CLI
#: resolved outside, which usually lives under the now-hidden ``$HOME``.
SANDBOX_BIN = "/hyperloom-bin"

#: Read-only system trees. ``/sys`` is not optional: ROCm reads GPU topology
#: from ``/sys/class/kfd``, and without it ``torch.cuda.device_count()`` returns
#: 0 -- no error, no warning, just no GPUs.
_SYSTEM_ROOTS: tuple[str, ...] = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/opt", "/sys")

#: GPU character devices. ``--dev-bind`` rather than ``--bind``: a plain bind
#: mount of a device node does not carry device access through.
_GPU_DEVICES: tuple[str, ...] = ("/dev/kfd", "/dev/dri")

# bwrap flags, named so the invocation below reads as intent rather than strings.
_RO_BIND = "--ro-bind"
_RO_BIND_TRY = "--ro-bind-try"
_BIND_TRY = "--bind-try"
_DEV_BIND_TRY = "--dev-bind-try"
_TMPFS = "--tmpfs"
_SETENV = "--setenv"


class SandboxError(RuntimeError):
    """The run asked to be sandboxed and the sandbox could not be established."""


@dataclass(frozen=True)
class SandboxRoots:
    """Paths a command declares from its own arguments: outputs read-write, inputs read-only."""

    writable: tuple[Path, ...] = ()
    readonly: tuple[Path, ...] = ()


def resolve_sandbox_mode(env: Mapping[str, str] | None = None) -> str:
    """Return the validated sandbox mode.

    Args:
        env: Environment to read; defaults to the process environment.

    Returns:
        One of :data:`SANDBOX_MODES`.

    Raises:
        SandboxError: The configured mode is not recognised. Unknown modes fail
            rather than degrading, so a typo cannot silently disable isolation.
    """
    source = os.environ if env is None else env
    mode = (source.get(SANDBOX_MODE_ENV) or DEFAULT_SANDBOX_MODE).strip().lower()
    if mode not in SANDBOX_MODES:
        raise SandboxError(f"{SANDBOX_MODE_ENV}={mode!r} is not one of {', '.join(SANDBOX_MODES)}")
    return mode


def is_sandboxed(env: Mapping[str, str] | None = None) -> bool:
    """Whether this process is already running inside the sandbox."""
    source = os.environ if env is None else env
    return (source.get(SANDBOX_ACTIVE_ENV) or "").strip() == "1"


def interpreter_support_dir(prefix: str | None = None) -> Path | None:
    """Return the interpreter distribution a venv depends on, if it is external.

    A ``uv``-managed venv symlinks ``bin/python`` to a standalone build outside
    itself, so binding only the venv leaves a dangling interpreter. ``pyvenv.cfg``
    records that location in ``home``, and records it *unresolved* -- pointing at
    a version-alias directory (``cpython-3.12-…``) which is itself a symlink to
    the fully-versioned one (``cpython-3.12.13-…``). Binding the resolved target
    instead of the alias breaks the chain, so the declared value is used verbatim.

    The parent of ``home`` is bound, not ``home`` itself: the interpreter needs
    its sibling ``lib/``, and without it Python aborts with "Could not find
    platform independent libraries".

    Args:
        prefix: Virtual environment root; defaults to :data:`sys.prefix`.

    Returns:
        The directory to bind, or ``None`` when the interpreter lives inside the
        venv or under a system tree that is already bound.
    """
    root = Path(prefix or sys.prefix)
    config = root / "pyvenv.cfg"
    if not config.is_file():
        return None
    home = ""
    for line in config.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "home":
            home = value.strip()
            break
    if not home:
        return None
    support = Path(home).parent
    if _is_within(support, root) or any(_is_within(support, Path(r)) for r in _SYSTEM_ROOTS):
        return None
    return support


def interpreter_roots(python: Path) -> tuple[Path, ...]:
    """Directories another venv's interpreter needs inside the sandbox: the venv and its distribution.

    ``python`` is the venv's ``bin/python``, unresolved. A system interpreter needs
    nothing beyond the system trees, which are always bound.
    """
    prefix = python.parent.parent
    if not (prefix / "pyvenv.cfg").is_file():
        return ()
    support = interpreter_support_dir(str(prefix))
    return (prefix,) if support is None else (prefix, support)


def _is_within(path: Path, parent: Path) -> bool:
    """Whether ``path`` is ``parent`` or sits beneath it."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def package_root() -> Path | None:
    """Return the tree this ``hyperloom`` was imported from, if it needs binding.

    A wheel install lives inside the venv and is already covered. A source
    checkout on ``PYTHONPATH`` is not: without binding it the re-executed
    process silently imports a *different* Hyperloom -- whichever one the venv
    happens to contain -- or none at all. Silently running other code than the
    caller's is worse than failing, so the importing tree travels with it.
    """
    import hyperloom

    origin = getattr(hyperloom, "__file__", None)
    if not origin:
        return None
    root = Path(origin).resolve().parent.parent
    venv = Path(sys.prefix)
    if _is_within(root, venv) or any(_is_within(root, Path(r)) for r in _SYSTEM_ROOTS):
        return None
    return root


def writable_roots_from_env(env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Operator-declared writable paths, from :data:`SANDBOX_WRITABLE_ENV`."""
    source = os.environ if env is None else env
    raw = (source.get(SANDBOX_WRITABLE_ENV) or "").strip()
    return tuple(Path(p) for p in raw.split(os.pathsep) if p.strip())


def readonly_roots_from_env(env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Operator-declared read-only paths, from :data:`SANDBOX_READONLY_ENV`."""
    source = os.environ if env is None else env
    raw = (source.get(SANDBOX_READONLY_ENV) or "").strip()
    return tuple(Path(p) for p in raw.split(os.pathsep) if p.strip())


def build_bwrap_args(
    *,
    bwrap: str,
    venv: Path,
    writable_roots: Sequence[Path] = (),
    readonly_roots: Sequence[Path] = (),
    support_dir: Path | None = None,
    package_dir: Path | None = None,
    claude_cli: Path | None = None,
    search_path: str = "",
) -> list[str]:
    """Assemble the bubblewrap invocation that the re-exec runs under.

    Args:
        bwrap: Path to the ``bwrap`` executable.
        venv: Virtual environment root, bound read-only.
        writable_roots: Paths bound read-write. The session directory belongs
            here; everything else stays read-only so a run cannot quietly edit
            installed packages or the operator's checkouts.
        readonly_roots: Paths bound read-only -- inputs the run must not edit
            (model weights, datasets) but that live outside the venv/package
            roots already covered automatically.
        support_dir: External interpreter distribution, from
            :func:`interpreter_support_dir`.
        package_dir: Source tree this Hyperloom was imported from, from
            :func:`package_root`, when it sits outside the venv.
        claude_cli: The Claude CLI resolved outside the sandbox. Its directory
            is bound read-only and it is linked as ``claude`` on a ``$PATH``
            entry of its own, so every lookup by name finds it.
        search_path: The caller's ``$PATH``, kept after that entry.

    Returns:
        Argument list up to and including ``--``; the caller appends the command.
    """
    args = [bwrap, "--die-with-parent"]
    for root in _SYSTEM_ROOTS:
        args += [_RO_BIND_TRY, root, root]
    # A fresh /dev, then the GPU nodes bound back in. /dev/shm is its own tmpfs:
    # the one inside bwrap's minimal /dev is not what Ray's object store expects.
    args += ["--dev", "/dev", _TMPFS, "/dev/shm"]
    for device in _GPU_DEVICES:
        args += [_DEV_BIND_TRY, device, device]
    args += ["--proc", "/proc", _TMPFS, "/tmp"]
    for path in (venv, support_dir, package_dir):
        if path is not None:
            args += [_RO_BIND, str(path), str(path)]
    for root in readonly_roots:
        args += [_RO_BIND_TRY, str(root), str(root)]
    for root in writable_roots:
        args += [_BIND_TRY, str(root), str(root)]
    if claude_cli is not None:
        args += [_RO_BIND, str(claude_cli.parent), str(claude_cli.parent)]
        args += ["--dir", SANDBOX_BIN, "--symlink", str(claude_cli), f"{SANDBOX_BIN}/claude"]
        args += [_SETENV, "PATH", os.pathsep.join(filter(None, (SANDBOX_BIN, search_path)))]
    # A private home, and TMPDIR pointed inside it. An inherited TMPDIR naming an
    # unbound host path is not a missing-directory error: ROCm JIT-compiles its
    # blit kernels at device init, the compiler cannot write its temporaries, and
    # the process dies with SIGSEGV and no diagnostic.
    args += [_TMPFS, SANDBOX_HOME, _SETENV, "HOME", SANDBOX_HOME]
    args += [_SETENV, "TMPDIR", "/tmp"]
    args += [_SETENV, SANDBOX_ACTIVE_ENV, "1"]
    args.append("--")
    return args


def probe_sandbox_capability(bwrap: str, *, runner=subprocess.run) -> bool:
    """Whether this kernel will actually establish the sandbox.

    Finding the executable proves nothing: a kernel with unprivileged user
    namespaces disabled, or a container that forbids them, refuses the mount.
    Running the real namespace operation is the only honest check -- the same
    reasoning as the Codex probe.
    """
    try:
        completed = runner(
            [bwrap, "--unshare-user", "--ro-bind", "/", "/", "/bin/true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0


def verify_sandbox_bindings(
    writable_roots: Sequence[Path] | None = None,
    readonly_roots: Sequence[Path] | None = None,
) -> None:
    """Assert, from inside the sandbox, that the declared binds took.

    bwrap exits 0 for a mount it never performed if something rewrote its
    arguments on the way in -- a shim earlier on ``PATH``, a site wrapper, a
    policy filter. The bind is then simply absent, with no error anywhere, and
    the first thing to touch the directory reports a missing file. That reads
    as a mistyped workspace or a skipped install step, not as a sandbox that
    dropped the mount, which is an expensive thing to debug from the wrong end.

    Declared roots are checked because they are the silent ones. A missing venv
    stops the interpreter; a missing output directory just looks like user
    error.

    Writable roots are verified by writing rather than with :func:`os.access`,
    which cannot tell a read-only mount from a permission bit and, for uid 0,
    reports success on both. Read-only roots get only an existence check --
    a write-test there would fail by design and isn't a bug.

    Args:
        writable_roots: Paths to check; defaults to the declared set.
        readonly_roots: Paths to check; defaults to the declared set.

    Raises:
        SandboxError: A declared root is absent, or a writable one isn't
            actually writable here.
    """
    writable = writable_roots_from_env() if writable_roots is None else writable_roots
    readonly = readonly_roots_from_env() if readonly_roots is None else readonly_roots
    hint = "check for a bwrap wrapper or shim ahead of it on PATH (`which -a bwrap`)"
    for root in readonly:
        if not root.is_dir():
            raise SandboxError(
                f"{SANDBOX_READONLY_ENV} declared {root}, but it does not exist inside "
                f"the sandbox -- the bind was not performed. bwrap reported no error, so {hint}."
            )
    for root in writable:
        if not root.is_dir():
            raise SandboxError(
                f"{SANDBOX_WRITABLE_ENV} declared {root}, but it does not exist inside "
                f"the sandbox -- the bind was not performed. bwrap reported no error, so {hint}."
            )
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".hyperloom-sandbox-"):
                pass
        except OSError as exc:
            raise SandboxError(
                f"{SANDBOX_WRITABLE_ENV} declared {root}, but it is not writable inside "
                f"the sandbox ({exc}). It was bound read-only or not at all, so {hint}."
            ) from exc


def _declare_roots(roots: SandboxRoots) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Merge the command's roots into the operator's, create missing outputs, and publish the result.

    Publishing through the env vars lets :func:`verify_sandbox_bindings` check the
    derived binds too, from inside the sandbox.
    """
    writable = tuple(dict.fromkeys((*writable_roots_from_env(), *roots.writable)))
    readonly = tuple(dict.fromkeys((*readonly_roots_from_env(), *roots.readonly)))
    for root in writable:
        root.mkdir(parents=True, exist_ok=True)
    os.environ[SANDBOX_WRITABLE_ENV] = os.pathsep.join(map(str, writable))
    os.environ[SANDBOX_READONLY_ENV] = os.pathsep.join(map(str, readonly))
    return writable, readonly


def reexec_if_required(argv: Sequence[str] | None = None, roots: SandboxRoots = SandboxRoots()) -> None:
    """Re-execute this process inside the sandbox when one was asked for.

    Returns immediately when the mode is ``off``. When the process is already
    sandboxed it returns too, after :func:`verify_sandbox_bindings` confirms
    the binds took. Otherwise this call does not return: the process is
    replaced.

    Args:
        argv: The ``hyperloom`` sub-command and its arguments -- what
            :func:`hyperloom.cli.main` received, not :data:`sys.argv`. The
            replacement is spelled ``python -m hyperloom <argv>`` rather than
            rebuilt from ``sys.argv``, because ``sys.argv`` does not survive
            every entry point: under ``python -c`` it holds only ``['-c']`` and
            the body is unrecoverable. Naming the module is exact for the
            console script, ``-m``, and an embedded caller alike.
        roots: What the command derived from its own arguments, bound in
            addition to the operator's ``HYPERLOOM_SANDBOX_*`` roots.

    Raises:
        SandboxError: A sandbox was requested and cannot be established. Never
            downgraded to an unsandboxed run -- the caller asked for isolation,
            and silently not providing it is the one outcome worse than failing.
    """
    mode = resolve_sandbox_mode()
    if mode == MODE_OFF:
        return
    if is_sandboxed():
        # Second pass, now inside the sandbox: confirm what was asked for is
        # actually here before any command relies on it.
        verify_sandbox_bindings()
        return
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SandboxError(f"{SANDBOX_MODE_ENV}=bwrap requires the bwrap executable on PATH")
    if not probe_sandbox_capability(bwrap):
        raise SandboxError(
            f"{SANDBOX_MODE_ENV}=bwrap: this kernel refuses to create the sandbox "
            "(unprivileged user namespaces may be disabled). Refusing to continue "
            "unsandboxed."
        )
    venv = Path(sys.prefix)
    command = [sys.executable, "-m", "hyperloom", *(list(sys.argv[1:]) if argv is None else list(argv))]
    claude_cli = resolve_claude_executable()
    writable, readonly = _declare_roots(roots)
    args = build_bwrap_args(
        bwrap=bwrap,
        venv=venv,
        writable_roots=writable,
        readonly_roots=readonly,
        support_dir=interpreter_support_dir(),
        package_dir=package_root(),
        claude_cli=Path(claude_cli).resolve() if claude_cli else None,
        search_path=os.environ.get("PATH", ""),
    )
    os.execv(bwrap, [*args, *command])
