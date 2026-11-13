# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the whole ``hyperloom`` process inside bubblewrap, when the operator asks for it.

The sandboxes elsewhere in this tree each confine one *kind* of work: a specialist
(:mod:`hyperloom.orchestrator.specialists.launch_env`), an agent or dev sandbox, a test
sandbox (:mod:`hyperloom.sandbox.confined`), a native compile. Each answers "what may this
agent reach". None answers the operator's question -- "I ran ``hyperloom optimize`` against
a tree I do not fully trust; bound what the *run* can touch" -- which is what this does.

The two layers nest rather than replace each other, and both are wanted: the outer one
bounds the operator's blast radius, the inner one bounds the agent. A specialist launched
inside this sandbox probes bubblewrap again and gets its own, narrower, confinement; that
is why nothing here disables the inner tier. Verified working two levels deep at
bubblewrap 0.9.0.

Off by default. ``HYPERLOOM_REEXEC=bwrap`` turns it on and anything else is refused, so a
typo cannot silently leave a run unconfined.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from hyperloom_common.bwrap import SYSTEM_ROOTS, BwrapPolicy, Scratch, bwrap_probe, interpreter_roots
from hyperloom.common.agent_cli import resolve_claude_executable
from hyperloom.common.confinement import toolchain_paths

#: Whether to re-exec, and into what. ``off`` (default) or ``bwrap``.
REEXEC_ENV = "HYPERLOOM_REEXEC"
#: Set inside the sandbox, so the second pass verifies instead of re-execing.
ACTIVE_ENV = "HYPERLOOM_REEXEC_ACTIVE"
#: Operator-declared writable paths, ``os.pathsep``-separated. Republished with the
#: command's own, so the in-sandbox check covers the derived binds too.
WRITABLE_ENV = "HYPERLOOM_REEXEC_WRITABLE"
#: Operator-declared read-only paths, ``os.pathsep``-separated.
READONLY_ENV = "HYPERLOOM_REEXEC_READONLY"
#: Where the sandbox's ``/tmp`` is taken from; the host's ``/tmp`` when unset.
TMP_ENV = "HYPERLOOM_REEXEC_TMP"
#: The bubblewrap to use, when the one first on ``PATH`` is not the one wanted. A site that
#: installs a ``bwrap`` wrapper for something else -- an IDE's own sandbox, say -- leaves it
#: ahead of ``/usr/bin/bwrap``, and the probe then judges the wrapper rather than bubblewrap.
BWRAP_ENV = "HYPERLOOM_REEXEC_BWRAP"

MODE_OFF = "off"
MODE_BWRAP = "bwrap"
REEXEC_MODES: tuple[str, ...] = (MODE_OFF, MODE_BWRAP)

_HINT = "check for a bwrap wrapper or shim ahead of it on PATH (`which -a bwrap`)"


class ReexecError(RuntimeError):
    """A sandbox was asked for and cannot be established. Never downgraded to an unconfined run."""


@dataclass(frozen=True)
class ReexecRoots:
    """What a sub-command derived from its own arguments, bound beside the operator's."""

    writable: tuple[Path, ...] = field(default_factory=tuple)
    readonly: tuple[Path, ...] = field(default_factory=tuple)


def resolve_mode(env: Mapping[str, str] | None = None) -> str:
    """The requested mode. Fails closed: an unrecognised value is refused, not ignored.

    Raises:
        ReexecError: ``REEXEC_ENV`` names something other than a known mode.
    """
    source = os.environ if env is None else env
    mode = (source.get(REEXEC_ENV) or MODE_OFF).strip().lower()
    if mode not in REEXEC_MODES:
        raise ReexecError(f"{REEXEC_ENV}={mode!r} is not one of {', '.join(REEXEC_MODES)}")
    return mode


def is_reexeced(env: Mapping[str, str] | None = None) -> bool:
    """Whether this process is already the one inside the sandbox."""
    source = os.environ if env is None else env
    return bool((source.get(ACTIVE_ENV) or "").strip())


def _roots_from_env(name: str, env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    source = os.environ if env is None else env
    raw = (source.get(name) or "").strip()
    return tuple(Path(p) for p in raw.split(os.pathsep) if p.strip())


def foreign_interpreter_roots(python: Path) -> tuple[Path, ...]:
    """What *another* venv's interpreter needs bound: the venv and the distribution behind it.

    :func:`hyperloom_common.bwrap.interpreter_roots` answers this for the running process by
    reading ``sys.prefix``/``sys.base_prefix``, and is the right tool there. It cannot answer
    it for a venv named on the command line -- the benchmark's ``--python`` -- without running
    that interpreter, which is a subprocess on an operator-named binary before any sandbox
    exists. ``pyvenv.cfg`` answers it statically instead.

    A ``uv``-managed venv symlinks ``bin/python`` to a standalone build outside itself and
    records that build in ``home`` *unresolved*, naming a version-alias directory
    (``cpython-3.12-…``) that is itself a link to the fully-versioned one. Both are returned:
    binding only the alias leaves its target unbound and the interpreter dead inside.

    The *parent* of ``home`` is bound, not ``home``: the interpreter needs its sibling
    ``lib/``, without which Python aborts with "Could not find platform independent libraries".
    """
    prefix = python.parent.parent
    config = prefix / "pyvenv.cfg"
    if not config.is_file():
        return ()
    home = ""
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "home":
            home = value.strip()
            break
    if not home:
        return (prefix,)
    support = Path(home).parent
    if _within(support, prefix) or any(_within(support, Path(root)) for root in SYSTEM_ROOTS):
        return (prefix,)
    resolved = support.resolve()
    return (prefix, support) if resolved == support else (prefix, support, resolved)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def package_root() -> Path | None:
    """The tree this ``hyperloom`` was imported from, when binding the venv does not cover it.

    A wheel install lives inside the venv. A source checkout on ``PYTHONPATH`` does not, and
    without it the replacement process silently imports a *different* Hyperloom -- whichever
    the venv happens to hold, or none. Running other code than the caller's is worse than
    failing, so the importing tree travels with the process.
    """
    import hyperloom

    origin = getattr(hyperloom, "__file__", None)
    if not origin:
        return None
    root = Path(origin).resolve().parent.parent
    if root.name == "src" and (root.parent / "pyproject.toml").is_file():
        root = root.parent
    if _within(root, Path(sys.prefix)) or any(_within(root, Path(system)) for system in SYSTEM_ROOTS):
        return None
    return root


def _declare_roots(roots: ReexecRoots) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Merge the command's roots into the operator's, create missing outputs, and republish.

    Republishing is what lets :func:`verify_bindings` check the *derived* binds from inside,
    not only the ones the operator typed.
    """
    writable = tuple(dict.fromkeys((*_roots_from_env(WRITABLE_ENV), *roots.writable)))
    readonly = tuple(dict.fromkeys((*_roots_from_env(READONLY_ENV), *roots.readonly)))
    for root in writable:
        root.mkdir(parents=True, exist_ok=True)
    os.environ[WRITABLE_ENV] = os.pathsep.join(map(str, writable))
    os.environ[READONLY_ENV] = os.pathsep.join(map(str, readonly))
    return writable, readonly


def verify_bindings(writable: Sequence[Path] | None = None, readonly: Sequence[Path] | None = None) -> None:
    """Assert from inside the sandbox that the declared binds actually took.

    bubblewrap exits 0 for a mount it never performed if something rewrote its arguments on
    the way in -- a shim earlier on ``PATH``, a site wrapper, a policy filter. The bind is
    then simply absent, with no error anywhere, and the first thing to touch the directory
    reports a missing file. That reads as a mistyped workspace or a skipped install step
    rather than a sandbox that dropped the mount, which is expensive to debug from that end.
    This host has exactly such a shim at ``~/.local/bin/bwrap``.

    The probe's canary asserts that bubblewrap *can* confine; this asserts that *these*
    paths arrived. Writable roots are checked by writing: :func:`os.access` cannot tell a
    read-only mount from a permission bit, and for uid 0 reports success on both. Read-only
    roots get an existence check -- a write test there fails by design.

    Raises:
        ReexecError: A declared root is absent, or a writable one is not writable here.
    """
    for root in _roots_from_env(READONLY_ENV) if readonly is None else readonly:
        if not root.is_dir():
            raise ReexecError(
                f"{READONLY_ENV} declared {root}, which is not inside the sandbox: the bind was "
                f"not performed and bubblewrap reported no error, so {_HINT}."
            )
    for root in _roots_from_env(WRITABLE_ENV) if writable is None else writable:
        if not root.is_dir():
            raise ReexecError(
                f"{WRITABLE_ENV} declared {root}, which is not inside the sandbox: the bind was "
                f"not performed and bubblewrap reported no error, so {_HINT}."
            )
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".hyperloom-reexec-"):
                pass
        except OSError as exc:
            raise ReexecError(
                f"{WRITABLE_ENV} declared {root}, which is not writable inside the sandbox "
                f"({exc}): it was bound read-only or not at all, so {_HINT}."
            ) from exc


def _policy(writable: Sequence[Path], readonly: Sequence[Path], home: Path, scratch: Scratch) -> BwrapPolicy:
    """What the re-executed process may reach.

    Built as a :class:`BwrapPolicy` rather than through :class:`~hyperloom.common.confinement.Confinement`
    on purpose. ``Confinement`` and ``credential_paths`` exist to keep the app's own keys away
    from an *agent*; here the process being confined **is** the app, and masking its credentials
    would take the Claude login out from under the thing that owns it.
    """
    claude = resolve_claude_executable()
    toolchain = toolchain_paths(os.environ, claude, writable=tuple(writable)) if claude else ()
    read_only = (
        *SYSTEM_ROOTS,
        *map(str, interpreter_roots()),
        *([str(root)] if (root := package_root()) else []),
        *map(str, toolchain),
        *map(str, readonly),
    )
    return BwrapPolicy(
        read_only=tuple(dict.fromkeys(read_only)),
        writable=tuple(dict.fromkeys(map(str, writable))),
        devices=tuple(node for node in ("/dev/kfd", "/dev/dri") if os.path.exists(node)),
        tmp=scratch.tmp,
        shm=scratch.shm,
        # HOME inside the sandbox is a writable directory of the run's own: the operator's
        # may hold credentials this run has no business reading, and TMPDIR is pinned to the
        # sandbox's /tmp because an inherited one pointing outside it is simply absent here.
        marks=((ACTIVE_ENV, "1"), ("HOME", str(home)), ("TMPDIR", "/tmp")),
        die_with_parent=True,
    )


def reexec_if_required(argv: Sequence[str] | None = None, roots: ReexecRoots = ReexecRoots()) -> None:
    """Replace this process with itself inside the sandbox, when one was asked for.

    Returns immediately when the mode is ``off``, and after :func:`verify_bindings` when this
    is already the inside process. Otherwise it does not return.

    Args:
        argv: The sub-command and its arguments, as ``hyperloom.cli.main`` received them --
            not :data:`sys.argv`. The replacement is spelled ``python -m hyperloom <argv>``
            rather than rebuilt from ``sys.argv``, which does not survive every entry point:
            under ``python -c`` it holds only ``['-c']`` and the body is unrecoverable.
        roots: What the sub-command derived from its own arguments, bound beside the
            operator's declared ones.

    Raises:
        ReexecError: A sandbox was requested and cannot be established.
    """
    if resolve_mode() == MODE_OFF:
        return
    if is_reexeced():
        verify_bindings()
        return
    # The operator's own input is validated before the probe: a mistyped path is theirs to
    # fix either way, and reporting the probe's failure instead would send them after the
    # wrong thing.
    declared_tmp = (os.environ.get(TMP_ENV) or "").strip()
    if declared_tmp and not Path(declared_tmp).is_dir():
        raise ReexecError(f"{TMP_ENV} names {declared_tmp}, which is not an existing directory")
    declared_bwrap = (os.environ.get(BWRAP_ENV) or "").strip()
    if declared_bwrap and not os.access(declared_bwrap, os.X_OK):
        raise ReexecError(f"{BWRAP_ENV} names {declared_bwrap}, which is not an executable here")
    if not declared_bwrap and shutil.which("bwrap") is None:
        raise ReexecError(f"{REEXEC_ENV}={MODE_BWRAP} needs the bwrap executable on PATH, or {BWRAP_ENV} set")
    probe = bwrap_probe(declared_bwrap or None)
    if not probe.ok:
        raise ReexecError(
            f"{REEXEC_ENV}={MODE_BWRAP}: {probe.path or 'bubblewrap'} cannot sandbox here "
            f"({'; '.join(probe.problems) or 'no reason reported'}). Refusing to run unconfined. "
            f"If that is a wrapper rather than bubblewrap itself, name the real one in {BWRAP_ENV}."
        )
    if declared_tmp:
        # Scratch.create() takes its /tmp from tempfile, which reads TMPDIR.
        os.environ["TMPDIR"] = declared_tmp
    writable, readonly = _declare_roots(roots)
    home = Path(writable[0]) / ".hyperloom-home" if writable else Path(tempfile.mkdtemp(prefix="hl-home-"))
    home.mkdir(parents=True, exist_ok=True)
    scratch = Scratch.create()
    command = [sys.executable, "-m", "hyperloom", *(list(sys.argv[1:]) if argv is None else list(argv))]
    line = _policy(writable, (*readonly, home.parent), home, scratch).argv(probe, command)
    os.execv(line[0], line)
