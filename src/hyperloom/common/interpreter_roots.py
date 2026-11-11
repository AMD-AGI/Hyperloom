# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which directories another venv's interpreter needs, for whatever is binding them."""

from __future__ import annotations

from pathlib import Path

from hyperloom_common.bwrap import SYSTEM_ROOTS


def within(path: Path, parent: Path) -> bool:
    """Whether ``path`` is ``parent`` or sits under it."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


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
    if within(support, prefix) or any(within(support, Path(root)) for root in SYSTEM_ROOTS):
        return (prefix,)
    resolved = support.resolve()
    return (prefix, support) if resolved == support else (prefix, support, resolved)
