# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Start-up hook of a local test's server: aiter writes its merged tuned tables below the test, not below ``/tmp``.

When an ``AITER_CONFIG_*`` variable lists several tables, aiter's ``AITER_CONFIG.update_config_files``
(``aiter/jit/core.py``) writes their merged view to ``Path("/tmp/aiter_configs/")``, a literal in the function body that
no variable and no ``TMPDIR`` changes. Local sandboxes share ``/tmp``, so two tests that give one variable different
tables overwrite each other's merged file between the write and the read.

The toolbox copies this file as ``sitecustomize.py`` into a directory of the test's own, puts that directory first on the
server's ``PYTHONPATH`` and names the test's merge directory in ``HYPERLOOM_AITER_MERGE_DIR``. Once ``aiter.jit.core``
has been imported, the hook replaces that one string constant in the function's code object, in this process only; the
installed aiter is never written. What the hook cannot do it leaves undone and says so in a note below the directory it
lives in; it never stops a process from starting or aiter from importing.

The file is standalone (standard library only) and runs in every Python process of the server, so it imports nothing
it does not need to start.
"""

from __future__ import annotations

import os
import sys
import time

MERGE_DIR_ENV = "HYPERLOOM_AITER_MERGE_DIR"
STARTED_NOTE = "hook.json"
CHAIN_FAILED_NOTE = "chain-failed.json"
NOTES_DIR = "seen"
_MODULE = "aiter.jit.core"
_HARD_CODED = "/tmp/aiter_configs"
_HERE = os.path.dirname(os.path.realpath(__file__))


def _write_note(path: str, **fields: object) -> None:
    """Create ``path`` with ``fields``; a note that exists or cannot be written is left as it is."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handle = open(path, "x", encoding="utf-8")
    except OSError:
        return
    import json

    with handle:
        json.dump({"pid": os.getpid(), **fields}, handle)


def redirect_merge_directory(core: object, directory: str) -> str | None:
    """Point ``core``'s merged-view directory at ``directory``; ``None`` when done, else why it could not be."""
    function = getattr(getattr(core, "AITER_CONFIG", None), "update_config_files", None)
    code = getattr(function, "__code__", None)
    if code is None:
        return "aiter.jit.core has no AITER_CONFIG.update_config_files"
    places = [
        i for i, value in enumerate(code.co_consts) if isinstance(value, str) and value.rstrip("/") == _HARD_CODED
    ]
    if len(places) != 1:
        return f"update_config_files names {len(places)} places for the merged view, expected one"
    constants = list(code.co_consts)
    constants[places[0]] = directory
    try:
        function.__code__ = code.replace(co_consts=tuple(constants))
    except (TypeError, ValueError) as exc:
        return f"update_config_files cannot be rebuilt: {type(exc).__name__}: {exc}"
    return None


def _note_aiter(directory: str, reason: str | None, origin: object = None) -> None:
    _write_note(
        os.path.join(_HERE, NOTES_DIR, f"{os.getpid()}-{time.time_ns()}.json"),
        module=_MODULE,
        origin=origin,
        merge_dir=directory,
        patched=reason is None,
        reason=reason,
    )


def _redirected(module: object, directory: str) -> None:
    try:
        os.makedirs(directory, exist_ok=True)
        reason = redirect_merge_directory(module, directory)
    except OSError as exc:
        reason = f"the merge directory cannot be made: {type(exc).__name__}: {exc}"
    _note_aiter(directory, reason, getattr(module, "__file__", None))


class _AiterCoreFinder:
    """Lets ``aiter.jit.core`` load as it always does, then redirects its merge directory once it has."""

    def __init__(self, directory: str) -> None:
        self._directory = directory

    def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
        if fullname != _MODULE:
            return None
        spec = None
        for finder in sys.meta_path:
            find = getattr(finder, "find_spec", None)
            if finder is not self and find is not None:
                spec = find(fullname, path, target)
                if spec is not None:
                    break
        loader = getattr(spec, "loader", None)
        execute = getattr(loader, "exec_module", None)
        directory = self._directory
        if spec is not None and execute is None:
            _note_aiter(directory, "the loader of aiter.jit.core has no exec_module")
            return spec
        if spec is None:
            return None

        def exec_module(module):  # type: ignore[no-untyped-def]
            execute(module)
            if self in sys.meta_path:
                sys.meta_path.remove(self)
            _redirected(module, directory)

        try:
            loader.exec_module = exec_module
        except (AttributeError, TypeError) as exc:
            _note_aiter(directory, f"the loader of aiter.jit.core cannot be wrapped: {type(exc).__name__}: {exc}")
        return spec


def _chain() -> None:
    """Run the ``sitecustomize`` this file shadows (the distribution's, for one), as Python would have run it.

    Its failure is noted and raised again: ``site`` reports a failing ``sitecustomize`` and goes on, as it would for the
    original.
    """
    for entry in sys.path:
        folder = os.path.realpath(entry) if entry else ""
        candidate = os.path.join(folder, "sitecustomize.py")
        if folder and folder != _HERE and os.path.isfile(candidate):
            try:
                with open(candidate, "rb") as handle:
                    exec(
                        compile(handle.read(), candidate, "exec"), {"__name__": "sitecustomize", "__file__": candidate}
                    )
            except Exception as exc:
                _write_note(os.path.join(_HERE, CHAIN_FAILED_NOTE), reason=f"{type(exc).__name__}: {exc}")
                raise
            return


def _install() -> None:
    directory = os.environ.get(MERGE_DIR_ENV, "")
    if os.path.isabs(directory):
        _write_note(os.path.join(_HERE, STARTED_NOTE), merge_dir=directory)
        if _MODULE in sys.modules:
            _redirected(sys.modules[_MODULE], directory)
        else:
            sys.meta_path.insert(0, _AiterCoreFinder(directory))
    _chain()


if __name__ == "sitecustomize":
    _install()
