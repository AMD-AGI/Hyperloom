# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Start-up hook of a test's server: every process notes where it imported each package the test rebuilt from.

A farm (``hyperloom.measure.farm``) puts the rebuilt packages first on the server's ``PYTHONPATH``, but a process can
still import the image's copy: a worker inherits its parent's ``sys.path``, and anything that put a directory holding the
image's package ahead of the farm in the parent (aiter puts its meta directory first) sends every worker to the image. The
patch is then inert in the processes that run the model, and the test measures the image.

The farm copies this file as ``sitecustomize.py`` into its ``site`` directory, beside a ``WATCH_FILE`` naming the packages
and the directory the notes go to. In every Python process started with that ``PYTHONPATH``, the first time one of the
packages is executed the hook writes ``<pid>-<package>.json`` with the file it was loaded from. It changes nothing the
import does; what it cannot do it leaves undone, and it never stops a process from starting.

The file is standalone (standard library only) and runs in every Python process of the server.
"""

from __future__ import annotations

import json
import os
import sys

WATCH_FILE = "hyperloom-import-watch.json"
PACKAGES_KEY = "packages"
NOTES_KEY = "notes"
PACKAGE_FIELD = "package"
ORIGIN_FIELD = "origin"
_HERE = os.path.dirname(os.path.realpath(__file__))


def note_import(notes: str, package: str, origin: object) -> None:
    """Write the note of ``package`` for this process; one that exists or cannot be written is left as it is."""
    try:
        os.makedirs(notes, exist_ok=True)
        handle = open(os.path.join(notes, f"{os.getpid()}-{package}.json"), "x", encoding="utf-8")
    except OSError:
        return
    with handle:
        json.dump({"pid": os.getpid(), PACKAGE_FIELD: package, ORIGIN_FIELD: origin}, handle)


class OriginFinder:
    """Finds a watched package as the other finders do, and notes the file it is executed from."""

    def __init__(self, packages: frozenset[str], notes: str) -> None:
        self._packages = packages
        self._notes = notes

    def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
        if fullname not in self._packages:
            return None
        spec = next((found for found in self._others(fullname, path, target) if found is not None), None)
        execute = getattr(getattr(spec, "loader", None), "exec_module", None)
        if execute is None:
            return spec
        notes = self._notes

        def exec_module(module):  # type: ignore[no-untyped-def]
            note_import(notes, fullname, getattr(module, "__file__", None))
            execute(module)

        try:
            spec.loader.exec_module = exec_module
        except (AttributeError, TypeError):
            note_import(notes, fullname, spec.origin)
        return spec

    def _others(self, fullname, path, target):  # type: ignore[no-untyped-def]
        for finder in list(sys.meta_path):
            find = getattr(finder, "find_spec", None)
            if finder is not self and find is not None:
                yield find(fullname, path, target)


def _chain() -> None:
    """Run the ``sitecustomize`` this file shadows: the first one on ``sys.path`` after the directory of this one."""
    folders = [os.path.realpath(entry) for entry in sys.path if entry]
    after = folders[folders.index(_HERE) + 1 :] if _HERE in folders else folders
    for folder in after:
        candidate = os.path.join(folder, "sitecustomize.py")
        if os.path.isfile(candidate):
            with open(candidate, "rb") as handle:
                code = compile(handle.read(), candidate, "exec")
            exec(code, {"__name__": "sitecustomize", "__file__": candidate})
            return


def _install() -> None:
    try:
        with open(os.path.join(_HERE, WATCH_FILE), encoding="utf-8") as handle:
            watch = json.load(handle)
        packages, notes = frozenset(watch[PACKAGES_KEY]), str(watch[NOTES_KEY])
    except (OSError, ValueError, KeyError, TypeError):
        return
    for package in packages & set(sys.modules):
        note_import(notes, package, getattr(sys.modules[package], "__file__", None))
    sys.meta_path.insert(0, OriginFinder(packages, notes))


if __name__ == "sitecustomize":
    _install()
    _chain()
