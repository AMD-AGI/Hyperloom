# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which of aiter's prebuilt modules a patch to its sources makes stale.

An image ships aiter's compiled modules (``aiter/jit/*.so``) and aiter builds a module only when its ``.so`` is missing.
A patch to ``csrc`` or ``hsa`` changes what a module would build to, so the test's own JIT directory must not link the
image's copy of a module the patch touches: the server then builds it on its first call, from the patched sources, unless a CPU
build made it before the test (:mod:`hyperloom.measure.runner.native.aiter_modules`).

``optCompilerConfig.json`` says what each module is built from. Its values are Python expressions that aiter evaluates
(``f'{AITER_CSRC_DIR}/pybind/x.cu'``); the ones that name a path are expanded here, and an entry that cannot be
expanded is read as touched, because a module that is wrongly rebuilt only costs time, and one that is wrongly kept
measures the old code. A module is touched when a patched path is

* one of its sources, or below a source directory;
* in the closure of the quoted or angled ``#include`` of its sources, searched where the build searches (the including
  file's directory, the module's ``extra_include`` directories and ``csrc/include``);
* its code generator script, or below the script's directory;
* below ``hsa`` while its generator is ``hsa/codegen.py``: the whole module list for ``codegen.py`` itself or a path
  that names no module, else the modules whose ``-m <name>`` the path's ``hsa/<arch>/<name>/`` directory names;
* in a module whose entry in the config differs from the image's.

A module generated from a tuned table that differs from the image's is touched too (``framework.table_modules``).

A prebuilt name that is not an entry of the config is a module aiter names at run time (``mha_varlen_fwd_<variant>``
from ``module_mha_varlen_fwd``, ``module_moe_ck2stages_<variant>``); it follows the entry it starts with, and one that
follows none is read as touched whenever anything is.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.outside_facts import (
    AITER_META_DIR_ENV,
    AITER_META_MARKER_DIR,
)
from hyperloom.measure.jit.key import env_names_in

__all__ = [
    "CONFIG_FILE",
    "ESTIMATED_MODULE_SECONDS",
    "MAX_REBUILD_SECONDS",
    "SourceTree",
    "Touched",
    "build_env_names",
    "config_path",
    "prebuilt_names",
    "read_config",
    "touched_entries",
    "touched_modules",
]

CONFIG_FILE = "optCompilerConfig.json"
#: The directory of the aiter package that holds its builder, its module config and its prebuilt modules.
_JIT_DIR = "jit"
#: What one module takes to build: the 63 to 66 s that the heaviest modules (mha_batch_prefill, module_moe_ck2stages) took in
#: earlier tests' server logs. The single-lineage flow gives aiter 40 minutes to build in (``targeted_build``); a patch that
#: would take longer than that to rebuild before its test starts is refused when the test is accepted.
ESTIMATED_MODULE_SECONDS = 65
MAX_REBUILD_SECONDS = 40 * 60
_PLACEHOLDERS = {
    "AITER_CSRC_DIR": AITER_META_MARKER_DIR,
    "AITER_GRADLIB_DIR": "gradlib",
    AITER_META_DIR_ENV: "",
    "AITER_ROOT_DIR": "",
    "CK_DIR": "3rdparty/composable_kernel",
}
_DEFAULT_INCLUDES = ("csrc/include", "csrc/include/torch")
_STRING = re.compile(r"""f?(?:'([^']*)'|"([^"]*)")""")
_PLACEHOLDER = re.compile(r"\{([A-Za-z_]+)\}")
_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]+[<"]([^>"]+)[>"]', re.MULTILINE)
_CODEGEN = "hsa/codegen.py"
#: The code below the meta directory that builds a module whose own code cannot be told: every generator there is.
_ALL_GENERATORS = (AITER_META_MARKER_DIR, "hsa")
_MODULE_OPTION = re.compile(r"(?:^|\s)-m\s+(\S+)")
_MODULE_PREFIX = "module_"
_MAX_SOURCE_BYTES = 4 * 1024**2


@dataclass(frozen=True)
class Touched:
    """The modules a patch touches: ``names`` are the config entries and the prebuilt names to keep out of the JIT
    directory, ``rebuilt`` the prebuilt ones among them, which the server builds again when it calls them."""

    names: frozenset[str] = frozenset()
    rebuilt: frozenset[str] = frozenset()

    def __or__(self, other: Touched) -> Touched:
        return Touched(self.names | other.names, self.rebuilt | other.rebuilt)


@dataclass(frozen=True)
class SourceTree:
    """The files below aiter's meta directory as a test sees them: the directory ``root``, with ``overlay`` over it (a
    path's new bytes, or ``None`` for a path the patch deletes)."""

    root: Path
    overlay: Mapping[str, bytes | None] = field(default_factory=dict)

    def read(self, relative: str) -> str | None:
        """The text of the file, ``None`` when there is none (or it is not text we should scan)."""
        if relative in self.overlay:
            data = self.overlay[relative]
        else:
            try:
                with open(self.root / relative, "rb") as handle:
                    data = handle.read(_MAX_SOURCE_BYTES)
            except OSError:
                return None
        return None if data is None else data.decode("utf-8", errors="replace")

    def files_under(self, directory: str) -> list[str]:
        """The files below ``directory``, the overlay's included and its deletions left out, in a stable order."""
        found = set()
        for current, _, names in os.walk(self.root / directory):
            found.update(Path(current, name).relative_to(self.root).as_posix() for name in names)
        found.update(path for path in self.overlay if path.startswith(f"{directory}/"))
        return sorted(path for path in found if self.overlay.get(path, b"") is not None)


def config_path(package: Path) -> Path:
    """Where the aiter package at ``package`` keeps its module config."""
    return package / _JIT_DIR / CONFIG_FILE


def read_config(path: Path | bytes) -> dict[str, Any]:
    """The module config: the file at ``path``, or the bytes of one; ``OSError`` or ``ValueError`` when it is not one."""
    data = path if isinstance(path, bytes) else path.read_bytes()
    document = json.loads(data)
    if not isinstance(document, dict):
        raise ValueError("the module config is not a mapping of module name to entry")
    return document


def prebuilt_names(package: Path) -> frozenset[str]:
    """The modules the aiter package at ``package`` ships built (the ``.so`` files of its ``jit`` directory)."""
    try:
        return frozenset(path.stem for path in (package / _JIT_DIR).iterdir() if path.suffix == ".so")
    except OSError:
        return frozenset()


class _Unexpandable(Exception):
    pass


def _literal(expression: Any) -> str:
    """The text of an entry written as a (formatted) string literal; anything else is read at run time and is unreadable here."""
    match = _STRING.fullmatch(expression) if isinstance(expression, str) else None
    if match is None:
        raise _Unexpandable
    return match.group(1) if match.group(1) is not None else match.group(2)


def _path_of(text: str) -> str:
    """The path below the meta directory that ``text`` names once the directories aiter defines are put in."""
    if any(name not in _PLACEHOLDERS for name in _PLACEHOLDER.findall(text)):
        raise _Unexpandable
    expanded = _PLACEHOLDER.sub(lambda found: _PLACEHOLDERS[found.group(1)], text)
    if "{" in expanded or "}" in expanded:
        raise _Unexpandable
    return posixpath.normpath(expanded.strip("/")) if expanded.strip("/") else ""


def _script_of(command: Any) -> tuple[str, list[str]]:
    """The generator script a ``blob_gen_cmd`` runs and, for ``hsa/codegen.py``, the modules ``-m`` names."""
    text = _literal(command)
    words = text.split(maxsplit=1)
    script = _path_of(words[0]) if words else ""
    return script, _MODULE_OPTION.findall(text) if script == _CODEGEN else []


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


@dataclass(frozen=True)
class _Inputs:
    """What one module is built from, as paths below the meta directory."""

    sources: tuple[str, ...]
    includes: tuple[str, ...]
    scripts: tuple[str, ...]
    hsa_modules: frozenset[str]


def _inputs(entry: Any) -> _Inputs:
    if not isinstance(entry, Mapping):
        raise _Unexpandable
    sources = tuple(_path_of(_literal(source)) for source in _list(entry.get("srcs", [])))
    includes = tuple(_path_of(_literal(path)) for path in _list(entry.get("extra_include", []))) + _DEFAULT_INCLUDES
    commands = [command for command in _list(entry.get("blob_gen_cmd", "")) if command not in ("", "''", '""')]
    generators = [_script_of(command) for command in commands]
    return _Inputs(
        sources,
        includes,
        tuple(script for script, _ in generators),
        frozenset(name for _, modules in generators for name in modules),
    )


class _Closure:
    """The files a module's sources reach by ``#include``, found in the tree; each file is read once."""

    def __init__(self, tree: SourceTree, patched: Collection[str]) -> None:
        self._tree = tree
        self._patched = frozenset(patched)
        self._raw: dict[str, tuple[str, ...]] = {}

    def _includes_of(self, relative: str) -> tuple[str, ...]:
        if relative not in self._raw:
            text = self._tree.read(relative)
            self._raw[relative] = () if text is None else tuple(_INCLUDE.findall(text))
        return self._raw[relative]

    def of(self, inputs: _Inputs) -> set[str]:
        pending = [path for source in inputs.sources for path in self._files(source)]
        seen = set(pending)
        while pending:
            current = pending.pop()
            for target in self._includes_of(current):
                for directory in (posixpath.dirname(current), *inputs.includes):
                    found = posixpath.normpath(posixpath.join(directory, target))
                    if found not in seen and found.startswith("csrc/") and self._exists(found):
                        seen.add(found)
                        pending.append(found)
                        break
        return seen

    def _exists(self, relative: str) -> bool:
        """A file the patch deletes still counts: the sources that include it were built with it."""
        return relative in self._patched or self._tree.read(relative) is not None

    def _files(self, source: str) -> list[str]:
        if source.startswith("3rdparty/") or self._tree.read(source) is not None:
            return [source]
        return self._tree.files_under(source) or [source]


def _under(path: str, directory: str) -> bool:
    return bool(directory) and (path == directory or path.startswith(f"{directory}/"))


def _hsa_touched(inputs: _Inputs, patched: Collection[str], known: frozenset[str]) -> bool:
    if not inputs.hsa_modules:
        return False
    for path in (path for path in patched if _under(path, "hsa")):
        parts = path.split("/")
        named = parts[2] if len(parts) >= 4 and path != _CODEGEN else None
        if named not in known or named in inputs.hsa_modules:
            return True
    return False


def _entry_touched(
    entry: Mapping[str, Any], closure: _Closure, patched: Collection[str], known_hsa: frozenset[str]
) -> bool:
    try:
        inputs = _inputs(entry)
    except _Unexpandable:
        return True
    reached = closure.of(inputs) if any(_under(path, AITER_META_MARKER_DIR) for path in patched) else set()
    scripts = [(script, posixpath.dirname(script)) for script in inputs.scripts if script != _CODEGEN]
    return (
        any(_under(path, source) for path in patched for source in inputs.sources)
        or any(path in reached for path in patched)
        or any(path == script or _under(path, directory) for path in patched for script, directory in scripts)
        or _hsa_touched(inputs, patched, known_hsa)
    )


def _hsa_names(config: Mapping[str, Any]) -> frozenset[str]:
    names: set[str] = set()
    for entry in config.values():
        try:
            names |= _inputs(entry).hsa_modules
        except _Unexpandable:
            continue
    return frozenset(names)


def _entry_of(name: str, config: Mapping[str, Any]) -> str | None:
    """The config entry a prebuilt name belongs to: itself, or the entry with the longest key (or key without
    ``module_``) that the name starts with."""
    if name in config:
        return name
    matches = [
        (len(prefix), key)
        for key in config
        for prefix in (key, key.removeprefix(_MODULE_PREFIX))
        if name.startswith(f"{prefix}_")
    ]
    return max(matches)[1] if matches else None


def touched_modules(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    tree: SourceTree,
    patched: Iterable[str],
    prebuilt: Collection[str],
) -> Touched:
    """The modules a patch to the meta directory (``patched`` paths, below it) and to the module config (``before`` the
    image's, ``after`` the test's) touches, with ``tree`` the sources the test builds from."""
    paths = sorted(set(patched))
    closure, known_hsa = _Closure(tree, paths), _hsa_names(after)
    keys = {
        key
        for key, entry in after.items()
        if entry != before.get(key) or (paths and _entry_touched(entry, closure, paths, known_hsa))
    }
    return touched_entries(keys, after, prebuilt)


def touched_entries(keys: Collection[str], config: Mapping[str, Any], prebuilt: Collection[str]) -> Touched:
    """The entries ``keys`` of ``config`` as touched modules, with the prebuilt names that follow them: what a module
    generated from a tuned table that differs from the image's is
    (:func:`~hyperloom.measure.framework.table_modules.stale_table_modules`)."""
    names = set(keys)
    for name in prebuilt:
        owner = _entry_of(name, config)
        if (owner in keys) or (owner is None and keys):
            names.add(name)
    return Touched(frozenset(names), frozenset(names & set(prebuilt)))


def build_env_names(package: Path, meta: Path, config: Mapping[str, Any], modules: Collection[str]) -> frozenset[str]:
    """The environment variables the build of ``modules`` (entries of ``config``) reads, from the aiter package at
    ``package`` and the meta directory at ``meta``.

    That is aiter's builder, the package's ``jit`` directory (``core.py`` and the helpers it imports), and each module's
    own Python: a source that is a Python file or a directory, and the directory of each code generator script (a
    generator imports what is beside it; ``hsa/codegen.py``'s is ``hsa``). A module that is no entry of ``config``, or
    whose entry cannot be expanded, reads every generator of ``csrc`` and ``hsa``. The rest of the tree (``setup.py``, the
    Triton kernels below ``aiter/ops``) is not the build's and is not read: a variable only it reads cannot change the
    module. Both sides of a CPU build of aiter's modules compare the variables this names, so they compare the same ones.
    """
    places: set[str] = set()
    for name in modules:
        try:
            inputs = _inputs(config[name])
        except (KeyError, _Unexpandable):
            places.update(_ALL_GENERATORS)
            continue
        places.update(inputs.sources)
        places.update(posixpath.dirname(script) or script for script in inputs.scripts)
    return env_names_in([package / _JIT_DIR, *(meta / place for place in sorted(places) if place)])
