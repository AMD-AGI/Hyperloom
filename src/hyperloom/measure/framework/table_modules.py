# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which of aiter's prebuilt modules a test's tuned tables make stale.

aiter generates the kernels of some GEMM modules from a tuned table and the module holds only the kernels the table
names: their ``blob_gen_cmd`` in ``optCompilerConfig.json`` passes the generator ``{AITER_CONFIGS.<VARIABLE>_FILE}``,
the table the variable resolves to, or a file below ``{AITER_ROOT_DIR}/aiter/configs``. The image's prebuilt module was
generated from the image's tables and aiter builds a module only when its ``.so`` is missing, so a test whose table names
other kernels must build its own: with the image's, its server stops on a kernel that is not in the compiled registry.

``AITER_CONFIG.get_config_file`` (``aiter/jit/core.py``) resolves a variable to its value when it is set (the tables it
lists, merged when there are several), else to the variable's default table below ``aiter/configs`` with the per-model
tables of ``aiter/configs/model_configs`` whose name holds the variable's tuned name. A table reaches a test through the
variable (an ``AITER_CONFIG_*`` of the sheet, or the table a kernel tune made), or through a patch to ``configs``; the
test's tables and the image's are compared by the bytes of the files each resolves to. What cannot be resolved counts as
different, and a module whose command names the tables in a way not read here is stale whenever any table differs: a
module that is wrongly rebuilt only costs time, one that is wrongly kept is never measured.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.measure.framework.rows import DEFAULT_TABLE_PLACEHOLDER
from hyperloom.measure.framework.tables import AITER_VARIABLE_PREFIX, TableVariable, read_variables

__all__ = ["TableUse", "file_identity", "named_table_modules", "stale_table_modules", "table_modules"]

_TABLE_FILE = re.compile(r"\{AITER_CONFIGS\.(AITER_CONFIG_[A-Za-z0-9_]+)_FILE\}")
_CONFIGS_FILE = re.compile(r"\{AITER_ROOT_DIR\}/aiter/(configs/[^\s:{}'\"]+)")
_TABLES_OBJECT = "AITER_CONFIGS"
_CONFIGS_DIR, _MODEL_CONFIGS = "configs", "configs/model_configs"
_UNTUNED = "untuned"
_DIGEST_CHUNK = 1 << 20
_COMMAND = "blob_gen_cmd"


def file_identity(path: str | Path) -> str:
    """The content of the file at ``path`` as a digest, or what says it could not be read."""
    try:
        with open(path, "rb") as handle:
            digest = hashlib.sha256()
            while chunk := handle.read(_DIGEST_CHUNK):
                digest.update(chunk)
    except OSError:
        return f"unreadable:{path}"
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class TableUse:
    """The tables one module is generated from: ``variables`` it resolves, ``files`` (patterns below the package) it
    names directly, and whether its command names tables in a way not read here (``unread``)."""

    variables: frozenset[str] = frozenset()
    files: tuple[str, ...] = ()
    unread: bool = False


def _use(entry: Any) -> TableUse | None:
    """What ``entry`` of the config generates its module from, ``None`` when no table."""
    if not isinstance(entry, Mapping):
        return TableUse(unread=True)
    raw = entry.get(_COMMAND, "")
    commands = raw if isinstance(raw, list) else [raw]
    if any(not isinstance(command, str) for command in commands):
        return TableUse(unread=True)
    text = " ".join(commands)
    variables = frozenset(_TABLE_FILE.findall(text))
    files = tuple(_CONFIGS_FILE.findall(text))
    unread = _TABLES_OBJECT in _TABLE_FILE.sub("", text)
    return TableUse(variables, files, unread) if variables or files or unread else None


def table_modules(config: Mapping[str, Any]) -> dict[str, TableUse]:
    """The entries of the module config that are generated from a tuned table, with the tables."""
    return {key: use for key, entry in config.items() if (use := _use(entry)) is not None}


class _Tables:
    """The tables one aiter package (``package``) resolves under ``environment``; ``image`` resolves the image's default
    table that a rows-only value names. Every file is read once."""

    def __init__(self, package: Path, environment: Mapping[str, str], image: _Tables | None = None) -> None:
        self.package = package
        self._environment = environment
        self._image = image
        self._variables: dict[str, TableVariable] | None = None
        self._resolved: dict[str, tuple[str, ...]] = {}

    def _known(self) -> dict[str, TableVariable]:
        if self._variables is None:
            self._variables = read_variables(self.package / "jit" / "core.py")
        return self._variables

    def defaults(self, name: str) -> list[str] | None:
        """The files aiter reads for ``name`` when it is unset, ``None`` when the package does not say."""
        variable = self._known().get(name)
        if variable is None:
            return None
        models = sorted(
            str(path)
            for path in (self.package / _MODEL_CONFIGS).glob(f"*{variable.tuned}*.csv")
            if path.is_file() and _UNTUNED not in path.name
        )
        return [str(self.package / _CONFIGS_DIR / variable.default), *models]

    def value(self, name: str) -> str:
        return (self._environment.get(name) or "").strip()

    def variable(self, name: str) -> tuple[str, ...]:
        """What ``name`` resolves to, as the content of its files."""
        if name not in self._resolved:
            self._resolved[name] = self._resolve(name)
        return self._resolved[name]

    def _resolve(self, name: str) -> tuple[str, ...]:
        value = self.value(name)
        if not value:
            listed = self.defaults(name)
            return self.tree() if listed is None else _contents(listed)
        listed = []
        for part in value.split(os.pathsep):
            image_default = (
                None if part != DEFAULT_TABLE_PLACEHOLDER or self._image is None else self._image.defaults(name)
            )
            listed.extend(image_default if image_default is not None else [part])
        return _contents(listed)

    def files(self, patterns: Iterable[str]) -> tuple[str, ...]:
        """The files below the package that ``patterns`` match, by name and content."""
        found = []
        for pattern in patterns:
            directory = self.package / os.path.dirname(pattern)
            try:
                names = sorted(path.name for path in directory.iterdir() if path.is_file())
            except OSError:
                names = []
            matched = fnmatch.filter(names, os.path.basename(pattern))
            found.extend(f"{pattern}={name}:{file_identity(directory / name)}" for name in matched)
            if not matched:
                found.append(f"{pattern}=none")
        return tuple(found)

    def tree(self) -> tuple[str, ...]:
        """Every table below ``configs``: what stands for a variable the package does not say how it resolves."""
        root = self.package / _CONFIGS_DIR
        files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
        return ("tree", *(f"{path.relative_to(root).as_posix()}:{file_identity(path)}" for path in files))


def _contents(listed: list[str]) -> tuple[str, ...]:
    """What aiter reads from the files ``listed``: one file as it is, several merged with the missing ones left out."""
    if len(listed) == 1:
        return (file_identity(listed[0]),)
    return tuple(file_identity(path) for path in listed if os.path.isfile(path))


def _differs(use: TableUse, image: _Tables, test: _Tables) -> bool:
    return any(test.variable(name) != image.variable(name) for name in use.variables) or (
        bool(use.files) and test.files(use.files) != image.files(use.files)
    )


def _anything_differs(image: _Tables, test: _Tables, named: Collection[str]) -> bool:
    """Whether any table the test's aiter can read differs from the image's: a variable it sets, or the tables below
    ``configs``."""
    return any(test.variable(name) != image.variable(name) for name in named) or (
        test.package != image.package and test.tree() != image.tree()
    )


def stale_table_modules(
    config: Mapping[str, Any], image: Path, package: Path, environments: Iterable[Mapping[str, str]]
) -> frozenset[str]:
    """The entries of ``config`` whose tuned tables, as the aiter package at ``package`` resolves them under any of
    ``environments`` (a test's servers), differ from the tables the image's package (``image``) resolves unset."""
    uses = table_modules(config)
    if not uses:
        return frozenset()
    image_tables = _Tables(image, {})
    stale: set[str] = set()
    for environment in environments:
        test = _Tables(package, environment, image_tables)
        named = sorted(name for name in environment if name.startswith(AITER_VARIABLE_PREFIX) and test.value(name))
        if test.package == image_tables.package and not named:
            continue
        anything = None
        for key, use in uses.items():
            if key in stale:
                continue
            if use.unread:
                anything = _anything_differs(image_tables, test, named) if anything is None else anything
                differs = anything
            else:
                differs = _differs(use, image_tables, test)
            if differs:
                stale.add(key)
    return frozenset(stale)


def _reads(use: TableUse, path: str, known: Mapping[str, TableVariable]) -> bool:
    """Whether the module of ``use`` can read the file at ``path`` (below the package) as one of its tables."""
    if use.unread or any(fnmatch.fnmatchcase(path, pattern) for pattern in use.files):
        return True
    for name in use.variables:
        variable = known.get(name)
        if variable is None or path == f"{_CONFIGS_DIR}/{variable.default}":
            return True
        model = path.removeprefix(f"{_MODEL_CONFIGS}/")
        if model != path and fnmatch.fnmatchcase(model, f"*{variable.tuned}*.csv") and _UNTUNED not in model:
            return True
    return False


def named_table_modules(
    config: Mapping[str, Any], package: Path, variables: Collection[str], paths: Collection[str]
) -> frozenset[str]:
    """The entries of ``config`` whose tables a test may change, before it runs: the ones that resolve one of
    ``variables`` (the ``AITER_CONFIG_*`` the test sets), or read one of ``paths`` (the files below the aiter package at
    ``package`` that it patches). What the test's tables hold is not known yet, so every table it names counts."""
    known = read_variables(package / "jit" / "core.py")
    configs = [path for path in paths if path.startswith(f"{_CONFIGS_DIR}/")]
    return frozenset(
        key
        for key, use in table_modules(config).items()
        if (use.variables & set(variables)) or (use.unread and variables) or any(_reads(use, p, known) for p in configs)
    )
