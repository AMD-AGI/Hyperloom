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
test's tables and the image's are compared by the bytes of the files each resolves to (several in the order aiter
merges them only where that order changes what a generator makes, see ``_contents``). What cannot be resolved counts as
different, and a module whose command names the tables in a way not read here is stale whenever any table differs: a
module that is wrongly rebuilt only costs time, one that is wrongly kept is never measured.
"""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import math
import os
import re
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.outside_facts import TUNED_LIBTYPE_COLUMN
from hyperloom.measure.framework.rows import DEFAULT_TABLE_PLACEHOLDER
from hyperloom.measure.framework.tables import AITER_VARIABLE_PREFIX, TableVariable, read_variables

__all__ = ["TableUse", "default_tables", "file_identity", "named_table_modules", "stale_table_modules", "table_modules"]

_TABLE_FILE = re.compile(r"\{AITER_CONFIGS\.(AITER_CONFIG_[A-Za-z0-9_]+)_FILE\}")
_CONFIGS_FILE = re.compile(r"\{AITER_ROOT_DIR\}/aiter/(configs/[^\s:{}'\"]+)")
_TABLES_OBJECT = "AITER_CONFIGS"
_CONFIGS_DIR, _MODEL_CONFIGS = "configs", "configs/model_configs"
_UNTUNED = "untuned"
_DIGEST_CHUNK = 1 << 20
_COMMAND = "blob_gen_cmd"
_GENERATOR = re.compile(r"\{AITER_CSRC_DIR\}/([^\s:{}'\"]+\.py)")
_SHAPE = ("gfx", "cu_num", "M", "N", "K")
_KIND_SHAPE = (TUNED_LIBTYPE_COLUMN, *_SHAPE)
_BATCHED_SHAPE = ("gfx", "cu_num", "B", "M", "N", "K")
#: The columns each generator of aiter e7d2453f25e5 (its path below ``csrc``) keys the rows of its table by: a row replaces
#: an earlier one of its key (``chip_info.build_tune_dict``, ``build_tune_dict_batched``), and a generator that keeps the
#: rows of one ``libtype`` keys within it. ``None`` for the one that compiles the kernels of every row (opus: their union).
#: A generator not listed keys by nothing, so that any row may replace any earlier one.
_GENERATOR_KEYS: dict[str, tuple[str, ...] | None] = {
    "ck_gemm_a8w8/gen_instances.py": _SHAPE,
    "ck_gemm_a4w4_blockscale/gen_instances.py": _SHAPE,
    "ck_gemm_a8w8_bpreshuffle/gen_instances.py": _KIND_SHAPE,
    "cktile_gemm_a8w8_bpreshuffle/gen_instances.py": _KIND_SHAPE,
    "ck_gemm_a8w8_blockscale/gen_instances.py": _KIND_SHAPE,
    "ck_gemm_a8w8_blockscale/gen_instances_cktile.py": _KIND_SHAPE,
    "ck_gemm_a8w8_blockscale_bpreshuffle/gen_instances.py": _KIND_SHAPE,
    "ck_batched_gemm_a8w8/gen_instances.py": _BATCHED_SHAPE,
    "ck_batched_gemm_bf16/gen_instances.py": _BATCHED_SHAPE,
    "opus_gemm/gen_instances.py": None,
}
#: The keys of a variable whose generators are not known: one key that every row has.
_ANY_ROW: frozenset[tuple[str, ...]] = frozenset({()})


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
    names directly, whether its command names tables in a way not read here (``unread``), and the ``generators`` it runs
    (their paths below ``csrc``)."""

    variables: frozenset[str] = frozenset()
    files: tuple[str, ...] = ()
    unread: bool = False
    generators: tuple[str, ...] = ()


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
    generators = tuple(_GENERATOR.findall(text))
    return TableUse(variables, files, unread, generators) if variables or files or unread else None


def _per_model(package: Path, variable: TableVariable) -> Iterator[str]:
    """The per-model tables of ``variable`` in the aiter package at ``package``, in the order their directory lists them:
    the order ``AITER_CONFIG.get_config_file`` (``Path.glob``) merges them in."""
    for path in (package / _MODEL_CONFIGS).glob(f"*{variable.tuned}*.csv"):
        if path.is_file() and _UNTUNED not in path.name:
            yield str(path)


def default_tables(package: Path, variable: TableVariable) -> list[str]:
    """The files the aiter package at ``package`` reads for ``variable`` when it is unset: its default table, then the
    per-model tables whose name holds its tuned name, by name."""
    return [str(package / _CONFIGS_DIR / variable.default), *sorted(_per_model(package, variable))]


def table_modules(config: Mapping[str, Any]) -> dict[str, TableUse]:
    """The entries of the module config that are generated from a tuned table, with the tables."""
    return {key: use for key, entry in config.items() if (use := _use(entry)) is not None}


def _generator_keys(use: TableUse) -> set[tuple[str, ...]]:
    if not use.generators:
        return set(_ANY_ROW)
    keys = (_GENERATOR_KEYS.get(generator, ()) for generator in use.generators)
    return {key for key in keys if key is not None}


def _row_keys(uses: Iterable[TableUse]) -> dict[str, frozenset[tuple[str, ...]]]:
    """For each variable a module is generated from, the keys its generators key a row by; none at all when a module
    names tables in a way not read here, as it may read any variable by any key."""
    found: dict[str, set[tuple[str, ...]]] = {}
    for use in uses:
        if use.unread:
            return {}
        keys = _generator_keys(use)
        for name in use.variables:
            found.setdefault(name, set()).update(keys)
    return {name: frozenset(keys) for name, keys in found.items()}


class _Tables:
    """The tables one aiter package (``package``) resolves under ``environment``; ``image`` resolves the image's default
    table that a rows-only value names. Every variable is resolved once."""

    def __init__(
        self,
        package: Path,
        environment: Mapping[str, str],
        keys: Mapping[str, frozenset[tuple[str, ...]]],
        image: _Tables | None = None,
    ) -> None:
        self.package = package
        self._environment = environment
        self._keys = keys
        self._image = image
        self._variables: dict[str, TableVariable] | None = None
        self._resolved: dict[str, tuple[str, ...]] = {}

    def _known(self) -> dict[str, TableVariable]:
        if self._variables is None:
            self._variables = read_variables(self.package / "jit" / "core.py")
        return self._variables

    def defaults(self, name: str) -> list[str] | None:
        """The files aiter reads for ``name`` when it is unset, in the order it merges them, ``None`` when the package
        does not say."""
        variable = self._known().get(name)
        if variable is None:
            return None
        return [str(self.package / _CONFIGS_DIR / variable.default), *_per_model(self.package, variable)]

    def value(self, name: str) -> str:
        return (self._environment.get(name) or "").strip()

    def variable(self, name: str) -> tuple[str, ...]:
        """What ``name`` resolves to, as the content of its files."""
        if name not in self._resolved:
            self._resolved[name] = self._resolve(name)
        return self._resolved[name]

    def _resolve(self, name: str) -> tuple[str, ...]:
        value = self.value(name)
        keys = self._keys.get(name, _ANY_ROW)
        if not value:
            listed = self.defaults(name)
            return self.tree() if listed is None else _contents(listed, keys)
        listed = []
        for part in value.split(os.pathsep):
            image_default = (
                None if part != DEFAULT_TABLE_PLACEHOLDER or self._image is None else self._image.defaults(name)
            )
            listed.extend(image_default if image_default is not None else [part])
        return _contents(listed, keys)

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


def _cell(value: str) -> object:
    """A cell of a key as the generators read it: a number as its value (``64`` is ``64.0``), else its text."""
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return text
    return int(number) if number.is_integer() else number


def _shares_a_key(files: list[str], keys: Collection[tuple[str, ...]]) -> bool:
    """Whether a row of one of ``files`` has the key of a row of another, under one of ``keys`` cut to the columns every
    file with rows has; a file without rows shares no key, whatever its header, and one that cannot be read as a table
    counts as sharing every key."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    for path in files:
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                rows = [row for row in csv.reader(handle) if row]
        except (OSError, UnicodeDecodeError, csv.Error):
            return True
        tables.append((rows[0], rows[1:]) if rows else ([], []))
    for key in keys:
        columns = [column for column in key if all(column in header for header, rows in tables if rows)]
        first: dict[tuple[object, ...], int] = {}
        for index, (header, rows) in enumerate(tables):
            if not rows:
                continue
            places = [header.index(column) for column in columns]
            for row in rows:
                cells = tuple(_cell(row[place]) if place < len(row) else "" for place in places)
                if first.setdefault(cells, index) != index:
                    return True
    return False


def _contents(listed: list[str], keys: Collection[tuple[str, ...]]) -> tuple[str, ...]:
    """What aiter reads from the files ``listed``: one file as it is, several merged with the missing ones left out.

    aiter merges several in the order listed: a variable's value as it lists them, else the default table and then the
    per-model tables as their directory lists them, which differs between two directories holding the same files (the
    image's ``model_configs``, a test's private copy of it). Its merge refuses two rows only when they agree on its own
    key (``tables.table_key``), which holds columns a generator does not key by (``q_dtype_w``), and a generator keeps
    the last row of each of its ``keys``. So when a key occurs in more than one of the files the order can change the
    kernels a module holds, and they are their contents in the order listed; the image's are in the order its directory
    lists them, which is what the image's own server merges (the order its prebuilt module was generated in is not
    recorded). When no key does, each key's row comes from one file whatever the order, the generated module holds the
    same kernels for the same shapes, and they are their contents in no order."""
    if len(listed) == 1:
        return (file_identity(listed[0]),)
    files = [path for path in listed if os.path.isfile(path)]
    contents = tuple(file_identity(path) for path in files)
    return contents if _shares_a_key(files, keys) else tuple(sorted(contents))


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
    keys = _row_keys(uses.values())
    image_tables = _Tables(image, {}, keys)
    stale: set[str] = set()
    for environment in environments:
        test = _Tables(package, environment, keys, image_tables)
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
