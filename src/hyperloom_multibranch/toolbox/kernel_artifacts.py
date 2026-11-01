# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A tuned table a specialist hands over as bytes, named as the :class:`~hyperloom.contracts.recipe.KernelArtifact`
its delta carries.

The reference is minted here from the content, ``kernel/<kind>/<target>/<sha256><ext>``, and never from a name the
caller picks: two branches that tune the same table under the same target put the same bytes and get the same
reference, so a merge dedupes them by equality instead of reporting a slot conflict on identical bytes.

The service reads one file per variable, and aiter rewrites a table that lists a shape twice, so a variable loads one
table of one target. The binding belongs to the session, not to a branch: the first table put for a variable, by
whichever branch, binds it to that table's target, and from then on any branch may put its own table under that
target (a new content reference under the same target; no other branch's artifact changes), while a table of another
target name for the same variable is refused. Every shape of an operator goes in one file under that target.

A put may leave the target out: a variable that is bound already takes the table under its bound target (the artifact
names it), and one that is not bound is refused as ``target_required``, since nothing says which target it is to be bound
to. Which of the two a put gets, and the binding itself, are decided in one step under one lock, so a put that arrives
while another binds the variable sees that binding.

A table put with ``rows_only`` holds only the rows a specialist tuned, so a branch need not upload the image's own: the
test's variable names the image's default table and then that file, and a shape in both is read from the file
(``hyperloom.measure.framework.rows``). It binds the variable to its target like any table.

A TunableOp results file (the variable ``hyperloom.common.outside_facts.TUNABLEOP_TABLE_ENV``) is such a table too:
PyTorch reads it from a file named after the variable's value with the device ordinal put in, so the compiler places it
under the name of every ordinal.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.aiter_jit_cache import resolve_package_root
from hyperloom.common.outside_facts import SGLANG_IMAGE_AITER_CONFIGS
from hyperloom.contracts.recipe import KERNEL_ROOT, KernelArtifact, KernelKind, RecipeError
from hyperloom.measure.compile import (
    KERNEL_LOAD_KEY,
    LOAD_ROWS_ONLY_KEY,
    LOAD_VIA_ENV_FILE,
    TABLE_EXTENSION,
    kernel_load_env,
    kernel_rows_only,
)
from hyperloom.measure.framework.rows import cell
from hyperloom.measure.framework.tables import read_variables, table_key
from hyperloom.measure.hosted_spec import MAX_ATTACHMENT_BYTES
from hyperloom_multibranch.contracts.toolbox import RequestRejected

__all__ = [
    "CODE_INVALID",
    "CODE_TARGET_REQUIRED",
    "CODE_VARIABLE_TAKEN",
    "TARGET_PATTERN",
    "KernelRowsConflict",
    "KernelRowsError",
    "KernelRowsMixed",
    "KernelRowsUnmergeable",
    "KernelVariables",
    "image_aiter_configs",
    "mint_kernel_artifact",
    "shape_order",
    "union_rows",
]

#: What a target may be: one path segment, since the target is a directory of the reference.
TARGET_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._=,+-]{0,127}")
CODE_INVALID = "kernel_artifact_invalid"
CODE_TARGET_REQUIRED = "target_required"
CODE_VARIABLE_TAKEN = "kernel_variable_taken"


def image_aiter_configs() -> Path:
    """The image's aiter tables: ``configs`` of the aiter its servers import, found without importing it (a vLLM image
    keeps it in its Python's site-packages), else where an SGLang image keeps it (``SGLANG_IMAGE_AITER_CONFIGS``)."""

    package = resolve_package_root()
    configs = None if package is None else package / "configs"
    return configs if configs is not None and configs.is_dir() else Path(SGLANG_IMAGE_AITER_CONFIGS)


def mint_kernel_artifact(
    kind: KernelKind, target: str, env: str, data: bytes, rows_only: bool = False
) -> KernelArtifact:
    """The artifact for the table ``data`` that ``env`` points the service at; its reference follows from the content.

    ``rows_only`` says ``data`` is only the rows to read over the image's default table, not a table of its own.

    Raises :class:`RequestRejected` (``kernel_artifact_invalid``) for an empty or oversized table, a target that is not
    one path segment, a ``kind`` or ``env`` the measurement has no way to load, and rows with no row.
    """

    if not data:
        raise RequestRejected("a kernel table is empty", CODE_INVALID)
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise RequestRejected(f"a kernel table is {len(data)} bytes, the cap is {MAX_ATTACHMENT_BYTES}", CODE_INVALID)
    if not TARGET_PATTERN.fullmatch(target):
        raise RequestRejected(f"a kernel target must match {TARGET_PATTERN.pattern}, got {target!r}", CODE_INVALID)
    sha256 = hashlib.sha256(data).hexdigest()
    ref = f"{KERNEL_ROOT}/{kind.value}/{target}/{sha256}{TABLE_EXTENSION}"
    load: dict[str, object] = {"via": LOAD_VIA_ENV_FILE, "env": env}
    if rows_only:
        load[LOAD_ROWS_ONLY_KEY] = True
    try:
        artifact = KernelArtifact(kind, target, ref, sha256, {KERNEL_LOAD_KEY: load})
        kernel_load_env(artifact)
        kernel_rows_only(artifact)
    except RecipeError as exc:
        raise RequestRejected(str(exc), CODE_INVALID) from None
    if rows_only:
        _check_rows(data)
    return artifact


def _check_rows(data: bytes) -> None:
    """Rows are a CSV with a header and at least one row; the columns are the test's to check against the image."""

    try:
        records = [record for record in csv.reader(io.StringIO(data.decode())) if record]
    except (UnicodeDecodeError, csv.Error) as exc:
        raise RequestRejected(f"the rows are not a CSV: {exc}", CODE_INVALID) from None
    if len(records) < 2:
        raise RequestRejected("rows_only needs the header line and at least one row of tuned rows", CODE_INVALID)


class KernelVariables:
    """The target each variable loads a table for, among the tables this toolbox has put, whichever branch put them.

    The binding lives as long as the toolbox (one per session), is not per branch, and is never released; the compiler
    refuses a recipe that loads two tables on one variable. :meth:`put` resolves the target, mints the artifact and binds
    the variable under one lock, so no put sees a binding half made.
    """

    def __init__(self) -> None:
        self._bound: dict[str, tuple[KernelKind, str]] = {}
        self._lock = threading.Lock()

    def put(
        self, kind: KernelKind, target: str | None, env: str, data: bytes, rows_only: bool = False
    ) -> KernelArtifact:
        """The artifact for the table ``data`` of ``env``, and ``env`` bound to its target.

        ``target`` ``None`` is the target ``env`` is bound to. Raises :class:`RequestRejected`: ``target_required`` for
        no target when ``env`` is bound to none, ``kernel_variable_taken`` when ``env`` is bound to another target, and
        whatever :func:`mint_kernel_artifact` refuses; a refused put binds nothing.
        """

        with self._lock:
            artifact = mint_kernel_artifact(kind, self._target_for(env, target), env, data, rows_only)
            self._bind(artifact)
        return artifact

    def _target_for(self, env: str, target: str | None) -> str:
        if target is not None:
            return target
        held = self._bound.get(env)
        if held is None:
            raise RequestRejected(
                f"no target was given and {env} is bound to none in this session: no table has been put for it yet. "
                "Name the operator or shape the table serves as target (one path segment); that first put binds "
                f"{env} to it for the session, and later puts may leave the target out to take the bound one",
                CODE_TARGET_REQUIRED,
            )
        return held[1]

    def _bind(self, artifact: KernelArtifact) -> None:
        env = kernel_load_env(artifact)
        held = self._bound.setdefault(env, artifact.slot)
        if held != artifact.slot:
            kind, target = held
            raise RequestRejected(
                f"{env} is bound to target {target!r} ({kind.value} table) in this session, not to a branch. Put your "
                f"table for {env} with target={target!r}: it is stored under its own reference and changes no other "
                f"branch's artifact. Only a different target name for {env} is refused, because the service reads one "
                f"file per variable: carry every shape of it in one table under {target!r}, or only your tuned rows "
                f"with rows_only",
                CODE_VARIABLE_TAKEN,
            )


class KernelRowsError(ValueError):
    """Two tables of one variable cannot be made one table."""


class KernelRowsConflict(KernelRowsError):
    """Both tables set a shape, and to different rows: ``shapes`` names each of them by the columns that make a shape."""

    def __init__(self, shapes: Sequence[Mapping[str, str]]) -> None:
        self.shapes = tuple(dict(shape) for shape in shapes)
        listed = "; ".join(
            " ".join(f"{name}={value}" for name, value in shape.items()) for shape in self.shapes[:_NAMED]
        )
        more = f"; and {len(self.shapes) - _NAMED} more" if len(self.shapes) > _NAMED else ""
        super().__init__(f"both tables set {len(self.shapes)} shape(s) to different rows: {listed}{more}")


class KernelRowsMixed(KernelRowsError):
    """One table is a set of rows over the image's default table and the other is a whole table: not one kind."""


class KernelRowsUnmergeable(KernelRowsError):
    """The tables cannot be read as two tables of one variable (header, shape key, a repeated shape)."""


#: How many conflicting shapes the message names; ``KernelRowsConflict.shapes`` has them all.
_NAMED = 8


@dataclass(frozen=True)
class _Table:
    header: tuple[str, ...]
    rows: list[tuple[str, ...]]


def union_rows(
    first: bytes,
    second: bytes,
    env: str,
    *,
    rows_only: tuple[bool, bool] = (False, False),
    configs: Path | None = None,
) -> bytes:
    """The rows of two tables for one ``AITER_CONFIG_*`` variable, as one table with one row to a shape.

    A shape is what ``framework.tables.table_key`` says it is for ``env``; that reads the install's untuned sibling of
    the table, so this runs where the image's aiter is reachable (``configs``, default the SGLang image's
    ``aiter/configs``, with ``jit/core.py`` beside it). A shape in both tables with the same row is kept once. The
    tables' cells are compared as aiter reads them (``64`` is ``64.0``), and the text kept is the smaller of the two.

    The result is the same for the same two tables in either order and in any row order, so it is a content reference:
    the rows are in the numeric order of their key, under the header order that sorts first, in the line ending both
    tables use (``\\n`` unless both are ``\\r\\n``). ``rows_only`` says of each table whether it is a rows-only set
    (``mint_kernel_artifact``'s flag); the union is one when both are and a whole table when neither is, so the flag to
    mint it with is the one both had.

    Raises:
        KernelRowsConflict: When both tables set a shape and the rows differ; it names every such shape.
        KernelRowsMixed: When one table is rows-only and the other is not.
        KernelRowsUnmergeable: When a table is not a CSV, the two headers are not one set of columns, the install gives
            ``env`` no shape key, or a table lists one shape twice (aiter refuses such a table).
    """
    first_only, second_only = rows_only
    if first_only != second_only:
        raise KernelRowsMixed(
            f"the first table is {_kind(first_only)} and the second is {_kind(second_only)}; a rows-only set "
            "is read over the image's default table and a whole table replaces it, so they do not make one table"
        )
    left, right = _read(first, "first"), _read(second, "second")
    if set(left.header) != set(right.header):
        raise KernelRowsUnmergeable(
            f"the tables have different columns ({', '.join(left.header)} against {', '.join(right.header)}); only "
            "two tables of one variable are united, and no column is filled in"
        )
    header = min(left.header, right.header)
    key = _shape_key(env, header, configs)
    merged = _by_shape(left, header, key, "first")
    conflicts = []
    for shape, row in _by_shape(right, header, key, "second").items():
        held = merged.get(shape)
        if held is None:
            merged[shape] = row
        elif [*map(cell, held)] == [*map(cell, row)]:
            merged[shape] = min(held, row)
        else:
            conflicts.append(shape)
    if conflicts:
        raise KernelRowsConflict([dict(zip(key, shape, strict=True)) for shape in sorted(conflicts, key=shape_order)])
    crlf = b"\r\n" in first and b"\r\n" in second
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\r\n" if crlf else "\n")
    writer.writerow(header)
    writer.writerows(merged[shape] for shape in sorted(merged, key=shape_order))
    return out.getvalue().encode()


def _kind(rows_only: bool) -> str:
    return "rows-only" if rows_only else "a whole table"


def _read(data: bytes, which: str) -> _Table:
    try:
        records = [tuple(record) for record in csv.reader(io.StringIO(data.decode(), newline="")) if record]
    except (UnicodeDecodeError, csv.Error) as exc:
        raise KernelRowsUnmergeable(f"the {which} table is not a CSV: {exc}") from None
    if not records:
        raise KernelRowsUnmergeable(f"the {which} table is empty: it has no header line")
    header, *rows = records
    if len(set(header)) != len(header):
        raise KernelRowsUnmergeable(f"the {which} table names a column twice: {', '.join(header)}")
    short = next((number for number, row in enumerate(rows, 2) if len(row) != len(header)), None)
    if short is not None:
        raise KernelRowsUnmergeable(f"line {short} of the {which} table does not have the header's {len(header)} cells")
    return _Table(header, rows)


def _shape_key(env: str, header: Sequence[str], configs: Path | None) -> tuple[str, ...]:
    root = image_aiter_configs() if configs is None else Path(configs)
    variable = read_variables(root.parent / "jit" / "core.py").get(env)
    key = None if variable is None else table_key(root, variable.tuned, header)
    if not key:
        raise KernelRowsUnmergeable(
            f"no shape key for {env}: the aiter install at {root.parent} does not declare it, or has no untuned table to "
            "read the key from. The rows can only be united where the image's aiter is reachable"
        )
    return key


def _by_shape(
    table: _Table, header: Sequence[str], key: Sequence[str], which: str
) -> dict[tuple[str, ...], tuple[str, ...]]:
    """The table's rows by shape, each with its cells in ``header``'s order."""
    place = {name: index for index, name in enumerate(table.header)}
    shapes: dict[tuple[str, ...], tuple[str, ...]] = {}
    for original in table.rows:
        row = tuple(original[place[name]] for name in header)
        shape = tuple(cell(row[header.index(name)]) for name in key)
        if shape in shapes:
            raise KernelRowsUnmergeable(
                f"the {which} table lists the shape {dict(zip(key, shape, strict=True))} more than once; aiter refuses "
                "a table that does, so give each shape one row"
            )
        shapes[shape] = row
    return shapes


def shape_order(shape: tuple[str, ...]) -> tuple[tuple[int, float, str], ...]:
    """The sort key of a shape (the values of its key columns): numbers in their numeric order, then the rest as text, the
    same for the same shapes whatever their input order."""
    ordered = []
    for value in shape:
        try:
            number = float(value)
        except ValueError:
            number = math.nan
        ordered.append((0, number, "") if math.isfinite(number) else (1, 0.0, value))
    return tuple(ordered)
