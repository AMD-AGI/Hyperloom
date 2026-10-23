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

A table put with ``rows_only`` holds only the rows a specialist tuned, so a branch need not upload the image's own: the
test's variable names the image's default table and then that file, and a shape in both is read from the file
(``hyperloom.measure.framework_rows``). It binds the variable to its target like any table.

A TunableOp results file (the variable ``hyperloom.common.outside_facts.TUNABLEOP_TABLE_ENV``) is such a table too:
PyTorch reads it from a file named after the variable's value with the device ordinal put in, so the compiler places it
under the name of every ordinal.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re

from hyperloom.contracts.recipe import KERNEL_ROOT, KernelArtifact, KernelKind, RecipeError
from hyperloom.measure.compile import (
    KERNEL_LOAD_KEY,
    LOAD_ROWS_ONLY_KEY,
    LOAD_VIA_ENV_FILE,
    TABLE_EXTENSION,
    kernel_load_env,
    kernel_rows_only,
)
from hyperloom.measure.hosted_spec import MAX_ATTACHMENT_BYTES
from hyperloom_multibranch.contracts.toolbox import RequestRejected

#: What a target may be: one path segment, since the target is a directory of the reference.
TARGET_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._=,+-]{0,127}")
CODE_INVALID = "kernel_artifact_invalid"
CODE_VARIABLE_TAKEN = "kernel_variable_taken"


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
    refuses a recipe that loads two tables on one variable.
    """

    def __init__(self) -> None:
        self._bound: dict[str, tuple[KernelKind, str]] = {}

    def bind(self, artifact: KernelArtifact) -> None:
        """Bind the variable of ``artifact`` to its target; refused when the variable is bound to another one."""

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
