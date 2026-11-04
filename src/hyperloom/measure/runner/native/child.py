# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The compile of one laid tree, in the build runner's child: :func:`main` with ``[WORK]``.

The runner starts it under ``python -I`` with this module imported from the tree the runner itself runs from (see
:mod:`.isolated`), never from the interpreter's installed ``hyperloom``.

:class:`~hyperloom.measure.runner.native.isolated.IsolatedBuilds` lays the tree below ``WORK`` and writes ``job.json``;
this process, started under the runner's isolation, compiles it (:func:`~hyperloom.measure.runner.native.builder.
compile_laid`) and writes what it staged to ``staged.json``: the record, or ``{"infra": reason}`` for a machine failure.
It holds no key and signs nothing.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.runner.native.builder import BuildMachine, LaidTree, compile_laid
from hyperloom.measure.runner.native.isolated import JOB_FILE, STAGED_FILE
from hyperloom.measure.runner.native.recipes import NativeRecipe

__all__ = ["main"]


def _recipe(factory: str) -> NativeRecipe:
    module, _, name = factory.partition(":")
    return getattr(importlib.import_module(module), name)()


def _laid(job: dict) -> LaidTree:
    machine = job["machine"]
    return LaidTree(
        recipe=_recipe(job["factory"]),
        machine=BuildMachine(
            machine["image_digest"], machine["builder_image_digest"], machine["gfx_arch"], machine["python"]
        ),
        tracked=Path(job["tracked"]),
        image_root=Path(job["image_root"]),
        laid=frozenset(job["laid"]),
        environment=dict(job.get("environment", {})),
    )


def main(argv: Sequence[str] | None = None) -> int:
    work = Path((sys.argv[1:] if argv is None else argv)[0])
    job = json.loads((work / JOB_FILE).read_text(encoding="utf-8"))
    try:
        staged = compile_laid(_laid(job), work)
        document: dict[str, object] = {
            "status": staged.status,
            "outputs": [output.document() for output in staged.outputs],
            "toolchain": dict(staged.toolchain),
            "log": staged.log,
            "failure": staged.failure,
        }
    except BuildInfraError as exc:
        document = {"infra": str(exc)}
    (work / STAGED_FILE).write_text(json.dumps(document), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
