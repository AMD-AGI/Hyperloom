# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The parity build: does this builder reproduce the image's own native binary from the unpatched tree?

Once per (image, component) the unpatched pinned tree is compiled with the recipe and every binary the image has under
the recipe's output globs is compared with the one built: the dynamic symbols each exports (``nm -D --defined-only``)
must be the same set, no output may be missing or extra, and the built package must import. Any difference refuses
native editing of that component on that image: its binary was built with flags this builder does not reproduce, and a
measurement against it would compare two different compilers, not two candidates.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom.measure.compile import BlobStore, EnvRoot
from hyperloom.measure.runner.native.builder import BuildRequest, compile_tree, tree_dir
from hyperloom.measure.jit.native.record import FAILED

__all__ = ["ParityReport", "exported_symbols", "import_problem", "parity_build"]

_NM_TIMEOUT_SEC = 120
_IMPORT_TIMEOUT_SEC = 300
_SHOWN = 5

Symbols = Callable[[Path], frozenset[str]]
Importer = Callable[[str, Path, str], str | None]


@dataclass(frozen=True)
class ParityReport:
    """``differences`` is empty when the build reproduces the image; else one line per difference."""

    differences: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.differences


def exported_symbols(path: Path) -> frozenset[str]:
    """The names of the dynamic symbols ``path`` defines, as ``nm -D --defined-only`` lists them."""
    done = subprocess.run(
        ["nm", "-D", "--defined-only", "--format=posix", str(path)],
        capture_output=True,
        text=True,
        timeout=_NM_TIMEOUT_SEC,
        check=True,
    )
    return frozenset(line.split()[0] for line in done.stdout.splitlines() if line.strip())


def import_problem(python: str, package_parent: Path, import_name: str) -> str | None:
    """Why ``import_name`` does not import from ``package_parent`` with ``python``; ``None`` when it does."""
    env = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONPATH": str(package_parent), "PYTHONNOUSERSITE": "1"}
    done = subprocess.run(
        [python, "-c", f"import {import_name}"],
        cwd=str(package_parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=_IMPORT_TIMEOUT_SEC,
        check=False,
    )
    return None if done.returncode == 0 else (done.stderr.strip().splitlines() or [f"exit {done.returncode}"])[-1]


def parity_build(
    request: BuildRequest,
    blobs: BlobStore,
    scratch: Path,
    *,
    symbols: Symbols = exported_symbols,
    importer: Importer = import_problem,
) -> ParityReport:
    """Compile ``request``'s tree without its patches and compare it with the image's binaries.

    Raises:
        BuildInfraError: When the compile fails for machine reasons; parity is then unknown, not failed.
    """
    recipe = request.recipe
    unpatched = replace(request, stack=replace(request.stack, patches=()))
    staged = compile_tree(
        unpatched, EnvRoot(recipe.component, recipe.import_name, None, None, ()), blobs, scratch, against_image=False
    )
    if staged.status == FAILED:
        return ParityReport((f"the unpatched tree does not build: {staged.failure}",))
    built = tree_dir(scratch) / recipe.package_dir
    image_root = Path(request.image_root)
    image_outputs = {
        path.relative_to(image_root).as_posix() for glob in recipe.outputs for path in image_root.glob(glob)
    }
    differences = _compare(
        sorted(image_outputs), {output.path for output in staged.outputs}, built, image_root, symbols
    )
    problem = importer(request.machine.python, built, recipe.import_name)
    if problem is not None:
        differences.append(f"the built package does not import: {problem}")
    return ParityReport(tuple(differences))


def _compare(
    image_outputs: list[str], built_outputs: set[str], built: Path, image_root: Path, symbols: Symbols
) -> list[str]:
    differences = [
        f"{path}: the image has it, the build did not make it" for path in image_outputs if path not in built_outputs
    ]
    differences += [
        f"{path}: built, but the image has no such file" for path in sorted(built_outputs - set(image_outputs))
    ]
    for path in image_outputs:
        if path in built_outputs:
            ours, theirs = symbols(built / path), symbols(image_root / path)
            if ours != theirs:
                differences.append(_symbol_difference(path, ours, theirs))
    return differences


def _symbol_difference(path: str, ours: frozenset[str], theirs: frozenset[str]) -> str:
    missing, extra = sorted(theirs - ours), sorted(ours - theirs)
    return (
        f"{path}: {len(missing)} exported symbols missing (for example {', '.join(missing[:_SHOWN]) or '-'}), "
        f"{len(extra)} extra (for example {', '.join(extra[:_SHOWN]) or '-'})"
    )
