# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Materialize a forge-loop run directory for one SIKL definition, with the bundle copied beside the
workspace: the driver rewrites it on every measurement, which the workspace guard would read as tampering."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hyperloom.sikl.definition import Definition

ASSETS = Path(__file__).resolve().parent / "assets"
DRIVER_TEMPLATE = ASSETS / "sikl_forge_driver.py"
CANDIDATE_AUTHOR = "forge-candidate"

GITIGNORE = """__pycache__/
*.pyc
*.log
build/
forge_experiments/
"""

KERNEL_HEADER = '"""Candidate for {definition}, starting from its {origin}; keep `run`."""\n\n'

# The decorator that defines a GPU kernel in each candidate language, by its import path.
KERNEL_DECORATORS = {
    "flydsl": "flydsl.compiler.kernel",
    "triton": "triton.jit",
    "gluon": "triton.experimental.gluon.jit",
}


class StartingKernelError(RuntimeError):
    """No starting kernel passes every workload under the harness, or the harness gave no verdict."""


@dataclass(frozen=True)
class Run:
    """The directories one `hyperloom sikl` run owns."""

    root: Path
    workspace: Path
    bundle: Path
    handoff: Path

    @property
    def kernel(self) -> Path:
        return self.workspace / "kernel.py"

    @property
    def driver(self) -> Path:
        return self.workspace / "driver.py"

    @property
    def program_md(self) -> Path:
        return self.workspace / "program.md"

    @property
    def experiments(self) -> Path:
        return self.workspace / "forge_experiments"


def candidate_name(definition: Definition) -> str:
    """The solution name the loop's kernel is published under in the bundle."""
    return f"{definition.name}__forge"


def candidate_solution(definition: Definition, run: Run) -> Path:
    """Where the driver publishes the loop's kernel for the harness to score."""
    return (
        run.bundle
        / "solutions"
        / CANDIDATE_AUTHOR
        / definition.op_type
        / definition.name
        / f"{candidate_name(definition)}.json"
    )


def _render_driver(definition: Definition, run: Run, sikl_root: Path) -> str:
    template = DRIVER_TEMPLATE.read_text(encoding="utf-8")
    substitutions = {
        "@BUNDLE@": str(run.bundle),
        "@SIKL_ROOT@": str(sikl_root),
        "@DEFINITION@": definition.name,
        "@CANDIDATE_AUTHOR@": CANDIDATE_AUTHOR,
        "@CANDIDATE_NAME@": candidate_name(definition),
        "@OP_TYPE@": definition.op_type,
        "@TARGET@": json.dumps(definition.target),
        "@LANGUAGE@": definition.language,
        "@KERNEL_DECORATOR@": KERNEL_DECORATORS[definition.language],
        "@STARTING_DIGESTS@": json.dumps(sorted(_starting_digests(definition))),
    }
    for placeholder, value in substitutions.items():
        template = template.replace(placeholder, value)
    unfilled = sorted(key for key in substitutions if key in template)
    if unfilled:
        raise ValueError(f"driver template still carries unfilled placeholders: {unfilled}")
    return template


def commit(run: Run) -> None:
    """forge-loop's keep/revert is git: one commit, made once every workspace document exists."""
    git = ["git", "-C", str(run.workspace)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "config", "user.email", "hyperloom-sikl@local"], check=True)
    subprocess.run([*git, "config", "user.name", "hyperloom-sikl"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "hyperloom sikl: initial workspace"], check=True)


def materialize(definition: Definition, bundle: Path, root: Path, *, sikl_root: Path) -> Run:
    """Create the run directory and return the paths forge-loop is handed. A non-empty directory is
    refused: the initial commit takes the whole workspace, leftovers included."""
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"run directory {root} is not empty; give each run its own directory")
    run = Run(root=root, workspace=root / "workspace", bundle=root / "bundle", handoff=root / "handoff")
    run.workspace.mkdir(parents=True, exist_ok=True)
    run.handoff.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle, run.bundle, dirs_exist_ok=True)

    _write_kernel(definition, run, "reference")
    run.driver.write_text(_render_driver(definition, run, sikl_root), encoding="utf-8")
    (run.workspace / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    return run


def _starting_kernels(definition: Definition) -> dict[str, str]:
    """Each kernel the run may start from, by origin, as `kernel.py` holds it."""
    sources = {"reference": definition.reference_source, "baseline": definition.baseline_source}
    return {
        origin: KERNEL_HEADER.format(definition=definition.name, origin=origin) + source
        for origin, source in sources.items()
        if source is not None
    }


def _starting_digests(definition: Definition) -> set[str]:
    return {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in _starting_kernels(definition).values()}


def _write_kernel(definition: Definition, run: Run, origin: str) -> None:
    run.kernel.write_text(_starting_kernels(definition)[origin], encoding="utf-8")


def _passes_every_workload(run: Run) -> bool:
    completed = subprocess.run([sys.executable, str(run.driver)], cwd=run.workspace, capture_output=True, text=True)
    verdict = re.search(r"^allclose: (True|False)$", completed.stdout, re.MULTILINE)
    if verdict is None:
        raise StartingKernelError(f"the harness gave no verdict:\n{completed.stderr[-2000:]}")
    return verdict.group(1) == "True"


def choose_starting_kernel(definition: Definition, run: Run) -> str:
    """Keep the first of the reference and the baseline that passes every workload, and return which.

    The harness checks correctness under graph capture, which a reference that syncs with the host fails."""
    for origin in _starting_kernels(definition):
        _write_kernel(definition, run, origin)
        if _passes_every_workload(run):
            return origin
    raise StartingKernelError(f"neither the reference nor the baseline of {definition.name} passes every workload")
