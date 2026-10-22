# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Materialize a forge-loop run directory for one SIKL schema. The schema copy and SIKL's validation reports
live beside the workspace, not in it: the driver writes them on every measurement, which the workspace guard
would read as tampering."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hyperloom.sikl.schema import Schema

ASSETS = Path(__file__).resolve().parent / "assets"
DRIVER_TEMPLATE = ASSETS / "sikl_forge_driver.py"
API_PARAMS_PROBE = ASSETS / "sikl_api_params.py"

GITIGNORE = """__pycache__/
*.pyc
*.log
build/
forge_experiments/
"""

# The decorator that defines a GPU kernel in each candidate language, by its import path.
KERNEL_DECORATORS = {
    "flydsl": "flydsl.compiler.kernel",
    "triton": "triton.jit",
    "gluon": "triton.experimental.gluon.jit",
}

# A torch statement that computes a gemm-contract schema into its preallocated output, by the operand rank.
_TORCH_GEMMS = {
    2: "torch.matmul(a, b.t() if trans_b else b, out=out)",
    3: "torch.bmm(a, b.transpose(1, 2) if trans_b else b, out=out)",
}


class StartingKernelError(RuntimeError):
    """The run has no starting kernel, or SIKL does not pass it on every workload."""


@dataclass(frozen=True)
class Run:
    """The directories one `hyperloom sikl` run owns."""

    root: Path
    workspace: Path
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

    @property
    def schema_json(self) -> Path:
        return self.root / "schema.json"

    @property
    def validate_dir(self) -> Path:
        return self.root / "sikl_validate"


def _api_params(schema: Schema, sikl_root: Path) -> list[str]:
    """The keyword arguments SIKL calls the schema's entry with, as its contract states them."""
    env = dict(os.environ, PYTHONPATH=str(sikl_root))
    completed = subprocess.run(
        [sys.executable, str(API_PARAMS_PROBE), str(schema.path)], env=env, capture_output=True, text=True
    )
    if completed.returncode:
        raise StartingKernelError(f"SIKL could not read {schema.path}:\n{completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def torch_starting_kernel(schema: Schema, sikl_root: Path) -> str:
    """A torch implementation of a gemm-contract schema, written into its preallocated output."""
    if schema.op_type != "gemm":
        raise StartingKernelError(f"no built-in starting kernel for op {schema.op_type!r}; pass --starting-kernel")
    rank = len(schema.inputs["a"]["shape"])
    params = _api_params(schema, sikl_root)
    return (
        f'"""Starting point for {schema.name}: torch writes the product into the preallocated output."""\n\n'
        "import torch\n\n\n"
        f"def run(*, {', '.join(params)}):\n"
        f"    {_TORCH_GEMMS[rank]}\n"
    )


def _render_driver(schema: Schema, run: Run, *, sikl_root: Path, language: str, starting_kernel: str) -> str:
    template = DRIVER_TEMPLATE.read_text(encoding="utf-8")
    substitutions = {
        "@SIKL_ROOT@": str(sikl_root),
        "@SCHEMA_JSON@": str(run.schema_json),
        "@OP_DIR@": schema.op_dir,
        "@LANGUAGE@": language,
        "@KERNEL_DECORATOR@": KERNEL_DECORATORS[language],
        "@VALIDATE_DIR@": str(run.validate_dir),
        "@STARTING_DIGESTS@": json.dumps([hashlib.sha256(starting_kernel.encode("utf-8")).hexdigest()]),
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


def materialize(schema: Schema, root: Path, *, sikl_root: Path, language: str, starting_kernel: str) -> Run:
    """Create the run directory and return the paths forge-loop is handed. A non-empty directory is
    refused: the initial commit takes the whole workspace, leftovers included."""
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"run directory {root} is not empty; give each run its own directory")
    run = Run(root=root, workspace=root / "workspace", handoff=root / "handoff")
    run.workspace.mkdir(parents=True, exist_ok=True)
    run.handoff.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(schema.path, run.schema_json)
    run.kernel.write_text(starting_kernel, encoding="utf-8")
    run.driver.write_text(
        _render_driver(schema, run, sikl_root=sikl_root, language=language, starting_kernel=starting_kernel),
        encoding="utf-8",
    )
    (run.workspace / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    return run


def check_starting_kernel(run: Run) -> None:
    """Stop before dispatch unless SIKL passes the starting kernel on every workload: the loop measures every
    candidate from it."""
    completed = subprocess.run([sys.executable, str(run.driver)], cwd=run.workspace, capture_output=True, text=True)
    verdict = re.search(r"^allclose: (True|False)$", completed.stdout, re.MULTILINE)
    if verdict is None:
        raise StartingKernelError(f"SIKL gave no verdict:\n{completed.stdout[-2000:]}{completed.stderr[-2000:]}")
    if verdict.group(1) != "True":
        findings = "\n".join(line for line in completed.stdout.splitlines() if line.startswith("sikl_findings:"))
        raise StartingKernelError(f"SIKL does not pass the starting kernel on every workload:\n{findings}")
