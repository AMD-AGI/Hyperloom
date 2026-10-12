# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optional self-check for a CPU patch specialist's worktree.

Installs the worktree's framework package into a private venv next to the
worktree, imports it, byte-compiles the changed files and runs optional pytest
targets, then prints a JSON result::

    python -m hyperloom.orchestrator.specialists.selfcheck \\
        --worktree <worktree> --package-dir python/sglang --package sglang [--pytest <target> ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from hyperloom.agents.framework.isolation import create_venv
from hyperloom.orchestrator.enablement.runtime.build_utils import write_rocm_torch_constraints

_OUTPUT_TAIL_CHARS = 2000


def project_dir(worktree: Path, package_dir: str) -> Path:
    """Nearest directory at or above ``package_dir`` holding ``pyproject.toml`` or ``setup.py``."""
    start = worktree / package_dir
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / "setup.py").is_file():
            return candidate
        if candidate == worktree:
            break
    raise FileNotFoundError(f"no pyproject.toml or setup.py at or above {start} inside {worktree}")


def _step(name: str, argv: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    """Run one check and record its outcome."""
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    return {
        "step": name,
        "ok": result.returncode == 0,
        "output_tail": (result.stdout + result.stderr)[-_OUTPUT_TAIL_CHARS:],
    }


def run_selfcheck(*, worktree: Path, package_dir: str, package: str, pytest_targets: list[str]) -> dict[str, Any]:
    """Install, import, compile and test the worktree's package in ``<workspace>/selfcheck_venv``."""
    venv_dir = worktree.parent / "selfcheck_venv"
    python = venv_dir / "bin" / "python"
    if not python.is_file():
        create_venv(venv_dir)
    constraints = venv_dir / "torch_constraints.txt"
    write_rocm_torch_constraints(str(python), str(constraints))

    install = [str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation"]
    install += ["--constraint", str(constraints), "-e", str(project_dir(worktree, package_dir))]
    steps = [_step("pip_install", install, cwd=worktree, timeout_sec=3600)]
    if not steps[0]["ok"]:
        return {"ok": False, "steps": steps}

    steps.append(_step("import", [str(python), "-c", f"import {package}"], cwd=worktree.parent, timeout_sec=600))
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"], cwd=worktree, capture_output=True, text=True, check=True
    ).stdout.split()
    changed_py = [path for path in changed if path.endswith(".py") and (worktree / path).is_file()]
    if changed_py:
        steps.append(_step("py_compile", [str(python), "-m", "py_compile", *changed_py], cwd=worktree, timeout_sec=600))
    if pytest_targets:
        pytest = [str(python), "-m", "pytest", "-q", "--tb=short", *pytest_targets]
        steps.append(_step("pytest", pytest, cwd=worktree, timeout_sec=1800))
    return {"ok": all(step["ok"] for step in steps), "steps": steps}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the self-check and print its JSON result."""
    parser = argparse.ArgumentParser(prog="specialist_selfcheck", description=__doc__.splitlines()[0])
    parser.add_argument("--worktree", required=True, type=Path, help="The specialist's git worktree.")
    parser.add_argument("--package-dir", required=True, help="Framework package directory, relative to the worktree.")
    parser.add_argument("--package", required=True, help="Import name of the framework package.")
    parser.add_argument(
        "--pytest", action="append", default=[], dest="pytest_targets", help="Pytest target (repeatable)."
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    result = run_selfcheck(
        worktree=args.worktree, package_dir=args.package_dir, package=args.package, pytest_targets=args.pytest_targets
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
