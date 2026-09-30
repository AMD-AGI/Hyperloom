# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optional worktree self-validation helper for CPU patch specialists.

Creates a private venv, installs the worktree's Python package into it using
``pip install --no-deps --no-build-isolation``, compiles the changed files, and
optionally runs pytest targets.  Results are printed as JSON to stdout.

Usage (from inside a specialist worktree)::

    python -m hyperloom.orchestrator.specialists.selfcheck \\
        --worktree /path/to/worktree \\
        [--project python] \\
        [--pytest tests/unit/test_scheduler.py]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(argv: list[str], *, cwd: Path | None = None, timeout_sec: int = 3600) -> tuple[int, str]:
    """Run a subprocess and return (returncode, combined output tail)."""
    try:
        result = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        tail = (result.stdout + result.stderr)[-2000:]
        return result.returncode, tail
    except subprocess.TimeoutExpired:
        return 1, f"timed out after {timeout_sec}s"
    except OSError as exc:
        return 1, repr(exc)


def _find_constraint_path(venv_dir: Path) -> Path:
    """Write a constraints file pinning torch from the venv's site-packages."""
    python = venv_dir / "bin" / "python"
    constraint_path = venv_dir / "torch_constraints.txt"
    try:
        from hyperloom.orchestrator.enablement.runtime.build_utils import write_rocm_torch_constraints

        write_rocm_torch_constraints(str(python), str(constraint_path))
    except Exception:
        constraint_path.write_text("", encoding="utf-8")
    return constraint_path


def run_selfcheck(
    *,
    worktree: Path,
    project_rel: str = "",
    pytest_targets: list[str] | None = None,
) -> dict[str, Any]:
    """Validate the worktree in a private venv.

    Steps:
    1. Create venv with ``--system-site-packages`` in ``worktree/.selfcheck_venv``.
    2. Write torch constraints so pip cannot downgrade torch.
    3. ``pip install --no-deps --no-build-isolation -e <project_dir>``.
    4. Import the package in the venv's Python.
    5. ``py_compile`` files changed since HEAD (``git diff --name-only HEAD``).
    6. Optionally run pytest targets.

    Args:
        worktree: Path to the specialist's git worktree.
        project_rel: Relative path inside ``worktree`` to the pip-installable
            project directory (e.g. ``python`` for sglang).  Empty means the
            worktree root itself.
        pytest_targets: Optional list of pytest paths or node ids to run.

    Returns:
        A dict with at least ``ok: bool`` and ``steps: list[dict]``.
    """
    from hyperloom.agents.framework.isolation import create_venv

    venv_dir = worktree / ".selfcheck_venv"
    project_dir = (worktree / project_rel) if project_rel else worktree
    python = venv_dir / "bin" / "python"
    steps: list[dict[str, Any]] = []

    try:
        create_venv(venv_dir)
    except Exception as exc:
        return {"ok": False, "error": f"venv creation failed: {exc!r}", "steps": steps}

    constraint_path = _find_constraint_path(venv_dir)

    install_args = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "--constraint",
        str(constraint_path),
        "-e",
        str(project_dir),
    ]
    rc, out = _run(install_args, cwd=worktree, timeout_sec=3600)
    steps.append({"step": "pip_install", "ok": rc == 0, "output_tail": out})
    if rc != 0:
        return {"ok": False, "error": "pip install failed", "steps": steps}

    rc, out = _run([str(python), "-c", "import importlib; importlib.import_module('sglang')"], cwd=worktree)
    if rc != 0:
        package_name = project_dir.name
        rc, out = _run([str(python), "-c", f"import importlib; importlib.import_module({package_name!r})"], cwd=worktree)
    steps.append({"step": "import_check", "ok": rc == 0, "output_tail": out})

    diff_rc, diff_out = _run(["git", "diff", "--name-only", "HEAD"], cwd=worktree, timeout_sec=30)
    changed_py = [
        line.strip()
        for line in (diff_out or "").splitlines()
        if line.strip().endswith(".py") and (worktree / line.strip()).is_file()
    ]
    if changed_py:
        compile_rc, compile_out = _run(
            [str(python), "-m", "py_compile", *changed_py],
            cwd=worktree,
            timeout_sec=60,
        )
        steps.append({"step": "py_compile", "ok": compile_rc == 0, "files": changed_py, "output_tail": compile_out})
    else:
        steps.append({"step": "py_compile", "ok": True, "files": [], "output_tail": ""})

    if pytest_targets:
        pytest_rc, pytest_out = _run(
            [str(python), "-m", "pytest", *pytest_targets, "--tb=short", "-q"],
            cwd=worktree,
            timeout_sec=600,
        )
        steps.append({"step": "pytest", "ok": pytest_rc == 0, "output_tail": pytest_out})

    ok = all(s["ok"] for s in steps)
    return {"ok": ok, "steps": steps}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run selfcheck and print a JSON result to stdout."""
    parser = argparse.ArgumentParser(
        prog="specialist_selfcheck",
        description="Validate a worktree's Python package in a private venv.",
    )
    parser.add_argument("--worktree", required=True, help="Path to the specialist git worktree.")
    parser.add_argument(
        "--project",
        default="",
        help="Relative path inside the worktree to the pip-installable project directory.",
    )
    parser.add_argument(
        "--pytest",
        action="append",
        dest="pytest_targets",
        default=[],
        help="Pytest target to run (repeatable).",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    result = run_selfcheck(
        worktree=Path(args.worktree),
        project_rel=args.project or "",
        pytest_targets=args.pytest_targets or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


__all__ = ["run_selfcheck"]


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
