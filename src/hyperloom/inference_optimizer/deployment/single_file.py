# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Export the sole source artifact of a script-owned benchmark."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess


def build_single_file(state: dict, root: Path, hook: Path, dest: Path) -> dict:
    """Capture a clean, accepted optimization module without wrapping its API."""
    if hook.parent != root.resolve() or not state.get("current_best"):
        raise ValueError("No accepted single-file workload to export")
    tracked_result = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True)
    status_result = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True)
    if tracked_result.returncode or status_result.returncode:
        raise ValueError("Single-file export requires its original git checkout")
    tracked = tracked_result.stdout.decode().strip("\0").split("\0")
    dirty = status_result.stdout
    if tracked != [hook.name] or dirty:
        raise ValueError("Single-file export requires a clean checkout tracking only the optimization module")
    content = hook.read_bytes()
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise ValueError("Optimization module is not valid Python") from exc
    if not any(isinstance(node, ast.FunctionDef) and node.name == "hyperloom_optimize" for node in tree.body):
        raise ValueError("Module must define hyperloom_optimize(model)")
    (dest / hook.name).write_bytes(content)
    return {
        "schema_version": 1,
        "status": "exported",
        "reasons": [],
        "kind": "single_file",
        "entrypoint": "hyperloom_optimize.hyperloom_optimize",
        "files": {hook.name: hashlib.sha256(content).hexdigest()},
        "validation": "not_run",
        "accepted_performance": {"tput": state["current_best"].get("tput")},
        "usage": "Copy hyperloom_optimize.py beside the original user script and run that script in its original runtime.",
    }
