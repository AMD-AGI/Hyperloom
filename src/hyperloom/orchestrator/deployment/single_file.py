# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Export the sole source artifact of a script-owned benchmark."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess

from .settings_prelude import apply_settings_prelude, plan_settings


def _is_pycache_noise(line: bytes) -> bool:
    path = line[3:] if len(line) > 3 else b""
    parts = path.split(b"/")
    return b"__pycache__" in parts or path.endswith(b".pyc")


def build_single_file(state: dict, root: Path, hook: Path, dest: Path) -> dict:
    """Capture a clean, accepted optimization module without wrapping its API."""
    if hook.parent != root.resolve() or not state.get("current_best"):
        raise ValueError("No accepted single-file workload to export")
    tracked_result = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True)
    status_result = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True)
    if tracked_result.returncode or status_result.returncode:
        raise ValueError("Single-file export requires its original git checkout")
    tracked = tracked_result.stdout.decode().strip("\0").split("\0")
    dirty_lines = [line for line in status_result.stdout.split(b"\n") if line and not _is_pycache_noise(line)]
    dirty = b"\n".join(dirty_lines)
    if tracked != [hook.name] or dirty:
        raise ValueError("Single-file export requires a clean checkout tracking only the optimization module")
    content = hook.read_bytes()
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise ValueError("Optimization module is not valid Python") from exc
    if not any(isinstance(node, ast.FunctionDef) and node.name == "hyperloom_optimize" for node in tree.body):
        raise ValueError("Module must define hyperloom_optimize(model)")
    base_digest = hashlib.sha256(content).hexdigest()
    best = state.get("current_best") or {}
    plan = plan_settings(best.get("extra_envs"), unset_envs=best.get("unset_envs") or ())
    content = apply_settings_prelude(content, plan)
    (dest / hook.name).write_bytes(content)
    return {
        "schema_version": 2,
        "status": "exported",
        "reasons": [],
        "kind": "single_file",
        "entrypoint": "hyperloom_optimize.hyperloom_optimize",
        "files": {hook.name: hashlib.sha256(content).hexdigest()},
        "settings": {
            "source": "current_best.extra_envs",
            "base_sha256": base_digest,
            "applied_env": dict(plan.applied_env),
            "applied_api": list(plan.applied_api),
            "unset_env": list(plan.unset_env),
            "excluded": list(plan.excluded),
            "untranslated": list(plan.untranslated),
        },
        "warnings": [f"settings_not_carried: {entry['name']} ({entry['reason']})" for entry in plan.untranslated],
        "validation": "not_run",
        "accepted_performance": {"tput": state["current_best"].get("tput")},
        "usage": "Copy hyperloom_optimize.py beside the original user script and run that script in its original runtime. "
        "Accepted runtime settings are baked into the module; it sets them on import, so it does not depend on the "
        "launching environment.",
    }
