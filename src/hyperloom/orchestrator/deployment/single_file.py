# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Export the sole source artifact of a script-owned benchmark."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from .settings_prelude import apply_settings_prelude, plan_settings


def _sibling_import(tree: ast.Module, root: Path, hook: Path) -> str | None:
    """Name a module the hook imports from beside itself, or ``None``.

    Only the hook is exported, so an import that resolves to a sibling file ships
    broken. The git check this replaces caught it as "tracks more than one file";
    reading the imports tests the property itself, and ignores scratch files the
    agent left behind that nothing imports.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level:
            return node.module or "."  # a relative import cannot resolve from one file
    stem = hook.stem
    for name in sorted(names):
        if name == stem:
            continue
        if (root / f"{name}.py").is_file() or (root / name / "__init__.py").is_file():
            return name
    return None


def build_single_file(state: dict, root: Path, hook: Path, dest: Path) -> dict:
    """Capture a clean, accepted optimization module without wrapping its API.

    The module is taken from disk rather than re-derived, because the caller has
    already proven it byte-identical to the accepted stack snapshots:
    ``export._verify_accepted_source`` compares every file recorded in the
    optimization stack against the captured tree, requiring each snapshot to be
    ``complete`` first. Those snapshots are written by the executor at KEEP time,
    so they *are* the bytes that were benchmarked -- a stronger statement than any
    "matches HEAD" check can make, and one that does not require the operator to
    keep their single file inside a repository.
    """
    if hook.parent != root.resolve() or not state.get("current_best"):
        raise ValueError("No accepted single-file workload to export")
    content = hook.read_bytes()
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise ValueError("Optimization module is not valid Python") from exc
    if not any(isinstance(node, ast.FunctionDef) and node.name == "hyperloom_optimize" for node in tree.body):
        raise ValueError("Module must define hyperloom_optimize(model)")
    sibling = _sibling_import(tree, root, hook)
    if sibling is not None:
        raise ValueError(f"Optimization module imports {sibling!r} from beside itself; the export ships one file")
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
        "Accepted runtime settings are baked into the module and applied on import -- import it once, early, before any "
        "other import (side effect only), and again at the usual call site after the model is loaded.",
    }
