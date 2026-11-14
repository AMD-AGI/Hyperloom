# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Export the sole source artifact of a script-owned benchmark."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from hyperloom.common.optimization_scope import OptimizationScope

from .settings_prelude import apply_settings_prelude, plan_settings


def _sibling_import(tree: ast.Module, root: Path, hook: Path) -> str | None:
    """Name a module the hook imports from beside itself, or ``None``.

    Only the hook is exported, so an import that resolves to a sibling file ships
    broken. Reading the imports tests that property itself, and ignores scratch files
    the agent left behind that nothing imports.

    Import statements only. A dynamic import names its target at runtime, so no reading
    of the source settles it, and matching the literal cases would only catch code no
    one writes to reach a file beside them.
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
        # A directory counts whether or not it holds __init__.py: since PEP 420 a
        # plain directory imports fine as a namespace package.
        if (root / f"{name}.py").is_file() or (root / name).is_dir():
            return name
    return None


# Unambiguous markers of an ahead-of-time export path. ``torch.export.export`` is
# deliberately not one: ``export`` alone is too common an attribute name to key on.
_AOTI_MARKERS = frozenset({"aoti_compile_and_package", "aoti_load_package"})


def uses_aoti(tree: ast.Module) -> bool:
    """Report whether the module compiles or loads an AOTInductor package."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _AOTI_MARKERS:
            return True
        if isinstance(node, ast.Name) and node.id in _AOTI_MARKERS:
            return True
        if isinstance(node, ast.ImportFrom):
            if any(alias.name in _AOTI_MARKERS for alias in node.names):
                return True
    return False


def _with_aoti_deployment_cost(result: dict, warnings: list[str], usage: str) -> dict:
    """Record what an exported AOTInductor module costs to deploy.

    Capture never executes the module, so the package is built at the deployment site on
    first call. Nothing else in this metadata can express that cost, and plan_settings
    deliberately refuses to carry the absolute cache paths it needs (classified
    "unportable_absolute_path"), so the module resolves them itself.
    """
    result["runtime_requirements"] = {
        "compiles_on_first_call": True,
        "writable_cache_dir": "TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR must point at writable local disk",
        "rocm_devel_headers": "building the package needs the ROCm development package; loading it does not",
        "local_tmpdir": "TMPDIR must be local disk: the build's cleanup fails on a network filesystem, "
        "and every load unpacks the whole package there",
        "pinned_toolchain": "an AOTInductor package is tied to the torch and ROCm build it was compiled "
        "against; a platform upgrade requires a fresh compile and a re-run of the accuracy gate",
    }
    result["warnings"] = warnings + [
        "aoti_first_call_compiles: the first call builds the AOTInductor package and takes "
        "far longer than accepted_performance.tput, which measures steady state only"
    ]
    result["usage"] = usage + (
        " This module compiles an AOTInductor package on its first call, so budget for that on a cold "
        "deployment and see runtime_requirements for what the build needs."
    )
    return result


def build_single_file(state: dict, root: Path, scope: OptimizationScope, dest: Path) -> dict:
    """Capture a clean, accepted optimization module without wrapping its API.

    The module is read from disk: the caller (``export._verify_accepted_source``) has
    already proven it byte-identical to the complete KEEP-time stack snapshots, which
    does not require the operator to keep the file in a repository.
    """
    hook = scope.file
    if hook.parent != root.resolve() or not state.get("current_best"):
        raise ValueError("No accepted single-file workload to export")
    content = hook.read_bytes()
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise ValueError("Optimization module is not valid Python") from exc
    if not any(isinstance(node, ast.FunctionDef) and node.name == scope.function for node in tree.body):
        raise ValueError(f"Module must define {scope.function}(model)")
    sibling = _sibling_import(tree, root, hook)
    if sibling is not None:
        raise ValueError(f"Optimization module imports {sibling!r} from beside itself; the export ships one file")
    base_digest = hashlib.sha256(content).hexdigest()
    best = state.get("current_best") or {}
    plan = plan_settings(best.get("extra_envs"), unset_envs=best.get("unset_envs") or ())
    content = apply_settings_prelude(content, plan, scope.function)
    (dest / hook.name).write_bytes(content)
    warnings = [f"settings_not_carried: {entry['name']} ({entry['reason']})" for entry in plan.untranslated]
    usage = (
        f"Copy {hook.name} beside the original user script and run that script in its original runtime. "
        "Accepted runtime settings are baked into the module and applied on import -- import it once, early, "
        "before any other import (side effect only), and again at the usual call site after the model is loaded."
    )
    result = {
        "schema_version": 2,
        "status": "exported",
        "reasons": [],
        "kind": "single_file",
        "entrypoint": f"{hook.stem}.{scope.function}",
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
        "warnings": warnings,
        "validation": "not_run",
        "accepted_performance": {"tput": state["current_best"].get("tput")},
        "usage": usage,
    }
    if not uses_aoti(tree):
        return result
    return _with_aoti_deployment_cost(result, warnings, usage)
