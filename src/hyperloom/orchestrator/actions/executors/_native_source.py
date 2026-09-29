# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attest the deployed files and import roots of native optimizer candidates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def source_file_hashes(paths: Sequence[str | Path]) -> dict[str, str]:
    """Bind applied source bytes, refusing missing targets instead of omitting them."""
    files: dict[str, str] = {}
    for raw in paths:
        path = Path(raw).expanduser().absolute()
        if not path.is_file():
            raise ValueError(f"Native candidate source target is not a file: {path}")
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(files.items()))


def applied_source_evidence(
    framework_root: Path | None,
    patches: Sequence[Path],
    artifacts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Describe the realized files of an already-applied source transaction."""
    from ._patch_snapshot import patch_declared_ops

    operations: dict[str, str] = {}
    if patches and framework_root is None:
        raise ValueError("Native source transaction requires its applied framework root")
    if framework_root is not None:
        for relative, operation in patch_declared_ops(framework_root, list(patches)).items():
            operations[str((framework_root / relative).absolute())] = operation
    for artifact in artifacts:
        target = Path(str(artifact["root"])) / str(artifact["rel_target"])
        operations[str(target.absolute())] = "upsert"
    if (patches or artifacts) and not operations:
        raise ValueError("Native source transaction did not identify any applied files")
    return _attest_source_operations(operations)


def _attest_source_operations(operations: Mapping[str, str]) -> dict[str, Any]:
    absent = sorted(path for path, operation in operations.items() if operation == "delete")
    if any(os.path.lexists(path) for path in absent):
        raise ValueError("Native source deletion was not applied")
    return {
        "source_files": source_file_hashes([path for path in operations if path not in absent]),
        "absent_source_files": absent,
    }


def _kernel_source_operations(applied: Mapping[str, Any]) -> dict[str, str]:
    manifest = json.loads(Path(str(applied["manifest_path"])).read_text(encoding="utf-8"))
    backups = manifest.get("source_backups") or []
    if backups:
        return {
            str(Path(row["path"]).absolute()): "delete" if row.get("disposition") == "deleted" else "upsert"
            for row in backups
        }
    target = applied.get("target_file") or manifest.get("target_file")
    if not target:
        raise ValueError("Native kernel transaction did not record its deployed source paths")
    return {str(Path(target).absolute()): "upsert"}


def warm_source_evidence(params: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Bind the successful patches retained by baseline's warm replay transaction."""
    from ._patch_snapshot import patch_declared_ops

    operations: dict[str, str] = {}
    for applied in params.get("warm_kernel_apply_results") or []:
        operations.update(_kernel_source_operations(applied))
    statuses = params.get("_warm_patch_statuses") or []
    if len(statuses) != len(params.get("patches") or []):
        raise ValueError("Native warm replay has incomplete source patch application evidence")
    for index, status in enumerate(statuses):
        if status.get("status") not in {
            "applied",
            "applied_3way",
            "applied_nogit",
            "already_present",
            "present_in_dirty_worktree",
        }:
            raise ValueError("Native warm replay cannot silently omit a requested source patch")
        patch = output_dir / "warm_patches" / f"{index:03d}_{Path(status.get('patch_ref') or '').stem or 'patch'}.diff"
        root = Path(status["target_repo"])
        operations.update({str((root / path).absolute()): op for path, op in patch_declared_ops(root, [patch]).items()})
    return _attest_source_operations(operations)


def kernel_source_evidence(applied: Mapping[str, Any]) -> dict[str, Any]:
    """Bind every write/deletion recorded by the kernel deployment transaction."""
    return _attest_source_operations(_kernel_source_operations(applied))


_SOURCE_PACKAGES = ("sglang", "vllm", "aiter", "sgl_kernel", "flydsl", "flashinfer", "torch")
_IMPORT_PROBE = """import importlib.util, json, sys
result = {}
for name in json.loads(sys.argv[1]):
    spec = importlib.util.find_spec(name)
    result[name] = {
        'origin': None if spec is None else spec.origin,
        'locations': [] if spec is None else list(spec.submodule_search_locations or []),
    }
    if name == 'sitecustomize':
        result[name]['loaded'] = sys.modules.get(name) is not None
print(json.dumps(result))
"""


def _source_import_anchor(path: Path, overlay_roots: Sequence[Path]) -> tuple[str, Path]:
    for root in overlay_roots:
        if path.is_relative_to(root):
            return "sitecustomize", root / "sitecustomize.py"
    for parent in path.parents:
        if parent.name in _SOURCE_PACKAGES and (parent / "__init__.py").is_file():
            return parent.name, parent
    for parent in path.parents:
        if not any((parent / marker).is_file() for marker in ("pyproject.toml", "setup.py", "setup.cfg")):
            continue
        for name in _SOURCE_PACKAGES:
            for package in (parent / name, parent / "python" / name):
                if (package / "__init__.py").is_file():
                    return name, package
    raise ValueError(f"Native source file has no verifiable framework/package import root: {path}")


def _server_python(benchmark: Mapping[str, Any], overrides: Mapping[str, Any], env: Mapping[str, str]) -> str:
    evidence = (benchmark.get("workload_spec") or {}).get("server_launch") or {}
    argv = evidence.get("base_argv") or []
    executable = str(overrides.get("executable") or (argv[0] if argv else ""))
    resolved = shutil.which(executable, path=env.get("PATH")) if executable else None
    if not resolved:
        raise ValueError("Native source validation requires the verified server executable or a runtime override")
    path = Path(resolved)
    if str(benchmark.get("framework") or "").lower() == "vllm":
        # The vLLM console entrypoint belongs to the same environment as its Python.
        python = path.parent / "python"
        if not python.is_file():
            raise ValueError(f"Native vLLM entrypoint has no sibling Python: {path}")
        return str(python)
    return str(path)


def verify_native_source_imports(benchmark: dict[str, Any]) -> None:
    """Resolve candidate package origins with the actual server interpreter and env."""
    overrides = (benchmark.get("agentx") or {}).get("launch_overrides") or {}
    targets = list(overrides.get("source_files") or {}) + list(overrides.get("absent_source_files") or [])
    if not targets:
        return
    for raw, digest in (overrides.get("source_files") or {}).items():
        if source_file_hashes([raw]).get(str(Path(raw).absolute())) != digest:
            raise ValueError(f"Native source bytes changed after candidate application: {raw}")
    if any(os.path.lexists(path) for path in overrides.get("absent_source_files") or []):
        raise ValueError("Native deleted source reappeared before launch")
    env = {**os.environ, **{str(k): str(v) for k, v in (benchmark.get("envs") or {}).items()}}
    launch = (benchmark.get("workload_spec") or {}).get("server_launch") or {}
    env.update({str(k): str(v) for k, v in (launch.get("runtime_environment") or {}).items()})
    env.update(overrides.get("env") or {})
    for name in overrides.get("unset_env") or []:
        env.pop(name, None)
    overlay_roots = [
        Path(entry).absolute()
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and (Path(entry) / "sitecustomize.py").is_file()
    ]
    expected: dict[str, Path] = {}
    for raw in targets:
        module, anchor = _source_import_anchor(Path(raw).absolute(), overlay_roots)
        if module in expected and expected[module] != anchor:
            raise ValueError(f"Native candidate has conflicting import roots for {module}")
        expected[module] = anchor
    proc = subprocess.run(
        [_server_python(benchmark, overrides, env), "-c", _IMPORT_PROBE, json.dumps(sorted(expected))],
        env=env,
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    lines = proc.stdout.strip().splitlines()
    if not lines:
        raise ValueError("Native source import probe returned no evidence")
    observed = json.loads(lines[-1])
    for module, anchor in expected.items():
        record = observed.get(module) or {}
        if module == "sitecustomize" and record.get("loaded") is not True:
            raise ValueError(f"Native candidate overlay did not load successfully: {anchor}")
        origin = record.get("origin")
        locations = record.get("locations") or []
        matches = (
            bool(origin)
            and (Path(origin).resolve() == anchor.resolve() or Path(origin).resolve().is_relative_to(anchor.resolve()))
        ) or any(Path(location).resolve() == anchor.resolve() for location in locations)
        if not matches:
            raise ValueError(f"Native candidate source is not imported from {anchor}: {module} resolved to {record}")
    benchmark.setdefault("workload_spec", {})["source_imports"] = observed
