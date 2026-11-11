# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Standalone inference API, copied into the deployment without Hyperloom dependencies."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
_runtime_directory: tempfile.TemporaryDirectory | None = None


def _manifest() -> dict:
    manifest = json.loads((ROOT / "deployment.json").read_text())
    if manifest["status"] != "exported":
        raise RuntimeError(f"Deployment is incomplete: {manifest['reasons']}")
    for rel, digest in manifest["files"].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Deployment file changed: {rel}")
    return manifest


def _prepare(manifest: dict) -> None:
    global _runtime_directory
    if _runtime_directory is None:
        _runtime_directory = tempfile.TemporaryDirectory(prefix="model-inference-")
        if (ROOT / "assets").exists():
            shutil.copytree(ROOT / "assets", Path(_runtime_directory.name) / "assets")
    runtime_root = Path(_runtime_directory.name)
    for name in manifest["cache_env"]:
        (runtime_root / "cache" / name).mkdir(parents=True, exist_ok=True)
    for name in manifest["unset_envs"]:
        os.environ.pop(name, None)
    for name, value in manifest["env"].items():
        os.environ[name] = value.replace("${DEPLOYMENT_ROOT}", str(ROOT)).replace("${RUNTIME_ROOT}", str(runtime_root))
    for rel in manifest["python_paths"]:
        path = str(ROOT / rel)
        if path not in sys.path:
            sys.path.insert(0, path)
    # sitecustomize has already run before a library consumer calls load_model.
    overlay = ROOT / "overlay/files/sitecustomize.py"
    if overlay.is_file() and "_deployment_overlay" not in sys.modules:
        spec = importlib.util.spec_from_file_location("_deployment_overlay", overlay)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)


def load_model(weights: str):
    """Load the optimized callable; call it with the adapter's documented inputs.

    Call before importing the workload or torch so accepted runtime settings
    take effect. Weights are supplied externally and are never downloaded here.
    """
    manifest = _manifest()
    _prepare(manifest)
    adapter = importlib.import_module(manifest["adapter"])
    return adapter.load_model(weights)


def check_environment(manifest: dict) -> list[str]:
    """Return differences from the environment observed during export."""
    mismatches = []
    expected_python = manifest["environment_closure"]["interpreter_tag"]
    if sys.version.split()[0] != expected_python:
        mismatches.append(f"Python: expected {expected_python}, found {sys.version.split()[0]}")
    for name, expected in manifest["environment_closure"]["distributions"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, found {actual}")
    return mismatches


def main() -> None:
    """Run inference or the adapter's quality gate in this fresh process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--report", default="validation.json")
    args = parser.parse_args()
    if not args.validate and not (args.input and args.output):
        parser.error("inference requires --input and --output")
    if args.validate:
        # Overwrite a previous success before loading anything that may fail.
        report = {
            "passed": False,
            "deployment_sha256": hashlib.sha256((ROOT / "deployment.json").read_bytes()).hexdigest(),
        }
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    manifest = _manifest()
    model = load_model(args.weights)
    adapter = importlib.import_module(manifest["adapter"])
    if args.validate:
        gate = adapter.validate(model)
        mismatches = check_environment(manifest)
        report.update(quality_gate=gate, environment_mismatches=mismatches)
        report["passed"] = isinstance(gate, dict) and gate.get("passed") is True and not mismatches
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise SystemExit("Standalone validation failed; see " + args.report)
    else:
        adapter.write_output(model(adapter.read_input(args.input)), args.output)


if __name__ == "__main__":
    main()
