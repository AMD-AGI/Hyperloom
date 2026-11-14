# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Capture custom inference contracts using the source and environment recipe machinery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from hyperloom_common.io import atomic_write_json
from hyperloom.common.env_safety import is_credential_name
from hyperloom.orchestrator.enablement.recipe.keep_probe import probe_environment_closure
from hyperloom.orchestrator.source_snapshot import (
    SourceLayer,
    missing_source_snapshot,
    snapshot_source_layer,
    source_layers_in_order,
)

CONTRACT_NAME = "hyperloom_inference.json"
_MAX_BYTES = 128 * 1024 * 1024
#: Where a framework deployment keeps the captured source tree, relative to the package root.
_SOURCE_FILES = "source/files"
#: The status of a deployment that produced an artifact, and where a single-file one records it.
_EXPORTED = "exported"
_SINGLE_FILE_METADATA = "reports/deployment.json"


def _files(root: Path, patterns: list[str]) -> list[str]:
    selected = set()
    for pattern in patterns:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError(f"Source patterns must be relative: {pattern}")
        matches = [p for p in root.glob(pattern) if p.is_file()]
        if not matches:
            raise ValueError(f"Source pattern matched no files: {pattern}")
        for path in matches:
            rel = path.relative_to(root)
            if any(part in {".git", "__pycache__", ".env"} for part in rel.parts):
                continue
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Source link escapes its root: {rel}")
            selected.add(str(rel))
    if sum((root / p).stat().st_size for p in selected) > _MAX_BYTES:
        raise ValueError("Source capture exceeds 128 MiB; declare weights externally")
    return sorted(selected)


def _capture(root: Path, patterns: list[str], dest: Path) -> dict:
    snapshot = snapshot_source_layer(
        SourceLayer(framework_root=root, base_sha=None, rel_paths=_files(root, patterns)),
        dest_dir=dest,
        provenance="custom_inference_export",
    )
    if not snapshot or not snapshot["complete"]:
        raise ValueError("Source snapshot is incomplete")
    portable = {k: v for k, v in snapshot.items() if k not in {"framework_root", "snapshot_dir"}}
    atomic_write_json(dest / "manifest.json", portable)
    return portable


def _read_contract(root: Path) -> dict:
    contract = json.loads((root / CONTRACT_NAME).read_text())
    if not isinstance(contract, dict):
        raise ValueError("Custom inference contract must be a JSON object")
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported custom inference contract version")
    if not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*", contract.get("adapter", "")):
        raise ValueError("adapter must be an importable Python module")
    for name in ("source_files", "input_contract", "output_contract", "interpreter", "runtime"):
        if not contract.get(name):
            raise ValueError(f"Missing inference contract field: {name}")
    for name in ("source_files", "python_paths", "assets", "exclude_env", "cache_env"):
        values = contract.get(name, [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"{name} must be a list of strings")
    return contract


def _environment(state: dict, contract: dict, mappings: dict[str, str]) -> tuple[dict, list]:
    best = state.get("current_best") or {}
    env = {**state.get("reference_envs", {}), **state.get("operator_extra_env", {}), **best.get("extra_envs", {})}
    for name in contract.get("cache_env", []):
        env.setdefault(name, "")
    excluded = set(contract.get("exclude_env", []))
    unset = list(best.get("unset_envs") or [])
    output = {}
    for name, raw in env.items():
        if name in excluded or name in unset:
            continue
        # is_credential_name, not is_secret_shaped_env_name: this value is written into a
        # file that leaves the machine, so it is held to the same rule as the other
        # operator-facing artifact (reference_script, which redacts on the same check).
        # The looser one misses AWS_ACCESS_KEY_ID, *_PRIVATE_KEY, *_SIGNATURE and *_AUTH,
        # so half of a credential pair could ship under a refusal for the other half.
        if is_credential_name(name):
            raise ValueError(f"Inference environment requires an external credential: {name}")
        if name in contract.get("cache_env", []):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError("Invalid cache environment variable name")
            output[name] = "${RUNTIME_ROOT}/cache/" + name
            continue
        value = str(raw)
        if value.startswith("/") and ":" not in value:
            value = str(Path(value).resolve())
        for source, target in sorted(mappings.items(), key=lambda item: -len(item[0])):
            if value == source or value.startswith(source + "/"):
                anchor = "${RUNTIME_ROOT}/" if target.startswith("assets/") else "${DEPLOYMENT_ROOT}/"
                value = anchor + target + value[len(source) :]
                break
        # A whole value, not one embedded in a flag: `--ckpt /abs/path` passes. Matching
        # a `/` after a space or `=` would catch it and also refuse `--ratio 1/2`, and a
        # missed path fails loudly at the destination where a false refusal would throw
        # away the run's deliverable.
        if value.startswith("/") or ":/" in value:
            raise ValueError(f"Unmapped runtime path in {name}; declare it in assets or exclude_env")
        output[name] = value
    return output, unset


def _capture_assets(contract: dict, dest: Path, mappings: dict[str, str]) -> list[dict]:
    snapshots = []
    for index, raw in enumerate(contract.get("assets", [])):
        source = Path(raw).resolve()
        root = source if source.is_dir() else source.parent
        family = "%d" in source.name
        patterns = ["**/*"] if source.is_dir() else [source.name.replace("%d", "[0-9]*")]
        relative = f"assets/{index}"
        snapshots.append(_capture(root, patterns, dest / relative))
        if family:
            mappings[str(root)] = relative + "/files"
        else:
            mappings[str(source)] = relative + "/files" + ("" if source.is_dir() else "/" + source.name)
    return snapshots


def _verify_accepted_source(state: dict, captured: Path) -> set[str]:
    """Check the captured tree against the accepted stack; return what it verified.

    An entry with no snapshot is skipped, because a config-only KEEP touches no file and
    so records none -- but only after ``missing_source_snapshot`` rules out the case where
    one *landed* code and recorded nothing anyway. That entry changed the tree being
    captured and nothing proves what it changed, so it is refused rather than skipped;
    until now it was only logged, at lift time, long after the export read past it.

    The returned set lets a caller add its own check on top. The single-file path does,
    looking for its declared module; the contract path cannot, since it has no one file
    to look for -- which is why the refusal has to live here rather than in the caller.

    The source inert KEEPs committed is checked in the order it landed between them.
    """
    expected = {}
    # Every inert layer, unfiltered -- unlike ``_inert_layers_in``, which drops the ones
    # that landed after the lift. This checks the tree on disk, which holds all of them
    # since an inert KEEP is never reverted; that one reconstructs the tree a past
    # measurement ran on. Different questions, so the asymmetry is deliberate.
    for entry in source_layers_in_order(state.get("optimization_stack", []), state.get("inert_source_layers", [])):
        if not entry.get("source_snapshot"):
            if missing_source_snapshot(entry):
                name = str(entry.get("variant_name") or entry.get("name") or "a KEEP")
                raise ValueError(f"{name} landed a source layer with no snapshot; nothing proves what it changed")
            continue
        snapshot = Path(entry["source_snapshot"])
        manifest = json.loads((snapshot / "manifest.json").read_text())
        if not manifest.get("complete"):
            raise ValueError("Accepted source snapshot is incomplete")
        for record in manifest["files"]:
            rel = record["rel"]
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError("Invalid accepted source path")
            expected[rel] = (snapshot / "files" / rel).read_bytes() if record["op"] == "upsert" else None
    for rel, content in expected.items():
        path = captured / rel
        if (content is None and path.exists()) or (
            content is not None and (not path.is_file() or path.read_bytes() != content)
        ):
            raise ValueError(f"Captured source differs from the accepted stack: {rel}")
    return set(expected)


def _capture_payloads(state: dict, contract: dict, root: Path, dest: Path) -> tuple[dict, dict, list]:
    snapshots = {"source": _capture(root, contract["source_files"], dest / "source")}
    # The returned set is not read here: a contract export declares many source files and
    # has no single one to require. It raises on everything it can judge -- a captured file
    # that differs from its snapshot, and a layer that landed code without recording one.
    _verify_accepted_source(state, dest / _SOURCE_FILES)
    mappings = {str(root.resolve()): _SOURCE_FILES}
    python_paths = []
    for rel in contract.get("python_paths", ["."]):
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("python_paths must be relative to the source root")
        python_paths.append(str(Path(_SOURCE_FILES) / rel))
    overlay = state["current_best"].get("final_overlay")
    if overlay:
        snapshots["overlay"] = _capture(Path(overlay), ["**/*"], dest / "overlay")
        mappings[str(Path(overlay).resolve())] = "overlay/files"
        python_paths.append("overlay/files")
    snapshots["assets"] = _capture_assets(contract, dest, mappings)
    return snapshots, mappings, python_paths


def _capture_recipe(session: Path, dest: Path) -> None:
    from hyperloom.inference_optimizer.breakdown.session_package import deliverable
    from hyperloom.orchestrator.enablement.recipe.sufficiency import referenced_payloads

    breakdown = session / "session_breakdown.json"
    if not breakdown.is_file():
        return
    section = json.loads(breakdown.read_text()).get("enablement", {})
    if not section:
        return
    fields = (
        "recipe_steps",
        "replay_sufficiency",
        "roots",
        "source_snapshots",
        "accepted_config",
        "runtime_provenance",
        "environment_closure",
        "installed_versions_at_keep",
        "dependency_closure_status",
    )
    recipe = {k: section[k] for k in fields if k in section}
    references = referenced_payloads(recipe)
    if deliverable(session, references) != references:
        raise ValueError("Enablement recipe references unavailable payloads")
    for rel, _digest in references:
        source = session / rel
        if not source.resolve().is_relative_to(session.resolve()):
            raise ValueError("Enablement recipe payload escapes the session")
        target = dest / "recipe" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_write_json(dest / "recipe/enablement.json", recipe)


def _single_file_deployment(state: dict, root: Path, dest: Path) -> dict | None:
    """Export the declared module, or ``None`` when the session declares no single-file scope."""
    from hyperloom.common.optimization_scope import optimization_scope

    scope = optimization_scope(str(state.get("bypass_scripts_dir") or ""))
    if scope is None:
        return None
    from .single_file import build_single_file

    # Fail closed: the module is the whole deliverable, so exporting it unverified
    # would ship bytes nothing proves were benchmarked.
    if scope.file.name not in _verify_accepted_source(state, root):
        raise ValueError(f"No accepted source snapshot covers {scope.file.name}; nothing proves what was benchmarked")
    return build_single_file(state, root, scope, dest)


def _build(session: Path, state: dict, dest: Path) -> dict:
    if not state.get("framework_repo_path"):
        raise ValueError("Framework source root is unavailable")
    root = Path(state.get("framework_repo_path") or "")
    single_file = _single_file_deployment(state, root, dest)
    if single_file is not None:
        return single_file
    contract = _read_contract(root)
    best = state.get("current_best") or {}
    if not best:
        raise ValueError("No accepted configuration to export")
    snapshots, mappings, python_paths = _capture_payloads(state, contract, root, dest)
    env, unset = _environment(state, contract, mappings)
    probe_env = dict(os.environ)
    probe_env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in unset:
        probe_env.pop(name, None)
    probe_env.update(
        {k: v.replace("${DEPLOYMENT_ROOT}", str(dest)).replace("${RUNTIME_ROOT}", str(dest)) for k, v in env.items()}
    )
    probe_env["PYTHONPATH"] = os.pathsep.join(str(dest / rel) for rel in reversed(python_paths))
    closure, _ = probe_environment_closure(interpreter=contract["interpreter"], env=probe_env)
    if not closure:
        raise ValueError("Accepted interpreter environment closure is unavailable")
    shutil.copyfile(Path(__file__).with_name("runtime.py"), dest / "inference.py")
    _capture_recipe(session, dest)
    if sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) > _MAX_BYTES:
        raise ValueError("Deployment exceeds 128 MiB; declare weights externally")
    return {
        "schema_version": 1,
        "status": _EXPORTED,
        "reasons": [],
        "validation": "not_run",
        "adapter": contract["adapter"],
        "input_contract": contract["input_contract"],
        "output_contract": contract["output_contract"],
        "runtime": contract["runtime"],
        "environment_closure": closure,
        "env": env,
        "unset_envs": unset,
        "cache_env": [name for name in contract.get("cache_env", []) if name in env],
        "python_paths": python_paths,
        "snapshots": snapshots,
        "accepted_performance": {k: best[k] for k in ("tput", "e2el_mean_ms") if k in best},
        "files": {
            str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(dest.rglob("*"))
            if p.is_file()
        },
    }


def _declares_single_file(state: dict[str, Any]) -> bool:
    """Whether this session's workload declares a single-file optimization scope."""
    from hyperloom.common.optimization_scope import optimization_scope

    try:
        return optimization_scope(str(state.get("bypass_scripts_dir") or "")) is not None
    except ValueError:
        return False


def export_custom_inference(session_dir: str | Path, state: dict[str, Any]) -> dict | None:
    """Replace the export on each CLOSE; a missing contract produces an explicit refusal.

    Capture performs no model execution. The standalone --validate command owns
    fresh-process correctness, independently of campaign benchmark acceptance.
    """
    if state.get("framework") != "custom":
        return None
    session = Path(session_dir)
    dest = session / "deployment"
    with tempfile.TemporaryDirectory(prefix=".deployment-", dir=session) as tmp:
        staged = Path(tmp)
        try:
            result = _build(session, state, staged)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            shutil.rmtree(staged)
            staged.mkdir()
            result = {"schema_version": 1, "status": "incomplete", "reasons": [str(exc)], "validation": "not_run"}
        # A refusal carries no "kind", so it must not be read as "not single_file" and
        # routed to deployment/: the rmtree below then removes the previously exported
        # hook while reports/deployment.json still reads "exported", with a digest for a
        # file that is no longer there. The declaration decides, not the result.
        single_file = result.get("kind") == "single_file" or _declares_single_file(state)
        metadata = session / _SINGLE_FILE_METADATA if single_file else staged / "deployment.json"
        atomic_write_json(metadata, result)
        if dest.exists():
            shutil.rmtree(dest)
        staged.rename(dest)
    if result["status"] == _EXPORTED:
        from hyperloom.inference_optimizer.breakdown.session_package import deliverable

        expected = {(f"deployment/{rel}", digest) for rel, digest in result["files"].items()}
        expected.add((_SINGLE_FILE_METADATA if single_file else "deployment/deployment.json", ""))
        if deliverable(session, expected) != expected:
            result.update(
                status="incomplete",
                reasons=["artifact_not_self_contained: session package limits omit deployment files"],
            )
            atomic_write_json(session / _SINGLE_FILE_METADATA if single_file else dest / "deployment.json", result)
    return result


def main() -> None:
    """Export an existing session without starting an optimization campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    result = export_custom_inference(args.session, json.loads((args.session / "state.json").read_text()))
    if result is not None:
        from hyperloom_common.io import WriteOptions, atomic_write_text
        from hyperloom.inference_optimizer.reference_script import LaunchSettings, render_reference_script

        atomic_write_text(
            args.session / "current_setting.sh",
            render_reference_script(framework="custom", launch=LaunchSettings(server_args="")),
            options=WriteOptions(mode=0o755),
        )
    print(json.dumps(result, indent=2))
    if not result or result["status"] != _EXPORTED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
