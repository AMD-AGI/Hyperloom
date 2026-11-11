# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hand over the accepted single-file optimization module as the campaign's deliverable.

Nothing here compiles or runs the module. "Export" in this package means copying out the
bytes a benchmark already accepted -- unrelated to an ahead-of-time ``torch.export``,
which is something the delivered module may do on the operator's own machine.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from hyperloom_common.io import atomic_write_json
from hyperloom.orchestrator.source_snapshot import missing_source_snapshot, source_layers_in_order

#: The status of a deployment that produced an artifact, and where it records it.
_EXPORTED = "exported"
_SINGLE_FILE_METADATA = "reports/deployment.json"


def _verify_accepted_source(state: dict, captured: Path) -> set[str]:
    """Check the captured tree against the accepted stack; return what it verified.

    An entry with no snapshot is skipped, because a config-only KEEP touches no file and
    so records none -- but only after ``missing_source_snapshot`` rules out the case where
    one *landed* code and recorded nothing anyway. That entry changed the tree being
    captured and nothing proves what it changed, so it is refused rather than skipped.

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


def _build(state: dict, dest: Path) -> dict:
    """Export the declared module, verified against the stack that was accepted."""
    from hyperloom.common.optimization_scope import optimization_scope

    from .single_file import build_single_file

    if not state.get("framework_repo_path"):
        raise ValueError("Framework source root is unavailable")
    root = Path(state["framework_repo_path"])
    scope = optimization_scope(str(state.get("bypass_scripts_dir") or ""))
    if scope is None:
        raise ValueError("Session declares no single-file optimization scope")
    # Fail closed: the module is the whole deliverable, so exporting it unverified
    # would ship bytes nothing proves were benchmarked.
    verified = _verify_accepted_source(state, root)
    if scope.file.name not in verified:
        raise ValueError(f"No accepted source snapshot covers {scope.file.name}; nothing proves what was benchmarked")
    # The export ships one file, so a stack that changed more than the module measured a
    # configuration the operator cannot be handed. Patches are refused at apply time and
    # cannot get here; a non-diff artifact installed inside this root can.
    beyond = sorted(verified - {scope.file.name})
    if beyond:
        raise ValueError(f"The accepted stack changed files the export cannot ship: {', '.join(beyond)}")
    return build_single_file(state, root, scope, dest)


def _declares_single_file(state: dict[str, Any]) -> bool:
    """Whether this session asked for a single-file export.

    A declaration that cannot be read still counts as one: the operator wrote it, so the
    refusal belongs on the CLOSE step rather than reading as "no export was wanted".
    """
    from hyperloom.common.optimization_scope import optimization_scope

    try:
        return optimization_scope(str(state.get("bypass_scripts_dir") or "")) is not None
    except ValueError:
        return True


def single_file_deliverable(session_dir: str | Path, state: dict[str, Any]) -> dict | None:
    """Replace the deliverable on each CLOSE; ``None`` when the session declared no scope.

    Nothing is executed here: the campaign's own benchmark is what accepted the module,
    and this hands over the bytes it accepted.
    """
    if state.get("framework") != "custom" or not _declares_single_file(state):
        return None
    session = Path(session_dir)
    dest = session / "deployment"
    with tempfile.TemporaryDirectory(prefix=".deployment-", dir=session) as tmp:
        staged = Path(tmp)
        try:
            result = _build(state, staged)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            shutil.rmtree(staged)
            staged.mkdir()
            result = {"schema_version": 2, "status": "incomplete", "reasons": [str(exc)], "validation": "not_run"}
        atomic_write_json(session / _SINGLE_FILE_METADATA, result)
        if dest.exists():
            shutil.rmtree(dest)
        staged.rename(dest)
    if result["status"] == _EXPORTED:
        from hyperloom.inference_optimizer.breakdown.session_package import deliverable as packaged_payloads

        expected = {(f"deployment/{rel}", digest) for rel, digest in result["files"].items()}
        expected.add((_SINGLE_FILE_METADATA, ""))
        if packaged_payloads(session, expected) != expected:
            result.update(
                status="incomplete",
                reasons=["artifact_not_self_contained: session package limits omit deployment files"],
            )
            atomic_write_json(session / _SINGLE_FILE_METADATA, result)
    return result
