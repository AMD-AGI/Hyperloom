# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The artifacts a patch ships beside its diff: where each one lands, and whether it is usable where it lands."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_float
from hyperloom.common.gpu_identity import row_dispatches_on
from hyperloom.inference_optimizer.framework_paths import resolve_kernel_search_roots
from hyperloom.inference_optimizer.gpu_types import amd_gpu_dispatch_identity

from ._nogit_patch import _is_within


@dataclass
class _ArtifactSpec:
    """One resolved non-diff tuned artifact to install at integration.

    Attributes:
        source: Absolute path to the artifact file inside the specialist
            workspace / worktree (sandbox-validated).
        target: Absolute install path inside a framework root (no escape).
        rel_target: The framework-relative target, normalized to the matched
            root via ``_resolve_artifact_target`` (an author's absolute target
            is converted to this relative form). Used for reporting AND as the
            framework-relative key for the durable KEEP source snapshot.
        root: The root ``rel_target`` is relative to. The KEEP
            source snapshot is keyed on one root, so an artifact installed into
            a different tree than the patches must be recognisable as such.
        kind: Free-form artifact kind label (e.g. ``config_json``).
        description: Free-form human description.
    """

    source: Path
    target: Path
    rel_target: str
    root: Path
    kind: str = ""
    description: str = ""


def _artifact_candidates(root: Path, rel: str) -> list[Path]:
    """The joins a root admits for a framework-relative artifact target.

    A pip-installed root is the package directory itself, so a target that
    repeats the package name (``vllm/model_executor/...``) belongs under the
    root's parent; a checkout holds the package one level down and joins
    directly. Both are offered and the caller picks by which parent exists,
    the same rule ``_resolve_focus_dir`` applies to prompt focus directories.

    Args:
        root: A framework search root.
        rel: The framework-relative target.

    Returns:
        Candidate absolute paths, direct join first.
    """
    candidates = [(root / rel).resolve()]
    if (root / "__init__.py").is_file():
        candidates.append((root.parent / rel).resolve())
    return candidates


def _resolve_artifact_target(rel_target: str) -> tuple[Path, str, Path] | None:
    """Resolve an artifact target (framework-relative, or absolute) to a path.

    A relative target picks the framework root whose tree already contains the
    target's parent directory; else the first existing root. On a pip-installed
    root both joins in :func:`_artifact_candidates` are tried, so a ``vllm/...``
    config lands beside the package rather than under a doubled ``vllm/vllm/``.
    Either way the resolved path must stay within the chosen root, so a ``..``
    cannot walk out of the tree it names.

    Args:
        rel_target: The install path authored by the specialist (framework-
            relative, or absolute).

    Returns:
        A ``(absolute_target, framework_relative_target, root)`` tuple, or
        ``None`` when nothing resolves safely. Callers MUST persist the relative
        target AND the root so the durable KEEP source snapshot captures the
        installed file even when the author used an absolute path.
    """
    rel = (rel_target or "").strip()
    if not rel or ".." in Path(rel).parts:
        return None
    roots = [Path(r).resolve() for r in resolve_kernel_search_roots()]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        return None
    if Path(rel).is_absolute():
        cand = Path(rel).resolve()
        for root in roots:
            if _is_within(cand, root):
                return cand, cand.relative_to(root).as_posix(), root
        return None
    # Prefer a root whose tree already holds the target's parent dir.
    for root in roots:
        for cand in _artifact_candidates(root, rel):
            if _is_within(cand, root) and cand.parent.is_dir():
                return cand, cand.relative_to(root).as_posix(), root
    for root in roots:
        for cand in _artifact_candidates(root, rel):
            if _is_within(cand, root):
                return cand, cand.relative_to(root).as_posix(), root
    return None


def _resolve_artifact_specs(
    *,
    specialist_workspace: Path,
    explicit_artifacts: list[dict[str, Any]] | None,
    done_payload: dict[str, Any] | None,
) -> tuple[list[_ArtifactSpec], list[dict[str, str]]]:
    """Resolve non-diff tuned artifacts to install.

    Order: ``params.artifacts`` → ``specialist_done.artifacts_written``. Each
    entry is ``{source, target, kind, description}``: ``source`` is resolved
    inside the specialist workspace/worktree (sandbox) and ``target`` is
    resolved inside a framework root. Malformed / out-of-sandbox entries are
    dropped and reported.

    Args:
        specialist_workspace: The specialist task workspace.
        explicit_artifacts: ``params.artifacts`` override list, or ``None``.
        done_payload: The parsed ``specialist_done.json`` payload, or ``None``.

    Returns:
        A ``(specs, errors)`` tuple: resolved specs, plus per-entry error
        records (``{artifact, error}``) for entries that could not be resolved.
    """
    raw: list[Any] = []
    if explicit_artifacts:
        raw = list(explicit_artifacts)
    elif done_payload and isinstance(done_payload.get("artifacts_written"), list):
        raw = list(done_payload["artifacts_written"])

    allowed_roots = [
        (specialist_workspace / "worktree").resolve(),
        specialist_workspace.resolve(),
    ]
    specs: list[_ArtifactSpec] = []
    errors: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            errors.append({"artifact": str(entry), "error": "not_a_mapping"})
            continue
        src_rel = str(entry.get("source") or "").strip()
        tgt_rel = str(entry.get("target") or "").strip()
        if not src_rel or not tgt_rel:
            errors.append({"artifact": json.dumps(entry), "error": "missing_source_or_target"})
            continue
        # Resolve source inside the workspace sandbox.
        src = Path(src_rel)
        if not src.is_absolute():
            for base in (specialist_workspace / "worktree", specialist_workspace):
                cand = base / src_rel
                if cand.exists():
                    src = cand
                    break
        if not src.resolve().is_file():
            errors.append({"artifact": src_rel, "error": "source_not_found"})
            continue
        src_resolved = src.resolve()
        if not any(_is_within(src_resolved, root) for root in allowed_roots):
            errors.append({"artifact": src_rel, "error": "source_outside_workspace"})
            continue
        resolved = _resolve_artifact_target(tgt_rel)
        if resolved is None:
            errors.append({"artifact": tgt_rel, "error": "target_unresolved_or_escapes_root"})
            continue
        target, rel_norm, root = resolved
        specs.append(
            _ArtifactSpec(
                source=src_resolved,
                target=target,
                rel_target=rel_norm,
                root=root,
                kind=str(entry.get("kind") or "").strip(),
                description=str(entry.get("description") or "").strip(),
            )
        )
    return specs, errors


def _is_aiter_gemm_model_config(spec: _ArtifactSpec) -> bool:
    """Return whether an artifact is an AITER runtime GEMM model-config CSV."""
    if spec.source.suffix.lower() != ".csv":
        return False
    kind = spec.kind.strip().lower()
    target = spec.target.as_posix().lower()
    filename = spec.target.name.lower()
    is_aiter_model_config = "/aiter/configs/model_configs/" in target
    return is_aiter_model_config and (kind == "model_config" or "_tuned_gemm" in filename)


def _validate_aiter_gemm_artifacts(
    specs: list[_ArtifactSpec],
    *,
    gpu_type: str | None,
) -> list[dict[str, str]]:
    """Reject AITER GEMM CSVs that cannot dispatch on the target GPU.

    A model-config seed containing rows for another architecture, or placeholder
    rows that still require an offline tuning step, has no runtime effect. Such
    an artifact must not enter E2E promotion because benchmark variance could
    otherwise attribute an unrelated gain to it.
    """
    identity = amd_gpu_dispatch_identity(gpu_type)
    if identity is None:
        return []
    expected_gfx, expected_cu_num = identity
    errors: list[dict[str, str]] = []
    required_columns = {"gfx", "cu_num", "us", "kernelName"}

    for spec in specs:
        if not _is_aiter_gemm_model_config(spec):
            continue
        base_error = {
            "artifact": str(spec.source),
            "expected_gfx": expected_gfx,
            "expected_cu_num": str(expected_cu_num),
        }
        try:
            with spec.source.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
                if not required_columns.issubset(fieldnames):
                    errors.append(
                        {
                            **base_error,
                            "error": "invalid_aiter_gemm_schema",
                        }
                    )
                    continue
                rows = list(reader)
        except (OSError, UnicodeError, csv.Error):
            errors.append({**base_error, "error": "invalid_aiter_gemm_csv"})
            continue

        target_rows = [row for row in rows if row_dispatches_on(row, identity)]
        if not target_rows:
            errors.append({**base_error, "error": "no_target_gpu_rows"})
            continue

        invalid_target_rows = 0
        for row in target_rows:
            kernel_name = str(row.get("kernelName") or "").strip().lower()
            if to_float(row.get("us"), 0.0) <= 0.0 or "placeholder" in kernel_name:
                invalid_target_rows += 1
        if invalid_target_rows:
            errors.append(
                {
                    **base_error,
                    "error": "target_gpu_rows_not_runtime_ready",
                    "invalid_rows": str(invalid_target_rows),
                    "target_rows": str(len(target_rows)),
                }
            )
    return errors
