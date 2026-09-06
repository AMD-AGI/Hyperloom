# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Portable projections of the durable enablement state onto the breakdown.

Every projection here drops host- and session-local material: an absolute path
names nothing on a fresh image, and a credential value names something a recipe
must never carry. What survives is what a consumer can act on -- a rebuild path,
a digest, a class name, an anchor plus a relative path.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from hyperloom.common.env_safety import is_secret_shaped_env_name

from .credentials import classify_credential_class, strip_url_userinfo

#: ``accepted_config`` keys Hyperloom's own revalidation applies. Two of the five
#: are subtractive, so a projection that drops them replays a *superset* of the
#: verified argv and env.
ACCEPTED_CONFIG_KEYS: tuple[str, ...] = (
    "extra_envs",
    "extra_server_args",
    "remove_args",
    "unset_envs",
    "args_mode",
)

_LAUNCH_EVIDENCE_KEYS: tuple[str, ...] = (
    "schema_version",
    "framework",
    "recipe_digest",
    "requested_server_args",
    "observed_server_launch_flags",
    "observed_server_identity",
    "observed_model_binding",
    "requested_model_digest",
)

#: Dropped from ``observed_server_identity`` under the rule that drops
#: ``model_path``: ``served_model_name`` defaults to the model path on SGLang,
#: so it is a path field wearing a name field's label.
_IDENTITY_PATH_FIELDS: frozenset[str] = frozenset({"model_path", "tokenizer_path", "served_model_name"})

_BUILD_INPUT_KEYS: tuple[str, ...] = (
    "component",
    "repo_url",
    "ref",
    "resolved_sha",
    "gpu_arch",
    "max_jobs",
    "torch_constraint_mode",
    "build_command",
    "env_keys",
    "env_digest",
    "ambient_keys",
    "ambient_digest",
    "credential_class",
    "credential_channels",
)


def root_id_for(path: str) -> str:
    """Return the stable, non-revealing id of the root at ``path``.

    A digest rather than the path itself: the id travels into the breakdown,
    where an absolute host path names nothing a consumer can resolve and is
    internal environment information besides.
    """
    return hashlib.sha256(str(path or "").encode("utf-8")).hexdigest()[:16]


def project_accepted_config(accepted_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project all five launch-affecting ``accepted_config`` keys.

    A key whose durable value is empty is omitted rather than emitted as a
    fabricated default, so absence still reads as absence.
    """
    source = dict(accepted_config or {})
    out: dict[str, Any] = {}
    for key in ACCEPTED_CONFIG_KEYS:
        value = source.get(key)
        if value in (None, "", [], {}):
            continue
        if key == "extra_envs" and isinstance(value, dict):
            out[key] = {str(k): str(v) for k, v in value.items()}
        elif key in ("remove_args", "unset_envs") and isinstance(value, (list, tuple)):
            out[key] = [str(v) for v in value]
        else:
            out[key] = str(value)
    return out


def _is_filesystem_path(value: str) -> bool:
    text = str(value or "")
    return text.startswith("/") or text.startswith("~/")


def project_runtime_provenance(enablement: Mapping[str, Any]) -> dict[str, Any] | None:
    """Export the graded runtime as a rebuild path, never as a location.

    ``to_runtime_override()`` projects attempt-local absolute locations into a
    venv that is explicitly not archived, so emitting those strings would publish
    paths no replay can resolve, and a flag beside them would not make them
    resolvable. What travels is which slots the launch bound, the inputs that
    re-create the venv, and the build that produced it.
    """
    from ...framework.stack_actions import FrameworkRuntime

    runtime = enablement.get("active_runtime")
    if not isinstance(runtime, dict) or not runtime:
        return None
    override = FrameworkRuntime.from_state(runtime).to_runtime_override()
    if not override:
        return None
    return {
        "override_keys": sorted(override),
        "acquisition": _project_acquisition(enablement.get("kept_stack_action")),
        "build_task_id": str(enablement.get("runtime_build_task_id") or "") or None,
        # Path-valued entries are attempt directories the rebuild re-derives; the
        # rest are build switches that change what gets launched.
        "runtime_env": {
            str(k): str(v) for k, v in (override.get("runtime_env") or {}).items() if not _is_filesystem_path(str(v))
        },
    }


def _project_acquisition(action: Any) -> dict[str, Any] | None:
    """Reduce the kept stack action to the inputs that re-create its venv."""
    if not isinstance(action, dict) or not action:
        return None
    credential_class = None
    for raw in (action.get("repo_url"), action.get("index_url")):
        credential_class = credential_class or classify_credential_class(f"pip install --index-url {raw or ''}")
    resolved = action.get("resolved_packages")
    return {
        "acquisition_method": str(action.get("acquisition_method") or ""),
        "repo_url": strip_url_userinfo(str(action.get("repo_url") or "")),
        "ref": str(action.get("ref") or ""),
        "index_url": strip_url_userinfo(str(action.get("index_url") or "")),
        "packages": [str(p) for p in (action.get("packages") or [])],
        "resolved_ref": str(action.get("resolved_ref") or "") or None,
        "resolved_packages": dict(resolved) if isinstance(resolved, dict) and resolved else None,
        "credential_class": credential_class,
        "credential_channels": [str(c) for c in (action.get("credential_channels") or [])],
    }


def project_launch_evidence(evidence: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, bool]:
    """Project ``build_launch_evidence`` onto its persistable subset.

    Every ``*_path`` key and ``model_path`` are dropped outright; the requested
    env is reduced to its sorted key list with credential-shaped names removed;
    the argv strings pass through the shipped publish-boundary sanitizer.

    Returns:
        The projection, and whether the sanitizer refused an argv it could not
        represent. A refusal omits the key rather than emitting a partially
        represented launch line.
    """
    if not isinstance(evidence, Mapping) or not evidence:
        return None, False
    from ...knowledge.remote_recipe.sanitize import sanitize_publish_server_args

    out: dict[str, Any] = {}
    argv_refused = False
    for key in _LAUNCH_EVIDENCE_KEYS:
        if key not in evidence:
            continue
        value = evidence[key]
        if key in ("requested_server_args", "observed_server_launch_flags"):
            try:
                out[key] = sanitize_publish_server_args(str(value or ""))
            except ValueError:
                argv_refused = True
        elif key == "observed_server_identity" and isinstance(value, Mapping):
            out[key] = {str(k): v for k, v in value.items() if str(k) not in _IDENTITY_PATH_FIELDS}
        else:
            out[key] = dict(value) if isinstance(value, Mapping) else value
    requested_env = evidence.get("requested_server_env")
    if isinstance(requested_env, Mapping):
        out["requested_server_env_keys"] = sorted(str(k) for k in requested_env if not is_secret_shaped_env_name(k))
    warm = evidence.get("warm_reuse")
    if isinstance(warm, Mapping):
        out["warm_reuse"] = {
            "reused_ready_server": bool(warm.get("reused_ready_server")),
            "provenance": str(warm.get("provenance") or ""),
        }
    return out, argv_refused


def project_roots(roots: Any) -> list[dict[str, Any]]:
    """Project root records, dropping the absolute path each one binds."""
    out: list[dict[str, Any]] = []
    for record in roots or []:
        if not isinstance(record, Mapping) or not record.get("id"):
            continue
        target = record.get("replay_target")
        target = target if isinstance(target, Mapping) else {}
        out.append(
            {
                "id": str(record.get("id")),
                "kind": str(record.get("kind") or "other"),
                "contributions": sorted({str(c) for c in (record.get("contributions") or [])}),
                "is_git": bool(record.get("is_git")),
                "base_sha": str(record.get("base_sha") or "") or None,
                "replay_target": {
                    "anchor": str(target.get("anchor") or "unmappable"),
                    "rel": str(target.get("rel") or ""),
                },
            }
        )
    return out


def project_source_snapshots(snapshots: Any) -> list[dict[str, Any]]:
    """Project snapshot manifests, replacing both absolute path fields.

    ``framework_root`` is replaced by the root record's ``replay_target`` and
    ``snapshot_dir`` by ``snapshot_ref``, the overlay directory relative to the
    session root the replay bundle is handed: a manifest without that reference
    names changes no consumer can fetch.
    """
    out: list[dict[str, Any]] = []
    for manifest in snapshots or []:
        if not isinstance(manifest, Mapping) or not manifest.get("root_id"):
            continue
        out.append(
            {
                "root_id": str(manifest.get("root_id")),
                "schema_version": int(manifest.get("schema_version") or 0),
                "snapshot_ref": str(manifest.get("snapshot_ref") or ""),
                "base_sha": str(manifest.get("base_sha") or "") or None,
                "provenance": str(manifest.get("provenance") or ""),
                "import_root": str(manifest.get("import_root") or ""),
                "complete": bool(manifest.get("complete")),
                "files": [
                    {"rel": str(f.get("rel") or ""), "op": str(f.get("op") or "")}
                    for f in (manifest.get("files") or [])
                    if isinstance(f, Mapping)
                ],
            }
        )
    return out


def project_build_inputs(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Project an attempt row's recorded build inputs onto the emitted set.

    Every URL is emitted with its userinfo stripped and the class it carried
    recorded instead; the build command travels as an identity, never as text.
    """
    inputs = (row or {}).get("build_inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        return None
    out: dict[str, Any] = {key: inputs.get(key) for key in _BUILD_INPUT_KEYS}
    out["repo_url"] = strip_url_userinfo(str(inputs.get("repo_url") or "")) or None
    command = inputs.get("build_command")
    out["build_command"] = dict(command) if isinstance(command, Mapping) and command else None
    out["env_keys"] = [str(k) for k in (inputs.get("env_keys") or [])]
    out["ambient_keys"] = [str(k) for k in (inputs.get("ambient_keys") or [])]
    out["credential_channels"] = [str(c) for c in (inputs.get("credential_channels") or [])]
    return out
