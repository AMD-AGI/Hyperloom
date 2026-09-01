# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Final-state value builders for Remote Recipe KB V2."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Mapping

from hyperloom.inference_optimizer.breakdown.agent_ownership import (
    LEVER_CONFIG,
    LEVER_ENABLEMENT,
    LEVER_SOURCE_PATCH,
    LEVER_UPSTREAM_PR,
    patch_lever_kind,
    patch_owner_phase,
)
from .models import (
    CONFIG_SECTION,
    KERNEL_SECTION,
    MAX_FILE_BYTES,
    PATCH_SECTION,
    RECIPE_SECTIONS,
    Artifact,
    KnowledgeBundle,
    RecipeScope,
    RemoteRecipeValidationError,
    extract_knowledge_artifact_refs,
    validate_relative_path,
)
from .sanitize import (
    HOST_ORIGIN_KEY,
    sanitize_publish_env_mapping,
    sanitize_publish_server_args,
    sanitize_shared_knowledge,
)
from ...specialists.patch_safety import parse_patch_targets

log = logging.getLogger(__name__)

CURRENT_KNOWLEDGE_SCHEMA_VERSION = 1
RECORD_KIND_HYPERLOOM_RECIPE = "hyperloom_recipe"

_PATH_KEYS = (
    "artifact_path",
    "final_report_path",
    "patch",
    "patch_path",
    "report_path",
    "tuned_file",
)
_PATH_LIST_KEYS = (
    "artifact_files",
    "artifacts",
    "changed_files",
    "patches",
    "patches_applied",
)
_SOURCE_METADATA_KEYS = (
    "source_file",
    "source_files",
    "target_file",
    "target_files",
)
_IGNORED_ACTIONS = {"replay_warm_recipe", "profile", "roofline", "conc_sweep", "sweep"}
_OVERLAY_REF_RE = re.compile(r"^patch/overlays/(\d{6})/(\d+)-([^/]+)\.patch$")
_OVERLAY_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _patch_declared_targets(path: Any) -> tuple[str, ...]:
    """Return every safe target declared by a unified diff."""
    patch = Path(str(path or ""))
    try:
        text = patch.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RemoteRecipeValidationError(f"cannot read accepted patch targets: {patch}") from exc
    try:
        return parse_patch_targets(text).all
    except ValueError as exc:
        raise RemoteRecipeValidationError(str(exc)) from exc


def _kernel_apply_root(patch_source: Any, target_file: Any, declared: Any = None) -> str:
    """Return the directory a kernel patch's declared paths are relative to.

    The absolute target and the patch's own relative path for it pin the root
    between them: the root is what remains of the target once its declared tail
    is removed. This is the inverse of what replay does when it places the
    patch, so recording it here is what lets replay skip searching for a tree
    the diff merely happens to fit.

    Args:
        patch_source: The patch file, read for its declared targets when
            ``declared`` is not supplied.
        target_file: An absolute path the patch was applied to.
        declared: The patch's declared relative targets, when already parsed.

    Returns:
        The absolute root, or ``""`` when the target does not end with any
        declared path -- which means the two do not describe the same apply.
    """
    target = Path(str(target_file or ""))
    if not target.is_absolute():
        return ""
    targets = tuple(declared) if declared is not None else _patch_declared_targets(patch_source)
    for relative in targets:
        parts = Path(str(relative)).parts
        if parts and len(parts) < len(target.parts) and target.parts[-len(parts) :] == parts:
            return str(Path(*target.parts[: -len(parts)]))
    return ""


def _kernel_host_origin(apply_root: str) -> dict[str, Any]:
    """Wrap a kernel apply root in the one subtree that may carry host paths.

    ``sanitize_shared_knowledge`` strips absolute paths everywhere else, so this
    key is what makes the root survive publication.
    """
    root = str(apply_root or "").strip()
    return {HOST_ORIGIN_KEY: {"apply_root": root}} if root.startswith("/") else {}


def _positive_int(value: Any) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _workload_shape(state: Any) -> dict[str, int]:
    """Return the replay-sensitive workload dimensions."""
    extra = _mapping(getattr(state, "baseline_workload_extra", {}))
    shape: dict[str, int] = {}
    for key in ("tp", "conc", "isl", "osl"):
        value = _positive_int(getattr(state, key, None))
        if value is None:
            value = _positive_int(extra.get(key))
        if value is not None:
            shape[key] = value
    return shape


class _Files:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifacts: list[Artifact] = []
        self.refs: set[str] = set()
        self._sources: dict[tuple[Path, str, str], str] = {}

    def add(self, source: Any, *, category: str, kind: str, name: str = "") -> str:
        raw = str(source or "").strip()
        if not raw:
            return ""
        src = Path(raw)
        if src.is_symlink():
            raise RemoteRecipeValidationError(f"artifact source must not be a symlink: {src}")
        if not src.is_file():
            return ""
        if src.stat().st_size > MAX_FILE_BYTES:
            raise RemoteRecipeValidationError(f"artifact {src} exceeds the {MAX_FILE_BYTES}-byte KB Store limit")
        resolved = src.resolve()
        source_key = (resolved, category, kind)
        if source_key in self._sources:
            ref = self._sources[source_key]
            self.refs.add(ref)
            return ref
        basename = Path(name or src.name).name or "artifact"
        rel = f"{category}/{kind}/{basename}"
        occupied = {item.path for item in self.artifacts}
        if rel in occupied:
            suffix = hashlib.sha256(str(resolved).encode()).hexdigest()[:10]
            rel = f"{category}/{kind}/{src.stem}-{suffix}{src.suffix}"
        destination = self.root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)
        artifact = Artifact(path=rel, source=destination, kind=kind, meta={"origin": category})
        self.artifacts.append(artifact)
        self.refs.add(rel)
        self._sources[source_key] = rel
        return rel

    def validate_adoption(self, source: Path, rel: str) -> None:
        """Fail before merge when a staged file cannot safely own ``rel``."""
        if source.is_symlink():
            raise RemoteRecipeValidationError(f"artifact source must not be a symlink: {source}")
        if source.stat().st_size > MAX_FILE_BYTES:
            raise RemoteRecipeValidationError(f"artifact {source} exceeds the {MAX_FILE_BYTES}-byte KB Store limit")
        if rel in self.refs:
            existing = next(
                (artifact.source for artifact in self.artifacts if artifact.path == rel),
                None,
            )
            if (
                existing is None
                or existing.stat().st_size != source.stat().st_size
                or existing.read_bytes() != source.read_bytes()
            ):
                raise RemoteRecipeValidationError(f"conflicting artifact content for shared ref: {rel}")

    def adopt(self, source: Path, rel: str) -> str:
        """Take a file that already carries its final ``category/kind/name``."""
        self.validate_adoption(source, rel)
        if rel in self.refs:
            return rel
        destination = self.root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        category = rel.split("/", 1)[0]
        kind = rel.split("/")[1] if rel.count("/") >= 2 else "artifacts"
        self.artifacts.append(
            Artifact(
                path=rel,
                source=destination,
                kind=kind,
                meta={"origin": category, "staged": True},
            )
        )
        self.refs.add(rel)
        return rel

    def adopt_with_rename(self, source: Path, rel: str) -> str:
        """Adopt an existing ref, renaming only a content conflict."""
        if source.is_symlink() or not source.is_file():
            raise RemoteRecipeValidationError(f"artifact source must be a regular file: {source}")
        if source.stat().st_size > MAX_FILE_BYTES:
            raise RemoteRecipeValidationError(f"artifact {source} exceeds the {MAX_FILE_BYTES}-byte KB Store limit")
        normalized = validate_relative_path(rel)
        if normalized not in self.refs:
            return self.adopt(source, normalized)
        existing = next(
            (artifact.source for artifact in self.artifacts if artifact.path == normalized),
            None,
        )
        if (
            existing is not None
            and existing.stat().st_size == source.stat().st_size
            and existing.read_bytes() == source.read_bytes()
        ):
            return normalized
        path = Path(normalized)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:10]
        candidate = path.with_name(f"{path.stem}-{digest}{path.suffix}").as_posix()
        index = 1
        while candidate in self.refs:
            occupied = next(
                (artifact.source for artifact in self.artifacts if artifact.path == candidate),
                None,
            )
            if occupied is not None and occupied.read_bytes() == source.read_bytes():
                return candidate
            candidate = path.with_name(f"{path.stem}-{digest}-{index}{path.suffix}").as_posix()
            index += 1
        return self.adopt(source, candidate)

    def write(self, text: str, *, rel: str, kind: str) -> str:
        destination = self.root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        self.artifacts.append(Artifact(path=rel, source=destination, kind=kind))
        self.refs.add(rel)
        return rel

    def prune_superseded(self, knowledge: Mapping[str, Any]) -> None:
        """Drop scraped files superseded by staged columns; reject staged orphans."""
        paths = {artifact.path for artifact in self.artifacts}
        referenced = extract_knowledge_artifact_refs(knowledge, paths)
        unreferenced = paths - referenced
        staged_orphans = sorted(
            artifact.path
            for artifact in self.artifacts
            if artifact.path in unreferenced and artifact.meta.get("staged")
        )
        if staged_orphans:
            raise RemoteRecipeValidationError(f"staged artifacts absent from final knowledge: {staged_orphans!r}")
        retained: list[Artifact] = []
        for artifact in self.artifacts:
            if artifact.path in unreferenced:
                artifact.source.unlink(missing_ok=True)
                self.refs.discard(artifact.path)
                continue
            retained.append(artifact)
        self.artifacts = retained


#: Levers that never belong to the kernel column. They are consulted before the
#: phase because config and source work can both be dispatched from inside the
#: kernel phase, so the phase alone does not say which column owns an entry.
_NON_KERNEL_LEVERS = frozenset(
    {
        LEVER_CONFIG,
        LEVER_ENABLEMENT,
        LEVER_SOURCE_PATCH,
        LEVER_UPSTREAM_PR,
    }
)
_KERNEL_ACTIONS = frozenset(
    {
        "fusion",
        "geak_e2e",
        "gemm_tuning",
        "integrate",
        "kernel_opt",
    }
)


def _is_kernel_entry(entry: Mapping[str, Any]) -> bool:
    """Whether a stack entry belongs to the kernel column.

    Kernel work is published from its own sub-columns, so the same entry must
    not also land in the config layer and be replayed twice.
    """
    if patch_lever_kind(entry) in _NON_KERNEL_LEVERS:
        return False
    action = str(entry.get("action") or "").strip().lower()
    if action in ("explore", "framework"):
        return False
    # Pre-``lever_kind`` rows fall back to the phase that recorded them.
    phase = (
        patch_owner_phase(entry)
        if action.startswith("integrate_patch")
        else str(entry.get("source_phase") or "").strip().upper()
    )
    if phase in ("EXPLORE", "FRAMEWORK_AGENT"):
        return False
    return phase in ("KERNEL", "KERNEL_AGENT") or action in _KERNEL_ACTIONS


def _apply_recipe_delta(
    config: dict[str, Any],
    delta: Mapping[str, Any],
) -> dict[str, Any]:
    from ...actions.executors._grid_server_args import compose_server_args
    from ...loop.coordinator_helpers import _dedupe_extra_server_args

    mode = str(delta.get("args_mode") or "append").strip().lower()
    if mode not in {"append", "replace"}:
        raise RemoteRecipeValidationError(f"unsupported recipe args_mode: {mode!r}")
    args = compose_server_args(
        inherited_args="",
        base_extra_args=str(config.get("extra_server_args") or ""),
        variant_extra_args=str(delta.get("extra_server_args") or ""),
        remove_args=delta.get("remove_args"),
        args_mode=mode,
    )
    envs = dict(_mapping(config.get("extra_envs")))
    for key in delta.get("unset_envs") or []:
        envs.pop(str(key), None)
    raw_envs = _mapping(delta.get("extra_envs"))
    framework_arg_envs = sorted(key for key in raw_envs if key.startswith("EXTRA_") and key.endswith("_ARGS"))
    if framework_arg_envs:
        raise RemoteRecipeValidationError(
            f"recipe_delta must carry framework arguments in extra_server_args, not envs: {framework_arg_envs!r}"
        )
    envs.update(raw_envs)
    return {
        "extra_server_args": _dedupe_extra_server_args(args),
        "extra_envs": envs,
    }


def build_publishable_recipe_config(state: Any) -> dict[str, Any]:
    """Build the cross-session optimization layer, excluding runtime bases."""
    config: dict[str, Any] = {
        "extra_server_args": "",
        "extra_envs": {},
    }
    outcome = _mapping(getattr(state, "warm_replay_outcome", {}))
    for raw in getattr(state, "optimization_stack", []) or []:
        entry = _mapping(raw)
        if not entry:
            continue
        action = str(entry.get("action") or "").strip().lower()
        if action == "replay_warm_recipe":
            if str(outcome.get("status") or "") != "reproduced":
                raise RemoteRecipeValidationError("replay_warm_recipe stack entry was not reproduced")
            raw_delta = entry.get("recipe_delta")
            if not isinstance(raw_delta, Mapping):
                raise RemoteRecipeValidationError("reproduced warm replay is missing recipe_delta")
            delta = dict(raw_delta)
            config = _apply_recipe_delta(
                {"extra_server_args": "", "extra_envs": {}},
                delta,
            )
            continue
        if (
            action in _IGNORED_ACTIONS
            or entry.get("baseline_enablement")
            or entry.get("attribution_eligible") is False
            or entry.get("recipe_publishable") is False
            or _is_kernel_entry(entry)
        ):
            continue
        delta = _mapping(entry.get("recipe_delta"))
        config_signal = bool(
            str(entry.get("candidate_extra_server_args") or "").strip()
            or _mapping(entry.get("candidate_extra_envs"))
            or entry.get("remove_args")
            or entry.get("unset_envs")
            or str(entry.get("args_mode") or "").strip().lower() == "replace"
        )
        if not delta:
            if config_signal:
                raise RemoteRecipeValidationError(
                    "publishable config KEEP is missing recipe_delta: "
                    f"action={action!r} variant={entry.get('variant_name')!r}"
                )
            continue
        config = _apply_recipe_delta(config, delta)
    return {
        "extra_server_args": sanitize_publish_server_args(str(config.get("extra_server_args") or "")),
        "extra_envs": sanitize_publish_env_mapping(_mapping(config.get("extra_envs"))),
    }


def _externalize_record(
    record: Mapping[str, Any],
    files: _Files,
    category: str,
    *,
    required_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Preserve a result record while replacing known local file fields with refs."""
    out = dict(record)
    for key in _SOURCE_METADATA_KEYS:
        out.pop(key, None)
    required = required_keys or set()
    for key in _PATH_KEYS:
        if key not in out:
            continue
        original = out.get(key)
        ref = files.add(
            original,
            category=category,
            kind="patches" if "patch" in key else "artifacts",
        )
        if ref:
            out[key] = ref
        else:
            if key in required and str(original or "").strip():
                raise RemoteRecipeValidationError(f"accepted {category} {key} cannot be materialized: {original!r}")
            out.pop(key, None)
    for key in _PATH_LIST_KEYS:
        if key not in out:
            continue
        values = out.get(key) or []
        if isinstance(values, (str, Path)):
            values = [values]
        refs = [
            ref
            for value in values
            if (
                ref := files.add(
                    value,
                    category=category,
                    kind="patches" if "patch" in key else "artifacts",
                )
            )
        ]
        out[key] = refs
    out.setdefault("phase", "KERNEL_AGENT")
    return out


def build_kernel_gemm_value(state: Any, files: _Files) -> dict[str, Any]:
    """Build only accepted GEMM optimizations from the final stack/result."""
    stack_rows = [
        dict(item)
        for item in (getattr(state, "optimization_stack", []) or [])
        if isinstance(item, Mapping) and str(item.get("action") or "").lower() == "gemm_tuning"
    ]
    last = _mapping(getattr(state, "last_gemm_tuning", {}))
    accepted_last = str(last.get("decision") or "").upper() == "KEEP" or str(last.get("status") or "").lower() == "kept"
    if stack_rows and accepted_last:
        stack_rows[-1] = {**last, **stack_rows[-1]}
    optimizations = []
    for row in stack_rows:
        optimizations.append(
            _externalize_record(
                {**row, "phase": row.get("phase") or "KERNEL_AGENT"},
                files,
                "kernel/gemm",
                required_keys={"tuned_file"},
            )
        )
    return {"optimizations": optimizations}


def build_kernel_fusion_value(state: Any, files: _Files) -> dict[str, Any]:
    """Build only E2E-accepted fusion solution records."""
    stack_rows = [
        dict(item)
        for item in (getattr(state, "optimization_stack", []) or [])
        if isinstance(item, Mapping) and str(item.get("action") or "").lower() == "fusion"
    ]
    result = _mapping(getattr(state, "last_fusion", {}))
    integrated = _mapping(getattr(state, "last_fusion_integrate", {}))
    if not stack_rows or str(integrated.get("decision") or "").upper() != "KEEP":
        return {"items": []}
    patch_source = stack_rows[-1].get("patch_path") or result.get("patch")
    if not str(patch_source or "").strip():
        raise RemoteRecipeValidationError("accepted kernel/fusion is missing its patch")
    patch_ref = files.add(patch_source, category="kernel/fusion", kind="patches")
    if not patch_ref:
        raise RemoteRecipeValidationError(f"accepted kernel/fusion patch cannot be materialized: {patch_source!r}")
    declared = _patch_declared_targets(patch_source)
    # Recorded now or never: replay has no way back to the checkout this was
    # measured on, and a record that cannot name it cannot be replayed.
    apply_root = str(result.get("kernel_repo") or "").strip() or _kernel_apply_root(
        patch_source,
        stack_rows[-1].get("target_file") or result.get("source_file") or result.get("target_file"),
        declared,
    )
    if not apply_root.startswith("/"):
        # Degrade per item, not per session: a rootless kernel item would poison
        # the whole combined warm replay (kernel_apply_root_missing), so it is
        # dropped from the Recipe rather than published broken. Aborting the
        # entire publish here would also throw away config, patch, and the other
        # kernel columns, which is a worse outcome than losing this one item.
        log.warning(
            "kernel/fusion KEEP dropped from Recipe: cannot derive the checkout it "
            "was applied into (patch=%r); publishing the rest of the session",
            patch_source,
        )
        files.discard(patch_ref)
        return {"items": []}
    integrated_for_publish = dict(integrated)
    for key in (
        "target_file",
        "target_files",
        "source_file",
        "source_files",
        "artifact_files",
    ):
        integrated_for_publish.pop(key, None)
    e2e_record = _externalize_record(
        integrated_for_publish,
        files,
        "kernel/fusion",
    )
    e2e_record.pop("target_file", None)
    e2e_record.pop("target_files", None)
    record = {
        **result,
        **stack_rows[-1],
        "e2e": e2e_record,
        "phase": str(stack_rows[-1].get("phase") or "KERNEL_AGENT"),
        "patch": patch_ref,
        **_kernel_host_origin(apply_root),
    }
    # Remove duplicate local-path aliases after establishing canonical refs.
    # ``kernel_repo`` goes too: the apply root's one home is host_origin, and
    # left here the sanitizer would strip it and leave the field silently empty.
    for key in (
        "patch_path",
        "target_file",
        "target_files",
        "source_file",
        "source_files",
        "artifact_files",
        "kernel_repo",
    ):
        record.pop(key, None)
    return {"items": [record]}


def match_rewrite_attempt(
    integrate: Mapping[str, Any],
    attempts: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Find micro evidence for an E2E-integrated stack row, strongest key first."""
    candidates = [(str(key), raw) for key, raw in attempts.items() if isinstance(raw, Mapping)]
    criteria = (
        (
            str(integrate.get("integration_id") or ""),
            lambda key, raw: str(raw.get("integration_id") or "") == str(integrate.get("integration_id") or ""),
        ),
        (
            str(integrate.get("task_group_key") or ""),
            lambda key, raw: str(raw.get("task_group_key") or "") == str(integrate.get("task_group_key") or ""),
        ),
        (
            str(integrate.get("kernel_id") or ""),
            lambda key, raw: (
                str(raw.get("kernel_id") or raw.get("current_kernel_id") or key)
                == str(integrate.get("kernel_id") or "")
            ),
        ),
        (
            str(integrate.get("patch_path") or ""),
            lambda key, raw: (
                str(raw.get("last_artifact_path") or raw.get("artifact_path") or "")
                == str(integrate.get("patch_path") or "")
            ),
        ),
    )
    for expected, predicate in criteria:
        if not expected:
            continue
        for key, raw in candidates:
            if predicate(key, raw):
                return raw
    return {}


def build_kernel_rewrite_value(state: Any, files: _Files) -> dict[str, Any]:
    """Build rewrite rows exclusively from E2E-integrated stack entries."""
    attempts = getattr(state, "kernel_opt_task_attempts", {}) or getattr(state, "kernel_opt_attempts", {}) or {}
    rows: list[dict[str, Any]] = []
    if not isinstance(attempts, Mapping):
        attempts = {}
    integrated = [
        dict(entry)
        for entry in (getattr(state, "optimization_stack", []) or [])
        if isinstance(entry, Mapping) and str(entry.get("action") or "").lower() == "integrate"
    ]
    for entry in integrated:
        raw = match_rewrite_attempt(entry, attempts)
        kernel_name = str(
            raw.get("kernel_name")
            or raw.get("current_kernel_id")
            or raw.get("kernel_id")
            or entry.get("kernel_id")
            or "unknown"
        )
        speedup = _number(raw.get("last_micro_speedup") or raw.get("speedup"))
        integration_id = str(entry.get("integration_id") or "")
        slug = hashlib.sha256(
            f"{integration_id}|{entry.get('kernel_id')}|{entry.get('patch_path')}".encode()
        ).hexdigest()[:10]
        # The integrated stack row is authoritative.  A matched micro attempt
        # may only fill a path that older stack rows omitted.
        patch_source = entry.get("patch_path") or raw.get("last_artifact_path") or raw.get("artifact_path")
        patch = files.add(
            patch_source,
            category="kernel/rewrite",
            kind="patches",
        )
        if not patch:
            raise RemoteRecipeValidationError(
                "accepted kernel/rewrite patch cannot be materialized: "
                f"integration_id={integration_id!r} patch={patch_source!r}"
            )
        declared = _patch_declared_targets(patch_source)
        # Recorded now or never: replay has no way back to the checkout this was
        # measured on, and a record that cannot name it cannot be replayed.
        apply_root = str(raw.get("last_deploy_repo_root") or "").strip() or _kernel_apply_root(
            patch_source,
            entry.get("target_file"),
            declared,
        )
        if not apply_root.startswith("/"):
            # Drop this one item, not the whole session: a rootless kernel item
            # would poison the combined warm replay, and raising would take
            # config/patch/other kernel items down with it.
            log.warning(
                "kernel/rewrite KEEP dropped from Recipe: cannot derive the checkout it "
                "was applied into (integration_id=%r); publishing the rest of the session",
                integration_id,
            )
            files.discard(patch)
            continue
        e2e_gain = _number(entry.get("gain_pct"))
        optimized_throughput = _number(entry.get("tput"))
        experience = files.write(
            "\n".join(
                (
                    f"# Kernel rewrite: {kernel_name}",
                    "",
                    "- Phase: KERNEL_AGENT",
                    "- Decision: KEEP",
                    f"- Measured speedup: {speedup:g}x",
                    f"- E2E gain: {e2e_gain:g}%",
                    f"- Optimized throughput: {optimized_throughput:g}",
                    f"- Patch: {patch or 'unavailable'}",
                    "",
                )
            ),
            rel=f"kernel/rewrite/experience/{slug}.md",
            kind="experience",
        )
        rows.append(
            {
                "id": integration_id or str(entry.get("task_group_key") or entry.get("kernel_id") or f"rewrite-{slug}"),
                "phase": "KERNEL_AGENT",
                "kernel_name": kernel_name,
                "speedup": speedup,
                "e2e_gain_pct": e2e_gain,
                "optimized_throughput": optimized_throughput,
                "experience_document": experience,
                "patch": patch,
                **_kernel_host_origin(apply_root),
            }
        )
    return {"items": rows}


def _experience(state: Any, name: str) -> list[Any]:
    value = getattr(state, name, []) or []
    return list(value) if isinstance(value, (list, tuple)) else []


def _worked_from_stack(stack: list[dict[str, Any]], gains: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(stack):
        action = str(entry.get("action") or "").strip().lower()
        if action in _IGNORED_ACTIONS:
            continue
        gain = gains[index] if index < len(gains) else entry.get("gain_pct")
        rows.append(
            {
                "name": str(entry.get("variant_name") or entry.get("kernel_name") or entry.get("kernel_id") or action),
                "action": action,
                "phase": str(entry.get("source_phase") or ""),
                "gain_pct": gain,
            }
        )
    return rows


def has_new_keep(state: Any) -> bool:
    """True when the promoted KEEP-only stack has a performance optimization entry.

    ``optimization_stack`` is the accepted stack, not the attempt ledger;
    individual rows therefore do not carry a redundant KEEP decision.
    Pre-baseline enablement KEEPs (``baseline_enablement``) establish a runnable
    anchor but are not performance optimizations; they alone do not qualify for
    KB writeback. ``recipe_publishable`` is deliberately not consulted: it filters
    the config layer inside :func:`build_publishable_recipe_config`, and gating the
    whole write on it would publish nothing for an enablement-only session.
    """
    for raw in getattr(state, "optimization_stack", []) or []:
        if not isinstance(raw, Mapping):
            continue
        action = str(raw.get("action") or "").strip().lower()
        if action in _IGNORED_ACTIONS:
            continue
        if raw.get("baseline_enablement"):
            continue
        return True
    return False


def _adopt_replayed_prior(
    state: Any,
    sections: Any,
    value: dict[str, Any],
    files: _Files,
    stack: list[dict[str, Any]],
) -> None:
    """Carry forward the exact prior overlays only after replay reproduced."""
    outcome = _mapping(getattr(state, "warm_replay_outcome", {}))
    if str(outcome.get("status") or "") != "reproduced":
        return
    replayed_refs = {str(ref) for ref in (outcome.get("replayed_patch_refs") or []) if str(ref)}
    if not replayed_refs:
        return
    warm_root = getattr(sections, "warm_start_dir", None)
    if warm_root is None:
        raise RemoteRecipeValidationError("replayed prior overlays have no warm-start artifact root")
    warm_root = Path(warm_root)
    recipe_path = warm_root / "recipe.json"
    if not recipe_path.is_file():
        raise RemoteRecipeValidationError("replayed prior overlays are missing warm-start recipe.json")
    try:
        import json

        document = json.loads(recipe_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"recipe.json is not a JSON object, got {type(document).__name__}")
    except (OSError, ValueError) as exc:
        raise RemoteRecipeValidationError(f"cannot read replayed prior recipe: {exc}") from exc
    prior_value = _mapping(document.get("value"))
    if not prior_value:
        knowledge = _mapping(document.get("knowledge"))
        prior_value = _mapping(knowledge.get("value"))

    replay_index = next(
        (index for index, entry in enumerate(stack) if str(entry.get("action") or "").lower() == "replay_warm_recipe"),
        -1,
    )
    if replay_index < 0:
        raise RemoteRecipeValidationError("replayed prior overlays have no replay_warm_recipe stack entry")
    # The prior column already lists its overlays in replay order, so the order
    # they are re-adopted in is the order they were replayed in.
    candidates = [
        str(ref) for ref in (_mapping(prior_value.get(PATCH_SECTION)).get("patches") or []) if str(ref) in replayed_refs
    ]
    missing_metadata = replayed_refs - set(candidates)
    if missing_metadata:
        raise RemoteRecipeValidationError(
            f"successfully replayed prior overlays are absent from prior knowledge: {sorted(missing_metadata)!r}"
        )

    # Prior apply roots keyed by the prior ref, so a re-homed overlay carries
    # forward the checkout it must be applied into. Without this the adopted
    # overlay publishes rootless and the next generation's replay skips the
    # whole Recipe with framework_apply_root_missing.
    prior_roots: dict[str, str] = {}
    for prov_row in _mapping(prior_value.get(PATCH_SECTION)).get("provenance") or []:
        if not isinstance(prov_row, Mapping):
            continue
        origin = prov_row.get("host_origin")
        if not isinstance(origin, Mapping):
            continue
        for ref, root in (origin.get("apply_roots") or {}).items():
            root_str = str(root or "").strip()
            if str(ref) and root_str:
                prior_roots[str(ref)] = root_str

    adopted_roots: dict[str, str] = {}
    member_index = 0
    seen: set[str] = set()
    files_root = warm_root / "files"
    if files_root.is_symlink():
        raise RemoteRecipeValidationError("replayed prior files root must not be a symlink")
    resolved_root = files_root.resolve()
    for old_ref in candidates:
        if old_ref in seen:
            continue
        seen.add(old_ref)
        normalized_ref = validate_relative_path(old_ref)
        source = files_root / normalized_ref
        cursor = files_root
        for part in Path(normalized_ref).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise RemoteRecipeValidationError(
                    f"successfully replayed prior overlay resolves through a symlink: {old_ref!r}"
                )
        try:
            source.resolve().relative_to(resolved_root)
        except ValueError as exc:
            raise RemoteRecipeValidationError(f"replayed prior overlay escapes files root: {old_ref!r}") from exc
        if not source.is_file():
            raise RemoteRecipeValidationError(f"successfully replayed prior overlay is missing: {old_ref!r}")
        try:
            source.read_bytes()
        except OSError as exc:
            raise RemoteRecipeValidationError(
                f"cannot read successfully replayed prior overlay {old_ref!r}: {exc}"
            ) from exc
        match = _OVERLAY_REF_RE.match(old_ref)
        old_name = match.group(3) if match else source.stem
        safe_name = _OVERLAY_NAME_RE.sub("-", old_name).strip("._-") or "patch"
        new_ref = f"{PATCH_SECTION}/overlays/{replay_index:06d}/{member_index:02d}-{safe_name}.patch"
        try:
            files.adopt(source, new_ref)
        except (OSError, ValueError) as exc:
            raise RemoteRecipeValidationError(
                f"cannot adopt successfully replayed prior overlay {old_ref!r}: {exc}"
            ) from exc
        node = _mapping(value.get(PATCH_SECTION))
        refs = [str(ref) for ref in (node.get("patches") or []) if str(ref)]
        if new_ref not in refs:
            refs.append(new_ref)
        node["patches"] = sorted(refs)
        value[PATCH_SECTION] = node
        if prior_root := prior_roots.get(old_ref):
            adopted_roots[new_ref] = prior_root
        member_index += 1

    # Re-home the prior apply roots onto the new refs, under the replay stack
    # index the adopted overlays now live at. Every adopted overlay whose prior
    # record named a root keeps it, so a re-published Recipe stays replayable
    # rather than losing its provenance one generation on. An overlay whose
    # prior record was already rootless stays rootless -- nothing is invented.
    if adopted_roots:
        node = _mapping(value.get(PATCH_SECTION))
        rows = [dict(row) for row in (node.get("provenance") or []) if isinstance(row, Mapping)]
        row = next((r for r in rows if int(r.get("stack_index") or -1) == replay_index), None)
        if row is None:
            row = {
                "stack_index": replay_index,
                "base_sha": "",
                "complete": True,
                "artifacts_outside_root": 0,
                "realized": True,
            }
            rows.append(row)
        origin = dict(row.get("host_origin") or {})
        merged_roots = dict(origin.get("apply_roots") or {})
        merged_roots.update(adopted_roots)
        origin["apply_roots"] = merged_roots
        row["host_origin"] = origin
        rows.sort(key=lambda r: int(r.get("stack_index") or 0))
        node["provenance"] = rows
        value[PATCH_SECTION] = node


def _remap_artifact_refs(value: Any, refs: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {key: _remap_artifact_refs(item, refs) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_artifact_refs(item, refs) for item in value]
    if isinstance(value, tuple):
        return tuple(_remap_artifact_refs(item, refs) for item in value)
    if isinstance(value, str):
        return refs.get(value, value)
    return value


def _unique_rows(rows: list[Any]) -> list[Any]:
    seen: set[str] = set()
    merged: list[Any] = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _adopt_prior_kernel(
    state: Any,
    sections: Any,
    value: dict[str, Any],
    files: "_Files",
) -> None:
    """Carry prior Kernel rows only after that Kernel replay was reproduced."""
    replay = _mapping(getattr(state, "warm_replay_outcome", {}))
    kernel_replay = _mapping(replay.get("kernel"))
    if (
        str(replay.get("status") or "") != "reproduced"
        or str(kernel_replay.get("status") or "") != "kept"
        or (_positive_int(kernel_replay.get("kept")) or 0) <= 0
    ):
        return
    list_keys = {
        "gemm": "optimizations",
        "fusion": "items",
        "rewrite": "items",
    }
    selected_kernel: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for raw in getattr(state, "warm_kernel_kb_plan", []) or []:
        if not isinstance(raw, Mapping) or str(raw.get("decision") or "").upper() != "KEEP":
            continue
        column = str(raw.get("column") or "").lower()
        list_key = list_keys.get(column)
        row = raw.get("recipe_row")
        if list_key is None or not isinstance(row, Mapping):
            continue
        selected_kernel.setdefault(column, {list_key: []})[list_key].append(dict(row))
    if not selected_kernel:
        return
    prior = sections.read("kernel")
    if prior is None:
        return
    files_root = Path(sections.warm_start_dir) / "files"
    prior_paths = {
        source.relative_to(files_root).as_posix()
        for source in prior.files
        if source.is_file() and not source.is_symlink()
    }
    prior_knowledge = sanitize_shared_knowledge(selected_kernel)
    referenced = extract_knowledge_artifact_refs(prior_knowledge, prior_paths)
    missing = referenced - prior_paths
    if missing:
        raise RemoteRecipeValidationError(f"prior kernel artifacts are missing: {sorted(missing)!r}")
    remapped: dict[str, str] = {}
    for ref in sorted(referenced):
        source = files_root / validate_relative_path(ref)
        remapped[ref] = files.adopt_with_rename(source, ref)

    kernel = _mapping(value.get("kernel"))
    prior_kernel = _mapping(_remap_artifact_refs(prior_knowledge, remapped))
    for column, list_key in list_keys.items():
        prior_column = _mapping(prior_kernel.get(column))
        if not prior_column:
            continue
        current_column = _mapping(kernel.get(column))
        prior_rows = prior_column.get(list_key)
        current_rows = current_column.get(list_key)
        kernel[column] = {
            **prior_column,
            **current_column,
            list_key: _unique_rows(
                [
                    *(prior_rows if isinstance(prior_rows, list) else []),
                    *(current_rows if isinstance(current_rows, list) else []),
                ]
            ),
        }
    value["kernel"] = kernel


def _validate_patch_column(value: Mapping[str, Any]) -> None:
    """Fail closed on an overlay ref the replay order cannot be derived from.

    The refs carry the stack and member index the column is ordered by, so a
    malformed one would silently reorder or drop an overlay on replay.
    """
    refs = _mapping(value.get(PATCH_SECTION)).get("patches") or []
    seen: set[str] = set()
    for raw_ref in refs:
        ref = str(raw_ref or "")
        if _OVERLAY_REF_RE.match(ref) is None:
            raise RemoteRecipeValidationError(f"value.{PATCH_SECTION}.patches contains an invalid overlay ref: {ref!r}")
        if ref in seen:
            raise RemoteRecipeValidationError(f"value.{PATCH_SECTION}.patches contains a duplicate: {ref!r}")
        seen.add(ref)


def _is_ref_list(value: Any) -> bool:
    """Whether ``value`` is a flat list of refs, so two producers can be unioned."""
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def merge_staged_sections(
    value: dict[str, Any],
    sections: Any,
    files: "_Files",
    *,
    only: Collection[str] | None = None,
    required: Collection[str] | None = None,
) -> list[str]:
    """Merge each staged column into ``value``, adopting the files it names.

    A column owns its own shape, so the staged knowledge map is carried across
    whole rather than reduced to a known set of keys. Only ref *lists* are
    unioned with what ``value`` already holds, because those are the one place
    two producers legitimately contribute to the same column: an adopted prior
    overlay and a freshly staged one.
    """
    merged: list[str] = []
    required_names = set(required or ())
    for name in sections.sections():
        if only is not None and name not in only:
            continue
        try:
            staged = sections.staged(name)
            if staged is None:
                qualifier = "required " if name in required_names else ""
                raise RemoteRecipeValidationError(f"{qualifier}staged section {name!r} cannot be read")
            if not staged.knowledge:
                qualifier = "required " if name in required_names else ""
                raise RemoteRecipeValidationError(f"{qualifier}staged section {name!r} is empty")
            staged_files = [
                (
                    source,
                    source.relative_to(sections.files_dir).as_posix(),
                )
                for source in staged.files
                if source.is_file() and not source.is_symlink()
            ]
            staged_paths = {rel for _, rel in staged_files}
            staged_refs = extract_knowledge_artifact_refs(staged.knowledge, staged_paths)
            missing = staged_refs - staged_paths
            orphaned = staged_paths - staged_refs
            if missing or orphaned:
                raise RemoteRecipeValidationError(
                    f"staged section {name!r} file mismatch: missing={sorted(missing)!r} orphaned={sorted(orphaned)!r}"
                )
            for source, rel in staged_files:
                files.validate_adoption(source, rel)
            for source, rel in staged_files:
                files.adopt(source, rel)
            current = value.get(name) if isinstance(value.get(name), Mapping) else {}
            combined = dict(staged.knowledge)
            for ref_key, before in current.items():
                after = combined.get(ref_key, [])
                if not _is_ref_list(before) or not _is_ref_list(after):
                    continue
                combined[ref_key] = list(dict.fromkeys(str(ref) for ref in [*before, *after] if str(ref)))
            value[name] = combined
            merged.append(name)
        except Exception as exc:
            if isinstance(exc, RemoteRecipeValidationError):
                raise
            qualifier = "required " if name in required_names else ""
            raise RemoteRecipeValidationError(f"{qualifier}staged section {name!r} is invalid: {exc}") from exc
    return merged


def build_remote_knowledge(
    state: Any,
    files_dir: str | Path,
    *,
    sections: Any,
) -> KnowledgeBundle:
    """Construct the final opaque knowledge document and temporary files tree.

    Every column is staged through its own facade and then merged, so this owns
    the assembly rule and none of the columns' shapes. ``config`` and ``kernel``
    are published here from the settled stack; ``patch`` was staged member by
    member as each KEEP landed, because its bytes do not outlive the worktree
    they came from.
    """
    if sections is None:
        raise RemoteRecipeValidationError("a Recipe can only be built from a staged draft")
    scope = RecipeScope.from_state(state)
    pending_sections = list(getattr(state, "kb_stage_outbox", []) or [])
    blocking_sections = [
        row for row in pending_sections if not (isinstance(row, Mapping) and row.get("missing_patch_sources"))
    ]
    dropped_sections = [
        row
        for row in [
            *pending_sections,
            *(getattr(state, "kb_stage_dead_letter", []) or []),
        ]
        if isinstance(row, Mapping) and row.get("missing_patch_sources")
    ]
    if blocking_sections:
        raise RemoteRecipeValidationError(
            "required section staging is incomplete: "
            f"{[row.get('id') for row in blocking_sections if isinstance(row, Mapping)]!r}"
        )
    root = Path(files_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = _Files(root)
    stack = [
        {**dict(item), "__stack_index": index}
        for index, item in enumerate(getattr(state, "optimization_stack", []) or [])
        if isinstance(item, Mapping)
    ]
    dropped_entries: set[tuple[str, int]] = set()
    for row in dropped_sections:
        try:
            stack_index = int(row.get("stack_index") or 0)
        except (TypeError, ValueError):
            continue
        dropped_entries.add((str(row.get("owner") or "").upper(), stack_index))
    # ``config`` and ``kernel`` are staged unconditionally just below, so only
    # ``patch`` is conditional: a KEEP that harvested overlays demands it, and a
    # record published without them would replay a weaker stack than measured.
    required_columns = {CONFIG_SECTION, KERNEL_SECTION}
    if any(
        (owner := str(item.get("kb_required_owner") or "").strip().upper())
        and (owner, int(item["__stack_index"])) not in dropped_entries
        for item in stack
    ):
        required_columns.add(PATCH_SECTION)
    current_best = _mapping(getattr(state, "current_best", {}))
    optimized_throughput = _number(current_best.get("tput"))
    validated_gain = _number(getattr(state, "cumulative_gain_validated", 0.0))
    gains = list(getattr(state, "gain_per_stack_entry", []) or [])
    worked = _experience(state, "what_worked") or _worked_from_stack(stack, gains)

    from ..agent_kb import ConfigKB, KernelAgentKB

    ConfigKB(sections).stage(build_publishable_recipe_config(state))
    KernelAgentKB(sections).stage_from_state(state, kernel_optimizer=scope.kernel_optimizer)

    value: dict[str, Any] = {name: {} for name in RECIPE_SECTIONS}
    staged_sections = merge_staged_sections(
        value,
        sections,
        files,
        only=RECIPE_SECTIONS,
        required=required_columns,
    )
    missing_columns = required_columns - set(staged_sections)
    if missing_columns:
        raise RemoteRecipeValidationError(f"required staged columns are missing: {sorted(missing_columns)!r}")
    # Carrying the prior record forward runs last: it unions rows and refs into
    # the assembled columns, so it must see what staging already contributed.
    _adopt_replayed_prior(state, sections, value, files, stack)
    if scope.kernel_optimizer == "forge":
        _adopt_prior_kernel(state, sections, value, files)
    _validate_patch_column(value)
    knowledge = sanitize_shared_knowledge(
        {
            "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
            "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
            "optimized_throughput": optimized_throughput,
            "validated_e2e_gain": validated_gain,
            "workload_shape": _workload_shape(state),
            "value": value,
            "what_worked": worked,
            "what_failed": _experience(state, "last_action_failures"),
            "remaining_gaps": _experience(state, "gaps"),
            "lessons": _experience(state, "warm_start_lessons"),
            "pitfalls": _experience(state, "warm_start_pitfalls"),
            "provenance": {
                "producer": "hyperloom-inference-optimizer",
                "kernel_optimizer": scope.kernel_optimizer,
                "phase": "CLOSE",
                "session_id": str(getattr(state, "recipe_kb_session_id", "") or getattr(state, "session_id", "")),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "optimization_stack_length": len(stack),
                "staged_sections": staged_sections,
                "dropped_staged_sections": list(
                    dict.fromkeys(str(row.get("id") or "") for row in dropped_sections if str(row.get("id") or ""))
                ),
            },
        }
    )
    files.prune_superseded(knowledge)
    bundle = KnowledgeBundle(knowledge=knowledge, artifacts=files.artifacts)
    bundle.validate()
    return bundle


def has_replay_material(document: Mapping[str, Any]) -> bool:
    """Report nonempty config, patch, or kernel material from a View."""
    knowledge = _mapping(document.get("knowledge")) or dict(document)
    value = _mapping(knowledge.get("value"))
    if not value:
        return False
    config = _mapping(value.get(CONFIG_SECTION))
    if str(config.get("extra_server_args") or "").strip() or _mapping(config.get("extra_envs")):
        return True
    # ``provenance`` is deliberately not consulted: it describes the overlays
    # rather than being replayable itself.
    patches = _mapping(value.get(PATCH_SECTION)).get("patches")
    if isinstance(patches, list) and patches:
        return True

    def _nonempty(raw: Any) -> bool:
        if isinstance(raw, Mapping):
            return any(_nonempty(item) for item in raw.values())
        if isinstance(raw, (list, tuple, set)):
            return any(_nonempty(item) for item in raw)
        return bool(raw)

    return _nonempty(value.get(KERNEL_SECTION))


def knowledge_to_warm_recipe(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and project the current Hyperloom Recipe contract for PRELUDE."""
    knowledge = _mapping(document.get("knowledge")) or dict(document)
    version = knowledge.get("knowledge_schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != CURRENT_KNOWLEDGE_SCHEMA_VERSION:
        raise RemoteRecipeValidationError(
            "record knowledge_schema_version does not match the current "
            f"Hyperloom contract ({CURRENT_KNOWLEDGE_SCHEMA_VERSION})"
        )
    if knowledge.get("record_kind") != RECORD_KIND_HYPERLOOM_RECIPE:
        raise RemoteRecipeValidationError(f"record_kind must be {RECORD_KIND_HYPERLOOM_RECIPE!r}")
    raw_value = knowledge.get("value")
    if not isinstance(raw_value, Mapping):
        raise RemoteRecipeValidationError("current Recipe is missing value")
    value = dict(raw_value)
    for section in RECIPE_SECTIONS:
        if not isinstance(value.get(section), Mapping):
            raise RemoteRecipeValidationError(f"current Recipe is missing value.{section}")
    raw_patches = _mapping(value.get(PATCH_SECTION)).get("patches") or []
    if not isinstance(raw_patches, list) or not all(isinstance(ref, str) for ref in raw_patches):
        raise RemoteRecipeValidationError(f"current Recipe value.{PATCH_SECTION}.patches must be a flat string list")
    for ref in raw_patches:
        validate_relative_path(ref)
    session_id = str(document.get("session_id") or "")
    validated_gain = _number(knowledge.get("validated_e2e_gain"))
    view = _mapping(document.get("view"))
    replayable = bool(view.get("replayable")) if isinstance(view.get("replayable"), bool) else True
    row = {
        "canonical_id": str(document.get("canonical_id") or ""),
        "best_throughput": _number(knowledge.get("optimized_throughput")),
        "validated_gain_pct": validated_gain,
        "what_worked": list(knowledge.get("what_worked") or []),
        "what_failed": list(knowledge.get("what_failed") or []),
        "remaining_gaps": list(knowledge.get("remaining_gaps") or []),
        "lessons": list(knowledge.get("lessons") or []),
        "pitfalls": list(knowledge.get("pitfalls") or []),
        "sessions": ([{"session_id": session_id, "gain_pct": validated_gain}] if session_id else []),
        "provenance": _mapping(knowledge.get("provenance")),
        "remote_session_id": session_id,
        "remote_schema_version": int(document.get("schema_version") or 2),
        "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
        "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
        "view": view,
        "view_source": str(view.get("source") or "current"),
        "replayable": replayable,
        "replay_material_available": (replayable and has_replay_material(document)),
        "replay_disabled_reason": str(view.get("replay_disabled_reason") or ""),
    }
    for key, value in _mapping(knowledge.get("workload_shape")).items():
        if key in {"tp", "conc", "isl", "osl"}:
            resolved = _positive_int(value)
            if resolved is not None:
                row[key] = resolved
    return row


__all__ = [
    "CURRENT_KNOWLEDGE_SCHEMA_VERSION",
    "RECORD_KIND_HYPERLOOM_RECIPE",
    "build_kernel_fusion_value",
    "build_kernel_gemm_value",
    "build_kernel_rewrite_value",
    "build_publishable_recipe_config",
    "build_remote_knowledge",
    "has_replay_material",
    "knowledge_to_warm_recipe",
    "has_new_keep",
    "match_rewrite_attempt",
    "merge_staged_sections",
]
