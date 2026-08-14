# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Final-state value builders for Remote Recipe KB V2."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Mapping

from .models import (
    MAX_FILE_BYTES,
    Artifact,
    KnowledgeBundle,
    RemoteRecipeValidationError,
    extract_knowledge_artifact_refs,
    validate_relative_path,
)
from .sanitize import (
    sanitize_publish_env_mapping,
    sanitize_publish_server_args,
    sanitize_shared_knowledge,
)

log = logging.getLogger(__name__)

CURRENT_KNOWLEDGE_SCHEMA_VERSION = 1
RECORD_KIND_HYPERLOOM_RECIPE = "hyperloom_recipe"

_PATH_KEYS = (
    "artifact_path",
    "final_report_path",
    "patch",
    "patch_path",
    "report_path",
    "source_file",
    "target_file",
    "tuned_file",
)
_PATH_LIST_KEYS = (
    "artifact_files",
    "artifacts",
    "changed_files",
    "patches",
    "patches_applied",
    "source_files",
    "target_files",
)
_IGNORED_ACTIONS = {"replay_warm_recipe", "profile", "roofline", "conc_sweep", "sweep"}
_OVERLAY_REF_RE = re.compile(
    r"^(explore|framework)/overlays/(\d{6})/(\d+)-([^/]+)\.patch$"
)
_OVERLAY_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


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
            raise RemoteRecipeValidationError(
                f"artifact {src} exceeds the {MAX_FILE_BYTES}-byte KB Store limit"
            )
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

    def add_tree(self, source: Any, *, category: str, kind: str) -> list[str]:
        """Copy a required artifact directory while preserving its relative tree."""
        raw = str(source or "").strip()
        if not raw:
            return []
        root = Path(raw)
        if root.is_symlink() or not root.is_dir():
            raise RemoteRecipeValidationError(
                f"accepted {category} artifact tree cannot be materialized: {root}"
            )
        refs: list[str] = []
        for src in sorted(root.rglob("*")):
            if src.is_symlink():
                raise RemoteRecipeValidationError(
                    f"accepted {category} artifact tree contains a symlink: {src}"
                )
            if not src.is_file():
                continue
            if src.stat().st_size > MAX_FILE_BYTES:
                raise RemoteRecipeValidationError(
                    f"artifact {src} exceeds the {MAX_FILE_BYTES}-byte KB Store limit"
                )
            relative = src.relative_to(root).as_posix()
            rel = f"{category}/{kind}/{relative}"
            destination = self.root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, destination)
            self.artifacts.append(
                Artifact(path=rel, source=destination, kind=kind, meta={"origin": category})
            )
            self.refs.add(rel)
            refs.append(rel)
        return refs

    def validate_adoption(self, source: Path, rel: str) -> None:
        """Fail before merge when a staged file cannot safely own ``rel``."""
        if source.is_symlink():
            raise RemoteRecipeValidationError(f"artifact source must not be a symlink: {source}")
        if source.stat().st_size > MAX_FILE_BYTES:
            raise RemoteRecipeValidationError(
                f"artifact {source} exceeds the {MAX_FILE_BYTES}-byte KB Store limit"
            )
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
                raise RemoteRecipeValidationError(
                    f"conflicting artifact content for shared ref: {rel}"
                )

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
            raise RemoteRecipeValidationError(
                "staged artifacts absent from final knowledge: "
                f"{staged_orphans!r}"
            )
        retained: list[Artifact] = []
        for artifact in self.artifacts:
            if artifact.path in unreferenced:
                artifact.source.unlink(missing_ok=True)
                self.refs.discard(artifact.path)
                continue
            retained.append(artifact)
        self.artifacts = retained


def _entry_origin(entry: Mapping[str, Any]) -> str:
    phase = str(entry.get("source_phase") or "").strip().upper()
    action = str(entry.get("action") or "").strip().lower()
    if phase == "FRAMEWORK_AGENT" or action == "framework":
        return "framework"
    if phase == "EXPLORE" or action == "explore":
        return "explore"
    if phase in ("KERNEL", "KERNEL_AGENT") or action in (
        "geak_e2e",
        "gemm_tuning",
        "fusion",
        "integrate",
        "kernel_opt",
    ):
        return "kernel"
    return ""


def _config_from(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {"extra_server_args": "", "extra_envs": {}}
    source: Mapping[str, Any] = entries[-1]
    args = str(
        source.get("effective_extra_server_args")
        or source.get("extra_server_args")
        or source.get("candidate_extra_server_args")
        or ""
    ).strip()
    envs: dict[str, Any] = {}
    for entry in entries:
        for key in entry.get("unset_envs") or []:
            envs.pop(str(key), None)
        envs.update(_mapping(entry.get("extra_envs")))
    if not envs:
        envs = _mapping(source.get("extra_envs"))
    return {
        "extra_server_args": sanitize_publish_server_args(args),
        "extra_envs": sanitize_publish_env_mapping(envs),
    }


def _config_from_current_best(state: Any) -> dict[str, Any]:
    current = _mapping(getattr(state, "current_best", {}))
    args = str(
        current.get("effective_extra_server_args")
        or current.get("extra_server_args")
        or ""
    ).strip()
    return {
        "extra_server_args": sanitize_publish_server_args(args),
        "extra_envs": sanitize_publish_env_mapping(
            _mapping(current.get("extra_envs"))
        ),
    }


def _entry_files(entries: list[dict[str, Any]], files: _Files, category: str) -> tuple[list[str], list[str]]:
    patches: list[str] = []
    artifacts: list[str] = []
    for entry in entries:
        try:
            stack_index = int(entry.get("__stack_index", -1))
        except (TypeError, ValueError):
            stack_index = -1
        patch_member = 0
        seen_patch_sources: set[str] = set()

        def add_value(raw: Any, *, kind: str) -> str:
            nonlocal patch_member
            if kind != "patches" or stack_index < 0:
                return files.add(raw, category=category, kind=kind)
            source = Path(str(raw or ""))
            if not source.is_file():
                return ""
            source_key = str(source.resolve())
            if source_key in seen_patch_sources:
                return ""
            seen_patch_sources.add(source_key)
            stem = source.name
            for suffix in (".patch", ".diff"):
                if stem.lower().endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            safe_name = _OVERLAY_NAME_RE.sub("-", stem).strip("._-") or "patch"
            rel = (
                f"{category}/overlays/{stack_index:06d}/"
                f"{patch_member:02d}-{safe_name}.patch"
            )
            patch_member += 1
            return files.adopt(source, rel)

        for key in _PATH_KEYS:
            raw = entry.get(key)
            if not raw:
                continue
            kind = "patches" if "patch" in key else "artifacts"
            ref = add_value(raw, kind=kind)
            if ref:
                (patches if kind == "patches" else artifacts).append(ref)
        for key in _PATH_LIST_KEYS:
            raw_values = entry.get(key) or []
            if isinstance(raw_values, (str, Path)):
                raw_values = [raw_values]
            if not isinstance(raw_values, (list, tuple, set)):
                continue
            kind = "patches" if "patch" in key else "artifacts"
            for raw in raw_values:
                ref = add_value(raw, kind=kind)
                if ref:
                    (patches if kind == "patches" else artifacts).append(ref)
    return list(dict.fromkeys(patches)), list(dict.fromkeys(artifacts))


def build_explore_value(
    state: Any,
    entries: list[dict[str, Any]],
    files: _Files,
) -> dict[str, Any]:
    """Build EXPLORE-origin patch and artifact references."""
    patches, artifacts = _entry_files(entries, files, "explore")
    return {
        "patches": patches,
        "artifacts": artifacts,
    }


def build_framework_value(
    state: Any,
    entries: list[dict[str, Any]],
    files: _Files,
) -> dict[str, Any]:
    """Build FRAMEWORK-origin patch and artifact references."""
    patches, artifacts = _entry_files(entries, files, "framework")
    return {
        "patches": patches,
        "artifacts": artifacts,
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
                raise RemoteRecipeValidationError(
                    f"accepted {category} {key} cannot be materialized: {original!r}"
                )
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
            if (ref := files.add(
                value,
                category=category,
                kind="patches" if "patch" in key else "artifacts",
            ))
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
    accepted_last = str(last.get("decision") or "").upper() == "KEEP" or str(
        last.get("status") or ""
    ).lower() == "kept"
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
    target_source = stack_rows[-1].get("target_file") or result.get("source_file")
    if not str(patch_source or "").strip() or not str(target_source or "").strip():
        raise RemoteRecipeValidationError(
            "accepted kernel/fusion is missing its patch or target file"
        )
    patch_ref = files.add(patch_source, category="kernel/fusion", kind="patches")
    target_ref = files.add(target_source, category="kernel/fusion", kind="artifacts")
    if not patch_ref or not target_ref:
        raise RemoteRecipeValidationError(
            "accepted kernel/fusion patch or target cannot be materialized: "
            f"patch={patch_source!r} target={target_source!r}"
        )
    record = {
        **result,
        **stack_rows[-1],
        "e2e": _externalize_record(integrated, files, "kernel/fusion"),
        "phase": str(stack_rows[-1].get("phase") or "KERNEL_AGENT"),
        "patch": patch_ref,
        "source_file": target_ref,
    }
    # Remove duplicate local-path aliases after establishing canonical refs.
    record.pop("patch_path", None)
    record.pop("target_file", None)
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
            lambda key, raw: str(
                raw.get("kernel_id") or raw.get("current_kernel_id") or key
            ) == str(integrate.get("kernel_id") or ""),
        ),
        (
            str(integrate.get("patch_path") or ""),
            lambda key, raw: str(raw.get("last_artifact_path") or raw.get("artifact_path") or "")
            == str(integrate.get("patch_path") or ""),
        ),
        (
            str(integrate.get("target_file") or ""),
            lambda key, raw: str(raw.get("last_source_file") or raw.get("source_file") or "")
            == str(integrate.get("target_file") or ""),
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
        source_source = entry.get("target_file") or raw.get("last_source_file") or raw.get("source_file")
        patch = files.add(
            patch_source,
            category="kernel/rewrite",
            kind="patches",
        )
        source = files.add(
            source_source,
            category="kernel/rewrite",
            kind="source",
        )
        if not patch or not source:
            raise RemoteRecipeValidationError(
                "accepted kernel/rewrite patch or source cannot be materialized: "
                f"integration_id={integration_id!r} patch={patch_source!r} "
                f"source={source_source!r}"
            )
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
                    f"- Source: {source or 'unavailable'}",
                    "",
                )
            ),
            rel=f"kernel/rewrite/experience/{slug}.md",
            kind="experience",
        )
        rows.append(
            {
                "id": integration_id
                or str(entry.get("task_group_key") or entry.get("kernel_id") or f"rewrite-{slug}"),
                "phase": "KERNEL_AGENT",
                "kernel_name": kernel_name,
                "speedup": speedup,
                "e2e_gain_pct": e2e_gain,
                "optimized_throughput": optimized_throughput,
                "experience_document": experience,
                "patch": patch,
                "source_files": [source] if source else [],
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
                "name": str(
                    entry.get("variant_name")
                    or entry.get("kernel_name")
                    or entry.get("kernel_id")
                    or action
                ),
                "action": action,
                "phase": str(entry.get("source_phase") or ""),
                "gain_pct": gain,
            }
        )
    return rows


def has_new_keep(state: Any) -> bool:
    """True when the promoted KEEP-only stack has a non-replay entry.

    ``optimization_stack`` is the accepted stack, not the attempt ledger;
    individual rows therefore do not carry a redundant KEEP decision.
    """
    for raw in getattr(state, "optimization_stack", []) or []:
        if not isinstance(raw, Mapping):
            continue
        action = str(raw.get("action") or "").strip().lower()
        if action not in _IGNORED_ACTIONS:
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
    replayed_refs = {
        str(ref)
        for ref in (outcome.get("replayed_patch_refs") or [])
        if str(ref)
    }
    if not replayed_refs:
        return
    warm_root = getattr(sections, "warm_start_dir", None)
    if warm_root is None:
        raise RemoteRecipeValidationError(
            "replayed prior overlays have no warm-start artifact root"
        )
    warm_root = Path(warm_root)
    recipe_path = warm_root / "recipe.json"
    if not recipe_path.is_file():
        raise RemoteRecipeValidationError(
            "replayed prior overlays are missing warm-start recipe.json"
        )
    try:
        import json

        document = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RemoteRecipeValidationError(
            f"cannot read replayed prior recipe: {exc}"
        ) from exc
    prior_value = _mapping(document.get("value"))
    if not prior_value:
        knowledge = _mapping(document.get("knowledge"))
        prior_value = _mapping(knowledge.get("value"))

    replay_index = next(
        (
            index
            for index, entry in enumerate(stack)
            if str(entry.get("action") or "").lower() == "replay_warm_recipe"
        ),
        -1,
    )
    if replay_index < 0:
        raise RemoteRecipeValidationError(
            "replayed prior overlays have no replay_warm_recipe stack entry"
        )
    prior_timeline = prior_value.get("patch_timeline")
    candidates: list[tuple[str, str]] = []
    if isinstance(prior_timeline, list):
        for row in prior_timeline:
            ref = str(row or "")
            owner = ref.split("/", 1)[0].lower()
            if owner in {"explore", "framework"} and ref in replayed_refs:
                candidates.append((owner, ref))
    if not candidates:
        for owner in ("explore", "framework"):
            for ref in _mapping(prior_value.get(owner)).get("patches") or []:
                if str(ref) in replayed_refs:
                    candidates.append((owner, str(ref)))
    candidate_refs = {ref for _owner, ref in candidates}
    missing_metadata = replayed_refs - candidate_refs
    if missing_metadata:
        raise RemoteRecipeValidationError(
            "successfully replayed prior overlays are absent from prior "
            f"knowledge: {sorted(missing_metadata)!r}"
        )

    member_index = 0
    seen: set[str] = set()
    files_root = warm_root / "files"
    if files_root.is_symlink():
        raise RemoteRecipeValidationError(
            "replayed prior files root must not be a symlink"
        )
    resolved_root = files_root.resolve()
    for owner, old_ref in candidates:
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
                    "successfully replayed prior overlay resolves through a "
                    f"symlink: {old_ref!r}"
                )
        try:
            source.resolve().relative_to(resolved_root)
        except ValueError as exc:
            raise RemoteRecipeValidationError(
                f"replayed prior overlay escapes files root: {old_ref!r}"
            ) from exc
        if not source.is_file():
            raise RemoteRecipeValidationError(
                f"successfully replayed prior overlay is missing: {old_ref!r}"
            )
        try:
            source.read_bytes()
        except OSError as exc:
            raise RemoteRecipeValidationError(
                f"cannot read successfully replayed prior overlay {old_ref!r}: {exc}"
            ) from exc
        match = _OVERLAY_REF_RE.match(old_ref)
        old_name = match.group(4) if match else source.stem
        safe_name = _OVERLAY_NAME_RE.sub("-", old_name).strip("._-") or "patch"
        new_ref = (
            f"{owner}/overlays/{replay_index:06d}/"
            f"{member_index:02d}-{safe_name}.patch"
        )
        try:
            files.adopt(source, new_ref)
        except (OSError, ValueError) as exc:
            raise RemoteRecipeValidationError(
                f"cannot adopt successfully replayed prior overlay {old_ref!r}: {exc}"
            ) from exc
        node = _mapping(value.get(owner))
        refs = [str(ref) for ref in (node.get("patches") or []) if str(ref)]
        if new_ref not in refs:
            refs.append(new_ref)
        node["patches"] = refs
        value[owner] = node
        member_index += 1


def _patch_timeline(
    value: Mapping[str, Any],
    _stack: list[dict[str, Any]],
) -> list[str]:
    """Build the global replay order from section overlay refs."""
    rows: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    for owner in ("explore", "framework"):
        node = _mapping(value.get(owner))
        for raw_ref in node.get("patches") or []:
            ref = str(raw_ref or "")
            match = _OVERLAY_REF_RE.match(ref)
            if match is None or ref in seen:
                continue
            seen.add(ref)
            stack_index = int(match.group(2))
            member_index = int(match.group(3))
            rows.append((stack_index, member_index, owner, ref))
    rows.sort()
    return [ref for _stack_index, _member_index, _owner, ref in rows]


def _adopt_replayed_kernel(
    state: Any,
    sections: Any,
    value: dict[str, Any],
    files: "_Files",
    stack: list[dict[str, Any]],
) -> None:
    """Carry the kernel rows accepted by the combined Recipe replay forward."""
    outcome = _mapping(getattr(state, "warm_replay_outcome", {}))
    kernel_outcome = _mapping(outcome.get("kernel"))
    if (
        str(outcome.get("status") or "") != "reproduced"
        or str(kernel_outcome.get("status") or "") != "kept"
        or not any(_mapping(item.get("kernel_replay")) for item in stack)
    ):
        return
    plan = [
        dict(entry)
        for entry in (getattr(state, "warm_kernel_kb_plan", []) or [])
        if isinstance(entry, Mapping)
    ]
    if not plan:
        raise RemoteRecipeValidationError(
            "combined replay kept kernel content without a carry-forward plan"
        )
    prior = sections.read("kernel")
    if prior is None:
        raise RemoteRecipeValidationError(
            "combined replay kept kernel content but the inference Recipe "
            "has no kernel section"
        )
    files_root = Path(sections.warm_start_dir) / "files"
    prior_paths = {
        source.relative_to(files_root).as_posix()
        for source in prior.files
        if source.is_file() and not source.is_symlink()
    }
    adopted: set[str] = set()
    rows_by_column: dict[str, list[dict[str, Any]]] = {
        "gemm": [],
        "fusion": [],
        "rewrite": [],
    }
    for entry in plan:
        column = str(entry.get("column") or "")
        row = entry.get("recipe_row")
        if column not in rows_by_column or not isinstance(row, Mapping):
            raise RemoteRecipeValidationError(
                "combined replay kernel plan cannot reproduce its exact Recipe row"
            )
        copied_row = dict(row)
        refs = extract_knowledge_artifact_refs(copied_row, prior_paths)
        for ref in sorted(refs):
            source = files_root / validate_relative_path(ref)
            if ref not in prior_paths:
                raise RemoteRecipeValidationError(
                    f"combined replay kernel artifact is missing: {ref!r}"
                )
            if ref not in adopted:
                files.validate_adoption(source, ref)
                files.adopt(source, ref)
                adopted.add(ref)
        rows_by_column[column].append(copied_row)

    kernel = _mapping(value.get("kernel"))
    list_keys = {
        "gemm": "optimizations",
        "fusion": "items",
        "rewrite": "items",
    }
    for column, accepted_rows in rows_by_column.items():
        if not accepted_rows:
            continue
        list_key = list_keys[column]
        prior_column = _mapping(prior.knowledge.get(column))
        current_column = _mapping(kernel.get(column))
        combined_rows = [
            *(
                current_column.get(list_key)
                if isinstance(current_column.get(list_key), list)
                else []
            ),
            *accepted_rows,
        ]
        kernel[column] = {
            **{
                key: item
                for key, item in prior_column.items()
                if key not in {list_key, "files"}
            },
            **current_column,
            list_key: combined_rows,
        }
    value["kernel"] = kernel


def merge_staged_sections(
    value: dict[str, Any],
    sections: Any,
    files: "_Files",
    *,
    only: Collection[str] | None = None,
    required: Collection[str] | None = None,
) -> list[str]:
    """Overlay agent-staged sections onto the values scraped from the stack.

    An agent that stages a section owns the keys it wrote; keys it left alone
    keep whatever the stack scrape produced. That is what lets a section-aware
    agent and a not-yet-migrated one publish into the same document.

    ``only`` restricts the overlay to named sections. A document that carries
    one agent's columns must not adopt another's files, which would land as
    artifacts nothing in the knowledge references.
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
                raise RemoteRecipeValidationError(
                    f"{qualifier}staged section {name!r} cannot be read"
                )
            if not staged.knowledge:
                qualifier = "required " if name in required_names else ""
                raise RemoteRecipeValidationError(
                    f"{qualifier}staged section {name!r} is empty"
                )
            staged_files = [
                (
                    source,
                    source.relative_to(sections.files_dir).as_posix(),
                )
                for source in staged.files
                if source.is_file() and not source.is_symlink()
            ]
            staged_paths = {rel for _, rel in staged_files}
            staged_refs = (
                {
                    str(ref)
                    for key in ("patches", "artifacts")
                    for ref in (staged.knowledge.get(key) or [])
                    if str(ref).strip()
                }
                if name in required_names
                else extract_knowledge_artifact_refs(
                    staged.knowledge,
                    staged_paths,
                )
            )
            missing = staged_refs - staged_paths
            orphaned = staged_paths - staged_refs
            if missing or orphaned:
                raise RemoteRecipeValidationError(
                    f"staged section {name!r} file mismatch: "
                    f"missing={sorted(missing)!r} orphaned={sorted(orphaned)!r}"
                )
            for source, rel in staged_files:
                files.validate_adoption(source, rel)
            for source, rel in staged_files:
                files.adopt(source, rel)
            current = value.get(name)
            combined = {
                "patches": (
                    list(current.get("patches") or [])
                    if isinstance(current, Mapping)
                    else []
                ),
                "artifacts": (
                    list(current.get("artifacts") or [])
                    if isinstance(current, Mapping)
                    else []
                ),
            }
            # Owner sections contain patch/artifact refs only.
            for ref_key in ("patches", "artifacts"):
                before = (
                    list(current.get(ref_key) or [])
                    if isinstance(current, Mapping)
                    else []
                )
                after = list(staged.knowledge.get(ref_key) or [])
                if before or after:
                    combined[ref_key] = list(
                        dict.fromkeys(
                            str(ref) for ref in [*before, *after] if str(ref)
                        )
                    )
            value[name] = combined
            merged.append(name)
        except Exception as exc:
            if isinstance(exc, RemoteRecipeValidationError):
                raise
            qualifier = "required " if name in required_names else ""
            raise RemoteRecipeValidationError(
                f"{qualifier}staged section {name!r} is invalid: {exc}"
            ) from exc
    return merged


def build_remote_knowledge(
    state: Any,
    files_dir: str | Path,
    *,
    sections: Any = None,
) -> KnowledgeBundle:
    """Construct the final opaque knowledge document and temporary files tree."""
    pending_sections = list(
        getattr(state, "kb_stage_outbox", []) or []
    )
    if pending_sections:
        raise RemoteRecipeValidationError(
            "required section staging is incomplete: "
            f"{[row.get('id') for row in pending_sections if isinstance(row, Mapping)]!r}"
        )
    root = Path(files_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = _Files(root)
    stack = [
        {**dict(item), "__stack_index": index}
        for index, item in enumerate(getattr(state, "optimization_stack", []) or [])
        if isinstance(item, Mapping)
    ]
    owner_names = {
        "EXPLORE": "explore",
        "FRAMEWORK_AGENT": "framework",
    }
    required_patch_owners = {
        owner_names[owner]
        for item in stack
        if (
            owner := str(item.get("kb_required_owner") or "").upper()
        ) in owner_names
    }
    explore_entries = [item for item in stack if _entry_origin(item) == "explore"]
    framework_entries = [item for item in stack if _entry_origin(item) == "framework"]
    current_best = _mapping(getattr(state, "current_best", {}))
    optimized_throughput = _number(current_best.get("tput"))
    validated_gain = _number(
        getattr(state, "cumulative_gain_validated", 0.0)
        or getattr(state, "cumulative_gain", 0.0)
    )
    gains = list(getattr(state, "gain_per_stack_entry", []) or [])
    worked = _experience(state, "what_worked") or _worked_from_stack(stack, gains)
    if sections is None:
        # Legacy callers without section staging retain their stack-derived
        # owner snapshots; remote current records use value.config below.
        explore_value = build_explore_value(state, explore_entries, files)
        framework_value = build_framework_value(state, framework_entries, files)
        explore_value.update(_config_from(explore_entries))
        framework_value.update(_config_from(framework_entries))
    else:
        explore_value = {"patches": [], "artifacts": []}
        framework_value = {"patches": [], "artifacts": []}
    value = {
        "config": _config_from_current_best(state),
        "explore": explore_value,
        "framework": framework_value,
        "kernel": {
            "gemm": build_kernel_gemm_value(state, files),
            "fusion": build_kernel_fusion_value(state, files),
            "rewrite": build_kernel_rewrite_value(state, files),
        },
    }
    if sections is not None:
        _adopt_replayed_prior(state, sections, value, files, stack)
        _adopt_replayed_kernel(state, sections, value, files, stack)
    staged_sections = (
        merge_staged_sections(
            value,
            sections,
            files,
            only=("explore", "framework"),
            required=required_patch_owners,
        )
        if sections is not None
        else []
    )
    missing_required_owners = required_patch_owners - set(staged_sections)
    if missing_required_owners:
        raise RemoteRecipeValidationError(
            "required staged owner sections are missing: "
            f"{sorted(missing_required_owners)!r}"
        )
    value["patch_timeline"] = _patch_timeline(value, stack)
    knowledge = sanitize_shared_knowledge(
        {
            "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
            "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
            "optimized_throughput": optimized_throughput,
            "validated_e2e_gain": validated_gain,
            "value": value,
            "what_worked": worked,
            "what_failed": _experience(state, "last_action_failures"),
            "remaining_gaps": _experience(state, "gaps"),
            "lessons": _experience(state, "warm_start_lessons"),
            "pitfalls": _experience(state, "warm_start_pitfalls"),
            "provenance": {
                "producer": "hyperloom-inference-optimizer",
                "phase": "CLOSE",
                "session_id": str(
                    getattr(state, "recipe_kb_session_id", "")
                    or getattr(state, "session_id", "")
                ),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "optimization_stack_length": len(stack),
                "staged_sections": staged_sections,
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
    config = _mapping(value.get("config"))
    if (
        str(config.get("extra_server_args") or "").strip()
        or _mapping(config.get("extra_envs"))
    ):
        return True
    # Legacy records stored config under owner sections.
    for section_name in ("explore", "framework"):
        section = _mapping(value.get(section_name))
        if not section:
            continue
        if str(section.get("extra_server_args") or "").strip():
            return True
        envs = section.get("extra_envs")
        if isinstance(envs, Mapping) and envs:
            return True
        patches = section.get("patches")
        if isinstance(patches, list) and patches:
            return True
    timeline = value.get("patch_timeline")
    if isinstance(timeline, list) and timeline:
        return True

    def _nonempty(raw: Any) -> bool:
        if isinstance(raw, Mapping):
            return any(_nonempty(item) for item in raw.values())
        if isinstance(raw, (list, tuple, set)):
            return any(_nonempty(item) for item in raw)
        return bool(raw)

    return _nonempty(value.get("kernel"))


def knowledge_to_warm_recipe(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and project the current Hyperloom Recipe contract for PRELUDE."""
    knowledge = _mapping(document.get("knowledge")) or dict(document)
    version = knowledge.get("knowledge_schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != CURRENT_KNOWLEDGE_SCHEMA_VERSION
    ):
        raise RemoteRecipeValidationError(
            "record knowledge_schema_version does not match the current "
            f"Hyperloom contract ({CURRENT_KNOWLEDGE_SCHEMA_VERSION})"
        )
    if knowledge.get("record_kind") != RECORD_KIND_HYPERLOOM_RECIPE:
        raise RemoteRecipeValidationError(
            f"record_kind must be {RECORD_KIND_HYPERLOOM_RECIPE!r}"
        )
    raw_value = knowledge.get("value")
    if not isinstance(raw_value, Mapping):
        raise RemoteRecipeValidationError("current Recipe is missing value")
    value = dict(raw_value)
    for section in ("explore", "framework", "kernel"):
        if not isinstance(value.get(section), Mapping):
            raise RemoteRecipeValidationError(
                f"current Recipe is missing value.{section}"
            )
    raw_timeline = value.get("patch_timeline")
    if not isinstance(raw_timeline, list) or not all(
        isinstance(ref, str) for ref in raw_timeline
    ):
        raise RemoteRecipeValidationError(
            "current Recipe value.patch_timeline must be a flat string list"
        )
    for ref in raw_timeline:
        validate_relative_path(ref)
    session_id = str(document.get("session_id") or "")
    validated_gain = _number(knowledge.get("validated_e2e_gain"))
    view = _mapping(document.get("view"))
    replayable = (
        bool(view.get("replayable"))
        if isinstance(view.get("replayable"), bool)
        else True
    )
    return {
        "canonical_id": str(document.get("canonical_id") or ""),
        "best_throughput": _number(knowledge.get("optimized_throughput")),
        "validated_gain_pct": validated_gain,
        "what_worked": list(knowledge.get("what_worked") or []),
        "what_failed": list(knowledge.get("what_failed") or []),
        "remaining_gaps": list(knowledge.get("remaining_gaps") or []),
        "lessons": list(knowledge.get("lessons") or []),
        "pitfalls": list(knowledge.get("pitfalls") or []),
        "sessions": (
            [{"session_id": session_id, "gain_pct": validated_gain}]
            if session_id
            else []
        ),
        "provenance": _mapping(knowledge.get("provenance")),
        "remote_session_id": session_id,
        "remote_schema_version": int(document.get("schema_version") or 2),
        "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
        "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
        "view": view,
        "view_source": str(view.get("source") or "current"),
        "replayable": replayable,
        "replay_material_available": (
            replayable and has_replay_material(document)
        ),
        "replay_disabled_reason": str(
            view.get("replay_disabled_reason") or ""
        ),
    }


__all__ = [
    "CURRENT_KNOWLEDGE_SCHEMA_VERSION",
    "RECORD_KIND_HYPERLOOM_RECIPE",
    "build_explore_value",
    "build_framework_value",
    "build_kernel_fusion_value",
    "build_kernel_gemm_value",
    "build_kernel_rewrite_value",
    "build_remote_knowledge",
    "has_replay_material",
    "knowledge_to_warm_recipe",
    "has_new_keep",
    "match_rewrite_attempt",
    "merge_staged_sections",
]
