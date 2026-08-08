# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Final-state value builders for Remote Recipe KB V2."""

from __future__ import annotations

import hashlib
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .models import MAX_FILE_BYTES, Artifact, KnowledgeBundle, RemoteRecipeValidationError

_PATH_KEYS = {
    "artifact_path",
    "final_report_path",
    "patch",
    "patch_path",
    "report_path",
    "source_file",
    "target_file",
    "tuned_file",
}
_PATH_LIST_KEYS = {
    "artifact_files",
    "artifacts",
    "changed_files",
    "patches",
    "patches_applied",
    "source_files",
    "target_files",
}
_IGNORED_ACTIONS = {"replay_warm_recipe", "profile", "roofline", "conc_sweep", "sweep"}


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

    def write(self, text: str, *, rel: str, kind: str) -> str:
        destination = self.root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        self.artifacts.append(Artifact(path=rel, source=destination, kind=kind))
        self.refs.add(rel)
        return rel


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
    return {"extra_server_args": args, "extra_envs": {str(k): str(v) for k, v in envs.items()}}


def _entry_files(entries: list[dict[str, Any]], files: _Files, category: str) -> tuple[list[str], list[str]]:
    patches: list[str] = []
    artifacts: list[str] = []
    for entry in entries:
        for key in _PATH_KEYS:
            raw = entry.get(key)
            if not raw:
                continue
            kind = "patches" if "patch" in key else "artifacts"
            ref = files.add(raw, category=category, kind=kind)
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
                ref = files.add(raw, category=category, kind=kind)
                if ref:
                    (patches if kind == "patches" else artifacts).append(ref)
    return list(dict.fromkeys(patches)), list(dict.fromkeys(artifacts))


def build_explore_value(
    state: Any,
    entries: list[dict[str, Any]],
    files: _Files,
) -> dict[str, Any]:
    """Build final cumulative EXPLORE config and EXPLORE-origin file references."""
    patches, artifacts = _entry_files(entries, files, "explore")
    return {
        **_config_from(entries),
        "patches": patches,
        "artifacts": artifacts,
    }


def build_framework_value(
    state: Any,
    entries: list[dict[str, Any]],
    files: _Files,
) -> dict[str, Any]:
    """Build final FRAMEWORK config/env and FRAMEWORK-origin file references."""
    patches, artifacts = _entry_files(entries, files, "framework")
    return {
        **_config_from(entries),
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


def build_kernel_geak_value(state: Any, files: _Files) -> dict[str, Any]:
    """Build only same-harness E2E-validated GEAK winners and their artifacts."""
    stack_rows = [
        dict(item)
        for item in (getattr(state, "optimization_stack", []) or [])
        if isinstance(item, Mapping) and str(item.get("action") or "").lower() == "geak_e2e"
    ]
    if not stack_rows:
        return {"items": []}
    result = _mapping(getattr(state, "geak_result", {}))
    if str(result.get("status") or "").lower() != "ok":
        result = {}
    current_best = _mapping(getattr(state, "current_best", {}))
    baseline = _number(getattr(state, "baseline_tput", 0.0))
    rows: list[dict[str, Any]] = []
    for entry in stack_rows:
        optimized_throughput = _number(entry.get("tput"))
        gain = (
            (optimized_throughput - baseline) / baseline * 100.0
            if baseline > 0 and optimized_throughput > 0
            else 0.0
        )

        def add_required_file(source: Any, *, kind: str) -> str:
            if not str(source or "").strip():
                return ""
            ref = files.add(source, category="kernel/geak", kind=kind)
            if not ref:
                raise RemoteRecipeValidationError(
                    f"accepted kernel/geak {kind} cannot be materialized: {source!r}"
                )
            return ref

        report = add_required_file(
            entry.get("report_path") or result.get("report_path"),
            kind="reports",
        )
        launch_script = add_required_file(
            current_best.get("geak_launch_script") or result.get("final_launch_script"),
            kind="launch",
        )
        bench_script = add_required_file(
            current_best.get("geak_bench_script") or result.get("bench_script"),
            kind="launch",
        )
        overlay_source = (
            entry.get("final_overlay")
            or current_best.get("final_overlay")
            or result.get("final_overlay")
        )
        overlay_files: list[str] = []
        if str(overlay_source or "").strip():
            overlay_root = Path(str(overlay_source))
            nested_overlay = overlay_root / "overlay"
            if overlay_root.is_dir() and nested_overlay.is_dir():
                overlay_root = nested_overlay
            overlay_files = files.add_tree(
                overlay_root,
                category="kernel/geak",
                kind="overlay",
            )
            if not overlay_files:
                raise RemoteRecipeValidationError(
                    f"accepted kernel/geak overlay is empty: {overlay_root}"
                )
        row_id = hashlib.sha256(
            (
                f"{entry.get('ts')}|{entry.get('variant_name')}|"
                f"{optimized_throughput}|{overlay_source}"
            ).encode()
        ).hexdigest()[:16]
        rows.append(
            {
                "id": f"geak-{row_id}",
                "optimized_throughput": optimized_throughput,
                "e2e_gain_pct": gain,
                "extra_server_args": str(
                    entry.get("candidate_extra_server_args")
                    or entry.get("extra_server_args")
                    or ""
                ),
                "extra_envs": {
                    str(key): str(value)
                    for key, value in _mapping(entry.get("extra_envs")).items()
                },
                "accepted_kernels": list(entry.get("accepted_kernels") or []),
                "accepted_heads": list(entry.get("accepted_heads") or []),
                "self_reported_speedup": _number(result.get("throughput_speedup")),
                "alignment": _mapping(current_best.get("geak_alignment")),
                "report": report,
                "launch_script": launch_script,
                "bench_script": bench_script,
                "overlay_files": overlay_files,
            }
        )
    return {"items": rows}


def _match_rewrite_attempt(
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
        raw = _match_rewrite_attempt(entry, attempts)
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
    """True only when the final stack contains a non-replay, meaningful KEEP."""
    for raw in getattr(state, "optimization_stack", []) or []:
        if not isinstance(raw, Mapping):
            continue
        action = str(raw.get("action") or "").strip().lower()
        if action not in _IGNORED_ACTIONS:
            return True
    return False


def build_remote_knowledge(state: Any, files_dir: str | Path) -> KnowledgeBundle:
    """Construct the final opaque knowledge document and temporary files tree."""
    root = Path(files_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = _Files(root)
    stack = [dict(item) for item in (getattr(state, "optimization_stack", []) or []) if isinstance(item, Mapping)]
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
    knowledge = {
        "knowledge_schema_version": 2,
        "optimized_throughput": optimized_throughput,
        "validated_e2e_gain": validated_gain,
        "value": {
            "explore": build_explore_value(state, explore_entries, files),
            "framework": build_framework_value(state, framework_entries, files),
            "kernel": {
                "geak": build_kernel_geak_value(state, files),
                "gemm": build_kernel_gemm_value(state, files),
                "fusion": build_kernel_fusion_value(state, files),
                "rewrite": build_kernel_rewrite_value(state, files),
            },
        },
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
        },
    }
    bundle = KnowledgeBundle(knowledge=knowledge, artifacts=files.artifacts)
    bundle.validate(files.refs)
    return bundle


def convert_v1_recipe_to_knowledge(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap one legacy RecipeKB row for lossless storage in KB Store."""
    legacy = dict(recipe)
    best_config = _mapping(legacy.get("best_config"))
    if best_config:
        legacy["best_config"] = {
            "extra_server_args": str(
                best_config.get("extra_server_args")
                or best_config.get("args")
                or ""
            ),
            "extra_envs": _mapping(
                best_config.get("extra_envs") or best_config.get("envs")
            ),
        }
    return {
        "knowledge_schema_version": 1,
        "optimized_throughput": _number(recipe.get("best_throughput")),
        "validated_e2e_gain": _number(recipe.get("validated_gain_pct") or recipe.get("gain_pct")),
        "value": {
            "legacy_recipe": legacy,
        },
        "provenance": {
            "producer": "hyperloom-v1-converter",
            "source_schema": 1,
        },
    }


def envelope_to_v1_recipe(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Project remote knowledge into the stable config-only warm Recipe shape."""
    knowledge = _mapping(envelope.get("knowledge")) or dict(envelope)
    value = _mapping(knowledge.get("value"))
    raw_version = knowledge.get("knowledge_schema_version")
    if raw_version is None:
        raw_version = 1 if "best_config" in knowledge or "legacy_recipe" in value else 2
    try:
        knowledge_version = int(raw_version)
    except (TypeError, ValueError):
        knowledge_version = 0
    if knowledge_version == 1:
        legacy = _mapping(value.get("legacy_recipe")) or knowledge
        row = dict(legacy)
        row["canonical_id"] = str(
            envelope.get("canonical_id") or row.get("canonical_id") or ""
        )
        row.setdefault("prs_tested", [])
        row["remote_session_id"] = str(envelope.get("session_id") or "")
        row["remote_schema_version"] = int(envelope.get("schema_version") or 2)
        row["knowledge_schema_version"] = 1
        return row
    if knowledge_version != 2:
        raise RemoteRecipeValidationError(
            f"unsupported knowledge_schema_version: {raw_version!r}"
        )
    explore = _mapping(value.get("explore"))
    explore_args = str(explore.get("extra_server_args") or "").strip()
    explore_envs = {
        str(key): str(value)
        for key, value in _mapping(explore.get("extra_envs")).items()
    }
    session_id = str(envelope.get("session_id") or "")
    validated_gain = _number(knowledge.get("validated_e2e_gain"))
    return {
        "canonical_id": str(envelope.get("canonical_id") or ""),
        "best_config": {
            "extra_server_args": explore_args,
            "extra_envs": explore_envs,
        },
        "best_throughput": _number(knowledge.get("optimized_throughput")),
        "validated_gain_pct": validated_gain,
        "what_worked": list(knowledge.get("what_worked") or []),
        "what_failed": list(knowledge.get("what_failed") or []),
        "remaining_gaps": list(knowledge.get("remaining_gaps") or []),
        "lessons": list(knowledge.get("lessons") or []),
        "pitfalls": list(knowledge.get("pitfalls") or []),
        # Phase 1 intentionally replays config/env only. Omitting historical PR
        # payloads keeps the unchanged replay executor out of its patch path.
        "prs_tested": [],
        "sessions": (
            [{"session_id": session_id, "gain_pct": validated_gain}]
            if session_id
            else []
        ),
        "provenance": _mapping(knowledge.get("provenance")),
        "remote_session_id": session_id,
        "remote_schema_version": int(envelope.get("schema_version") or 2),
        "knowledge_schema_version": 2,
    }


__all__ = [
    "build_explore_value",
    "build_framework_value",
    "build_kernel_fusion_value",
    "build_kernel_geak_value",
    "build_kernel_gemm_value",
    "build_kernel_rewrite_value",
    "build_remote_knowledge",
    "convert_v1_recipe_to_knowledge",
    "envelope_to_v1_recipe",
    "has_new_keep",
]
