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

from hyperloom.common.gpu_identity import gfx_arch_for_gpu_type
from hyperloom.common.platform_probe import platform_fingerprint
from hyperloom.inference_optimizer.breakdown.agent_ownership import (
    SECTION_BY_LEVER,
    patch_lever_kind,
    patch_owner_phase,
)
from .models import (
    MAX_FILE_BYTES,
    Artifact,
    KnowledgeBundle,
    RecipeScope,
    RemoteRecipeValidationError,
    extract_knowledge_artifact_refs,
    validate_relative_path,
)
from .sanitize import (
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
_OVERLAY_REF_RE = re.compile(r"^(explore|framework)/overlays/(\d{6})/(\d+)-([^/]+)\.patch$")
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


#: Snapshot keys that are host-local and must never enter the shared Recipe.
_ROOFLINE_PATH_KEYS = frozenset(
    {
        "analysis_md_path",
        "kernel_roofline_path",
        "trace_input",
        "workspace",
        "workspace_path",
    }
)
_ROOFLINE_ARM_NUMBERS = (
    "achieved_tok_per_sec",
    "theoretical_peak_tok_per_sec",
    "roofline_mem_ceiling_tok_per_sec",
    "roofline_cmp_ceiling_tok_per_sec",
    "within_roofline_pct",
    "gap_to_roofline_pct",
    "within_roofline_pct_uncapped",
    "e2e_mean_ms",
    "roofline_ideal_ms",
    "compute_pct",
    "idle_pct",
    "comm_pct",
)
_ROOFLINE_TOP_KERNEL_KEYS = ("name", "gpu_pct", "efficiency_pct", "bound_type")


def _finite_or_none(value: Any) -> float | None:
    """Return a finite float, or ``None`` when the input is missing/non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _publishable_roofline_arm(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Project one snapshot into the path-free arm the miner reads."""
    arm: dict[str, Any] = {}
    for key in _ROOFLINE_ARM_NUMBERS:
        number = _finite_or_none(snapshot.get(key))
        if number is not None:
            arm[key] = number
    exceeded = snapshot.get("roofline_ceiling_exceeded")
    if isinstance(exceeded, bool):
        arm["roofline_ceiling_exceeded"] = exceeded
    bound = str(snapshot.get("roofline_bound_kind") or "").strip()
    if bound:
        arm["roofline_bound_kind"] = bound
    bottleneck = str(snapshot.get("top_bottleneck") or "").strip()
    if bottleneck:
        arm["top_bottleneck"] = bottleneck
    raw_kernel = snapshot.get("top_kernel")
    if isinstance(raw_kernel, Mapping):
        kernel: dict[str, Any] = {}
        name = str(raw_kernel.get("name") or "").strip()
        if name:
            kernel["name"] = name
        for key in _ROOFLINE_TOP_KERNEL_KEYS[1:]:
            number = _finite_or_none(raw_kernel.get(key))
            if number is not None:
                kernel[key] = number
            else:
                text = str(raw_kernel.get(key) or "").strip()
                if key == "bound_type" and text:
                    kernel[key] = text
        if kernel:
            arm["top_kernel"] = kernel
    return arm


def _roofline_arm_usable(arm: Mapping[str, Any]) -> bool:
    """True when an arm can support a roof or a measured point."""
    peak = _finite_or_none(arm.get("theoretical_peak_tok_per_sec")) or 0.0
    ideal = _finite_or_none(arm.get("roofline_ideal_ms")) or 0.0
    achieved = _finite_or_none(arm.get("achieved_tok_per_sec")) or 0.0
    e2e = _finite_or_none(arm.get("e2e_mean_ms")) or 0.0
    return (peak > 0 or ideal > 0) and (achieved > 0 or e2e > 0)


def build_publishable_roofline(state: Any) -> dict[str, Any] | None:
    """Compact baseline/optimized roofline pair for Recipe KB mining.

    Prefers the session's TraceLens/bypass snapshot history. Falls back to the
    CPU-only ``baseline_roofline_ceiling`` when no trace snapshot is usable.
    Returns ``None`` rather than a partial roof so miners do not treat a missing
    measurement as zero headroom. Host paths are dropped here; sanitizer is a
    second net, not the contract.
    """
    snapshots = [item for item in (getattr(state, "roofline_snapshots", None) or []) if isinstance(item, Mapping)]
    ceiling = getattr(state, "baseline_roofline_ceiling", None)
    source = "trace"
    if not snapshots:
        if isinstance(ceiling, Mapping) and ceiling:
            snapshots = [ceiling]
            source = "analytic_backup"
        else:
            return None

    baseline_arm = _publishable_roofline_arm(snapshots[0])
    optimized_snap = snapshots[-1] if snapshots else None
    optimized_arm = _publishable_roofline_arm(optimized_snap) if optimized_snap is not None else {}
    if not _roofline_arm_usable(baseline_arm):
        if source == "trace" and isinstance(ceiling, Mapping) and ceiling:
            baseline_arm = _publishable_roofline_arm(ceiling)
            source = "analytic_backup"
            optimized_arm = {}
        if not _roofline_arm_usable(baseline_arm):
            return None

    peak_src = optimized_arm if _roofline_arm_usable(optimized_arm) else baseline_arm
    record: dict[str, Any] = {
        "source": source,
        "baseline": baseline_arm,
    }
    unit = str(
        (optimized_snap or {}).get("throughput_unit")
        or snapshots[0].get("throughput_unit")
        or ""
    ).strip()
    if unit:
        record["throughput_unit"] = unit
    bound = str(
        (optimized_snap or {}).get("roofline_bound_kind")
        or snapshots[0].get("roofline_bound_kind")
        or peak_src.get("roofline_bound_kind")
        or ""
    ).strip()
    if bound and bound != "unknown":
        record["bound_kind"] = bound
    peak = _finite_or_none(peak_src.get("theoretical_peak_tok_per_sec"))
    if peak is not None and peak > 0:
        record["theoretical_peak_tok_per_sec"] = peak
    mem = _finite_or_none(peak_src.get("roofline_mem_ceiling_tok_per_sec"))
    if mem is not None and mem > 0:
        record["roofline_mem_ceiling_tok_per_sec"] = mem
    cmp = _finite_or_none(peak_src.get("roofline_cmp_ceiling_tok_per_sec"))
    if cmp is not None and cmp > 0:
        record["roofline_cmp_ceiling_tok_per_sec"] = cmp
    # A lone snapshot fills both arms from the same measurement, so publishing
    # an optimized arm would claim a post-optimization profile that never ran
    # and hand miners a zero baseline-to-optimized delta.
    if _roofline_arm_usable(optimized_arm) and source == "trace" and len(snapshots) > 1:
        record["optimized"] = optimized_arm
    for key in _ROOFLINE_PATH_KEYS:
        record.pop(key, None)
        baseline_arm.pop(key, None)
        optimized_arm.pop(key, None)
    return record


def _session_gpu_type(state: Any) -> str:
    """Session board identity, already probe-corrected by the CLI when present."""
    return str(getattr(state, "gpu_type", "") or getattr(state, "runner_type", "") or "").strip().lower()


def _session_multi_node(state: Any) -> bool | None:
    """Whether this session spans multiple nodes, or ``None`` when unknown.

    Mirrors the run-report fingerprint: a True marker means the CPU/GPU sample
    may be the orchestrator rather than the benchmark node.
    """
    raw = getattr(state, "multi_node", None)
    if isinstance(raw, bool):
        return raw
    try:
        from ...actions.executors._multi_node_env import is_multi_node

        return bool(is_multi_node())
    except Exception:  # noqa: BLE001 - unknown is not False
        return None


def _compact_stack(stack: Any) -> dict[str, str]:
    """Keep resolved stack versions; drop ``unknown`` placeholders."""
    if not isinstance(stack, Mapping):
        return {}
    out: dict[str, str] = {}
    for key in ("rocm", "aiter", "sglang", "vllm"):
        value = str(stack.get(key) or "").strip()
        if value and value.lower() != "unknown":
            out[key] = value
    return out


def build_publishable_platform(
    state: Any,
    *,
    fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Host + GPU facts for Recipe KB mining, without node identity.

    Projects the same probe as ``final.json`` (``platform_fingerprint``) into
    ``value.platform``. Hostname is dropped so a shared store cannot inventory
    machines. ``gpu_type`` comes from the session (the KB's hardware dimension)
    and is a stronger gfx source than a PATH-dependent probe.

    Returns ``None`` when neither a CPU sysfs sample nor a session GPU type
    is available, so miners do not treat a missing host as a default EPYC.
    """
    gpu_type = _session_gpu_type(state)
    multi_node = _session_multi_node(state)
    raw = dict(fingerprint) if fingerprint is not None else platform_fingerprint(gpu_type or None, multi_node=multi_node)
    status = str(raw.get("status") or "").strip() or "error"

    cpu: dict[str, Any] = {}
    for key in ("cpu", "smt", "sockets", "numa_nodes", "nps", "governor", "boost", "kernel"):
        value = raw.get(key)
        if value in (None, ""):
            continue
        cpu[key if key != "cpu" else "model"] = value

    gpu_block = raw.get("gpu") if isinstance(raw.get("gpu"), Mapping) else {}
    gfx = str(gpu_block.get("gfx_arch") or "").strip()
    if not gfx or gfx.lower() == "unknown":
        gfx = gfx_arch_for_gpu_type(gpu_type) or ""
    gpu: dict[str, Any] = {}
    if gpu_type:
        gpu["gpu_type"] = gpu_type
    if gfx:
        gpu["gfx_arch"] = gfx
    host_count = gpu_block.get("host_count")
    if isinstance(host_count, int) and not isinstance(host_count, bool) and host_count > 0:
        gpu["host_count"] = host_count
    driver = str(gpu_block.get("amdgpu_driver") or "").strip()
    if driver and driver.lower() != "unknown":
        gpu["amdgpu_driver"] = driver

    record: dict[str, Any] = {"status": status}
    reason = str(raw.get("reason") or "").strip()
    if reason and status != "ok":
        record["reason"] = reason
    if multi_node is not None:
        record["multi_node_session"] = multi_node
    elif raw.get("multi_node_session") is not None:
        record["multi_node_session"] = bool(raw.get("multi_node_session"))
    if cpu:
        record["cpu"] = cpu
    if gpu:
        record["gpu"] = gpu
    stack = _compact_stack(raw.get("stack"))
    if stack:
        record["stack"] = stack
    if "cpu" not in record and "gpu" not in record:
        return None
    return record


_DO_NOT_REPEAT_CAP = 32
_DO_NOT_REPEAT_REASON_MAX = 160
#: Slots held back for kernel skips so a long explore phase cannot crowd them
#: out. Unused reserve flows back to explore rows, and vice versa.
_DO_NOT_REPEAT_KERNEL_RESERVE = _DO_NOT_REPEAT_CAP // 2
_KERNEL_SKIP_ACTIONS = frozenset(
    {
        "fusion",
        "geak_e2e",
        "gemm_tuning",
        "integrate",
        "kernel_opt",
        "rewrite",
    }
)
_FRAMEWORK_SKIP_ACTIONS = frozenset({"integrate_patch", "framework"})


def _reason_line(*parts: Any) -> str:
    """One-line skip reason; empty when the only content looks like a host path."""
    text = " ".join(str(part).strip() for part in parts if part not in (None, ""))
    text = " ".join(text.split())
    if not text or text.startswith("/") or ":\\" in text[:3]:
        return ""
    return text[:_DO_NOT_REPEAT_REASON_MAX]


def _skip_explore_row(row: Mapping[str, Any], *, reason: str = "") -> dict[str, Any] | None:
    """Return a path-free explore skip, or ``None`` when it cannot be fingerprinted.

    A skip list is an optimization, not replay material, so an unsanitizable
    row is dropped rather than raised. Rejected configs are the likeliest place
    to find broken quoting, and losing one skip must never cost the session its
    whole published recipe.
    """
    try:
        args = sanitize_publish_server_args(
            str(row.get("extra_server_args") or row.get("candidate_extra_server_args") or "")
        )
    except ValueError:
        return None
    raw_envs = row.get("extra_envs") or row.get("candidate_extra_envs") or {}
    envs = sanitize_publish_env_mapping(raw_envs if isinstance(raw_envs, Mapping) else {})
    if not args and not envs:
        return None
    from ...actions.executors._canonical_fingerprint import canonical_fingerprint

    item: dict[str, Any] = {
        "kind": "explore",
        "fingerprint": canonical_fingerprint(args, envs),
        "extra_server_args": args,
        "extra_envs": envs,
    }
    error_class = str(row.get("error_class") or row.get("reason") or "").strip()
    if error_class and not error_class.startswith("/"):
        item["error_class"] = error_class[:64]
    name = str(row.get("name") or row.get("variant_name") or "").strip()
    if name:
        item["name"] = name[:120]
    why = _reason_line(reason, error_class, name)
    if why:
        item["reason"] = why
    return item


def _skip_kernel_row(kernel_id: str, *, action: str = "", reason: str = "") -> dict[str, Any] | None:
    kid = str(kernel_id or "").strip()
    if not kid or kid.startswith("/"):
        return None
    item: dict[str, Any] = {"kind": "kernel", "kernel_id": kid[:128]}
    act = str(action or "").strip().lower()
    if act:
        item["action"] = act
    why = _reason_line(reason, act, kid)
    if why:
        item["reason"] = why
    return item


def build_publishable_do_not_repeat(state: Any) -> list[dict[str, Any]]:
    """Compact skip list so the next run can refuse known-failed work without a prompt.

    Prefer the explore rejected ledger (already fingerprinted) and
    ``rejected_kernel_ids``. ``last_action_failures`` only contributes when it
    still carries args/envs or a kernel id — the live store's action/task_id
    rows are not skippable and are omitted rather than published as prose.
    """
    config_items: list[dict[str, Any]] = []
    kernel_items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _add(item: dict[str, Any] | None) -> None:
        if not item:
            return
        kind = str(item.get("kind") or "")
        if kind in {"explore", "framework"}:
            token = str(item.get("fingerprint") or "")
            bucket = config_items
        elif kind == "kernel":
            token = str(item.get("kernel_id") or "")
            bucket = kernel_items
        else:
            return
        key = (kind, token)
        if not token or key in seen:
            return
        seen.add(key)
        bucket.append(item)

    explore = getattr(state, "explore_search", None) or {}
    rejected = explore.get("rejected") if isinstance(explore, Mapping) else None
    if isinstance(rejected, list):
        for row in rejected:
            if isinstance(row, Mapping):
                _add(_skip_explore_row(row, reason=str(row.get("reason") or "")))

    for row in _experience(state, "last_action_failures"):
        if not isinstance(row, Mapping):
            continue
        action = str(row.get("action") or "").strip().lower()
        _add(_skip_explore_row(row, reason=str(row.get("error_class") or "")))
        kid = str(row.get("kernel_id") or row.get("variant_name") or "").strip()
        if action in _KERNEL_SKIP_ACTIONS or kid:
            if action in _KERNEL_SKIP_ACTIONS or action.startswith("kernel"):
                _add(_skip_kernel_row(kid, action=action, reason=str(row.get("error_class") or "")))
        if action in _FRAMEWORK_SKIP_ACTIONS:
            framed = _skip_explore_row(row, reason=str(row.get("error_class") or ""))
            if framed is not None:
                framed = {**framed, "kind": "framework"}
                _add(framed)

    for kid in getattr(state, "rejected_kernel_ids", None) or []:
        _add(_skip_kernel_row(str(kid), reason="rejected_kernel_ids"))

    config_budget = _DO_NOT_REPEAT_CAP - min(len(kernel_items), _DO_NOT_REPEAT_KERNEL_RESERVE)
    items = config_items[:config_budget]
    return items + kernel_items[: _DO_NOT_REPEAT_CAP - len(items)]


_TOKEN_BUCKET_INTS = (
    "total_in",
    "total_out",
    "total_cache_creation",
    "total_cache_read",
    "total_reasoning_out",
    "calls",
)
_TOKEN_PHASE_CAP = 16


def _session_dir(state: Any) -> Path | None:
    raw = getattr(state, "session_dir", None) or getattr(state, "_session_dir", None)
    if raw in (None, ""):
        return None
    path = Path(str(raw))
    return path if path.is_dir() else None


def _elapsed_minutes_from_state(state: Any) -> float | None:
    raw = getattr(state, "elapsed_minutes", None)
    if callable(raw):
        try:
            raw = raw()
        except Exception:  # noqa: BLE001 — CLOSE must still publish the rest
            raw = None
    number = _finite_or_none(raw)
    if number is None or number <= 0:
        return None
    return round(number, 3)


def _compact_token_bucket(raw: Any) -> dict[str, Any]:
    """Project a token bucket to the compact counters miners read."""
    if not isinstance(raw, Mapping):
        return {}
    bucket = {key: int(raw.get(key, 0) or 0) for key in _TOKEN_BUCKET_INTS}
    from hyperloom.inference_optimizer.breakdown.collectors.decision import _token_convenience

    compact = _token_convenience(bucket)
    if int(compact.get("calls", 0) or 0) <= 0 and int(compact.get("grand_total", 0) or 0) <= 0:
        return {}
    return compact


def _token_usage_from_ledger(session_dir: Path) -> dict[str, Any] | None:
    """Roll up ``reports/trace/llm_calls.jsonl`` without waiting for breakdown."""
    from hyperloom.inference_optimizer.breakdown.collectors.decision import (
        _empty_token_bucket,
        _fold_call_into_bucket,
        _load_llm_calls,
    )

    warnings: list[str] = []
    calls = _load_llm_calls(session_dir, warnings)
    if not calls:
        return None
    session_total = _empty_token_bucket()
    by_phase: dict[str, dict[str, int]] = {}
    for call in calls:
        _fold_call_into_bucket(session_total, call)
        phase = str(call.get("phase") or "").strip() or "unattributed"
        _fold_call_into_bucket(by_phase.setdefault(phase, _empty_token_bucket()), call)
    total = _compact_token_bucket(session_total)
    if not total:
        return None
    phases: dict[str, dict[str, Any]] = {}
    for phase, bucket in list(by_phase.items())[:_TOKEN_PHASE_CAP]:
        compact = _compact_token_bucket(bucket)
        if compact:
            phases[phase] = compact
    record: dict[str, Any] = {"session_total": total, "source": "llm_calls"}
    if phases:
        record["by_phase"] = phases
    return record


def build_publishable_token_usage(state: Any) -> dict[str, Any] | None:
    """Compact LLM spend + wall clock for no-run savings estimates.

    CLOSE runs before ``session_breakdown.json`` exists, so this prefers an
    explicit ``state.token_usage`` and otherwise folds the live
    ``llm_calls.jsonl`` ledger. ``elapsed_minutes`` comes from SharedState.
    Returns ``None`` when neither spend nor elapsed is known.
    """
    record: dict[str, Any] = {}
    elapsed = _elapsed_minutes_from_state(state)
    if elapsed is not None:
        record["elapsed_minutes"] = elapsed
    usage = getattr(state, "token_usage", None)
    compact: dict[str, Any] | None = None
    if isinstance(usage, Mapping) and usage:
        total = _compact_token_bucket(usage.get("session_total") if "session_total" in usage else usage)
        if total:
            compact = {"session_total": total, "source": "state"}
            raw_phases = usage.get("by_phase")
            if isinstance(raw_phases, Mapping):
                phases: dict[str, dict[str, Any]] = {}
                for phase, bucket in list(raw_phases.items())[:_TOKEN_PHASE_CAP]:
                    item = _compact_token_bucket(bucket)
                    if item:
                        phases[str(phase)] = item
                if phases:
                    compact["by_phase"] = phases
    if compact is None:
        session_dir = _session_dir(state)
        if session_dir is not None:
            compact = _token_usage_from_ledger(session_dir)
    if compact:
        record["token_usage"] = compact
    return record or None


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


def _entry_origin(entry: Mapping[str, Any]) -> str:
    action = str(entry.get("action") or "").strip().lower()
    lever_section = SECTION_BY_LEVER.get(patch_lever_kind(entry))
    if lever_section:
        return lever_section
    # Pre-``lever_kind`` rows fall back to the phase that recorded them.
    phase = (
        patch_owner_phase(entry)
        if action.startswith("integrate_patch")
        else str(entry.get("source_phase") or "").strip().upper()
    )
    # Action before phase: both levers run inside FRAMEWORK_AGENT, so the
    # phase alone would file every config win under ``framework``.
    if action == "framework":
        return "framework"
    if action == "explore":
        return "explore"
    if phase == "FRAMEWORK_AGENT":
        return "framework"
    if phase == "EXPLORE":
        return "explore"
    if phase in ("KERNEL", "KERNEL_AGENT") or action in (
        "geak_e2e",
        "gemm_tuning",
        "fusion",
        "collective",
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
            or _entry_origin(entry) == "kernel"
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


def _entry_files(
    entries: list[dict[str, Any]], files: _Files, category: str
) -> tuple[list[str], list[str], dict[str, str]]:
    patches: list[str] = []
    artifacts: list[str] = []
    patch_roots: dict[str, str] = {}
    for entry in entries:
        try:
            stack_index = int(entry.get("__stack_index", -1))
        except (TypeError, ValueError):
            stack_index = -1
        patch_member = 0
        seen_patch_sources: set[str] = set()
        entry_framework_root = str(entry.get("framework_root") or "").strip()

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
            rel = f"{category}/overlays/{stack_index:06d}/{patch_member:02d}-{safe_name}.patch"
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
                if kind == "patches" and entry_framework_root:
                    patch_roots.setdefault(ref, entry_framework_root)
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
                    if kind == "patches" and entry_framework_root:
                        patch_roots.setdefault(ref, entry_framework_root)
    # first-wins, matching the dedupe below: a ref shared by two entries keeps
    # the root of the one whose position it also keeps.
    return list(dict.fromkeys(patches)), list(dict.fromkeys(artifacts)), patch_roots


def _section_value(patches: list[str], artifacts: list[str], patch_roots: dict[str, str]) -> dict[str, Any]:
    """Assemble one owner section, omitting ``patch_roots`` when nothing recorded one."""
    value: dict[str, Any] = {"patches": patches, "artifacts": artifacts}
    if patch_roots:
        value["patch_roots"] = patch_roots
    return value


def build_explore_value(
    state: Any,
    entries: list[dict[str, Any]],
    files: _Files,
) -> dict[str, Any]:
    """Build EXPLORE-origin patch and artifact references."""
    return _section_value(*_entry_files(entries, files, "explore"))


def build_framework_value(
    state: Any,
    entries: list[dict[str, Any]],
    files: _Files,
) -> dict[str, Any]:
    """Build FRAMEWORK-origin patch and artifact references."""
    return _section_value(*_entry_files(entries, files, "framework"))


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
    _patch_declared_targets(patch_source)
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
    }
    # Remove duplicate local-path aliases after establishing canonical refs.
    for key in (
        "patch_path",
        "target_file",
        "target_files",
        "source_file",
        "source_files",
        "artifact_files",
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
        _patch_declared_targets(patch_source)
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
            f"successfully replayed prior overlays are absent from prior knowledge: {sorted(missing_metadata)!r}"
        )

    member_index = 0
    seen: set[str] = set()
    files_root = warm_root / "files"
    if files_root.is_symlink():
        raise RemoteRecipeValidationError("replayed prior files root must not be a symlink")
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
        old_name = match.group(4) if match else source.stem
        safe_name = _OVERLAY_NAME_RE.sub("-", old_name).strip("._-") or "patch"
        new_ref = f"{owner}/overlays/{replay_index:06d}/{member_index:02d}-{safe_name}.patch"
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


def _patch_timeline(value: Mapping[str, Any]) -> list[str]:
    """Build the global replay order from section overlay refs."""
    rows: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    for owner in ("explore", "framework"):
        node = _mapping(value.get(owner))
        for raw_ref in node.get("patches") or []:
            ref = str(raw_ref or "")
            match = _OVERLAY_REF_RE.match(ref)
            if match is None or match.group(1) != owner:
                raise RemoteRecipeValidationError(f"value.{owner}.patches contains an invalid overlay ref: {ref!r}")
            if ref in seen:
                raise RemoteRecipeValidationError(f"owner patch refs contain a duplicate: {ref!r}")
            seen.add(ref)
            stack_index = int(match.group(2))
            member_index = int(match.group(3))
            rows.append((stack_index, member_index, owner, ref))
    rows.sort()
    return [ref for _stack_index, _member_index, _owner, ref in rows]


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


def merge_staged_sections(
    value: dict[str, Any],
    sections: Any,
    files: "_Files",
    *,
    only: Collection[str] | None = None,
    required: Collection[str] | None = None,
) -> list[str]:
    """Merge staged patch and artifact refs into their owner sections."""
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
                    f"staged section {name!r} file mismatch: missing={sorted(missing)!r} orphaned={sorted(orphaned)!r}"
                )
            for source, rel in staged_files:
                files.validate_adoption(source, rel)
            for source, rel in staged_files:
                files.adopt(source, rel)
            current = value.get(name)
            combined = {
                "patches": (list(current.get("patches") or []) if isinstance(current, Mapping) else []),
                "artifacts": (list(current.get("artifacts") or []) if isinstance(current, Mapping) else []),
            }
            # Owner sections contain patch/artifact refs only.
            for ref_key in ("patches", "artifacts"):
                before = list(current.get(ref_key) or []) if isinstance(current, Mapping) else []
                after = list(staged.knowledge.get(ref_key) or [])
                if before or after:
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
    sections: Any = None,
) -> KnowledgeBundle:
    """Construct the final opaque knowledge document and temporary files tree."""
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
    # ``kb_required_owner`` is a recorded phase label; both spellings still map
    # to their section so a resumed pre-merge session stages correctly.
    owner_names = {
        "EXPLORE": "explore",
        "FRAMEWORK_AGENT": "framework",
    }
    dropped_entries: set[tuple[str, int]] = set()
    for row in dropped_sections:
        try:
            stack_index = int(row.get("stack_index") or 0)
        except (TypeError, ValueError):
            continue
        dropped_entries.add((str(row.get("owner") or "").upper(), stack_index))
    required_patch_owners = {
        owner_names[owner]
        for item in stack
        if (owner := str(item.get("kb_required_owner") or "").upper()) in owner_names
        and (owner, int(item["__stack_index"])) not in dropped_entries
    }
    explore_entries = [item for item in stack if _entry_origin(item) == "explore"]
    framework_entries = [item for item in stack if _entry_origin(item) == "framework"]
    current_best = _mapping(getattr(state, "current_best", {}))
    optimized_throughput = _number(current_best.get("tput"))
    validated_gain = _number(getattr(state, "cumulative_gain_validated", 0.0))
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
    kernel_value = (
        {
            "gemm": {"optimizations": []},
            "fusion": {"items": []},
            "rewrite": {"items": []},
        }
        if scope.kernel_optimizer == "geak"
        else {
            "gemm": build_kernel_gemm_value(state, files),
            "fusion": build_kernel_fusion_value(state, files),
            "rewrite": build_kernel_rewrite_value(state, files),
        }
    )
    value = {
        "config": build_publishable_recipe_config(state),
        "explore": explore_value,
        "framework": framework_value,
        "kernel": kernel_value,
    }
    if sections is not None:
        _adopt_replayed_prior(state, sections, value, files, stack)
        if scope.kernel_optimizer == "forge":
            _adopt_prior_kernel(state, sections, value, files)
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
            f"required staged owner sections are missing: {sorted(missing_required_owners)!r}"
        )
    value["patch_timeline"] = _patch_timeline(value)
    roofline = build_publishable_roofline(state)
    if roofline:
        value["roofline"] = roofline
    platform = build_publishable_platform(state)
    if platform:
        value["platform"] = platform
    do_not_repeat = build_publishable_do_not_repeat(state)
    if do_not_repeat:
        value["do_not_repeat"] = do_not_repeat
    cost = build_publishable_token_usage(state)
    if cost:
        if "elapsed_minutes" in cost:
            value["elapsed_minutes"] = cost["elapsed_minutes"]
        if "token_usage" in cost:
            value["token_usage"] = cost["token_usage"]
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
    config = _mapping(value.get("config"))
    if str(config.get("extra_server_args") or "").strip() or _mapping(config.get("extra_envs")):
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
        for material in ("patches", "artifacts"):
            items = section.get(material)
            if isinstance(items, list) and items:
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
    for section in ("explore", "framework", "kernel"):
        if not isinstance(value.get(section), Mapping):
            raise RemoteRecipeValidationError(f"current Recipe is missing value.{section}")
    raw_timeline = value.get("patch_timeline")
    if not isinstance(raw_timeline, list) or not all(isinstance(ref, str) for ref in raw_timeline):
        raise RemoteRecipeValidationError("current Recipe value.patch_timeline must be a flat string list")
    for ref in raw_timeline:
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
    dnr = value.get("do_not_repeat")
    if isinstance(dnr, list) and dnr:
        row["do_not_repeat"] = [item for item in dnr if isinstance(item, dict)]
    usage = value.get("token_usage")
    if isinstance(usage, Mapping) and usage:
        row["token_usage"] = dict(usage)
    elapsed = _finite_or_none(value.get("elapsed_minutes"))
    if elapsed is not None:
        row["elapsed_minutes"] = elapsed
    for key, shape_value in _mapping(knowledge.get("workload_shape")).items():
        if key in {"tp", "conc", "isl", "osl"}:
            resolved = _positive_int(shape_value)
            if resolved is not None:
                row[key] = resolved
    return row


__all__ = [
    "CURRENT_KNOWLEDGE_SCHEMA_VERSION",
    "RECORD_KIND_HYPERLOOM_RECIPE",
    "build_explore_value",
    "build_framework_value",
    "build_kernel_fusion_value",
    "build_kernel_gemm_value",
    "build_kernel_rewrite_value",
    "build_publishable_recipe_config",
    "build_publishable_roofline",
    "build_publishable_platform",
    "build_publishable_do_not_repeat",
    "build_publishable_token_usage",
    "build_remote_knowledge",
    "has_replay_material",
    "knowledge_to_warm_recipe",
    "has_new_keep",
    "match_rewrite_attempt",
    "merge_staged_sections",
]
