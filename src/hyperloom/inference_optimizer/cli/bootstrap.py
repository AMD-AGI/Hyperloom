# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session bootstrap + summary helpers for the CLI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.coerce import to_unix
from hyperloom.common.env import forge_explicitly_enabled
from hyperloom.common.gpu_partition import published_shape
from hyperloom.common.perf_metric import is_agentx_mode
from hyperloom.common.timeutil import now_iso
from hyperloom.orchestrator.actions.executors._workload_envs import (
    agentx_enabled as _agentx_enabled,
)
from hyperloom.orchestrator.phases.machine_state import bank_phase_segment
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.common.workload_defaults import (
    DEFAULT_ISL,
    DEFAULT_OSL,
    DEFAULT_CONC,
    DEFAULT_TP,
    DEFAULT_EP,
    DEFAULT_PRECISION,
)
from ..session.paths import _SESSION_SKELETON, ENV_USER_DATA_PATH
from .model_gate import _load_model_arch, _load_model_config_tags
from ..model_config_utils import summarize_model_config

log = logging.getLogger(__name__)


def parse_operator_extra_env(args: argparse.Namespace) -> dict[str, str]:
    """Parse ``--extra-env NAME=VALUE`` pins into a mapping."""
    pins: dict[str, str] = {}
    for item in getattr(args, "extra_env", None) or []:
        key, sep, value = str(item).partition("=")
        if sep and key.strip():
            pins[key.strip()] = value
    return pins


def resolve_model_display_name(args: argparse.Namespace) -> str:
    """Resolve the canonical model identity used for session naming / display."""
    override = (getattr(args, "model_display_name", "") or "").strip()
    if override:
        return override
    return Path(str(getattr(args, "model", "") or "")).name


# Bump when a change makes previously recorded AgentX measurements incomparable.
# The MLPerf client is a different workload, but it is opt-in: the epoch stays
# so aiperf sessions remain resumable. The backend name is what resume compares.
AGENTX_MEASUREMENT_EPOCH = 1


def seed_grading(framework: str, benchmark_mode: str) -> dict[str, Any]:
    """Resolve the grading axis and its noise band once, at seed, so they can be recorded.

    The resolution reads ``HYPERLOOM_PERF_METRIC`` and ``HYPERLOOM_PERF_NOISE_PCT``. Deriving it again later -- in a
    resumed process, a re-baseline subprocess, or the breakdown export CLOSE drives from a subprocess that often did
    not inherit them -- can name an axis the session never graded on. This is the same reasoning that put
    ``benchmark_mode`` in the state rather than leaving it to the ambient var.
    """
    from hyperloom.common.perf_metric import (
        GRADED_INTVTY,
        GRADED_OUTPUT,
        intvty_serving_grading_enabled,
        parse_intvty_noise_pct,
    )

    from .. import framework_registry

    on_intvty = intvty_serving_grading_enabled(
        scriptable=framework_registry.is_scriptable(framework),
        benchmark_mode=benchmark_mode,
    )
    return {
        "objective": GRADED_INTVTY if on_intvty else GRADED_OUTPUT,
        "noise_pct": parse_intvty_noise_pct(),
    }


def agentx_state_is_stale(state: Any) -> str:
    """Return why a resumed session's AgentX state is unusable, or ``\"\"``."""
    want_mode = "agentx" if _agentx_enabled() else "synthetic"
    had_mode = str(getattr(state, "benchmark_mode", "") or "")
    if had_mode and had_mode != want_mode:
        return (
            f"session was measured in benchmark_mode={had_mode!r} but this run is "
            f"{want_mode!r}; the KEEP ledger keys on server args only, so the two "
            "sets of measurements would overwrite each other"
        )
    if want_mode == "agentx":
        had_epoch = int(getattr(state, "agentx_epoch", 0) or 0)
        if had_epoch != AGENTX_MEASUREMENT_EPOCH:
            return (
                f"session carries AgentX epoch {had_epoch}, this build measures "
                f"epoch {AGENTX_MEASUREMENT_EPOCH}; the recorded results describe "
                "a different workload and cannot anchor or be compared against"
            )
        from hyperloom.common.agentx_workload import agentic_backend

        # Sessions recorded before the backend was persisted are aiperf.
        had_backend = str(getattr(state, "agentx_backend", "") or "") or "aiperf"
        want_backend = agentic_backend()
        if had_backend != want_backend:
            return (
                f"session was measured with agentic backend {had_backend!r} but this "
                f"run is {want_backend!r}; the recorded results describe a different "
                "workload and cannot anchor or be compared against"
            )
    return ""


def latency_budget_scope_error(framework: str | None, requested_ms: float | None) -> str:
    """Return why ``--max-latency-ms`` does not apply to *framework*, or ``\"\"``.

    Scriptable workloads grade on output throughput alone and are the only frameworks compute partitioning places
    work for, so they are the only place a throughput-only gate can buy throughput with per-request latency. AgentX
    serving sessions already REVERT that trade on interactivity, and the fixed ISL/OSL serving mode takes no new
    capability, so the budget is refused there rather than silently doing nothing.
    """
    from .. import framework_registry

    if requested_ms is None or framework_registry.is_scriptable(framework):
        return ""
    name = str(framework or "").strip() or framework_registry.DEFAULT_FRAMEWORK
    return (
        f"--max-latency-ms applies only to scriptable frameworks (xdit, custom); {name!r} is a serving framework. "
        "On AgentX the interactivity objective already refuses a throughput gain bought with per-request latency"
    )


def _budget_resume_conflict(flag: str, unit: str, archived: float, requested: float | None) -> str:
    """Why a resume-time budget differs from the archived one, or ``\"\"``; omitting the flag keeps the archive."""
    if requested is None or float(requested) == archived:
        return ""
    recorded = f"{archived:g} {unit}" if archived > 0 else "no budget"
    return (
        f"{flag} {float(requested):g} differs from the {recorded} this session was "
        "graded under; its KEEPs would be judged against a constraint they were never measured "
        "for. Resume without the flag to keep the recorded budget, or start a fresh session"
    )


def latency_budget_resume_conflict(state: Any, requested_ms: float | None) -> str:
    """Return why ``--max-latency-ms`` cannot apply to a resumed session, or ``\"\"``.

    The recorded KEEPs were graded under the archived budget, so a different
    value would leave them judged against a constraint the new one does not
    state. Omitting the flag keeps the archived budget.
    """
    return _budget_resume_conflict(
        "--max-latency-ms", "ms", float(getattr(state, "latency_budget_ms", 0.0) or 0.0), requested_ms
    )


def power_budget_resume_conflict(state: Any, requested_w: float | None) -> str:
    """Return why ``--max-power-w`` cannot apply to a resumed session, or ``\"\"``."""
    return _budget_resume_conflict(
        "--max-power-w", "W", float(getattr(state, "power_budget_w", 0.0) or 0.0), requested_w
    )


def resolve_gpu_power_settings(
    *,
    power_cap_w: float | None,
    perf_level: str | None,
    nodes: int,
    read: Any = None,
) -> tuple[dict[str, Any], str]:
    """Read the cards' power settings and check the declared ones; ``(record, error)``.

    ``record`` is what the session stores and the platform fingerprint shows: the declared values and what each card
    reported. ``error`` is non-empty when a declared value does not hold, or cannot be checked, and the launch must stop.
    Nothing is set here; the operator sets power cap and perf level with ``amd-smi set`` before launch.
    """
    from hyperloom.common.gpu_power_settings import (
        GpuPowerSettingsError,
        declared_setting_problems,
        normalize_perf_level,
        read_gpu_power_settings,
        visible_gpu_indices,
    )

    declared: dict[str, Any] = {}
    if power_cap_w is not None:
        declared["power_cap_w"] = float(power_cap_w)
    if perf_level:
        declared["perf_level"] = normalize_perf_level(perf_level)
    if nodes >= 2:
        if declared:
            return {}, (
                "--gpu-power-cap-w / --gpu-perf-level cannot be checked on a multi-node session: they describe the "
                "benchmark nodes' cards, which this process cannot read, and an unverifiable assertion is not a "
                "satisfied one"
            )
        return {}, ""
    try:
        observed = (read or read_gpu_power_settings)()
    except GpuPowerSettingsError as exc:
        if declared:
            return {"declared": declared}, f"the declared GPU power settings cannot be checked: {exc}"
        return {}, ""
    gpus = visible_gpu_indices()
    record = {
        "declared": declared,
        "observed": {str(gpu): row for gpu, row in sorted(observed.items()) if gpus is None or gpu in gpus},
    }
    problems = declared_setting_problems(
        observed,
        power_cap_w=declared.get("power_cap_w"),
        perf_level=declared.get("perf_level"),
        gpus=gpus,
    )
    if problems:
        return record, (
            "the GPUs are not at the declared power settings (set them with amd-smi before launch): "
            + "; ".join(problems)
        )
    return record, ""


#: A card holding more than this much VRAM before the session has started anything has someone else's model resident,
#: and a cap is card-wide: setting it would change that tenant's run too.
_FOREIGN_RESIDENT_VRAM_MB = 2048.0


def gpu_power_ledger_dir() -> Path:
    """Where per-card power-settings records live: the workspace-shared runtime directory, so every session on the
    host that shares it sees the same records."""
    configured = (os.environ.get("HYPERLOOM_RUNTIME_DIR") or "").strip()
    root = Path(configured) if configured else Path(os.environ.get(ENV_USER_DATA_PATH) or ".") / "runtime"
    return root / "gpu_power_settings"


def apply_declared_gpu_power_settings(
    *,
    power_cap_w: float | None,
    perf_level: str | None,
    nodes: int,
    owner: str,
    ledger_dir: Path | None = None,
    read: Any = None,
    resident_vram: Any = None,
    apply: Any = None,
    restore: Any = None,
    lease_factory: Any = None,
) -> tuple[Any, dict[str, Any], str]:
    """Set the declared cap / perf level on the session's cards; ``(restore_callback, applied_record, error)``.

    Only reached with ``--apply-gpu-power-settings``. The originals are recorded under an exclusive per-card lease
    before anything is set, and ``restore_callback`` puts them back and releases the lease; the caller registers it to
    run at exit. When ``error`` is non-empty nothing was left changed and the launch must stop. The read-back check is
    still :func:`resolve_gpu_power_settings`, run after this, so an applied value is verified the same way an
    operator-set one is.
    """
    from hyperloom.common.gpu_power_settings import (
        GpuPowerSettingsError,
        PowerSettingsLease,
        apply_gpu_power_settings,
        normalize_perf_level,
        read_gpu_power_settings,
        read_resident_vram_mb,
        restore_gpu_power_settings,
        visible_gpu_indices,
    )

    read = read or read_gpu_power_settings
    resident_vram = resident_vram or read_resident_vram_mb
    apply = apply or apply_gpu_power_settings
    restore = restore or restore_gpu_power_settings
    lease_factory = lease_factory or PowerSettingsLease
    level = normalize_perf_level(perf_level) if perf_level else ""
    if power_cap_w is None and not level:
        return None, {}, "--apply-gpu-power-settings needs --gpu-power-cap-w and/or --gpu-perf-level to apply"
    if nodes >= 2:
        return None, {}, "--apply-gpu-power-settings cannot set the cards of a multi-node session from this process"

    try:
        observed = read()
        mask = visible_gpu_indices()
        gpus = set(observed) if mask is None else set(observed) & mask
        if not gpus:
            return None, {}, "no GPU the session can use was reported by amd-smi"
        lease = lease_factory(ledger_dir or gpu_power_ledger_dir(), gpus)
    except GpuPowerSettingsError as exc:
        return None, {}, f"cannot apply the declared GPU power settings: {exc}"

    # Only what this session changes is recorded, so only that is put back: restoring a perf level nobody declared
    # would reset a card left in MANUAL clocks for reasons this session knows nothing about.
    touched = [key for key, value in (("power_cap_w", power_cap_w), ("perf_level", level)) if value]

    def _originals_of(rows: Any) -> dict[int, dict[str, Any]]:
        return {gpu: {key: rows[gpu].get(key) for key in touched} for gpu in sorted(gpus) if gpu in rows}

    recovered: dict[int, dict[str, Any]] = {}
    try:
        orphans = lease.orphaned()
        if orphans:
            recovered = {gpu: dict(rec.get("original") or {}) for gpu, rec in orphans.items()}
            problems = restore(recovered)
            if problems:
                lease.release()
                return (
                    None,
                    {},
                    (
                        "a previous session left these cards at settings it applied and they could not be restored: "
                        + "; ".join(problems)
                    ),
                )
            lease.clear()
            observed = read()
        busy = sorted(gpu for gpu, used in resident_vram().items() if gpu in gpus and used > _FOREIGN_RESIDENT_VRAM_MB)
        if busy:
            lease.release()
            return (
                None,
                {},
                (
                    f"GPU(s) {', '.join(map(str, busy))} already hold a resident model; a power cap is card-wide and would "
                    "change that workload too. Free the cards or pin the session away from them"
                ),
            )
        originals = _originals_of(observed)
        # A value that cannot be read cannot be put back: restore would skip it, report success and clear the record,
        # leaving the card at the applied setting with nothing left to recover it.
        unreadable = sorted(
            gpu
            for gpu in gpus
            if gpu not in originals
            or any(
                originals[gpu].get(key) is None
                or isinstance(originals[gpu].get(key), bool)
                or originals[gpu].get(key) == ""
                for key in touched
            )
        )
        if unreadable:
            lease.release()
            return (
                None,
                {},
                (
                    f"the current {' / '.join(k.replace('_w', '').replace('_', ' ') for k in touched)} of GPU(s) "
                    f"{', '.join(map(str, unreadable))} could not be read, so it could not be restored after the session; "
                    "nothing was applied"
                ),
            )
        declared = {"power_cap_w": power_cap_w, "perf_level": level}
        lease.record(originals, applied={key: declared[key] for key in touched}, owner=owner)
    except GpuPowerSettingsError as exc:
        lease.release()
        return None, {}, f"cannot apply the declared GPU power settings: {exc}"

    done = False

    def _restore() -> list[str]:
        nonlocal done
        if done:
            return []
        done = True
        problems = restore(originals)
        if not problems:
            lease.clear()
        lease.release()
        return problems

    try:
        apply(gpus, power_cap_w=power_cap_w, perf_level=level)
    except GpuPowerSettingsError as exc:
        leftover = _restore()
        suffix = f" (restoring the originals also failed: {'; '.join(leftover)})" if leftover else ""
        return None, {}, f"cannot apply the declared GPU power settings: {exc}{suffix}"

    applied = {
        "by": "hyperloom",
        "gpus": sorted(gpus),
        "originals": {str(gpu): row for gpu, row in originals.items()},
    }
    if recovered:
        applied["recovered_from_orphan"] = sorted(recovered)
    return _restore, applied, ""


def orphaned_power_settings_warning(ledger_dir: Path | None = None) -> str:
    """A warning when a dead session left one of this session's cards at a value it applied; ``""`` otherwise."""
    from hyperloom.common.gpu_power_settings import orphaned_power_records, visible_gpu_indices

    orphans = orphaned_power_records(ledger_dir or gpu_power_ledger_dir(), visible_gpu_indices())
    if not orphans:
        return ""

    def _describe(values: Mapping[str, Any]) -> str:
        cap, level = values.get("power_cap_w"), values.get("perf_level")
        return ", ".join(
            ([f"cap {cap:g} W"] if isinstance(cap, (int, float)) else []) + ([f"perf level {level}"] if level else [])
        )

    parts = [
        f"GPU {gpu} at {_describe(record.get('applied') or {}) or '?'} from session {record.get('owner') or '?'}, "
        f"originally {_describe(record.get('original') or {}) or '?'}"
        for gpu, record in sorted(orphans.items())
    ]
    return (
        "these cards were left at power settings a Hyperloom session applied and never restored: "
        + "; ".join(parts)
        + ". Pass --apply-gpu-power-settings to restore them at launch, or restore them with amd-smi set"
    )


def _build_agentx_corpus_shape_seed() -> dict[str, Any]:
    """Return the canonical corpus shape, until a measurement replaces it."""
    from hyperloom.common.agentx_workload import MLPERF_CORPUS, is_mlperf_backend, mlperf_trajectories
    from hyperloom.inference_optimizer.agentx.mapping import (
        CANONICAL_CORPUS_DURATION_S,
        CANONICAL_CORPUS_ENTRIES,
        CANONICAL_CORPUS_LOADER,
        CANONICAL_ISL,
        CANONICAL_OSL,
        CANONICAL_PREFIX_CACHE_HIT,
    )

    if is_mlperf_backend():
        # The MLPerf corpus has no published shape; the first measurement supplies it.
        return {
            "corpus_loader": MLPERF_CORPUS,
            "corpus_entries": mlperf_trajectories(),
            "source": "canonical_mlperf",
        }

    return {
        "corpus_loader": CANONICAL_CORPUS_LOADER,
        "corpus_entries": CANONICAL_CORPUS_ENTRIES,
        "duration_s": float(CANONICAL_CORPUS_DURATION_S),
        "isl": dict(CANONICAL_ISL),
        "osl": dict(CANONICAL_OSL),
        "prefix_cache_hit": CANONICAL_PREFIX_CACHE_HIT,
        "source": "canonical",
    }


def _seed_shared_state(
    session_dir: Path,
    args: argparse.Namespace,
    *,
    session_id: str,
    compute_partition: dict[str, Any] | None = None,
    gpu_power_settings: dict[str, Any] | None = None,
) -> SharedState:
    """Construct and persist the initial :class:`SharedState` for a run."""
    # research_lane capacity is locked for the session; clamp to [0, ceiling].
    from hyperloom.common.visible_devices import detect_gpu_count
    from hyperloom.orchestrator.policy.gate import research_lane_ceiling

    research_lane_capacity = int(getattr(args, "research_lane_capacity", 1) or 1)
    research_lane_capacity = max(
        0,
        min(research_lane_ceiling(), research_lane_capacity),
    )
    gpu_specialist_capacity_raw = getattr(
        args,
        "gpu_specialist_capacity",
        None,
    )
    try:
        gpu_specialist_capacity = max(
            0,
            int(gpu_specialist_capacity_raw) if gpu_specialist_capacity_raw is not None else detect_gpu_count(),
        )
    except (TypeError, ValueError):
        gpu_specialist_capacity = detect_gpu_count()
    # Collect plateau threshold overrides; absent keys use defaults at compute time.
    plateau_overrides: dict[str, Any] = {}
    if getattr(args, "plateau_explore_keep_gain", None) is not None:
        plateau_overrides["explore_keep_gain_pct"] = float(args.plateau_explore_keep_gain)
    if getattr(args, "plateau_explore_empty_streak", None) is not None:
        plateau_overrides["explore_empty_streak"] = int(args.plateau_explore_empty_streak)
    if getattr(args, "plateau_explore_lookback", None) is not None:
        plateau_overrides["explore_lookback"] = int(args.plateau_explore_lookback)

    # Resolve int workload knobs from the CLI arg, applying the shared fallback default when unset.
    def _int_arg(arg_name: str, default: int) -> int:
        """Resolve an int workload knob from ``args``, else the fallback default."""
        val = getattr(args, arg_name, None)
        if val is None:
            return int(default)
        try:
            resolved = int(val)
        except (TypeError, ValueError):
            return int(default)
        return resolved if resolved > 0 else int(default)

    def _resolve_framework_version(args_in: Any) -> str:
        """Resolve ``framework_version`` for the recipe-snapshot canonical id."""
        explicit = (getattr(args_in, "framework_version", None) or "").strip() or (
            os.environ.get("FRAMEWORK_VERSION", "") or ""
        ).strip()
        if explicit:
            return explicit
        framework = (getattr(args_in, "framework", None) or "").strip() or (
            os.environ.get("FRAMEWORK", "") or ""
        ).strip()
        if not framework:
            return ""
        from ..recipe_snapshot_constants import (
            DEFAULT_FRAMEWORK_VERSION_SLUG,
            detect_framework_version,
        )

        detected = detect_framework_version(framework)
        # Treat the failure-slug as "no info".
        return "" if detected == DEFAULT_FRAMEWORK_VERSION_SLUG else detected

    # KB architecture tags from config.json; fresh-launch only.
    _cfg_tags = _load_model_config_tags(str(args.model))

    # Persisted for the session breakdown; the runtime reads the env directly.
    _kernel_optimizer_record = "forge" if forge_explicitly_enabled() else "geak"

    # Reference launch recipe (fresh-launch only, fail-soft): lowest-priority base for the baseline server args.
    _ref_args, _ref_envs, _ref_model, _ref_controls = _resolve_reference_recipe(args)

    # Canonical model identity (prefers the quantize prelude's pinned source name).
    _model_identity = resolve_model_display_name(args)
    benchmark_mode = "agentx" if _agentx_enabled() else "synthetic"
    if _agentx_enabled():
        from hyperloom.common.agentx_workload import agentic_backend

        agentx_backend = agentic_backend()
    else:
        agentx_backend = ""
    state = SharedState(
        session_id=session_id,
        claw_session_id=(os.environ.get("CLAW_SESSION_ID") or "").strip(),
        sandbox_user_id=(os.environ.get("SANDBOX_USER_ID") or "").strip(),
        model_name=_model_identity,
        model_path=str(args.model),
        model_class=args.model_class or "",
        # Advisory architecture profile; fresh-launch only. Soft-degrade to {}.
        model_arch=_load_model_arch(
            session_dir,
            _model_identity,
            str(args.model),
        ),
        # Architecture-identity tags from config.json.
        model_architectures=_cfg_tags.get("architectures", []),
        model_type=_cfg_tags.get("model_type", ""),
        # config.json structural summary, persisted for downstream collectors.
        model_info=summarize_model_config(str(args.model)),
        framework=os.environ.get("FRAMEWORK", "sglang"),
        # The only copy of the budget. Validated at the CLI, so anything that reaches here is usable, and archived
        # with the session so a resume restores it without a second source to reconcile.
        latency_budget_ms=float(getattr(args, "max_latency_ms", None) or 0.0),
        power_budget_w=float(getattr(args, "max_power_w", None) or 0.0),
        gpu_power_settings=dict(gpu_power_settings or {}),
        gpu_type=str(getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")),
        # Workload metadata mirrored from CLI/env.
        tp=_int_arg("tp", DEFAULT_TP),
        ep=_int_arg("ep", DEFAULT_EP),
        precision=(str(getattr(args, "precision", None) or DEFAULT_PRECISION).strip()),
        framework_version=_resolve_framework_version(args),
        conc=_int_arg("conc", DEFAULT_CONC),
        isl=_int_arg("isl", DEFAULT_ISL),
        osl=_int_arg("osl", DEFAULT_OSL),
        profile_osl=_int_arg("profile_osl", 0),
        max_model_len=_int_arg("max_model_len", 0),
        kernel_enabled=not getattr(args, "no_kernel", False),
        kernel_optimizer=_kernel_optimizer_record,
        # AgentX corpus shape: seeded from canonical constants if AgentX is on;
        # overwritten by the measured shape after every aiperf run.
        agentx_corpus_shape=_build_agentx_corpus_shape_seed() if benchmark_mode == "agentx" else {},
        baseline_tput=0.0,
        cumulative_gain_validated=0.0,
        reference_server_args=_ref_args,
        reference_envs=_ref_envs,
        reference_launch_controls=_ref_controls,
        reference_model=_ref_model,
        # Operator launch shape; the process env carries it for one process only, so a resume re-exports it from here
        # rather than from argv.
        operator_server_args=str(getattr(args, "server_args", "") or "").strip(),
        operator_extra_env=parse_operator_extra_env(args),
        bypass_scripts_dir=os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip(),
        framework_repo_path=os.environ.get("FRAMEWORK_REPO_PATH", "").strip(),
        benchmark_backend=os.environ.get("HYPERLOOM_BENCHMARK_BACKEND", "").strip().lower(),
        compute_partition=dict(compute_partition if compute_partition is not None else (published_shape() or {})),
        nodes=max(1, int(getattr(args, "nodes", 1) or 1)),
        warm_replay_enabled=not bool(getattr(args, "no_warm_replay", False)),
        **(
            {}
            if getattr(args, "warm_replay_min_confidence", None) is None
            else {"warm_replay_min_confidence": float(args.warm_replay_min_confidence)}
        ),
        max_minutes=int((args.max_hours or 0) * 60),
        research_lane_capacity=research_lane_capacity,
        gpu_specialist_capacity=gpu_specialist_capacity,
        plateau_overrides=plateau_overrides,
        enable_roofline=bool(
            getattr(args, "enable_roofline", True),
        ),
        # Standalone FRAMEWORK_AGENT phase; --no-framework-agent skips it.
        framework_agent_phase_enabled=not bool(getattr(args, "no_framework_agent", False)),
        # FRAMEWORK local-exploration arm; --no-framework-local-explore opts out.
        framework_local_explore_enabled=not bool(getattr(args, "no_framework_local_explore", False)),
        # Enablement self-heal lanes; --enablement off opts out.
        enablement_mode=str(getattr(args, "enablement", "all") or "all"),
        eval_disabled=bool(getattr(args, "no_eval", False)),
        research_scout_enabled=bool(getattr(args, "research_scout", True)),
        research_scout_interval=max(1, int(getattr(args, "research_scout_interval", 3) or 3)),
        static_recon_enabled=bool(getattr(args, "static_recon", True)),
        target_advisory_enabled=bool(getattr(args, "target_advisory", True)),
        recipe_sediment_enabled=bool(getattr(args, "recipe_sediment", True)),
        # SWEEP-phase concurrency sweep: defaults OFF under AgentX because each
        # rung is a 3600s window and the session grades at a fixed CONC.
        # Pass --enable-conc-sweep explicitly to override.
        conc_sweep_enabled=(
            not is_agentx_mode(benchmark_mode) if args.enable_conc_sweep is None else args.enable_conc_sweep
        ),
        benchmark_mode=benchmark_mode,
        agentx_epoch=AGENTX_MEASUREMENT_EPOCH if _agentx_enabled() else 0,
        agentx_backend=agentx_backend,
        grading=seed_grading(os.environ.get("FRAMEWORK", "sglang"), benchmark_mode),
        conc_sweep_concs=_parse_conc_sweep_concs(args, benchmark_mode),
        conc_sweep_total_budget_sec=int(
            getattr(args, "conc_sweep_total_budget_sec", 9000) or 0,
        ),
    )
    state.save(session_dir)
    return state


def _snapshot_system_prompts(
    session_dir: Path,
    *,
    prompts: dict[str, str],
    macro_cycle: int,
    orchestration_phase: str = "",
) -> None:
    """Persist each agent's effective system prompt via the shared snapshot writer.

    The orchestration role additionally writes a phase-scoped copy when ``orchestration_phase`` is set.
    """
    from hyperloom.orchestrator.prompts import write_prompt_snapshot

    for role, body in prompts.items():
        write_prompt_snapshot(session_dir, role, body, macro_cycle=macro_cycle)
    boot_phase = orchestration_phase.strip()
    if boot_phase and "orchestration" in prompts:
        write_prompt_snapshot(
            session_dir, "orchestration", prompts["orchestration"], phase=boot_phase, macro_cycle=macro_cycle
        )


def _print_session_skeleton(session_dir: Path) -> None:
    """Echo the freshly-created skeleton so launchers see the exact layout."""
    print(f"Session layout under {session_dir}:")
    for sub in _SESSION_SKELETON:
        marker = "ok" if (session_dir / sub).is_dir() else "MISSING"
        print(f"  [{marker}] {sub}/")
    print("  [ok] manifest.json (written first)")


def _print_final_summary(
    state: SharedState,
    stop_reason: str,
    session_dir: Path,
) -> None:
    """Print the end-of-run summary block to stdout."""
    print()
    print("================ Final summary ================")
    print(f"  stop_reason          : {stop_reason}")
    print(f"  session_id           : {state.session_id}")
    print(f"  model                : {state.model_name}")
    from .. import framework_registry

    print(
        f"  baseline             : {framework_registry.format_primary_metric(getattr(state, 'framework', ''), state.baseline_tput)}"
    )
    if stop_reason == "baseline_failed":
        failure_summary = _read_failure_summary(session_dir)
        if failure_summary and failure_summary.get("root_cause"):
            print(
                f"  root_cause           : "
                f"[{failure_summary.get('root_cause_type', 'unknown')}] "
                f"{failure_summary.get('root_cause')}"
            )
            if failure_summary.get("server_log"):
                print(f"  server_log           : {failure_summary.get('server_log')}")
    if state.cumulative_gain_validated_ts:
        stale = " ⚠ stack changed since validation" if state.optimization_stack_has_unvalidated_keeps() else ""
        print(
            f"  cumulative_gain_val  : {state.cumulative_gain_validated:.2f}% "
            f"(validated_at_stack_len={state.cumulative_gain_validated_stack_len}, "
            f"ts={state.cumulative_gain_validated_ts}){stale}"
        )
    else:
        print("  cumulative_gain_val  : 0.00% ⚠ never validated — no `explore` KEEP has landed yet")
    print(f"  current_best         : {state.current_best}")
    print(f"  pruned_families      : {state.pruned_families}")
    print(f"  crash_count          : {state.crash_count}")
    print("===============================================")


def _bank_previous_leg_phase_segment(state: SharedState) -> None:
    """Bank the phase time the stopped leg spent but never recorded."""
    boundary = state.leg_ended_ts or state.stop_ts
    stop_unix = min(to_unix(boundary, 0.0) or 0.0, time.time())
    if stop_unix <= 0.0:
        return
    bank_phase_segment(state, until_unix=stop_unix)


def _begin_resume_leg(state: SharedState) -> str:
    """Mark the start of a resumed run leg on ``state`` (caller persists).

    Stamps :attr:`SharedState.resumed_ts`, which dates the previous leg's CLOSE
    transition in ``phase_history`` as a previous leg's and stops the phase
    clock charging the gap between the two legs to the phase it stopped in.
    Clears the previous leg's terminal bookkeeping: a stale ``stop_reason``
    makes Orchestration heartbeats think the work is done, and a stale closing
    flag resumes straight into a wind-down already finished.

    The wall-clock budget is untouched. Elapsed time is summed forward across
    legs, so this leg starts from whatever the session has already spent;
    :meth:`SharedState.extend_budget_minutes` is the only way to lengthen it.

    Args:
        state (SharedState): The loaded session state, mutated in place.

    Returns:
        str: The timestamp stamped as this leg's boundary.
    """
    _bank_previous_leg_phase_segment(state)
    state.resumed_ts = now_iso()
    state.stop_reason = ""
    state.stop_ts = ""
    state.leg_ended_ts = ""
    state.closing_phase = False
    state.closing_started_unix = 0.0
    state.closing_report_task_id = ""
    state.close_sequence_done = False
    state.crash_count = 0
    state.teardown_timings_sec = {}
    state.begin_leg()
    return state.resumed_ts


def _resume_budget_lines(state: SharedState, *, extend_hours: float) -> list[str]:
    """Operator-facing notes about what budget the resumed leg actually has.

    Args:
        state: Loaded session state after :func:`_begin_resume_leg`.
        extend_hours: Hours this invocation granted via ``--extend-hours``.

    Returns:
        Lines to print, each already prefixed with ``  → ``.
    """
    elapsed_h = state.elapsed_minutes() / 60.0
    remaining_min = state.remaining_minutes()
    lines = [f"  → {elapsed_h:.2f}h charged to this session across every leg so far"]
    if remaining_min is None:
        lines.append("  → budget: unbounded")
        return lines
    lines.append(f"  → budget: {float(state.max_minutes) / 60.0:.2f}h total, {remaining_min / 60.0:.2f}h left")
    if extend_hours > 0.0:
        lines.append(f"  → --extend-hours added {extend_hours:.2f}h to the session budget")
    if remaining_min <= 0.0:
        lines.append(
            "  → WARNING: the budget is spent; this leg will close almost "
            "immediately. Pass --extend-hours to grant more, or start a fresh session"
        )
    return lines


def _reconcile_crash_count(state: SharedState, session_dir: Path) -> None:
    """Reconcile persisted ``crash_count`` (state.json + final.json) up to the live in-memory value."""
    live = int(getattr(state, "crash_count", 0) or 0)

    # state.json: reload, bump if stale, atomic re-save.
    try:
        disk_state = SharedState.load_or_init(session_dir)
        if int(disk_state.crash_count or 0) < live:
            disk_state.crash_count = live
            disk_state.save(session_dir)
    except Exception:
        log.exception("crash_count reconcile (state.json) failed (non-fatal)")

    try:
        from hyperloom.orchestrator.actions.executors.report import reconcile_final_crash_count

        reconcile_final_crash_count(session_dir, live)
    except Exception:
        log.exception("crash_count reconcile (final.json) failed (non-fatal)")


def _parse_conc_sweep_concs(args: argparse.Namespace, benchmark_mode: str) -> list[int]:
    """Parse ``--conc-sweep-concs`` into a list[int]; non-integers warned+dropped."""
    from hyperloom.orchestrator.kernel.conc_sweep import default_concs_for_mode

    fallback = default_concs_for_mode(benchmark_mode)
    raw = str(getattr(args, "conc_sweep_concs", "") or "").strip()
    if not raw:
        return fallback
    out: list[int] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except ValueError:
            log.warning("conc_sweep: ignoring non-integer CONC token %r", t)
    return out or fallback


def _read_failure_summary(session_dir: Path) -> dict | None:
    """Read ``reports/final.json``'s ``failure_summary`` block, if present."""
    try:
        from ..session.session_paths import reports_dir

        final_json = reports_dir(session_dir) / "final.json"
        data = json.loads(final_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    fs = data.get("failure_summary") if isinstance(data, dict) else None
    return fs if isinstance(fs, dict) else None


def _resolve_reference_recipe(
    args: argparse.Namespace,
) -> tuple[str, dict[str, str], str, dict[str, Any]]:
    """Resolve the reference launch recipe for a fresh launch."""
    source = (getattr(args, "reference_script", None) or "").strip()
    if not source:
        return ("", {}, "", {})

    framework = (os.environ.get("FRAMEWORK", "") or "sglang").strip().lower()
    from ..reference_script import parse_reference_script

    try:
        recipe = parse_reference_script(source, framework=framework)
    except Exception as exc:
        print(f"ERROR: --reference-script {source!r} could not be parsed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    controls = getattr(recipe, "launch_controls", {})
    if not recipe.server_args and not recipe.envs and not controls:
        print(
            f"ERROR: --reference-script {source!r} lifted no server flags and no env exports",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"Reference script: {source} ({len(recipe.server_args.split())} arg tokens, {len(recipe.envs)} env(s))")
    return (recipe.server_args, dict(recipe.envs), recipe.model or "", dict(controls))
