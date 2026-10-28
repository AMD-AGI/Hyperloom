# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One forge-loop run, from the campaign it resolves to the result it publishes, with no command line attached."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from kernelforge.loop.bootstrap.errors import ForgeLoopError, ForgeLoopUsageError, InvalidOptionValue
from kernelforge.loop.bootstrap.lanes import LaneBinding
from kernelforge.loop.bootstrap.pr_references import _collect_pr_kb_context, _write_pr_provenance
from kernelforge.loop.bootstrap.preparation import (
    _AssemblyPreparation,
    _prepare_assembly_task,
    _prepare_measurement_driver,
)
from kernelforge.loop.bootstrap.reporter import _Implementer, _RunReporter
from kernelforge.loop.bootstrap.request import ForgeLoopRequest, SpecialistProbeOptions, validate_roofline_target
from kernelforge.loop.bootstrap.run_context import _ForgeDeadline, _ForgeRun
from kernelforge.loop.bootstrap.session_budget import _forge_session_timeout_sec, _is_long_horizon
from kernelforge.loop.bootstrap.warm_start import _read_kb_warm_start, _WarmStart
from kernelforge.agent_backends.runtime import agent_runtime
from kernelforge.config import Config, KnowledgeConfig, agent_runtime_overrides
from kernelforge.loop.knowledge_integration import kb_reference_program_md
from kernelforge.tools.durable_io import atomic_write_json
from kernelforge.tools.git import git_head
from kernelforge.tracker.schema import ExperimentSubject

if TYPE_CHECKING:
    from kernelforge.loop.agent_relay import AgentAvailabilitySnapshot, AgentRelayConfig
    from kernelforge.loop.bootstrap.relay import SearchActorStartup
    from kernelforge.loop.campaign_config import CampaignConfig, CampaignConfigStore
    from kernelforge.loop.campaign_setup import CampaignResolution
    from kernelforge.loop.iteration.iteration_types import IterationConfig
    from kernelforge.loop.runner import IterationLoop
    from kernelforge.orchestrator.agent import ImplementerSetup
    from kernelforge.roofline_ceiling.standalone import AnalystAgent

# A high runaway backstop, not a time bound.
FORGE_IMPLEMENTER_TURN_BACKSTOP = 2000


def _make_ceiling_estimator(
    *,
    workspace_dir: str,
    driver_script: str,
    source_files: Sequence[str],
    analyst: AnalystAgent,
):
    """Build the callable the loop uses to estimate a ceiling.

    Only ``--roofline-ceiling on`` asks for one, because an estimate costs a
    profiler pass and an analyst session. Whether it is called is the loop's
    decision: a fresh campaign estimates, and a resumed one reads back the
    ceiling it already estimated. The analyst runs in its own session under its
    own role, so the backend is resolved here, where the registry already
    lives, and the loop is handed a callable instead of a dependency on it.

    The benchmark the estimate is taken against is the one the campaign scores,
    built by ``bench_command`` rather than assembled again here: a ceiling
    derived against a differently timed region would not be comparable with the
    latencies it is divided by.
    """

    async def estimate(*, case_ids: Sequence[str], case_ms: dict[str, float]):
        from kernelforge.evaluation.benchmark.runner import BenchRun, bench_command
        from kernelforge.roofline_ceiling.analyst import AnalystSession, CeilingTask
        from kernelforge.roofline_ceiling.standalone import resolve_analyst_backend
        from kernelforge.roofline_ceiling.estimate import estimate_ceiling
        from kernelforge.roofline_ceiling.evidence import KnownCases, PerformanceRun

        return await estimate_ceiling(
            resolve_analyst_backend(analyst),
            CeilingTask(
                run=PerformanceRun(bench_command(BenchRun(driver_script)), workspace_dir),
                kernel_files=[str(path) for path in source_files],
                driver_script=driver_script,
                known=KnownCases(list(case_ids), dict(case_ms)),
            ),
            AnalystSession(model=analyst.model, timeout_sec=analyst.timeout_sec),
        )

    return estimate


def _load_external_baseline(path: str) -> tuple[float, dict[str, float]]:
    """Read the ``--baseline-json`` scoring anchor, rejecting one that cannot anchor a speedup.

    Every number the run publishes divides by these per-case times, so a malformed or non-positive entry is refused
    here rather than silently producing ratios no consumer can interpret.
    """
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise InvalidOptionValue(f"--baseline-json could not be read as JSON: {error}")
    if not isinstance(payload, dict):
        raise InvalidOptionValue("--baseline-json must hold a JSON object")
    try:
        wall_ms = float(payload.get("wall_ms"))
    except (TypeError, ValueError):
        raise InvalidOptionValue("--baseline-json needs a numeric 'wall_ms'")
    if not math.isfinite(wall_ms) or wall_ms <= 0.0:
        raise InvalidOptionValue("--baseline-json 'wall_ms' must be finite and positive")
    raw_cases = payload.get("case_times")
    if not isinstance(raw_cases, dict) or not raw_cases:
        raise InvalidOptionValue("--baseline-json needs a non-empty 'case_times' object")
    case_times: dict[str, float] = {}
    for case_id, value in raw_cases.items():
        try:
            ms = float(value)
        except (TypeError, ValueError):
            raise InvalidOptionValue(f"--baseline-json case {case_id!r} has a non-numeric time")
        if not math.isfinite(ms) or ms <= 0.0:
            raise InvalidOptionValue(f"--baseline-json case {case_id!r} must be finite and positive")
        case_times[str(case_id)] = ms
    return wall_ms, case_times


def _profile_timeout_override(env_timeout, profile_timeout_sec: int) -> int:
    """The Analysis workflow deadline, overridden by the environment when it names a usable one.

    This is the deadline resume tooling sets; ``--profile-timeout-sec`` is the
    default when the environment is unset.
    """
    if env_timeout is None or env_timeout.strip() == "":
        return profile_timeout_sec
    try:
        override = int(env_timeout)
    except ValueError as error:
        raise ForgeLoopError("FORGE_PROFILE_TIMEOUT_SEC must be an integer") from error
    if override <= 0:
        raise ForgeLoopError("FORGE_PROFILE_TIMEOUT_SEC must be positive")
    return override


def _reject_incompatible_loop_options(*, producer: str) -> None:
    """Refuse an option combination the loop cannot honour, before anything is set up."""
    if producer:
        from kernelforge.knowledge.kb_store.identity.kernel_recipe import KERNEL_RECIPE_PRODUCERS

        if producer not in KERNEL_RECIPE_PRODUCERS:
            raise ForgeLoopUsageError(f"--producer must be one of: {', '.join(sorted(KERNEL_RECIPE_PRODUCERS))}")


def _resumed_profile_id(state_path: Path) -> str:
    """The Agent profile a resumed campaign was running, or ``""`` for a fresh one."""
    from kernelforge.loop.run_state import read_run_state

    try:
        return read_run_state(state_path).agent_relay.active_profile_id
    except ValueError as error:
        raise ForgeLoopError(str(error)) from error


def _check_workspace_matches_resume(*, resume: bool, campaign_store) -> None:
    """Refuse a workspace that does not match what ``--resume`` says about it.

    Resuming needs the records a prior campaign left; starting fresh needs a
    workspace that has none, so a second campaign cannot be read as a
    continuation of the first.
    """
    campaign_root = campaign_store.root
    state_path = campaign_root / "run_state.json"
    has_run_artifacts = (
        state_path.exists()
        or (campaign_root / "events.jsonl").exists()
        or any((campaign_root / "candidates").glob("iter_*"))
    )
    if resume and not state_path.is_file():
        raise ForgeLoopError("--resume requires <workspace>/forge_experiments/run_state.json")
    if resume and not campaign_store.exists():
        raise ForgeLoopError("--resume requires <workspace>/forge_experiments/campaign_config.json")
    if not resume and has_run_artifacts:
        raise ForgeLoopError("workspace already contains a Forge campaign; pass --resume to continue it")


def _record_experiment_accounting(tracker, *, experiment_id, llm_usage: dict, kb_experience) -> None:
    """Record what this run spent and what the KB gave it, against the experiment record.

    Accounting is best-effort on both counts: neither may fail a run that has
    already produced its result.
    """
    if not experiment_id:
        return
    try:
        if llm_usage.get("calls"):
            tracker.set_llm_usage(experiment_id, llm_usage)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Warning: failed to record LLM usage for experiment {experiment_id}: {exc}", file=sys.stderr)
    try:
        tracker.set_kb_experience(experiment_id, kb_experience)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Warning: failed to record KB experience for experiment {experiment_id}: {exc}", file=sys.stderr)


def _ensure_recovery_experiment(tracker, caller_experiment_id: str, *, kernel: str) -> None:
    """Make sure the caller-owned recovery record exists, so a hard kill still has one to recover from."""
    if not caller_experiment_id:
        return
    try:
        tracker.get(caller_experiment_id)
    except FileNotFoundError:
        tracker.create(
            ExperimentSubject(Path(kernel).stem, description=f"External recovery channel for {caller_experiment_id}"),
            experiment_id=caller_experiment_id,
        )


def _program_context(
    resolution,
    campaign_store,
    *,
    resume: bool,
    experience_kb: bool,
    workspace_dir: str,
) -> str:
    """The program context the implementer is given, plus any KB reference a resume should carry."""
    if resolution.save_deferred:
        # The immutable config (and its program.md) is not on disk yet; use the in-memory program text captured from
        # --program-md-file (matching read_program_md's "" for a campaign without program context).
        program_md = resolution.program_text or ""
    else:
        try:
            program_md = campaign_store.read_program_md(resolution.campaign)
        except (OSError, ValueError) as error:
            raise ForgeLoopError(str(error)) from error
    if resume and experience_kb:
        reference_pointer = kb_reference_program_md(workspace_dir)
        if reference_pointer:
            program_md = program_md + "\n\n" + reference_pointer
    return program_md


def _resolve_campaign(request: ForgeLoopRequest, workspace: Path, agent_relay) -> CampaignResolution:
    """The stored campaign on resume, else one created from this invocation's options."""
    from kernelforge.loop.campaign_config import CampaignInputs
    from kernelforge.loop.campaign_setup import resolve_campaign

    try:
        return resolve_campaign(
            str(workspace),
            CampaignInputs(
                kernel=request.campaign.kernel,
                driver=request.campaign.driver,
                snr_threshold=request.campaign.snr_threshold,
                source_files=request.identity.source_files,
                program_md_file=request.task_context.program_md_file,
                target_functions=request.identity.target_functions,
                operator_name=request.identity.operator_name,
                producer=request.knowledge.producer,
                kernel_backend=request.task_context.kernel_backend or "",
                git_branch=request.git_branch or "",
                gpu_target=request.gpu.gpu_target or "",
                gpu_type=request.gpu.gpu_type,
                task_type=request.identity.task_type,
                framework=request.identity.framework,
                nproc_per_node=request.loop_shape.nproc_per_node,
                bench_repeat=request.loop_shape.bench_repeat,
                commit_new_paths=tuple(request.loop_shape.commit_new_paths),
                agent_relay=agent_relay,
            ),
            resume=request.resume,
            prepare_task=request.prepare_task,
        )
    except (OSError, ValueError) as error:
        raise ForgeLoopError(str(error)) from error


def _session_config(
    probe: SpecialistProbeOptions,
    campaign: CampaignConfig,
    workspace: Path,
    knowledge_config: KnowledgeConfig,
    agent_runtime_overrides: dict,
) -> Config:
    """The Config this session runs under: the campaign's GPU and producer, plus only the overrides asked for."""
    overrides = {"gpu_target": campaign.gpu_target, "gpu_type": campaign.gpu_type, "producer": campaign.producer}
    # Only what was actually asked for: an override present with a None value still wins over ``Config.from_env``'s
    # environment lookup, which is what made the FORGE_SPECIALIST_PROBE* variables dead on this path.
    for _name, _value in (
        ("specialist_probe", probe.specialist_probe),
        ("specialist_probe_max", probe.specialist_probe_max),
        ("specialist_probe_budget_sec", probe.specialist_probe_budget_sec),
        ("specialist_probe_scratch_root", probe.specialist_probe_scratch_root),
    ):
        if _value is not None:
            overrides[_name] = _value
    overrides.update(agent_runtime_overrides)
    if campaign.gpu_target:
        os.environ["GPU_TARGET"] = campaign.gpu_target
    return Config.from_env(workspace=str(workspace), knowledge_config=knowledge_config, **overrides)


def _isolate_aiter_cache(experiments_dir: Path, max_cache_gb: float, source_files: list[str]) -> None:
    """Give this attempt its own AITER runtime-build cache, so parallel forge processes never evict its kernels.

    The pristine BASELINE shard is seeded with the package's prebuilt .so so the task-preparation preflight imports
    warm modules instead of cold-compiling the CK instance-factory TU (>26 min on gfx950, past the preflight timeout).
    """
    from kernelforge.tools.aiter_cache import (
        activate_aiter_cache_for_sources,
        configure_aiter_cache_isolation,
        seed_prebuilt_modules,
    )

    aiter_cache = configure_aiter_cache_isolation(experiments_dir, max_cache_bytes=int(max_cache_gb * 1024**3))
    print(f"  [aiter-cache] isolated runtime builds under {aiter_cache.cache_root}")
    baseline_cache = activate_aiter_cache_for_sources(source_files)
    if baseline_cache is not None:
        seed_stats = seed_prebuilt_modules(baseline_cache.aiter_jit_dir)
        print(
            f"  [aiter-cache] seeded {seed_stats['seeded']} prebuilt module(s)"
            f" (skipped {seed_stats['skipped']}, errors {seed_stats['errors']})"
            f" from {seed_stats['src'] or 'n/a'}"
        )


def _persist_prepared_campaign(run: _ForgeRun, workspace: Path, program_text: str | None) -> _ForgeRun:
    """Fix a fresh campaign's driver digest and pristine base commit from the post-preparation tree, and save it."""
    import dataclasses
    import hashlib

    from kernelforge.loop.campaign_config import derive_campaign_implementation_contract

    campaign = run.campaign
    iter_config = run.iter_config
    prepared_driver_sha256 = hashlib.sha256(Path(iter_config.driver_script).read_bytes()).hexdigest()
    prepared_base_commit = git_head(str(workspace)) or campaign.base_commit
    prepared_signature, prepared_identity = derive_campaign_implementation_contract(
        workspace_dir=str(workspace),
        kernel_path=campaign.kernel_path,
        source_files=campaign.source_files,
        framework=campaign.framework,
        base_commit=prepared_base_commit,
    )
    campaign = dataclasses.replace(
        campaign,
        driver_sha256=prepared_driver_sha256,
        base_commit=prepared_base_commit,
        implementation_signature=prepared_signature,
        implementation_identity=prepared_identity,
    )
    iter_config.canonical_driver_sha256 = campaign.driver_sha256
    iter_config.campaign_base_commit = campaign.base_commit
    iter_config.implementation_signature = campaign.implementation_signature
    iter_config.implementation_identity = dict(campaign.implementation_identity)
    try:
        run.campaign_store.save(campaign, program_md=program_text)
    except (OSError, ValueError) as error:
        raise ForgeLoopError(str(error)) from error
    return dataclasses.replace(run, campaign=campaign)


def _build_loop_runner(run: _ForgeRun, source_files: list[str], session_timeout_sec: int) -> IterationLoop:
    """Construct the IterationLoop, refusing a resume whose stored state it cannot continue."""
    from kernelforge.loop.runner import IterationLoop
    from kernelforge.roofline_ceiling.standalone import AnalystAgent

    request = run.request
    loop_runner = IterationLoop(
        run.iter_config,
        run.tracker,
        run.config,
        resume=request.resume,
        ceiling_estimator=(
            _make_ceiling_estimator(
                workspace_dir=request.campaign.workspace_dir,
                driver_script=run.iter_config.driver_script,
                source_files=source_files,
                # The runtime the implementer lanes settle on is resolved further down, after this point, and the
                # analyst is a different role anyway: it reads the same provider/model ladder the standalone command
                # reads, so the raw selection is what it needs and an empty one means "take the default".
                analyst=AnalystAgent(
                    provider=request.agent.agent_backend or "",
                    model=request.agent.model or "",
                    timeout_sec=session_timeout_sec,
                    sandbox_mode=run.config.agent_sandbox_mode,
                ),
            )
            if request.roofline_ceiling.enabled
            else None
        ),
    )
    if request.resume:
        try:
            loop_runner.validate_resume_preflight()
        except (OSError, ValueError) as error:
            raise ForgeLoopError(str(error)) from error
    return loop_runner


@dataclass(frozen=True)
class _RunInputs:
    """What one run starts from, before its campaign is resolved."""

    request: ForgeLoopRequest
    workspace: Path
    knowledge_config: KnowledgeConfig
    deadline: _ForgeDeadline
    publish_result: Callable[[dict], None]


@dataclass(frozen=True)
class _OpenedCampaign:
    """The campaign this run works on, resolved once its workspace is locked."""

    store: CampaignConfigStore
    state_path: Path
    resolution: CampaignResolution
    agent_runtime_overrides: dict


@dataclass(frozen=True)
class _AgentSession:
    """The agent configuration this run's sessions share, its model relay, and which providers can start."""

    config: Config
    relay: AgentRelayConfig | None
    availability: AgentAvailabilitySnapshot | None
    profile_timeout_sec: int


@dataclass(frozen=True)
class _PreparedCampaign:
    """The run after the task was made measurable, and what assembly preparation bound it to."""

    run: _ForgeRun
    assembly: _AssemblyPreparation


@dataclass(frozen=True)
class _SearchPlan:
    """What the search starts with: the implementer setup, the planning roles, and what the KB and assembly gave."""

    setup: ImplementerSetup
    source_files: list[str]
    specialist_definitions: dict
    profiling_enabled: bool
    critic_enabled: bool
    warm_start: _WarmStart
    assembly: _AssemblyPreparation


def run_forge_loop(request: ForgeLoopRequest, *, publish_result: Callable[[dict], None]) -> dict:
    """Run one forge-loop campaign and return its result.

    ``publish_result`` receives the result -- or, when task preparation fails, the failure -- as soon as it exists,
    before the run's PR provenance is written. A run that is refused raises :class:`ForgeLoopError`.
    """
    try:
        knowledge_config = KnowledgeConfig.from_env()
    except ValueError as exc:
        raise ForgeLoopError(str(exc)) from exc

    _reject_incompatible_loop_options(producer=request.knowledge.producer)

    from kernelforge.loop.run_state import WorkspaceLock, WorkspaceLockError

    # Absolute deadline shared by task preparation and optimization.
    deadline = _ForgeDeadline.for_budget(request.budget)

    if request.roles.profile_timeout_sec <= 0:
        raise ForgeLoopError("--profile-timeout-sec must be positive")

    workspace = Path(request.campaign.workspace_dir).resolve()
    workspace_lock = WorkspaceLock(workspace / "forge_experiments" / "workspace.lock")
    try:
        workspace_lock.acquire()
    except WorkspaceLockError as error:
        raise ForgeLoopError(str(error)) from error
    try:
        return _run_campaign(_RunInputs(request, workspace, knowledge_config, deadline, publish_result))
    finally:
        workspace_lock.release()


def _run_campaign(inputs: _RunInputs) -> dict:
    """The campaign under the workspace lock: resolve it, make it measurable, search it, and publish what it found."""
    request = inputs.request
    long_horizon = _is_long_horizon(request.budget.max_hours)
    opened = _open_campaign(inputs)
    session = _agent_session(inputs, opened)
    run = _new_run(inputs, opened, session, _iteration_config(inputs, opened.resolution.campaign))
    prepared = _prepare_campaign(inputs, run, opened)
    run, assembly = prepared.run, prepared.assembly

    # Resolved before the loop is built: the ceiling estimator captures it, and the raw option is ``None`` by default.
    session_timeout_sec = _forge_session_timeout_sec(request.budget.max_hours, request.budget.session_timeout_sec)

    # Construct the loop only after task preparation has resolved the profiling contract; IterationLoop snapshots that
    # readiness in its runtime state.
    loop_runner = _build_loop_runner(run, assembly.source_files, session_timeout_sec)

    warm_start = _read_kb_warm_start(run, assembly.program_md, assembly.kb_warmstart_enabled)
    # Source mapping, profiling, profile analysis, and potential analysis are produced together by the commit-bound
    # Analysis Agent.
    run.iter_config.program_md = warm_start.program_md

    profiling_enabled = bool(request.roles.profiling and long_horizon)
    from kernelforge.orchestrator.orchestration import default_specialist_definitions

    plan = _SearchPlan(
        setup=_implementer_setup(run, assembly.source_files, session_timeout_sec, profiling_enabled),
        source_files=assembly.source_files,
        specialist_definitions=default_specialist_definitions(),
        profiling_enabled=profiling_enabled,
        critic_enabled=bool(long_horizon),
        warm_start=warm_start,
        assembly=assembly,
    )
    actors = _start_actors(run, session, opened, plan)
    return _publish(run, loop_runner, _search(run, loop_runner, session, actors, plan))


def _open_campaign(inputs: _RunInputs) -> _OpenedCampaign:
    """Resolve the campaign this run creates or resumes, and the agent overrides its sessions start from."""
    from kernelforge.loop.bootstrap.relay import resolve_relay_policy
    from kernelforge.loop.campaign_config import CampaignConfigStore

    request, workspace = inputs.request, inputs.workspace
    campaign_store = CampaignConfigStore(str(workspace))
    _check_workspace_matches_resume(resume=request.resume, campaign_store=campaign_store)
    overrides = agent_runtime_overrides(request.agent)
    requested_relay_policy = resolve_relay_policy(
        Config.from_env(workspace=str(workspace), knowledge_config=inputs.knowledge_config, **overrides)
    )
    return _OpenedCampaign(
        store=campaign_store,
        state_path=campaign_store.root / "run_state.json",
        resolution=_resolve_campaign(request, workspace, requested_relay_policy),
        agent_runtime_overrides=overrides,
    )


def _agent_session(inputs: _RunInputs, opened: _OpenedCampaign) -> _AgentSession:
    """The session config, the model relay it resumes and which of its providers can start, and the output roots."""
    from kernelforge.loop.bootstrap.relay import probe_agent_profiles, resolve_session_relay

    request, workspace = inputs.request, inputs.workspace
    campaign = opened.resolution.campaign
    # Resume tooling may override the complete Analysis workflow deadline via FORGE_PROFILE_TIMEOUT_SEC;
    # --profile-timeout-sec is the default when the env is unset.
    profile_timeout_sec = _profile_timeout_override(
        os.environ.get("FORGE_PROFILE_TIMEOUT_SEC"), request.roles.profile_timeout_sec
    )
    config = _session_config(
        request.probe, campaign, workspace, inputs.knowledge_config, opened.agent_runtime_overrides
    )
    try:
        session_relay = resolve_session_relay(campaign.agent_relay, config)
    except ValueError as error:
        raise ForgeLoopError(str(error)) from error
    availability = probe_agent_profiles(session_relay, config=config, workspace=str(workspace))
    # Two roots by design: resume artifacts (run_state/candidates/best) always live under
    # <workspace>/forge_experiments (campaign_root); diagnostics and the external-recovery checkpoint go to
    # --experiments-dir when the caller supplies a distinct one (e.g. Hyperloom's output dir), else campaign_root.
    experiments_dir = request.artifacts.experiments_dir
    config.experiments_dir = Path(experiments_dir).resolve() if experiments_dir else opened.store.root
    config.experiments_dir.mkdir(parents=True, exist_ok=True)
    source_files = [str((workspace / path).resolve()) for path in campaign.source_files]
    _isolate_aiter_cache(config.experiments_dir, request.artifacts.aiter_cache_max_gb, source_files)
    return _AgentSession(
        config=config, relay=session_relay, availability=availability, profile_timeout_sec=profile_timeout_sec
    )


def _iteration_config(inputs: _RunInputs, campaign: CampaignConfig) -> IterationConfig:
    """The loop's configuration: the campaign's task and measurement semantics, and this invocation's budget."""
    from kernelforge.loop.iteration.iteration_types import IterationConfig

    request, workspace, deadline = inputs.request, inputs.workspace, inputs.deadline
    # Measurement semantics come from the campaign, never from this invocation's defaults.
    iter_config = IterationConfig(
        kernel_file=str((workspace / campaign.kernel_path).resolve()),
        driver_script=str((workspace / campaign.driver_path).resolve()),
        canonical_driver_sha256=campaign.driver_sha256,
        campaign_base_commit=campaign.base_commit,
        snr_threshold=campaign.snr_threshold,
        max_time_hours=max(0.05, request.budget.max_hours),
        deadline_unix=deadline.deadline_unix - deadline.finalize_reserve_sec,
        git_branch=campaign.git_branch,
        workspace_dir=request.campaign.workspace_dir,
        experiment_id=request.artifacts.experiment_id or "",
        backend=campaign.kernel_backend.split("-", 1)[0],
        kernel_backend=campaign.kernel_backend,
        task_type=campaign.task_type,
        source_files=[str((workspace / path).resolve()) for path in campaign.source_files],
        target_functions=list(campaign.target_functions),
        # Operator E2E time share (percent) for the baseline potential estimator.
        operator_name=campaign.operator_name,
        implementation_signature=campaign.implementation_signature,
        implementation_identity=dict(campaign.implementation_identity),
        # Rank count for collective tasks; selects the per-rank profiling backend.
        nproc_per_node=campaign.nproc_per_node,
        # Measurement fidelity: in-process repeats within each independent bench.
        bench_repeat=campaign.bench_repeat,
        lanes=request.loop_shape.lanes,
        merge_stacking=request.loop_shape.merge_stacking,
        # New files a KEEP may carry; a REVERT removes exactly the same set. From the campaign, so a resumed session
        # can still ship or remove the new file an earlier session was configured to.
        commit_new_paths=list(campaign.commit_new_paths),
        # The attainment of the per-shape ceiling that ends the run; the ceiling itself is estimated inside the loop.
        roofline_target=validate_roofline_target(request.roofline_target, request.roofline_ceiling.enabled),
    )
    if request.baseline_json:
        # The anchor every speedup divides by, and the wall time published beside it, both come from the caller's
        # measurement. ``baseline_wall_ms`` is deliberately left unset so the loop still benches the kernel it starts
        # from -- that bench is the search start and the first incumbent, not the anchor.
        external_baseline_ms, external_baseline_case_times = _load_external_baseline(request.baseline_json)
        iter_config.baseline_case_times = dict(external_baseline_case_times)
        iter_config.pristine_baseline_wall_ms = external_baseline_ms
    return iter_config


def _new_run(
    inputs: _RunInputs, opened: _OpenedCampaign, session: _AgentSession, iter_config: IterationConfig
) -> _ForgeRun:
    """The run's collaborators: its experiment tracker, its usage ledger, and the recovery experiment a caller reads."""
    from kernelforge.tracker import ExperimentTracker, UsageAccumulator, UsageLedgerFile

    config = session.config
    tracker = ExperimentTracker(config.experiments_dir)
    # Published at a fixed path from the first call onwards, so a caller that never asked for --result-json, or that
    # loses this process outright, can still read what the run spent.
    usage_ledger = UsageLedgerFile(config.experiments_dir)
    usage = UsageAccumulator(on_update=usage_ledger.publish)
    usage_ledger.publish(usage.totals())

    run = _ForgeRun(
        request=inputs.request,
        campaign=opened.resolution.campaign,
        campaign_store=opened.store,
        config=config,
        iter_config=iter_config,
        tracker=tracker,
        usage=usage,
        deadline=inputs.deadline,
        publish_result=inputs.publish_result,
    )
    # The caller-owned experiment ID is an EXTERNAL recovery channel, deliberately independent of the internal
    # per-segment experiment identity (each resume segment gets a fresh ID so the campaign parent/child chain stays
    # intact).
    _ensure_recovery_experiment(tracker, run.caller_experiment_id, kernel=iter_config.kernel_file)
    return run


def _prepare_campaign(inputs: _RunInputs, run: _ForgeRun, opened: _OpenedCampaign) -> _PreparedCampaign:
    """Make the task measurable before the loop snapshots it: the driver, a deferred campaign save, and assembly."""
    request = inputs.request
    resolution = opened.resolution
    program_md = _program_context(
        resolution,
        opened.store,
        resume=request.resume,
        experience_kb=request.knowledge.experience_kb,
        workspace_dir=request.campaign.workspace_dir,
    )

    # Pre-loop task preparation (fresh campaigns only): ensure the driver conforms to the loop's stdout contract
    # BEFORE base_sha is captured, so any scaffolding the prep step commits becomes part of pristine (never the
    # solution diff).
    if request.prepare_task and not request.resume:
        _prepare_measurement_driver(run, program_md)

    if resolution.save_deferred:
        run = _persist_prepared_campaign(run, inputs.workspace, resolution.program_text)

    return _PreparedCampaign(run=run, assembly=_prepare_assembly_task(run, program_md))


def _implementer_setup(
    run: _ForgeRun, source_files: list[str], session_timeout_sec: int, profiling_enabled: bool
) -> ImplementerSetup:
    """What every implementer session is given: the task, the in-session gate it must pass, and its own budget."""
    from kernelforge.loop.gate_settings import GateMeasurement
    from kernelforge.orchestrator.agent import ImplementerSetup, ImplementerTask, SessionOptions

    iter_config, config = run.iter_config, run.config
    # A turn cap never bounded time: it fired on 2.2% of sessions, so a session that neither converged nor capped ran
    # until something outside killed it.
    config.max_turns = FORGE_IMPLEMENTER_TURN_BACKSTOP
    print(
        f"  Implementer session budget: {session_timeout_sec}s "
        f"(campaign budget {run.request.budget.max_hours:g}h; turn backstop {config.max_turns})"
    )
    # PR references are independent of the experience-KB lifecycle.
    pr_task_context, pr_kb_repo = _collect_pr_kb_context(run)
    return ImplementerSetup(
        ImplementerTask(
            program_md=iter_config.program_md,
            kernel_backend_name=iter_config.kernel_backend,
            pre_task_context=pr_task_context,
            pr_kb_repo=pr_kb_repo,
            task_type=iter_config.task_type,
            source_files=source_files,
            # Function names, not paths: nothing to rebind onto a lane.
            target_functions=iter_config.target_functions,
            profiling_enabled=profiling_enabled,
            commit_new_paths=iter_config.commit_new_paths,
        ),
        GateMeasurement(
            driver_script=iter_config.driver_script,
            snr_threshold=iter_config.snr_threshold,
            # No hard edit budget: a session making steady progress must not be cut off for editing a lot.
            max_blocks=10,
            stage_timeout_sec=iter_config.validate_stage_timeout_sec,
            bench_timeout_sec=iter_config.bench_timeout_sec,
            bench_repeat=iter_config.bench_repeat,
        ),
        SessionOptions(
            session_timeout_sec=session_timeout_sec,
            permission_mode=run.request.roles.permission_mode,
            agent_backend=agent_runtime(config).provider,
            # Hook-capable providers gate before stop; resumable providers apply the same canonical gate between turns.
            insession_gate=True,
        ),
    )


def _start_actors(
    run: _ForgeRun, session: _AgentSession, opened: _OpenedCampaign, plan: _SearchPlan
) -> SearchActorStartup:
    """Start the search actors of every relay member that can run, resuming on the profile the last session used."""
    from kernelforge.loop.bootstrap.relay import SearchActorBuildContext, start_search_actors

    request, iter_config = run.request, run.iter_config
    actor_builder = SearchActorBuildContext(
        config=run.config,
        setup=plan.setup,
        lane_binding=LaneBinding(iter_config.workspace_dir, iter_config.driver_script, plan.source_files),
        lanes=request.loop_shape.lanes,
        program_md=iter_config.program_md,
        specialist_definitions=plan.specialist_definitions,
        critic_enabled=plan.critic_enabled,
        usage=run.usage,
        supervisor_backend=request.roles.supervisor_backend,
    )
    try:
        return start_search_actors(
            session.relay,
            actor_builder,
            availability=session.availability,
            workspace=str(Path(iter_config.workspace_dir).resolve()),
            resume_profile_id=_resumed_profile_id(opened.state_path),
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ForgeLoopError(f"Agent startup availability check failed: {error}") from error


def _search(
    run: _ForgeRun, loop_runner: IterationLoop, session: _AgentSession, actors: SearchActorStartup, plan: _SearchPlan
) -> _RunReporter:
    """Run the loop to its end with the actors that started, and return the reporter of its result."""
    from kernelforge.loop.iteration.iteration_types import LoopAgents, LoopCallbacks
    from kernelforge.orchestrator.analysis import make_analysis_agent_service

    start_bundle = actors.start_bundle
    analysis_service = make_analysis_agent_service(
        config=actors.analysis_config(run.config),
        usage=run.usage,
        timeout_sec=session.profile_timeout_sec,
        profiling_enabled=plan.profiling_enabled,
    )
    print("  Analysis Agent: " + ("profiled" if plan.profiling_enabled else "static-only"))
    print(f"  Orchestration: enabled with parallel specialists ({', '.join(sorted(plan.specialist_definitions))})")
    print(
        "  Plan Critic: "
        + (
            "enabled (long-horizon, same backend/model)"
            if plan.critic_enabled
            else "disabled (requires --max-hours > 2)"
        )
    )

    reporter = _RunReporter(
        run=run,
        loop_runner=loop_runner,
        warm_start=plan.warm_start,
        assembly_preparation=plan.assembly.preparation,
        implementer=_Implementer(start_bundle.profile.provider, start_bundle.profile.model),
    )
    asyncio.run(
        loop_runner.run(
            LoopAgents(
                agent_fn=start_bundle.agent_fn,
                agent_factory=start_bundle.agent_factory if run.request.loop_shape.lanes > 1 else None,
                analysis_service=analysis_service,
                orchestration_service=start_bundle.orchestration_service,
                supervisor_fn=start_bundle.supervisor_fn,
                search_actors=actors.loop_search_actors,
            ),
            callbacks=LoopCallbacks(
                on_best_committed=reporter.checkpoint_on_best_committed,
                on_best_ready=reporter.publish_remote_best,
            ),
            usage=run.usage,
            workspace_lock_held=True,
        )
    )
    return reporter


def _publish(run: _ForgeRun, loop_runner: IterationLoop, reporter: _RunReporter) -> dict:
    """Publish the final best, account for the run, and hand its result to the caller before writing provenance."""
    kb_experience = reporter.publish_final_best(llm_summary=run.deadline.remaining(reserve=30.0) >= 60.0)
    loop_runner._checkpoint_llm_usage()
    experiment_id = getattr(loop_runner.experiment, "experiment_id", None)
    loop_runner.llm_usage = run.usage.totals()
    _record_experiment_accounting(
        run.tracker,
        experiment_id=experiment_id,
        llm_usage=loop_runner.llm_usage,
        kb_experience=kb_experience,
    )

    # Final result: full kb_experience.
    result = reporter.build_result(kb_experience=kb_experience)
    result_json = run.request.artifacts.result_json
    if result_json:
        atomic_write_json(result_json, result)
    run.publish_result(result)
    _write_pr_provenance(
        workspace_dir=run.request.campaign.workspace_dir,
        surfaced=run.iter_config.pr_reference_labels,
        winning_iteration=int(result.get("best_iteration") or 0),
        experiment_id=str(result.get("experiment_id") or ""),
    )
    return result
