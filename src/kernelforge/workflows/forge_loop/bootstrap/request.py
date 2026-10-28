# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What one forge-loop run is asked to do, grouped by concern; the command line and in-process callers build it."""

from __future__ import annotations

import math
from dataclasses import dataclass

from kernelforge.config import AgentRuntimeOptions, CeilingSwitch, GpuOptions
from kernelforge.workflows.forge_loop.bootstrap.errors import InvalidOptionValue


@dataclass(frozen=True)
class CampaignTarget:
    """The kernel, driver and workspace a campaign optimizes, and the correctness threshold."""

    kernel: str | None
    driver: str | None
    workspace_dir: str
    snr_threshold: float


@dataclass(frozen=True)
class TimeBudget:
    """The campaign's runtime budget and its absolute deadline."""

    max_hours: float
    session_timeout_sec: int | None
    deadline_unix: float


@dataclass(frozen=True)
class TaskContext:
    """The kernel backend and the task context files handed to preparation."""

    kernel_backend: str | None
    program_md_file: str | None
    invocation_spec_file: str | None


@dataclass(frozen=True)
class RunArtifacts:
    """Where the run writes, how caches are bounded, and how the run is identified."""

    experiments_dir: str | None
    aiter_cache_max_gb: float
    experiment_id: str | None
    experience_id: str
    result_json: str | None


@dataclass(frozen=True)
class AgentRoleOptions:
    """Provider-facing knobs of the implementer, Analysis and supervisor roles."""

    permission_mode: str | None
    profile_timeout_sec: int
    supervisor_backend: str
    profiling: bool


@dataclass(frozen=True)
class LoopShape:
    """How the loop measures and how many candidates each round explores."""

    nproc_per_node: int
    lanes: int
    merge_stacking: bool
    bench_repeat: int
    commit_new_paths: tuple[str, ...]


@dataclass(frozen=True)
class TaskIdentity:
    """The task type and the names that identify the optimized operator."""

    task_type: str
    source_files: str
    target_functions: str
    framework: str
    operator_name: str


@dataclass(frozen=True)
class KnowledgeOptions:
    """The experience-KB and PR-KB switches, and the producer the records belong to."""

    experience_kb: bool
    kb_warmstart_enabled: bool
    producer: str
    pr_kb: bool | None


@dataclass(frozen=True)
class SpecialistProbeOptions:
    """Overrides for the planning specialists' scratch-tree probes."""

    specialist_probe: bool | None
    specialist_probe_max: int | None
    specialist_probe_budget_sec: float | None
    specialist_probe_scratch_root: str | None


@dataclass(frozen=True)
class ForgeLoopRequest:
    """Every option of one forge-loop run, grouped by concern."""

    campaign: CampaignTarget
    budget: TimeBudget
    git_branch: str | None
    gpu: GpuOptions
    task_context: TaskContext
    artifacts: RunArtifacts
    agent: AgentRuntimeOptions
    roles: AgentRoleOptions
    loop_shape: LoopShape
    roofline_ceiling: CeilingSwitch
    roofline_target: float
    target_mean_case_speedup: float
    wiki_guided: bool
    prepare_task: bool
    identity: TaskIdentity
    knowledge: KnowledgeOptions
    probe: SpecialistProbeOptions
    baseline_json: str
    resume: bool


def validate_roofline_target(target: float, ceiling_enabled: bool) -> float:
    """Check the attainment target is a fraction, and that a ceiling can reach it.

    A target above one is unreachable by construction -- attainment is bounded
    by the ceiling it divides by -- and a target without ``--roofline-ceiling
    on`` is a stop condition that can never fire. Both are refused rather than
    run, because either one leaves a campaign quietly ignoring the limit its
    operator thought they had set.
    """
    value = float(target or 0.0)
    if value <= 0:
        return 0.0
    if value > 1.0:
        raise InvalidOptionValue(
            f"--roofline-target {value} is a fraction of the ceiling, so it cannot exceed 1.0",
            param_hint="--roofline-target",
        )
    if not ceiling_enabled:
        raise InvalidOptionValue(
            "--roofline-target needs a ceiling to measure against; pass --roofline-ceiling on",
            param_hint="--roofline-target",
        )
    return value


def validate_wiki_guided(enabled: bool, knowledge_wiki_root: str) -> bool:
    """Check guided planning has a knowledge base to build its batch from.

    Without a knowledge base there is nothing to build a batch from, so the flag would plan exactly as without it
    while the operator believes otherwise; it is refused rather than run.
    """
    if enabled and not str(knowledge_wiki_root or "").strip():
        raise InvalidOptionValue(
            "--wiki-guided needs a knowledge base to build its batch from; set FORGE_KNOWLEDGE_WIKI_ROOT",
            param_hint="--wiki-guided",
        )
    return bool(enabled)


_SPEEDUP_TARGET_OPTION = "--target-mean-case-speedup"


def validate_speedup_target(target: float) -> float:
    """Check the speedup target is a finite speedup the baseline has not already met; zero turns it off.

    The baseline's mean case speedup is one by definition, so any other target
    at or below one, a negative one included, would end every campaign before
    its first round, and a target that is not finite can never be reached.
    Both are refused rather than run: a mistyped target must not leave a
    campaign without the stop its operator believes it has.
    """
    value = float(target or 0.0)
    if value == 0:
        return 0.0
    if not math.isfinite(value):
        raise InvalidOptionValue(
            f"{_SPEEDUP_TARGET_OPTION} {value} can never be reached; give a finite speedup above 1.0, or 0 for none",
            param_hint=_SPEEDUP_TARGET_OPTION,
        )
    if value <= 1.0:
        raise InvalidOptionValue(
            f"{_SPEEDUP_TARGET_OPTION} {value} is at or below 1.0, which the baseline already meets; "
            "give a speedup above 1.0, or 0 for none",
            param_hint=_SPEEDUP_TARGET_OPTION,
        )
    return value


__all__ = [
    "AgentRoleOptions",
    "CampaignTarget",
    "ForgeLoopRequest",
    "KnowledgeOptions",
    "LoopShape",
    "RunArtifacts",
    "SpecialistProbeOptions",
    "TaskContext",
    "TaskIdentity",
    "TimeBudget",
    "validate_roofline_target",
    "validate_speedup_target",
    "validate_wiki_guided",
]
