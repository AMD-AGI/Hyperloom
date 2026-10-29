# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``forge-loop`` command: parse one invocation, run it, and print its result for the caller that started it."""

from __future__ import annotations

import sys

import click

from kernelforge.cli.option_groups import option_group
from kernelforge.cli.options import (
    _agent_runtime_options,
    _gpu_options,
    _roofline_ceiling_option,
    _validate_max_hours,
    _validate_optional_agent_provider,
)
from kernelforge.evaluation.scoring import DEFAULT_SNR_THRESHOLD_DB
from kernelforge.loop.api import (
    ForgeLoopError,
    ForgeLoopUsageError,
    InvalidOptionValue,
    TaskPreparationFailed,
    format_forge_loop_result,
    run_forge_loop,
)
from kernelforge.loop.bootstrap.request import (
    AgentRoleOptions,
    CampaignTarget,
    ForgeLoopRequest,
    KnowledgeOptions,
    LoopShape,
    RunArtifacts,
    SpecialistProbeOptions,
    TaskContext,
    TaskIdentity,
    TimeBudget,
)

ANALYSIS_TIMEOUT_SEC = 7200


@click.command("forge-loop")
@option_group(
    "request",
    ForgeLoopRequest,
    option_group(
        "campaign",
        CampaignTarget,
        click.option("--kernel", default=None, help="Fresh campaign: kernel file to optimize"),
        click.option("--driver", default=None, help="Fresh campaign: validation/bench driver"),
        click.option("--workspace", "workspace_dir", required=True, help="Git workspace dir"),
        click.option(
            "--snr-threshold",
            default=DEFAULT_SNR_THRESHOLD_DB,
            type=float,
            help="Fresh campaign: threshold in dB the driver's correctness "
            "suite must clear (stored immutably in the campaign config; "
            "ignored on --resume). A KEEP needs this and a measured gain "
            "over the incumbent; assembly adds the task's own suite.",
        ),
    ),
    option_group(
        "budget",
        TimeBudget,
        click.option(
            "--max-hours",
            default=1.0,
            type=float,
            callback=_validate_max_hours,
            help="Max runtime hours (default: 1.0, minimum 1.0). Budgets "
            ">2 hours enable Analysis profiling and, for single-lane "
            "rounds, Plan Critic review.",
        ),
        click.option(
            "--session-timeout-sec",
            default=None,
            type=click.IntRange(min=1),
            help="Wall-clock budget for one implementer session (seconds). Overrides "
            "the value computed from --max-hours; the claude backend cuts a "
            "session at this deadline and the session is told about it.",
        ),
        click.option(
            "--deadline-unix",
            default=0.0,
            type=float,
            help="Absolute UNIX deadline shared by preparation and optimization.",
        ),
    ),
    click.option(
        "--git-branch",
        default=None,
        help="Fresh campaign: development branch to optimize on (checked out "
        "before the immutable campaign config is snapshotted).",
    ),
    _gpu_options("Hardware SKU for KB identities, e.g. mi355x"),
    option_group(
        "task_context",
        TaskContext,
        click.option(
            "--kernel-backend",
            default=None,
            help=("Fresh campaign: kernel backend override. Unsupported kernel backends fall back to flydsl."),
        ),
        click.option(
            "--program-md-file", default=None, help="Fresh campaign: optional task context copied into the campaign"
        ),
        click.option(
            "--invocation-spec-file",
            default=None,
            help="Path to a Hyperloom invocation-spec JSON used by task preparation.",
        ),
    ),
    option_group(
        "artifacts",
        RunArtifacts,
        click.option(
            "--experiments-dir",
            default=None,
            help="Diagnostics/checkpoint root (profiles, optimization_potential, "
            "tracker checkpoint). Defaults to <workspace>/forge_experiments. "
            "Resume artifacts always live under the workspace regardless.",
        ),
        click.option(
            "--aiter-cache-max-gb",
            default=4.0,
            type=click.FloatRange(min=0.0),
            help=(
                "Per-attempt AITER cache soft limit in GiB (default: 4). "
                "LRU pruning targets 75% of the limit; 0 disables in-run pruning."
            ),
        ),
        click.option(
            "--experiment-id",
            default=None,
            help="Caller-owned experiment ID; recorded for external checkpoint recovery.",
        ),
        click.option(
            "--experience-id",
            default="",
            help="Unique KB run identity, independent of the checkpoint experiment ID.",
        ),
        click.option("--result-json", default=None, help="Write the result dict here (also printed)"),
    ),
    _agent_runtime_options,
    option_group(
        "roles",
        AgentRoleOptions,
        click.option("--permission-mode", default=None, help="Provider permission mode when supported"),
        click.option(
            "--profile-timeout-sec",
            default=ANALYSIS_TIMEOUT_SEC,
            type=int,
            help="Ceiling (seconds) for the single complete Analysis Agent "
            "session. The Agent persists phase and case artifacts for "
            "resume when the deadline is reached. Default 7200 (2 hours).",
        ),
        click.option(
            "--supervisor-backend",
            default="",
            callback=_validate_optional_agent_provider,
            help="Registered provider for the AVO supervisor. Omit to follow "
            "the provider of the Agent profile running the search, so one "
            "--agent-backend value controls every local agent.",
        ),
        click.option(
            "--profiling/--no-profiling",
            default=True,
            help="Allow Analysis hardware profiling and Implementer "
            "self-profiling guidance for long-horizon runs (>2 hours). "
            "Shorter runs keep Analysis static-only and omit that guidance. "
            "--no-profiling disables collection for every duration.",
        ),
    ),
    option_group(
        "loop_shape",
        LoopShape,
        click.option(
            "--nproc-per-node",
            default=1,
            type=click.IntRange(min=1),
            help="Ranks the driver self-launches via torchrun (collective tasks "
            "such as all-reduce). >1 profiles EVERY rank in its own "
            "rocprofv3 session, because wrapping the driver would only "
            "profile the launcher process, which runs no kernel. Default 1 "
            "(single-GPU, unchanged behavior).",
        ),
        click.option(
            "--lanes",
            default=3,
            type=click.IntRange(min=1, max=8),
            help="Implementer lanes per round. Above 1 the round's analysis is "
            "partitioned into that many non-overlapping plans, each run "
            "concurrently in its own workspace copy, and each candidate is "
            "measured on its own. Default 3: the lanes of a round run "
            "concurrently, so a lane costs a session rather than a share "
            "of the round's wall clock, and three is what the three "
            "specialist analyses can be divided into. The partition "
            "returns fewer when the evidence supports fewer. Above 1 "
            "needs a provider that declares session_env and is refused on "
            "one that does not; a provider without stop_hooks runs and is "
            "warned, because it can waste a round but not misreport one.",
        ),
        click.option(
            "--merge-stacking/--no-merge-stacking",
            default=True,
            help="Once consecutive iterations stop producing a new best, spend "
            "one iteration measuring two archived rejected gains applied "
            "together, chosen for winning on different cases. Costs a "
            "measurement but no Implementer session. Default on; this "
            "applies at every --lanes setting, so turn it off to compare "
            "against a run that predates it.",
        ),
        click.option(
            "--bench-repeat",
            default=1,
            type=click.IntRange(min=1),
            help="How many times each bench repeats its measurement in-process, "
            "reporting the per-case median. Default 1 (single shot). >1 "
            "shrinks run-to-run spread. Requires a driver that accepts "
            "--repeat; the flag is omitted entirely when this is 1.",
        ),
        click.option(
            "--commit-new-path",
            "commit_new_paths",
            multiple=True,
            help="Workspace-relative path or glob naming a file the agent may "
            "CREATE and still have committed with a KEEP (repeatable, e.g. "
            "--commit-new-path configs/*.json). Untracked files are "
            "otherwise never staged and never removed by a REVERT. A '*' "
            "does not cross a directory separator and '**' is rejected; "
            "name each level instead. Protected measurement paths are "
            "never admitted however they are spelled. Immutable per "
            "campaign: it is snapshotted into campaign_config.json and "
            "read back on --resume.",
        ),
    ),
    _roofline_ceiling_option(
        "Estimate the per-shape theoretical achievable latency the campaign "
        "measures its attainment against. 'off' (default) skips it. 'on' estimates "
        "it on a fresh campaign once the baseline is measured, which costs a "
        "profiler pass and an analyst session, and on --resume reuses the ceiling "
        "the campaign already estimated, estimating again only when that report "
        "can no longer be read. Pass it on every --resume that should keep it."
    ),
    click.option(
        "--roofline-target",
        default=0.0,
        type=float,
        help="Stop the campaign once mean per-case attainment (ceiling / measured, "
        "equal-weight across scored cases) reaches this fraction, e.g. 0.86. "
        "Requires a ceiling. Zero (default) disables the gate: a ceiling is an "
        "estimate whose arithmetic nothing checks, so stopping on one is opt-in. "
        "It never affects KEEP, which stays a measurement.",
    ),
    click.option(
        "--prepare-task/--no-prepare-task",
        default=True,
        help="Pre-loop task preparation (default on, fresh campaigns only). "
        "Before the loop, run a deterministic preflight of the driver "
        "against the loop's stdout contract; if it fails invoke ONE agent "
        "that authors/repairs the graph-timed measurement driver (never "
        "the kernel/source), then re-check. Skipped on --resume, whose "
        "driver contract is already fixed by the campaign.",
    ),
    option_group(
        "identity",
        TaskIdentity,
        click.option(
            "--task-type",
            default="",
            help="Task type (e.g. flydsl2flydsl, repository, image_kernel). "
            "'repository'/'image_kernel' enable multi-file / whole-repo "
            "handling; anything else keeps the single-file behavior.",
        ),
        click.option(
            "--source-files",
            default="",
            help="Comma/newline-separated implementation entry points used for "
            "orientation, profiling, JIT hints, and KB identity. This is not "
            "an edit allowlist; --kernel remains the anchor.",
        ),
        click.option(
            "--target-functions",
            default="",
            help="Comma-separated target kernel/function hints. Used for PMC "
            "filtering, source mapping, and agent orientation; it does not "
            "restrict which functions may be edited.",
        ),
        click.option(
            "--framework",
            default="",
            help="Explicit framework identity for the experience KB slug "
            "(vllm/sglang/aiter, or 'standalone' for a framework-less "
            "file). Authoritative when given; otherwise inferred from the "
            "file that defines the target operation, falling back to "
            "'unknown' when no known owner is found.",
        ),
        click.option(
            "--operator-name",
            default="",
            help="Logical operator identity used by profiling and the "
            "experience page key (for example, the traced operation name).",
        ),
    ),
    option_group(
        "knowledge",
        KnowledgeOptions,
        click.option(
            "--experience-kb/--no-experience-kb",
            default=True,
            help="Read and publish forge-loop experience KB entries (default on). "
            "Internal callers with their own KB lifecycle, such as "
            "forge-rewrite-by-flydsl, disable this explicitly.",
        ),
        click.option(
            "--kb-warmstart/--no-kb-warmstart",
            "kb_warmstart_enabled",
            default=True,
            help="Apply the best matching KB solution before iteration 1 (default on, "
            "and only with --experience-kb). A caller that prepares the workspace "
            "itself, such as forge-fuse, turns this off to keep publishing while "
            "never replaying a stored patch over a tree it already staged.",
        ),
        click.option(
            "--producer",
            default="",
            help="System owning the candidate stream these records belong to (default: "
            "the forge-loop's own). A producer has its own index in the KB "
            "identity scheme, so a pipeline driving this command as a subprocess "
            "can keep its records out of the kernel campaigns' ranking.",
        ),
        click.option(
            "--pr-kb/--no-pr-kb",
            default=None,
            help="Inject upstream pull-request references from the PR Monitor "
            "as Implementer prior knowledge (default off). Falls back to "
            "PR_KB_ENABLE when unset.",
        ),
    ),
    option_group(
        "probe",
        SpecialistProbeOptions,
        click.option(
            "--specialist-probe/--no-specialist-probe",
            default=None,
            help="Let the read-only planning specialists measure one variant per probe "
            "in a scratch tree, instead of only arguing about a dispatch constant "
            "(default on). Each probe re-runs the workspace driver for one case "
            "with declared constants overridden; it queues on the same device lock "
            "the fan-out lanes take, and never touches the canonical tree. Falls "
            "back to FORGE_SPECIALIST_PROBE when unset.",
        ),
        click.option(
            "--specialist-probe-max",
            default=None,
            type=click.IntRange(min=1),
            help="Probes ONE analysis round may make in total, shared by every "
            "specialist it dispatches (default 6). Every call counts, including "
            "one that is refused. Falls back to FORGE_SPECIALIST_PROBE_MAX when "
            "unset.",
        ),
        click.option(
            "--specialist-probe-budget-sec",
            default=None,
            type=click.FloatRange(min=1.0),
            help="Seconds on the device ONE analysis round may spend probing, shared by "
            "every specialist it dispatches (default 600). Cut down further at "
            "call time so no probe can leave a specialist without the time to "
            "write its analysis. Falls back to FORGE_SPECIALIST_PROBE_BUDGET_SEC "
            "when unset.",
        ),
        click.option(
            "--specialist-probe-scratch-root",
            default=None,
            help="Where the round scratch trees are created. Must be absolute and lie "
            "outside the workspace. Default: <experiments-dir>/specialist_probe, "
            "or a sibling of the workspace when that would land inside it. Falls "
            "back to FORGE_SPECIALIST_PROBE_SCRATCH_ROOT when unset.",
        ),
    ),
    click.option(
        "--baseline-json",
        default="",
        help="JSON file holding a scoring anchor measured outside this loop: "
        '{"wall_ms": <float>, "case_times": {"<case id>": <ms>}}. Every speedup '
        "this run reports then divides by those per-case times instead of by the "
        "kernel the run starts from, which is what a caller wants when that "
        "kernel already replaced something else (a rewrite's port replacing its "
        "source). The loop still benches the starting kernel, but as the search "
        "start rather than the anchor. Omit it and the loop anchors on its own "
        "first bench, as it always has.",
    ),
    click.option("--resume", is_flag=True, help="Resume the campaign stored in the exact workspace"),
)
def forge_loop(request: ForgeLoopRequest):
    """Run ONE Forge IterationLoop as a standalone subprocess (CLI-ized kernel backend)."""
    try:
        run_forge_loop(request, publish_result=lambda result: click.echo(format_forge_loop_result(result)))
    except TaskPreparationFailed as failure:
        # The published result already carries the reason; the status alone tells the caller it failed.
        sys.exit(failure.exit_code)
    except InvalidOptionValue as error:
        raise click.BadParameter(str(error), param_hint=error.param_hint or None) from error
    except ForgeLoopUsageError as error:
        raise click.UsageError(str(error)) from error
    except ForgeLoopError as error:
        failure = click.ClickException(str(error))
        failure.exit_code = error.exit_code
        raise failure from error
