# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry point for kernelforge."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import click

from kernelforge.llm.git import git
from kernelforge.config import Config
from kernelforge.knowledge.experience_store import (
    REMOTE_BACKEND_KB_STORE,
    KnowledgeConfig,
)
from kernelforge.knowledge.experience_integration import (
    WarmStartRollbackError,
    kb_reference_program_md,
    kb_read_status,
    kb_warmstart,
    mark_kb_reference_rejected,
    write_experience_to_kb,
)
from kernelforge.loop.recovery import (
    atomic_write_json,
    publish_warm_start_recovery,
    rollback_unpublished_warm_start,
)
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB

if TYPE_CHECKING:
    # Imported lazily at runtime to keep CLI startup off the knowledge stack.
    from kernelforge.knowledge.pr_monitor_refs import PRRefsResult


MIN_MAX_HOURS = 1.0  # a run shorter than this can't complete a productive campaign
LONG_HORIZON_THRESHOLD_HOURS = 2.0
# A high runaway backstop, not a time bound. The turn cap never bounded wall
# clock -- it fired on only 2.2% of sessions and a session could spend hours
# well under it -- so the wall-clock budget below is what ends a long session.
# This ceiling exists only to stop a truly pathological loop (an agent stuck
# retrying the same edit forever) from spending without limit; a healthy session
# hands off long before reaching it.
FORGE_IMPLEMENTER_TURN_BACKSTOP = 2000
# One implementer session's wall-clock budget is a FUNCTION of the campaign, not
# a fixed number. The fraction caps a single session at a slice of the campaign
# so one stuck session cannot eat a whole short run; the floor keeps even a 1h
# run's session long enough to read+edit+build+bench; the ceiling keeps a long
# overnight campaign admitting many sessions instead of a few marathons. Sized
# off the TOTAL budget, not what remains, because the worst runaways are the
# earliest iterations. See :func:`_forge_session_timeout_sec`.
FORGE_SESSION_BUDGET_FRACTION = 0.15
FORGE_SESSION_BUDGET_MIN_MINUTES = 90
FORGE_SESSION_BUDGET_MAX_MINUTES = 210
# End-to-end wall clock for every PR reference lookup of one run: preflight,
# repository listing, path probing, discovery, and detail enrichment.
PR_KB_BUDGET_SEC = 45.0
ANALYSIS_TIMEOUT_SEC = 7200
# Suffix naming one lane's private AITER build cache, placed beside the lane copy
# rather than inside it. Beside, because the campaign cache a lane reads today is
# outside the lane copy too: moving it inside would put every compiled artifact
# into the lane's own worktree, where a backend that requires a session to create
# no untracked files rejects the whole lane. The round's fan-out removes the
# directory holding the lane copies, and the cache with it.
_LANE_AITER_CACHE_SUFFIX = ".aiter-cache"


def _initial_remote_publication_state(warm: dict) -> dict:
    """Seed publication authority from an already-materialized KB warm-start."""
    commit = str(warm.get("applied_commit") or "")
    solution = str(warm.get("solution_slug") or "")
    if warm.get("applied") and commit and solution:
        return {
            "status": "published",
            "state": "materialized_from_remote",
            "source": "existing_warm_start_solution",
            "solution_slug": solution,
            "best_commit": commit,
            "pending_commit": "",
            "last_attempted_commit": "",
            "published_commit": commit,
            "last_result": {
                "written": False,
                "reason": "existing_warm_start_solution",
                "solution": solution,
            },
        }
    return {
        "status": "not_attempted",
        "state": "not_attempted",
        "source": "",
        "solution_slug": "",
        "best_commit": "",
        "pending_commit": "",
        "last_attempted_commit": "",
        "published_commit": "",
        "last_result": None,
    }


def _warm_start_publication_covers(state: dict, commit: str) -> bool:
    """Whether ``commit`` is already represented by the consumed KB solution."""
    return bool(
        commit
        and state.get("source") == "existing_warm_start_solution"
        and state.get("published_commit") == commit
        and not state.get("pending_commit")
    )


def _record_remote_publication_result(
    state: dict,
    *,
    commit: str,
    result: dict,
) -> None:
    """Apply one campaign publication attempt without erasing prior authority."""
    state["best_commit"] = commit
    state["last_result"] = result
    if result.get("written"):
        state["published_commit"] = commit
        state["pending_commit"] = ""
        state["status"] = "published"
        state["state"] = "published"
        state["source"] = "campaign_publication"
        state["solution_slug"] = str(result.get("solution") or "")
        return
    reason = str(result.get("reason") or "error")
    if reason in {
        "not_configured",
        "missing_gpu_type",
        "no_improvement",
        "empty_diff",
        "not_better_than_kb",
    }:
        state["pending_commit"] = ""
        state["status"] = reason
        state["state"] = reason
        state["source"] = "campaign_publication"
        return
    state["status"] = "pending_retry"
    state["state"] = "pending_retry"
    state["source"] = "campaign_publication"


def _remote_publication_view(state: dict, best_commit: str) -> dict:
    """Return the authoritative local-versus-remote best publication status."""
    local_best = str(best_commit or "")
    published = str(state.get("published_commit") or "")
    pending = str(state.get("pending_commit") or "")
    return {
        key: value
        for key, value in {
            **state,
            "best_commit": local_best,
            "local_best_commit": local_best,
            "pending_commit": pending,
            "published_commit": published,
            "latest_best_published": bool(local_best and published == local_best and not pending),
        }.items()
        if key != "last_result"
    }


def _persist_declared_spec(invocation_spec_file: str, driver: str) -> None:
    """Place the declared invocation spec beside the driver that reads it.

    Preparation does this for a driver it authors, and a driver that already
    conforms skips preparation entirely -- so without this the spec stays only
    wherever the operator passed it from. A driver that derives its cases from
    the task reads the spec while benchmarking, so it would be reading a path on
    a machine and at a time nobody controls: edited later, the measured suite
    changes silently, and a resumed campaign measures something its own baseline
    never did.

    Failing to place it is not fatal -- ``_materialize_invocation_spec`` refuses
    a destination the caller already owns, and the driver conformed without it.
    """
    if not invocation_spec_file:
        return
    from kernelforge.loop.task_preparer import _materialize_invocation_spec

    destination, _ = _materialize_invocation_spec(
        invocation_spec_file,
        Path(driver).resolve().parent,
    )
    if destination is None:
        print(
            f"  [prepare] could not place {invocation_spec_file} beside {driver}; "
            "the driver will only be able to read it from where it was passed"
        )


def _validate_max_hours(ctx, param, value):
    """Reject a runtime budget below the minimum a real campaign needs.

    The loop refuses to START an iteration once less than its configured
    ``IterationConfig.budget_reserve_sec`` remains, so a sub-floor budget leaves
    a uselessly small iteration window: the campaign would finalize after little
    or no work and still exit 0. Applied to the `forge-loop` command.
    """
    if value is not None and value < MIN_MAX_HOURS:
        raise click.BadParameter(
            f"must be >= {MIN_MAX_HOURS} (a forge run needs at least {MIN_MAX_HOURS:g} hour to be productive)"
        )
    return value


def _normalize_gpu_type(_ctx, _param, value):
    """Canonicalize the hardware SKU used in KB identities.

    An omitted option keeps the stable default so a campaign resumes the same
    way whatever the process environment says. An explicitly empty one is a
    caller reporting that it cannot name the card, and is passed through so the
    KB layer refuses rather than filing the run under a guess.
    """
    if value is None:
        return "mi355x"
    return str(value).strip().lower()


def _pr_kb_enabled(flag: bool | None) -> bool:
    """Resolve the PR KB switch: CLI flag wins, env is the fallback, default off."""
    if flag is not None:
        return bool(flag)
    return os.environ.get("PR_KB_ENABLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _git_remote_url(workspace: Path) -> str:
    """Return the origin URL, or ``""`` when it cannot be read."""
    try:
        result = git(
            "remote",
            "get-url",
            "origin",
            cwd=workspace,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        click.echo(f"Warning: failed to read git origin: {error}", err=True)
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _pr_refs_event_fields(reason: str, stats: dict) -> dict:
    """Build the PR refresh event appended after campaign initialization."""
    return {
        "position": "A",
        "reason": reason or "ok",
        "degraded_reason": stats.get("degraded_reason"),
        "candidates": stats.get("candidates"),
        "surfaced": stats.get("surfaced"),
        "injected_entries": stats.get("injected_entries"),
        "injected_bytes": stats.get("injected_bytes"),
        "http_calls": stats.get("http_calls"),
        "skipped_cached_empty": stats.get("skipped_cached_empty"),
        "from_snapshot": stats.get("from_snapshot"),
        "distill_absent": stats.get("distill_absent"),
        "distill_dropped": stats.get("distill_dropped"),
        "fallback_used": stats.get("fallback_used"),
        "relevance_dropped": stats.get("relevance_dropped"),
    }


def _collect_pr_references(
    *,
    workspace_dir: str,
    kernel_backend: str,
    git_remote: str,
    source_files: Iterable[str],
    operator_name: str,
    target_functions: Iterable[str],
    budget_sec: float,
) -> PRRefsResult | None:
    """Run position A, absorbing every recoverable failure into a warning.

    Returns None when the lookup could not run, so the campaign proceeds with
    no upstream references instead of inheriting this subsystem's failure.
    """
    from kernelforge.knowledge.pr_monitor_refs import (
        PR_KB_RECOVERABLE,
        collect_references,
    )

    try:
        return collect_references(
            workspace_dir=workspace_dir,
            kernel_backend=kernel_backend,
            git_remote=git_remote,
            source_files=source_files,
            operator_name=operator_name,
            target_functions=target_functions,
            budget_sec=budget_sec,
            # The campaign freshness guard runs inside the loop; until it passes
            # this invocation may not modify the workspace it was pointed at.
            persist=False,
        )
    except PR_KB_RECOVERABLE as error:
        print(f"  [pr-kb] unavailable ({type(error).__name__}: {error})")
        return None


def _write_pr_provenance(
    *,
    workspace_dir: str,
    surfaced: tuple[str, ...],
    winning_iteration: int,
    experiment_id: str = "",
) -> None:
    """Write exposure data for the references injected into this run.

    Runs after the result sentinel, so every failure degrades to a warning
    rather than changing the exit status of a finished run. Free-form lesson
    text is deliberately not parsed into adoption classifications.
    """
    if not surfaced:
        return

    from kernelforge.knowledge.pr_monitor_refs import (
        PR_KB_RECOVERABLE,
        write_provenance,
    )

    try:
        write_provenance(
            workspace_dir,
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "winning_iteration": winning_iteration,
                "surfaced": list(surfaced),
            },
        )
    except PR_KB_RECOVERABLE as error:
        click.echo(f"Warning: failed to write PR provenance: {error}", err=True)


def _forge_session_timeout_sec(max_hours: float, override_sec: int | None) -> int:
    """Wall-clock budget for one implementer session, in seconds.

    ``--session-timeout-sec`` (``override_sec``) wins when given; otherwise the
    budget is sized from the campaign per the constants above:
    ``min(MAX, max(MIN, FRACTION * total_campaign_minutes))``.
    """
    if override_sec is not None:
        return int(override_sec)
    total_min = float(max_hours) * 60.0
    session_min = min(
        FORGE_SESSION_BUDGET_MAX_MINUTES,
        max(
            FORGE_SESSION_BUDGET_MIN_MINUTES,
            FORGE_SESSION_BUDGET_FRACTION * total_min,
        ),
    )
    return int(round(session_min * 60.0))


def _is_long_horizon(max_hours: float) -> bool:
    """Whether one campaign session enables expensive long-horizon agents."""
    return float(max_hours) > LONG_HORIZON_THRESHOLD_HOURS


def _validate_agent_provider(ctx, param, value):
    """Validate a dynamic built-in or entry-point provider name."""
    if value is None:
        return None
    from kernelforge.agent_backends import get_agent_provider

    try:
        return get_agent_provider(value).name
    except ValueError as exc:
        raise click.BadParameter(str(exc), ctx=ctx, param=param) from exc


def _validate_fallback_provider(ctx, param, value):
    """Validate a fallback provider while allowing explicit disablement."""
    if value is not None and value.strip().lower() in {"", "none", "off"}:
        return ""
    return _validate_agent_provider(ctx, param, value)


def _validate_optional_agent_provider(ctx, param, value):
    """Validate an optional provider override, empty meaning "inherit"."""
    if value is None or not value.strip():
        return ""
    return _validate_agent_provider(ctx, param, value)


def _agent_runtime_options(func):
    """Attach provider-neutral Agent runtime options to one command."""
    decorators = (
        click.option("--model", default=None, help="Selected provider model"),
        click.option(
            "--agent-backend",
            default=None,
            callback=_validate_agent_provider,
            help="Registered local Agent provider",
        ),
        click.option(
            "--agent-cli",
            default=None,
            help="Provider executable path or command",
        ),
        click.option(
            "--agent-timeout-sec",
            type=click.IntRange(min=1),
            default=None,
            help="Timeout for one Agent session",
        ),
        click.option(
            "--agent-reasoning-effort",
            default=None,
            help="Provider reasoning effort value",
        ),
        click.option(
            "--agent-sandbox-mode",
            default=None,
            help="Provider sandbox mode",
        ),
        click.option(
            "--agent-fallback-provider",
            default=None,
            callback=_validate_fallback_provider,
            help="Fallback provider name, or 'none'",
        ),
        click.option(
            "--agent-precheck/--no-agent-precheck",
            default=None,
            help="Enable provider preflight and capability probe",
        ),
        click.option(
            "--agent-options-json",
            default=None,
            help="Provider extension options as a JSON object",
        ),
    )
    for decorator in reversed(decorators):
        func = decorator(func)
    return func


def _agent_runtime_overrides(
    *,
    model: str | None,
    agent_backend: str | None,
    agent_cli: str | None,
    agent_timeout_sec: int | None,
    agent_reasoning_effort: str | None,
    agent_sandbox_mode: str | None,
    agent_fallback_provider: str | None,
    agent_precheck: bool | None,
    agent_options_json: str | None,
) -> dict:
    """Convert provider-neutral CLI values into Config overrides."""
    values = {
        "agent_model": model,
        "agent_backend": agent_backend,
        "agent_cli": agent_cli,
        "agent_timeout_sec": agent_timeout_sec,
        "agent_reasoning_effort": agent_reasoning_effort,
        "agent_sandbox_mode": agent_sandbox_mode,
        "agent_fallback_provider": agent_fallback_provider,
        "agent_precheck": agent_precheck,
    }
    overrides = {key: value for key, value in values.items() if value is not None}
    if agent_options_json is not None:
        try:
            options = json.loads(agent_options_json)
        except json.JSONDecodeError as exc:
            raise click.BadParameter(
                f"invalid JSON: {exc}",
                param_hint="--agent-options-json",
            ) from exc
        if not isinstance(options, dict):
            raise click.BadParameter(
                "must be a JSON object",
                param_hint="--agent-options-json",
            )
        overrides["agent_options"] = options
    return overrides


@click.group()
@click.version_option(package_name="hyperloom-inference_optimizer")
def main():
    """Kernel Agents — Agentic GPU kernel development system."""
    pass


def _lane_workspace_path(value: str, *, lane_dir: str, workspace_dir: str, label: str) -> str:
    """Rebind one canonical workspace path onto a lane's own copy of it.

    A lane is handed paths in order to edit them, so a canonical path handed to a
    lane is an edit into the campaign workspace that the lane's own diff will
    never report. A path that cannot be rebound is refused rather than passed
    through as it is.

    A relative path is read against the campaign workspace, which is what it
    names. Resolving it against the process cwd instead would silently rebind
    whatever happens to sit at the same relative position under wherever forge
    was launched from -- or, more often, refuse a path that was perfectly valid.
    """
    workspace_root = Path(workspace_dir).resolve()
    value_path = Path(value)
    resolved = value_path.resolve() if value_path.is_absolute() else (workspace_root / value_path).resolve()
    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError as error:
        raise ValueError(
            f"lane isolation cannot rebind the {label} {resolved} onto "
            f"{lane_dir}: it is outside the campaign workspace {workspace_root}"
        ) from error
    return str(Path(lane_dir).resolve() / relative)


def _assert_lane_session_cwd(*, kernel_path: str, workspace: str, lane_dir: str) -> None:
    """Fail unless every directory the session could start in is inside the lane.

    An Implementer session runs in the kernel file's directory, or in
    ``config.workspace`` when the resolved provider requires the workspace as its
    cwd -- the codex provider does. Which of the two it takes is decided inside
    the session from the backend it ends up with, so both are checked here from
    the same two inputs the session reads.

    A session that starts outside its lane writes its edits and runs its shell
    commands in the canonical workspace, where the lane's own diff will never
    report them and the tree every lane shares is measured instead.
    """
    lane_root = Path(lane_dir).resolve()
    candidates = {"kernel directory": Path(kernel_path).resolve().parent}
    if workspace:
        candidates["configured workspace"] = Path(workspace).resolve()
    outside = sorted(
        f"{label} {candidate}"
        for label, candidate in candidates.items()
        if candidate != lane_root and lane_root not in candidate.parents
    )
    if outside:
        raise ValueError(f"lane session would run outside its lane {lane_root}: " + "; ".join(outside))


# What a lane needs its provider to do, and what a provider that does not do it
# costs. Both are guarantees the lane code arranges but cannot itself enforce:
# it builds the hooks and the environment overlay, and the provider decides
# whether either one reaches the session.
_LANE_PROVIDER_REQUIREMENTS = (
    (
        "stop_hooks",
        "run the callbacks in AgentRunSpec.hooks, which is what denies a lane "
        "an edit to the driver, harness or oracle while its session can still "
        "be saved",
    ),
    (
        "session_env",
        "apply AgentRunSpec.env to the session it spawns, which is what gives "
        "each lane its own AITER build cache instead of one they share, where "
        "a lane can measure a module a sibling compiled",
    ),
)


def _require_lane_provider_capabilities(provider: str, lanes: int) -> None:
    """Refuse concurrent lanes on a provider that cannot keep a lane's promises.

    Raising is the point. A safety property that holds on one backend and not
    another is not a property, and running fewer lanes than were asked for would
    answer the operator's request with a different one, so they are told which
    provider is missing what and choose for themselves.

    One lane is never refused: it is the whole campaign, with nothing to be
    isolated from and no sibling to be confused with, and it is what a refusal
    offers as the way forward.
    """
    if lanes < 2:
        return
    from kernelforge.agent_backends.registry import get_agent_provider

    capabilities = get_agent_provider(provider).capabilities
    missing = [
        f"{name} (it must {detail})"
        for name, detail in _LANE_PROVIDER_REQUIREMENTS
        if not getattr(capabilities, name, False)
    ]
    if not missing:
        return
    raise click.ClickException(
        f"--lanes {lanes} needs guarantees that agent provider {provider!r} "
        "does not declare: " + "; ".join(missing) + ". Re-run with --lanes 1, "
        "or select a provider that declares them."
    )


def _make_lane_agent_factory(
    *,
    make_agent,
    config: Config,
    workspace_dir: str,
    driver: str,
    source_files: Iterable[str],
    session_kwargs: dict,
):
    """Build the factory that binds one Implementer session to one lane.

    ``session_kwargs`` are the inputs a lane shares with the canonical session
    (prompt context, budgets, backend). Everything that names a path is rebound
    onto the lane's own copy of the workspace, because a lane is handed those
    paths in order to edit them.

    The lane's serialized driver is the second factory argument. It is handed to
    the session as the command to run the driver through, because the device
    lock lives in that wrapper and a session that runs the driver beside it
    takes no lock at all.

    A lane runs the in-session gate for its protection hooks alone. They deny an
    edit or a shell write to the measurement surface while the session is still
    running, which is the last point at which the rest of that session can be
    saved: a lane diff that touches the driver, harness or oracle is refused at
    the boundary, and the implementation work in the same diff is refused with
    it. The gate's Stop hook is left out because it benchmarks, and lanes run
    concurrently while the device times one thing at a time. Each lane's
    candidate is measured once by the loop instead, under the ordinary KEEP
    protocol.
    """
    from kernelforge.agent_backends.base import session_environment
    from kernelforge.loop.aiter_cache import child_cache_environment

    def factory(lane_dir: str, serialized_driver: str | None):
        # Each lane gets its own Config: a provider that requires the workspace
        # as its cwd would otherwise start every lane's session in the shared
        # canonical workspace.
        lane_root = Path(lane_dir).resolve()
        lane_config = replace(config, workspace=str(lane_root))
        lane_agent = make_agent(
            config=lane_config,
            insession_gate=True,
            insession_gate_stop_check=False,
            driver_script=_lane_workspace_path(
                driver,
                lane_dir=lane_dir,
                workspace_dir=workspace_dir,
                label="driver",
            ),
            # The lock-taking wrapper the round installed for this lane. The
            # protected file stays the driver above; this only changes what the
            # session is told to execute, which is the only thing that makes the
            # lock more than advisory.
            interposed_driver_path=serialized_driver,
            source_files=[
                _lane_workspace_path(
                    path,
                    lane_dir=lane_dir,
                    workspace_dir=workspace_dir,
                    label="source file",
                )
                for path in source_files
            ],
            profiling_enabled=False,
            **session_kwargs,
        )
        # Each lane compiles a different edit of the same kernel. aiter imports a
        # JIT module by name and never checks the .so against the source it was
        # built from, so lanes sharing one build cache load each other's binaries
        # and each one measures a kernel it did not write. The cache cannot be
        # selected by writing os.environ -- every lane is in this process, so the
        # last write would be every lane's -- so it is handed to the lane's own
        # provider subprocess instead. Raises rather than returning a lane that
        # would compile into the shared cache.
        lane_env = child_cache_environment(lane_root.with_name(lane_root.name + _LANE_AITER_CACHE_SUFFIX))

        async def session(kernel_path: str, plan: str) -> str:
            _assert_lane_session_cwd(
                kernel_path=kernel_path,
                workspace=lane_config.workspace,
                lane_dir=lane_dir,
            )
            with session_environment(lane_env):
                return await lane_agent(kernel_path, plan)

        return session

    return factory


# ─── Autonomous Forge loop command ───


def _resolve_nomination(*, auto, nomination_input, kernel, resume):
    """Run one nomination pass, or return None when --auto was not requested.

    Args:
        auto: Whether self-nomination was requested.
        nomination_input: Path to the nomination request JSON.
        kernel: The --kernel value, which --auto is meant to supply instead.
        resume: Whether this is a resumed campaign.

    Returns:
        The resolution, or ``None`` when running in the named-kernel mode.

    Raises:
        click.ClickException: On a conflicting or incomplete invocation, or when
            the nomination picked more targets than this build can execute.
    """
    if not auto:
        if nomination_input:
            raise click.ClickException("--nomination-input requires --auto")
        return None
    if resume:
        raise click.ClickException("--auto cannot be combined with --resume; a resumed campaign has its target")
    if not nomination_input:
        raise click.ClickException("--auto requires --nomination-input")
    if kernel:
        raise click.ClickException("--auto derives the kernel from the nomination; do not pass --kernel")

    from kernelforge.nomination import NominationError, resolve

    try:
        resolution = resolve(nomination_input)
    except NominationError as error:
        raise click.ClickException(str(error)) from error
    if len(resolution.targets) > 1:
        # Running several targets in one call needs a per-target base commit and
        # per-target scratch paths, neither of which exists yet. Refuse loudly
        # rather than silently optimizing only the first pick.
        raise click.ClickException(
            f"nomination picked {len(resolution.targets)} targets but multi-target execution "
            "is not implemented; lower max_kernels to 1"
        )
    return resolution


def _nominated_patches(resolution, *, campaign_root, best_commit, micro_speedup):
    """Build the ``patches`` array for a nominated run that reached a best.

    Args:
        resolution: The nomination that chose this run's target.
        campaign_root: Campaign directory holding the published best bundle.
        best_commit: Commit the best result was published from.
        micro_speedup: Forge's own mean-case speedup, a queue tiebreaker only.

    Returns:
        One entry when a published patch exists, otherwise an empty list.
    """
    from kernelforge.nomination import patch_entry

    if not best_commit or not resolution.targets:
        return []
    root = Path(campaign_root)
    try:
        manifest = json.loads((root / "best" / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    # The manifest records the patch relative to the campaign root, and the
    # published directory is versioned, so the path cannot be assumed.
    relative = str(manifest.get("patch_path") or "").strip() if isinstance(manifest, dict) else ""
    if not relative:
        return []
    patch_path = root / relative
    if not patch_path.is_file():
        return []
    return [
        patch_entry(
            resolution.targets[0],
            patch_path=str(patch_path),
            base_commit=best_commit,
            micro_speedup=float(micro_speedup or 0.0),
        )
    ]


def _emit_empty_nomination(resolution) -> None:
    """Report a nomination that selected nothing, as a clean success."""
    payload = json.dumps(
        {
            "patches": [],
            "nomination": resolution.summary.to_dict(),
            "improved": False,
        }
    )
    click.echo(f"__FORGE_RESULT__{payload}__FORGE_RESULT__")


@main.command("forge-loop")
@click.option("--kernel", default=None, help="Fresh campaign: kernel file to optimize")
@click.option("--driver", default=None, help="Fresh campaign: validation/bench driver")
@click.option("--workspace", "workspace_dir", required=True, help="Git workspace dir")
@click.option(
    "--auto",
    is_flag=True,
    help="Pick the kernels here instead of being handed one. Requires "
    "--nomination-input; --kernel is then derived from the nomination and "
    "must not be passed. Off by default, so a run without it is unchanged.",
)
@click.option(
    "--nomination-input",
    default="",
    help="Path to the nomination request JSON (trace, candidate list, lane "
    "budget, target ceiling). Only read under --auto.",
)
@click.option(
    "--snr-threshold",
    default=DEFAULT_SNR_THRESHOLD_DB,
    type=float,
    help="Fresh campaign: SNR pre-filter threshold in dB (stored "
    "immutably in the campaign config; ignored on --resume). A "
    "KEEP is decided by the task's own correctness_command, not "
    "by this value.",
)
@click.option(
    "--max-hours",
    default=1.0,
    type=float,
    callback=_validate_max_hours,
    help="Max runtime hours (default: 1.0, minimum 1.0). Budgets "
    ">2 hours enable Analysis profiling and, for single-lane "
    "rounds, Plan Critic review.",
)
@click.option(
    "--session-timeout-sec",
    default=None,
    type=click.IntRange(min=1),
    help="Wall-clock budget for one implementer session (seconds). Overrides "
    "the value computed from --max-hours; the claude backend cuts a "
    "session at this deadline and the session is told about it.",
)
@click.option(
    "--deadline-unix",
    default=0.0,
    type=float,
    help="Absolute UNIX deadline shared by preparation and optimization.",
)
@click.option(
    "--git-branch",
    default=None,
    help="Fresh campaign: development branch to optimize on (checked out "
    "before the immutable campaign config is snapshotted).",
)
@click.option(
    "--gpu-target",
    default=None,
    help="ROCm compilation architecture, e.g. gfx950 (also exported to env)",
)
@click.option(
    "--gpu-type",
    default=None,
    callback=_normalize_gpu_type,
    help="Hardware SKU for KB identities, e.g. mi355x",
)
@click.option(
    "--kernel-backend",
    default=None,
    help=("Fresh campaign: kernel backend override. Unsupported kernel backends fall back to flydsl."),
)
@click.option("--program-md-file", default=None, help="Fresh campaign: optional task context copied into the campaign")
@click.option(
    "--invocation-spec-file",
    default=None,
    help="Path to a Hyperloom invocation-spec JSON used by task preparation.",
)
@click.option(
    "--experiments-dir",
    default=None,
    help="Diagnostics/checkpoint root (profiles, optimization_potential, "
    "tracker checkpoint). Defaults to <workspace>/forge_experiments. "
    "Resume artifacts always live under the workspace regardless.",
)
@click.option(
    "--aiter-cache-max-gb",
    default=4.0,
    type=click.FloatRange(min=0.0),
    help=(
        "Per-attempt AITER cache soft limit in GiB (default: 4). "
        "LRU pruning targets 75% of the limit; 0 disables in-run pruning."
    ),
)
@click.option(
    "--experiment-id",
    default=None,
    help="Caller-owned experiment ID; recorded for external checkpoint recovery.",
)
@click.option(
    "--experience-id",
    default="",
    help="Unique KB run identity, independent of the checkpoint experiment ID.",
)
@click.option("--result-json", default=None, help="Write the result dict here (also printed)")
@_agent_runtime_options
@click.option("--permission-mode", default=None, help="Provider permission mode when supported")
@click.option(
    "--profile-timeout-sec",
    default=ANALYSIS_TIMEOUT_SEC,
    type=int,
    help="Ceiling (seconds) for the single complete Analysis Agent "
    "session. The Agent persists phase and case artifacts for "
    "resume when the deadline is reached. Default 7200 (2 hours).",
)
@click.option(
    "--supervisor-backend",
    default="",
    callback=_validate_optional_agent_provider,
    help="Registered provider for the AVO supervisor. Omit to follow "
    "the effectively resolved Implementer provider, so one "
    "--agent-backend value controls every local agent.",
)
@click.option(
    "--profiling/--no-profiling",
    default=True,
    help="Allow Analysis hardware profiling and Implementer "
    "self-profiling guidance for long-horizon runs (>2 hours). "
    "Shorter runs keep Analysis static-only and omit that guidance. "
    "--no-profiling disables collection for every duration.",
)
@click.option(
    "--nproc-per-node",
    default=1,
    type=click.IntRange(min=1),
    help="Ranks the driver self-launches via torchrun (collective tasks "
    "such as all-reduce). >1 profiles EVERY rank in its own "
    "rocprofv3 session, because wrapping the driver would only "
    "profile the launcher process, which runs no kernel. Default 1 "
    "(single-GPU, unchanged behavior).",
)
@click.option(
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
    "needs a provider that declares stop_hooks and session_env, "
    "and is refused on one that does not.",
)
@click.option(
    "--merge-stacking/--no-merge-stacking",
    default=True,
    help="Once consecutive iterations stop producing a new best, spend "
    "one iteration measuring two archived rejected gains applied "
    "together, chosen for winning on different cases. Costs a "
    "measurement but no Implementer session. Default on; this "
    "applies at every --lanes setting, so turn it off to compare "
    "against a run that predates it.",
)
@click.option(
    "--bench-repeat",
    default=1,
    type=click.IntRange(min=1),
    help="How many times each bench repeats its measurement in-process, "
    "reporting the per-case median. Default 1 (single shot). >1 "
    "shrinks run-to-run spread. Requires a driver that accepts "
    "--repeat; the flag is omitted entirely when this is 1.",
)
@click.option(
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
)
@click.option(
    "--prepare-task/--no-prepare-task",
    default=True,
    help="Pre-loop task preparation (default on, fresh campaigns only). "
    "Before the loop, run a deterministic preflight of the driver "
    "against the loop's stdout contract; if it fails invoke ONE agent "
    "that authors/repairs the graph-timed measurement driver (never "
    "the kernel/source), then re-check. Skipped on --resume, whose "
    "driver contract is already fixed by the campaign.",
)
@click.option(
    "--task-type",
    default="",
    help="Task type (e.g. flydsl2flydsl, repository, image_kernel). "
    "'repository'/'image_kernel' enable multi-file / whole-repo "
    "handling; anything else keeps the single-file behavior.",
)
@click.option(
    "--source-files",
    default="",
    help="Comma/newline-separated implementation entry points used for "
    "orientation, profiling, JIT hints, and KB identity. This is not "
    "an edit allowlist; --kernel remains the anchor.",
)
@click.option(
    "--target-functions",
    default="",
    help="Comma-separated target kernel/function hints. Used for PMC "
    "filtering, source mapping, and agent orientation; it does not "
    "restrict which functions may be edited.",
)
@click.option(
    "--framework",
    default="",
    help="Explicit framework identity for the experience KB slug "
    "(vllm/sglang/aiter, or 'standalone' for a framework-less "
    "file). Authoritative when given; otherwise inferred from the "
    "file that defines the target operation, falling back to "
    "'unknown' when no known owner is found.",
)
@click.option(
    "--operator-name",
    default="",
    help="Logical operator identity used by profiling and the "
    "experience page key (for example, the traced operation name).",
)
@click.option(
    "--experience-kb/--no-experience-kb",
    default=True,
    help="Read and publish forge-loop experience KB entries (default on). "
    "Internal callers with their own KB lifecycle, such as "
    "forge-rewrite-by-flydsl, disable this explicitly.",
)
@click.option(
    "--kb-warmstart/--no-kb-warmstart",
    "kb_warmstart_enabled",
    default=True,
    help="Apply the best matching KB solution before iteration 1 (default on, "
    "and only with --experience-kb). A caller that prepares the workspace "
    "itself, such as forge-fuse, turns this off to keep publishing while "
    "never replaying a stored patch over a tree it already staged.",
)
@click.option(
    "--producer",
    default="",
    help="System owning the candidate stream these records belong to (default: "
    "the forge-loop's own). A producer has its own index in the KB "
    "identity scheme, so a pipeline driving this command as a subprocess "
    "can keep its records out of the kernel campaigns' ranking.",
)
@click.option(
    "--return-after-read-kb",
    "--return-after-read-KB",
    "return_after_read_kb",
    is_flag=True,
    default=False,
    help="Return before Iteration 1 when a KB solution applies cleanly, passes "
    "current correctness, and improves current performance.",
)
@click.option(
    "--pr-kb/--no-pr-kb",
    default=None,
    help="Inject upstream pull-request references from the Primus Cortex PR "
    "Monitor as Implementer prior knowledge (default off). Falls back to "
    "PR_KB_ENABLE when unset.",
)
@click.option(
    "--specialist-probe/--no-specialist-probe",
    default=None,
    help="Let the read-only planning specialists measure one variant per probe "
    "in a scratch tree, instead of only arguing about a dispatch constant "
    "(default on). Each probe re-runs the workspace driver for one case "
    "with declared constants overridden; it queues on the same device lock "
    "the fan-out lanes take, and never touches the canonical tree. Falls "
    "back to FORGE_SPECIALIST_PROBE when unset.",
)
@click.option(
    "--specialist-probe-max",
    default=None,
    type=click.IntRange(min=1),
    help="Probes ONE analysis round may make in total, shared by every "
    "specialist it dispatches (default 6). Every call counts, including "
    "one that is refused. Falls back to FORGE_SPECIALIST_PROBE_MAX when "
    "unset.",
)
@click.option(
    "--specialist-probe-budget-sec",
    default=None,
    type=click.FloatRange(min=1.0),
    help="Seconds on the device ONE analysis round may spend probing, shared by "
    "every specialist it dispatches (default 600). Cut down further at "
    "call time so no probe can leave a specialist without the time to "
    "write its analysis. Falls back to FORGE_SPECIALIST_PROBE_BUDGET_SEC "
    "when unset.",
)
@click.option(
    "--specialist-probe-scratch-root",
    default=None,
    help="Where the round scratch trees are created. Must be absolute and lie "
    "outside the workspace. Default: <experiments-dir>/specialist_probe, "
    "or a sibling of the workspace when that would land inside it. Falls "
    "back to FORGE_SPECIALIST_PROBE_SCRATCH_ROOT when unset.",
)
@click.option("--resume", is_flag=True, help="Resume the campaign stored in the exact workspace")
def forge_loop(
    kernel,
    driver,
    workspace_dir,
    auto,
    nomination_input,
    snr_threshold,
    max_hours,
    session_timeout_sec,
    deadline_unix,
    git_branch,
    gpu_target,
    gpu_type,
    kernel_backend,
    program_md_file,
    invocation_spec_file,
    experiments_dir,
    aiter_cache_max_gb,
    experiment_id,
    experience_id,
    result_json,
    model,
    agent_backend,
    agent_cli,
    agent_timeout_sec,
    agent_reasoning_effort,
    agent_sandbox_mode,
    agent_fallback_provider,
    agent_precheck,
    agent_options_json,
    permission_mode,
    supervisor_backend,
    profile_timeout_sec,
    profiling,
    prepare_task,
    task_type,
    source_files,
    target_functions,
    framework,
    operator_name,
    experience_kb,
    kb_warmstart_enabled,
    producer,
    return_after_read_kb,
    pr_kb,
    resume,
    nproc_per_node,
    bench_repeat,
    lanes,
    merge_stacking,
    specialist_probe,
    specialist_probe_max,
    specialist_probe_budget_sec,
    specialist_probe_scratch_root,
    commit_new_paths,
):
    """Run ONE Forge IterationLoop as a standalone subprocess (CLI-ized kernel backend).

    This is the subprocess entry the Hyperloom forge backend shells out to, so
    the LLM-driven loop runs in an isolated, hard-killable process (like GEAK)
    instead of in-process. Hyperloom owns worktree/in-place prep + export +
    restore; this command owns only baseline -> agent -> validate -> bench -> keep.
    Emits a JSON result dict (baseline_ms / best_ms / mean_case_speedup /
    improved / experiment_id / iteration_count) to stdout, sentinel-wrapped for
    mixed-output parsing.

    The campaign is resumable: its immutable inputs are snapshotted into
    <workspace>/forge_experiments/campaign_config.json and each session's control
    state into run_state.json, so --resume continues an interrupted campaign.

    Under --auto the kernel is nominated here rather than named by Hyperloom, and
    the result carries a ``patches`` array plus nomination counts.
    """
    nomination = _resolve_nomination(
        auto=auto,
        nomination_input=nomination_input,
        kernel=kernel,
        resume=resume,
    )
    if nomination is not None:
        if not nomination.targets:
            # No eligible candidate is an answer, not a failure: Hyperloom's
            # phase latch needs a clean exit to stop waiting on this lane.
            _emit_empty_nomination(nomination)
            return
        kernel = nomination.targets[0].source_file
    long_horizon = _is_long_horizon(max_hours)
    critic_enabled = bool(long_horizon)
    try:
        knowledge_config = KnowledgeConfig.from_env()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    import dataclasses as _dataclasses
    import hashlib as _hashlib

    if return_after_read_kb and not experience_kb:
        raise click.UsageError("--return-after-read-kb cannot be used with --no-experience-kb")
    if return_after_read_kb and not kb_warmstart_enabled:
        raise click.UsageError("--return-after-read-kb cannot be used with --no-kb-warmstart")
    if producer:
        from kernelforge.knowledge.kernel_identity import KERNEL_RECIPE_PRODUCERS

        if producer not in KERNEL_RECIPE_PRODUCERS:
            raise click.UsageError(f"--producer must be one of: {', '.join(sorted(KERNEL_RECIPE_PRODUCERS))}")

    from kernelforge.knowledge.experience_integration import git_head
    from kernelforge.loop.campaign_config import (
        CampaignConfigStore,
        derive_campaign_implementation_contract,
    )
    from kernelforge.loop.run_state import WorkspaceLock, WorkspaceLockError

    # Absolute deadline shared by task preparation and optimization. Derived from
    # --max-hours when the caller does not pass one, so the same time-budget
    # bookkeeping applies to standalone runs. The loop itself is also time-driven
    # (max_time_hours) and finalizes gracefully within budget, so this is a shared
    # clock for the pre-loop phases rather than a hard cancellation of the loop.
    if deadline_unix <= 0:
        deadline_unix = time.time() + max_hours * 3600.0
    finalize_reserve_sec = max(
        30.0,
        float(os.environ.get("FORGE_FINALIZE_RESERVE_SEC", "120") or 120),
    )

    def _remaining(*, reserve: float = 0.0) -> float:
        return max(0.0, deadline_unix - time.time() - reserve)

    def _require_time(phase: str, minimum: float = 1.0) -> None:
        if _remaining(reserve=finalize_reserve_sec) < minimum:
            raise click.ClickException(f"absolute Forge deadline exhausted before {phase}")

    if profile_timeout_sec <= 0:
        raise click.ClickException("--profile-timeout-sec must be positive")

    workspace = Path(workspace_dir).resolve()
    workspace_lock = WorkspaceLock(workspace / "forge_experiments" / "workspace.lock")
    try:
        workspace_lock.acquire()
    except WorkspaceLockError as error:
        raise click.ClickException(str(error)) from error
    click.get_current_context().call_on_close(workspace_lock.release)

    from kernelforge.loop.campaign_setup import resolve_campaign

    campaign_store = CampaignConfigStore(str(workspace))
    campaign_root = campaign_store.root
    state_path = campaign_root / "run_state.json"
    has_run_artifacts = (
        state_path.exists()
        or (campaign_root / "events.jsonl").exists()
        or any((campaign_root / "candidates").glob("iter_*"))
    )
    if resume and not state_path.is_file():
        raise click.ClickException("--resume requires <workspace>/forge_experiments/run_state.json")
    if resume and not campaign_store.exists():
        raise click.ClickException("--resume requires <workspace>/forge_experiments/campaign_config.json")
    if not resume and has_run_artifacts:
        raise click.ClickException("workspace already contains a Forge campaign; pass --resume to continue it")

    try:
        resolution = resolve_campaign(
            str(workspace),
            resume=resume,
            prepare_task=prepare_task,
            kernel=kernel,
            driver=driver,
            source_files=source_files or "",
            program_md_file=program_md_file,
            target_functions=target_functions or "",
            operator_name=operator_name or "",
            producer=producer or "",
            kernel_backend=kernel_backend or "",
            git_branch=git_branch or "",
            gpu_target=gpu_target or "",
            gpu_type=gpu_type,
            task_type=task_type or "",
            framework=framework or "",
            snr_threshold=snr_threshold,
            nproc_per_node=nproc_per_node,
            bench_repeat=bench_repeat,
            commit_new_paths=list(commit_new_paths),
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    campaign = resolution.campaign
    program_text = resolution.program_text
    campaign_save_deferred = resolution.save_deferred

    # Resume tooling may override the complete Analysis workflow deadline via
    # FORGE_PROFILE_TIMEOUT_SEC; --profile-timeout-sec is the default
    # when the env is unset. Validated AFTER the campaign config is persisted so
    # a config-only run still leaves a resumable pending config on a malformed
    # value (the caller can fix the env and retry without re-supplying inputs).
    _env_timeout = os.environ.get("FORGE_PROFILE_TIMEOUT_SEC")
    if _env_timeout is not None and _env_timeout.strip() != "":
        try:
            profile_timeout_sec = int(_env_timeout)
        except ValueError as error:
            raise click.ClickException("FORGE_PROFILE_TIMEOUT_SEC must be an integer") from error
        if profile_timeout_sec <= 0:
            raise click.ClickException("FORGE_PROFILE_TIMEOUT_SEC must be positive")

    kernel = str((workspace / campaign.kernel_path).resolve())
    driver = str((workspace / campaign.driver_path).resolve())
    source_files_list = [str((workspace / path).resolve()) for path in campaign.source_files]
    target_functions_list = list(campaign.target_functions)
    snr_threshold = campaign.snr_threshold
    gpu_target = campaign.gpu_target
    gpu_type = campaign.gpu_type
    kernel_backend = campaign.kernel_backend
    task_type = campaign.task_type
    git_branch = campaign.git_branch
    framework = campaign.framework
    operator_name = campaign.operator_name
    producer = campaign.producer
    # Measurement semantics come from the campaign, never from this invocation's
    # defaults. A resumed TP4 run that fell back to nproc=1 / single-shot would
    # compare its candidates against an incumbent measured
    # under different rules, and would profile the launcher instead of the ranks.
    nproc_per_node = campaign.nproc_per_node
    bench_repeat = campaign.bench_repeat
    # From the campaign for the same reason: a resumed session that fell back
    # to an empty allowlist could neither ship nor remove the new file an
    # earlier session was configured to.
    commit_new_paths = list(campaign.commit_new_paths)
    profiling_enabled = bool(profiling and long_horizon)

    overrides = {"gpu_target": gpu_target}
    overrides["gpu_type"] = gpu_type
    overrides["producer"] = producer
    # Only what was actually asked for: an override present with a None value
    # still wins over ``Config.from_env``'s environment lookup, which is what
    # made the FORGE_SPECIALIST_PROBE* variables dead on this path.
    for _name, _value in (
        ("specialist_probe", specialist_probe),
        ("specialist_probe_max", specialist_probe_max),
        ("specialist_probe_budget_sec", specialist_probe_budget_sec),
        ("specialist_probe_scratch_root", specialist_probe_scratch_root),
    ):
        if _value is not None:
            overrides[_name] = _value
    overrides.update(
        _agent_runtime_overrides(
            model=model,
            agent_backend=agent_backend,
            agent_cli=agent_cli,
            agent_timeout_sec=agent_timeout_sec,
            agent_reasoning_effort=agent_reasoning_effort,
            agent_sandbox_mode=agent_sandbox_mode,
            agent_fallback_provider=agent_fallback_provider,
            agent_precheck=agent_precheck,
            agent_options_json=agent_options_json,
        )
    )
    if gpu_target:
        os.environ["GPU_TARGET"] = gpu_target
    config = Config.from_env(
        workspace=str(workspace),
        knowledge_config=knowledge_config,
        **overrides,
    )
    # Two roots by design: resume artifacts (run_state/candidates/best) always
    # live under <workspace>/forge_experiments (campaign_root); diagnostics and
    # the external-recovery checkpoint go to --experiments-dir when the caller
    # supplies a distinct one (e.g. Hyperloom's output dir), else campaign_root.
    config.experiments_dir = Path(experiments_dir).resolve() if experiments_dir else campaign_root
    config.experiments_dir.mkdir(parents=True, exist_ok=True)

    # AITER cache isolation: give this attempt its own runtime-build cache so
    # parallel forge processes never share/evict each other's kernels.
    from kernelforge.loop.aiter_cache import (
        activate_aiter_cache_for_sources,
        configure_aiter_cache_isolation,
        seed_prebuilt_modules,
    )

    aiter_cache = configure_aiter_cache_isolation(
        config.experiments_dir,
        max_cache_bytes=int(aiter_cache_max_gb * 1024**3),
    )
    print(f"  [aiter-cache] isolated runtime builds under {aiter_cache.cache_root}")
    baseline_cache = activate_aiter_cache_for_sources(source_files_list)

    # Seed the pristine BASELINE shard with the package's prebuilt .so so the
    # task-preparation preflight imports warm modules instead of cold-compiling
    # the CK instance-factory TU (>26 min on gfx950, which blows the preflight
    # timeout and leaves the driver stuck as a placeholder). Safe here only
    # because this shard holds pristine source; once the loop edits a source it
    # re-activates a fresh content-keyed shard that is never seeded and compiles
    # normally, so an edit is never measured against a stale prebuilt module.
    if baseline_cache is not None:
        seed_stats = seed_prebuilt_modules(baseline_cache.aiter_jit_dir)
        print(
            f"  [aiter-cache] seeded {seed_stats['seeded']} prebuilt module(s)"
            f" (skipped {seed_stats['skipped']}, errors {seed_stats['errors']})"
            f" from {seed_stats['src'] or 'n/a'}"
        )

    from kernelforge.loop.runner import IterationLoop
    from kernelforge.loop.runner import IterationConfig
    from kernelforge.loop.scoring import (
        aggregate_regression_detail,
        warm_start_improvement_flags,
    )
    from kernelforge.tracker import ExperimentTracker, UsageAccumulator
    from kernelforge.orchestrator.agent import make_agent_fn

    iter_config = IterationConfig(
        kernel_file=kernel,
        driver_script=driver,
        canonical_driver_sha256=campaign.driver_sha256,
        campaign_base_commit=campaign.base_commit,
        snr_threshold=snr_threshold,
        max_time_hours=max(0.05, max_hours),
        deadline_unix=deadline_unix - finalize_reserve_sec,
        git_branch=git_branch,
        workspace_dir=workspace_dir,
        experiment_id=experiment_id or "",
        backend=(kernel_backend or "").split("-", 1)[0],
        kernel_backend=kernel_backend or "",
        task_type=task_type,
        source_files=source_files_list,
        target_functions=target_functions_list,
        # Operator E2E time share (percent) for the baseline potential estimator.
        operator_name=operator_name,
        implementation_signature=campaign.implementation_signature,
        implementation_identity=dict(campaign.implementation_identity),
        # Rank count for collective tasks; selects the per-rank profiling backend.
        nproc_per_node=nproc_per_node,
        # Measurement fidelity: in-process repeats within each independent bench.
        bench_repeat=bench_repeat,
        lanes=lanes,
        merge_stacking=merge_stacking,
        # New files a KEEP may carry; a REVERT removes exactly the same set.
        commit_new_paths=commit_new_paths,
    )
    tracker = ExperimentTracker(config.experiments_dir)
    usage = UsageAccumulator()

    # The caller-owned experiment ID is an EXTERNAL recovery channel, deliberately
    # independent of the internal per-segment experiment identity (each resume
    # segment gets a fresh ID so the campaign parent/child chain stays intact).
    # An external caller that hard-kills this process on its own wall clock reads
    # <experiments-dir>/<experiment-id>.json to salvage the last validated best,
    # so the record must exist before the first KEEP -- set_checkpoint raises
    # FileNotFoundError on an unknown ID.
    caller_experiment_id = (experiment_id or "").strip()
    if caller_experiment_id:
        try:
            tracker.get(caller_experiment_id)
        except FileNotFoundError:
            tracker.create(
                task_id=Path(kernel).stem,
                description=f"External recovery channel for {caller_experiment_id}",
                experiment_id=caller_experiment_id,
            )

    if campaign_save_deferred:
        # The immutable config (and its program.md) is not on disk yet; use the
        # in-memory program text captured from --program-md-file (matching
        # read_program_md's "" for a campaign without program context).
        program_md = program_text or ""
    else:
        try:
            program_md = campaign_store.read_program_md(campaign)
        except (OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error
    if resume and experience_kb:
        reference_pointer = kb_reference_program_md(workspace_dir)
        if reference_pointer:
            program_md = program_md + "\n\n" + reference_pointer

    # Pre-loop task preparation (fresh campaigns only): ensure the driver conforms
    # to the loop's stdout contract BEFORE base_sha is captured, so any scaffolding
    # the prep step commits becomes part of pristine (never the solution diff).
    # Skipped on --resume, whose pristine base and driver contract are already
    # fixed by the immutable campaign; re-preparing would corrupt the resumed base.
    if prepare_task and not resume:
        from kernelforge.loop.task_preparer import (
            declared_case_ids,
            preflight_task,
            prepare_task_sync,
        )

        # Preparation runs the driver, so it needs the rank count before the
        # loop that normally exports it exists. Without this a TP4 campaign on
        # an 8-GPU node is prepared and probed at 8 ranks and only then switched
        # to 4, so the contract is verified against a configuration the campaign
        # never measures.
        if nproc_per_node > 1:
            os.environ["FORGE_NPROC_PER_NODE"] = str(nproc_per_node)
        else:
            os.environ.pop("FORGE_NPROC_PER_NODE", None)

        _require_time("task preparation", 10.0)
        # The spec's declared suite gates this too: a driver that times a subset
        # of the task's cases is not "already conforming", it just measures less
        # than the task asks for, and accepting it here skips the only step that
        # would have repaired it. Derived once, here, and handed to preparation as
        # well: derived twice, the two copies agreed only while preparation's own
        # materialization of the spec kept succeeding.
        try:
            expected_case_ids = declared_case_ids(invocation_spec_file)
        except ValueError as exc:
            # Continuing would switch the driver's case check off and spend the
            # whole run measuring a suite the operator never got to state.
            raise click.ClickException(str(exc)) from exc
        pf = preflight_task(
            driver=driver,
            snr_threshold=snr_threshold,
            require_graph=True,
            require_profile=True,
            deadline_unix=deadline_unix - finalize_reserve_sec,
            expected_case_ids=expected_case_ids,
        )
        if pf.ok:
            print("  [prepare] task already conforms to the driver contract; skipping")
            _persist_declared_spec(invocation_spec_file or "", driver)
        else:
            print(f"  [prepare] task does not conform ({pf.summary()}); invoking prep agent...")
            # The budget that actually applies is min(wall, what the per-kernel
            # deadline leaves), and it decides how many attempts ever start.
            # Without it in the log, diagnosing "FAILED after 2 attempt(s)"
            # meant reverse-engineering the wall from audit timestamps.
            from kernelforge.loop import task_preparer as _tp

            _prep_wall = min(
                float(_tp.PREPARE_MAX_WALL_SEC),
                max(0.0, deadline_unix - finalize_reserve_sec - time.time()),
            )
            print(
                f"  [prepare] budget: wall={_prep_wall:.0f}s "
                f"attempt_cap={_tp.PER_ATTEMPT_CAP_SEC}s "
                f"max_attempts={_tp.PREPARE_MAX_ATTEMPTS}"
            )
            prep = prepare_task_sync(
                config=config,
                workspace_dir=workspace_dir,
                kernel=kernel,
                driver=driver,
                program_md=program_md,
                target_functions=target_functions_list,
                source_files=source_files_list,
                kernel_backend=kernel_backend,
                snr_threshold=snr_threshold,
                preflight=pf,
                invocation_spec_file=invocation_spec_file or "",
                expected_case_ids=expected_case_ids,
                # Let the default PREPARE_MAX_WALL_SEC (3000s, sized for a cold-JIT
                # preflight) apply; prepare_task clamps it to the per-kernel
                # deadline_unix below. A local min(1200, ...) here silently defeated
                # that raised budget, timing out slow cold preflights.
                deadline_unix=deadline_unix - finalize_reserve_sec,
                # A collective task needs a driver that launches its own ranks;
                # the preparer cannot infer that from the kernel source.
                nproc_per_node=nproc_per_node,
                read_only_files=[path for path in (program_md_file, invocation_spec_file) if path],
                usage=usage,
            )
            if prep.ok:
                # Cap the file list: an external driver bundle can legitimately
                # publish dozens of files, and one run scrolled ~700 cache paths
                # through the operator's log for a 3-file change.
                _shown = prep.wrote_files[:12]
                _extra = len(prep.wrote_files) - len(_shown)
                print(
                    f"  [prepare] prepared measurement driver in {prep.attempts} attempt(s); "
                    f"wrote {', '.join(_shown) or '(none)'}" + (f" (+{_extra} more)" if _extra > 0 else "")
                )
                # Telemetry only: never let a missing field break a good prep.
                _pf_sec = getattr(prep.final_preflight, "duration_sec", 0.0) or 0.0
                if _pf_sec:
                    _stages = ", ".join(
                        f"{name}={detail['seconds']:.0f}s"
                        for name, detail in (getattr(prep.final_preflight, "details", {}) or {}).items()
                        if isinstance(detail, dict) and "seconds" in detail
                    )
                    print(f"  [prepare] preflight: {_pf_sec:.0f}s" + (f" ({_stages})" if _stages else ""))
                if prep.audit_dir:
                    print(f"  [prepare] audit: {prep.audit_dir}")
            else:
                # prep.message carries the real failure reason (e.g. "driver
                # conformed but commit didn't land"); the preflight summary can
                # read "ok" even when prep failed downstream, so lead with
                # prep.message and append the preflight summary as extra context.
                detail = prep.message or (prep.final_preflight.summary() if prep.final_preflight else "")
                if prep.message and prep.final_preflight:
                    detail = f"{prep.message} | preflight: {prep.final_preflight.summary()}"
                disposition = "rolled back" if prep.rolled_back else "workspace preserved"
                print(f"  [prepare] FAILED after {prep.attempts} attempt(s); {disposition}. {detail}")
                err_result = {
                    "error": "task_preparation_failed",
                    "detail": detail,
                    "attempts": prep.attempts,
                    "rolled_back": prep.rolled_back,
                    "experiment_id": None,
                }
                if prep.audit_dir:
                    err_result["task_preparation_audit_dir"] = prep.audit_dir
                err_payload = json.dumps(err_result)
                if result_json:
                    atomic_write_json(result_json, err_result)
                click.echo(f"__FORGE_RESULT__{err_payload}__FORGE_RESULT__")
                sys.exit(2)

    # Fix the fresh campaign's canonical driver digest and pristine base_commit
    # from the POST-preparation state and persist the immutable config now. Task
    # preparation above may have repaired the driver (new digest) and committed
    # its scaffolding (new HEAD); anchoring the digest and base here keeps the
    # campaign consistent with the driver the loop validates and the pristine base
    # its solution diff is measured against. When prep made no changes, the digest
    # and base are simply re-confirmed.
    if campaign_save_deferred:
        prepared_driver_sha256 = _hashlib.sha256(Path(driver).read_bytes()).hexdigest()
        prepared_base_commit = git_head(str(workspace)) or campaign.base_commit
        prepared_signature, prepared_identity = derive_campaign_implementation_contract(
            workspace_dir=str(workspace),
            kernel_path=campaign.kernel_path,
            source_files=campaign.source_files,
            framework=campaign.framework,
            base_commit=prepared_base_commit,
        )
        campaign = _dataclasses.replace(
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
            campaign_store.save(campaign, program_md=program_text)
        except (OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error

    # Construct the loop only after task preparation has resolved the profiling
    # contract; IterationLoop snapshots that readiness in its runtime state.
    loop_runner = IterationLoop(iter_config, tracker, config, resume=resume)

    if resume:
        try:
            loop_runner.validate_resume_preflight()
        except (OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error

    # Pristine HEAD — anchors the cumulative solution diff written back at the end.
    # Persisted from the fresh campaign so resume publications remain cumulative.
    base_sha = campaign.base_commit

    # KB warm-start: apply the best prior solution as the starting point and inject
    # its experience into the prompt, so the agent continues from the best-known
    # state instead of from scratch. Fresh campaigns only; fully best-effort —
    # cold-starts if gbrain is unconfigured/unreachable or anything errors.
    kb_pristine_baseline_ms = None
    kb_reused_speedup = None
    warm = {
        "candidate": False,
        "read_reason": "resume" if resume else "deadline",
        "read_error": "",
    }
    warm_start_result = None
    if experience_kb and kb_warmstart_enabled and not resume and _remaining(reserve=finalize_reserve_sec) >= 600:
        try:
            warm = kb_warmstart(
                config=config,
                kernel=kernel,
                driver=driver,
                workspace_dir=workspace_dir,
                kernel_backend=kernel_backend,
                target_functions=target_functions_list,
                framework=framework,
                snr_threshold=snr_threshold,
                source_files=source_files_list,
                operator_name=operator_name,
                bench_repeat=bench_repeat,
                canonical_timeout_cap_sec=(iter_config.validate_stage_timeout_sec),
            )
        except WarmStartRollbackError as error:
            failure = click.ClickException(f"warm-start rollback failed; workspace may be inconsistent: {error}")
            failure.exit_code = 2
            raise failure from error
        warm.setdefault(
            "read_reason",
            "hit" if warm.get("candidate") else "solution_pages_missing",
        )
        warm.setdefault("read_error", "")
    elif not resume and not experience_kb:
        warm["read_reason"] = "disabled"
        print("  [kb] experience KB disabled by caller")
    elif not resume:
        print("  [kb] warm-start skipped: absolute deadline reserve")
    if warm.get("candidate"):
        kb_pristine_baseline_ms = warm.get("pristine_ms")
        if warm.get("applied"):
            # The floor this campaign starts from. Ending here means the run
            # reproduced a recorded solution rather than finding one.
            kb_reused_speedup = warm.get("mean_case_speedup")
            iter_config.publication_baseline_wall_ms = warm.get("pristine_ms")
            try:
                published = publish_warm_start_recovery(
                    workspace_dir=workspace_dir,
                    base_commit=base_sha,
                    warm=warm,
                    caller_experiment_id=caller_experiment_id,
                    experience_id=experience_id,
                    tracker=tracker,
                    result_json=result_json,
                )
                if published:
                    warm_start_result = dict(published)
                    iter_config.warm_start_publication = dict(published)
                if published and published.get("persistence_degraded"):
                    print(
                        "  [warm-start] recovery publication degraded: "
                        + "; ".join(published.get("persistence_errors") or []),
                        flush=True,
                    )
                print(
                    "  [warm-start] published recoverable best before iteration 1",
                    flush=True,
                )
            except Exception as error:
                try:
                    rollback_unpublished_warm_start(
                        workspace_dir,
                        base_commit=base_sha,
                        result_json=result_json,
                    )
                except Exception as rollback_error:
                    raise click.ClickException(
                        "failed to publish validated warm-start and could not "
                        f"restore pristine workspace: {rollback_error}"
                    ) from rollback_error
                warm["applied"] = False
                warm["keep_baseline_ms"] = warm.get("pristine_ms")
                mark_kb_reference_rejected(
                    workspace_dir,
                    int(warm.get("applied_rank") or 0),
                    "publication_failed",
                )
                warm["program_md_addition"] = kb_reference_program_md(
                    workspace_dir,
                    detect_applied=False,
                ) or warm.get(
                    "reference_program_md_addition",
                    "## Warm-start reference only\n"
                    "The prior patch could not be published durably and was "
                    "removed before optimization.",
                )
                iter_config.publication_baseline_wall_ms = None
                print(
                    "  [warm-start] recovery publication failed; restored "
                    f"pristine source and continuing reference-only ({error})",
                    flush=True,
                )
        if warm.get("program_md_addition"):
            program_md = program_md + "\n\n" + warm["program_md_addition"]
        # Keep the pristine raw baseline immutable. Warm-start performance is a
        # separate mean case speedup anchor, never an overloaded wall-time scalar.
        if warm.get("pristine_ms"):
            iter_config.baseline_wall_ms = warm["pristine_ms"]
        if warm.get("baseline_case_times"):
            iter_config.baseline_case_times = dict(warm["baseline_case_times"])
        iter_config.preloop_baseline_unscored_cases = list(warm.get("baseline_unscored_cases") or [])
        if warm.get("applied"):
            iter_config.pristine_baseline_wall_ms = warm.get("pristine_ms")
            iter_config.warm_start_wall_ms = warm.get("keep_baseline_ms")
            iter_config.warm_start_mean_case_speedup = warm.get("mean_case_speedup")
            iter_config.warm_start_bench = {
                "case_times": dict(warm.get("case_times") or {}),
                "unscored_cases": list(warm.get("unscored_cases") or []),
            }
            iter_config.warm_start_commit = warm.get("applied_commit", "")
            iter_config.warm_start_solution_slug = warm.get("solution_slug", "")

    if return_after_read_kb and warm.get("applied"):
        if not isinstance(warm_start_result, dict):
            raise click.ClickException("validated KB warm-start has no recoverable result")
        pristine_ms = float(warm["pristine_ms"])
        best_ms = float(warm["keep_baseline_ms"])
        mean_case_speedup = float(warm["mean_case_speedup"])
        best_commit = str(warm.get("applied_commit") or "")
        publication_state = _initial_remote_publication_state(warm)
        publication = _remote_publication_view(
            publication_state,
            best_commit,
        )
        kb_experience = {
            "read": kb_read_status(warm),
            "write": publication_state["last_result"],
            "publication": publication,
        }
        result = {
            **warm_start_result,
            "pristine_baseline_ms": pristine_ms,
            "search_start_ms": best_ms,
            "mean_case_speedup": mean_case_speedup,
            "search_start_mean_case_speedup": mean_case_speedup,
            # The published manifest already withholds the badge when the wall
            # times contradict the score; this result JSON used to assert the
            # improvement outright, so the same run answered differently
            # depending on which artifact a reader picked.
            **warm_start_improvement_flags(
                pristine_ms=pristine_ms,
                best_ms=best_ms,
                mean_case_speedup=mean_case_speedup,
            ),
            "incremental_improved": False,
            "improved_during_search": False,
            "total_speedup": mean_case_speedup,
            "incremental_speedup": 1.0,
            "remote_publication": publication,
            "kb_experience": kb_experience,
            "llm_usage": usage.totals(),
            "returned_after_read_kb": True,
        }
        payload = json.dumps(result)
        if result_json:
            atomic_write_json(result_json, result)
        print(
            "  [kb] validated warm-start accepted; returning before iteration 1",
            flush=True,
        )
        click.echo(f"__FORGE_RESULT__{payload}__FORGE_RESULT__")
        return

    # Source mapping, profiling, profile analysis, and potential analysis are
    # produced together by the commit-bound Analysis Agent.
    iter_config.program_md = program_md

    # Hook-capable providers gate before stop; resumable providers apply the same
    # canonical gate between turns. The outer loop remains final authority.
    gate_on = True
    # No hard edit budget: a session making steady progress must not be cut off
    # for editing a lot. The sole budget is block-based (``max_blocks``): after
    # that many BLOCKed non-converging stops the gate cleanly allows the stop, so
    # the session ends resumable and the summarizer can write a full lesson. The
    # provider turn ceiling remains only a high runaway backstop.
    max_blocks = 10
    # A turn cap never bounded time: it fired on 2.2% of sessions, so a session
    # that neither converged nor capped ran until something outside killed it.
    # The wall-clock deadline below is the real per-session budget; the turn cap
    # is now only a fixed high backstop against a pathological loop.
    config.max_turns = FORGE_IMPLEMENTER_TURN_BACKSTOP
    session_timeout_sec = _forge_session_timeout_sec(max_hours, session_timeout_sec)
    print(
        f"  Implementer session budget: {session_timeout_sec}s "
        f"(campaign budget {max_hours:g}h; turn backstop {config.max_turns})"
    )
    # PR references are independent of the experience-KB lifecycle. Keep them
    # out of program_md so commit-bound Analysis and specialist orchestration
    # remain grounded in measured evidence rather than external reference text.
    pr_task_context = ""
    pr_kb_repo = ""
    if _pr_kb_enabled(pr_kb):
        from kernelforge.knowledge.pr_query_context import (
            REASON_LOCAL_FAILURE,
            REASON_SKIPPED_DEADLINE,
        )

        if _remaining(reserve=finalize_reserve_sec) < 20.0:
            print("  [pr-kb] skipped (deadline)")
            iter_config.pr_kb_event = _pr_refs_event_fields(
                REASON_SKIPPED_DEADLINE,
                {},
            )
        else:
            pr_result = _collect_pr_references(
                workspace_dir=workspace_dir,
                kernel_backend=kernel_backend or "",
                git_remote=_git_remote_url(workspace),
                source_files=campaign.source_files,
                operator_name=operator_name or "",
                target_functions=target_functions_list,
                budget_sec=min(PR_KB_BUDGET_SEC, _remaining(reserve=finalize_reserve_sec)),
            )
            if pr_result is None:
                iter_config.pr_kb_event = _pr_refs_event_fields(
                    REASON_LOCAL_FAILURE,
                    {"degraded_reason": REASON_LOCAL_FAILURE},
                )
            else:
                pr_task_context = pr_result.prompt_context
                pr_kb_repo = pr_result.repo
                iter_config.pr_reference_context = pr_task_context
                iter_config.pr_reference_labels = tuple(
                    f"{reference.repo}#{reference.number}" for reference in pr_result.references
                )
                iter_config.pr_kb_event = _pr_refs_event_fields(pr_result.reason, pr_result.stats)
                iter_config.pr_kb_snapshot = pr_result.pending_snapshot
                if pr_result.injected:
                    print(
                        f"  [pr-kb] injected "
                        f"{pr_result.stats.get('injected_entries', 0)} "
                        f"references ({pr_result.stats.get('injected_bytes', 0)} B)"
                    )
                else:
                    print(f"  [pr-kb] no references ({pr_result.reason or 'empty'})")

    selected_runtime = config.agent_runtime()
    agent_fn = make_agent_fn(
        config=config,
        program_md=program_md,
        pre_task_context=pr_task_context,
        pr_kb_repo=pr_kb_repo,
        kernel_backend_name=kernel_backend,
        insession_gate=gate_on,
        driver_script=driver,
        snr_threshold=snr_threshold,
        max_blocks=max_blocks,
        session_timeout_sec=session_timeout_sec,
        validation_timeout_sec=iter_config.validate_stage_timeout_sec,
        bench_timeout_sec=iter_config.bench_timeout_sec,
        bench_repeat=bench_repeat,
        permission_mode=permission_mode,
        task_type=task_type,
        source_files=source_files_list,
        target_functions=target_functions_list,
        profiling_enabled=profiling_enabled,
        agent_backend=selected_runtime.provider,
        usage=usage,
    )
    effective_implementer = getattr(
        agent_fn,
        "backend_name",
        selected_runtime.provider,
    )
    effective_implementer_model = getattr(
        agent_fn,
        "backend_model",
        selected_runtime.model,
    )
    print(f"  Implementer: {effective_implementer} / {effective_implementer_model}")

    # Checked against the backend a lane actually resolves to, not the one that
    # was asked for: a lane repeats the canonical session's resolution from the
    # same runtime, so a fallback that moved the canonical session moved the
    # lanes with it. Raised here, before the campaign spends anything on a round
    # that could not have been measured honestly.
    _require_lane_provider_capabilities(effective_implementer, lanes)

    _lane_agent_factory = _make_lane_agent_factory(
        make_agent=make_agent_fn,
        config=config,
        workspace_dir=iter_config.workspace_dir,
        driver=driver,
        source_files=source_files_list,
        session_kwargs={
            "program_md": program_md,
            "pre_task_context": pr_task_context,
            "pr_kb_repo": pr_kb_repo,
            "kernel_backend_name": kernel_backend,
            "snr_threshold": snr_threshold,
            "max_blocks": max_blocks,
            "session_timeout_sec": session_timeout_sec,
            "validation_timeout_sec": iter_config.validate_stage_timeout_sec,
            "bench_timeout_sec": iter_config.bench_timeout_sec,
            "bench_repeat": bench_repeat,
            "permission_mode": permission_mode,
            "task_type": task_type,
            # Function names, not paths: nothing to rebind onto a lane.
            "target_functions": target_functions_list,
            "agent_backend": selected_runtime.provider,
            "usage": usage,
        },
    )

    from kernelforge.orchestrator.analysis import make_analysis_agent_service

    analysis_service = make_analysis_agent_service(
        config=config,
        usage=usage,
        timeout_sec=profile_timeout_sec,
        profiling_enabled=profiling_enabled,
    )
    analysis_mode = "profiled" if profiling_enabled else "static-only"
    print("  Analysis Agent: " + analysis_mode)

    from kernelforge.orchestrator.orchestration import (
        default_specialist_definitions,
        make_orchestration_service,
    )

    specialist_definitions = default_specialist_definitions()
    orchestration_service = make_orchestration_service(
        config=config,
        usage=usage,
        definitions=specialist_definitions,
        enable_plan_critic=critic_enabled,
    )
    print(f"  Orchestration: enabled with parallel specialists ({', '.join(sorted(specialist_definitions))})")
    print(
        "  Plan Critic: "
        + ("enabled (long-horizon, same backend/model)" if critic_enabled else "disabled (requires --max-hours > 2)")
    )

    # AVO supervisor (always on): reviews the trajectory on a stall and injects
    # fresh directions. It follows the effectively resolved Implementer backend so one
    # --agent-backend value controls every local agent; --supervisor-backend stays
    # available for callers that need a heterogeneous reviewer.
    from kernelforge.orchestrator.supervisor import make_supervisor_fn

    sup_backend = supervisor_backend or effective_implementer
    supervisor_fn = make_supervisor_fn(
        program_md=program_md,
        gpu_target=config.gpu_target,
        backend=sup_backend,
        usage=usage,
        config=config,
    )
    # Report the backend/model resolved by the shared backend preflight.
    eff_backend = getattr(supervisor_fn, "backend_name", sup_backend)
    eff_model = getattr(
        supervisor_fn,
        "backend_model",
        config.agent_runtime().model,
    )
    fell_back = f" (requested {sup_backend}, fell back)" if eff_backend != sup_backend else ""
    print(f"  Supervisor: {eff_backend} / {eff_model}{fell_back}")

    remote_publication = _initial_remote_publication_state(warm)

    def _build_result(kb_experience) -> dict:
        """Assemble the loop result dict from live runner state.

        Shared by the per-new-best interim snapshot (kb_experience=None) and the
        final write (full kb_experience). Reading live state means an interim call
        always reflects the latest VERIFIED best.
        """
        search_start_ms = getattr(loop_runner.ic, "warm_start_wall_ms", None) or getattr(
            loop_runner.ic, "baseline_wall_ms", None
        )
        search_start_mean_case_speedup = getattr(loop_runner.ic, "warm_start_mean_case_speedup", None) or 1.0
        pristine_ms = (
            getattr(loop_runner.ic, "pristine_baseline_wall_ms", None)
            or getattr(loop_runner.ic, "publication_baseline_wall_ms", None)
            or kb_pristine_baseline_ms
            or search_start_ms
        )
        best = getattr(loop_runner, "best_wall_ms", None)
        total_speedup = getattr(loop_runner, "best_mean_case_speedup", None)
        incremental_speedup = (
            float(total_speedup) / float(search_start_mean_case_speedup)
            if total_speedup and search_start_mean_case_speedup
            else None
        )
        exp_id = getattr(loop_runner.experiment, "experiment_id", None)
        state_best = loop_runner.run_state.best
        best_iteration = getattr(state_best, "iteration", 0)
        best_commit = getattr(state_best, "commit_hash", "")
        # A validated warm-start is published before IterationLoop creates a
        # run-state best. Preserve that stronger pristine->warm result until an
        # iteration KEEP supersedes it.
        if not best_commit:
            try:
                published = json.loads((campaign_root / "best_result.json").read_text())
            except Exception:
                published = {}
            if published.get("correctness_passed") is True and int(published.get("iteration", -1)) == 0:
                pristine_ms = published.get("pristine_baseline_ms") or published.get("baseline_wall_ms") or pristine_ms
                search_start_ms = published.get("search_start_ms") or published.get("best_wall_ms") or search_start_ms
                best = published.get("best_wall_ms")
                total_speedup = published.get("mean_case_speedup")
                search_start_mean_case_speedup = (
                    published.get("search_start_mean_case_speedup") or total_speedup or search_start_mean_case_speedup
                )
                incremental_speedup = 1.0
                best_iteration = 0
                best_commit = str(published.get("commit_hash") or "")
        # KEEP is decided on the mean of per-case speedups while these are
        # aggregate wall times, so the two can legitimately disagree by a hair.
        # A claimed improvement that is not actually faster overall is recorded
        # by name and withdrawn from `improved` instead of carrying a PASS badge.
        aggregate_regression = aggregate_regression_detail(
            baseline_ms=pristine_ms,
            best_ms=best,
            mean_case_speedup=total_speedup,
        )
        result = {
            "baseline_ms": pristine_ms,
            "pristine_baseline_ms": pristine_ms,
            "search_start_ms": search_start_ms,
            "best_ms": best,
            "mean_case_speedup": total_speedup,
            "search_start_mean_case_speedup": (search_start_mean_case_speedup),
            "aggregate_regression": aggregate_regression,
            "improved": bool(total_speedup and total_speedup > 1.0) and not aggregate_regression,
            "total_improved": bool(total_speedup and total_speedup > 1.0) and not aggregate_regression,
            "incremental_improved": bool(total_speedup and total_speedup > search_start_mean_case_speedup),
            "improved_during_search": bool(total_speedup and total_speedup > search_start_mean_case_speedup),
            # Reported so a consumer can tell a faster transfer from a cheaper
            # barrier; wall time alone cannot say which one a kept kernel bought.
            "case_bandwidth": dict(getattr(loop_runner, "last_case_bandwidth", {}) or {}),
            "total_speedup": total_speedup,
            "incremental_speedup": incremental_speedup,
            "experiment_id": exp_id,
            "campaign_id": getattr(loop_runner.run_state, "campaign_id", ""),
            "session_index": getattr(loop_runner.run_state, "session_index", 0),
            "segment_index": getattr(loop_runner.experiment, "segment_index", 0),
            "next_iteration": getattr(loop_runner.run_state, "next_iteration", 1),
            "best_iteration": best_iteration,
            "best_commit": best_commit,
            "remote_publication": _remote_publication_view(
                remote_publication,
                best_commit,
            ),
            "best_manifest": str(campaign_root / "best" / "manifest.json"),
            "optimization_report": str(campaign_root / "optimization_report.md"),
            "optimization_history": str(campaign_root / "optimization_history.md"),
            "persistence_degraded": bool(getattr(loop_runner, "persistence_degraded", False)),
            "persistence_errors": list(getattr(loop_runner, "persistence_errors", [])),
            "iteration_count": 0,
            "kb_experience": kb_experience,
            "agent_backend": effective_implementer,
            "agent_model": effective_implementer_model,
            "llm_usage": getattr(loop_runner, "llm_usage", {}) or {},
        }
        if exp_id:
            try:
                completed_experiment = tracker.get(exp_id)
                result["iteration_count"] = len(completed_experiment.iterations)
                result["checkpoint"] = completed_experiment.checkpoint
            except Exception:
                # Tracker metadata is optional on incomplete runs; final result
                # emission must remain available so callers can reject it cleanly.
                pass
        if nomination is not None:
            result["nomination"] = nomination.summary.to_dict()
            result["patches"] = _nominated_patches(
                nomination,
                campaign_root=campaign_root,
                best_commit=best_commit,
                micro_speedup=total_speedup,
            )
        return result

    def _write_result_json(result: dict) -> None:
        """Persist the result dict to --result-json (best-effort; never raises)."""
        if not result_json:
            return
        try:
            atomic_write_json(result_json, result)
        except Exception as e:  # noqa: BLE001 - a snapshot write must never break the loop
            print(f"  [forge-loop] result-json snapshot skipped ({e})", flush=True)

    def _attempt_remote_publication(
        *,
        commit: str,
        llm_summary: bool,
        incremental_summary: dict | None = None,
        snr_db_override: float | None = None,
    ) -> dict:
        """Publish the current durable best idempotently within this process."""
        if not experience_kb:
            return {"written": False, "reason": "disabled"}
        if _warm_start_publication_covers(remote_publication, commit):
            return dict(remote_publication["last_result"])
        if (
            not llm_summary
            and commit
            and remote_publication["published_commit"] == commit
            and not remote_publication["pending_commit"]
        ):
            return remote_publication["last_result"] or {
                "written": True,
                "reason": "already_published",
            }
        remote_publication["pending_commit"] = commit
        remote_publication["best_commit"] = commit
        remote_publication["last_attempted_commit"] = commit
        remote_publication["state"] = "publishing"
        remote_publication["source"] = "campaign_publication"
        try:
            status = write_experience_to_kb(
                config=config,
                loop_runner=loop_runner,
                workspace_dir=workspace_dir,
                kernel=kernel,
                kernel_backend=kernel_backend,
                gpu_target=config.gpu_target,
                base_sha=base_sha,
                pristine_baseline_ms=kb_pristine_baseline_ms,
                reused_speedup=kb_reused_speedup,
                source_files=source_files_list,
                target_functions=target_functions_list,
                framework=framework,
                experience_id=experience_id,
                operator_name=operator_name,
                llm_summary=llm_summary,
                incremental_summary=incremental_summary,
                snr_db_override=snr_db_override,
                usage=usage,
            )
        except Exception as e:  # noqa: BLE001 - checkpoint publish must never break the loop
            status = {"written": False, "reason": f"error:{e!r}"}
        _record_remote_publication_result(
            remote_publication,
            commit=commit,
            result=status,
        )
        return status

    # The runner invokes this after the KEEP commit, run state, event, and local
    # best artifact are durable, but before post-KEEP profiling.
    def _publish_remote_best(result) -> None:
        if not getattr(result, "kept", False):
            return
        commit = str(getattr(result, "commit_hash", "") or git_head(workspace_dir))
        _attempt_remote_publication(
            commit=commit,
            llm_summary=False,
            incremental_summary={
                "category": "",
                "strategy": str(getattr(loop_runner.run_state.best, "plan", "") or ""),
                "recipe": "",
                "lessons": "",
            },
            snr_db_override=getattr(result, "snr_db", None),
        )
        _write_result_json(_build_result(kb_experience=None))

    def _checkpoint_on_best_committed(result) -> None:
        """Persist the durable best (external recovery) before optional post-KEEP profiling.

        Written onto the live campaign-segment experiment record; the campaign's
        own run_state.json remains the primary resume mechanism.
        """
        if not getattr(result, "kept", False):
            return
        experiment = loop_runner.experiment
        best_commit = str(getattr(result, "commit_hash", "") or git_head(workspace_dir))
        search_start_ms = getattr(loop_runner.ic, "warm_start_wall_ms", None) or getattr(
            loop_runner.ic, "baseline_wall_ms", None
        )
        search_start_mean_case_speedup = getattr(loop_runner.ic, "warm_start_mean_case_speedup", None) or 1.0
        baseline_ms = (
            getattr(loop_runner.ic, "pristine_baseline_wall_ms", None)
            or getattr(loop_runner.ic, "publication_baseline_wall_ms", None)
            or search_start_ms
        )
        best_ms = getattr(result, "wall_ms", None)
        mean_case_speedup = getattr(result, "mean_case_speedup", None)
        aggregate_regression = aggregate_regression_detail(
            baseline_ms=baseline_ms,
            best_ms=best_ms,
            mean_case_speedup=mean_case_speedup,
        )
        checkpoint = {
            "schema_version": 1,
            "state": "best_committed",
            "decision": "KEEP",
            "experiment_id": (caller_experiment_id or (experiment.experiment_id if experiment is not None else "")),
            "base_commit": base_sha,
            "best_commit": best_commit,
            "best_iteration": int(getattr(result, "iteration", 0) or 0),
            "baseline_ms": baseline_ms,
            "pristine_baseline_ms": baseline_ms,
            "search_start_ms": search_start_ms,
            "best_ms": best_ms,
            "mean_case_speedup": mean_case_speedup,
            "search_start_mean_case_speedup": (search_start_mean_case_speedup),
            "aggregate_regression": aggregate_regression,
            "improved": bool(mean_case_speedup and mean_case_speedup > 1.0) and not aggregate_regression,
            "total_improved": bool(mean_case_speedup and mean_case_speedup > 1.0) and not aggregate_regression,
            "incremental_improved": bool(mean_case_speedup and mean_case_speedup > search_start_mean_case_speedup),
            "improved_during_search": bool(mean_case_speedup and mean_case_speedup > search_start_mean_case_speedup),
            "total_speedup": mean_case_speedup,
            "incremental_speedup": (
                float(mean_case_speedup) / float(search_start_mean_case_speedup)
                if mean_case_speedup and search_start_mean_case_speedup
                else None
            ),
            "validation_passed": bool(getattr(result, "validation_passed", False)),
            "validation_summary": str(getattr(result, "validation_summary", "") or ""),
            "snr_db": getattr(result, "snr_db", None),
        }
        if experiment is not None or caller_experiment_id:
            try:
                if experiment is not None:
                    tracker.set_checkpoint(experiment.experiment_id, checkpoint)
                # Mirror onto the caller-owned record so an external hard kill can
                # still recover this KEEP; refreshed on every resumed segment.
                if caller_experiment_id:
                    tracker.set_checkpoint(caller_experiment_id, checkpoint)
            except Exception as e:  # noqa: BLE001 - never break the loop
                print(f"  [checkpoint] skipped ({e})", flush=True)
        _write_result_json(_build_result(kb_experience=None))

    asyncio.run(
        loop_runner.run(
            agent_fn=agent_fn,
            agent_factory=_lane_agent_factory if lanes > 1 else None,
            analysis_service=analysis_service,
            orchestration_service=orchestration_service,
            supervisor_fn=supervisor_fn,
            on_best_committed=_checkpoint_on_best_committed,
            on_best_ready=_publish_remote_best,
            usage=usage,
            workspace_lock_held=True,
        )
    )

    # Final graceful write: upgrade the (possibly interim) per-run solution page
    # with the precise LLM-generated summary. Always attempted; fully best-effort
    # (never affects the run's result or exit code).
    final_best_commit = str(getattr(loop_runner.run_state.best, "commit_hash", "") or "")
    kb_write = _attempt_remote_publication(
        commit=final_best_commit,
        llm_summary=_remaining(reserve=30.0) >= 60.0,
    )
    if remote_publication["pending_commit"]:
        kb_write = _attempt_remote_publication(
            commit=final_best_commit,
            llm_summary=False,
        )
    loop_runner._checkpoint_llm_usage()
    kb_experience = {
        "read": kb_read_status(warm),
        "write": kb_write,
        "publication": _remote_publication_view(
            remote_publication,
            final_best_commit,
        ),
    }
    experiment_id = getattr(loop_runner.experiment, "experiment_id", None)
    loop_runner.llm_usage = usage.totals()
    if experiment_id:
        try:
            if loop_runner.llm_usage.get("calls"):
                tracker.set_llm_usage(experiment_id, loop_runner.llm_usage)
        except Exception as exc:
            click.echo(
                f"Warning: failed to record LLM usage for experiment {experiment_id}: {exc}",
                err=True,
            )
        try:
            tracker.set_kb_experience(experiment_id, kb_experience)
        except Exception as exc:
            click.echo(
                f"Warning: failed to record KB experience for experiment {experiment_id}: {exc}",
                err=True,
            )

    # Final result: full kb_experience. On a clean exit this overwrites any
    # per-new-best interim snapshot written during the loop above.
    result = _build_result(kb_experience=kb_experience)
    payload = json.dumps(result)
    if result_json:
        atomic_write_json(result_json, result)
    # Sentinel-wrapped so the caller can extract it from mixed loop stdout.
    click.echo(f"__FORGE_RESULT__{payload}__FORGE_RESULT__")
    _write_pr_provenance(
        workspace_dir=workspace_dir,
        surfaced=iter_config.pr_reference_labels,
        winning_iteration=int(result.get("best_iteration") or 0),
        experiment_id=str(result.get("experiment_id") or ""),
    )


def _validate_rewrite_framework(_ctx, _param, value):
    """Accept only a framework the capability handshake advertises."""
    from kernelforge.rewrite_by_flydsl.protocol import SUPPORTED_FRAMEWORKS

    cleaned = (value or "").strip().lower()
    if cleaned and cleaned not in SUPPORTED_FRAMEWORKS:
        raise click.BadParameter(f"unsupported framework {value!r}; expected one of " + ", ".join(SUPPORTED_FRAMEWORKS))
    return cleaned


def _emit_rewrite_capabilities(ctx, _param, value):
    """Answer the capability handshake before any required option is parsed."""
    if not value or ctx.resilient_parsing:
        return
    from kernelforge.rewrite_by_flydsl.protocol import capabilities

    click.echo(json.dumps(capabilities(), indent=2, sort_keys=True))
    ctx.exit()


def _emit_rewrite_applyback_contract(ctx, _param, value):
    """Publish producer-owned example documents for cross-repository checks."""
    if not value or ctx.resilient_parsing:
        return
    from kernelforge.rewrite_by_flydsl.protocol import applyback_contract_example

    click.echo(json.dumps(applyback_contract_example(), indent=2, sort_keys=True))
    ctx.exit()


@main.command("forge-rewrite-by-flydsl")
@click.option(
    "--capabilities-json",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_emit_rewrite_capabilities,
    help="Print the machine-readable rewrite capability handshake and exit.",
)
@click.option(
    "--applyback-contract-json",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_emit_rewrite_applyback_contract,
    help="Print producer-authored apply-back manifest and outer-result examples.",
)
@click.option(
    "--source-kernel", required=True, help="Path to the source kernel to rewrite (e.g. a Triton .py or a .hip)"
)
@click.option(
    "--driver",
    required=True,
    help="Path to the rewrite measurement driver. A conforming driver is "
    "used unchanged; otherwise --prepare-driver authors or repairs "
    "this self-contained file.",
)
@click.option(
    "--prepare-driver/--no-prepare-driver",
    default=True,
    show_default=True,
    help="Author or repair a non-conforming dual-path rewrite driver before PORT.",
)
@click.option(
    "--invocation-spec-file",
    default="",
    help="Optional invocation evidence JSON used only by rewrite driver preparation.",
)
@click.option(
    "--logical-op-name",
    "--op-name",
    "op_name",
    required=True,
    help="Stable logical identity of the workload (a namespace or "
    "punctuation is allowed). KernelForge derives the FlyDSL factory "
    "symbol from it and reports the symbol in the result; never "
    "re-derive it downstream. --op-name is a deprecated alias.",
)
@click.option("--workspace", "workspace_dir", required=True, help="Git workspace dir")
@click.option("--experiments-dir", required=True, help="Where to write forge_experiments")
@click.option(
    "--target-functions",
    default="",
    help="Comma-separated source kernel entry names (the @triton.jit name, "
    "or the __global__ function name for HIP/CUDA)",
)
@click.option(
    "--source-language",
    default="",
    help="Language the source kernel is written in; one of the "
    "source_languages reported by --capabilities-json. Inferred "
    "from the file when omitted, but a caller whose profiler saw "
    "the kernel run should state it: a traced Triton kernel lives "
    "in a .py that names no language.",
)
@click.option(
    "--source-entry",
    default="",
    help="Host callable in the source that runs the kernel, used as the "
    "live correctness oracle + baseline: ref(x)->y. Auto-discovered "
    "if omitted.",
)
@click.option("--shapes-json", default="[]", help="JSON list of {M,N,dtype} shapes driving correctness + benchmark")
@click.option("--snr-threshold", default=DEFAULT_SNR_THRESHOLD_DB, type=float)
@click.option(
    "--flydsl-kernel-name", default="kernel.py", help="Filename of the produced FlyDSL kernel in the workspace"
)
@click.option(
    "--gpu-target",
    default=None,
    help="ROCm compilation architecture, e.g. gfx950 (also exported to env)",
)
@click.option(
    "--gpu-type",
    default=None,
    callback=_normalize_gpu_type,
    help="Hardware SKU for rewrite KB identities, e.g. mi355x",
)
@click.option(
    "--rewrite-kb/--no-rewrite-kb",
    default=True,
    show_default=True,
    help="Read and publish rewrite recipes.",
)
@click.option("--model", default=None, help="LLM model (overrides KERNEL_AGENTS_MODEL)")
@click.option("--permission-mode", default=None, help="Claude permission mode (default: acceptEdits)")
@click.option("--max-port-attempts", default=3, type=int, help="Max correctness-only port sessions before giving up")
@click.option(
    "--max-applyback-attempts",
    default=2,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum clean-room framework integration sessions.",
)
@click.option(
    "--max-hours",
    default=1.0,
    type=float,
    callback=_validate_max_hours,
    help="Total rewrite runtime budget (hours, minimum 1.0)",
)
@click.option(
    "--deadline-unix",
    default=0.0,
    type=float,
    help="Absolute UNIX deadline for PORT, OPTIMIZE, and apply-back finalization.",
)
@click.option(
    "--framework",
    default="",
    callback=_validate_rewrite_framework,
    help="Target framework for the apply-back patch (aiter, vllm, or sglang). "
    "Inferred from the source path when omitted.",
)
@click.option(
    "--applyback-import-module",
    "applyback_import_modules",
    multiple=True,
    help="Import target required to load before and after apply-back. Repeat for "
    "multiple modules; defaults to the source module inferred from its package.",
)
@click.option(
    "--git-branch",
    default="forge-rewrite-optimize",
    help="Development branch used by the nested FlyDSL forge-loop.",
)
@click.option(
    "--supervisor-backend", default="codex", help="OPTIMIZE supervisor backend on stall: 'codex' (default) or 'claude'"
)
@click.option(
    "--profile-timeout-sec", default=3600, type=int, help="OPTIMIZE: ceiling for the complete Analysis Agent workflow"
)
@click.option("--result-json", default=None, help="Write the result dict here (also printed)")
def forge_rewrite(
    source_kernel,
    driver,
    prepare_driver,
    invocation_spec_file,
    op_name,
    workspace_dir,
    experiments_dir,
    target_functions,
    source_language,
    source_entry,
    shapes_json,
    snr_threshold,
    flydsl_kernel_name,
    gpu_target,
    gpu_type,
    rewrite_kb,
    model,
    permission_mode,
    max_port_attempts,
    max_applyback_attempts,
    max_hours,
    deadline_unix,
    framework,
    applyback_import_modules,
    git_branch,
    supervisor_backend,
    profile_timeout_sec,
    result_json,
):
    """Rewrite a source kernel into FlyDSL and optimize it via forge-loop.

    Ports the source kernel (Triton, HIP, CUDA or C++) into an equivalent FlyDSL kernel
    (correctness-only PORT phase), then hands the FlyDSL kernel to forge-loop for
    optimization. With an existing framework git base, the final 20 minutes are
    reserved for one agent session that converts the verified best FlyDSL kernel
    into a cumulative framework apply-back patch. A conforming task driver is
    reused unchanged; otherwise the rewrite-specific preparation stage authors
    or repairs it from source and optional invocation evidence. The driver uses
    the ORIGINAL kernel as a live oracle + baseline and defines the operator's
    I/O, so this works for any operator (not just rowwise). Emits a
    JSON result (source_ms / flydsl_best_ms / speedup / correct), using the same
    __FORGE_RESULT__ patch-consumer contract as forge-loop.

    Example:
        kernelforge forge-rewrite-by-flydsl --source-kernel softmax.py \\
            --logical-op-name softmax \\
            --driver rewrite_driver.py --source-entry softmax \\
            --target-functions softmax_kernel_online \\
            --workspace /ws --experiments-dir /ws/forge_experiments \\
            --shapes-json '[{"M":8192,"N":8192,"dtype":"fp16"}]' --gpu-target gfx942
    """
    import os
    import re as _re

    if "--op-name" in sys.argv:
        click.echo(
            "warning: --op-name is deprecated; use --logical-op-name.",
            err=True,
        )

    overrides = {}
    if gpu_target:
        overrides["gpu_target"] = gpu_target
        os.environ["GPU_TARGET"] = gpu_target
    overrides["gpu_type"] = gpu_type
    if model:
        # Config exposes the model as ``agent_model`` (from_env keys on it); a
        # bare ``model`` override is silently ignored.
        overrides["agent_model"] = model
    rewrite_kb_enabled = bool(rewrite_kb)
    # A disabled KB must not validate ambient remote credentials.
    try:
        rewrite_knowledge_config = KnowledgeConfig.from_env(
            mode="local" if not rewrite_kb_enabled else None,
            remote_backend=REMOTE_BACKEND_KB_STORE,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    config = Config.from_env(
        workspace=workspace_dir,
        knowledge_config=rewrite_knowledge_config,
        **overrides,
    )

    targets = [t.strip() for t in _re.split(r"[,\n]", target_functions) if t.strip()]
    try:
        shapes = json.loads(shapes_json) if shapes_json else []
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"--shapes-json is not valid JSON: {e}")

    # Root containers: claude CLI bypassPermissions needs IS_SANDBOX=1.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.environ.setdefault("IS_SANDBOX", "1")

    from kernelforge.rewrite_by_flydsl import run_rewrite

    result = run_rewrite(
        op_name=op_name,
        source_kernel=source_kernel,
        driver=driver,
        workspace=workspace_dir,
        experiments_dir=experiments_dir,
        target_functions=targets,
        config=config,
        source_entry=source_entry,
        source_language=source_language,
        shapes=shapes,
        snr_threshold=snr_threshold,
        flydsl_kernel_name=flydsl_kernel_name,
        max_port_attempts=max_port_attempts,
        optimize_max_hours=max_hours,
        permission_mode=permission_mode,
        supervisor_backend=supervisor_backend,
        profile_timeout_sec=profile_timeout_sec,
        result_json=result_json,
        deadline_unix=deadline_unix,
        framework=framework,
        optimize_git_branch=git_branch,
        prepare_driver=prepare_driver,
        invocation_spec_file=invocation_spec_file,
        applyback_import_modules=applyback_import_modules,
        max_applyback_attempts=max_applyback_attempts,
        rewrite_kb_enabled=rewrite_kb_enabled,
    )
    # The structured result and sentinel were already emitted for callers to parse.
    # Also exit non-zero on a FAILED rewrite so pure shell/CI (which checks $?, not
    # the sentinel) cannot misread failure as success. A correct-but-not-faster port
    # is still a SUCCESS (speedup is a separate metric) -> key off port_ok.
    if not (result or {}).get("success"):
        raise SystemExit(1)


def _register_forge_fuse() -> None:
    """Attach the fusion pipeline's own Click command under `forge-fuse`.

    Importing it lazily keeps `kernelforge --help` free of the fusion
    pipeline's import cost, which pulls in the trace and validation stack.
    """
    from kernelforge.fusion.command import run as forge_fuse

    main.add_command(forge_fuse, name="forge-fuse")


_register_forge_fuse()


def _register_gemm_tune() -> None:
    """Attach the deterministic GEMM tuner under `gemm-tune`.

    It used to be a separate distribution with its own `forge-gemm-tune`
    console script; folding it in means forge ships exactly one CLI, and this
    is where its subcommands (`run`, `plan`, `evidence`) join it.
    """
    from kernelforge.gemm_tune.cli import gemm_tune

    main.add_command(gemm_tune, name="gemm-tune")


_register_gemm_tune()


if __name__ == "__main__":
    main()
