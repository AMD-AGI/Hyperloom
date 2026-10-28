# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Implementer agent factory — builds the per-iteration kernel-editing agent."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable


from kernelforge.agent_backends import (
    AgentRunSpec,
    AgentToolPolicy,
    create_registered_backend,
    resolve_agent_runtime,
    StdioMcpServer,
)
from kernelforge.agent_backends.runtime import agent_runtime, agent_runtime_settings
from kernelforge.agent_backends.session_resume import run_session_with_api_resume
from kernelforge.config import Config, resolve_agent_model
from kernelforge.workspace.edit_policy import FORGE_PROTECTED_PATHS, TOOL_OWNED_UNTRACKED_GLOBS
from kernelforge.knowledge.pr_knowledge.agent_tools import TOOL_NAMES as PR_TOOL_NAMES
from kernelforge.orchestrator.gate_tag import gate_outcome_tag
from kernelforge.loop.insession_gate import GateMeasurement, GateScope, InSessionGate
from kernelforge.loop.lessons import SessionEnd
from kernelforge.evaluation.scoring import (
    KEEP_MEASUREMENT_COUNT,
    KEEP_MIN_MARGIN_FRACTION,
    GateScoring,
    keep_t_critical,
)
from kernelforge.tracker.usage import UsageAccumulator

log = logging.getLogger(__name__)

# Repository / image_kernel tasks ship the correctness reference + tests INSIDE the repo tree (e.g. AITER's
# op_tests/.../test_<op>.py), which the in-session gate's default protected globs do not catch.
_REPO_EXTRA_PROTECTED_GLOBS = [
    "test_*.py",
    "*_test.py",
    "*_ref.py",
    "*_reference.py",
    "conftest.py",
]

# task_type values that mean "a full source tree, not a self-contained snippet".
_REPO_TASK_TYPES = {"repository", "image_kernel"}

# Wall-clock fallback for a session that never sized its own budget (every make_agent_fn caller except the forge-loop,
# e.g. the PORT loop).
_DEFAULT_SESSION_TIMEOUT_SEC = 90 * 60

# PR KB settings forwarded to the MCP child.
_PR_KB_CHILD_ENV_VARS = (
    "KB_STORE_URL",
    "HYPERLOOM_PR_MONITOR_ENABLED",
    "PR_KB_TIMEOUT_SEC",
    "PR_KB_BUDGET_SEC",
    "PR_KB_TOP_K",
    "PR_KB_CANDIDATE_CAP",
    "PR_KB_MIN_WORTH",
    "PR_KB_FALLBACK_MIN_WORTH",
)

_WORKSPACE_HYGIENE_RULE = (
    "Do NOT create or leave new non-ignored files outside the campaign's "
    "explicit --commit-new-path allowlist. Run "
    "one-off checks inline; if a temporary file is unavoidable, place it "
    "under forge_experiments/ and remove it before ending the turn."
)

# The counterpart to the rule above: that one keeps the workspace clean, this one keeps the artifact clean. Both
# are stated here rather than in a knowledge card because a card is read on demand and this holds every iteration.
_DELIVERABLE_HYGIENE_RULE = (
    "What you submit is a finished operator, not a scratchpad: when you end "
    "the turn it carries no `print` and reads no `os.environ` of its own. A "
    "sweep knob or a probe print may live in the kernel while you search; "
    "before ending the turn, replace each knob with the constant it selected, "
    "delete each probe, and re-run the driver on the file you submit. A knob "
    "left behind is indistinguishable, to everyone downstream, from live "
    "configuration. The one exception is an option a library you call "
    "exposes no other way: set that, and say in a comment why there is no "
    "API for it."
)


def _pr_kb_child_env() -> dict[str, str]:
    """Collect the PR KB settings that must reach the position-C server."""
    return {name: os.environ[name] for name in _PR_KB_CHILD_ENV_VARS if os.environ.get(name, "").strip()}


def _deadline_section(deadline_sec: int) -> str:
    """Tell the session its own wall-clock bound."""
    deadline_min = max(1, round(deadline_sec / 60))
    return (
        "## Session deadline\n"
        f"You have about {deadline_min} minutes of wall-clock for "
        "THIS session before it is cut off; a session cut off mid-turn hands "
        "off nothing. Well before then, stop improving and submit the best "
        "candidate you actually have through the clean handoff "
        "(candidate_submitted) — do not reach for a larger gain you cannot "
        "land in time.\n"
    )


def _gate_scoring_context(scoring: GateScoring) -> str:
    """Tell the session the exact rule its candidate will be judged by."""
    return (
        "\n\n## Authoritative mean case scoring state\n"
        "Each of three independent measurements is scored as "
        "`mean(pristine_case_ms / candidate_case_ms)` with equal case "
        "weight. The mean of those three scores must beat the "
        "current best by at least "
        f"{keep_t_critical(KEEP_MEASUREMENT_COUNT):g} standard "
        "errors of that mean -- a one-sided 95% Student-t test on "
        "the three scores -- floored at "
        f"{KEEP_MIN_MARGIN_FRACTION:.2%} of the current best. A "
        "quiet measurement therefore earns a small gain and a noisy "
        "one does not; repeatability is worth as much as speed.\n"
        f"Fixed pristine per-case ms: {dict(scoring.baseline_case_times)}\n"
        f"Current best pristine-relative score: {scoring.best_mean_case_speedup}.\n"
        "Raw `mean_ms` is diagnostic and never decides KEEP/REVERT."
    )


def _gate_scope(
    source_files: list[str],
    is_repo_task: bool,
    extra_protected_globs: list[str] | None,
    extra_protected_paths: list[str] | None,
) -> GateScope:
    """What every session's gate protects, before the per-session kernel and workspace are known."""
    return GateScope(
        # Declared sources seed profiling/JIT orientation.
        target_files=(source_files or None),
        # Combine repo reference/test globs (repo tasks) with any caller-supplied protected globs (e.g. the
        # rewrite port loop protects the source kernel it ports FROM, which is also the correctness oracle).
        extra_protected_globs=(_REPO_EXTRA_PROTECTED_GLOBS if is_repo_task else []) + list(extra_protected_globs or []),
        # Exact-path measurement files (e.g. the PORT phase's source kernel, which the driver imports as the
        # oracle).
        extra_protected_paths=extra_protected_paths,
    )


def _backend_expertise_section(config: Config, task: ImplementerTask, source_files: list[str]) -> str:
    """The kernel backend's prompt framed for this loop, or "" for a backend without one."""
    # Load the kernel backend's prompt as extra domain context; without it the agent gets only the generic
    # instructions below and misses backend discipline (e.g. ck's tile/pipeline tuning + stale-.cuda.o cleaning).
    from kernelforge.kernel_backends.base import build_single_kernel_backend_prompt

    kernel_backend_context = build_single_kernel_backend_prompt(
        config, task.kernel_backend_name, task_type=task.task_type, source_paths=source_files
    )
    if not kernel_backend_context:
        return ""
    # The backend prompts name the STEPS (build, run the driver, profile) but not the mechanism, because only this
    # loop knows it: this agent has Bash and the driver documented above, and no build/test/bench/pmc tools. So this
    # framing covers only what is specific to this loop.
    # Profiling off means the loop hands the session no profiler, so this framing must not promise one. (The loaded
    # kernel_backend_context is backend domain knowledge and is left as-is.)
    self_verbs = (
        "build, run, and profile the kernel YOURSELF via Bash"
        if task.profiling_enabled
        else "build and run the kernel YOURSELF via Bash"
    )
    return (
        f"{chr(10)}## Backend Expertise ({task.kernel_backend_name}){chr(10)}"
        "Backend guidance for choosing and implementing your edit. In this "
        f"loop you {self_verbs} to verify every change before finishing. "
        "After you finish, the loop also runs an SNR pre-filter + benchmark "
        "pass on your final kernel, and accepts it only if the task's own "
        "correctness suite passes too."
        f"{chr(10)}{chr(10)}{kernel_backend_context}"
    )


def _resolve_implementer_backend(config: Config, agent_backend: str | None, usage) -> tuple[Any, str]:
    """The backend this loop's sessions run on, and the provider that was asked for.

    They differ when the requested provider is unavailable and the registry
    falls back, which is said out loud rather than left for a later surprise.
    """
    runtime = agent_runtime(config)
    if agent_backend and agent_backend.strip().lower() != runtime.provider:
        runtime = resolve_agent_runtime(
            agent_backend,
            # ``config.agent_model`` belongs to the provider that was resolved,
            # not to the one being switched to; carrying it across the switch
            # is how a Claude model id reaches the OpenAI-protocol gateway.
            model=resolve_agent_model(agent_backend),
            # The executable and provider options, like the model, belong to the resolved provider.
            settings=replace(agent_runtime_settings(config), executable="", options={}),
        )
    requested_backend = runtime.provider
    precheck_cwd = (
        str(Path(config.workspace).resolve())
        if config.workspace and Path(config.workspace).is_dir()
        else str(Path.cwd())
    )
    backend = create_registered_backend(runtime, probe_cwd=precheck_cwd, usage=usage)
    if backend.name != requested_backend:
        reason = getattr(backend, "fallback_reason", "")
        reason_suffix = f" ({reason})" if reason else ""
        print(
            f"  [agent] {requested_backend} backend unavailable; falling back to {backend.name}{reason_suffix}",
            file=sys.stderr,
            flush=True,
        )
    return backend, requested_backend


def _session_cwd(kernel_path: str, workspace: Path | None, backend: Any) -> str:
    """Where an implementer session runs: beside the kernel, or at the workspace root for a provider that needs it."""
    if backend.capabilities.requires_workspace_cwd and workspace is not None and workspace.is_dir():
        return str(workspace.resolve())
    return str(Path(kernel_path).parent)


def _bullet_list(items: list[str]) -> str:
    """Render orientation hints as a bullet list for a prompt section."""
    return "\n".join(f"  - {item}" for item in items)


def _target_section(
    kernel_path: str,
    *,
    is_repo_task: bool,
    source_files: list[str],
    target_functions: list[str],
    assembly_session: bool,
) -> str:
    """The prompt section naming what this session may edit, and what is only a hint.

    On a repository task the declared source files and target functions are
    orientation, never the edit boundary; an assembly session is the opposite,
    with everything but the declared ``.s`` frozen.
    """
    if assembly_session:
        return (
            "## Editable assembly source\n"
            + _bullet_list(source_files)
            + "\nAll other tracked files are frozen, including the Python launcher and reference.\n"
        )
    if not (is_repo_task and (source_files or target_functions)):
        return f"## Target kernel\n{kernel_path}\n"
    section = (
        f"## Target kernel (anchor)\n{kernel_path}\n\n"
        "## Declared implementation entry points (starting hints, not an "
        "edit allowlist)\n"
        f"{_bullet_list(source_files or [kernel_path])}\n"
        "\nYou may edit any existing tracked implementation file in the "
        "workspace. Only protected driver, harness, test, scoring, and "
        "reference files are forbidden.\n"
    )
    if target_functions:
        section += (
            "\n## Target function hints\n"
            f"{_bullet_list(target_functions)}\n"
            "\nThese names guide profiling and code navigation; they do not "
            "restrict which functions may be edited. Trace them across the "
            "workspace (imports, dispatch, "
            "@triton.jit / __global__ definitions) and edit where the "
            "performance-relevant code actually lives. Let the profiler "
            "tell you which one is hot — not every listed function is "
            "necessarily exercised by the benchmark.\n"
        )
    return section


async def _run_outer_gate_turns(gate, backend, run_spec: AgentRunSpec, run_result, usage):
    """Drive the gate's Stop decision from out here, for a backend that does not run our hooks.

    Returns the final result, how many turns it took, and any workspace-safety
    error a resume raised -- which the caller must still fold into the gate's
    integrity verdict.
    """

    def count_result_target_edits(result) -> int:
        """Count incremental target edits from a hookless backend turn."""
        reported = getattr(result, "target_edit_count", None)
        if reported is not None:
            return max(0, int(reported))
        return gate.count_target_edits(run_spec.cwd, result.file_changes)

    continuation_turns = 1
    integrity_error: BaseException | None = None
    gate.edit_count += count_result_target_edits(run_result)
    while True:
        decision = await gate._on_stop({}, None, None)
        if decision.get("decision") != "block":
            break
        if not backend.capabilities.resumable or not run_result.session_id or not hasattr(backend, "resume"):
            gate.end_reason = "resume_unavailable"
            gate.findings.append("The gate requested another turn but no resumable session was available.")
            break

        feedback = (
            "The canonical Forge gate rejected your current candidate. "
            "Continue the SAME optimization session: inspect the concrete "
            "failure below, edit only the allowed kernel source files, "
            "re-run any useful checks, and finish with an updated PLAN: "
            "line. Do not modify git state or measurement files."
            "\n\n## Canonical gate feedback\n"
            f"{decision.get('reason', '')}"
        )
        try:
            resume_targets = list(run_spec.target_files)
            resume_targets.extend(str((Path(run_spec.cwd) / path).resolve()) for path in run_result.file_changes)
            resumed = await backend.resume(
                replace(run_spec, target_files=list(dict.fromkeys(resume_targets)), allow_dirty_targets=True),
                run_result.session_id,
                feedback,
                usage=usage,
            )
        except Exception as exc:  # isolation boundary: one continuation turn; the outer gate stays final
            log.warning("outer-gate resume failed: %s", exc, exc_info=True)
            if getattr(exc, "agent_safety_rejection", False):
                integrity_error = exc
            gate.end_reason = "resume_error"
            gate.findings.append(f"Agent resume failed: {type(exc).__name__}: {exc}")
            break
        continuation_turns += 1
        gate.edit_count += count_result_target_edits(resumed)
        resumed.tool_calls = [*run_result.tool_calls, *resumed.tool_calls]
        resumed.findings = [*run_result.findings, *resumed.findings]
        # A turn that left a benchmark running poisons every turn after it: resuming the session does not free
        # the device, so the contention has to outlive the turn that reported it.
        if not resumed.workspace_contention:
            resumed.workspace_contention = run_result.workspace_contention
        if not resumed.session_id:
            resumed.session_id = run_result.session_id
        run_result = resumed
    return run_result, continuation_turns, integrity_error


def _parse_output_tag(text: str, tag: str, cap: int, *, last: bool = False) -> str:
    """Pull a `TAG: ...` one-liner out of the agent output, sanitized: cut any trailing injected control text
    and cap the length.
    """
    found = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith(tag):
            continue
        value = stripped[len(tag) :].strip()
        for marker in ("Stop hook feedback", "## ", "Make your change"):
            index = value.find(marker)
            if index != -1:
                value = value[:index]
        value = value.strip()
        if len(value) > cap:
            # Truncate on a word boundary (not mid-word) + ellipsis.
            cut = value[:cap].rsplit(" ", 1)[0].rstrip(" ,;:-")
            value = (cut or value[:cap]) + "…"
        found = value
        if not last:
            break
    return found


def _session_end_reason(*, gate, run_result) -> str:
    """Why this session ended, for per-iteration analysis.

    The gate allowing a safe candidate handoff is ``candidate_submitted``;
    otherwise the SDK ended the query -- a turn cap (the subtype mentions
    ``max_turns``), another SDK error, or the agent voluntarily stopping
    ("success").
    """
    if any(line.strip().upper() == "SUBMIT_CANDIDATE" for line in run_result.text.splitlines()):
        return "candidate_submitted"
    if gate is not None and gate.end_reason:
        return gate.end_reason
    if run_result.end_reason and run_result.end_reason != "agent_stopped":
        return run_result.end_reason
    if "max_turns" in run_result.subtype:
        return "turn_cap"
    if run_result.subtype and run_result.subtype != "success":
        return f"sdk_{run_result.subtype}"
    return "agent_stopped"


def _fill_session_sink(
    session_sink: dict,
    *,
    gate,
    run_result,
    ending: SessionEnd,
    summarize,
) -> None:
    """Hand the runner what only this scope saw of the session.

    The runner never sees the ``AgentRunResult``, so this is the only place a
    workspace the reaper could not clear can reach the code that decides
    whether to run the canonical measurement.
    """
    # PLAN — one-sentence description of THIS iteration's FINAL modification.
    session_sink["plan"] = _parse_output_tag(run_result.text, "PLAN:", 160, last=True)
    session_sink["session_id"] = run_result.session_id
    session_sink["summarize"] = summarize
    session_sink["end_reason"] = ending.end_reason
    session_sink["turns"] = ending.turns
    session_sink["workspace_contention"] = run_result.workspace_contention
    if gate is not None:
        session_sink["findings"] = gate.findings_blob()
        session_sink["edit_count"] = gate.edit_count
        session_sink["gate_passed"] = gate.passed
        session_sink["wall_ms"] = gate.last_wall_ms
        session_sink["mean_case_speedup"] = gate.last_mean_case_speedup
        session_sink["benchmark_measurement"] = gate.last_bench_result
    elif run_result.findings:
        session_sink["findings"] = "\n---\n".join(run_result.findings)
        session_sink["edit_count"] = run_result.edit_count


@dataclass(frozen=True)
class ImplementerTask:
    """What an Implementer session works on: its program, backend, context, sources and protected files."""

    program_md: str
    kernel_backend_name: str = "ck"
    pre_task_context: str = ""
    pr_kb_repo: str = ""
    task_type: str = ""
    source_files: list[str] | None = None
    target_functions: list[str] | None = None
    profiling_enabled: bool = True
    commit_new_paths: list[str] | None = None
    extra_protected_globs: list[str] | None = None
    extra_protected_paths: list[str] | None = None


@dataclass(frozen=True)
class SessionOptions:
    """How each Implementer session runs: its budget, permission mode, provider and in-session gate."""

    # One implementer session's wall-clock budget (see kernelforge.cli.session_budget._forge_session_timeout_sec).
    session_timeout_sec: int | None = None
    permission_mode: str | None = None
    agent_backend: str | None = None
    insession_gate: bool = False
    # Whether an enabled gate also installs its Stop hook, which runs canonical correctness and a benchmark before the
    # session may end.
    insession_gate_stop_check: bool = True


@dataclass(frozen=True)
class ImplementerSetup:
    """Everything an Implementer session is built from besides the config: task, gate measurement and options."""

    task: ImplementerTask
    measurement: GateMeasurement = field(default_factory=GateMeasurement)
    session: SessionOptions = field(default_factory=SessionOptions)


def make_agent_fn(
    config: Config,
    setup: ImplementerSetup,
    *,
    usage: "UsageAccumulator | None" = None,
) -> Callable[..., Awaitable[str]]:
    """Create an agent_fn callback for the autonomous iteration loop."""
    task, measurement, session = setup.task, setup.measurement, setup.session
    program_md, kernel_backend_name, pre_task_context = task.program_md, task.kernel_backend_name, task.pre_task_context
    pr_kb_repo, profiling_enabled = task.pr_kb_repo, task.profiling_enabled
    extra_protected_globs = task.extra_protected_globs
    extra_protected_paths = list(task.extra_protected_paths or [])
    commit_new_paths = list(task.commit_new_paths or [])
    driver_script, max_blocks = measurement.driver_script, measurement.max_blocks
    permission_mode = session.permission_mode
    backend, requested_backend = _resolve_implementer_backend(config, session.agent_backend, usage)

    # Multi-file / repository awareness.
    source_files = list(task.source_files or [])
    target_functions = list(task.target_functions or [])
    is_repo_task = task.task_type.strip().lower() in _REPO_TASK_TYPES
    # Assembly preparation has already verified the session's single .s source.
    assembly_session = kernel_backend_name == "assembly"
    if assembly_session:
        from kernelforge.assembly.prepare import frozen_paths

        extra_protected_paths += frozen_paths(config.workspace, source_files)
        preparation_dir = Path(config.workspace) / "forge_experiments" / "assembly_preparation"
        extra_protected_paths.extend(str(path) for path in preparation_dir.glob("*") if path.is_file())
        commit_new_paths = []
        is_repo_task = False

    kernel_backend_section = _backend_expertise_section(config, task, source_files)

    prior_knowledge_section = f"{chr(10)}## Prior Knowledge{chr(10)}{pre_task_context}" if pre_task_context else ""

    # Stable across every iteration of a loop — placed in system_prompt so the underlying CLI's prompt cache reuses it
    # instead of re-billing each call.
    driver_base = Path(driver_script).name if driver_script else "forge_driver.py"
    # What the session is told to execute.
    driver_run = measurement.interposed_driver_path or driver_base
    # Empty unless a wrapper was interposed, so a session without one carries a byte-identical prompt.
    driver_interposition_section = (
        ""
        if driver_run == driver_base
        else f"""
## Run the driver through this command
    python3 {driver_run}
Never `python3 {driver_base}` directly. That wrapper passes every argument
through to {driver_base} unchanged, writes nothing of its own and returns the
driver's exit status — but it first takes a lock on the GPU this session shares
with others running right now. Timing two kernels on one device at once corrupts
both numbers, including the ones this session is judged on.
"""
    )
    profiling_dir = Path(config.local_knowledge_dir) / "common_methodology" / "profiling"
    # Suppressed when profiling is disabled so the implementer prompt carries no profiling affordance/hint at all.
    self_profiling_section = (
        ""
        if not profiling_enabled
        else f"""
## Self-profiling (optional, on demand)
If you need hardware profiling to decide the next change — which bottleneck (compute / bandwidth /
latency-occupancy), cache hit rates, occupancy, arithmetic intensity — profile the kernel YOURSELF
instead of guessing rocprof-compute's CLI:
    python3 {profiling_dir}/rocpc_profile.py --driver {driver_run} [--roofline] [--kernel <index>]
It prints rocprof-compute's Top-Stats + System Speed-of-Light tables (+ the Roofline section with
--roofline) and keeps the raw workload. Find your kernel's index in the printed Top-Stats table and
re-run with --kernel <index> to isolate its Speed-of-Light. Then classify the bottleneck (compute /
bandwidth / latency-occupancy) yourself using measure_triage.md and measure_roofline.md in
{profiling_dir} ; the script's mechanics are in {profiling_dir}/measure_rocpc_workflow.md . Use the
script rather than raw `rocprof-compute` (it finds a usable interpreter, runs profile+analyze, and
prints the tables). If this environment lacks rocprof-compute's deps the script prints an
"unavailable — skipping" notice and exits without data; just proceed without profiling (don't retry it).
"""
    )

    # Register on-demand PR tools only for a resolved repo and MCP backend.
    pr_mcp_servers: dict[str, StdioMcpServer] = {}
    if pr_kb_repo and backend.capabilities.mcp:
        pr_mcp_servers["pr_monitor"] = StdioMcpServer(
            command=sys.executable,
            args=("-m", "kernelforge.knowledge.pr_knowledge.mcp_server"),
            env={"PR_KB_REPO": pr_kb_repo, **_pr_kb_child_env()},
            startup_timeout_sec=15,
            # Claude ignores this field; the REST client also enforces a budget.
            tool_timeout_sec=60,
            tools=tuple(f"mcp__pr_monitor__{name}" for name in PR_TOOL_NAMES),
        )

    base_system_prompt = f"""\
You are a GPU kernel optimization agent in an autonomous iteration loop.

## Your Task
Each invocation: read the target kernel, review the recent iteration history,
and modify it to improve performance. You MAY make multiple changes across
several places this iteration — you are not limited to a single change. Then
explain your rationale in one sentence.

## Program (what to optimize and how)
{program_md}
{prior_knowledge_section}
{kernel_backend_section}

## Instructions
1. Use Read to fetch the current kernel file (path supplied per invocation).
2. Study the "Prior experience" section: it has a trajectory table of every past
   attempt + score, full diffs of the best/near-miss attempts, and a path to the
   solution archive (forge_experiments/candidates/iter_NNN/). Read prior kernels
   there to compare approaches, reuse a good idea, or COMBINE two — the file on
   disk is only the current best, not the full search history.
3. Identify the highest-impact change(s) based on PMC data and past results.
4. Use Edit to make your change(s) — you may edit multiple places this iteration,
   not just one. The target file may NOT be self-contained: if the
   performance-relevant code lives in other workspace files it includes/imports
   (headers, modules, dispatch/config, JIT templates), trace them with Grep/Glob
   and make the change there too — such cross-file edits are kept/reverted
   together with the kernel. Never edit the measurement / driver / harness files
   (those edits are blocked). That boundary is the whole of it: gaming means
   changing what measures you. Caching, memoization and compile-time
   specialization are ordinary optimizations and are NOT gaming, provided they
   still compute the right answer for the inputs the harness supplies. Before
   refusing a mechanism because it would compromise the benchmark, read the
   harness — it is in the workspace — and cite the lines that say so. An
   assumption about what the harness does is not a reason.
5. {_WORKSPACE_HYGIENE_RULE}
6. {_DELIVERABLE_HYGIENE_RULE}
7. As your last output, output one line starting with `PLAN:` — a SHORT headline
   (≤ ~12 words, one clause, plain prose, NO code/syntax) naming the optimization
   now in the file that will be committed and benchmarked, e.g. "vectorize global
   loads to 128-bit". Name only what you KEPT, not abandoned attempts or bug-fix
   minutiae. This is the iteration's headline.

You do NOT write a takeaway for the next iteration here. After this session ends
you will be asked, in a separate turn, to record everything you explored — so
keep exploring until you are done rather than reserving effort for a summary.
{driver_interposition_section}
{self_profiling_section}
## Tool usage — token discipline
Every Bash invocation's stdout/stderr is billed back to you on the next turn.
Never `cat` a whole file — use the Read tool (it's cheaper than a shell pipe).
"""

    # In-session self-correction mode: the agent may build/test/fix itself inside ONE session.
    gate_enabled = bool(session.insession_gate and driver_script)
    # Without the stop check the gate is protection only: its hooks deny an edit or a shell write to the measurement
    # surface while the session runs, and nothing decides when the session may end.
    gate_stop_check = gate_enabled and session.insession_gate_stop_check
    gate_scope = _gate_scope(source_files, is_repo_task, extra_protected_globs, extra_protected_paths)
    gate_system_prompt = f"""\
You are a GPU kernel optimization agent working in ONE self-correcting session.

## Your Task
Improve the target kernel's performance while keeping it numerically correct.
Unlike a one-shot edit, you own a full edit→verify→fix loop IN THIS SESSION:
edit the kernel, build/run it yourself to check, read any error, and fix it —
repeat until the kernel is CORRECT and FASTER than the current best.

## Program (what to optimize and how)
{program_md}
{prior_knowledge_section}
{kernel_backend_section}

## The measurement driver ({driver_base}) — you run it, you never edit it
`{driver_base}` in the current directory is the SAME script the outer loop uses to
judge your kernel. It is yours to READ and to RUN; it is NOT yours to change.
- BEFORE you optimize, READ it (`Read {driver_base}`). It is the ground truth for
  WHAT you are optimizing: the target op(s), the exact shapes/cases that get
  scored, the correctness reference, and how correctness and performance are
  measured. Do not guess these from the kernel — confirm them here.
- Get a CORRECTNESS verdict yourself:
    `python3 {driver_run}`
  prints a correctness line (e.g. `SNR: <db> dB` or `allclose: True/False`);
  this always runs every scored case.
- Get a PERFORMANCE number yourself:
    `python3 {driver_run} --warmup 3 --iters 20 --bench-mode`
  prints per-case `case_ms: <case> <ms>` plus one `mean_ms: <ms>` aggregate
  (the arithmetic mean across cases). `mean_ms` is diagnostic only. The score is
  the equal-weight arithmetic mean of per-case speedups:
  `mean(pristine_case_ms / candidate_case_ms)`. Run this before you stop; the
  gate reports the resulting mean case speedup.
- You may run these (and read the driver) as often as you like — executing the
  driver is always allowed. What is HARD-DENIED is any EDIT or shell write to it
  (Edit/Write, `>`/`>>` redirects, `sed -i`, `rm`/`mv`/`cp`/`tee`, in-process
  `open(...,'w')`, etc.). Optimize the kernel, never the measurement.
{driver_interposition_section}
## How to work (IMPORTANT)
1. Use Read to inspect the target kernel (path supplied per invocation). Also
   study the "Prior experience" section: it has a trajectory table of every past
   attempt + score, full diffs of the best/near-miss attempts, and a path to the
   solution archive (forge_experiments/candidates/iter_NNN/) where every prior
   kernel is saved in full. Read prior kernels there to compare, reuse, or COMBINE
   approaches — the file on disk is only the current best, not the search history.
2. Use Edit to make a change with a clear hypothesis.
3. VERIFY IT YOURSELF with Bash before finishing, using the driver documented
   above (run from the current directory):
     `python3 {driver_run}` → full-suite correctness verdict
     `python3 {driver_run} --warmup 3 --iters 20 --bench-mode`
                              → per-case timings plus diagnostic `mean_ms`
   If correctness fails or a compile error is raised, READ the error and FIX it —
   do not stop on a broken kernel. Optimize the mean of per-case speedups,
   not raw `mean_ms`; the in-session gate computes the authoritative score.
4. As soon as you have the FIRST candidate that you believe is correct and
   faster than the current best, STOP exploring. Submit it immediately to the
   outer loop, which is the sole authority for canonical correctness, benchmark,
   KEEP, and REVERT. Do not spend the remaining Session searching for a larger
   gain after you already have a plausible improvement.
5. There is NO hard edit budget — make as many changes to non-protected tracked
   implementation files as a direction needs, verifying as you go. What IS bounded is fruitless
   stopping: each time you STOP with a candidate that is correct but not faster
   than best, the gate sends you back to keep improving, and after {max_blocks}
   such rejected stops it ends the session and hands off. So do not circle
   micro-tweaks inside the noise band — if a direction is not beating best,
   switch to a materially different one rather than resubmitting near-ties.

## Rules
- The target kernel file may NOT be self-contained. If the performance-relevant
  code lives in other workspace files it includes/imports (headers, modules,
  dispatch/config layers, JIT template sources), trace them with Grep/Glob/Read
  and edit those files too — you are not restricted to the single target file.
  The loop stages and keeps/reverts ALL your tracked source edits together, so a
  cross-file change is validated and benchmarked as one unit.
- Do NOT change the kernel's public function signature or delete needed imports.
- {"Optimize only the selected assembly; keep the frontend, launcher and ABI frozen." if kernel_backend_name == "assembly" else "Keep the kernel in its original backend/DSL (do not rewrite in another language)."}
- Do NOT edit the test harness / driver (the files that measure your kernel);
  such edits are blocked. Optimize the kernel, not the measurement. That is the
  whole boundary: gaming means changing what measures you. Caching, memoization
  and compile-time specialization are ordinary optimizations and are NOT gaming,
  provided they still compute the right answer for the inputs the harness
  supplies. Before refusing a mechanism because it would compromise the
  benchmark, read the harness — it is in the workspace — and cite the lines that
  say so. An assumption about what the harness does is not a reason.
- {_WORKSPACE_HYGIENE_RULE}
- {_DELIVERABLE_HYGIENE_RULE}
- As your VERY LAST output, after all edits/fixes are done and the kernel is in
  its final state, output one line starting with `PLAN:` — a SHORT headline
  (≤ ~12 words, one clause, plain prose, NO code/syntax) naming the optimization
  NOW IN THE FILE (the one that will be committed and benchmarked), e.g. "raise
  num_stages to 3 to pipeline loads". If you tried several things this session,
  name only what you KEPT, not abandoned attempts or bug-fix minutiae. This is
  the iteration's headline.
- Then output exactly `SUBMIT_CANDIDATE` on its own final line and end the
  Session. This hands the current tracked diff to outer canonical validation.
- You do NOT write a takeaway for the next iteration here. Once this Session
  ends you will be asked, in a separate turn, to record every direction you
  explored — including the ones you abandoned. So spend this Session exploring,
  and do not hold back effort for a summary.

{self_profiling_section}
## Tool usage — token discipline
Never `cat` a whole file — use the Read tool.
"""

    # One value drives both the run spec's hard deadline and the deadline the session is told, so the enforced cut
    # and the stated cut can never disagree.
    session_deadline_sec = (
        session.session_timeout_sec
        if session.session_timeout_sec is not None
        else max(backend.runtime.timeout_sec, _DEFAULT_SESSION_TIMEOUT_SEC)
    )

    async def agent_fn(
        kernel_path: str,
        experiment_history: str,
        session_sink: dict | None = None,
        gate_scoring: GateScoring = GateScoring(),
    ) -> str:
        # A caller that reads nothing back still gets a sink, so the session records into one place unconditionally.
        if session_sink is None:
            session_sink = {}
        # Mark the session before entering the provider.
        progress_log: list[str] = []
        # IterationLoop sets this before invoking arbitrary agent callbacks; setdefault provides the same contract
        # when make_agent_fn is used standalone without obscuring the outer loop's earlier marker.
        session_sink.setdefault("session_started", True)
        session_sink["progress_log"] = progress_log

        target_section = _target_section(
            kernel_path,
            is_repo_task=is_repo_task,
            source_files=source_files,
            target_functions=target_functions,
            assembly_session=assembly_session,
        )

        prompt = f"""\
{target_section}
{_deadline_section(session_deadline_sec)}
## Prior experience (distilled constraints + recent iterations)
{experiment_history if experiment_history else "(none — this is iteration 1)"}

Make your change(s) now.
"""

        # One fixed interaction budget per candidate Session.
        turn_cap = config.max_turns
        configured_workspace = Path(config.workspace) if config.workspace else None
        run_cwd = _session_cwd(kernel_path, configured_workspace, backend)

        gate = None
        system_prompt = base_system_prompt
        hooks = None
        if gate_enabled:
            gate = InSessionGate(
                measurement,
                # The declared tree, not one guessed from the driver's location: forge-fuse keeps its driver in the
                # run's output dir, which is outside the workspace and is not a repository.
                replace(gate_scope, kernel_file=kernel_path, workspace=configured_workspace),
                baseline_case_times=gate_scoring.baseline_case_times,
                best_mean_case_speedup=gate_scoring.best_mean_case_speedup,
            )
            if gate_stop_check:
                system_prompt = gate_system_prompt + _gate_scoring_context(gate_scoring)
            # Built only for a provider that runs them. A backend which ignores
            # ``AgentRunSpec.hooks`` drops the whole group without a word, so
            # attaching one anyway makes this call site read as protection the
            # session does not have -- the confusion the outer gate below
            # exists to answer. Keyed on the backend that was resolved rather
            # than the one that was asked for, so a provider fallback carries
            # the decision with it.
            if backend.capabilities.stop_hooks:
                hooks = gate.make_agent_hooks(stop_check=gate_stop_check)

        run_spec = AgentRunSpec(
            system_prompt=system_prompt,
            user_prompt=prompt,
            role="implementer",
            cwd=run_cwd,
            writable=True,
            timeout_sec=session_deadline_sec,
            tool_policy=AgentToolPolicy(
                read=True,
                search=True,
                write=True,
                shell=True,
                max_turns=turn_cap,
                permission_mode=permission_mode or "",
                bare=gate is None,
                thinking_budget_tokens=3000,
            ),
            target_files=(source_files or [kernel_path]),
            commit_new_paths=list(commit_new_paths),
            driver_script=driver_script or "",
            protected_path_rules=FORGE_PROTECTED_PATHS,
            protected_globs=list(gate_scope.extra_protected_globs),
            # The loop writes its own ledger into the workspace it hands the implementer, and the kernel's runtime
            # leaves a JIT cache there, so every iteration starts from a worktree the caller already dirtied.
            allow_dirty_baseline=True,
            ignored_untracked_globs=list(TOOL_OWNED_UNTRACKED_GLOBS),
            protected_paths=list(extra_protected_paths),
            hooks=hooks,
            mcp_servers=pr_mcp_servers,
            progress_log=progress_log,
        )

        def _finalize_integrity(error: BaseException | None = None) -> None:
            if gate is None:
                return
            reason = gate.finalize_integrity()
            if error is not None and getattr(error, "agent_safety_rejection", False):
                reason = reason or (f"backend workspace safety rejection: {type(error).__name__}: {error}")
                gate.integrity_reason = reason
                gate.integrity_violation = True
                gate.integrity_verdict = "violation"
                finding = f"Protected workspace integrity violation: {reason}"
                if finding not in gate.findings:
                    gate.findings.append(finding[:1200])
            session_sink["integrity_verdict"] = gate.integrity_verdict
            session_sink["integrity_violation"] = gate.integrity_violation
            session_sink["integrity_reason"] = gate.integrity_reason
            session_sink["integrity_restore"] = gate.restore_protected_files

        # A candidate Session is expensive: by the time the gateway drops it, the agent has usually already read the
        # kernel, edited it, and paid for a build+bench.
        try:
            run_result = await run_session_with_api_resume(
                backend,
                run_spec,
                usage=usage,
            )
        except BaseException as error:
            _finalize_integrity(error)
            raise
        num_turns = run_result.num_turns
        integrity_error: BaseException | None = None
        # A backend that does not run our hooks gets the same Stop decision driven from out here, between explicit
        # resume turns.
        uses_outer_gate = gate_stop_check and not backend.capabilities.stop_hooks
        if uses_outer_gate:
            run_result, num_turns, integrity_error = await _run_outer_gate_turns(
                gate, backend, run_spec, run_result, usage
            )

        # Stop hooks are not guaranteed to run: turn caps, SDK failures, and cancellation can all terminate a session
        # first.
        _finalize_integrity(integrity_error)

        end_reason = _session_end_reason(gate=gate, run_result=run_result)

        edit_count, passed = run_result.edit_count, "-"
        text = run_result.text or "no rationale provided"
        if gate is not None:
            edit_count, passed = gate.edit_count, gate.passed
            text = f"{gate_outcome_tag(gate, end_reason, num_turns)} {text}"
        # One structured line per session (stdout is captured by the caller's logs) so a run's end-reason distribution
        # is analyzable without the LLM.
        print(
            f"  [session-end] backend={backend.name} reason={end_reason} "
            f"edits={edit_count if edit_count else '-'} "
            f"turns={num_turns if num_turns is not None else '?'} "
            f"pass={passed}",
            flush=True,
        )

        # Hand back structured session info so the runner can feed the ExperienceLedger with the gate's objective
        # findings, and (via ``summarize``) ask this exact session to record what it explored.
        _fill_session_sink(
            session_sink,
            gate=gate,
            run_result=run_result,
            ending=SessionEnd(end_reason, turns=num_turns),
            summarize=_make_session_summarizer(
                backend=backend,
                spec=run_spec,
                session_id=run_result.session_id,
                usage=usage,
            ),
        )

        return text[:200]

    _describe_agent_fn(agent_fn, backend, requested_backend)
    return agent_fn


def _describe_agent_fn(agent_fn: Callable[..., Awaitable[str]], backend, requested_backend: str) -> None:
    """Expose the backend serving ``agent_fn`` and the way to release it to the callers that build actors from it."""
    setattr(agent_fn, "backend_name", backend.name)
    setattr(agent_fn, "backend_model", backend.runtime.model)
    setattr(agent_fn, "requested_backend", requested_backend)
    # Only a backend that keeps processes between sessions defines close(); the rest, out-of-tree providers included,
    # have nothing to release.
    setattr(agent_fn, "close", getattr(backend, "close", lambda: None))


def _make_session_summarizer(
    *,
    backend,
    spec: AgentRunSpec,
    session_id: str,
    usage=None,
) -> Callable[[str], Awaitable[str]] | None:
    """Build an async callable that resumes ONE finished implementer session."""
    if not session_id or not getattr(backend.capabilities, "resumable", False):
        return None
    if not hasattr(backend, "resume"):
        return None

    from kernelforge.loop.lessons import SUMMARIZER_ROLE

    implementer_policy = spec.tool_policy or AgentToolPolicy()
    summary_spec = replace(
        spec,
        # The implementer's system prompt and tool definitions open the cached prefix the whole session was billed
        # against; changing either makes the summary rewrite that prefix instead of reading it. The summarizer role
        # travels in the user turn, and a provider that describes tools beyond the narrower tool_policy below must
        # refuse them whatever the permission mode or settings say (the Claude backend denies them in a PreToolUse
        # hook).
        declared_tool_policy=implementer_policy,
        writable=False,
        hooks=None,
        allow_dirty_targets=True,
        allow_untracked=True,
        read_only_resume=True,
        protected_globs=["*"],
        # Preserve the implementer's progress log as a stable fallback record.
        progress_log=None,
        tool_policy=replace(
            implementer_policy,
            read=True,
            search=True,
            write=False,
            shell=False,
            extra_tools=(),
            # Enough turns to check a path or a number it half-remembers, not enough to start exploring the workspace.
            max_turns=4,
        ),
    )

    async def summarize(prompt: str) -> str:
        result = await backend.resume(summary_spec, session_id, f"{SUMMARIZER_ROLE}\n\n{prompt}", usage=usage)
        return result.text or ""

    return summarize
