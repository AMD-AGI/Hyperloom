# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Implementer agent factory — builds the per-iteration kernel-editing agent."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Awaitable


from kernelforge.agent_backends import (
    AgentRunSpec,
    AgentToolPolicy,
    create_registered_backend,
    resolve_agent_runtime,
    StdioMcpServer,
)
from kernelforge.agent_backends.session_resume import run_session_with_api_resume
from kernelforge.config import Config
from kernelforge.mcp_server.pr_stdio_server import TOOL_NAMES as PR_TOOL_NAMES
from kernelforge.loop.scoring import (
    DEFAULT_SNR_THRESHOLD_DB,
    KEEP_MEASUREMENT_COUNT,
    KEEP_MIN_MARGIN_FRACTION,
    keep_t_critical,
)
from kernelforge.tracker.usage import UsageAccumulator
from kernelforge import rtk

# Repository / image_kernel tasks ship the correctness reference + tests INSIDE
# the repo tree (e.g. AITER's op_tests/.../test_<op>.py), which the in-session
# gate's default protected globs do not catch. These extra globs stop the agent
# from editing the reference to game the correctness gate. Applied ONLY for
# repository/image_kernel tasks so single-file tasks are unaffected. A target
# source file always stays editable (the gate short-circuits target files).
_REPO_EXTRA_PROTECTED_GLOBS = [
    "test_*.py",
    "*_test.py",
    "*_ref.py",
    "*_reference.py",
    "conftest.py",
]

# task_type values that mean "a full source tree, not a self-contained snippet".
_REPO_TASK_TYPES = {"repository", "image_kernel"}

# Wall-clock fallback for a session that never sized its own budget (every
# make_agent_fn caller except the forge-loop, e.g. the PORT loop). The claude
# backend used to IGNORE the run spec's timeout, so these callers were bounded
# only by the turn cap; now that it HONOURS it, falling back to the provider's
# 30-minute runtime default would truncate a legitimate correctness/PORT
# session mid-work -- a cold CK build alone can take ~26 minutes. Fall back to
# the same 90-minute floor the forge budget uses as "enough to read + edit +
# build + bench", so honouring the timeout does not silently shorten sessions
# that pre-date the change. A caller that wants a tighter or looser bound passes
# session_timeout_sec explicitly.
_DEFAULT_SESSION_TIMEOUT_SEC = 90 * 60

# PR KB settings forwarded to the MCP child.
_PR_KB_CHILD_ENV_VARS = (
    "PRIMUS_CORTEX_PR_API",
    "PR_KB_TIMEOUT_SEC",
    "PR_KB_BUDGET_SEC",
    "PR_KB_TOP_K",
    "PR_KB_CANDIDATE_CAP",
    "PR_KB_MIN_WORTH",
    "PR_KB_FALLBACK_MIN_WORTH",
)


def _pr_kb_child_env() -> dict[str, str]:
    """Collect the PR KB settings that must reach the position-C server."""
    return {name: os.environ[name] for name in _PR_KB_CHILD_ENV_VARS if os.environ.get(name, "").strip()}


def make_agent_fn(
    config: Config,
    program_md: str,
    kernel_backend_name: str = "ck",
    pre_task_context: str = "",
    pr_kb_repo: str = "",
    usage: "UsageAccumulator | None" = None,
    insession_gate: bool = False,
    # Whether an enabled gate also installs its Stop hook, which runs canonical
    # correctness and a benchmark before the session may end. False installs the
    # gate's protection hooks alone, for a caller whose sessions run at the same
    # time as each other: the device times one thing at a time, so such a
    # session must not benchmark itself. Ignored when the gate is off.
    insession_gate_stop_check: bool = True,
    driver_script: str | None = None,
    # The wrapper script the session is told to run the driver through, when
    # that is not the driver itself. A concurrent lane is given one that takes
    # the shared device lock first, and the lock only works if the session runs
    # it: the driver sitting beside it stays readable and runnable, so naming
    # the wrapper here rather than in a per-invocation note is what keeps the
    # instruction from contradicting itself. The driver named by
    # ``driver_script`` remains the protected file and the one to read.
    interposed_driver_path: str | None = None,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    max_blocks: int = 10,
    # One implementer session's wall-clock budget (see cli._forge_session_timeout_sec).
    # None falls back to the backend runtime's own timeout: the same value is used
    # for the run spec AND the deadline the session is told, so the two never
    # disagree. This is what the claude backend now enforces -- a turn cap never
    # bounded time (it fired on 2.2% of sessions), so a long session ran until
    # something outside killed it.
    session_timeout_sec: int | None = None,
    validation_timeout_sec: int = 1800,
    bench_timeout_sec: int = 300,
    bench_repeat: int = 1,
    permission_mode: str | None = None,
    task_type: str = "",
    source_files: list[str] | None = None,
    target_functions: list[str] | None = None,
    profiling_enabled: bool = True,
    agent_backend: str | None = None,
    extra_protected_globs: list[str] | None = None,
    extra_protected_paths: list[str] | None = None,
    correctness_only: bool = False,
) -> Callable[..., Awaitable[str]]:
    """Create an agent_fn callback for the autonomous iteration loop.

    Returns an async function with signature:
        async fn(kernel_path: str, experiment_history_json: str) -> str

    Each call runs one implementer session through the configured backend that:
      1. Reads the kernel file
      2. Reviews experiment history (last 5 iterations)
      3. Proposes one or more modifications to the kernel
      4. Returns the rationale for the change

    When a :class:`~kernelforge.tracker.usage.UsageAccumulator` is supplied
    via ``usage``, terminal provider usage is folded into it so the loop can
    persist the run's total LLM cost.
    """
    runtime = config.agent_runtime()
    if agent_backend and agent_backend.strip().lower() != runtime.provider:
        runtime = resolve_agent_runtime(
            agent_backend,
            model=config.agent_model,
            executable=config.agent_cli,
            timeout_sec=config.agent_timeout_sec,
            reasoning_effort=config.agent_reasoning_effort,
            sandbox_mode=config.agent_sandbox_mode,
            precheck=config.agent_precheck,
            fallback_provider=config.agent_fallback_provider,
            options=config.agent_options,
        )
    requested_backend = runtime.provider
    precheck_cwd = (
        str(Path(config.workspace).resolve())
        if config.workspace and Path(config.workspace).is_dir()
        else str(Path.cwd())
    )
    backend = create_registered_backend(
        runtime,
        probe_cwd=precheck_cwd,
        usage=usage,
    )
    if backend.name != requested_backend:
        reason = getattr(backend, "fallback_reason", "")
        reason_suffix = f" ({reason})" if reason else ""
        print(
            f"  [agent] {requested_backend} backend unavailable; falling back to {backend.name}{reason_suffix}",
            file=sys.stderr,
            flush=True,
        )
    elif backend.runtime.model != runtime.model:
        reason = getattr(backend, "model_fallback_reason", "")
        reason_suffix = f" ({reason})" if reason else ""
        print(
            f"  [agent] model {runtime.model} unavailable; falling back to {backend.runtime.model}{reason_suffix}",
            file=sys.stderr,
            flush=True,
        )
    backend_model = backend.runtime.model

    # Multi-file / repository awareness. For a single-file task these stay empty
    # and every branch below collapses to the original single-file behavior.
    source_files = [f for f in (source_files or []) if f]
    target_functions = [f for f in (target_functions or []) if f]
    is_repo_task = (task_type or "").strip().lower() in _REPO_TASK_TYPES

    def _bullets(items: list[str]) -> str:
        return "\n".join(f"  - {i}" for i in items)

    # Load the kernel backend's prompt as extra domain context; without it the agent
    # gets only the generic instructions below and misses backend discipline
    # (e.g. ck's tile/pipeline tuning + stale-.cuda.o cleaning).
    kernel_backend_context = ""
    try:
        from kernelforge.kernel_backends.base import build_single_kernel_backend_prompt

        kernel_backend_context = build_single_kernel_backend_prompt(
            config, kernel_backend_name, task_type=task_type, source_paths=source_files
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"  Warning: kernel_backend prompt load failed for {kernel_backend_name} ({e}); using generic prompt",
            file=sys.stderr,
        )

    # Kernel-backend prompts name build/test/bench/pmc/registers as if they were tools.
    # This agent has Bash instead, so frame those names as shell steps it runs
    # and verifies itself, rather than forbidding them.
    kernel_backend_section = ""
    if kernel_backend_context:
        # Drop the profile/pmc mentions from this framing when profiling is
        # disabled, so the implementer prompt carries no profiling guidance. (The
        # loaded kernel_backend_context is backend domain knowledge and is left as-is.)
        _self_verbs = (
            "build, run, and profile the kernel YOURSELF via the Bash tool (compile, run the driver, run a profiler)"
            if profiling_enabled
            else "build and run the kernel YOURSELF via the Bash tool (compile, run the driver)"
        )
        _self_tools = (
            "`build`/`test`/`bench`/`pmc`/`registers`" if profiling_enabled else "`build`/`test`/`bench`/`registers`"
        )
        kernel_backend_section = (
            f"{chr(10)}## Backend Expertise ({kernel_backend_name}){chr(10)}"
            "Backend guidance for choosing and implementing your edit. In this "
            f"loop you {_self_verbs} to verify every change before finishing. "
            f"Where the guidance below names {_self_tools} tools, run those steps "
            "as shell commands via Bash. After you finish, the loop also runs an "
            "SNR pre-filter + benchmark pass on your final kernel, and accepts it "
            "only if the task's own correctness suite passes too."
            f"{chr(10)}{chr(10)}{kernel_backend_context}"
        )

    # `rtk` (token filter) is advertised to the agent ONLY when it's actually on
    # PATH; otherwise the agent would prefix every shell command with a missing
    # binary (command not found). Mirrors kernelforge.rtk.wrap_command, which
    # no-ops the same way. When rtk is absent the whole paragraph is dropped.
    if rtk.is_available():
        _rtk_guidance = (
            "Always prefix shell commands with `rtk` — it filters verbose output (ninja,\n"
            "cmake, git, grep, find, ls, rocprofv3, etc.) for 60-90% fewer tokens, and\n"
            "passes through unchanged for unknown commands. Examples:\n"
            "  - `rtk git diff` instead of `git diff`\n"
            "  - `rtk grep -r foo .` instead of `grep -r foo .`\n"
            "  - `rtk ninja -j4` instead of `ninja -j4`\n"
            "  - `rtk ls path/` instead of `ls path/`\n"
        )
        _rtk_guidance_terse = (
            "Prefix noisy shell commands with `rtk` to filter verbose output (ninja, cmake,\n"
            "git, grep, find, ls, rocprofv3, …) for 60-90% fewer tokens; it passes unknown\n"
            "commands through unchanged. "
        )
    else:
        _rtk_guidance = ""
        _rtk_guidance_terse = ""

    workspace_hygiene_rule = (
        "Do NOT create or leave new non-ignored files in the workspace. Run "
        "one-off checks inline; if a temporary file is unavoidable, place it "
        "under forge_experiments/ and remove it before ending the turn."
    )

    # Stable across every iteration of a loop — placed in system_prompt so the
    # underlying CLI's prompt cache reuses it instead of re-billing each call.
    # On-demand self-profiling affordance: point the agent at the canonical
    # profiling script + docs so it profiles its OWN kernel when it needs data,
    # instead of guessing rocprof-compute's CLI or tripping its dependency gate.
    driver_base = Path(driver_script).name if driver_script else "forge_driver.py"
    # What the session is told to execute. Identical to the driver unless the
    # caller interposed a wrapper, so the ordinary session's prompt is unchanged.
    driver_run = interposed_driver_path or driver_base
    # Empty unless a wrapper was interposed, so a session without one carries a
    # byte-identical prompt. It belongs in the system prompt rather than in a
    # per-invocation note because it is a hard requirement for the measurement
    # to mean anything, and a long session drifts away from its first message.
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
    # Suppressed when profiling is disabled so the implementer prompt carries no
    # profiling affordance/hint at all.
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
            args=("-m", "kernelforge.mcp_server.pr_stdio_server"),
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
{f"{chr(10)}## Prior Knowledge{chr(10)}{pre_task_context}" if pre_task_context else ""}
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
5. {workspace_hygiene_rule}
6. As your last output, output one line starting with `PLAN:` — a SHORT headline
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
{_rtk_guidance}Never `cat` a whole file — use the Read tool (it's cheaper than a shell pipe).
"""

    # In-session self-correction mode: the agent may build/test/fix itself inside
    # ONE session. Claude gates on a Stop hook while Codex runs the same canonical
    # gate between explicit resume turns; the outer loop stays the only canonical
    # validation authority.
    gate_enabled = bool(insession_gate and driver_script)
    # Without the stop check the gate is protection only: its hooks deny an edit
    # or a shell write to the measurement surface while the session runs, and
    # nothing decides when the session may end. The self-correcting prompt below
    # describes a gate that rejects a stop, so a session that has no such gate
    # keeps the ordinary prompt instead of being told about one.
    gate_stop_check = gate_enabled and insession_gate_stop_check
    gate_system_prompt = f"""\
You are a GPU kernel optimization agent working in ONE self-correcting session.

## Your Task
Improve the target kernel's performance while keeping it numerically correct.
Unlike a one-shot edit, you own a full edit→verify→fix loop IN THIS SESSION:
edit the kernel, build/run it yourself to check, read any error, and fix it —
repeat until the kernel is CORRECT and FASTER than the current best.

## Program (what to optimize and how)
{program_md}
{f"{chr(10)}## Prior Knowledge{chr(10)}{pre_task_context}" if pre_task_context else ""}
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
- Keep the kernel in its original backend/DSL (do not rewrite in another language).
- Do NOT edit the test harness / driver (the files that measure your kernel);
  such edits are blocked. Optimize the kernel, not the measurement. That is the
  whole boundary: gaming means changing what measures you. Caching, memoization
  and compile-time specialization are ordinary optimizations and are NOT gaming,
  provided they still compute the right answer for the inputs the harness
  supplies. Before refusing a mechanism because it would compromise the
  benchmark, read the harness — it is in the workspace — and cite the lines that
  say so. An assumption about what the harness does is not a reason.
- {workspace_hygiene_rule}
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
{_rtk_guidance_terse}Never `cat` a whole file — use the Read tool.
"""

    async def agent_fn(
        kernel_path: str,
        experiment_history: str,
        session_sink: dict | None = None,
        baseline_case_times: dict | None = None,
        best_mean_case_speedup: float | None = None,
    ) -> str:
        # Mark the session before entering the provider. If the backend raises
        # before returning an AgentRunResult (turn cap, cancellation, transport
        # failure), the outer loop still knows an agent actually ran and can
        # persist an outcome-only lesson instead of treating it as a baseline.
        progress_log: list[str] = []
        if session_sink is not None:
            # IterationLoop sets this before invoking arbitrary agent callbacks;
            # setdefault provides the same contract when make_agent_fn is used
            # standalone without obscuring the outer loop's earlier marker.
            session_sink.setdefault("session_started", True)
            session_sink["progress_log"] = progress_log

        # Repository tasks: declared source files and target functions are
        # orientation hints, never the edit boundary. The hard boundary is the
        # protected measurement surface enforced by the gate/backend.
        if is_repo_task and (source_files or target_functions):
            files_for_prompt = source_files or [kernel_path]
            target_section = (
                f"## Target kernel (anchor)\n{kernel_path}\n\n"
                "## Declared implementation entry points (starting hints, not an "
                "edit allowlist)\n"
                f"{_bullets(files_for_prompt)}\n"
                "\nYou may edit any existing tracked implementation file in the "
                "workspace. Only protected driver, harness, test, scoring, and "
                "reference files are forbidden.\n"
            )
            if target_functions:
                target_section += (
                    "\n## Target function hints\n"
                    f"{_bullets(target_functions)}\n"
                    "\nThese names guide profiling and code navigation; they do not "
                    "restrict which functions may be edited. Trace them across the "
                    "workspace (imports, dispatch, "
                    "@triton.jit / __global__ definitions) and edit where the "
                    "performance-relevant code actually lives. Let the profiler "
                    "tell you which one is hot — not every listed function is "
                    "necessarily exercised by the benchmark.\n"
                )
        else:
            target_section = f"## Target kernel\n{kernel_path}\n"

        # One value drives both the run spec's hard deadline and the deadline
        # the session is told, so the enforced cut and the stated cut can never
        # disagree. A caller that never sized a session budget falls back to a
        # sane 90-minute floor (see _DEFAULT_SESSION_TIMEOUT_SEC), never the
        # provider's 30-minute runtime default, which the claude backend now
        # honours and would otherwise use to truncate a legitimate PORT session.
        session_deadline_sec = (
            session_timeout_sec
            if session_timeout_sec is not None
            else max(backend.runtime.timeout_sec, _DEFAULT_SESSION_TIMEOUT_SEC)
        )
        session_deadline_min = max(1, round(session_deadline_sec / 60))

        # Tell the session its own wall-clock bound. The claude backend now cuts
        # a session at this deadline; a session that only learns of it by being
        # killed mid-turn hands off nothing, so ask it to land the best candidate
        # it actually has through the clean handoff (candidate_submitted) before
        # the clock runs out rather than chasing a larger gain it cannot finish.
        deadline_section = (
            "## Session deadline\n"
            f"You have about {session_deadline_min} minutes of wall-clock for "
            "THIS session before it is cut off; a session cut off mid-turn hands "
            "off nothing. Well before then, stop improving and submit the best "
            "candidate you actually have through the clean handoff "
            "(candidate_submitted) — do not reach for a larger gain you cannot "
            "land in time.\n"
        )

        prompt = f"""\
{target_section}
{deadline_section}
## Prior experience (distilled constraints + recent iterations)
{experiment_history if experiment_history else "(none — this is iteration 1)"}

Make your change(s) now.
"""

        # One fixed interaction budget per candidate Session. Campaign duration
        # controls how many Sessions are admitted; turns bound each Session.
        turn_cap = config.max_turns
        cwd = str(Path(kernel_path).parent)
        run_cwd = cwd
        configured_workspace = Path(config.workspace) if config.workspace else None
        if (
            backend.capabilities.requires_workspace_cwd
            and configured_workspace is not None
            and configured_workspace.is_dir()
        ):
            run_cwd = str(configured_workspace.resolve())

        gate = None
        system_prompt = base_system_prompt
        if gate_enabled:
            from kernelforge.loop.insession_gate import InSessionGate

            gate = InSessionGate(
                driver_script=driver_script,
                snr_threshold=snr_threshold,
                baseline_case_times=baseline_case_times,
                best_mean_case_speedup=best_mean_case_speedup,
                kernel_file=kernel_path,
                max_blocks=max_blocks,
                stage_timeout_sec=validation_timeout_sec,
                bench_timeout_sec=bench_timeout_sec,
                bench_repeat=bench_repeat,
                # Declared sources seed profiling/JIT orientation. Edit counting
                # covers every non-protected implementation file.
                target_files=(source_files or None),
                # Combine repo reference/test globs (repo tasks) with any
                # caller-supplied protected globs (e.g. the rewrite port loop
                # protects the source kernel it ports FROM, which is also the
                # correctness oracle). Both are additive; None keeps prior behavior.
                extra_protected_globs=(
                    (_REPO_EXTRA_PROTECTED_GLOBS if is_repo_task else []) + list(extra_protected_globs or [])
                )
                or None,
                # Exact-path measurement files (e.g. the PORT phase's source kernel,
                # which the driver imports as the oracle). Same tier as the driver.
                extra_protected_paths=extra_protected_paths,
                # PORT (and any correctness-only phase): require only correctness;
                # the gate skips the perf benchmark entirely.
                correctness_only=correctness_only,
                # The interposed command, so the hooks refuse a driver run that
                # goes around it. Naming it in the prompt above states the
                # requirement; this is what holds it.
                interposed_driver_path=interposed_driver_path,
                # The declared tree, not one guessed from the driver's location:
                # forge-fuse keeps its driver in the run's output dir, which is
                # outside the workspace and is not a repository.
                workspace=configured_workspace,
            )
            if gate_stop_check:
                scoring_context = (
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
                    f"Fixed pristine per-case ms: {dict(baseline_case_times or {})}\n"
                    f"Current best pristine-relative score: {best_mean_case_speedup}.\n"
                    "Raw `mean_ms` is diagnostic and never decides KEEP/REVERT."
                )
                system_prompt = gate_system_prompt + scoring_context

        run_spec = AgentRunSpec(
            system_prompt=system_prompt,
            user_prompt=prompt,
            cwd=run_cwd,
            writable=True,
            timeout_sec=session_deadline_sec,
            reasoning_effort="max",
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
            driver_script=driver_script or "",
            protected_globs=((_REPO_EXTRA_PROTECTED_GLOBS if is_repo_task else []) + list(extra_protected_globs or [])),
            # The loop writes its own ledger into the workspace it hands the
            # implementer, and the kernel's runtime leaves a JIT cache there, so
            # every iteration starts from a worktree the caller already
            # dirtied. Judging the turn against HEAD refuses it for that
            # inherited state before the agent is asked anything; judge it
            # against what it inherited instead.
            allow_dirty_baseline=True,
            # A profiler writes where it is run, and it is run here. rocprofv3
            # drops these two next to the driver; a session was failed for them
            # rather than for anything it did. Named rather than forgiven
            # wholesale (``allow_untracked``), so every path nobody declared is
            # still refused.
            ignored_untracked_globs=[
                # Both entries must reach any depth: the profiler runs in
                # ``run_cwd``, which is the kernel file's parent (see above),
                # while the guard reports paths relative to the git toplevel.
                # ``fnmatch`` crosses "/", so the ``*_results.db`` form already
                # does; ``.rocprofv3/`` has to be spelled out twice.
                ".rocprofv3/*",
                "*/.rocprofv3/*",
                "*_results.db",
            ],
            protected_paths=list(extra_protected_paths or []),
            hooks=(gate.make_agent_hooks(stop_check=gate_stop_check) if gate is not None else None),
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
            if session_sink is not None:
                session_sink["integrity_verdict"] = gate.integrity_verdict
                session_sink["integrity_violation"] = gate.integrity_violation
                session_sink["integrity_reason"] = gate.integrity_reason
                session_sink["integrity_restore"] = gate.restore_protected_files

        # A candidate Session is expensive: by the time the gateway drops it,
        # the agent has usually already read the kernel, edited it, and paid for
        # a build+bench. Resume the SAME session on an API failure so that work
        # survives; a turn cap or a deadline is left alone, because those mean
        # the agent answered.
        try:
            run_result = await run_session_with_api_resume(
                backend,
                run_spec,
                usage=usage,
            )
        except BaseException as error:
            _finalize_integrity(error)
            raise
        continuation_turns = 1
        integrity_error: BaseException | None = None
        # A backend that does not run our hooks gets the same Stop decision
        # driven from out here, between explicit resume turns. Only the mode that
        # asked for that decision gets it: without the stop check there is no
        # Stop hook to stand in for, and running one here would benchmark.
        uses_outer_gate = gate_stop_check and not backend.capabilities.stop_hooks
        if uses_outer_gate:

            def count_result_target_edits(result) -> int:
                """Count incremental target edits from a hookless backend turn."""
                reported = getattr(result, "target_edit_count", None)
                if reported is not None:
                    return max(0, int(reported))
                return gate.count_target_edits(
                    run_spec.cwd,
                    result.file_changes,
                )

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
                    resume_targets.extend(
                        str((Path(run_spec.cwd) / path).resolve()) for path in run_result.file_changes
                    )
                    resumed = await backend.resume(
                        replace(
                            run_spec,
                            target_files=list(dict.fromkeys(resume_targets)),
                            allow_dirty_targets=True,
                        ),
                        run_result.session_id,
                        feedback,
                        usage=usage,
                    )
                except Exception as exc:  # noqa: BLE001 - outer gate remains final
                    if getattr(exc, "agent_safety_rejection", False):
                        integrity_error = exc
                    gate.end_reason = "resume_error"
                    gate.findings.append(f"Agent resume failed: {type(exc).__name__}: {exc}")
                    break
                continuation_turns += 1
                gate.edit_count += count_result_target_edits(resumed)
                resumed.tool_calls = [
                    *run_result.tool_calls,
                    *resumed.tool_calls,
                ]
                resumed.findings = [
                    *run_result.findings,
                    *resumed.findings,
                ]
                # A turn that left a benchmark running poisons every turn after
                # it: resuming the session does not free the device, so the
                # contention has to outlive the turn that reported it.
                if not resumed.workspace_contention:
                    resumed.workspace_contention = run_result.workspace_contention
                if not resumed.session_id:
                    resumed.session_id = run_result.session_id
                run_result = resumed

        # Stop hooks are not guaranteed to run: turn caps, SDK failures, and
        # cancellation can all terminate a session first. This final scan is the
        # authoritative protected-integrity state for the outer runner.
        _finalize_integrity(integrity_error)

        full = run_result.text
        result_subtype = run_result.subtype
        num_turns = continuation_turns if uses_outer_gate else run_result.num_turns

        def _parse_tag(tag: str, cap: int, *, last: bool = False) -> str:
            """Pull a `TAG: ...` one-liner out of the agent output, sanitized:
            cut any trailing injected control text and cap the length.

            With ``last=True`` return the LAST occurrence rather than the first.
            An in-session gate session can emit the tag on several turns (the
            agent edits, the gate rejects the stop, it edits again…); the final
            occurrence is written after the kernel has converged, so it is the
            one that matches the code actually committed and benchmarked.
            """
            found = ""
            for ln in full.splitlines():
                s = ln.strip()
                if s.upper().startswith(tag):
                    val = s[len(tag) :].strip()
                    for marker in ("Stop hook feedback", "## ", "Make your change"):
                        idx = val.find(marker)
                        if idx != -1:
                            val = val[:idx]
                    val = val.strip()
                    if len(val) > cap:
                        # Truncate on a word boundary (not mid-word) + ellipsis.
                        cut = val[:cap].rsplit(" ", 1)[0].rstrip(" ,;:-")
                        val = (cut or val[:cap]) + "…"
                    found = val
                    if not last:
                        break
            return found

        # PLAN — one-sentence description of THIS iteration's FINAL modification.
        # Take the LAST occurrence: after any in-session edit/fix cycles, the
        # agent's closing PLAN describes the change that is now in the file (the
        # net change that gets committed + benchmarked), not an abandoned attempt.
        plan = _parse_tag("PLAN:", 160, last=True)

        submitted = any(line.strip().upper() == "SUBMIT_CANDIDATE" for line in full.splitlines())

        # Explain WHY this session ended, for per-iteration analysis:
        #   * gate allowed a safe candidate handoff -> candidate_submitted;
        #   * else the SDK ended the query — turn cap (subtype mentions
        #     max_turns), another SDK error, or the agent voluntarily stopped
        #     ("success"). A gate-enabled session that hits the turn cap never
        #     runs a Stop hook, so gate.end_reason stays "" and we land here.
        if submitted:
            end_reason = "candidate_submitted"
        elif gate is not None and gate.end_reason:
            end_reason = gate.end_reason
        elif run_result.end_reason and run_result.end_reason != "agent_stopped":
            end_reason = run_result.end_reason
        elif "max_turns" in result_subtype:
            end_reason = "turn_cap"
        elif result_subtype and result_subtype != "success":
            end_reason = f"sdk_{result_subtype}"
        else:
            end_reason = "agent_stopped"

        # One structured line per session (stdout is captured by the caller's
        # logs) so a run's end-reason distribution is analyzable without the LLM.
        edit_count = gate.edit_count if gate is not None else run_result.edit_count
        print(
            f"  [session-end] backend={backend.name} reason={end_reason} "
            f"edits={edit_count if edit_count else '-'} "
            f"turns={num_turns if num_turns is not None else '?'} "
            f"pass={gate.passed if gate is not None else '-'}",
            flush=True,
        )

        text = full or "no rationale provided"
        if gate is not None:
            # Surface the gate outcome so the outer loop's commit message / log
            # records whether the session self-converged.
            tag = f"[gate edits={gate.edit_count} pass={gate.passed} end={end_reason}"
            if num_turns is not None:
                tag += f" turns={num_turns}"
            if gate.last_mean_case_speedup is not None:
                tag += f" mean_case_speedup={gate.last_mean_case_speedup:.6f}x"
            if gate.last_wall_ms is not None:
                tag += f" raw_mean={gate.last_wall_ms:.4f}ms"
            tag += "]"
            text = f"{tag} {text}"

        # Hand back structured session info so the runner can feed the
        # ExperienceLedger with the gate's objective findings, and (via
        # ``summarize``) ask this exact session to record what it explored.
        if session_sink is not None:
            session_sink["plan"] = plan
            session_sink["session_id"] = run_result.session_id
            session_sink["summarize"] = _make_session_summarizer(
                backend=backend,
                spec=run_spec,
                session_id=run_result.session_id,
                usage=usage,
            )
            session_sink["end_reason"] = end_reason
            session_sink["turns"] = num_turns
            # The runner never sees the AgentRunResult, so this is the only
            # place a workspace the reaper could not clear can reach the code
            # that decides whether to run the canonical measurement.
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

        return text[:200]

    setattr(agent_fn, "backend_name", backend.name)
    setattr(agent_fn, "backend_model", backend_model)
    setattr(agent_fn, "requested_backend", requested_backend)
    return agent_fn


def _make_session_summarizer(
    *,
    backend,
    spec: AgentRunSpec,
    session_id: str,
    usage=None,
) -> Callable[[str], Awaitable[str]] | None:
    """Build an async callable that resumes ONE finished implementer session.

    The returned callable replays the session's full conversation and asks it a
    follow-up question, so the answer is grounded in everything the agent
    actually tried — including directions it abandoned, which exist in no other
    record. The resumed turn runs under a deliberately different policy than the
    session it continues:

      * ``hooks=None`` — the implementer session carries the in-session gate's Stop
        hook. Left attached, that hook would run correctness+bench and BLOCK the
        summarizing turn, pushing the agent back into editing the kernel.
      * read-only tools — the caller, not the model, persists the reply, so the
        session needs no write access (mirrors ``profile_analyst``).
      * ``read_only_resume`` — providers that guard the worktree may inspect any
        pre-existing Git-visible state, but must verify that the read-only turn
        leaves that state byte-for-byte unchanged.

    Returns ``None`` when the provider cannot resume, so the caller degrades to
    a lesson document carrying only the loop's machine-written outcome.
    """
    if not session_id or not getattr(backend.capabilities, "resumable", False):
        return None
    if not hasattr(backend, "resume"):
        return None

    from kernelforge.loop.lessons import SUMMARIZER_ROLE

    summary_spec = replace(
        spec,
        system_prompt=SUMMARIZER_ROLE,
        writable=False,
        hooks=None,
        allow_dirty_targets=True,
        allow_untracked=True,
        read_only_resume=True,
        protected_globs=["*"],
        reasoning_effort="high",
        # Preserve the implementer's progress log as a stable fallback record. The
        # summarizer turn has no reason to append its own activity to that list.
        progress_log=None,
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=False,
            shell=False,
            # Enough turns to check a path or a number it half-remembers, not
            # enough to start exploring the workspace.
            max_turns=4,
        ),
    )

    async def summarize(prompt: str) -> str:
        result = await backend.resume(summary_spec, session_id, prompt, usage=usage)
        return result.text or ""

    return summarize
