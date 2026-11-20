# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 3: author an env-gated fused kernel from a self-discovered recipe."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from kernelforge.agent_backends.base import (
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
)
from kernelforge.agent_backends.session_resume import is_api_failure

from .author_guard import AuthorSafetyError, AuthorWorkspaceGuard
from .author_hooks import target_file_hooks
from .emit import _FUSED_MODULE_MARKERS, _FUSED_MODULE_PREFIXES
from .llm_failure import (
    DEFAULT_BASE_DELAY_SEC,
    DEFAULT_MAX_DELAY_SEC,
    RETRYABLE_KINDS,
    classify_llm_error,
    env_setting,
    is_agent_safety_error,
    is_agent_timeout_error,
    retry_delay,
)
from .harness_contract import harness_contract, trace_kernels_block
from .validate import DEFAULT_TARGET_SPEEDUP
from kernelforge.config.edit_policy import FORGE_PROTECTED_PATHS

log = logging.getLogger("forge_fusion")

# Process-style author return codes.
AUTHOR_RC_OK = 0
AUTHOR_RC_FAILED = 1
AUTHOR_RC_SAFETY = 3
AUTHOR_RC_TIMEOUT = 124

# An author session costs minutes and gates a lane that costs hours, so a transport that
# never delivered an answer is worth one more call. Two attempts by default: past that the
# upstream is down rather than flaky, and the lane is better off reporting it.
DEFAULT_AUTHOR_ATTEMPTS = 2


def proven_fusion_fewshot() -> str:
    """Few-shot block of serving-validated decode fusions (worked examples)."""
    return """## Proven fusion examples (few-shot — these ALL passed real sglang serving e2e)
- ZAYA CCA QK post-processing (`ZAYA_FUSED_QK`): fold `_add_grouped_qk_means` +
  `_normalize_qk` (~15-20 tiny fp32 view/mean/add/mul/pow/sum/rsqrt ops) into ONE
  Triton kernel, one program per (token, k-head). +14.7% e2e alone.
- ZAYA ResidualScaling (`ZAYA_FUSED_RESIDUAL`): dual affine `(x+bias)*scale` on the
  hidden AND residual streams in ONE launch, bf16->fp32 in-kernel. QK+Residual
  together = +34.5% e2e.
- LFM2 (`LFM2_FUSED_RESIDUAL` / `LFM2_FUSED_SILU`): thread the per-layer residual
  adds into the next RMSNorm; merge w1|w3 SwiGLU into one GEMM + fused SiluAndMul.
  ~+16% e2e.
- Granite (`GRANITE_FUSED_RESIDUAL`): `scaled_add_rmsnorm` = `rmsnorm(x*scale + r)`,
  folding scalar-mul + residual-add + RMSNorm into ONE kernel; ~5e-9 vs eager.

## MANDATORY patterns from these serving-validated kernels
## (violating them passes microbench but CRASHES real serving — do NOT):
1. env-gated: with the flag UNSET the path is bit-for-bit the original eager code.
2. fp32 accumulation INSIDE the Triton kernel (cast bf16->fp32 in-kernel, not outside).
3. ONE Triton launch replacing the whole tiny-op chain (fewer launches = the win).
4. CUDA-GRAPH SAFE (CRITICAL): the kernel runs INSIDE the captured decode CUDA graph.
   Preallocate ALL outputs; use tl.constexpr for shapes; NO python-side allocation,
   NO `.item()`/`.cpu()`/torch host sync, NO data-dependent shapes in the decode hot
   path. A kernel that allocates or host-syncs per call passes a standalone microbench
   but SIGQUIT-crashes the sglang scheduler decode loop.
5. import the REAL eager op as the parity oracle; keep public signatures/imports intact.
6. ROCm-native Triton only; never reuse a CUDA-only framework fused op.

"""


def _arch_phrase(gpu_arch: str) -> str:
    """How to name the target GPU in a prompt."""
    arch = (gpu_arch or "").strip().lower()
    marketing = {"gfx950": "MI355X", "gfx942": "MI300X/MI325X"}.get(arch, "")
    if not arch:
        return "an AMD ROCm GPU"
    return f"AMD {marketing} ({arch})" if marketing else f"an AMD GPU ({arch})"


def _model_dir_block(model_path: str) -> str:
    """Name the model directory, because the alternative is that it gets searched for."""
    if not model_path:
        return ""
    return (
        f"- Model directory (config.json, tokenizer, weights): {model_path}\n"
        "  Read the model's own files from there rather than searching for them: "
        "`/` on this host includes multi-terabyte network mounts, and a single "
        "`find /` can outlast the whole authoring budget.\n"
    )


@dataclass(frozen=True)
class AuthorBrief:
    """What the author is told besides the recipe: framework, how it is validated, the bar, harness, GPU and model."""

    framework: str
    ab_hint: str
    target_speedup: float = DEFAULT_TARGET_SPEEDUP
    harness_path: str = ""
    gpu_arch: str = ""
    model_path: str = ""


_INTEGRATION_BLOCK = """## Existing operator integration (MANDATORY first path)
- Candidate kind: integration
- Existing ROCm operator: `{existing_operator}`
- Reproduce the eager boundary, then benchmark and wire the existing operator first.
- Do not author a replacement kernel unless the existing operator is incompatible or
  loses the exact-shape microbenchmark; record that evidence before falling back.
- The operator's numerics were never verified against THIS model's eager path. Before
  keeping it, record parity against the eager reference at the framework's own
  tolerance (reuse the rtol/atol the framework's tests use for this dtype; do not
  invent a looser one). A microbenchmark win alone is NOT sufficient to keep it.
- Report the measured max relative error alongside the speedup, so a fast but
  numerically worse operator is visible as such.

"""

_AUTHOR_REACH_RULES = """## Engineering discipline (MUST follow)
- The kernel MUST be REACHED by the decode path, not merely defined and exposed.
  Assigning it onto another module (`other_mod.my_fused_op = ...`) does nothing
  unless something already reads that name -- check that a reader exists before
  you rely on it. When the chain you are fusing lives in a file you may not edit,
  take over the call site from the file you may: rebind the method or attribute
  the chain already goes through, so the existing callers reach your kernel
  without changing any signature. Publishing a new name that nothing calls scores
  as a failed attempt, and it is the most common way a correct kernel is wasted.
- If you cannot reach the chain from the files you are allowed to touch, say so
  explicitly in your final message instead of leaving an unreachable kernel
  behind. That answer is useful; an inert one is not.
"""

_MULTI_INTEGRATION_BLOCK = """## Existing operator integrations (MANDATORY first paths)
Benchmark and wire these existing ROCm operators before authoring replacements:
{operators}
Do not author replacement kernels unless an operator is incompatible or loses its
exact-shape microbenchmark; record that evidence before falling back.
For each integrated operator, record parity against the eager reference at the
framework's own rtol/atol for this dtype and report the measured max relative error
next to the speedup. A microbenchmark win alone is NOT sufficient to keep it.

"""


def build_author_prompt(recipe: dict, brief: AuthorBrief) -> str:
    """Build the authoring prompt from a recipe dict (``Recipe.to_dict()``)."""
    framework, ab_hint, target_speedup = brief.framework, brief.ab_hint, brief.target_speedup
    harness_path, gpu_arch, model_path = brief.harness_path, brief.gpu_arch, brief.model_path
    shapes = recipe.get("shapes", {})
    hints = recipe.get("source_hints", [])
    env_flag = recipe.get("env_flag", "FUSED")
    harness_block = harness_contract(harness_path, env_flag) if harness_path else ""
    candidate_kind = str(recipe.get("candidate_kind") or "new_fusion")
    existing_operator = str(recipe.get("existing_operator") or "").strip()
    integration_block = ""
    if candidate_kind == "integration" and existing_operator:
        integration_block = _INTEGRATION_BLOCK.format(existing_operator=existing_operator)
    rocm_line = (
        "- TARGET IS ROCm (AMD GPU): author a ROCm-native Triton (or aiter) kernel. "
        "Do NOT reuse a framework CUDA-only fused op; verify it BUILDS and RUNS.\n"
        if recipe.get("rocm_native", True)
        else ""
    )
    return f"""You are optimizing the {framework} model file for a decode-path kernel fusion on
{_arch_phrase(gpu_arch)}, bf16 serving. Work autonomously; do not ask questions.

## Target
- Framework source file to edit: {recipe.get("source_file") or "(resolve it under the framework model dir)"}
{_model_dir_block(model_path)}- Fusion pattern: {recipe.get("pattern")}
- {recipe.get("description")}
{trace_kernels_block(recipe.get("trace_kernels"))}
## What to fuse (the recipe)
{recipe.get("fusion_math")}

## Representative decode shapes (from the model config + trace)
{shapes}

## How to localize it in the source
Grep the model file for these anchors and fuse the chain they mark:
{chr(10).join(f"  - {h}" for h in hints)}

{integration_block}{proven_fusion_fewshot()}{_AUTHOR_REACH_RULES}- The fusion MUST be env-gated by `{env_flag}`. With the flag UNSET the code path
  stays bit-for-bit the original eager path.
{rocm_line}- Cast to fp32 inside the fused kernel; one launch instead of the multi-op chain.
- NET launch count must DROP. Count the kernels the eager path launches per decode
  step and the kernels the fused path launches; the fused number must be strictly
  smaller. A scratch-fill, a separate cast or a contiguous copy added to feed your
  kernel can cancel the launches it saved while the chain alone still benchmarks
  faster. Report both counts alongside the speedup.
- CUDA-graph safe: no Python-side dynamic allocation or host sync in the decode
  hot path (preallocate outputs; use tl.constexpr for shapes).
- Keep all public function/class signatures and imports intact.
- Add a pure-torch reference and assert parity BEFORE trusting the kernel.
  {recipe.get("eager_reference_hint")}
- If Triton is unavailable, fall back to eager (never crash).
{harness_block}
## How to validate (the ONLY success signal)
Run this A/B (boots the model twice; eager vs `{env_flag}=1`), decode-step median:
    {ab_hint}
- SUCCESS = fused clearly faster than eager (target speedup >= {target_speedup:.2f}x).
- Also confirm correctness: greedy output with the flag on stays coherent vs off.
- Iterate: edit -> run A/B -> read speedup -> fix -> repeat until the target is met.

When done, stop.
Do not edit the A/B harness or hard-code any numbers.
"""


def build_multi_author_prompt(recipes: list[dict], brief: AuthorBrief) -> str:
    """Prompt to author SEVERAL confirmed fusions in one pass (each env-gated)."""
    if len(recipes) == 1:
        return build_author_prompt(recipes[0], brief)
    framework, ab_hint, target_speedup = brief.framework, brief.ab_hint, brief.target_speedup
    harness_path, gpu_arch, model_path = brief.harness_path, brief.gpu_arch, brief.model_path
    blocks = []
    all_flags = []
    for i, r in enumerate(recipes, 1):
        all_flags.append(r.get("env_flag", "FUSED"))
        blocks.append(
            f"### Fusion {i}: {r.get('pattern')} (env gate `{r.get('env_flag')}`)\n"
            f"{r.get('description')}\n"
            f"Math: {r.get('fusion_math')}\n"
            f"Source anchors: {', '.join(r.get('source_hints', []))}\n"
            f"Eager reference: {r.get('eager_reference_hint')}\n"
            f"Candidate kind: {r.get('candidate_kind', 'new_fusion')}\n"
            f"Existing operator: {r.get('existing_operator', '')}\n"
        )
    src = recipes[0].get("source_file") or "(resolve under the framework model dir)"
    shapes = recipes[0].get("shapes", {})
    harness_block = harness_contract(harness_path, " ".join(all_flags)) if harness_path else ""
    existing_operators = [
        str(recipe.get("existing_operator"))
        for recipe in recipes
        if recipe.get("candidate_kind") == "integration" and recipe.get("existing_operator")
    ]
    integration_block = ""
    if existing_operators:
        integration_block = _MULTI_INTEGRATION_BLOCK.format(
            operators=chr(10).join(f"- `{operator}`" for operator in existing_operators)
        )
    rocm_line = (
        "- TARGET IS ROCm (AMD GPU): author ROCm-native Triton/aiter kernels; do NOT "
        "reuse a framework CUDA-only fused op; verify each BUILDS and RUNS.\n"
        if any(r.get("rocm_native", True) for r in recipes)
        else ""
    )
    return f"""You are optimizing the {framework} model file `{src}` with SEVERAL decode-path
kernel fusions on {_arch_phrase(gpu_arch)}, bf16 serving. Work autonomously; no questions.

{_model_dir_block(model_path)}{trace_kernels_block(recipes[0].get("trace_kernels"))}
## Representative decode shapes (model config + trace)
{shapes}

## Fusions to author (each INDEPENDENTLY env-gated, default OFF = original eager path)
{chr(10).join(blocks)}

{integration_block}{proven_fusion_fewshot()}## Engineering discipline (MUST follow for every fusion)
- Each fusion is env-gated; with its flag UNSET the path is bit-for-bit eager.
{rocm_line}- Cast to fp32 inside the fused kernel; one launch instead of the op chain.
- CUDA-graph safe (preallocate outputs; tl.constexpr shapes; no host sync in decode).
- Keep public signatures/imports intact; import the REAL eager op for the parity ref.
- Add a parity self-check before trusting each kernel; fall back to eager if Triton is missing.
{harness_block}
## Validate (the ONLY success signal) — all flags ON together:
    {ab_hint}
- SUCCESS = combined fused clearly faster than eager (target speedup >= {target_speedup:.2f}x).
- Confirm greedy output stays coherent with all flags on. Iterate until the target is met.

When done, stop.
Do not edit the A/B harness or hard-code numbers.
"""


_AUTHOR_SYSTEM_PROMPT = """\
You are the authoring stage of KernelForge forge-fuse. Implement and validate
the requested source-level fusion autonomously. Keep system instructions separate
from the user task. Modify only the exact target files supplied by the caller;
never modify tests, benchmark oracles, git state, or unrelated source. Create the
target validation harness only when the user prompt requests it. Use the requested
working directory and finish with the result contract specified in the user prompt.
"""

_AUTHOR_NEW_MODULE_CLAUSE = """\
You may additionally create NEW fused-kernel helper modules that a target file
imports, but only directly inside: {directories}. Each one must be a .py file whose
name marks it as fused — it must contain {markers} or start with {prefixes} (e.g.
qwen3_fused_ops.py) — must not reuse the name of a file that already exists in that
directory, and must not be added to the git index. Anything else you create is
reverted and the whole attempt is rejected with it.
"""


def _quoted_name_list(values: tuple[str, ...]) -> str:
    """Render name fragments as an ``a, b or c`` list for the prompt."""
    quoted = [f"`{value}`" for value in values]
    if len(quoted) < 2:
        return "".join(quoted)
    return f"{', '.join(quoted[:-1])} or {quoted[-1]}"


def _author_system_prompt(new_module_dirs: list[str]) -> str:
    """State exactly the write scope the workspace guard is going to accept."""
    if not new_module_dirs:
        return _AUTHOR_SYSTEM_PROMPT
    clause = _AUTHOR_NEW_MODULE_CLAUSE.format(
        directories=", ".join(new_module_dirs),
        markers=_quoted_name_list(_FUSED_MODULE_MARKERS),
        prefixes=_quoted_name_list(_FUSED_MODULE_PREFIXES),
    )
    return f"{_AUTHOR_SYSTEM_PROMPT}{clause}"


def _write_registered_author_log(
    log_path: str,
    progress: list[str],
    *,
    text: str = "",
    error: str = "",
    created: tuple[str, ...] = (),
) -> None:
    """Persist streamed progress plus the final Agent result."""
    lines = list(progress)
    if created:
        lines.append(f"created new fusion module(s): {', '.join(created)}")
    if text.strip():
        lines.append(text.strip())
    if error.strip():
        lines.append(f"error: {error.strip()}")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(
        ("\n".join(lines).strip() + "\n") if lines else "",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class AuthorAgent:
    """The agent that authors: its backend (or a factory for one), model, device, turn cap, timeout, usage accumulator."""

    backend: Any
    gpu: str = "0"
    model: Optional[str] = None
    max_turns: int = 100
    timeout_s: int = 7200
    usage: Any = None


@dataclass(frozen=True)
class AuthorWorkspace:
    """Where an authoring session runs and logs, the files it may edit, and where it may add new modules."""

    workdir: str
    log_path: str
    target_files: Sequence[str] = ()
    new_module_dirs: Sequence[str] = ()


def _run_registered_author(prompt: str, workspace: AuthorWorkspace, agent: AuthorAgent) -> int:
    """Run authoring, retrying only a transport that never delivered an answer."""
    timeout_s = agent.timeout_s
    attempts = max(1, int(env_setting("FORGE_FUSION_AUTHOR_ATTEMPTS", DEFAULT_AUTHOR_ATTEMPTS, cast=int)))
    base_delay = float(env_setting("FORGE_FUSION_LLM_RETRY_BASE_SEC", DEFAULT_BASE_DELAY_SEC, cast=float))
    max_delay = float(env_setting("FORGE_FUSION_LLM_RETRY_MAX_SEC", DEFAULT_MAX_DELAY_SEC, cast=float))
    # llm_failure's 1800s default is sized for the short calls discover.py makes; an author attempt
    # is allowed timeout_s, which defaults to 7200. Defaulting to it here would make the gate below
    # read `elapsed + delay + 7200 >= 1800` on the first failure, so the retry could never start.
    # The default budget is what the configured attempts can legitimately cost; an operator who
    # sets the deadline still gets it enforced.
    default_deadline = attempts * (float(timeout_s) + max_delay)
    deadline = float(env_setting("FORGE_LLM_RETRY_DEADLINE_SEC", default_deadline, cast=float))
    started_at = time.monotonic()
    for attempt in range(1, attempts + 1):
        rc, retryable = _run_registered_author_once(prompt, workspace, agent)
        if not retryable or attempt >= attempts:
            return rc
        delay = retry_delay(attempt, base_sec=base_delay, max_sec=max_delay)
        elapsed = time.monotonic() - started_at
        # A retry that cannot plausibly finish must not be started: each attempt spends up to the
        # per-attempt timeout, so the clock is checked against what the next one would cost.
        if deadline > 0 and elapsed + delay + timeout_s >= deadline:
            log.warning(
                "author transport failed on attempt %d/%d with %.0fs of the %.0fs retry budget spent; not retrying",
                attempt,
                attempts,
                elapsed,
                deadline,
            )
            return rc
        log.warning(
            "author attempt %d/%d failed on the transport; retrying in %.0fs",
            attempt,
            attempts,
            delay,
        )
        time.sleep(delay)
    return rc


def _record_author_log(log_path: str, progress: list[str], **fields: Any) -> None:
    """Write the author log, best-effort.

    A log this run could not write is not a reason to lose the verdict it was
    recording, so the failure is reported and the caller carries on.
    """
    try:
        _write_registered_author_log(log_path, progress, **fields)
    except OSError:
        log.warning("could not write registered author log %s", log_path)


def _safety_detail(exc: AuthorSafetyError) -> str:
    """The guard's complaint, with the paths it named."""
    detail = str(exc)
    return f"{detail}; paths={', '.join(exc.paths[:20])}" if exc.paths else detail


def _author_run_error_verdict(run_error: Exception, *, backend: Any, timeout_s: int) -> tuple[int, bool]:
    """The exit code for an agent run that raised, and whether another attempt is worth spending."""
    # Checked ahead of the timeout markers: a provider safety stop is final even if its message happens to mention
    # a clock, and retrying one is exactly the anti-pattern the session-resume allowlist already refuses.
    if is_agent_safety_error(run_error):
        log.error(
            "%s author was stopped by a provider safety guard: %s: %s",
            backend.name,
            type(run_error).__name__,
            run_error,
        )
        return AUTHOR_RC_SAFETY, False
    if is_agent_timeout_error(run_error):
        log.warning(
            "%s author timed out after %ss: %s: %s",
            backend.name,
            timeout_s,
            type(run_error).__name__,
            run_error,
        )
        # The attempt just spent the whole per-attempt budget; another would spend it again.
        return AUTHOR_RC_TIMEOUT, False
    log.error("%s author failed: %s: %s", backend.name, type(run_error).__name__, run_error)
    return AUTHOR_RC_FAILED, classify_llm_error(run_error) in RETRYABLE_KINDS


def _author_result_verdict(result: Any, *, backend: Any) -> tuple[int, bool]:
    """The exit code for an agent run that returned, and whether another attempt is worth spending."""
    if result.end_reason == "agent_stopped":
        return AUTHOR_RC_OK, False
    log.warning(
        "%s author ended without success (end_reason=%s subtype=%s)",
        backend.name,
        result.end_reason,
        result.subtype or "none",
    )
    # The backends flatten a transport failure the SDK swallowed into this end_reason rather than an exception, so
    # session_resume owns which of them is worth another attempt; a turn cap is the task's own answer.
    return AUTHOR_RC_FAILED, is_api_failure(result)


def _run_author_session(
    prompt: str, workdir: str, agent: AuthorAgent, guard: AuthorWorkspaceGuard, progress: list[str]
) -> Any:
    """Run one agent session on the author's GPU, allowed to write exactly what the guard will accept."""
    session_timeout = max(1, int(agent.timeout_s))
    spec = AgentRunSpec(
        system_prompt=_author_system_prompt(guard.new_module_dirs),
        user_prompt=prompt,
        role="fusion author",
        cwd=workdir,
        model=agent.model,
        writable=True,
        timeout_sec=session_timeout,
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=True,
            shell=True,
            max_turns=max(1, int(agent.max_turns)),
        ),
        target_files=guard.target_files,
        allow_dirty_targets=True,
        allow_untracked=True,
        # The author phase runs in a worktree a long campaign has already left dirty in ways it never touches, so the
        # backend has to judge this turn against the pre-run snapshot rather than against a clean HEAD.
        allow_dirty_baseline=True,
        # Forge's measurement protections, and nothing beyond them; the outer provider-neutral transaction enforces
        # the exact target allowlist.
        protected_path_rules=FORGE_PROTECTED_PATHS,
        protected_globs=[],
        hooks=target_file_hooks(guard.target_files, workdir, allows_new_path=guard.permits_new_path),
        progress_log=progress,
    )
    previous_gpu = os.environ.get("HIP_VISIBLE_DEVICES")
    os.environ["HIP_VISIBLE_DEVICES"] = agent.gpu
    try:
        return asyncio.run(
            asyncio.wait_for(agent.backend.run(spec, usage=agent.usage), timeout=watchdog_timeout_sec(session_timeout))
        )
    finally:
        if previous_gpu is None:
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        else:
            os.environ["HIP_VISIBLE_DEVICES"] = previous_gpu


def _run_registered_author_once(prompt: str, workspace: AuthorWorkspace, agent: AuthorAgent) -> tuple[int, bool]:
    """One authoring run, with whether its failure was the transport's rather than the task's."""
    progress: list[str] = []
    requested_targets = list(dict.fromkeys(str(path) for path in workspace.target_files if str(path)))
    requested_module_dirs = list(dict.fromkeys(str(path) for path in workspace.new_module_dirs if str(path)))
    try:
        guard = AuthorWorkspaceGuard(workspace.workdir, requested_targets, requested_module_dirs)
    except AuthorSafetyError as exc:
        detail = _safety_detail(exc)
        heading = (
            "author workspace could not be inspected before run"
            if exc.transient
            else "author workspace safety rejected before run"
        )
        _record_author_log(workspace.log_path, progress, error=f"{heading}: {detail}")
        log.error("%s: %s", heading, detail)
        # A workspace the guard could not read is the box's problem, not the transport's.
        return (AUTHOR_RC_FAILED if exc.transient else AUTHOR_RC_SAFETY), False
    result = None
    run_error: BaseException | None = None
    try:
        result = _run_author_session(prompt, workspace.workdir, agent, guard, progress)
    except BaseException as exc:  # isolation boundary: one author attempt; restoration must always run
        log.warning("author agent run failed for %s", workspace.workdir, exc_info=True)
        run_error = exc
    text = str(getattr(result, "text", "") or "")

    def _with_run_error(reason: str) -> str:
        """Keep the session's own failure beside a verdict about the workspace."""
        if run_error is None:
            return reason
        return f"{reason}; the agent run also failed: {type(run_error).__name__}: {run_error}"

    def _rejected(reason: str, rc: int) -> tuple[int, bool]:
        """Record a workspace verdict and end the attempt; the retry loop never spends another one on it."""
        reason = _with_run_error(reason)
        _record_author_log(workspace.log_path, progress, text=text, error=reason)
        log.error("%s", reason)
        if run_error is not None and not isinstance(run_error, Exception):
            raise run_error
        return rc, False

    try:
        enforcement = guard.enforce()
    except AuthorSafetyError as exc:
        heading = (
            "author workspace restoration could not complete"
            if exc.transient
            else "author workspace safety restoration failed"
        )
        return _rejected(f"{heading}: {_safety_detail(exc)}", AUTHOR_RC_FAILED if exc.transient else AUTHOR_RC_SAFETY)
    except OSError as exc:
        detail = f"{type(exc).__name__}: internal workspace guard failure"
        # The agent-facing log stays content-free (a guard defect is not something the author can act on), but the
        # operator needs the traceback to fix it.
        log.exception("author workspace safety restoration failed: %s", _with_run_error(detail))
        return _rejected(f"author workspace safety rejection: {detail}", AUTHOR_RC_SAFETY)

    if enforcement.violations:
        violations = enforcement.violations
        detail = ", ".join(violations[:20])
        if len(violations) > 20:
            detail += f", ... ({len(violations)} paths)"
        # Deterministic: the same guard, worktree and prompt reject the next attempt the same way, and the loop is
        # told so rather than spending one on it.
        return _rejected(f"author workspace rejected and restored non-target paths: {detail}", AUTHOR_RC_SAFETY)

    if enforcement.created:
        log.info(
            "author created new fusion module(s): %s",
            ", ".join(enforcement.created),
        )

    if run_error is not None:
        if not isinstance(run_error, Exception):
            raise run_error
        _record_author_log(
            workspace.log_path,
            progress,
            error=f"{type(run_error).__name__}: {run_error}",
            created=enforcement.created,
        )
        return _author_run_error_verdict(run_error, backend=agent.backend, timeout_s=agent.timeout_s)

    assert result is not None
    _record_author_log(workspace.log_path, progress, text=text, created=enforcement.created)
    return _author_result_verdict(result, backend=agent.backend)


def run_author(prompt: str, workspace: AuthorWorkspace, agent: AuthorAgent) -> int:
    """Drive the selected Agent backend and return the legacy process-style code."""
    backend = agent.backend
    if callable(backend) and not hasattr(backend, "run"):
        backend = backend()
    selected_model = (
        str(agent.model or "").strip() or str(getattr(getattr(backend, "runtime", None), "model", "")).strip()
    )
    log.info(
        "running %s Agent author (model=%s max_turns=%d workdir=%s)",
        backend.name,
        selected_model,
        agent.max_turns,
        workspace.workdir,
    )
    return _run_registered_author(prompt, workspace, dataclasses.replace(agent, backend=backend, model=selected_model))
