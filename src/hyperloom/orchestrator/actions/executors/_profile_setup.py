# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TraceLens profile setup shared by fixed-sequence and native AgentX runs.

Runtime patching and shape discovery are shared with native AgentX.
Fixed-sequence capture planning and shell arguments stay on the synthetic path;
Magpie owns the native capture configuration.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from functools import cache
from typing import Any, Mapping, NamedTuple

from hyperloom.common.env import env_bool, env_flag, is_truthy
from hyperloom.common.workload_defaults import DEFAULT_CONC, DEFAULT_OSL
from hyperloom.inference_optimizer.model_config_utils import model_is_gemma2
from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path
from ._benchmark_interpreter import _resolve_probe_python

log = logging.getLogger(__name__)

# Profile-phase capture defaults. Trace size scales with captured decode steps;
# an oversized capture serializes too slowly and kills the engine. 128 steps is
# serialization-safe on a large TP=8 MoE. Tunable via
# HYPERLOOM_PROFILE_MAX_STEPS_CAP.
_DEFAULT_PROFILE_MAX_STEPS = 128
# Default profile OSL ceiling when --profile-osl / PROFILE_OSL is unset: the
# profile reuses min(served OSL, this) so its trace stays light.
_PROFILE_DEFAULT_OSL = 1024
# Speculative-decoding flags: SGLang ``--speculative-*``, vLLM ``--speculative-config`` (JSON or dotted keys),
# and the count each one bounds accepted tokens per step by, as (pattern, added to the count), in order of
# preference: the first pattern present decides. SGLang's draft depth beats its draft-token count, because a
# tree launch (``--speculative-eagle-topk`` > 1) verifies many more tokens than one step can accept.
_SPECULATIVE_FLAG = re.compile(r"--speculative[-_.]|--num[-_]speculative[-_]tokens")
_SPECULATIVE_BOUNDS = (
    (re.compile(r"--speculative[-_]num[-_]steps[=\s]+(\d+)"), 1),
    (re.compile(r"num[-_]speculative[-_]tokens[\"']?\s*[:=\s]\s*(\d+)"), 1),
    (re.compile(r"--speculative[-_]num[-_]draft[-_]tokens[=\s]+(\d+)"), 0),
)
_SPECULATIVE_FALLBACK_TOKENS_PER_STEP = 8
# vLLM ``--profiler-config`` keys that bound the capture window, and the delay an operator-set window defaults to.
_DELAY_ITERATIONS = "delay_iterations"
_MAX_ITERATIONS = "max_iterations"
_DEFAULT_PROFILE_DELAY_ITERS = 8
_VLLM_DELAY_FLAG = re.compile(r"(--profiler-config\.delay_iterations[=\s]+)\d+")

# ``HYPERLOOM_PROFILE_DEGRADED_REASON`` value meaning the TraceLens runtime patch
# was attempted and did not apply. Distinct from "never attempted": patching can
# be disabled for an image that already ships the patch.
_TRACELENS_PATCH_UNAVAILABLE = "tracelens_runtime_patch_unavailable"

# Installed ATOM (docker) TraceLens knobs. One subprocess; cached for the process.
_ATOM_CAPS_PROBE = (
    "import argparse\n"
    "from atom.model_engine.arg_utils import EngineArgs\n"
    "from atom.utils import envs\n"
    "p = argparse.ArgumentParser(); EngineArgs.add_cli_args(p)\n"
    "h = p.format_help()\n"
    'print(int("--mark-trace" in h))\n'
    'print(int(hasattr(envs, "ATOM_ENABLE_DETAILED_ANNOTATION")))\n'
    'print(int(hasattr(envs, "ATOM_PROFILER_MORE")))\n'
)


class _AtomTracelensCaps(NamedTuple):
    """Installed-atom support for TraceLens profile knobs."""

    mark_trace: bool
    detailed_annotation: bool
    profiler_more: bool


_ATOM_CAPS_NONE = _AtomTracelensCaps(False, False, False)


@cache
def _atom_tracelens_caps() -> _AtomTracelensCaps:
    """Probe the installed atom for ``--mark-trace`` and annotation envs.

    Fail-soft: import / help / parse errors return all-false so an older ATOM
    argparse never sees ``--mark-trace``. Cached after the first call.
    """
    try:
        proc = subprocess.run(
            [_resolve_probe_python("atom"), "-c", _ATOM_CAPS_PROBE],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning(
            "atom TraceLens caps probe failed (%s); omitting --mark-trace / annotation envs",
            exc,
        )
        return _ATOM_CAPS_NONE
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if proc.returncode != 0 or len(lines) < 3 or any(ln not in {"0", "1"} for ln in lines[:3]):
        log.warning(
            "atom TraceLens caps probe unavailable (exit=%s); omitting --mark-trace / annotation envs",
            proc.returncode,
        )
        return _ATOM_CAPS_NONE
    return _AtomTracelensCaps(
        mark_trace=lines[0] == "1",
        detailed_annotation=lines[1] == "1",
        profiler_more=lines[2] == "1",
    )


def _tracelens_patch_enabled() -> bool:
    """Read the ``HYPERLOOM_ENABLE_PATCH`` kill switch (default on).

    Set ``HYPERLOOM_ENABLE_PATCH=0`` to disable runtime patching of vLLM /
    SGLang. Default on because the patches are backward-compatible.

    With the switch off, no *server flag* that only a patched build accepts is
    injected (vLLM ``--profiler-config.detailed_trace_annotation``, SGLang
    ``--enable-shape-discovery-for-cuda-graph-profile``), because an unpatched
    argparse rejects them. The SGLang ``PROFILE_EXTRA_BODY`` annotations
    (``shape_discovery`` / ``detailed_annotations``) are a different case and are
    **kept**: they ride the ``/start_profile`` API, which an unpatched server
    accepts, and the switch is also how a pre-patched image opts out of runtime
    patching while still supporting them. Only a patch that was *attempted and
    failed* clears them, which is why that gate reads
    ``HYPERLOOM_PROFILE_DEGRADED_REASON`` (set solely on the attempted-and-failed
    path) rather than ``tracelens_patch_ok``.

    Returns:
        True when runtime patching is enabled (default), else False.
    """
    return os.environ.get("HYPERLOOM_ENABLE_PATCH", "1").strip() != "0"


_PROFILER_FLAG_RE = re.compile(r"--profiler-config\.(\w+)[=\s]+(\S+)")


def _profiler_flag_value(server_args: str, name: str) -> str | None:
    """The value vLLM would resolve for ``--profiler-config.<name>``, or None.

    Returns the LAST occurrence, because repeated flags are how this layer overrides
    earlier ones and vLLM's argparse resolves them last-wins. Accepts both
    ``--flag value`` and ``--flag=value``. Regex rather than tokenization: this runs
    after JSON-valued flags have been compacted, and one unbalanced quote must not
    take the whole guard down with it.
    """
    value: str | None = None
    for match in _PROFILER_FLAG_RE.finditer(server_args or ""):
        if match.group(1) == name:
            value = match.group(2)
    return value


def _profiler_bound_holds(name: str, value: str | None, *, cap: int) -> bool:
    """Whether the value already present keeps this flag's guarantee.

    Name-only matching is not enough for the two flags that decide whether the
    capture is bounded at all: vLLM reads ``max_iterations`` of 0 (or absent) as "no
    limit", and a frontend profiler left on tracks no iterations and captures the
    entire ``start_profile``..``stop_profile`` range. A guard that accepted those
    would report success while the run stayed unbounded, which is worse than not
    guarding -- the warning would send the next investigation the wrong way. The
    summary table is the same kind of flag: only an explicit false keeps vLLM from
    building it after the trace is written.

    Every other flag only decides what the trace contains or where it lands, so its
    presence is the whole contract.
    """
    if value is None:
        return False
    if name == _MAX_ITERATIONS:
        try:
            iterations = int(value)
        except ValueError:
            return False
        return 0 < iterations <= cap
    if name == "ignore_frontend":
        return is_truthy(value)
    if name == "torch_profiler_dump_cuda_time_total":
        return not is_truthy(value)
    return True


def _profile_steps_cap(policy_env: Mapping[str, Any]) -> int:
    raw = str(policy_env.get("HYPERLOOM_PROFILE_MAX_STEPS_CAP") or "").strip()
    try:
        cap = int(raw or _DEFAULT_PROFILE_MAX_STEPS)
    except ValueError:
        cap = _DEFAULT_PROFILE_MAX_STEPS
    return cap if cap > 0 else _DEFAULT_PROFILE_MAX_STEPS


def _explicit_capture_window(policy_env: Mapping[str, Any]) -> int:
    """The operator's ``HYPERLOOM_PROFILE_MAX_ITERS``, or 0 when the window is computed."""
    override = str(policy_env.get("HYPERLOOM_PROFILE_MAX_ITERS") or "").strip()
    return int(override) if override.isdigit() else 0


def _profile_capture_window(
    policy_env: Mapping[str, Any], *, cap: int, delay_iters: int = 0, steady_floor: int | None = None
) -> tuple[int, int]:
    """Resolve the synthetic workload's capture steps and delay."""
    max_iters = _explicit_capture_window(policy_env)
    if max_iters <= 0:
        return delay_iters, cap
    try:
        delay_iters = max(0, int(policy_env.get("HYPERLOOM_PROFILE_DELAY_ITERS") or _DEFAULT_PROFILE_DELAY_ITERS))
    except ValueError:
        delay_iters = _DEFAULT_PROFILE_DELAY_ITERS
    if steady_floor is not None and max_iters < steady_floor:
        log.warning("HYPERLOOM_PROFILE_MAX_ITERS=%d is below the steady-state floor of %d", max_iters, steady_floor)
    elif max_iters > cap:
        log.warning("HYPERLOOM_PROFILE_MAX_ITERS=%d exceeds the serialization-safe cap of %d", max_iters, cap)
    return delay_iters, max_iters


class ProfileInstrumentation(NamedTuple):
    """What one profile setup wrote, so later steps can restore it."""

    num_prompts: int
    atom_caps: _AtomTracelensCaps | None
    vllm_flags: list[tuple[str, str]]
    vllm_cap: int


def prepare_profile(bench: dict[str, Any], envs: dict[str, Any]) -> ProfileInstrumentation:
    """Prepare runtime instrumentation and the fixed-sequence capture workload."""
    plan = _capture_plan(envs)
    envs["OSL"] = plan.osl
    return _apply_framework_profile(
        bench,
        envs,
        delay_iters=plan.delay_iters,
        max_iters=plan.max_iters,
        num_prompts=plan.num_prompts,
    )


def apply_atom_profile_caps(envs: dict[str, Any], caps: _AtomTracelensCaps | None) -> None:
    """Write the ATOM TraceLens knobs the installed package accepts."""
    if caps is None:
        return
    if caps.detailed_annotation:
        envs["ATOM_ENABLE_DETAILED_ANNOTATION"] = "1"
    if caps.profiler_more:
        envs["ATOM_PROFILER_MORE"] = "1"
    if caps.mark_trace:
        extra = str(envs.get("EXTRA_ATOM_ARGS", "")).strip()
        if "--mark-trace" not in extra:
            envs["EXTRA_ATOM_ARGS"] = f"{extra} --mark-trace".strip()


class CapturePlan(NamedTuple):
    """Fixed-sequence engine window and the workload that can reach it."""

    delay_iters: int
    max_iters: int
    osl: int
    num_prompts: int


def _capture_plan(envs: dict[str, Any]) -> CapturePlan:
    """Return delay steps, captured steps, profile output length, and prompt count."""
    try:
        ratio = float(envs.get("RANDOM_RANGE_RATIO", 1.0))
    except (TypeError, ValueError):
        ratio = 1.0
    osl_val = int(envs.get("OSL") or DEFAULT_OSL)
    conc_val = int(envs.get("CONC") or DEFAULT_CONC)
    safe_conc = max(conc_val, 1)
    cap = _profile_steps_cap(os.environ)
    profile_osl_raw = os.environ.get("PROFILE_OSL", "").strip()
    profile_osl_explicit = profile_osl_raw.isdigit() and int(profile_osl_raw) > 0
    if profile_osl_explicit:
        osl_val = int(profile_osl_raw)
    else:
        osl_val = min(osl_val, _PROFILE_DEFAULT_OSL)
    osl_val = _fit_profile_osl(osl_val, cap=cap, ratio=ratio, safe_conc=safe_conc, explicit=profile_osl_explicit)
    safe_osl = max(osl_val, 1)
    steady_floor = math.ceil(safe_osl * (1.0 + ratio) / (2.0 * safe_conc))
    delay_iters, max_iters = _profile_capture_window(
        os.environ,
        cap=cap,
        delay_iters=max(0, int(osl_val * (ratio + 1) * 3 - cap / 2)),
        steady_floor=steady_floor,
    )
    return CapturePlan(
        delay_iters, max_iters, osl_val, _prompts_for_steps(delay_iters + max_iters, safe_conc, safe_osl)
    )


def _prompts_for_steps(steps: int, conc: int, osl: int) -> int:
    """Prompts whose one-token-per-step decode outlasts ``steps``, with the same margin again behind the window."""
    steps_to_prompts = max(1, (steps * conc + osl - 1) // osl)
    return max(conc, steps_to_prompts * 2)


def speculative_tokens_per_step(server_args: str) -> int:
    """The most tokens one decode forward step can emit under ``server_args``: 1 without speculative decoding.

    SGLang accepts at most ``--speculative-num-steps`` + 1 tokens per step, whatever the tree width; vLLM
    accepts at most ``num_speculative_tokens`` + 1. Only without either does ``--speculative-num-draft-tokens``
    bound it (DSpark, whose verify window is gamma + 1). A launch that enables speculation without stating a
    count gets a deliberately high fallback, because overestimating only moves the window earlier while
    underestimating loses it.
    """
    if not _SPECULATIVE_FLAG.search(server_args):
        return 1
    for pattern, extra in _SPECULATIVE_BOUNDS:
        values = pattern.findall(server_args)
        if values:
            return max(1, int(values[-1]) + extra)
    return _SPECULATIVE_FALLBACK_TOKENS_PER_STEP


def fit_capture_to_speculative(envs: dict[str, Any], framework_env: str) -> None:
    """Move a computed capture window to where a speculative-decoding server still reaches it.

    The window is counted in engine forward steps and was sized as if each step emits one token. With
    speculative decoding a step emits up to :func:`speculative_tokens_per_step` tokens, so the run ends in a
    fraction of those steps and a start step sized for one token per step is never reached: SGLang answers
    ``/start_profile`` and writes no trace. The delay is divided by that upper bound, and the prompt count is
    raised so the run still outlasts the window even when every draft token is accepted. Runs after every
    server-args merge, because the speculative flags usually arrive with the reference recipe or the candidate.
    An operator-set window (``HYPERLOOM_PROFILE_MAX_ITERS``) is left as written.
    """
    server_args = str(envs.get(framework_env, ""))
    tokens = speculative_tokens_per_step(server_args)
    if tokens == 1 or _explicit_capture_window(os.environ) > 0:
        return
    body = _profile_extra_body(envs)
    vllm_delay = _profiler_flag_value(server_args, _DELAY_ITERATIONS)
    if isinstance(body.get("start_step"), int):
        delay, steps = body["start_step"], int(body.get("num_steps") or 0)
        body["start_step"] = delay // tokens
        envs["PROFILE_EXTRA_BODY"] = json.dumps(body)
    elif vllm_delay is not None and vllm_delay.isdigit():
        delay, steps = int(vllm_delay), int(_profiler_flag_value(server_args, _MAX_ITERATIONS) or 0)
        envs[framework_env] = _VLLM_DELAY_FLAG.sub(rf"\g<1>{delay // tokens}", server_args)
    else:
        return
    conc, osl = max(int(envs.get("CONC") or DEFAULT_CONC), 1), max(int(envs.get("OSL") or DEFAULT_OSL), 1)
    needed = _prompts_for_steps((delay // tokens + steps) * tokens, conc, osl)
    envs["NUM_PROMPTS"] = max(int(envs.get("NUM_PROMPTS") or 0), needed)
    log.info(
        "speculative decoding emits up to %d tokens per step: profile start step %d -> %d",
        tokens,
        delay,
        delay // tokens,
    )


def _fit_profile_osl(osl: int, *, cap: int, ratio: float, safe_conc: int, explicit: bool) -> int:
    """Keep an explicit profile length, or shorten the automatic one until it fits."""
    steady_floor = math.ceil(max(osl, 1) * (1.0 + ratio) / (2.0 * safe_conc))
    if steady_floor <= cap:
        return osl
    if explicit:
        log.warning(
            "PROFILE_OSL=%d needs %d captured steps to reach steady "
            "state, above the profile cap of %d; the trace may lack a "
            "steady-state window (trace_split_no_steady_state). Lower "
            "--profile-osl or raise HYPERLOOM_PROFILE_MAX_STEPS_CAP.",
            osl,
            steady_floor,
            cap,
        )
        return osl
    fitted_osl = max(1, int(cap * 2 * safe_conc / (1.0 + ratio)))
    log.warning(
        "profile OSL %d would need %d captured steps to reach "
        "steady state (> cap %d); lowering profile OSL to %d so the "
        "capture stays serializable. Baseline/optimize unaffected.",
        osl,
        steady_floor,
        cap,
        fitted_osl,
    )
    return fitted_osl


def _apply_framework_profile(
    bench: dict[str, Any],
    envs: dict[str, Any],
    *,
    delay_iters: int,
    max_iters: int,
    num_prompts: int,
) -> ProfileInstrumentation:
    framework = str(bench.get("framework") or "").lower()
    patch_ok, sitecustomize = patch_for_profile(envs, framework)
    if "atom" in framework:
        caps = _atom_tracelens_caps()
        conc = int(envs.get("CONC") or DEFAULT_CONC)
        return ProfileInstrumentation(conc, caps, [], 0)
    if "vllm" in framework:
        flags = _apply_vllm_profile(envs, delay_iters=delay_iters, max_iters=max_iters, patch_ok=patch_ok)
        return ProfileInstrumentation(num_prompts, None, flags, max_iters)
    _apply_sglang_profile(
        bench,
        envs,
        capture_window=(delay_iters, max_iters),
        sitecustomize=sitecustomize,
        patch_ok=patch_ok,
    )
    return ProfileInstrumentation(num_prompts, None, [], 0)


def write_tracelens_patch_status(envs: dict[str, Any], *, attempted: bool, patched: bool, fw: str) -> None:
    """Write ``HYPERLOOM_TRACELENS_PATCH_STATUS`` and the degraded reason from a single owner."""
    if envs.get("HYPERLOOM_PROFILE_DEGRADED_REASON") == _TRACELENS_PATCH_UNAVAILABLE:
        envs.pop("HYPERLOOM_PROFILE_DEGRADED_REASON")
    if not attempted:
        envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] = "not_attempted"
    elif patched:
        envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] = "ok"
    else:
        envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] = "unavailable"
        envs["HYPERLOOM_PROFILE_DEGRADED_REASON"] = _TRACELENS_PATCH_UNAVAILABLE
        log.warning(
            "TraceLens runtime patch unavailable for framework=%s; "
            "profile will omit annotation-only flags and roofline "
            "analysis may be degraded.",
            fw or "<unset>",
        )


def patch_for_profile(envs: dict[str, Any], framework: str) -> tuple[bool, bool]:
    """Patch the installed server the way the fixed-sequence profile does."""
    from hyperloom.orchestrator.actions.executors._server_patcher import (
        ensure_sglang_patched_for_tracelens,
        ensure_vllm_patched_for_tracelens,
        resolve_sglang_shape_mode,
    )

    is_atom = "atom" in framework
    is_sglang = "sglang" in framework
    sitecustomize = is_sglang and resolve_sglang_shape_mode() == "sitecustomize"
    write_tracelens_patch_status(envs, attempted=False, patched=False, fw=framework)
    if is_atom or not _tracelens_patch_enabled():
        return False, sitecustomize
    if "vllm" in framework:
        patch_ok = ensure_vllm_patched_for_tracelens()
    else:
        patch_ok = ensure_sglang_patched_for_tracelens()
    if sitecustomize:
        envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] = "ok" if patch_ok else "unavailable"
        if not patch_ok:
            log.warning(
                "SGLang graph-capture patch (sglang_gc_patch) not applied; profiling "
                "continues on the unpatched server, where per-batch-size CUDA-graph "
                "capture can IndexError on multi-variant models (DSA dense+sparse). "
                "It needs TRACELENS_ROOT with an exact "
                "examples/custom_workflows/inference_analysis/sglang_gc_patch/sglang_<X_Y_Z> dir."
            )
        return False, sitecustomize
    write_tracelens_patch_status(envs, attempted=True, patched=patch_ok, fw=framework)
    return patch_ok, sitecustomize


#: The ``--profiler-config`` fields only TraceLens's vLLM patch adds (``_server_patcher``): a vLLM that was not patched
#: refuses them, so the server they are passed to must run a patched vLLM.
VLLM_PATCH_PROFILER_FIELDS: tuple[str, ...] = ("capture_torch_profiler", "detailed_trace_annotation")


def _apply_vllm_profile(
    envs: dict[str, Any],
    *,
    delay_iters: int,
    max_iters: int,
    patch_ok: bool,
) -> list[tuple[str, str]]:
    existing = str(envs.get("EXTRA_VLLM_ARGS", ""))
    flags = [
        (_DELAY_ITERATIONS, f"--profiler-config.delay_iterations {delay_iters}"),
        (_MAX_ITERATIONS, f"--profiler-config.max_iterations {max_iters}"),
    ]
    # Magpie's launcher puts ``profiler=torch`` before EXTRA_VLLM_ARGS. The argv
    # probe only sees EXTRA_VLLM_ARGS, and vLLM rejects the bounds without it.
    # A dir here would override Magpie's ``<workspace>/torch_trace`` by last-wins.
    if _profiler_flag_value(existing, "profiler") is None:
        flags.append(("profiler", "--profiler-config.profiler torch"))
    if patch_ok:
        flags.extend((field, f"--profiler-config.{field} True") for field in VLLM_PATCH_PROFILER_FIELDS)
    flags.append(("ignore_frontend", "--profiler-config.ignore_frontend True"))
    flags.append(("torch_profiler_dump_cuda_time_total", "--profiler-config.torch_profiler_dump_cuda_time_total False"))
    # Repeated flags last-win, so the computed cap replaces a YAML pin.
    # ``ignore_frontend`` is already in the shipped YAML; vLLM warns on the duplicate.
    profiler_args = " ".join(
        flag for sentinel, flag in flags if sentinel != "ignore_frontend" or "ignore_frontend" not in existing
    )
    if _DELAY_ITERATIONS not in existing:
        envs["EXTRA_VLLM_ARGS"] = f"{existing} {profiler_args}".strip()
    return flags


def annotations_supported(envs: Mapping[str, Any]) -> bool:
    """Whether the server this round launches will accept TraceLens step annotations.

    False only after a patch was attempted and failed: ``patch_for_profile`` writes the reason on
    exactly that path, and an image that ships the patch (``HYPERLOOM_ENABLE_PATCH=0``) keeps them.
    Both profile paths ask this one question -- the fixed-sequence request body because an unpatched
    server rejects the field, and the native path because Magpie probes the server whenever
    ``detailed_annotations`` is true and raises, rather than degrading, when the fields are absent.
    """
    return envs.get("HYPERLOOM_PROFILE_DEGRADED_REASON") != _TRACELENS_PATCH_UNAVAILABLE


def _apply_sglang_profile(
    bench: dict[str, Any],
    envs: dict[str, Any],
    *,
    capture_window: tuple[int, int],
    sitecustomize: bool,
    patch_ok: bool,
) -> None:
    delay_iters, max_iters = capture_window
    extra_body = _profile_extra_body(envs)
    extra_body["start_step"] = delay_iters
    extra_body["num_steps"] = max_iters
    shape_discovery = shape_discovery_enabled(bench)
    if sitecustomize:
        apply_sglang_shape_tool(envs, shape_discovery=shape_discovery)
        # Unpatched SGLang rejects shape_discovery in the profiling request.
        extra_body.pop("shape_discovery", None)
        extra_body.setdefault("detailed_annotations", True)
        envs["PROFILE_EXTRA_BODY"] = json.dumps(extra_body)
        return
    degraded = not annotations_supported(envs)
    if degraded:
        shape_discovery = False
    extra_body["shape_discovery"] = shape_discovery
    if degraded:
        extra_body["detailed_annotations"] = False
    else:
        extra_body.setdefault("detailed_annotations", True)
    envs["PROFILE_EXTRA_BODY"] = json.dumps(extra_body)
    if patch_ok and shape_discovery:
        existing = str(envs.get("EXTRA_SGLANG_ARGS", ""))
        if "shape-discovery-for-cuda-graph-profile" not in existing:
            envs["EXTRA_SGLANG_ARGS"] = f"{existing} --enable-shape-discovery-for-cuda-graph-profile".strip()


def apply_sglang_shape_tool(envs: dict[str, Any], *, shape_discovery: bool) -> None:
    """Configure the server environment for kernel_shape_tool discovery."""
    from hyperloom.orchestrator.actions.executors._server_patcher import kernel_shape_tool_dir

    tool_dir = kernel_shape_tool_dir()
    if shape_discovery and tool_dir is not None:
        prepend_launch_path(envs, "PYTHONPATH", str(tool_dir))
        envs["TRACELENS_SHAPE_DISCOVERY"] = "1"
    else:
        envs["TRACELENS_SHAPE_DISCOVERY"] = "0"
        if shape_discovery and tool_dir is None:
            log.warning(
                "SGLang shape mode=sitecustomize but kernel_shape_tool "
                "not found under TRACELENS_ROOT; shapes will be absent "
                "(set TRACELENS_ROOT to an NFS path visible to the server).",
            )


def shape_discovery_enabled(bench: dict[str, Any]) -> bool:
    enabled = env_flag("HYPERLOOM_PROFILE_SHAPE_DISCOVERY", default=True)
    forced = env_bool("HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE")
    model = str(bench.get("model") or "")
    if enabled and not forced and model_is_gemma2(model):
        enabled = False
        log.info(
            "Gemma2 roofline: disabling shape-discovery to avoid "
            "CUDA-graph capture crash (hipErrorStreamCapture"
            "Unsupported); CUDA graph + profiling kept. Set "
            "HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE=1 to override.",
        )
    return enabled


def _profile_extra_body(envs: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = json.loads(str(envs.get("PROFILE_EXTRA_BODY", "{}")))
    except (ValueError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}
