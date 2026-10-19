# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a profile run captures: the decode window, its caps, and the per-framework flags.

A profile measures a shape rather than a throughput, so the captured decode
window, the request count needed to reach it and the profiler knobs each
framework reads are settled here. :mod:`~._workload_envs` drives this module
once per materialization and carries the result in a :class:`_ProfileConfig`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from functools import cache
from typing import Any, Mapping, NamedTuple, Sequence

from hyperloom.common.env import env_bool, env_flag, is_truthy
from hyperloom.inference_optimizer.model_config_utils import (
    _load_model_config_dict,
    _model_is_gemma2,
)
from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path
from ._benchmark_interpreter import _resolve_probe_python

log = logging.getLogger(__name__)

# Profile-phase capture defaults. Trace size scales with captured decode steps;
# an oversized capture serializes too slowly and kills the engine. 128 steps is
# serialization-safe on a large TP=8 MoE. Tunable via
# HYPERLOOM_PROFILE_MAX_STEPS_CAP.
_DEFAULT_PROFILE_MAX_STEPS = 128
# AgentX counterpart of the cap above. The 128 is calibrated in decode steps
# against the synthetic 1024/1024 shape; an agentic step carries a measured ISL
# p50 of 56k-96k tokens, so the same step count buffers orders of magnitude more
# profiler events in HOST RAM. Measured on DeepSeek-V4-Pro: eight vLLM workers at
# 113-127 GB each, Ray reported 1012/1024 GB and killed the capture three times.
# The AgentX client bounds the window by wall clock anyway (~20s of steady
# state), so the extra steps buy nothing. HYPERLOOM_PROFILE_MAX_ITERS overrides.
# Default profile OSL ceiling when --profile-osl / PROFILE_OSL is unset: the
# profile reuses min(served OSL, this) so its trace stays light.
_PROFILE_DEFAULT_OSL = 1024
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
    if name == "max_iterations":
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


def _profile_capture_window(
    policy_env: Mapping[str, Any], *, cap: int, delay_iters: int = 0, steady_floor: int | None = None
) -> tuple[int, int]:
    """Resolve the synthetic workload's capture steps and delay."""
    override = str(policy_env.get("HYPERLOOM_PROFILE_MAX_ITERS") or "").strip()
    if not override.isdigit() or int(override) <= 0:
        return delay_iters, cap
    max_iters = int(override)
    try:
        delay_iters = max(0, int(policy_env.get("HYPERLOOM_PROFILE_DELAY_ITERS") or 8))
    except ValueError:
        delay_iters = 8
    if steady_floor is not None and max_iters < steady_floor:
        log.warning("HYPERLOOM_PROFILE_MAX_ITERS=%d is below the steady-state floor of %d", max_iters, steady_floor)
    elif max_iters > cap:
        log.warning("HYPERLOOM_PROFILE_MAX_ITERS=%d exceeds the serialization-safe cap of %d", max_iters, cap)
    return delay_iters, max_iters


class _ProfileConfig(NamedTuple):
    """What shaping a profile run capped, for the phases that read it back."""

    osl_val: int
    conc_val: int
    atom_caps: _AtomTracelensCaps | None = None
    #: The serialization-safe capture cap, so the re-assertion at the end of
    #: materialization can reject a ``max_iterations`` above it (or non-positive,
    #: which vLLM reads as no limit) instead of accepting the flag on its name alone.
    pending_vllm_profiler_cap: int = 0
    #: ``(sentinel, flag)`` pairs remembered so that same re-assertion can restore
    #: exactly the profiler flags a later step dropped, without re-stating the ones
    #: that survived. A dropped ``max_iterations`` is an OOM, not a cosmetic problem.
    pending_vllm_profiler_flags: Sequence[tuple[str, str]] = ()
    profile_num_prompts: int | None = None


class _ProfileKnobs(NamedTuple):
    """What the capture window and the patched framework settled for a profile."""

    framework: str
    delay_iters: int
    max_iters: int
    sglang_sitecustomize: bool
    tracelens_patch_ok: bool


class _ProfileWindow(NamedTuple):
    """The decode window a profile capture is bounded to, and what it costs in prompts."""

    osl_val: int
    delay_iters: int
    max_iters: int
    num_prompts: int


class _ProfileOsl(NamedTuple):
    """The output length a profile serves, and the captured steps its steady state needs."""

    osl_val: int
    steady_floor: int


def _profile_range_ratio(envs: Mapping[str, Any]) -> float:
    """The request-length spread the steady-state floor is computed against."""
    try:
        return float(envs.get("RANDOM_RANGE_RATIO", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _steady_state_floor(*, osl_val: int, conc_val: int, range_ratio: float) -> int:
    """Captured decode steps the splitter needs to isolate a steady-state window.

    Mirrors TraceLens ``find_steady_state_window``: a capture shorter than this is
    reported back as ``trace_split_no_steady_state``.
    """
    return math.ceil(max(osl_val, 1) * (1.0 + range_ratio) / (2.0 * max(conc_val, 1)))


def _fit_profile_osl(*, osl_val: int, conc_val: int, range_ratio: float, cap: int) -> _ProfileOsl:
    """Resolve the profile-scoped OSL, lowering it until its steady state fits ``cap``.

    ``PROFILE_OSL`` (via ``--profile-osl``) is honored as-is and only warned about;
    otherwise the profile serves ``min(served OSL, _PROFILE_DEFAULT_OSL)`` and is
    lowered further rather than losing its steady-state window.
    """
    raw = os.environ.get("PROFILE_OSL", "").strip()
    explicit = raw.isdigit() and int(raw) > 0
    osl_val = int(raw) if explicit else min(osl_val, _PROFILE_DEFAULT_OSL)
    steady_floor = _steady_state_floor(osl_val=osl_val, conc_val=conc_val, range_ratio=range_ratio)
    if steady_floor <= cap:
        return _ProfileOsl(osl_val=osl_val, steady_floor=steady_floor)
    if explicit:
        log.warning(
            "PROFILE_OSL=%d needs %d captured steps to reach steady "
            "state, above the profile cap of %d; the trace may lack a "
            "steady-state window (trace_split_no_steady_state). Lower "
            "--profile-osl or raise HYPERLOOM_PROFILE_MAX_STEPS_CAP.",
            osl_val,
            steady_floor,
            cap,
        )
        return _ProfileOsl(osl_val=osl_val, steady_floor=steady_floor)
    fitted_osl = max(1, int(cap * 2 * max(conc_val, 1) / (1.0 + range_ratio)))
    log.warning(
        "profile OSL %d would need %d captured steps to reach "
        "steady state (> cap %d); lowering profile OSL to %d so the "
        "capture stays serializable. Baseline/optimize unaffected.",
        osl_val,
        steady_floor,
        cap,
        fitted_osl,
    )
    return _ProfileOsl(
        osl_val=fitted_osl,
        steady_floor=_steady_state_floor(osl_val=fitted_osl, conc_val=conc_val, range_ratio=range_ratio),
    )


def _profile_prompt_count(*, osl_val: int, conc_val: int, captured_iters: int) -> int:
    """Prompts the engine needs to reach ``captured_iters`` decode steps.

    N prompts run roughly ``N * OSL / CONC`` iterations, so this inverts that and
    keeps a 2x buffer; running out of prompts ends the capture short of its window.
    """
    safe_conc = max(conc_val, 1)
    safe_osl = max(osl_val, 1)
    iters_to_prompts = max(1, (captured_iters * safe_conc + safe_osl - 1) // safe_osl)
    return max(safe_conc, iters_to_prompts * 2)


def _fit_profile_capture_window(envs: dict[str, Any], *, osl_val: int, conc_val: int) -> _ProfileWindow:
    """Bound what a profile captures to a window the engine can serialize."""
    cap = _profile_steps_cap(os.environ)
    range_ratio = _profile_range_ratio(envs)
    fitted = _fit_profile_osl(osl_val=osl_val, conc_val=conc_val, range_ratio=range_ratio, cap=cap)
    delay_iters, max_iters = _profile_capture_window(
        os.environ,
        cap=cap,
        delay_iters=max(0, int(fitted.osl_val * (range_ratio + 1) * 3 - cap / 2)),
        steady_floor=fitted.steady_floor,
    )
    return _ProfileWindow(
        osl_val=fitted.osl_val,
        delay_iters=delay_iters,
        max_iters=max_iters,
        num_prompts=_profile_prompt_count(
            osl_val=fitted.osl_val,
            conc_val=conc_val,
            captured_iters=delay_iters + max_iters,
        ),
    )


def _bound_vllm_profiler_capture(
    envs: dict[str, str],
    config: _ProfileConfig,
    knobs: _ProfileKnobs,
) -> _ProfileConfig:
    """Write the capture window into the flags vLLM's own profiler reads.

    ``profiler`` and ``torch_profiler_dir`` are normally set by Magpie's launcher
    script, not by this layer -- but that script appends its own flags *before*
    EXTRA_VLLM_ARGS in the real ``vllm serve`` invocation, so the argv preflight
    probe (which only sees EXTRA_VLLM_ARGS) checks capture_torch_profiler/
    delay_iterations/max_iterations against a ProfilerConfig that never saw
    ``profiler=torch`` or a trace dir. vLLM's validator requires both whenever
    those bounds are present, so this layer's profiler bounds would be treated as
    invalid and dropped instead of launched. Asserting ``profiler=torch`` keeps the
    probed fragment self-consistent. No ``torch_profiler_dir`` is appended: on the
    real ``vllm serve`` line Magpie's launcher emits ``<workspace>/torch_trace``
    *before* ``EXTRA_VLLM_ARGS``, and vLLM's last-wins merge would let a second one
    here override Magpie and send traces to the task root.
    """
    existing_vllm_args = str(envs.get("EXTRA_VLLM_ARGS", ""))
    profiler_flags = [
        ("delay_iterations", f"--profiler-config.delay_iterations {knobs.delay_iters}"),
        ("max_iterations", f"--profiler-config.max_iterations {knobs.max_iters}"),
    ]
    if _profiler_flag_value(existing_vllm_args, "profiler") is None:
        profiler_flags.append(("profiler", "--profiler-config.profiler torch"))
    if knobs.tracelens_patch_ok:
        profiler_flags.append(("capture_torch_profiler", "--profiler-config.capture_torch_profiler True"))
        profiler_flags.append(("detailed_trace_annotation", "--profiler-config.detailed_trace_annotation True"))
    # vLLM's AsyncLLM-side profiler tracks no iterations and captures the
    # whole start_profile..stop_profile range, so it has to stay off
    # wherever we bound the worker-side one. The YAML normally carries the
    # flag, but a replacing candidate wipes the YAML value too, so it
    # belongs in the set the re-assertion can restore.
    profiler_flags.append(("ignore_frontend", "--profiler-config.ignore_frontend True"))
    # vLLM's stop path builds a key_averages() summary table in Python on every rank. Nothing here reads
    # it, and on a long-prompt eager trace it blocks the engine and grows each worker by hundreds of GiB.
    profiler_flags.append(
        ("torch_profiler_dump_cuda_time_total", "--profiler-config.torch_profiler_dump_cuda_time_total False")
    )
    # Injected unconditionally so the computed cap WINS over anything the YAML
    # pins, via the repeated-flag last-wins that vLLM's argparse and this
    # layer both rely on. Filtering out a flag the YAML already carries would
    # hand the run e.g. a hand-written ``max_iterations 100000`` -- unbounded
    # in practice -- and quietly discard the serialization-safe budget
    # computed above; HYPERLOOM_PROFILE_MAX_ITERS is the override channel for
    # that, not the YAML. ``ignore_frontend`` is the exception: the shipped
    # profile YAML already sets it, and vLLM warns on duplicate keys.
    profiler_args = " ".join(
        flag
        for sentinel, flag in profiler_flags
        if sentinel != "ignore_frontend" or "ignore_frontend" not in existing_vllm_args
    )
    if "delay_iterations" not in existing_vllm_args:
        envs["EXTRA_VLLM_ARGS"] = f"{existing_vllm_args} {profiler_args}".strip()
    return config._replace(
        pending_vllm_profiler_cap=knobs.max_iters,
        pending_vllm_profiler_flags=profiler_flags,
    )


def _profile_request_body(envs: Mapping[str, Any]) -> dict[str, Any]:
    """The ``/start_profile`` body this run will send, as the envs already carry it."""
    try:
        parsed = json.loads(str(envs.get("PROFILE_EXTRA_BODY", "{}")))
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _shape_discovery_enabled(bench: Mapping[str, Any]) -> bool:
    """Whether this profile asks the server to record kernel shapes.

    Shape discovery balloons an eager + with_stack trace, so
    HYPERLOOM_PROFILE_SHAPE_DISCOVERY turns it off wholesale. Gemma2 crashes
    CUDA-graph capture with it, so the model gates it too;
    HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE=1 skips only that second gate.
    """
    if not env_flag("HYPERLOOM_PROFILE_SHAPE_DISCOVERY", default=True):
        return False
    if env_bool("HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE"):
        return True
    model = str(bench.get("model") or "")
    if not _model_is_gemma2(model):
        return True
    log.info(
        "Gemma2 roofline: disabling shape-discovery to avoid "
        "CUDA-graph capture crash (hipErrorStreamCapture"
        "Unsupported); CUDA graph + profiling kept. Set "
        "HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE=1 to override.",
    )
    if _load_model_config_dict(model) is None:
        log.warning(
            "Gemma2 detected via path heuristic (no readable "
            "config.json at %r); shape-discovery skip may be "
            "imprecise.",
            model,
        )
    return False


def _publish_shape_tool_path(envs: dict[str, str], body: dict[str, Any], *, shape_discovery: bool) -> None:
    """Point an unpatched SGLang at the kernel-shape tool and record the body it takes.

    Shapes reach this server through PYTHONPATH, not through a request-body flag
    or a CUDA-graph arg -- an unpatched SGLang rejects both.
    """
    body.pop("shape_discovery", None)
    body.setdefault("detailed_annotations", True)
    from ._server_patcher import kernel_shape_tool_dir

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
    envs["PROFILE_EXTRA_BODY"] = json.dumps(body)


def _publish_patched_shape_discovery(
    envs: dict[str, str],
    body: dict[str, Any],
    *,
    shape_discovery: bool,
    tracelens_patch_ok: bool,
) -> None:
    """Record the capture options only a patched SGLang build accepts.

    The gate is the degraded reason rather than ``tracelens_patch_ok``, because a
    patch can be baked into the image without ever being attempted here; only a
    patch that was attempted and failed clears the annotations.
    """
    degraded = envs.get("HYPERLOOM_PROFILE_DEGRADED_REASON") == _TRACELENS_PATCH_UNAVAILABLE
    body["shape_discovery"] = shape_discovery and not degraded
    body["detailed_annotations"] = False if degraded else body.get("detailed_annotations", True)
    # Written before the per-task extra_envs merge, so an extra_envs
    # PROFILE_EXTRA_BODY can still drop start_step/num_steps. Not
    # re-asserted for SGLang (unlike vLLM): "unbounded" vs "no capture"
    # for non-positive num_steps needs a SGLang-side answer.
    envs["PROFILE_EXTRA_BODY"] = json.dumps(body)
    if tracelens_patch_ok and body["shape_discovery"]:
        # Patched SGLang exposes this arg; unpatched errors on it.
        existing_sglang = str(envs.get("EXTRA_SGLANG_ARGS", ""))
        if "shape-discovery-for-cuda-graph-profile" not in existing_sglang:
            envs["EXTRA_SGLANG_ARGS"] = f"{existing_sglang} --enable-shape-discovery-for-cuda-graph-profile".strip()


def _request_sglang_profile_annotations(
    envs: dict[str, str],
    bench: dict[str, Any],
    knobs: _ProfileKnobs,
) -> None:
    """Ask the SGLang server for the capture window and annotations this profile needs."""
    body = _profile_request_body(envs)
    # Always override start_step/num_steps (the template has CONC=8 placeholders).
    body["start_step"] = knobs.delay_iters
    body["num_steps"] = knobs.max_iters
    shape_discovery = _shape_discovery_enabled(bench)
    if knobs.sglang_sitecustomize:
        _publish_shape_tool_path(envs, body, shape_discovery=shape_discovery)
    else:
        _publish_patched_shape_discovery(
            envs,
            body,
            shape_discovery=shape_discovery,
            tracelens_patch_ok=knobs.tracelens_patch_ok,
        )


def _shape_profile_trace(
    envs: dict[str, str],
    bench: dict[str, Any],
    config: _ProfileConfig,
    knobs: _ProfileKnobs,
) -> _ProfileConfig:
    """Shape the trace a profile run captures, per framework.

    ATOM drives its own iteration count, so the request count is pinned to the
    concurrency and the caps come from TraceLens; the serving frameworks take a
    capture window instead, written into the flags their profiler reads.
    """
    if "atom" in knobs.framework:
        # ATOM has no delay/max-iteration window; extra prompts only grow
        # the HTTP-bracketed trace. Force NUM_PROMPTS=CONC.
        return config._replace(atom_caps=_atom_tracelens_caps(), profile_num_prompts=config.conc_val)
    if "vllm" in knobs.framework:
        return _bound_vllm_profiler_capture(envs, config, knobs)
    _request_sglang_profile_annotations(envs, bench, knobs)
    return config
