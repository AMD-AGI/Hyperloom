"""Ask the installed engine what it supports, instead of hardcoding field names.

The failure this prevents is concrete, and it has already happened twice in
opposite directions on this project. The campaign container
(``0.1.1.dev50+geed1f3d0c``) exposes a full torch-profiler schedule on
``ProfilerConfig`` — ``delay_iterations``, ``warmup_iterations``,
``active_iterations``, ``wait_iterations``, ``max_iterations``. The vLLM source
checked out elsewhere on the same cluster (``0.1.dev12734+gd111bc53a``) has only
``delay_iterations`` and ``max_iterations``, and spells record-shapes
``torch_profiler_record_shapes``. A plan written against either tree breaks on
the other: against the newer build you leave the schedule unconfigured and
profile more than you meant to; against the older one you pass fields that do
not exist and lose a GPU-hour to a ``TypeError``.

Neither tree is "the" answer, which is the whole point. Profiling configuration
is the fastest-drifting part of an engine, an optimization agent runs against
whatever build the campaign pinned, and reading a *different* checkout is not
evidence about the pinned one. So: probe the dataclass on the build you are
actually running, keep what it has, and report what was dropped rather than
passing it and failing at engine init.
"""

from __future__ import annotations

import dataclasses
from typing import Any


def profiler_config_fields() -> set[str] | None:
    """Field names on the installed vLLM ``ProfilerConfig``, or None if absent.

    None means this build has no ``ProfilerConfig`` at all — an older vLLM
    gated on the ``VLLM_TORCH_PROFILER_DIR`` environment variable instead.
    """
    try:
        from vllm.config import ProfilerConfig  # type: ignore
    except Exception:
        return None
    try:
        return {f.name for f in dataclasses.fields(ProfilerConfig)}
    except Exception:
        # Pydantic dataclass without a dataclass view — fall back to annotations.
        return set(getattr(ProfilerConfig, "__annotations__", {}) or {}) or None


def build_profiler_config(
    trace_dir: str,
    *,
    delay_iterations: int = 0,
    warmup_iterations: int = 0,
    active_iterations: int = 0,
    wait_iterations: int = 0,
    max_iterations: int = 0,
    record_shapes: bool = False,
    use_gzip: bool = False,
    with_stack: bool = False,
    ignore_frontend: bool = True,
) -> tuple[dict[str, Any], dict]:
    """Return ``(profiler_config_kwargs, report)`` valid for the installed vLLM.

    Requested options the build does not support are dropped and listed in
    ``report["dropped"]``. Read that list: a dropped ``delay_iterations`` means
    the capture is **not** phase-isolated by the engine, and the caller has to
    fall back to isolating by subtraction rather than silently profile prefill
    and call it decode.

    The schedule knobs follow ``torch.profiler.schedule`` semantics where the
    build has them: ``wait`` (profiler off), ``delay`` (skip N engine
    iterations), ``warmup`` (profiler runs, data discarded), ``active``
    (iterations actually recorded). Builds that expose only ``delay_iterations``
    and ``max_iterations`` get the same window approximated by those two, and
    ``report["schedule"]`` says which of the two shapes you got — a profile
    taken with a warmup is not directly comparable to one taken without.

    Zero means "not requested" and is never sent, so a caller that does not care
    about the schedule inherits the engine's defaults rather than pinning them.

    ``with_stack`` defaults to False against vLLM's own default of True. Stack
    tracing is the single largest source of profiler overhead and nothing in
    per-kernel GPU attribution reads it.
    """
    available = profiler_config_fields()
    report: dict = {"available": sorted(available) if available else [], "dropped": {}}

    if available is None:
        report["mode"] = "env"
        report["note"] = (
            "no ProfilerConfig in this build; profiling is gated on the "
            "VLLM_TORCH_PROFILER_DIR environment variable"
        )
        return {}, report

    report["mode"] = "profiler_config"
    wanted: dict[str, Any] = {
        "profiler": "torch",
        "torch_profiler_dir": trace_dir,
        "torch_profiler_use_gzip": use_gzip,
        "torch_profiler_with_stack": with_stack,
        "ignore_frontend": ignore_frontend,
    }
    # Spelled differently across builds; send whichever this one has.
    for alias in ("torch_profiler_record_shapes", "record_shapes"):
        if alias in available:
            wanted[alias] = record_shapes
            break
    else:
        wanted["torch_profiler_record_shapes"] = record_shapes

    requested_schedule = {
        "wait_iterations": wait_iterations,
        "delay_iterations": delay_iterations,
        "warmup_iterations": warmup_iterations,
        "active_iterations": active_iterations,
        "max_iterations": max_iterations,
    }
    wanted.update({k: v for k, v in requested_schedule.items() if v})

    cfg: dict[str, Any] = {}
    for key, value in wanted.items():
        if key in available:
            cfg[key] = value
        else:
            report["dropped"][key] = value

    got = [k for k in requested_schedule if k in cfg]
    report["schedule"] = {
        "requested": {k: v for k, v in requested_schedule.items() if v},
        "applied": {k: cfg[k] for k in got},
        "shape": "full" if "active_iterations" in cfg else "delay_and_cap",
    }
    # A build without `active_iterations` limits the window with `max_iterations`
    # instead. If the caller asked for `active` and this build has no such field,
    # fall the request through to the cap rather than silently profiling forever.
    if active_iterations and "active_iterations" not in available and "max_iterations" in available:
        cfg["max_iterations"] = cfg.get("max_iterations") or active_iterations
        report["schedule"]["applied"]["max_iterations"] = cfg["max_iterations"]
        report["schedule"]["note"] = (
            "no active_iterations on this build; window capped with "
            "max_iterations instead (no discarded-warmup phase)"
        )

    # vLLM validates: torch_profiler_dir is required when profiler == "torch",
    # and is rejected when it is not. Never emit a half-configured pair.
    if cfg.get("profiler") == "torch" and not cfg.get("torch_profiler_dir"):
        cfg.pop("profiler", None)
        report["note"] = "torch_profiler_dir unsupported; profiler key withheld"

    report["phase_isolation_by_engine"] = "delay_iterations" in cfg
    return cfg, report


def engine_kwarg_names() -> set[str] | None:
    """Every keyword ``LLM(...)`` will accept on this build, or None if unknown.

    The union of ``EngineArgs`` fields (which is where ``LLM``'s ``**kwargs``
    land) and ``LLM.__init__``'s own named parameters.
    """
    names: set[str] = set()
    try:
        from vllm.engine.arg_utils import EngineArgs  # type: ignore

        names |= {f.name for f in dataclasses.fields(EngineArgs)}
    except Exception:
        try:
            from vllm.engine.arg_utils import EngineArgs  # type: ignore

            names |= set(getattr(EngineArgs, "__annotations__", {}) or {})
        except Exception:
            return None
    try:
        import inspect

        from vllm import LLM  # type: ignore

        names |= set(inspect.signature(LLM.__init__).parameters) - {"self", "kwargs"}
    except Exception:
        pass
    return names or None


def check_engine_kwargs(kwargs: dict[str, Any]) -> list[str]:
    """Keywords this build's engine will reject. Empty list means safe to pass.

    Worth calling before every ``LLM(...)``. The failure it catches is cheap to
    make and expensive to find: ``moe_backend="aiter"`` reads like an engine
    option, is not one on any build, and raises ``TypeError`` from
    ``EngineArgs(**kwargs)`` after the script has already printed its banner —
    which looks like "engine init failed" rather than "you passed a bad kwarg".

    Returns ``[]`` when the engine cannot be introspected; a build we cannot
    read is not a build we should block on.
    """
    known = engine_kwarg_names()
    if known is None:
        return []
    return sorted(k for k in kwargs if k not in known)


def rocm_profiling_env(
    *,
    use_aiter: bool = True,
    breakable_cudagraph: bool = True,
) -> dict[str, str]:
    """Environment a DSv4.x ROCm profiling run needs before the engine starts.

    These are **environment variables, not engine kwargs**. Passing e.g.
    ``moe_backend="aiter"`` to ``LLM(...)`` does not select the aiter MoE path —
    ``moe_backend`` is not an engine argument; it is an internal attribute of a
    quantisation method. The kwarg falls through to ``EngineArgs(**kwargs)`` and
    raises ``TypeError`` at construction, before any GPU work happens.

    ``VLLM_USE_BREAKABLE_CUDAGRAPH`` is not optional for DSv4.1-Flash:
    ``DeepseekV41ForCausalLM`` is not torch-compiled upstream, so the default
    ``cudagraph_mode=FULL_AND_PIECEWISE`` aborts at engine init without it.
    A profiling script that omits it dies in the same place a server would.
    """
    env: dict[str, str] = {"PYTHONUNBUFFERED": "1"}
    if use_aiter:
        env["VLLM_ROCM_USE_AITER"] = "1"
        env["VLLM_ROCM_USE_AITER_MOE"] = "1"
    if breakable_cudagraph:
        env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
    return env


def describe() -> dict:
    """One-call summary of what the installed engine can be asked for.

    Cheap enough to run at the top of any profiling script, and worth printing
    into the session log: it is the record of which build a profile came from.
    """
    fields = profiler_config_fields()
    info: dict = {"profiler_config_fields": sorted(fields) if fields else None}
    try:
        import vllm  # type: ignore

        info["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except Exception as exc:
        info["vllm_version"] = None
        info["vllm_import_error"] = f"{type(exc).__name__}: {exc}"

    if fields:
        info["supports_engine_phase_isolation"] = "delay_iterations" in fields
        info["supports_iteration_cap"] = "max_iterations" in fields
        info["supports_full_schedule"] = "active_iterations" in fields
        # Schedule fields that exist on SOME vLLM builds and not others. This is
        # a statement about THIS build only — a field listed here is not
        # "fictional", it is absent here, and the same name may well be present
        # in the container the campaign actually runs. Print it, do not
        # generalise from it.
        known_variable = (
            "warmup_iterations",
            "active_iterations",
            "wait_iterations",
            "delay_iterations",
            "max_iterations",
            "detailed_trace_annotation",
            "record_shapes",
            "torch_profiler_record_shapes",
        )
        info["schedule_fields_present"] = sorted(f for f in known_variable if f in fields)
        info["schedule_fields_absent_here"] = sorted(
            f for f in known_variable if f not in fields
        )
    return info
