# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helper for the ``explore`` executor's grid runs.

Takes a base Magpie YAML + a list of (name, extra_server_args, extra_envs)
variants, runs Magpie once per variant, parses ``benchmark_report.json``,
returns the winners.
"""

from __future__ import annotations

import fnmatch as _fnmatch
import json
import logging
import os
import re
import subprocess

from hyperloom.common.env import is_truthy

from ._grid_base import (
    GridVariant,
)

log = logging.getLogger(__name__)


def resolve_skip_spec(params: dict | None) -> str:
    """Resolve the active skip spec from task params + process env.

    ``params["skip_variants"]`` may be a list[str] or a single str; both are
    flattened to comma-joined form before pattern parsing. Resolution order
    is ``params["skip_variants"]`` > ``$SKIP_VARIANTS`` > ``""``.

    Args:
        params (dict | None): Task params; ``skip_variants`` (list/tuple/str)
            takes precedence over the environment when present and non-empty.

    Returns:
        str: The stripped skip spec string, or ``""`` when neither source
        supplies a value.
    """
    val = ""
    if params and "skip_variants" in params:
        raw = params.get("skip_variants")
        if isinstance(raw, (list, tuple)):
            val = ",".join(str(x) for x in raw if x is not None)
        elif raw is not None:
            val = str(raw)
    if not val.strip():
        val = os.environ.get("SKIP_VARIANTS", "")
    return (val or "").strip()


def _parse_skip_spec(spec: str) -> list[str]:
    """Split ``spec`` on commas and whitespace; drop empties.

    Newlines are treated as commas, then each comma-separated token is
    further split on whitespace so mixed separators all flatten into one
    list of patterns.

    Args:
        spec (str): Raw skip spec (e.g. ``"attn_*, sched_dfs\nvllm_aiter"``).

    Returns:
        list[str]: Non-empty, stripped pattern tokens in source order.
    """
    if not spec:
        return []
    out: list[str] = []
    for token in spec.replace("\n", ",").split(","):
        for sub in token.split():
            t = sub.strip()
            if t:
                out.append(t)
    return out


# Matches ``--cuda-graph-max-bs 64`` and ``--cuda_graph_max_bs=64``; captures
# the integer value.
_RE_CUDA_GRAPH_MAX_BS = re.compile(r"--cuda[-_]graph[-_]max[-_]bs[= ]+(\d+)")

# Multi-node grid prioritisation + invalid-variant filtering. In multi-node
# mode a ``--max-hours`` cut may stop the loop before the whole grid is benched,
# so we drop known regressions and reorder survivors so strong candidates run
# first. Both are STRICT no-ops outside multi-node mode. ``priority_tags`` ranks
# variants by the first tag matching their ``note``/``name`` (lower = higher);
# params priority is concatenated ahead of backends priority by callers.
_MN_PARAMS_PRIORITY: tuple[str, ...] = (
    "cuda_graph_max_bs",
    "max_running_requests",
    "chunked_prefill",
    "schedule",
    "decode_steps",
    "torch_compile",
    "mem_fraction",
)

_MN_BACKENDS_PRIORITY: tuple[str, ...] = (
    "tier1",
    "tier2",
    "tier3",
    "tier4",
    "tier5_comm",
    "tier5",
    "comm_custom_ar",
)


def _mn_priority_index(variant: "GridVariant", priority_tags: "tuple[str, ...] | list[str]") -> int:
    """Return the rank of ``variant`` against ``priority_tags`` (lower = first).

    Matches the first ``priority_tags`` entry that is a substring of the
    variant's ``note`` (falling back to ``name`` when ``note`` is empty).
    Untagged variants return ``len(priority_tags)`` so a stable sort sinks them
    to the end while preserving their relative order.

    Args:
        variant (GridVariant): The variant to rank.
        priority_tags (tuple[str, ...] | list[str]): Ordered category tags;
            lower index == higher priority.

    Returns:
        int: The index of the first matching tag, or ``len(priority_tags)`` when
        none match.
    """
    haystack = variant.note or variant.name or ""
    for idx, tag in enumerate(priority_tags):
        if tag and tag in haystack:
            return idx
    return len(priority_tags)


def reorder_grid_for_multi_node(
    grid: list["GridVariant"],
    *,
    priority_tags: "tuple[str, ...] | list[str]",
) -> list["GridVariant"]:
    """Stable-sort ``grid`` so likely multi-node winners run first.

    Single-node mode (``is_multi_node()`` False) returns ``grid`` unchanged,
    bit-for-bit (hard requirement: never alter single-node grid order). In
    multi-node mode variants are stably sorted by ``_mn_priority_index`` so
    tagged variants surface ahead of untagged ones and a ``--max-hours`` cut
    still benches the strong candidates.

    Args:
        grid (list[GridVariant]): The variants to reorder.
        priority_tags (tuple[str, ...] | list[str]): Ordered category tags used
            to rank variants.

    Returns:
        list[GridVariant]: ``grid`` unchanged in single-node mode, else a stably
        sorted copy with higher-priority variants first.
    """
    from ._multi_node_env import is_multi_node

    if not is_multi_node():
        return grid
    # ``sorted`` is stable, so ties keep input order.
    return sorted(grid, key=lambda v: _mn_priority_index(v, priority_tags))


def apply_multi_node_invalid_variants(
    grid: list["GridVariant"],
) -> tuple[list["GridVariant"], list[dict]]:
    """Drop variants that are known regressions/invalid on multi-node fabrics.

    Returns ``(kept, dropped)``. ``dropped`` entries carry ``name`` / ``source``
    / ``reason`` keys (matching the explore-loop ``skipped_dup`` shape). A
    STRICT no-op outside multi-node mode: returns ``(grid, [])`` so single-node
    behaviour is never altered.

    Current rule: ``--cuda-graph-max-bs N`` with ``N < $CONC`` regresses ~50%
    in multi-node mode (cuda-graph cache misses every cross-node decode tick),
    so it is dropped from the explore grid here.

    Args:
        grid (list[GridVariant]): The candidate variants to filter.

    Returns:
        tuple[list[GridVariant], list[dict]]: ``(kept, dropped)`` where dropped
        entries carry ``name``/``source``/``reason``; ``(grid, [])`` outside
        multi-node mode.
    """
    from ._multi_node_env import is_multi_node

    if not is_multi_node():
        return grid, []
    try:
        conc = int(os.environ.get("CONC", "64") or 64)
    except ValueError:
        conc = 64
    kept: list["GridVariant"] = []
    dropped: list[dict] = []
    for v in grid:
        m = _RE_CUDA_GRAPH_MAX_BS.search(v.extra_server_args or "")
        if conc > 0 and m and int(m.group(1)) < conc:
            dropped.append(
                {
                    "name": v.name,
                    "source": "multi_node_invalid",
                    "reason": (
                        f"cuda_graph_max_bs={m.group(1)} < CONC={conc} "
                        "(multi-node graph-cache miss is a known ~50% regression)"
                    ),
                }
            )
            continue
        kept.append(v)
    return kept, dropped


# Matches ``--moe-runner-backend aiter`` and ``--moe_runner_backend=aiter``.
_RE_AITER_MOE_RUNNER = re.compile(r"--moe[-_]runner[-_]backend[= ]+aiter\b")


def _operator_pinned_envs() -> dict[str, str]:
    """Return the operator's ``--extra-env`` pins from the CLI handoff env.

    The CLI serializes ``--extra-env NAME=VALUE`` pairs into
    ``$INFERENCE_OPTIMIZER_EXTRA_ENV`` (JSON object). Any parse failure yields
    an empty dict so the caller degrades to "no pin".

    Returns:
        dict[str, str]: The operator-pinned env, or ``{}`` when unset/invalid.
    """
    raw = os.environ.get("INFERENCE_OPTIMIZER_EXTRA_ENV", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def apply_aiter_moe_pin_filter(
    grid: list["GridVariant"],
) -> tuple[list["GridVariant"], list[dict]]:
    """Drop variants that re-enable the aiter MoE runner when it is pinned off.

    When the operator pins ``SGLANG_USE_AITER=0`` (via ``--extra-env``), the
    aiter fused-MoE runner is deliberately disabled (it hangs/crashes server
    launch on some ROCm images). Explore variants that flip it back on —
    ``SGLANG_USE_AITER=1`` in ``extra_envs`` (the master switch also gates the
    aiter MoE runner) or ``--moe-runner-backend aiter`` in ``extra_server_args``
    — would re-trigger that failure and burn the budget, so they are dropped.
    Aiter *attention* / allreduce / rmsnorm variants (which do NOT select the
    aiter MoE runner) are kept, since those are stable and can still win.

    A STRICT no-op when ``SGLANG_USE_AITER`` is not operator-pinned off.

    Args:
        grid (list[GridVariant]): The candidate variants to filter.

    Returns:
        tuple[list[GridVariant], list[dict]]: ``(kept, dropped)`` where dropped
        entries carry ``name``/``source``/``reason`` (``source`` is
        ``"aiter_moe_pinned_off"``).
    """
    pins = _operator_pinned_envs()
    aiter_pinned_off = "SGLANG_USE_AITER" in pins and not is_truthy(
        pins["SGLANG_USE_AITER"], default=True
    )
    if not aiter_pinned_off:
        return list(grid), []

    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        envs = {str(k): str(val) for k, val in (getattr(v, "extra_envs", None) or {}).items()}
        reenables_master = "SGLANG_USE_AITER" in envs and is_truthy(
            envs["SGLANG_USE_AITER"], default=False
        )
        selects_aiter_moe = bool(_RE_AITER_MOE_RUNNER.search(v.extra_server_args or ""))
        if not (reenables_master or selects_aiter_moe):
            kept.append(v)
            continue
        trigger = "SGLANG_USE_AITER=1" if reenables_master else "--moe-runner-backend aiter"
        dropped.append(
            {
                "name": v.name,
                "source": "aiter_moe_pinned_off",
                "reason": (
                    f"variant re-enables the aiter MoE runner ({trigger}) while the "
                    "operator pinned SGLANG_USE_AITER off"
                ),
            }
        )
    return kept, dropped


# Framework / hardware compatibility filter: each entry maps an
# ``extra_server_args`` substring to a required model class.
_COMPATIBILITY_FLAG_RULES: tuple[tuple[str, str], ...] = (
    ("--enable-flashinfer-mla", "mla"),
    ("--enable-deepep-moe", "moe"),
    ("--enable-ep-moe", "moe"),
)

# xDiT (diffusion) do-not-set blacklist — env knobs that crash or regress on
# FLUX.2-class DiT models with Ulysses SP. Each entry maps an env key to the
# set of forbidden values (``"*"`` = any truthy value) and a short reason.
# Enforced for the ``xdit`` framework only, in :func:`apply_compatibility_filter`.
_XDIT_ENV_BLACKLIST: dict[str, tuple[frozenset[str], str]] = {
    "XDIT_ATTENTION_BACKEND": (
        frozenset({"aiter_fp8", "aiter_sage", "aiter_sage_v2"}),
        "approximate/quantized attention regresses on Ulysses>=4 (attention is ~2% of FLOPS at small tokens/GPU)",
    ),
    "XDIT_USE_FP4_GEMMS": (frozenset({"*"}), "precision locked to BF16 (FP4 = different model)"),
    "XDIT_USE_FP8_GEMMS": (frozenset({"*"}), "precision locked to BF16 (FP8 = different model)"),
    "RCCL_MSCCL_ENABLE": (frozenset({"*"}), "MSCCL overhead > savings for 2.53MB per-pair A2A"),
    "PYTORCH_TUNABLEOP_TUNING": (frozenset({"*"}), "GPU memory fault when combined with torch.compile"),
    "TRITON_HIP_USE_ASYNC_COPY": (frozenset({"*"}), "crashes on MI355X (gfx950)"),
    "NCCL_PROTO": (frozenset({"ll", "LL"}), "NCCL_PROTO=LL regresses 10-22%; use LL128/SIMPLE"),
}

# Known crash combinations (all keys present + truthy → drop).
_XDIT_ENV_COMBO_BLACKLIST: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("AMD_DIRECT_DISPATCH", "AMDGCN_USE_BUFFER_OPS"),
        "AMD_DIRECT_DISPATCH=1 + AMDGCN_USE_BUFFER_OPS=1 is a known crash (-28.6%)",
    ),
)


def xdit_blacklist_reason(
    extra_envs: dict[str, str] | None,
) -> str | None:
    """Return a drop reason if a variant trips the xDiT do-not-set blacklist.

    Args:
        extra_envs (dict[str, str] | None): The variant's env overrides.

    Returns:
        str | None: A human-readable reason when blacklisted, else ``None``.
    """
    envs = {str(k): str(v) for k, v in (extra_envs or {}).items()}
    for key, (bad_values, reason) in _XDIT_ENV_BLACKLIST.items():
        if key not in envs:
            continue
        val = envs[key]
        if "*" in bad_values:
            if is_truthy(val, default=True):
                return f"{key}={val}: {reason}"
        elif val.strip().lower() in {b.lower() for b in bad_values}:
            return f"{key}={val}: {reason}"
    for keys, reason in _XDIT_ENV_COMBO_BLACKLIST:
        if all(k in envs and is_truthy(envs[k], default=True) for k in keys):
            return reason
    return None


# HY-WorldPlay (AR video) do-not-set blacklist — env knobs that change the model
# (precision / attention math / resolution) or are known to crash on MI355X
# (gfx950). Enforced for the ``worldplay`` framework only, in
# :func:`apply_compatibility_filter`. The vendored body already forces FP8 GEMMs
# OFF; blacklisting here stops explore from ever proposing the variant.
_WORLDPLAY_ENV_BLACKLIST: dict[str, tuple[frozenset[str], str]] = {
    "WORLDPLAY_USE_FP8_GEMMS": (frozenset({"*"}), "precision locked to BF16 (FP8 = different model)"),
    "WORLDPLAY_USE_FP8_GEMM": (frozenset({"*"}), "precision locked to BF16 (FP8 = different model)"),
    "WORLDPLAY_USE_FP4_GEMMS": (frozenset({"*"}), "precision locked to BF16 (FP4 = different model)"),
    "WORLDPLAY_USE_SAGEATTN": (frozenset({"*"}), "approximate attention alters the output distribution (different model)"),
    "WORLDPLAY_HEIGHT": (frozenset({"*"}), "resolution is part of the workload spec — not an optimization knob"),
    "WORLDPLAY_WIDTH": (frozenset({"*"}), "resolution is part of the workload spec — not an optimization knob"),
    "WORLDPLAY_NUM_FRAMES": (frozenset({"*"}), "frame count is part of the workload spec — not an optimization knob"),
    "WORLDPLAY_NUM_STEPS": (frozenset({"*"}), "step count is part of the workload spec — not an optimization knob"),
    "WORLDPLAY_FEW_STEP": (frozenset({"*"}), "step-distillation changes the model — not a BF16-safe knob"),
    "TRITON_HIP_USE_ASYNC_COPY": (frozenset({"*"}), "crashes on MI355X (gfx950)"),
    # Session 20260730T135601Z: "Memory access fault by GPU node-2" on this
    # pipeline, raised after MIOpen fails to find a grouped-conv solver. The
    # fault path is MIOpen, not torch.compile, so ENABLED alone is unsafe too.
    # A GPU memory fault outlives the variant that caused it and poisons every
    # later measurement, so this is refused rather than merely deprioritised.
    "PYTORCH_TUNABLEOP_ENABLED": (frozenset({"*"}), "faults the GPU on gfx950 with this pipeline (MIOpen grouped-conv solver miss)"),
    "PYTORCH_TUNABLEOP_TUNING": (frozenset({"*"}), "TunableOp tuning faults the GPU on gfx950 with this pipeline"),
}

# Measurement-contract keys: these set HOW the fps is sampled — ``--warmup``
# discarded full generations and ``--repeats`` timed generations aggregated into
# mean/std — not what runs. A variant that shrinks either still emits a
# legal-looking steady-state fps, but sampled differently from the 5-repeat
# baseline it is ranked against. Unlike ``_WORLDPLAY_ENV_BLACKLIST`` (which only
# fires on a truthy value) ANY override is a violation here, including ``0``:
# ``WORLDPLAY_WARMUP_CHUNKS=0`` is precisely the harmful setting, since it folds
# cold-start cost into the timed runs.
_WORLDPLAY_ENV_MEASUREMENT_LOCK: dict[str, str] = {
    "WORLDPLAY_REPEATS": "timed-repeat count is part of the baseline measurement contract (locked at 3)",
    "WORLDPLAY_WARMUP_CHUNKS": "warmup-generation count is part of the baseline measurement contract (locked at 1)",
}

# Known crash combinations (all keys present + truthy → drop).
_WORLDPLAY_ENV_COMBO_BLACKLIST: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("AMD_DIRECT_DISPATCH", "AMDGCN_USE_BUFFER_OPS"),
        "AMD_DIRECT_DISPATCH=1 + AMDGCN_USE_BUFFER_OPS=1 is a known crash",
    ),
)

# The customer's byte-identical ``bench_fps.py`` accepts ONLY this CLI surface;
# any other ``--flag`` makes argparse error out and the leg dies as
# ``no_measurement``. The explore LLM proposer keys off the *model's* native
# HunyuanVideo CLI (``--use_cache teacache``/``fbcache``/``magcache``,
# ``--attention_backend``, ``--enable_step_distill``, ``--cfg_distilled``,
# ``--enable_tiling`` …) — none of which the customer wrapper exposes — so those
# variants waste a full ~10-min leg for nothing. We drop them before dispatch and
# steer the proposer to ``WORLDPLAY_*`` env knobs + the accepted tunables.
_WORLDPLAY_ACCEPTED_SERVER_FLAGS: frozenset[str] = frozenset(
    {
        "--action_ckpt",
        "--aspect_ratio",
        "--dtype",
        "--enable_torch_compile",
        "--few_step",
        "--group_offloading",
        "--height",
        "--image_path",
        "--model_path",
        "--model_type",
        "--negative_prompt",
        "--num_inference_steps",
        "--offloading",
        "--out",
        "--pose",
        "--profile_steps",
        "--prompt",
        "--quality-calib-eps",
        "--quality-calib-margin",
        "--quality-calib-samples",
        "--quality-calibrate",
        "--quality-frames",
        "--quality-lpips-max",
        "--quality-mse-max",
        "--quality-ref",
        "--quality-ref-write",
        "--quality-ssim-min",
        "--repeats",
        "--resolution",
        "--seed",
        "--tag",
        "--torch_profiler_dir",
        "--transformer_resident_ar_rollout",
        "--video_length",
        "--warmup",
        "--width",
        "--worldplay-dir",
    }
)

# Accepted-but-forbidden: ``bench_fps.py`` takes these, but each changes the
# workload operating point (steps / frames / resolution) or the model
# (precision / step-distillation). ``_WORLDPLAY_ENV_BLACKLIST`` locks the
# env-keyed forms; this locks the equivalent CLI flags, which the env blacklist
# never sees when a variant passes them via ``extra_server_args``.
_WORLDPLAY_FORBIDDEN_SERVER_FLAGS: dict[str, str] = {
    "--num_inference_steps": "step count is part of the workload spec (locked at 50)",
    "--few_step": "step-distillation changes the model — out of scope (locked at 50 steps)",
    "--dtype": "precision locked to BF16 (dtype change = different model)",
    "--resolution": "resolution is part of the workload spec — not a tunable",
    "--height": "resolution is part of the workload spec — not a tunable",
    "--width": "resolution is part of the workload spec — not a tunable",
    "--video_length": "frame count is part of the workload spec (locked at 125)",
    "--aspect_ratio": "resolution/aspect is part of the workload spec — not a tunable",
    "--repeats": "timed-repeat count is part of the baseline measurement contract (locked at 3)",
    "--warmup": "warmup-generation count is part of the baseline measurement contract (locked at 1)",
}

# Matches a long option token (``--enable_torch_compile``, ``--quality-ref``),
# ignoring any ``=value`` suffix and short flags.
_RE_LONG_FLAG = re.compile(r"(--[A-Za-z][A-Za-z0-9_-]*)")

# Value markers that identify a step-distilled / few-step action checkpoint
# (e.g. ``ar_distilled_action_model``). The 50-step AR spec locks the action
# ckpt to the stock ``ar_model``; a distilled ckpt is a *different model* and is
# forbidden regardless of how many steps the same variant declares (a variant
# could pass ONLY the distilled ckpt and otherwise look 50-step-legit).
_WORLDPLAY_DISTILLED_CKPT_MARKERS: tuple[str, ...] = (
    "distill",
    "few_step",
    "fewstep",
    "few-step",
    "4step",
    "4-step",
)
# Captures the value after ``--action_ckpt`` (space- or ``=``-separated).
_RE_ACTION_CKPT = re.compile(r"--action_ckpt[=\s]+(\S+)")

# ``bench_fps.py`` declares these ``action="store_true"``, so they take NO value.
# Proposers routinely write ``--enable_torch_compile true`` (the shape every other
# accepted flag uses); argparse then rejects the trailing token and the leg dies as
# ``no_measurement`` seconds in, burning a whole propose/review/dispatch round.
# Dropping pre-dispatch costs nothing and lets the default seed grid take over.
_WORLDPLAY_STORE_TRUE_FLAGS: frozenset[str] = frozenset(
    {
        "--enable_torch_compile",
        "--few_step",
    }
)

_RE_STORE_TRUE_VALUE = re.compile(
    "(" + "|".join(sorted(re.escape(f) for f in _WORLDPLAY_STORE_TRUE_FLAGS)) + ")"
    r"(?![A-Za-z0-9_-])(?:=|\s+)(?!-)(\S+)"
)


def _ckpt_is_distilled(value: str | None) -> bool:
    """Whether an action-ckpt path names a distilled / few-step checkpoint.

    Args:
        value (str | None): The ``WORLDPLAY_ACTION_CKPT`` env value or the
            ``--action_ckpt`` CLI value.

    Returns:
        bool: ``True`` if the path contains any distillation marker.
    """
    v = (value or "").strip().lower()
    if not v:
        return False
    return any(m in v for m in _WORLDPLAY_DISTILLED_CKPT_MARKERS)


def worldplay_server_args_reason(
    extra_server_args: str | None,
) -> str | None:
    """Return a drop reason if a WorldPlay variant's CLI flags are unusable.

    Two independent checks against ``extra_server_args``: (1) any flag in
    :data:`_WORLDPLAY_FORBIDDEN_SERVER_FLAGS` (accepted by the wrapper but
    workload/precision-locked) is dropped with the lock reason; (2) any
    ``--flag`` outside :data:`_WORLDPLAY_ACCEPTED_SERVER_FLAGS` is dropped
    because the customer ``bench_fps.py`` would argparse-error → ``no_measurement``.

    Args:
        extra_server_args (str | None): The variant's ``EXTRA_WORLDPLAY_ARGS``
            payload (raw CLI string).

    Returns:
        str | None: A human-readable reason when droppable, else ``None``.
    """
    args = extra_server_args or ""
    if not args.strip():
        return None
    flags = _RE_LONG_FLAG.findall(args)
    # Forbidden-but-accepted first — the clearer, more specific message.
    for f in flags:
        if f in _WORLDPLAY_FORBIDDEN_SERVER_FLAGS:
            return f"{f}: {_WORLDPLAY_FORBIDDEN_SERVER_FLAGS[f]}"
    # Value-level: ``--action_ckpt`` is accepted by the wrapper, but a
    # distilled/few-step ckpt value is a different model (50-step ar_model lock).
    _ckpt_m = _RE_ACTION_CKPT.search(args)
    if _ckpt_m and _ckpt_is_distilled(_ckpt_m.group(1)):
        return (
            f"--action_ckpt {_ckpt_m.group(1)}: distilled/few-step checkpoint is "
            "a different model (50-step ar_model is the locked workload spec)"
        )
    # Shape-level: a store_true flag carrying a value argparse-errors even though
    # the flag itself is accepted, so it never reaches the unknown-flag check.
    _bool_m = _RE_STORE_TRUE_VALUE.search(args)
    if _bool_m:
        flag, value = _bool_m.group(1), _bool_m.group(2)
        env = "WORLDPLAY_USE_TORCH_COMPILE" if flag == "--enable_torch_compile" else "WORLDPLAY_FEW_STEP"
        return (
            f"{flag} {value}: store_true flag takes no value (argparse error → "
            f"no_measurement); pass the bare {flag} to enable it, or toggle "
            f"{env}=0/1 instead"
        )
    unknown = sorted({f for f in flags if f not in _WORLDPLAY_ACCEPTED_SERVER_FLAGS})
    if unknown:
        return (
            f"{', '.join(unknown)} not accepted by the customer bench_fps.py CLI "
            "(unknown flag → argparse error → no_measurement); propose WORLDPLAY_* "
            "env knobs or the accepted tunables (--enable_torch_compile / "
            "--group_offloading / --offloading / --transformer_resident_ar_rollout)"
        )
    return None


def worldplay_blacklist_reason(
    extra_envs: dict[str, str] | None,
) -> str | None:
    """Return a drop reason if a variant trips the WorldPlay do-not-set blacklist.

    Args:
        extra_envs (dict[str, str] | None): The variant's env overrides.

    Returns:
        str | None: A human-readable reason when blacklisted, else ``None``.
    """
    envs = {str(k): str(v) for k, v in (extra_envs or {}).items()}
    for key, (bad_values, reason) in _WORLDPLAY_ENV_BLACKLIST.items():
        if key not in envs:
            continue
        val = envs[key]
        if "*" in bad_values:
            if is_truthy(val, default=True):
                return f"{key}={val}: {reason}"
        elif val.strip().lower() in {b.lower() for b in bad_values}:
            return f"{key}={val}: {reason}"
    for key, reason in _WORLDPLAY_ENV_MEASUREMENT_LOCK.items():
        if key in envs:
            return f"{key}={envs[key]}: {reason}"
    for keys, reason in _WORLDPLAY_ENV_COMBO_BLACKLIST:
        if all(k in envs and is_truthy(envs[k], default=True) for k in keys):
            return reason
    # Value-level action-ckpt lock: a distilled/few-step checkpoint is a
    # different model even if the variant declares 50 steps.
    ckpt = envs.get("WORLDPLAY_ACTION_CKPT", "")
    if _ckpt_is_distilled(ckpt):
        return (
            f"WORLDPLAY_ACTION_CKPT={ckpt}: distilled/few-step checkpoint is a "
            "different model (50-step ar_model is the locked workload spec)"
        )
    return None


# Per-framework cache for ``_probe_server_help_text`` (avoids a subprocess per
# variant). Empty results are NOT cached so a transient failure re-probes.
_HELP_TEXT_CACHE: dict[str, str] = {}

# Per-framework ``--help`` extraction commands. Each is a single-shot
# ``python3 -c <inline>`` so the probe's 10s timeout covers the import cost.
_HELP_PROBE_COMMANDS: dict[str, tuple[str, ...]] = {
    "sglang": (
        "python3",
        "-c",
        "from sglang.launch_server import parser; parser.print_help()",
    ),
    "vllm": (
        "python3",
        "-c",
        "from vllm.entrypoints.openai.api_server import make_arg_parser; make_arg_parser(None).print_help()",
    ),
    # atom exposes EngineArgs.add_cli_args (mirrors vLLM).
    "atom": (
        "python3",
        "-c",
        "import argparse; from atom.model_engine.arg_utils import EngineArgs; "
        "p = argparse.ArgumentParser(); EngineArgs.add_cli_args(p); "
        "p.print_help()",
    ),
}


def _probe_server_help_text(framework: str) -> str:
    """Best-effort fetch of ``<framework> --help`` text for flag validation.

    Supported: ``sglang``, ``vllm``, ``atom``; unknown values return ``""``.
    Returns ``""`` on ANY failure — callers MUST treat empty as "unknown" and
    fall through to NOT filtering. Empty results are NOT cached.

    Args:
        framework (str): Framework name; matched case-insensitively.

    Returns:
        str: The combined stdout+stderr ``--help`` text, or ``""`` on any
        failure or unknown framework.
    """
    fw = (framework or "").strip().lower()
    if fw in _HELP_TEXT_CACHE:
        return _HELP_TEXT_CACHE[fw]
    cmd = _HELP_PROBE_COMMANDS.get(fw)
    if cmd is None:
        return ""
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
    except Exception:  # noqa: BLE001 — best-effort, see docstring
        out = ""
    if out:
        _HELP_TEXT_CACHE[fw] = out
    return out


def _detect_model_class(model_path: str) -> tuple[bool, bool]:
    """Heuristic detect of (is_mla_model, is_moe_model) from model path.

    Lowercased substring match — a cheap check to skip an obviously-wrong
    variant before a 10-min doomed sglang restart. Misclassifications cost at
    most one restart. MLA: DeepSeek (V2/V3/R1), GLM-5, Kimi-K2; MoE: MLA set +
    Qwen3-MoE.

    Args:
        model_path (str): Model path/identifier; matched as a lowercased
            substring.

    Returns:
        tuple[bool, bool]: ``(is_mla_model, is_moe_model)``.
    """
    p = model_path.lower()
    mla_keys = ("glm-5", "glm5", "deepseek", "kimi-k2", "kimi_k2", "kimi")
    moe_keys = (
        "glm-5",
        "glm5",
        "deepseek-v2",
        "deepseek-v3",
        "deepseek-r1",
        "kimi",
        "qwen3-moe",
        "qwen3_moe",
        "mixtral",
    )
    is_mla = any(k in p for k in mla_keys)
    is_moe = any(k in p for k in moe_keys)
    return is_mla, is_moe


def apply_compatibility_filter(
    grid: list["GridVariant"],
) -> tuple[list["GridVariant"], list[dict]]:
    """Skip variants known to be incompatible with current model/sglang.

    Two dimensions, each conservative (assume compatible) on probe failure:
    model class (MLA / MoE flags dropped when ``$MODEL_PATH`` lacks the family
    keyword), and sglang version (flags absent from ``launch_server --help``
    dropped). Returns the ``(kept, dropped)`` shape of ``apply_user_skip_list``.

    Args:
        grid (list[GridVariant]): The candidate variants to filter.

    Returns:
        tuple[list[GridVariant], list[dict]]: ``(kept, dropped)`` where dropped
        entries carry ``name``/``source``/``reason``.
    """
    model_path = os.environ.get("MODEL_PATH", "")
    if model_path:
        is_mla, is_moe = _detect_model_class(model_path)
    else:
        # No MODEL_PATH set -> can't detect -> assume compatible.
        is_mla, is_moe = True, True

    # Live framework's --help text; defaults to sglang for fixtures/callers
    # that don't thread ``benchmark.framework``.
    fw = (os.environ.get("FRAMEWORK", "") or "sglang").strip().lower()
    help_text = _probe_server_help_text(fw)
    help_available = bool(help_text)

    is_xdit = fw == "xdit"
    is_worldplay = fw == "worldplay"

    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        args = v.extra_server_args or ""
        skip_reason: str | None = None
        # xDiT do-not-set blacklist (env-keyed; precision lock + known crashes).
        if is_xdit:
            skip_reason = xdit_blacklist_reason(getattr(v, "extra_envs", None))
        # WorldPlay do-not-set blacklist (precision / workload-spec / crash lock),
        # checked on BOTH surfaces: env overrides and the raw CLI (extra_server_args).
        elif is_worldplay:
            skip_reason = worldplay_blacklist_reason(
                getattr(v, "extra_envs", None)
            ) or worldplay_server_args_reason(getattr(v, "extra_server_args", None))
        if skip_reason:
            dropped.append(
                {
                    "name": v.name,
                    "source": "worldplay_blacklist" if is_worldplay else "xdit_blacklist",
                    "reason": skip_reason,
                }
            )
            continue
        for flag, required_class in _COMPATIBILITY_FLAG_RULES:
            if flag not in args:
                continue
            # Model-class predicate
            class_ok = (required_class == "mla" and is_mla) or (required_class == "moe" and is_moe)
            if not class_ok:
                skip_reason = (
                    f"{flag} requires {required_class.upper()} model; "
                    f"MODEL_PATH={model_path!r} not recognised as "
                    f"{required_class.upper()}-class"
                )
                break
            # Framework flag-support predicate (only when help is readable).
            if help_available and flag not in help_text:
                skip_reason = f"{flag} not present in `{fw} --help` output; current {fw} version likely too old"
                break
        if skip_reason:
            dropped.append(
                {
                    "name": v.name,
                    "source": "compatibility_filter",
                    "reason": skip_reason,
                }
            )
        else:
            kept.append(v)
    return kept, dropped


def apply_user_skip_list(
    grid: list["GridVariant"],
    *,
    skip_spec: str,
) -> tuple[list["GridVariant"], list[dict]]:
    """Drop variants whose name matches any pattern in ``skip_spec``.

    Returns ``(kept, dropped)`` where each dropped entry is
    ``{"name", "reason", "source"}`` with source=``"user_skip"``.

    Args:
        grid (list[GridVariant]): The candidate variants to filter.
        skip_spec (str): Comma/whitespace skip patterns (exact or fnmatch glob)
            matched against ``GridVariant.name``.

    Returns:
        tuple[list[GridVariant], list[dict]]: ``(kept, dropped)`` where dropped
        entries carry ``name``/``source``/``reason``.
    """
    patterns = _parse_skip_spec(skip_spec)
    if not patterns:
        return list(grid), []

    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        matched_pat: str | None = None
        for pat in patterns:
            # Exact name first (cheaper), then fnmatch for globs.
            if pat == v.name or _fnmatch.fnmatchcase(v.name, pat):
                matched_pat = pat
                break
        if matched_pat is None:
            kept.append(v)
            continue
        dropped.append(
            {
                "name": v.name,
                "source": "user_skip",
                "reason": f"matched SKIP_VARIANTS pattern '{matched_pat}'",
            }
        )
    return kept, dropped
