# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pre-flight variant filtering and ordering for the explore grid.

Operates purely on lists of ``GridVariant``: user skip patterns, multi-node
invalid-variant drops and prioritisation, aiter-MoE pinning, xDiT env
blacklisting, and model/framework compatibility filtering. The filter helpers
return ``(kept, dropped)``; nothing here renders YAML, launches Magpie, or
reads ``benchmark_report.json`` — that lives in :mod:`._grid_runner`.
"""

from __future__ import annotations

import fnmatch as _fnmatch
import json
import logging
import os
import re
import subprocess
import time

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
    aiter_pinned_off = "SGLANG_USE_AITER" in pins and not is_truthy(pins["SGLANG_USE_AITER"], default=True)
    if not aiter_pinned_off:
        return list(grid), []

    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        envs = {str(k): str(val) for k, val in (getattr(v, "extra_envs", None) or {}).items()}
        reenables_master = "SGLANG_USE_AITER" in envs and is_truthy(envs["SGLANG_USE_AITER"], default=False)
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
# Applied for the ``xdit`` framework only, by :func:`apply_compatibility_filter`.
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


_HELP_TEXT_CACHE: dict[str, str] = {}

# Framework -> monotonic deadline before which a failed probe is not retried.
_HELP_PROBE_FAILED_UNTIL: dict[str, float] = {}
_HELP_PROBE_RETRY_SEC: float = 300.0

# Per-framework ``--help`` extraction commands. Each is a single-shot
# ``python3 -c <inline>`` so the probe's 10s timeout covers the import cost.
# Argv tails; the interpreter is resolved per framework at call time.
_HELP_PROBE_COMMANDS: dict[str, tuple[str, ...]] = {
    "sglang": (
        "-c",
        "from sglang.launch_server import parser; parser.print_help()",
    ),
    "vllm": (
        "-c",
        "from vllm.entrypoints.openai.api_server import make_arg_parser; make_arg_parser(None).print_help()",
    ),
    # atom exposes EngineArgs.add_cli_args (mirrors vLLM).
    "atom": (
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
    fall through to NOT filtering.

    A failure is held off for ``_HELP_PROBE_RETRY_SEC`` rather than re-paying a
    ten-second import on every variant, and is logged once: a silently empty
    probe means the flag rules stopped running with nothing to say so.

    Args:
        framework (str): Framework name; matched case-insensitively.

    Returns:
        str: The combined stdout+stderr ``--help`` text, or ``""`` on any
        failure or unknown framework.
    """
    fw = (framework or "").strip().lower()
    if fw in _HELP_TEXT_CACHE:
        return _HELP_TEXT_CACHE[fw]
    argv_tail = _HELP_PROBE_COMMANDS.get(fw)
    if argv_tail is None:
        return ""
    expiry = _HELP_PROBE_FAILED_UNTIL.get(fw)
    if expiry is not None and time.monotonic() < expiry:
        return ""
    # Deferred: _grid_runner imports this module at module scope.
    from ._grid_runner import _resolve_probe_python

    try:
        interpreter = _resolve_probe_python(fw)
        proc = subprocess.run(
            [interpreter, *argv_tail],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Only a clean exit is help text. stderr on a failed run is a
        # traceback, and treating that as help makes every flag look absent,
        # which drops the variants carrying them rather than sparing them.
        out = (proc.stdout or "") + (proc.stderr or "") if proc.returncode == 0 else ""
        reason = f"exit={proc.returncode}"
    except Exception as exc:  # noqa: BLE001 — best-effort, see docstring
        out, reason = "", repr(exc)
    if out:
        _HELP_TEXT_CACHE[fw] = out
        return out
    if fw not in _HELP_PROBE_FAILED_UNTIL:
        log.warning(
            "compatibility probe for %s produced no help text (%s); flag-version drops are disabled for it",
            fw,
            reason,
        )
    _HELP_PROBE_FAILED_UNTIL[fw] = time.monotonic() + _HELP_PROBE_RETRY_SEC
    return ""


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
    *,
    framework: str,
    model_path: str,
) -> tuple[list["GridVariant"], list[dict]]:
    """Skip variants known to be incompatible with current model/framework.

    Three dimensions, each conservative (assume compatible) when it cannot
    tell: the xDiT do-not-set list, model class (MLA / MoE flags dropped when
    the model path lacks the family keyword), and framework version (flags
    absent from the server's ``--help`` dropped). Returns the ``(kept,
    dropped)`` shape of ``apply_user_skip_list``.

    Args:
        grid (list[GridVariant]): The candidate variants to filter.
        framework (str): Framework the grid will run against.
        model_path (str): Model the grid will run against; ``""`` means unknown,
            which keeps every model-class-gated flag.

    Returns:
        tuple[list[GridVariant], list[dict]]: ``(kept, dropped)`` where dropped
        entries carry ``name``/``source``/``reason``.
    """
    if model_path:
        is_mla, is_moe = _detect_model_class(model_path)
    else:
        # Cannot detect -> assume compatible.
        is_mla, is_moe = True, True

    fw = framework.strip().lower()
    help_text = _probe_server_help_text(fw)
    help_available = bool(help_text)

    is_xdit = fw == "xdit"

    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        args = v.extra_server_args or ""
        skip_reason: str | None = None
        # xDiT do-not-set blacklist (env-keyed; precision lock + known crashes).
        if is_xdit:
            skip_reason = xdit_blacklist_reason(getattr(v, "extra_envs", None))
        if skip_reason:
            dropped.append(
                {
                    "name": v.name,
                    "source": "xdit_blacklist",
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
