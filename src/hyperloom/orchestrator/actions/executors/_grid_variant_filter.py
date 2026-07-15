# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared helper for the ``explore`` executor's grid runs.

Takes a base Magpie YAML + a list of (name, extra_server_args, extra_envs)
variants, runs Magpie once per variant, parses ``benchmark_report.json``,
returns the winners.
"""

from __future__ import annotations

import fnmatch as _fnmatch
import logging
import os
import re
import subprocess
from typing import Any

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
        "approximate/quantized attention regresses on Ulysses>=4 "
        "(attention is ~2% of FLOPS at small tokens/GPU)",
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

def _is_truthy_env(value: Any) -> bool:
    """Return whether an env-string value is truthy (set and not 0/false/off).

    Args:
        value (Any): Candidate env value.

    Returns:
        bool: ``True`` when the value is set to a non-falsey token.
    """
    return str(value).strip().lower() not in ("", "0", "false", "off", "no")

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
            if _is_truthy_env(val):
                return f"{key}={val}: {reason}"
        elif val.strip().lower() in {b.lower() for b in bad_values}:
            return f"{key}={val}: {reason}"
    for keys, reason in _XDIT_ENV_COMBO_BLACKLIST:
        if all(k in envs and _is_truthy_env(envs[k]) for k in keys):
            return reason
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
