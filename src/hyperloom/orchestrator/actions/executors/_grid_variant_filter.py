# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pre-flight variant filtering and ordering for the explore grid."""

from __future__ import annotations

import fnmatch as _fnmatch
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from hyperloom.common.env import is_truthy
from hyperloom.common.gpu_identity import is_gfx_arch

from ._grid_base import (
    GridVariant,
)
from hyperloom.inference_optimizer.grid_server_args import compose_server_args

log = logging.getLogger(__name__)


def resolve_skip_spec(params: dict | None) -> str:
    """Resolve the active skip spec from task params + process env."""
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
    """Split ``spec`` on commas and whitespace; drop empties."""
    if not spec:
        return []
    out: list[str] = []
    for token in spec.replace("\n", ",").split(","):
        for sub in token.split():
            t = sub.strip()
            if t:
                out.append(t)
    return out


# Matches ``--cuda-graph-max-bs 64`` and ``--cuda_graph_max_bs=64``; captures the integer value.
_RE_CUDA_GRAPH_MAX_BS = re.compile(r"--cuda[-_]graph[-_]max[-_]bs[= ]+(\d+)")

# Multi-node grid prioritisation + invalid-variant filtering.
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
    """Return the rank of ``variant`` against ``priority_tags`` (lower = first)."""
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
    """Stable-sort ``grid`` so likely multi-node winners run first."""
    from ._multi_node_env import is_multi_node

    if not is_multi_node():
        return grid
    # ``sorted`` is stable, so ties keep input order.
    return sorted(grid, key=lambda v: _mn_priority_index(v, priority_tags))


def apply_multi_node_invalid_variants(
    grid: list["GridVariant"],
) -> tuple[list["GridVariant"], list[dict]]:
    """Drop variants that are known regressions/invalid on multi-node fabrics."""
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
        if m and int(m.group(1)) < conc:
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
    """Return the operator's ``--extra-env`` pins from the CLI handoff env."""
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
    """Drop variants that re-enable the aiter MoE runner when it is pinned off."""
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


# Framework / hardware compatibility filter: each entry maps an ``extra_server_args`` substring to a required model
# class.
_COMPATIBILITY_FLAG_RULES: tuple[tuple[str, str], ...] = (
    ("--enable-flashinfer-mla", "mla"),
    ("--enable-deepep-moe", "moe"),
    ("--enable-ep-moe", "moe"),
)

# xDiT (diffusion) do-not-set blacklist — env knobs that crash or regress on FLUX.2-class DiT models with Ulysses SP.
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
    """Return a drop reason if a variant trips the xDiT do-not-set blacklist."""
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


# Framework -> (parser identity, help text). Keyed on identity rather than kept
# for the life of the process: a reinstalled framework must not be judged by the
# parser its predecessor printed.
_HELP_TEXT_CACHE: dict[str, tuple[str, str]] = {}

# Framework -> (parser identity, failed-probe retry deadline).
_HELP_PROBE_FAILURES: dict[str, tuple[str, float]] = {}
_HELP_PROBE_RETRY_SEC: float = 300.0
# Importing a serving framework to read its parser costs seconds (sglang: ~4s warm,
# more on a cold pod), and the probe is a blocking call inside an async executor.
_HELP_PROBE_TIMEOUT_SEC: float = 30.0

# Per-framework ``--help`` argv tails; resolve the interpreter at call time.
_HELP_PROBE_COMMANDS: dict[str, tuple[str, ...]] = {
    # Both build a parser and hand it to the framework's own registrar, the shape
    # `atom` already used: neither exposes a ready-made parser at module scope.
    "sglang": (
        "-c",
        "import argparse; from sglang.srt.server_args import ServerArgs; "
        "p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p); "
        "p.print_help()",
    ),
    "vllm": (
        "-c",
        "from vllm.entrypoints.openai.cli_args import make_arg_parser; "
        "from vllm.utils.argparse_utils import FlexibleArgumentParser; "
        "make_arg_parser(FlexibleArgumentParser()).print_help()",
    ),
    # atom exposes EngineArgs.add_cli_args (mirrors vLLM).
    "atom": (
        "-c",
        "import argparse; from atom.model_engine.arg_utils import EngineArgs; "
        "p = argparse.ArgumentParser(); EngineArgs.add_cli_args(p); "
        "p.print_help()",
    ),
}


def _framework_package_stamp(interpreter: str, fw: str) -> list[str | int] | None:
    """Stat the framework package the probe would import, or None if not locatable.

    Read off the filesystem, never imported: the parser lives in the probe
    interpreter's environment, not this process's, and importing a serving
    framework here is exactly the cost the cache exists to avoid.
    """
    try:
        root = Path(interpreter).resolve().parent.parent
        for init in sorted(root.glob(f"lib/python*/site-packages/{fw}/__init__.py")):
            stat = init.stat()
            return [str(init), stat.st_size, stat.st_mtime_ns]
    except OSError:
        return None
    return None


def _help_probe_launch_identity(
    interpreter: str,
    argv_tail: tuple[str, ...],
    package_stamp: list[str | int] | None,
) -> str:
    """Identify the parser a probe would print, without importing it here.

    The executable, its stat and the installed package's stat cover what changes
    the output: a rebuilt venv, a reinstalled framework, a different interpreter
    resolved out of the environment. Ambient env vars are deliberately absent --
    per-round tuning overrides rewrite them constantly, and folding those in
    would expire both the cache and the cooldown on every round.
    """
    executable = Path(shutil.which(interpreter) or interpreter)
    try:
        stat = executable.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_mode)
    except OSError:
        stamp = None
    payload = (interpreter, str(executable.resolve()), stamp, argv_tail, package_stamp)
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def _probe_server_help_text(framework: str) -> str:
    """Return the installed parser's help, reusing it while that parser is unchanged.

    Every step runs under one handler: this is a best-effort predicate, and a
    probe that cannot answer must leave the variant alone rather than fail the
    explore task around it.
    """
    fw = (framework or "").strip().lower()
    argv_tail = _HELP_PROBE_COMMANDS.get(fw)
    if argv_tail is None:
        return ""
    identity = ""
    try:
        from ._benchmark_interpreter import _resolve_probe_python

        interpreter = _resolve_probe_python(fw)
        package_stamp = _framework_package_stamp(interpreter, fw)
        identity = _help_probe_launch_identity(interpreter, argv_tail, package_stamp)
        cached = _HELP_TEXT_CACHE.get(fw)
        if cached is not None and cached[0] == identity:
            return cached[1]
        failure = _HELP_PROBE_FAILURES.get(fw)
        if failure is not None and failure[0] == identity and time.monotonic() < failure[1]:
            return ""
        proc = subprocess.run(
            [interpreter, *argv_tail],
            capture_output=True,
            text=True,
            timeout=_HELP_PROBE_TIMEOUT_SEC,
        )
        # Failed stdout/stderr can contain tracebacks, not supported flags.
        out = (proc.stdout or "") + (proc.stderr or "") if proc.returncode == 0 else ""
        if out:
            _HELP_PROBE_FAILURES.pop(fw, None)
            if package_stamp is not None:
                # Without a package to watch, a cached help text has no event that
                # would retire it, which is the staleness this cache had before.
                _HELP_TEXT_CACHE[fw] = (identity, out)
            return out
        reason = f"exit={proc.returncode}: {' '.join((proc.stderr or '').split())[-300:]}"
    except Exception as exc:  # noqa: BLE001 — see docstring: the filter degrades, it does not fail
        reason = repr(exc)
    previous = _HELP_PROBE_FAILURES.get(fw)
    already_reported = previous is not None and previous[0] == identity
    _HELP_PROBE_FAILURES[fw] = (identity, time.monotonic() + _HELP_PROBE_RETRY_SEC)
    if not already_reported:
        log.warning(
            "compatibility probe for %s produced no help text (%s); flag-version drops are disabled for it",
            fw,
            reason,
        )
    return ""


def _detect_model_class(model_path: str) -> tuple[bool, bool]:
    """Heuristic detect of (is_mla_model, is_moe_model) from model path."""
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


_UNSAFE_UNIFIED_ATTN_STACK = {
    "sglang": "0.5.20.dev20260920+gc610c40399",
    "aiter": "4ad99832823dde2315b361cbd3b54b1c5c12acd5",
    "rocm": "10.0.0",
}
_UNSAFE_UNIFIED_ATTN_REASON = "SGLANG_USE_AITER_UNIFIED_ATTN=1 is unsafe on the exact ROCm 10 Qwen3-14B-FP8 stack"
_FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
_DIST_VERSION_SHA_RE = re.compile(r"\+g([0-9a-f]{7,})")


def _fingerprint_component_matches(actual: str, expected: str) -> bool:
    """Match one stack-fingerprint component against a pinned commit, accepting the git-describe dist version a
    package reports when no explicit commit env var was exported.
    """
    if actual == expected:
        return True
    if not _FULL_SHA_RE.fullmatch(expected):
        return False
    found = _DIST_VERSION_SHA_RE.search(actual)
    return found is not None and expected.startswith(found.group(1))


def _matches_unsafe_unified_attn_stack(
    *,
    framework: str,
    model_path: str,
    gpu_type: str,
    stack_fingerprint: dict | None,
) -> bool:
    if framework.strip().lower() != "sglang" or not is_gfx_arch(gpu_type, "gfx950"):
        return False
    stack = stack_fingerprint if isinstance(stack_fingerprint, dict) else {}
    if any(
        not _fingerprint_component_matches(str(stack.get(key) or ""), value)
        for key, value in _UNSAFE_UNIFIED_ATTN_STACK.items()
    ):
        return False
    model_dir = Path(model_path)
    if "14b" not in model_dir.name.lower() or "fp8" not in model_dir.name.lower():
        return False
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(config, dict):
        return False
    quant = config.get("quantization_config") or {}
    if not isinstance(quant, dict):
        return False
    return (
        config.get("architectures") == ["Qwen3ForCausalLM"]
        and str(config.get("model_type") or "").lower() == "qwen3"
        and str(config.get("torch_dtype") or "").lower() in {"bf16", "bfloat16"}
        and config.get("head_dim") == 128
        and config.get("num_attention_heads") == 40
        and config.get("num_key_value_heads") == 8
        and str(quant.get("quant_method") or "").lower() == "fp8"
    )


def _last_server_arg(args: str, flags: tuple[str, ...], default: str) -> str:
    try:
        tokens = shlex.split(args)
    except ValueError:
        return ""
    value = default
    for idx, token in enumerate(tokens):
        for flag in flags:
            if token == flag and idx + 1 < len(tokens):
                value = tokens[idx + 1]
            elif token.startswith(f"{flag}="):
                value = token.split("=", 1)[1]
    return value


def _unsafe_unified_attn_reason(variant: GridVariant, *, base_server_args: str) -> str | None:
    envs = getattr(variant, "extra_envs", None) or {}
    if not is_truthy(envs.get("SGLANG_USE_AITER_UNIFIED_ATTN"), default=False):
        return None
    effective_args = compose_server_args(
        base_extra_args=base_server_args,
        variant_extra_args=variant.extra_server_args,
        remove_args=variant.remove_args,
        args_mode=variant.args_mode,
    )
    page_size = _last_server_arg(effective_args, ("--page-size", "--page_size"), "1")
    kv_dtype = _last_server_arg(
        effective_args,
        ("--kv-cache-dtype", "--kv_cache_dtype"),
        "auto",
    ).lower()
    if page_size == "1" and kv_dtype in {"auto", "bf16", "bfloat16"}:
        return _UNSAFE_UNIFIED_ATTN_REASON
    return None


def apply_compatibility_filter(
    grid: list["GridVariant"],
    *,
    framework: str,
    model_path: str,
    gpu_type: str = "",
    stack_fingerprint: dict | None = None,
    base_server_args: str = "",
) -> tuple[list["GridVariant"], list[dict]]:
    """Skip variants known to be incompatible with current model/framework."""
    if model_path:
        is_mla, is_moe = _detect_model_class(model_path)
    else:
        # Cannot detect -> assume compatible.
        is_mla, is_moe = True, True

    fw = framework.strip().lower()
    # The probe forks an interpreter that imports the framework's parser, so it is paid only
    # once a rule's help check is actually reached. ``None`` means "not probed in this call yet".
    help_text: str | None = None

    is_xdit = fw == "xdit"
    unsafe_unified_attn_stack = _matches_unsafe_unified_attn_stack(
        framework=fw,
        model_path=model_path,
        gpu_type=gpu_type,
        stack_fingerprint=stack_fingerprint,
    )

    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        args = v.extra_server_args or ""
        skip_reason: str | None = None
        if unsafe_unified_attn_stack:
            skip_reason = _unsafe_unified_attn_reason(v, base_server_args=base_server_args)
        if skip_reason:
            dropped.append(
                {
                    "name": v.name,
                    "source": "compatibility_filter",
                    "reason": skip_reason,
                }
            )
            continue
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
            if help_text is None:
                help_text = _probe_server_help_text(fw)
            if help_text and flag not in help_text:
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
    """Drop variants whose name matches any pattern in ``skip_spec``."""
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
