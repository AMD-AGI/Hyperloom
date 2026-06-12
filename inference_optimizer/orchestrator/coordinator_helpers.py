# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pure, self-contained helpers used by the Coordinator.

No dependency on Coordinator state; must not import ``coordinator`` (one-way
dependency).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _infer_model_class_from_config(model_path: str) -> str:
    """Infer a deterministic model_class from local model metadata."""
    import json
    raw_path = (model_path or "").strip()
    payload: dict[str, Any] = {}
    if raw_path:
        cfg = Path(raw_path) / "config.json"
        try:
            if cfg.is_file():
                data = json.loads(cfg.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    payload = data
        except Exception:  # noqa: BLE001 - best effort only.
            log.debug("model_class inference: failed to read %s", cfg, exc_info=True)

    text_parts: list[str] = [raw_path.lower()]
    arch = payload.get("architectures")
    if isinstance(arch, list):
        text_parts.extend(str(x).lower() for x in arch if x)
    elif arch:
        text_parts.append(str(arch).lower())
    for key in ("model_type", "attention_type", "attn_type"):
        if payload.get(key):
            text_parts.append(str(payload[key]).lower())
    text = " ".join(text_parts)

    def _positive_int(*keys: str) -> bool:
        """Whether any of the given payload keys holds a positive integer.

        Booleans are explicitly ignored (they are not treated as ints).

        Args:
            *keys: Payload keys to check.

        Returns:
            ``True`` if at least one key parses to an integer > 0.
        """
        for key in keys:
            val = payload.get(key)
            if isinstance(val, bool):
                continue
            try:
                if val is not None and int(val) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    is_moe = (
        _positive_int(
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
            "moe_num_experts",
        )
        or any(k in text for k in (
            "moe", "mixtral", "deepseek-v2", "deepseek-v3", "deepseek-r1",
            "kimi", "glm-5", "glm5",
        ))
    )
    is_mla = any(k in text for k in (
        "mla", "multi-head latent", "deepseek", "kimi", "glm-5", "glm5",
    ))
    is_nsa = any(k in text for k in (
        "nsa", "native sparse attention", "glm-5", "glm5",
    ))
    if is_moe and is_mla and is_nsa:
        return "moe_mla_nsa"
    if is_moe and is_mla:
        return "moe_mla"
    if is_moe:
        return "moe_swa"
    return "dense"


# task.params fields fingerprinted by the self-loop guard to detect a
# proposal repeating the same params after the same failure mode.
_BASELINE_FINGERPRINT_KEYS: tuple[str, ...] = (
    "benchmark_script",
    "result_dir",
    "extra_server_args",
    "extra_envs",
    "model_path",
    "gpu_type",
    "config_path",
    "disable_run_eval",
)
_BASELINE_SELF_LOOP_THRESHOLD: int = 2

# Flags whose argparse consumes multiple bare tokens before the next ``--``.
_MULTI_VALUE_SGLANG_FLAGS: frozenset[str] = frozenset({
    "--cuda-graph-bs",
    "--cuda-graph-max-bs",
})

_DEFAULT_ROOFLINE_WATERMARK_RATIO: float = 1.10  # 10% step over last roofline
_ROOFLINE_WATERMARK_RATIO_ENV: str = "HYPERLOOM_ROOFLINE_WATERMARK_RATIO"


def effective_closing_grace_sec(
    max_minutes: float | None,
    closing_grace_sec: float | None,
) -> float:
    """Resolve the closing-phase grace window after the wall-clock deadline.

    Explicit ``closing_grace_sec`` (including ``0`` to disable) wins;
    otherwise default to ``min(120, max_minutes * 60 * 0.02)``.
    """
    if closing_grace_sec is not None:
        return float(closing_grace_sec)
    return min(120.0, (max_minutes or 0.0) * 60.0 * 0.02)


def _parse_iso_unix(ts: str) -> float:
    """Parse an ISO 8601 UTC timestamp into unix seconds; ``0.0`` on failure.

    Naive timestamps are treated as UTC. Never raises.
    """
    s = (ts or "").strip()
    if not s:
        return 0.0
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _summarize_failed_variants(
    all_results: Any, *, max_entries: int = 10,
) -> list[dict[str, Any]]:
    """Project ``status=='failed'`` grid_runner rows into a compact list.

    Returns ``[{name, error_class, error_excerpt, extra_server_args}, ...]``
    (excerpt capped at 400 chars, at most ``max_entries`` rows).
    """
    if not isinstance(all_results, list):
        return []
    failed: list[dict[str, Any]] = []
    for row in all_results:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "") != "failed":
            continue
        err = str(row.get("error") or "")
        failed.append({
            "name": str(row.get("name") or ""),
            "error_class": str(row.get("error_class") or "") or None,
            "error_excerpt": err[:400] if err else None,
            "extra_server_args": str(row.get("extra_server_args") or ""),
        })
        if len(failed) >= max_entries:
            break
    return failed


def _parse_baseline_workload_extra(yaml_path: str) -> dict[str, Any]:
    """Extract KB workload-tag fields from a baseline-materialized Magpie YAML.

    Reads workload-shape fields outside ``_collect_workload_tags`` from
    ``benchmark.envs`` extra-args blobs and top-level ``benchmark`` fields.
    Defensive — parse errors return ``{}``.
    """
    import yaml as _yaml
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
    except (OSError, _yaml.YAMLError):
        return {}
    out: dict[str, Any] = {}
    bm = cfg.get("benchmark") if isinstance(cfg, dict) else None
    if not isinstance(bm, dict):
        return out
    for src, dst in (
        ("workload_mode", "workload_mode"),
        ("quant_scheme",  "quant_scheme"),
    ):
        v = bm.get(src)
        if v not in (None, "", 0):
            out[dst] = v
    envs = bm.get("envs") if isinstance(bm.get("envs"), dict) else {}
    extra_args_str = ""
    for env_key in ("EXTRA_SGLANG_ARGS", "EXTRA_VLLM_ARGS"):
        v = envs.get(env_key)
        if isinstance(v, str) and v.strip():
            extra_args_str = v.strip()
            break
    tokens = extra_args_str.split() if extra_args_str else []
    for i, tok in enumerate(tokens):
        if tok in ("--max-running-requests",) and i + 1 < len(tokens):
            try:
                out["max_running_requests"] = int(tokens[i + 1])
            except ValueError:
                pass
        elif tok in ("--max-num-seqs",) and i + 1 < len(tokens):
            try:
                out["max_num_seqs"] = int(tokens[i + 1])
            except ValueError:
                pass
        elif tok == "--enable-chunked-prefill":
            out["chunked_prefill_enabled"] = True
        elif tok == "--disable-chunked-prefill":
            out["chunked_prefill_enabled"] = False
        elif tok == "--enable-torch-compile":
            out["enable_torch_compile"] = True
    # Torch compile may also be a separate env var.
    if "enable_torch_compile" not in out:
        tc_env = envs.get("ENABLE_TORCH_COMPILE")
        if isinstance(tc_env, str):
            out["enable_torch_compile"] = tc_env.strip().lower() in (
                "1", "true", "yes", "on",
            )
    return out


def _baseline_params_fingerprint(params: dict[str, Any] | None) -> dict[str, Any]:
    """Project ``params`` to the keys that determine baseline behavior.

    Missing keys recorded as ``None``; ``extra_envs`` normalized to a sorted
    list of stringified ``[key, value]`` pairs so ordering doesn't affect
    equality.
    """
    params = params or {}
    out: dict[str, Any] = {}
    for key in _BASELINE_FINGERPRINT_KEYS:
        if key == "extra_envs":
            envs = params.get(key) or {}
            if isinstance(envs, dict):
                out[key] = sorted(
                    [str(k), str(v)] for k, v in envs.items()
                )
            else:
                out[key] = None
            continue
        value = params.get(key)
        out[key] = None if value is None else str(value)
    return out


def _resolve_roofline_watermark_ratio() -> float:
    """Resolve the roofline watermark ratio from ``$HYPERLOOM_ROOFLINE_WATERMARK_RATIO`` (fallback 1.10)."""
    raw = (os.environ.get(_ROOFLINE_WATERMARK_RATIO_ENV) or "").strip()
    if not raw:
        return _DEFAULT_ROOFLINE_WATERMARK_RATIO
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_ROOFLINE_WATERMARK_RATIO
    if val <= 1.0:
        return _DEFAULT_ROOFLINE_WATERMARK_RATIO
    return val


def _merge_cumulative_extra_sglang_args(
    base_args: str,
    candidate_args: str,
    full_args: str,
) -> str:
    """Build cumulative launch args for a KEEP without double-stacking.

    Prefer the full stack and dedupe, since joining ``base + candidate``
    when both are full stacks duplicates flags.
    """
    base = str(base_args or "").strip()
    candidate = str(candidate_args or "").strip()
    full = str(full_args or "").strip()
    if full and full != candidate:
        merged = full
    elif candidate and base:
        if candidate.startswith(base) or base in candidate.split():
            merged = candidate
        else:
            merged = f"{base} {candidate}".strip()
    else:
        merged = candidate or full or base
    return _dedupe_extra_server_args(merged)


_SPACE_VALUE_FLAGS = (
    "--json-model-override-args",
    "--override-generation-config",
    "--tool-call-parser",
)


def _dedupe_extra_server_args(args_str: str) -> str:
    """Collapse repeated ``--flag value`` pairs into a unique launch string.

    Keep each flag once with its last value (first-seen order preserved),
    since argparse ``action="store"`` only honors the last value. Flags in
    ``_MULTI_VALUE_SGLANG_FLAGS`` keep their multi-value runs. Args with
    JSON/space-valued flags are left untouched because downstream launch
    scripts expand them unquoted.
    """
    if not args_str:
        return ""
    if any(flag in args_str for flag in _SPACE_VALUE_FLAGS):
        return args_str
    tokens = args_str.split()
    pair_by_flag: dict[str, list[str]] = {}
    order: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            if "=" in t:
                flag, _, value = t.partition("=")
                values = [value] if value else []
                i += 1
            else:
                flag = t
                i += 1
                values = []
                if flag in _MULTI_VALUE_SGLANG_FLAGS:
                    while i < len(tokens) and not tokens[i].startswith("--"):
                        values.append(tokens[i])
                        i += 1
                elif i < len(tokens) and not tokens[i].startswith("--"):
                    values = [tokens[i]]
                    i += 1
            pair = [flag, *values] if values else [flag]
            if flag not in pair_by_flag:
                order.append(flag)
            pair_by_flag[flag] = pair
        else:
            # Stray positional token; preserve as-is.
            key = f"__positional_{len(order)}__"
            order.append(key)
            pair_by_flag[key] = [t]
            i += 1
    out: list[str] = []
    for k in order:
        out.extend(pair_by_flag[k])
    return " ".join(out)
