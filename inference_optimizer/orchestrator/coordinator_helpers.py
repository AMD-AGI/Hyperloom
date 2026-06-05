"""Pure, self-contained helpers used by the Coordinator.

These functions have no dependency on Coordinator state; they are kept
out of ``coordinator.py`` so the control-plane module stays focused on
the runtime loop. This module must not import ``coordinator`` (one-way
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


# Eight task.params fields that determine baseline (Magpie) behavior
# end-to-end; the self-loop guard fingerprints these to detect a proposal
# repeating the same params after the same failure mode fired N times.
# Tests override the threshold constant directly.
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
    (excerpt capped at 400 chars, at most ``max_entries`` rows) so the
    audit trail carries per-variant failure context the LLM can read
    before re-proposing the same variant.
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

    Reads workload-shape fields that affect ``best_config`` but live
    outside ``_collect_workload_tags`` (max_running_requests / max_num_seqs
    / chunked_prefill_enabled / enable_torch_compile / quant_scheme /
    workload_mode), parsed from ``benchmark.envs`` extra-args blobs and
    top-level ``benchmark`` fields. Read defensively — parse errors return
    ``{}`` and missing fields are simply absent.
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
    # Direct fields on benchmark — only present when operator added them.
    for src, dst in (
        ("workload_mode", "workload_mode"),
        ("quant_scheme",  "quant_scheme"),
    ):
        v = bm.get(src)
        if v not in (None, "", 0):
            out[dst] = v
    envs = bm.get("envs") if isinstance(bm.get("envs"), dict) else {}
    # Pick the framework-appropriate extra args blob.
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
    # Torch compile env can also live as a separate env var.
    if "enable_torch_compile" not in out:
        tc_env = envs.get("ENABLE_TORCH_COMPILE")
        if isinstance(tc_env, str):
            out["enable_torch_compile"] = tc_env.strip().lower() in (
                "1", "true", "yes", "on",
            )
    return out


def _baseline_params_fingerprint(params: dict[str, Any] | None) -> dict[str, Any]:
    """Project ``params`` to the keys that determine baseline behavior.

    Missing keys are recorded as ``None`` (absent vs explicit-null are
    indistinguishable, matching what the prompt sees). ``extra_envs`` is
    normalized to a sorted list of ``[key, value]`` pairs; all values are
    stringified so ordering doesn't affect equality.
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
    """Resolve the roofline watermark ratio (env-tunable, safe default).

    Reads ``$HYPERLOOM_ROOFLINE_WATERMARK_RATIO``; any unparseable or
    unsafe value (``<= 1.0`` would re-fire every tick) falls back to 1.10.
    """
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

    Explore variants usually record the *full* cumulative args for the
    kept stack layer; joining ``base + candidate`` when both are full
    stacks duplicates flags, so prefer the full stack and dedupe.
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


def _dedupe_extra_server_args(args_str: str) -> str:
    """Collapse repeated ``--flag value`` pairs into a unique launch string.

    SGLang/vLLM argparse is ``action="store"`` for almost every knob, so a
    repeated flag only honors its last value; ``final.extra_server_args``
    exists for dashboard/replay and should not show the same flag N times.
    Keep each flag once with its last value (first-seen order preserved).
    Flags in ``_MULTI_VALUE_SGLANG_FLAGS`` keep their multi-value runs; bare
    flags are deduped too.
    """
    if not args_str:
        return ""
    tokens = args_str.split()
    # last-wins-for-value, first-wins-for-order.
    pair_by_flag: dict[str, list[str]] = {}
    order: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            flag = t
            i += 1
            values: list[str] = []
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
            # Stray positional token; preserve as-is, in order.
            key = f"__positional_{len(order)}__"
            order.append(key)
            pair_by_flag[key] = [t]
            i += 1
    out: list[str] = []
    for k in order:
        out.extend(pair_by_flag[k])
    return " ".join(out)
