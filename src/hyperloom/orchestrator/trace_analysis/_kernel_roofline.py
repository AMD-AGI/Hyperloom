# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The per-kernel roofline payload: its fields, the model parameters behind it, and the runtime metadata it carries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hyperloom.common.gpu_identity import gfx_arch_for_gpu_type

from ._kernel_classification import _FLYDSL_SCAN_BYTES

# Canonical roofline_source provenance enum, shared with the bypass route so both
# emit the field from one vocabulary (see _roofline_source for the value ladder).
from ._roofline_source import ANALYTICAL as _RL_ANALYTICAL, PLACEHOLDER as _RL_PLACEHOLDER
from ._trace_shapes import _dtypes_from_shapes, _shape_call_entries
from hyperloom.inference_optimizer.framework_registry import tracelens_analysis_mode


def default_roofline_fields(candidates: list[dict[str, Any]]) -> None:
    """Give every candidate row the roofline fields it does not already carry, as unknown/None placeholders."""
    for item in candidates:
        if not isinstance(item, dict):
            continue
        item.setdefault("bottleneck", "unknown")
        item.setdefault("arithmetic_intensity", None)
        item.setdefault("compute_utilization_pct", None)
        item.setdefault("bandwidth_utilization_pct", None)
        item.setdefault("recommended_actions", [])


def _first_non_empty(*values: Any) -> Any:
    """Return the first argument that is neither ``None`` nor empty string.

    Args:
        *values (Any): Candidate values in priority order.

    Returns:
        Any: The first value that is not ``None`` or ``""``, else ``None``.
    """
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _kernel_roofline_row(candidate: dict[str, Any]) -> dict[str, Any]:
    """Project one hot-kernel candidate into the kernel-roofline view.

    Args:
        candidate (dict[str, Any]): A finalized hot-kernel candidate row.

    Returns:
        dict[str, Any]: A flattened roofline row with identity, cost, and
            (possibly ``None``) utilization/bottleneck fields.
    """
    arithmetic_intensity = _first_non_empty(
        candidate.get("arithmetic_intensity"),
        candidate.get("flops_per_byte"),
    )
    return {
        "kernel_id": candidate.get("kernel_id"),
        "name": candidate.get("name"),
        "gpu_pct": candidate.get("gpu_pct"),
        "duration_us": candidate.get("duration_us"),
        "call_count": candidate.get("call_count"),
        "kernel_category": candidate.get("kernel_category"),
        "source_file": candidate.get("source_file"),
        "bottleneck": _first_non_empty(
            candidate.get("bottleneck"),
            candidate.get("bound_type"),
        ),
        "bound_type": candidate.get("bound_type"),
        "arithmetic_intensity": arithmetic_intensity,
        "flops_per_byte": candidate.get("flops_per_byte"),
        "efficiency_percent": candidate.get("efficiency_percent"),
        "compute_utilization_pct": candidate.get("compute_utilization_pct"),
        "bandwidth_utilization_pct": candidate.get("bandwidth_utilization_pct"),
        "suggestion": candidate.get("suggestion") or "",
        "roofline_name": candidate.get("roofline_name"),
        "recommended_actions": list(candidate.get("recommended_actions") or []),
        "reusable_native_kernel": bool(candidate.get("reusable_native_kernel")),
        # Contract alignment (F6): emit the shared roofline_source provenance enum
        # so both routes carry it. TraceLens rows come from its per-op perf model
        # (analytical) when that model produced numbers, else placeholder.
        "roofline_source": _RL_ANALYTICAL
        if any(
            candidate.get(k) is not None
            for k in (
                "arithmetic_intensity",
                "flops_per_byte",
                "efficiency_percent",
                "compute_utilization_pct",
                "bandwidth_utilization_pct",
            )
        )
        else _RL_PLACEHOLDER,
    }


def build_kernel_roofline_payload(
    *,
    trace_input: str,
    trace_input_type: str,
    analysis_md_path: str,
    kernel_candidates_path: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the per-kernel roofline sidecar payload.

    A view over the candidates; missing counters stay null.

    Args:
        trace_input: The trace input path recorded in the payload.
        trace_input_type: Whether ``trace_input`` is a file or capture dir.
        analysis_md_path: Path to the source analysis report.
        kernel_candidates_path: Path to the candidates JSON.
        candidates: The finalized hot-kernel candidate rows.

    Returns:
        The kernel-roofline sidecar payload dict.
    """
    rows = [_kernel_roofline_row(candidate) for candidate in candidates if isinstance(candidate, dict)]
    return {
        "schema_version": 1,
        "source": "tracelens_analysis",
        "trace_input": trace_input,
        "trace_input_type": trace_input_type,
        "analysis_md_path": analysis_md_path,
        "kernel_candidates_path": kernel_candidates_path,
        "kernels": rows,
    }


def kernel_roofline_path_for_run(run_dir: Path, filename: str = "kernel_roofline.json") -> Path:
    """Return the stable session-level kernel roofline report path."""
    try:
        session_sub = run_dir.parent
        runs_dir = session_sub.parent
        kernel_agent_dir = runs_dir.parent
    except (IndexError, AttributeError):
        return run_dir / "reports" / filename
    if runs_dir.name == "runs" and kernel_agent_dir.name == "kernel-agent":
        session_dir = kernel_agent_dir.parent
        return session_dir / "reports" / filename
    # Backward-compatible flat run-dir layout.
    try:
        runs_dir_legacy = run_dir.parent
        kernel_agent_dir_legacy = runs_dir_legacy.parent
    except (IndexError, AttributeError):
        return run_dir / "reports" / filename
    if runs_dir_legacy.name == "runs" and kernel_agent_dir_legacy.name == "kernel-agent":
        session_dir = kernel_agent_dir_legacy.parent
        return session_dir / "reports" / filename
    return run_dir / "reports" / filename


def _candidate_model_config_paths(model_name: str) -> list[Path]:
    """Enumerate candidate ``config.json`` paths for a model name/path.

    Considers the value as a direct JSON path, a directory containing
    ``config.json``, and locations under ``$HYPERLOOM_MODELS_ROOT``.

    Args:
        model_name (str): A model name or filesystem path.

    Returns:
        list[Path]: Deduplicated candidate config paths in priority order;
            empty when ``model_name`` is blank.
    """
    text = str(model_name or "").strip()
    if not text:
        return []
    raw = Path(text).expanduser()
    candidates: list[Path] = []
    if raw.suffix == ".json":
        candidates.append(raw)
    candidates.append(raw / "config.json")
    # Optional models-root override (env-only; no hardcoded default). Standalone
    # kernel-agent tools cannot import hyperloom.common (the Ray/subprocess
    # sys.path contract in hyperloom.common.__init__), so this mirrors the shared
    # resolver's strategy (local path + HF hub cache) independently.
    _root = os.environ.get("HYPERLOOM_MODELS_ROOT", "").strip()
    if _root:
        candidates.append(Path(_root) / text / "config.json")
        candidates.append(Path(_root) / raw.name / "config.json")
    # HF hub cache (what vLLM/SGLang populate when given a repo id): the snapshot
    # commit-hash segment is not derivable by string, so let huggingface_hub
    # locate it (honoring HF_HOME / HF_HUB_CACHE). Best-effort; skipped when the
    # optional dep is absent or nothing is cached.
    try:
        from huggingface_hub import try_to_load_from_cache

        _hit = try_to_load_from_cache(repo_id=text, filename="config.json")
        if isinstance(_hit, str):
            candidates.append(Path(_hit))
    except (ImportError, OSError, ValueError):
        # No hub, no cache, or an unreadable one: the local candidates above still stand.
        pass
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def load_model_kernel_params(model_name: str) -> dict[str, Any]:
    """Read HF config.json and return attention parameters relevant to GEAK.

    Args:
        model_name (str): A model name or path used to locate ``config.json``.

    Returns:
        dict[str, Any]: Attention/MLA params (e.g. ``HEAD_SIZE``,
            ``NUM_ATTENTION_HEADS``) plus ``MODEL_CONFIG_PATH``; empty when no
            readable config is found.
    """
    for config_path in _candidate_model_config_paths(model_name):
        if not config_path.is_file():
            continue
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        params: dict[str, Any] = {
            "MODEL_CONFIG_PATH": str(config_path),
        }
        has_mla_dims = any(cfg.get(key) is not None for key in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
        if cfg.get("head_dim") is not None:
            params["HEAD_SIZE"] = cfg.get("head_dim")
        elif not has_mla_dims:
            hidden = cfg.get("hidden_size")
            heads = cfg.get("num_attention_heads")
            if isinstance(hidden, int) and isinstance(heads, int) and heads > 0 and hidden % heads == 0:
                params["HEAD_SIZE"] = hidden // heads
        for src, dst in (
            ("qk_nope_head_dim", "QK_NOPE_HEAD_DIM"),
            ("qk_rope_head_dim", "QK_ROPE_HEAD_DIM"),
            ("v_head_dim", "V_HEAD_DIM"),
            ("kv_lora_rank", "KV_LORA_RANK"),
            ("num_attention_heads", "NUM_ATTENTION_HEADS"),
            ("num_key_value_heads", "NUM_KEY_VALUE_HEADS"),
            ("hidden_size", "HIDDEN_SIZE"),
        ):
            if cfg.get(src) is not None:
                params[dst] = cfg[src]
        return params
    return {}


_FLYDSL_SMEM_MARKERS = ("SmemAllocator", "SmemPtr", "smem_alloc")

_FLYDSL_BUFFER_LOAD_MARKERS = (
    "make_buffer_tensor",
    "BufferCopy",
    "rocdl",
    "buffer_load",
)


def _resolve_flydsl_source_fallback() -> str:
    """Resolve the real FlyDSL MoE kernel source for synthetic pseudo-ops.

    Used for FlyDSL pseudo-ops, looking for
    ``$DSL2_ROOT/kernels/moe_gemm_2stage.py`` and known fallback roots.

    Returns:
        The first existing FlyDSL MoE kernel source path, or ``""`` when none
        exists.
    """
    roots = [
        os.environ.get("DSL2_ROOT", "").strip(),
        os.environ.get("FLYDSL_ROOT", "").strip(),
        "/opt/FlyDSL",
        "/sgl-workspace/flydsl",
    ]
    for root in roots:
        if not root:
            continue
        cand = os.path.join(root, "kernels", "moe_gemm_2stage.py")
        if os.path.isfile(cand):
            return cand
    return ""


def _flydsl_kernel_params(
    source_file: str,
    target_platform: str,
) -> dict[str, Any]:
    """Build FlyDSL-specific kernel params.

    Captures target arch, JIT cache state, and smem/buffer-load usage;
    best-effort and never raises.

    Args:
        source_file: The FlyDSL kernel source path.
        target_platform: The target GPU platform name.

    Returns:
        The FlyDSL kernel-params dict (possibly partial).
    """
    params: dict[str, Any] = {}
    arch = gfx_arch_for_gpu_type(target_platform)
    if arch:
        params["FLYDSL_TARGET_ARCH"] = arch
    cache_dir = os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", "").strip()
    if cache_dir:
        params["FLYDSL_AUTOTUNE_CACHE_DIR"] = cache_dir
    enable_cache = os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE", "").strip()
    if enable_cache:
        params["FLYDSL_RUNTIME_ENABLE_CACHE"] = enable_cache
    if source_file:
        try:
            with open(source_file, "r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(_FLYDSL_SCAN_BYTES)
        except OSError:
            head = ""
        if head:
            if any(m in head for m in _FLYDSL_SMEM_MARKERS):
                params["FLYDSL_USES_SMEM"] = True
            if any(m in head for m in _FLYDSL_BUFFER_LOAD_MARKERS):
                params["FLYDSL_USES_BUFFER_LOAD"] = True
    return params


def enrich_candidates_with_runtime_metadata(
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Attach stable runtime metadata fields before GEAK prompt generation.

    Mutates each candidate in place, filling framework, shapes/dtypes,
    model and FlyDSL kernel params, and runtime flags so the downstream
    prompt builder sees a consistent schema.

    Args:
        candidates (list[dict[str, Any]]): Hot-kernel candidate rows to enrich.
        args (argparse.Namespace): Parsed CLI args carrying framework, model
            name, target platform, and runtime flags.
    """
    framework = str(getattr(args, "framework", "") or "").strip()
    model_params = load_model_kernel_params(str(getattr(args, "model_name", "") or ""))
    target_platform = str(getattr(args, "target_platform", "") or "")
    runtime_flags = {
        "analysis_mode": tracelens_analysis_mode(framework),
        "target_platform": target_platform,
    }
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if framework:
            item.setdefault("framework", framework)
            item.setdefault("backend", framework)
        if "input_shapes" not in item:
            item["input_shapes"] = _shape_call_entries(
                item.get("shapes", []) or [],
                item.get("call_count"),
            )
            item["_input_shapes_synthetic"] = True
        item.setdefault("output_shapes", [])
        # Per-arg dtypes: TraceLens records them INLINE in each ``shapes`` string
        # ("(64,5120) bf16"); surface them as a clean, positionally-aligned list so
        # the GEAK harness allocates the correct-dtype tensors (fp8 weight vs bf16
        # activation). Falls back to any explicit ``dtypes`` field, then ``[]``.
        existing_dtypes = item.get("input_dtypes") or item.get("dtypes") or []
        if not existing_dtypes:
            existing_dtypes = _dtypes_from_shapes(item.get("shapes", []) or [])
        item["input_dtypes"] = existing_dtypes
        item.setdefault("output_dtypes", [])
        item.setdefault("runtime_args", {})
        item.setdefault("env_vars", {})
        item.setdefault("kernel_params", {})
        if model_params:
            params = item["kernel_params"]
            if isinstance(params, dict):
                for key, value in model_params.items():
                    params.setdefault(key, value)
        if item.get("source_type") == "flydsl":
            # FlyDSL pseudo-ops carry no real source_file; inject the FlyDSL MoE kernel source.
            _sf2 = str(item.get("source_file") or "").strip()
            if (not _sf2) or (not os.path.isfile(_sf2)):
                fb = _resolve_flydsl_source_fallback()
                if fb:
                    item["source_file"] = fb
            flydsl_params = _flydsl_kernel_params(
                str(item.get("source_file") or ""),
                target_platform,
            )
            params = item["kernel_params"]
            if isinstance(params, dict):
                for key, value in flydsl_params.items():
                    params.setdefault(key, value)
        flags = item.get("runtime_flags")
        if not isinstance(flags, dict):
            flags = {}
            item["runtime_flags"] = flags
        for key, value in runtime_flags.items():
            if value not in (None, ""):
                flags.setdefault(key, value)
        flags.setdefault("is_multigpu", bool(item.get("is_multigpu")))
        flags.setdefault("num_gpus_recommended", item.get("num_gpus_recommended"))
        # Kernel-class CONTRACT enrichment so GEAK's harness is faithful for
        # non-MoE kernels too (collectives need a real distributed reference;
        # attention needs causal/kv layout). Without these the harness can pass
        # a tautological or wrong-regime correctness check and a "win" won't
        # transfer to E2E. Derived from the kernel name + model config.
        _enrich_kernel_contract(item, model_params)


# Kernel-name tag -> torch.distributed op; the first match wins, so the bare "reduce" stays last.
_COLLECTIVE_OPS = {
    "all_reduce": "all_reduce",
    "allreduce": "all_reduce",
    "all_gather": "all_gather",
    "allgather": "all_gather",
    "reduce_scatter": "reduce_scatter",
    "reducescatter": "reduce_scatter",
    "all_to_all": "all_to_all",
    "alltoall": "all_to_all",
    "broadcast": "broadcast",
    # aiter cross_device_reduce_* kernels use all-reduce semantics.
    "cross_device_reduce": "all_reduce",
    "reduce": "reduce",
}


def _enrich_kernel_contract(item: dict[str, Any], model_params: dict[str, Any]) -> None:
    """Populate per-kernel-class contract fields used to build a faithful harness.

    - Collectives (all_reduce / all_gather / reduce_scatter / ...): record
      ``collective_op`` + ``world_size``/``tp_size`` (from model_params) so the
      harness can use ``torch.distributed.<op>`` as a real reference instead of
      a self-compare, and so a reviewer knows the kernel is comm-bound.
    - Attention (mha / flash / paged): record ``causal`` (decode/prefill attn
      is causal), ``kv_layout``, ``head_dim``/``num_heads`` from model_params,
      and the ``seqlen_regime`` so the harness benchmarks the served regime.
    These are advisory metadata; absent fields are simply not set.
    """
    name = str(item.get("name") or "").lower()
    contract = item.get("kernel_contract") or {}
    # --- collectives ---
    collective_op = next((op for tag, op in _COLLECTIVE_OPS.items() if tag in name), None)
    if item.get("is_multigpu") or collective_op:
        op = collective_op or "all_reduce"
        contract["kind"] = "collective"
        contract["collective_op"] = op
        tp = model_params.get("TP_SIZE") or model_params.get("TENSOR_PARALLEL_SIZE")
        ws = model_params.get("WORLD_SIZE") or tp or item.get("num_gpus_recommended")
        if ws:
            contract["world_size"] = int(ws)
        if tp:
            contract["tp_size"] = int(tp)
        contract.setdefault("reduce_op", "sum")
        contract["reference"] = f"torch.distributed.{op}"
        contract["e2e_note"] = "comm-bound collective: a 1-GPU GEAK slot cannot reproduce inter-GPU traffic"
    # --- attention ---
    elif any(t in name for t in ("mha", "flash", "attn", "attention", "paged")):
        contract["kind"] = "attention"
        contract["causal"] = True  # autoregressive serving attention is causal
        for src, dst in (
            ("HEAD_SIZE", "head_dim"),
            ("NUM_ATTENTION_HEADS", "num_heads"),
            ("NUM_KEY_VALUE_HEADS", "num_kv_heads"),
        ):
            if model_params.get(src) is not None:
                contract[dst] = model_params[src]
        contract.setdefault("kv_layout", "unknown")  # to be confirmed from trace; flag for reviewer
        contract["seqlen_regime"] = next((regime for regime in ("prefill", "decode") if regime in name), "mixed")
        contract["reference"] = "torch.nn.functional.scaled_dot_product_attention"
    # E2E-TRANSFER honesty flag: warn when a kernel-level win is unlikely to move
    # serving E2E so neither the pipeline nor a reviewer over-trusts a micro-speedup.
    #  - sealed-ASM launchers: asm_*.cu dispatches a prebuilt .co HSACO; only the
    #    host-side launcher is editable, so wins are tiny (observed E2E #1: 1.09x
    #    kernel -> 1.0008x E2E).
    #  - collectives on a 1-GPU GEAK slot: cannot reproduce inter-GPU traffic.
    src = str(item.get("source_file") or "").lower()
    reasons = []
    if "/py_itfs_cu/asm_" in src or (os.path.basename(src).startswith("asm_") and src.endswith(".cu")):
        reasons.append("sealed-ASM launcher (.co binary not editable; only host launcher)")
    if contract.get("kind") == "collective":
        reasons.append("comm-bound collective (needs multi-GPU GEAK slot to reproduce E2E traffic)")
    if reasons:
        item["e2e_transferable"] = False
        item["e2e_transfer_note"] = "; ".join(reasons)
    else:
        item.setdefault("e2e_transferable", True)
    if contract:
        item["kernel_contract"] = contract
