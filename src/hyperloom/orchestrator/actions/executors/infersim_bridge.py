# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferSim projection bridge.

Maps a Hyperloom benchmark spec (the ``benchmark`` block of a materialized
Magpie YAML: framework/model/precision + TP/CONC/ISL/OSL envs) onto Infera's
``infersim`` serving projection and returns the same throughput/latency
measurements a real serving benchmark would produce -- without booting a
server or touching a GPU.

This is the analytical inner-loop that lets the optimizer simulate candidate
serving configs and spend real GPU time only on the final validation, which is
the GPU-time reduction projected in the deck.

Design notes
------------
* Infera is an *optional* dependency. Everything here imports it lazily so the
  Hyperloom base install is unaffected; a missing/broken Infera surfaces as a
  structured error the runner turns into a failed report (never a crash).
* The projection is driven through Infera's own CLI plumbing
  (``build_parser().parse_known_args`` -> ``launch_projection_from_cli``) so we
  inherit its argument defaults and stay forward-compatible with new flags
  instead of hand-constructing its config dataclasses.
* Model selection is deliberately explicit: the operator points us at an
  InferSim model preset (``HYPERLOOM_INFERSIM_MODEL``) or a full workload YAML
  (``HYPERLOOM_INFERSIM_WORKLOAD``); a best-effort heuristic maps common HF
  model paths to presets so the common cases work with zero extra config.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Env knobs (all optional unless noted). Documented in the module docstring and
# the runner --help.
ENV_ROOT = "HYPERLOOM_INFERSIM_ROOT"  # path to the Infera checkout (added to sys.path)
ENV_WORKLOAD = "HYPERLOOM_INFERSIM_WORKLOAD"  # explicit InferSim workload YAML
ENV_MODEL = "HYPERLOOM_INFERSIM_MODEL"  # InferSim model preset name (e.g. gpt_oss_120B)
ENV_GPU_ARCH = "HYPERLOOM_INFERSIM_GPU_ARCH"  # e.g. mi355x (default)
ENV_HBM_GB = "HYPERLOOM_INFERSIM_HBM_GB"  # per-GPU HBM capacity, GB
ENV_EP = "HYPERLOOM_INFERSIM_EP"  # expert parallelism override
ENV_PP = "HYPERLOOM_INFERSIM_PP"  # pipeline parallelism override
ENV_KV_DTYPE = "HYPERLOOM_INFERSIM_KV_DTYPE"  # kv-cache dtype override
ENV_ANCHOR = "HYPERLOOM_INFERSIM_ANCHOR"  # single GPU anchor JSON (calibration)
ENV_ANCHOR_SCALING = "HYPERLOOM_INFERSIM_ANCHOR_SCALING"  # comma-sep TP-scaling anchors
ENV_ANCHOR_STORE = "HYPERLOOM_INFERSIM_ANCHOR_STORE"  # dir of warmup anchors (auto-select)
ENV_SERVING_MODEL = "HYPERLOOM_INFERSIM_SERVING_MODEL"  # continuous (default) | static

_DEFAULT_GPU_ARCH = "mi355x"
# Per-GPU HBM by arch (GB); only used when HBM is not supplied explicitly.
_ARCH_HBM_GB = {"mi300x": 192.0, "mi325x": 256.0, "mi355x": 288.0}

# Best-effort HF-path/name substring -> InferSim megatron preset. First match
# wins; extend freely. Override any time with HYPERLOOM_INFERSIM_MODEL.
_MODEL_HEURISTICS: tuple[tuple[str, str], ...] = (
    ("gpt-oss-120b", "gpt_oss_120B"),
    ("gpt-oss-20b", "gpt_oss_20B"),
    ("gpt_oss_120b", "gpt_oss_120B"),
    ("gpt_oss_20b", "gpt_oss_20B"),
    ("minimax-m2.5", "minimax_m2.5"),
    ("minimax_m2.5", "minimax_m2.5"),
    ("qwen3-235b", "qwen3_235B_A22B"),
    ("qwen3-32b", "qwen3_32B"),
    ("qwen3-30b", "qwen3_30B_A3B"),
    ("qwen3-14b", "qwen3_14B"),
    ("qwen3-4b", "qwen3_4B"),
    ("qwen2.5-72b", "qwen2.5_72B"),
    ("qwen2.5-32b", "qwen2.5_32B"),
    ("qwen2.5-14b", "qwen2.5_14B"),
    ("qwen2.5-7b", "qwen2.5_7B"),
    ("llama3.1-70b", "llama3.1_70B"),
    ("llama3.1-8b", "llama3.1_8B"),
    ("llama3.3-70b", "llama3.3_70B"),
    ("deepseek-v3", "deepseek_v3"),
    ("deepseek-v2", "deepseek_v2"),
    ("mixtral-8x22b", "mixtral_8x22B_v0.1"),
    ("mixtral-8x7b", "mixtral_8x7B_v0.1"),
)

# Bundled env-driven workload template used when only a preset name is known.
_TEMPLATE_WORKLOAD = (
    Path(__file__).resolve().parents[3]
    / "inference_optimizer"
    / "assets"
    / "infersim"
    / "infersim_workload.yaml"
)


class InfersimBridgeError(RuntimeError):
    """Raised for any recoverable bridge failure (bad config, import, etc.)."""


@dataclass
class ServingSpec:
    """Normalized serving request extracted from a Hyperloom benchmark block."""

    framework: str
    model_path: str
    tp: int = 1
    ep: int = 1
    pp: int = 1
    conc: int = 64
    isl: int = 1024
    osl: int = 1024
    weight_dtype: str = "bf16"
    kv_cache_dtype: str = "bf16"
    extra_server_args: str = ""


@dataclass
class ProjMetrics:
    """Projected serving metrics, mapped onto benchmark measurement fields."""

    output_throughput: float  # aggregate output tok/s (Magpie headline)
    request_throughput: float
    total_token_throughput: float
    ttft_ms: float
    tpot_ms: float
    itl_ms: float
    e2el_ms: float
    decode_tps_per_gpu: float
    memory_per_gpu_gb: float
    max_concurrency: int
    calibrated: bool = False
    replica_gpus: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _first_env_or(bench_envs: dict, key: str, default: Any) -> Any:
    """Ambient env wins over YAML envs (Magpie/bypass convention), then default."""
    val = os.environ.get(key)
    if val is None or str(val).strip() == "":
        val = bench_envs.get(key)
    if val is None or str(val).strip() == "":
        return default
    return val


def _precision_to_weight_dtype(precision: str) -> str:
    p = (precision or "").strip().lower()
    if p in ("fp8", "e4m3", "e5m2", "fp8_hybrid", "hybrid"):
        return "fp8"
    if p in ("mxfp4", "fp4"):
        return "mxfp4"
    return "bf16"


def _parse_server_arg_int(server_args: str, *flags: str) -> int | None:
    """Pull an int value for any of ``flags`` out of a server-arg string."""
    if not server_args:
        return None
    toks = server_args.split()
    for i, tok in enumerate(toks):
        for flag in flags:
            if tok == flag and i + 1 < len(toks):
                return _as_int(toks[i + 1], 0) or None
            if tok.startswith(flag + "="):
                return _as_int(tok.split("=", 1)[1], 0) or None
    return None


def _parse_server_arg_str(server_args: str, *flags: str) -> str | None:
    """Pull a string value for any of ``flags`` out of a server-arg string."""
    if not server_args:
        return None
    toks = server_args.split()
    for i, tok in enumerate(toks):
        for flag in flags:
            if tok == flag and i + 1 < len(toks):
                return toks[i + 1]
            if tok.startswith(flag + "="):
                return tok.split("=", 1)[1]
    return None


def spec_from_benchmark(bench: dict) -> ServingSpec:
    """Extract a :class:`ServingSpec` from a Magpie ``benchmark`` block."""
    bench = bench or {}
    envs = dict(bench.get("envs") or {})
    framework = str(bench.get("framework") or "sglang").lower()
    model_path = str(bench.get("model") or os.environ.get("MODEL", ""))
    extra_key = {
        "sglang": "EXTRA_SGLANG_ARGS",
        "vllm": "EXTRA_VLLM_ARGS",
        "atom": "EXTRA_ATOM_ARGS",
    }.get(framework, "")
    extra_args = str(_first_env_or(envs, extra_key, "")) if extra_key else ""

    tp = _as_int(_first_env_or(envs, "TP", 1), 1)
    # EP/PP: explicit bridge env, else parse from server args, else 1.
    ep = _as_int(os.environ.get(ENV_EP) or "", 0) or _parse_server_arg_int(
        extra_args, "--ep-size", "--expert-parallel-size", "--moe-ep-size"
    ) or 1
    pp = _as_int(os.environ.get(ENV_PP) or "", 0) or _parse_server_arg_int(
        extra_args, "--pp-size", "--pipeline-parallel-size"
    ) or 1

    weight_dtype = _precision_to_weight_dtype(str(bench.get("precision") or "bf16"))
    kv_dtype = str(os.environ.get(ENV_KV_DTYPE) or "bf16").lower()

    return ServingSpec(
        framework=framework,
        model_path=model_path,
        tp=max(1, tp),
        ep=max(1, ep),
        pp=max(1, pp),
        conc=max(1, _as_int(_first_env_or(envs, "CONC", 64), 64)),
        isl=max(1, _as_int(_first_env_or(envs, "ISL", 1024), 1024)),
        osl=max(1, _as_int(_first_env_or(envs, "OSL", 1024), 1024)),
        weight_dtype=weight_dtype,
        kv_cache_dtype=kv_dtype,
        extra_server_args=extra_args,
    )


def resolve_preset(model_path: str) -> str | None:
    """Best-effort map a model path/name to an InferSim preset name."""
    explicit = os.environ.get(ENV_MODEL)
    if explicit and explicit.strip():
        return explicit.strip()
    key = re.sub(r"[^a-z0-9.]+", "-", (model_path or "").lower())
    for needle, preset in _MODEL_HEURISTICS:
        if needle in key:
            return preset
    return None


@dataclass
class AnchorChoice:
    """The warmup anchor selected for a candidate, plus why it was chosen.

    ``regime_distance`` is the Hamming distance over InferSim's regime-defining
    axes (model/dtypes/attention-backend/cudagraph/aiter). Distance 0 means the
    candidate only moves along *transport* axes (TP/EP/PP, batch, concurrency,
    sequence lengths) and is fully reconstructable from this anchor -- i.e. no
    new GPU benchmark is needed. A non-zero distance means the candidate changes
    the kernel regime and warrants a fresh warmup anchor.
    """

    path: str
    regime_distance: int
    model: str | None = None
    needs_warmup: bool = False
    real_weights: bool = False


def recipe_from_spec(spec: ServingSpec) -> dict[str, Any]:
    """Canonical InferSim recipe dict for a Hyperloom serving spec."""
    attn = _parse_server_arg_str(spec.extra_server_args, "--attention-backend")
    return {
        "model": spec.model_path or None,
        "weight_dtype": spec.weight_dtype,
        "kv_cache_dtype": spec.kv_cache_dtype,
        "moe_expert_dtype": None,
        "attention_backend": attn,
        "cudagraph": None,
        "aiter": None,
        "tp": spec.tp,
        "pp": spec.pp,
        "ep": spec.ep,
        "batch": spec.conc,
        "concurrency": spec.conc,
        "input_len": spec.isl,
        "output_len": spec.osl,
    }


def select_anchor(spec: ServingSpec) -> AnchorChoice | None:
    """Pick the closest in-regime warmup anchor for ``spec``.

    Precedence: an explicit ``HYPERLOOM_INFERSIM_ANCHOR`` always wins; otherwise
    an anchor store directory is searched for the nearest anchor in regime space.
    Returns ``None`` when neither is configured (pure-analytical projection).
    """
    explicit = os.environ.get(ENV_ANCHOR)
    if explicit and Path(explicit).is_file():
        return AnchorChoice(path=explicit, regime_distance=0, model=spec.model_path)

    store_root = os.environ.get(ENV_ANCHOR_STORE)
    if not store_root or not Path(store_root).is_dir():
        return None

    _ensure_infera_importable()
    try:
        from infera.projection.core.projection.inference_projection.search.anchor_store import (
            AnchorStore,
        )
    except Exception as exc:  # noqa: BLE001
        raise InfersimBridgeError(f"cannot import InferSim AnchorStore: {exc}") from exc

    store = AnchorStore(store_root)
    recipe = recipe_from_spec(spec)
    entries = store.entries()
    if spec.model_path:
        named = [e for e in entries if e.get("model") in (None, spec.model_path)]
        entries = named or entries
    if not entries:
        return None

    from infera.projection.core.projection.inference_projection.search import regime

    def rank(entry: dict[str, Any]) -> tuple[int, int, float]:
        """Sort key: regime distance, then *fidelity*, then transport closeness.

        Fidelity matters as much as regime here: a dummy-weight anchor runs the
        same kernels but with synthetic MoE routing, so its decode curve is much
        flatter than a real-weights run. Ranking it below a real-weights anchor
        in the same regime is the difference between ~2% and ~30% error against
        measured serving.
        """
        dist = regime.regime_distance(recipe, dict(entry.get("regime") or {}))
        real = _anchor_is_real_weights(entry["path"])
        transport = entry.get("transport") or {}
        gap = 0.0
        for axis in ("tp", "ep", "pp"):
            av, rv = transport.get(axis), recipe.get(axis)
            if av and rv:
                gap += abs(float(av) - float(rv))
        return (dist, 0 if real else 1, gap)

    best = min(entries, key=rank)
    dist, fidelity_rank, _ = rank(best)
    return AnchorChoice(
        path=best["path"],
        regime_distance=int(dist),
        model=best.get("model"),
        needs_warmup=bool(dist),
        real_weights=(fidelity_rank == 0),
    )


def _anchor_is_real_weights(path: str) -> bool:
    """True when an anchor artifact was measured with real checkpoint weights."""
    try:
        import json

        with open(path) as fh:
            meta = (json.load(fh) or {}).get("meta") or {}
    except (OSError, ValueError):
        return False
    if meta.get("real_weights") is not None:
        return bool(meta["real_weights"])
    return str(meta.get("load_format") or "").lower() not in ("dummy", "")


def _resolve_workload_and_env(spec: ServingSpec) -> tuple[str, dict[str, str]]:
    """Return (workload_yaml_path, extra_env) for the projection.

    Precedence: an explicit ``HYPERLOOM_INFERSIM_WORKLOAD`` wins; otherwise a
    resolved preset name is fed to the bundled env-driven template via
    ``INFERSIM_MODEL``.
    """
    extra_env: dict[str, str] = {}
    explicit = os.environ.get(ENV_WORKLOAD)
    if explicit and Path(explicit).is_file():
        return str(Path(explicit).resolve()), extra_env

    preset = resolve_preset(spec.model_path)
    if not preset:
        raise InfersimBridgeError(
            f"could not resolve an InferSim model preset for model={spec.model_path!r}; "
            f"set {ENV_MODEL}=<preset> or {ENV_WORKLOAD}=<workload.yaml>"
        )
    if not _TEMPLATE_WORKLOAD.is_file():
        raise InfersimBridgeError(f"bundled workload template missing: {_TEMPLATE_WORKLOAD}")
    # The template reads INFERSIM_MODEL/TP/PP/EP; parallelism is *also* forced via
    # CLI overrides below so an explicit workload YAML is honored too.
    extra_env["INFERSIM_MODEL"] = preset
    return str(_TEMPLATE_WORKLOAD), extra_env


def _purge_foreign_infera(root: str) -> None:
    """Drop cached ``infera*`` modules not originating from ``root``.

    Another ``infera`` checkout may already be importable on the default path
    (and may lack the ``projection`` subpackage). Once imported it is cached in
    ``sys.modules``, so a later ``sys.path`` insert cannot override the
    top-level package. Purge any cached ``infera`` whose file is outside our
    root so the re-import resolves against ``HYPERLOOM_INFERSIM_ROOT``.
    """
    root_resolved = str(Path(root).resolve())
    for name in list(sys.modules):
        if name != "infera" and not name.startswith("infera."):
            continue
        mod = sys.modules.get(name)
        origin = getattr(mod, "__file__", None) or ""
        paths = list(getattr(mod, "__path__", []) or [])
        located = origin or (paths[0] if paths else "")
        if not located or not str(Path(located).resolve()).startswith(root_resolved):
            sys.modules.pop(name, None)


def _ensure_infera_importable() -> None:
    """Make ``infera.projection`` importable, honoring HYPERLOOM_INFERSIM_ROOT.

    When ``HYPERLOOM_INFERSIM_ROOT`` is set it takes precedence over any other
    ``infera`` on the path so the pinned Infera checkout is the one projected
    against.
    """
    root = os.environ.get(ENV_ROOT)
    if root and Path(root).is_dir():
        if sys.path[:1] != [root]:
            while root in sys.path:
                sys.path.remove(root)
            sys.path.insert(0, root)
        _purge_foreign_infera(root)

    try:
        import infera.projection  # noqa: F401
        return
    except Exception as exc:  # noqa: BLE001
        raise InfersimBridgeError(
            f"cannot import Infera 'infera.projection' (set {ENV_ROOT} to the Infera "
            f"checkout or pip install amd-infera[projection]): {exc}"
        ) from exc


def _build_argv(spec: ServingSpec, workload: str, anchor: AnchorChoice | None = None) -> list[str]:
    """Build the ``infersim inference`` argv for this serving spec."""
    gpu_arch = str(os.environ.get(ENV_GPU_ARCH) or _DEFAULT_GPU_ARCH).lower()
    hbm_gb = os.environ.get(ENV_HBM_GB) or _ARCH_HBM_GB.get(gpu_arch)
    serving_model = str(os.environ.get(ENV_SERVING_MODEL) or "continuous").lower()

    argv: list[str] = [
        "inference",
        "--config", workload,
        "--inference-mode", "both",
        "--profiling-mode", "simulate",
        "--serving-model", serving_model,
        "--input-len", str(spec.isl),
        "--output-len", str(spec.osl),
        "--inference-batch-size", str(spec.conc),
        "--max-concurrency", str(spec.conc),
        "--weight-dtype", spec.weight_dtype,
        "--kv-cache-dtype", spec.kv_cache_dtype,
        "--gpu-arch", gpu_arch,
    ]
    if hbm_gb:
        argv += ["--hbm-capacity-gb", str(hbm_gb)]

    if anchor is not None and Path(anchor.path).is_file():
        argv += ["--load-benchmark", anchor.path]
        argv += ["--profiling-mode", "both"]  # calibrate + report source
    scaling = os.environ.get(ENV_ANCHOR_SCALING)
    if scaling:
        for path in [p.strip() for p in scaling.split(",") if p.strip()]:
            argv += ["--load-benchmark-scaling", path]

    # Force parallelism via config overrides so an explicit workload YAML is
    # honored regardless of its baked-in values.
    argv += [
        f"tensor_model_parallel_size={spec.tp}",
        f"expert_model_parallel_size={spec.ep}",
        f"pipeline_model_parallel_size={spec.pp}",
    ]
    return argv


def project(spec: ServingSpec) -> ProjMetrics:
    """Run the InferSim projection for ``spec`` and return mapped metrics."""
    _ensure_infera_importable()
    workload, extra_env = _resolve_workload_and_env(spec)

    from infera.projection.cli import build_parser
    from infera.projection.core.projection.inference_projection import (
        launch_projection_from_cli,
    )

    anchor = select_anchor(spec)
    argv = _build_argv(spec, workload, anchor)
    # Template reads INFERSIM_* env; also expose TP/PP/EP for template default
    # interpolation (overrides above still win for explicit workloads).
    prev_env: dict[str, str | None] = {}
    inject = dict(extra_env)
    inject.setdefault("INFERSIM_TP", str(spec.tp))
    inject.setdefault("INFERSIM_PP", str(spec.pp))
    inject.setdefault("INFERSIM_EP", str(spec.ep))
    for key, val in inject.items():
        prev_env[key] = os.environ.get(key)
        os.environ[key] = val
    try:
        args, overrides = build_parser().parse_known_args(argv)
        results = launch_projection_from_cli(args, overrides)
    except InfersimBridgeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InfersimBridgeError(f"InferSim projection failed: {exc}") from exc
    finally:
        for key, val in prev_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    perf = results.get("performance")
    if perf is None:
        raise InfersimBridgeError("InferSim returned no performance projection")
    mem = results.get("memory")

    return _metrics_from_results(spec, perf, mem, anchor)


def _metrics_from_results(
    spec: ServingSpec, perf: Any, mem: Any, anchor: AnchorChoice | None = None
) -> ProjMetrics:
    """Map InferSim result objects onto benchmark measurement fields."""
    output_tps = float(getattr(perf, "decode_throughput_tps", 0.0) or 0.0)
    osl = max(1, spec.osl)
    isl = max(1, spec.isl)
    request_tps = output_tps / osl if osl else 0.0
    total_tps = output_tps * (isl + osl) / osl if osl else output_tps

    mem_gb = 0.0
    if mem is not None:
        total_bytes = float(getattr(mem, "total_bytes", 0) or 0)
        mem_gb = total_bytes / (1024.0 ** 3)
    extras = dict(getattr(perf, "extras", {}) or {})
    max_conc = int(extras.get("concurrency_used", 0) or extras.get("concurrency", 0) or spec.conc)
    if anchor is not None:
        # Provenance so a session can audit which warmup anchor served this
        # candidate and whether it stayed inside the anchor's regime.
        extras["anchor_path"] = anchor.path
        extras["anchor_regime_distance"] = anchor.regime_distance
        extras["anchor_needs_warmup"] = anchor.needs_warmup
        extras["anchor_real_weights"] = anchor.real_weights

    return ProjMetrics(
        output_throughput=output_tps,
        request_throughput=request_tps,
        total_token_throughput=total_tps,
        ttft_ms=float(getattr(perf, "ttft_ms", 0.0) or 0.0),
        tpot_ms=float(getattr(perf, "itl_ms", 0.0) or 0.0),
        itl_ms=float(getattr(perf, "itl_ms", 0.0) or 0.0),
        e2el_ms=float(getattr(perf, "request_latency_ms", 0.0) or 0.0),
        decode_tps_per_gpu=float(getattr(perf, "decode_throughput_tps_per_gpu", 0.0) or 0.0),
        memory_per_gpu_gb=mem_gb,
        max_concurrency=max_conc,
        calibrated=bool(extras.get("benchmark_calibrated", 0.0)),
        replica_gpus=int(getattr(perf, "replica_gpus", 0) or 0),
        extras=extras,
    )


def raw_result_from_metrics(spec: ServingSpec, m: ProjMetrics) -> dict[str, Any]:
    """Build an InferenceX-style flat result dict from projected metrics.

    Keys mirror what ``bypass_report.build_report`` / Magpie's result parser
    read, so the simulated run flows through Hyperloom's collectors unchanged.
    Percentiles are set to the mean (the point projection has no distribution;
    use the DES path for tails).
    """
    completed = max(1, m.max_concurrency)
    duration = (m.e2el_ms / 1000.0) if m.e2el_ms else 0.0
    return {
        "model_id": spec.model_path,
        "request_throughput": m.request_throughput,
        "output_throughput": m.output_throughput,
        "total_token_throughput": m.total_token_throughput,
        "completed": completed,
        "total_input_tokens": completed * spec.isl,
        "total_output_tokens": completed * spec.osl,
        "duration": duration,
        "mean_ttft_ms": m.ttft_ms,
        "median_ttft_ms": m.ttft_ms,
        "p99_ttft_ms": m.ttft_ms,
        "std_ttft_ms": 0.0,
        "mean_tpot_ms": m.tpot_ms,
        "median_tpot_ms": m.tpot_ms,
        "p99_tpot_ms": m.tpot_ms,
        "std_tpot_ms": 0.0,
        "mean_itl_ms": m.itl_ms,
        "median_itl_ms": m.itl_ms,
        "p99_itl_ms": m.itl_ms,
        "std_itl_ms": 0.0,
        "mean_e2el_ms": m.e2el_ms,
        "median_e2el_ms": m.e2el_ms,
        "p99_e2el_ms": m.e2el_ms,
        "std_e2el_ms": 0.0,
        # Non-Magpie diagnostics, carried for inspection/reporting.
        "infersim_decode_tps_per_gpu": m.decode_tps_per_gpu,
        "infersim_memory_per_gpu_gb": m.memory_per_gpu_gb,
        "infersim_calibrated": m.calibrated,
        "infersim_replica_gpus": m.replica_gpus,
        "infersim_tp": spec.tp,
        "infersim_ep": spec.ep,
        "infersim_pp": spec.pp,
    }
