# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferaSim projection bridge.

Maps a Hyperloom benchmark spec (the ``benchmark`` block of a materialized
Magpie YAML: framework/model/precision + TP/CONC/ISL/OSL envs) onto Infera's
``inferasim`` serving projection and returns projected throughput/latency.

The only consumer is EXPLORE's projection router (:mod:`_explore_projection`),
which compares a variant's projection with the stack's projection to decide
which variants are worth a real benchmark. A projected number is never
compared with a measured one and never decides a KEEP. In particular
``ProjMetrics.output_throughput`` is the projected aggregate decode rate, not
Magpie's ``output_throughput``.

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
  InferaSim model preset (``HYPERLOOM_INFERASIM_MODEL``) or a full workload YAML
  (``HYPERLOOM_INFERASIM_WORKLOAD``); a best-effort heuristic maps common HF
  model paths to presets so the common cases work with zero extra config.
* Three projection modes, picked by ``HYPERLOOM_INFERASIM_MODE``. ``simulate``
  prices kernels from models: no anchor is read, so no server is ever booted
  to produce one. ``benchmark`` calibrates against the nearest in-regime
  anchor (``HYPERLOOM_INFERASIM_ANCHOR`` / ``_ANCHOR_STORE`` /
  ``_ANCHOR_SCALING``). When the store has none for the candidate's regime, one
  is harvested with Infera's served-anchor benchmark and indexed into the
  store, so the next candidate in that regime reuses it
  (``HYPERLOOM_INFERASIM_HARVEST=0`` turns that off). ``auto`` (the default)
  is benchmark mode when the store already holds an in-regime anchor and
  simulate mode otherwise; it never harvests. Benchmark mode fails closed: if
  no anchor can be found or harvested, the projection raises rather than
  quietly returning an analytical number.
* The anchor store also fills itself: :func:`record_measured_anchor` turns a
  real EXPLORE decision round into a single-point served anchor, so the
  measurements a session already paid for calibrate the next projection.
* Throughput and latency come from Infera's discrete-event replay of a
  fixed-concurrency client (``HYPERLOOM_INFERASIM_ESTIMATOR=des``, the
  default), which ranked held-out configurations far better than the closed
  form did; ``analytical`` selects the closed form.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from hyperloom.inference_optimizer.framework_registry import server_args_env_name

# Env knobs (all optional unless noted). Documented in the module docstring and
# the runner --help.
ENV_ROOT = "HYPERLOOM_INFERASIM_ROOT"  # path to the Infera checkout (added to sys.path)
ENV_WORKLOAD = "HYPERLOOM_INFERASIM_WORKLOAD"  # explicit InferaSim workload YAML
ENV_MODEL = "HYPERLOOM_INFERASIM_MODEL"  # InferaSim model preset name (e.g. gpt_oss_120B)
ENV_GPU_ARCH = "HYPERLOOM_INFERASIM_GPU_ARCH"  # e.g. mi355x (default)
ENV_HBM_GB = "HYPERLOOM_INFERASIM_HBM_GB"  # per-GPU HBM capacity, GB
ENV_EP = "HYPERLOOM_INFERASIM_EP"  # expert parallelism override
ENV_PP = "HYPERLOOM_INFERASIM_PP"  # pipeline parallelism override
ENV_KV_DTYPE = "HYPERLOOM_INFERASIM_KV_DTYPE"  # kv-cache dtype override
ENV_ANCHOR = "HYPERLOOM_INFERASIM_ANCHOR"  # single GPU anchor JSON (calibration)
ENV_ANCHOR_SCALING = "HYPERLOOM_INFERASIM_ANCHOR_SCALING"  # comma-sep TP-scaling anchors
ENV_ANCHOR_STORE = "HYPERLOOM_INFERASIM_ANCHOR_STORE"  # dir of warmup anchors (auto-select)
ENV_SERVING_MODEL = "HYPERLOOM_INFERASIM_SERVING_MODEL"  # continuous (default) | static
ENV_MODE = "HYPERLOOM_INFERASIM_MODE"  # auto (default) | simulate | benchmark
ENV_ESTIMATOR = "HYPERLOOM_INFERASIM_ESTIMATOR"  # des (default) | analytical
ENV_HARVEST = "HYPERLOOM_INFERASIM_HARVEST"  # benchmark mode: harvest on a miss (default on)
ENV_HARVEST_PYTHON = "HYPERLOOM_INFERASIM_HARVEST_PYTHON"  # interpreter that can launch the engine
ENV_HARVEST_GPUS = "HYPERLOOM_INFERASIM_HARVEST_GPUS"  # GPUs the harvest may use (default min(TP*PP, 4))
ENV_HARVEST_TIMEOUT = "HYPERLOOM_INFERASIM_HARVEST_TIMEOUT_SEC"

log = logging.getLogger(__name__)

MODE_SIMULATE = "simulate"
MODE_BENCHMARK = "benchmark"
MODE_AUTO = "auto"
_MODES = (MODE_AUTO, MODE_SIMULATE, MODE_BENCHMARK)

ESTIMATOR_DES = "des"
ESTIMATOR_ANALYTICAL = "analytical"
_ESTIMATORS = (ESTIMATOR_DES, ESTIMATOR_ANALYTICAL)

_DEFAULT_GPU_ARCH = "mi355x"
# Per-GPU HBM by arch (GB); only used when HBM is not supplied explicitly.
_ARCH_HBM_GB = {"mi300x": 192.0, "mi325x": 256.0, "mi355x": 288.0}

# Best-effort HF-path/name substring -> InferaSim megatron preset. First match
# wins; extend freely. Override any time with HYPERLOOM_INFERASIM_MODEL.
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
    # Longest/most specific needles first: "deepseek-v2-lite" must not be
    # swallowed by "deepseek-v2", and R1 is the V3 architecture at 671B so it
    # resolves to the V3 config rather than a config of its own.
    ("deepseek-v2-lite", "deepseek_v2_lite"),
    ("deepseek-r1", "deepseek_v3_671b-fp8"),
    ("deepseek-v3", "deepseek_v3"),
    ("deepseek-v2", "deepseek_v2"),
    ("minimax-m2", "minimax_m2.5"),
    ("mixtral-8x22b", "mixtral_8x22B_v0.1"),
    ("mixtral-8x7b", "mixtral_8x7B_v0.1"),
)

# Bundled env-driven workload template used when only a preset name is known.
_TEMPLATE_WORKLOAD = (
    Path(__file__).resolve().parents[3] / "inference_optimizer" / "assets" / "inferasim" / "inferasim_workload.yaml"
)


class InferasimBridgeError(RuntimeError):
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
    num_prompts: int = 0
    # Image digest / engine version of the deployment, when known. A measured
    # anchor that recorded a different one is not used.
    runtime: dict[str, str] = field(default_factory=dict)


# Context, in tokens, out to which the projection has actually been checked
# against real vLLM measurements. A controlled sweep -- prefix caching off, three
# seeds, prompt length from 1k to 65,536 -- scores 6.0% across that range, which
# covers everything Hyperloom searches. Beyond it the decode step has never been
# compared to hardware, and the last extrapolation past a measured range hid a
# 46.3% error until someone ran the sweep.
VALIDATED_CONTEXT_TOKENS = 65536

# Above this the deployment spans nodes, and no inference measurement behind the
# projection ever crossed a node boundary.
SINGLE_NODE_GPUS = 8


def extrapolation_notes(spec: "ServingSpec", replica_gpus: int, calibrated: bool) -> list[str]:
    """Where this projection is being asked to work outside its evidence.

    Returned on every projection so a search cannot quietly trust a number that
    was produced by extrapolating along an axis nothing validated. These are not
    error bars -- the size of the error is not known, which is the point.
    """
    notes: list[str] = []
    context = int(spec.isl or 0) + int(spec.osl or 0)
    if context > VALIDATED_CONTEXT_TOKENS:
        notes.append(
            f"context {context} tokens exceeds the {VALIDATED_CONTEXT_TOKENS} "
            "this model has been checked against, and the decode step is known "
            "to be too flat in context; use for ranking, and anchor near the "
            "target context before believing the absolute latency"
        )
    if replica_gpus > SINGLE_NODE_GPUS:
        notes.append(
            f"{replica_gpus} GPUs spans nodes; the inter-node collective terms "
            "are derived, not measured, and nothing has checked them"
        )
    if not calibrated:
        notes.append(
            "no warmup anchor matched this model, so this is pure simulation with no measurement pinning its scale"
        )
    return notes


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


def _first_env_or(bench_envs: dict, key: str, default: Any, *, ambient: bool = True) -> Any:
    """Ambient env wins over YAML envs (Magpie/bypass convention), then default."""
    val = os.environ.get(key) if ambient else None
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


def _parse_kv_cache_dtype(server_args: str) -> str | None:
    """KV-cache dtype from a framework's server args, normalised for InferaSim.

    vLLM spells it ``--kv-cache-dtype fp8_e4m3`` (or ``fp8``, ``fp8_e5m2``) and
    sglang ``--kv-cache-dtype fp8_e4m3``; both mean the cache is stored in a
    single byte per element, which is what the projection needs to know. ``auto``
    means "follow the weights", which the caller's default already does.
    """
    raw = _parse_server_arg_str(server_args, "--kv-cache-dtype", "--kv_cache_dtype")
    if not raw:
        return None
    v = raw.strip().strip("'\"").lower()
    if v in ("auto", ""):
        return None
    if v.startswith("fp8"):
        return "fp8"
    if v in ("bf16", "bfloat16"):
        return "bf16"
    if v in ("fp16", "float16", "half"):
        return "fp16"
    return v


def parse_speculative(server_args: str) -> tuple[str | None, int]:
    """Speculative-decoding ``(method, k)`` from a framework's server args.

    Each serving framework spells this differently, and Hyperloom's EXPLORE grid
    emits all three, so all three are recognised:

    * vLLM   ``--speculative-config '{"method": "deepseek_mtp",
      "num_speculative_tokens": 3}'``
    * SGLang ``--speculative-algorithm NEXTN --speculative-num-steps 3``
    * atom   ``--method mtp --num-speculative-tokens 3``

    Returns ``(None, 0)`` when the args request no speculation. ``k`` defaults to
    1 when a method is named without a token count, matching every framework's
    own default.
    """
    if not server_args:
        return None, 0

    method = _parse_server_arg_str(server_args, "--speculative-algorithm")
    k = _parse_server_arg_int(
        server_args,
        "--speculative-num-steps",
        "--num-speculative-tokens",
        "--speculative-num-draft-tokens",
        "--speculative-tokens",
    )

    raw = _parse_server_arg_str(server_args, "--speculative-config")
    if raw:
        # The JSON is usually quoted as a single shell token, but a bare
        # unquoted blob would have been split on spaces by the tokenizer; fall
        # back to a regex over the whole string in that case.
        try:
            cfg = json.loads(raw.strip("'\""))
        except (ValueError, TypeError):
            cfg = {}
            m = re.search(r'"method"\s*:\s*"([^"]+)"', server_args)
            if m:
                cfg["method"] = m.group(1)
            m = re.search(r'"num_speculative_tokens"\s*:\s*(\d+)', server_args)
            if m:
                cfg["num_speculative_tokens"] = int(m.group(1))
        if isinstance(cfg, dict):
            method = cfg.get("method") or method
            k = int(cfg.get("num_speculative_tokens") or 0) or k

    # atom spells the method as ``--method mtp``; only honour that spelling when
    # it names a speculative method, since ``--method`` is a generic flag name.
    if not method:
        generic = _parse_server_arg_str(server_args, "--method")
        if generic and generic.strip().lower() in ("mtp", "eagle", "eagle3", "nextn"):
            method = generic

    if not method:
        return None, 0
    return str(method), max(1, int(k or 0))


def _parse_server_arg_int(server_args: str, *flags: str) -> int:
    """Integer value for any of ``flags``, or 0 when absent/non-numeric."""
    raw = _parse_server_arg_str(server_args, *flags)
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return 0


def spec_from_benchmark(bench: dict, *, ambient: bool = True) -> ServingSpec:
    """Extract a :class:`ServingSpec` from a Magpie ``benchmark`` block.

    ``ambient=False`` reads the block alone: no process env and no
    ``HYPERLOOM_INFERASIM_*`` overrides. That is what a caller comparing two
    materialized configs needs, since an exported ``CONC`` would otherwise
    project every variant at the same concurrency.
    """
    bench = bench or {}

    def override(name: str) -> str:
        return (os.environ.get(name) or "") if ambient else ""

    envs = dict(bench.get("envs") or {})
    framework = str(bench.get("framework") or "sglang").lower()
    model_path = str(bench.get("model") or override("MODEL"))
    # Infera names the engine as a free string, so a build ("mori-sglang") is a
    # regime of its own and reaches us as one. Resolving the args env through the
    # registry rather than an exact table means such a build still has its server
    # args read -- the hand-rolled table returned nothing for it, which silently
    # dropped the attention backend, KV dtype and speculative flags that decide
    # the regime.
    extra_args = str(_first_env_or(envs, server_args_env_name(framework), "", ambient=ambient))

    tp = _as_int(_first_env_or(envs, "TP", 1, ambient=ambient), 1)
    # EP/PP: explicit bridge env, else parse from server args, else 1.
    ep = (
        _as_int(override(ENV_EP), 0)
        or _parse_server_arg_int(extra_args, "--ep-size", "--expert-parallel-size", "--moe-ep-size")
        or 1
    )
    pp = _as_int(override(ENV_PP), 0) or _parse_server_arg_int(extra_args, "--pp-size", "--pipeline-parallel-size") or 1

    weight_dtype = _precision_to_weight_dtype(str(bench.get("precision") or "bf16"))
    # KV dtype: explicit bridge env wins, else the server arg the variant sets,
    # else bf16. Reading the arg matters because an explore grid changes the KV
    # dtype by passing this flag and nothing else -- and the projection prices KV
    # dtype perfectly well, so ignoring the flag made a lever the model *can*
    # see look like one it cannot, and projected an fp8 candidate as bf16.
    kv_dtype = str(override(ENV_KV_DTYPE) or _parse_kv_cache_dtype(extra_args) or "bf16").lower()

    conc = max(1, _as_int(_first_env_or(envs, "CONC", 64, ambient=ambient), 64))
    # A running batch cannot exceed the scheduler's cap on concurrent sequences,
    # so a variant that lowers it lowers the batch the decode step actually runs.
    max_seqs = _parse_server_arg_int(extra_args, "--max-num-seqs", "--max-running-requests")
    if max_seqs and max_seqs > 0:
        conc = min(conc, max_seqs)

    return ServingSpec(
        framework=framework,
        model_path=model_path,
        tp=max(1, tp),
        ep=max(1, ep),
        pp=max(1, pp),
        conc=conc,
        isl=max(1, _as_int(_first_env_or(envs, "ISL", 1024, ambient=ambient), 1024)),
        osl=max(1, _as_int(_first_env_or(envs, "OSL", 1024, ambient=ambient), 1024)),
        weight_dtype=weight_dtype,
        kv_cache_dtype=kv_dtype,
        extra_server_args=extra_args,
        num_prompts=max(0, _as_int(_first_env_or(envs, "NUM_PROMPTS", 0, ambient=ambient), 0)),
    )


def resolve_preset(model_path: str) -> str | None:
    """Best-effort map a model path/name to an InferaSim preset name."""
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

    ``regime_distance`` is the Hamming distance over InferaSim's regime-defining
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
    served: bool = False


def recipe_from_spec(spec: ServingSpec) -> dict[str, Any]:
    """Canonical InferaSim recipe dict for a Hyperloom serving spec."""
    attn = _parse_server_arg_str(spec.extra_server_args, "--attention-backend")
    method, k = parse_speculative(spec.extra_server_args)
    return {
        "model": spec.model_path or None,
        # The engines differ in scheduler, paging and kernel selection, so one
        # engine's anchor cannot price another's run. Omitting this axis leaves
        # it missing on our side, and regime_distance skips an axis missing on
        # either side -- a vLLM and an SGLang anchor would both score 0 here.
        "engine": spec.framework or None,
        "weight_dtype": spec.weight_dtype,
        "kv_cache_dtype": spec.kv_cache_dtype,
        "moe_expert_dtype": None,
        "attention_backend": attn,
        "cudagraph": None,
        "aiter": None,
        # A speculating candidate emits >1 token per step, so an anchor measured
        # without speculation cannot price it. Putting this on the recipe is what
        # makes the regime signature reject such an anchor instead of silently
        # reporting a plain-decode number for it.
        "speculative": f"spec:{k}" if method else "off",
        # The part the projection is priced for. Harvested anchors do not
        # record one and match anything; a measured anchor from another GPU
        # does not.
        "gpu_arch": gpu_arch(),
        "tp": spec.tp,
        "pp": spec.pp,
        "ep": spec.ep,
        "batch": spec.conc,
        "concurrency": spec.conc,
        "input_len": spec.isl,
        "output_len": spec.osl,
    }


def gpu_arch(name: str | None = None) -> str:
    """Canonical part name (``mi355x``) for ``name``, else the projected arch."""
    raw = str(name or os.environ.get(ENV_GPU_ARCH) or _DEFAULT_GPU_ARCH).strip().lower()
    match = re.search(r"mi\d{3}x", raw)
    return match.group(0) if match else raw


def projection_mode() -> str:
    """Return the projection mode named by ``HYPERLOOM_INFERASIM_MODE``.

    Unset or blank means ``auto``. Anything else unrecognised is an error
    rather than a fallback: a typo for ``benchmark`` would otherwise quietly
    drop calibration and report analytical numbers as if they were anchored.
    """
    raw = str(os.environ.get(ENV_MODE) or "").strip().lower()
    if not raw:
        return MODE_AUTO
    if raw not in _MODES:
        raise InferasimBridgeError(f"{ENV_MODE}={raw!r} is not one of {', '.join(_MODES)}")
    return raw


def estimator() -> str:
    """The estimator named by ``HYPERLOOM_INFERASIM_ESTIMATOR`` (default ``des``)."""
    raw = str(os.environ.get(ENV_ESTIMATOR) or "").strip().lower()
    if not raw:
        return ESTIMATOR_DES
    if raw not in _ESTIMATORS:
        raise InferasimBridgeError(f"{ENV_ESTIMATOR}={raw!r} is not one of {', '.join(_ESTIMATORS)}")
    return raw


def resolve_mode(spec: ServingSpec) -> str:
    """The concrete mode ``spec`` is projected in: ``auto`` picks per anchor.

    ``auto`` calibrates when the store already holds an in-regime anchor for
    ``spec`` and simulates otherwise. Callers comparing several specs resolve
    it once, on the reference, and project every spec in that mode.
    """
    mode = projection_mode()
    if mode != MODE_AUTO:
        return mode
    try:
        anchor = select_anchor(spec)
    except InferasimBridgeError:
        return MODE_SIMULATE
    return MODE_BENCHMARK if anchor is not None and anchor.regime_distance == 0 else MODE_SIMULATE


def select_anchor(spec: ServingSpec) -> AnchorChoice | None:
    """Pick the closest in-regime warmup anchor for ``spec``.

    Precedence: an explicit ``HYPERLOOM_INFERASIM_ANCHOR`` always wins; otherwise
    an anchor store directory is searched for the nearest anchor in regime space.
    Returns ``None`` when neither is configured (pure-analytical projection).
    """
    explicit = os.environ.get(ENV_ANCHOR)
    if explicit and Path(explicit).is_file():
        # Pinned by an operator, but still gated: a corrupt curve would silently
        # propagate into every projection made from it.
        if not anchor_curve_is_sane(explicit):
            return None
        return AnchorChoice(path=explicit, regime_distance=0, model=spec.model_path)

    store_root = os.environ.get(ENV_ANCHOR_STORE)
    if not store_root or not Path(store_root).is_dir():
        return None

    _ensure_infera_importable()
    try:
        from infera.projection.core.projection.inference_projection.search.anchor_store import (
            AnchorStore,
        )
    except Exception as exc:
        raise InferasimBridgeError(f"cannot import InferaSim AnchorStore: {exc}") from exc

    store = AnchorStore(store_root)
    recipe = recipe_from_spec(spec)

    from infera.projection.core.projection.inference_projection.search import regime

    # Only this model's warmups may calibrate this model. The names arrive in
    # different spellings -- the spec carries a checkout path or a preset, the
    # artifact carries a HuggingFace id -- so they are compared loosely rather
    # than for equality, which never held and left the filter inert.
    #
    # An anchor with no recorded model is allowed: a structural warmup names no
    # checkpoint, and the regime axes still have to agree before it is used.
    # But when the store holds anchors and none of them are for this model, the
    # answer is that there is no anchor. This used to fall back to using any
    # anchor at all, which calibrated qwen3 against gpt-oss and returned the
    # same decode step for every model in the sweep, labelled "calibrated".
    # A target has two names and either one identifies it. The checkout path
    # ("/models/DeepSeek-R1") usually shares a spelling with the artifact's id,
    # while the preset names an architecture and can legitimately cover several
    # checkpoints -- deepseek_v3 is the preset for DeepSeek-R1, and matching only
    # on that would reject R1's own warmup.
    names = [n for n in (spec.model_path, resolve_preset(spec.model_path)) if n]
    entries = [
        e for e in store.entries() if not e.get("model") or any(regime.models_match(n, e["model"]) for n in names)
    ]
    if not entries:
        return None

    def rank(entry: dict[str, Any]) -> tuple[int, int, int, float]:
        """Sort key: regime distance, fidelity, provenance, transport closeness.

        Fidelity matters as much as regime here: a dummy-weight anchor runs the
        same kernels but with synthetic MoE routing, so its decode curve is much
        flatter than a real-weights run. Ranking it below a real-weights anchor
        in the same regime is the difference between ~2% and ~30% error against
        measured serving. Provenance is the same argument one level down -- an
        offline anchor does not even run the served kernels.
        """
        dist = regime.regime_distance(recipe, dict(entry.get("regime") or {}))
        real = _anchor_is_real_weights(entry["path"])
        served = _anchor_is_served(entry["path"])
        transport = entry.get("transport") or {}
        gap = 0.0
        for axis in ("tp", "ep", "pp"):
            av, rv = transport.get(axis), recipe.get(axis)
            if av and rv:
                gap += abs(float(av) - float(rv))
        return (dist, 0 if real else 1, 0 if served else 1, gap)

    usable = [e for e in entries if anchor_curve_is_sane(e["path"]) and not _runtime_conflicts(e["path"], spec.runtime)]
    if not usable:
        return None
    best = min(usable, key=rank)
    dist, fidelity_rank, served_rank, _ = rank(best)
    return AnchorChoice(
        path=best["path"],
        regime_distance=int(dist),
        model=best.get("model"),
        needs_warmup=bool(dist),
        real_weights=(fidelity_rank == 0),
        served=(served_rank == 0),
    )


# A decode step at a larger batch does strictly more work, so measured decode
# latency must not fall as batch rises. Small drops are ordinary run-to-run
# noise; a large one means the measurement itself is broken (the harness times
# decode by differencing two generate() calls, which degenerates when the longer
# call is not actually longer). Such an artifact is not merely imprecise -- it
# poisons every projection that anchors on it, so it is rejected outright.
_ANCHOR_MONOTONIC_TOLERANCE = 0.15


def anchor_curve_is_sane(path: str) -> bool:
    """False when an artifact's measured decode curve is physically impossible."""
    try:
        with open(path) as fh:
            doc = json.load(fh) or {}
    except (OSError, ValueError):
        return False
    return anchor_curve_is_sane_doc(doc)


def anchor_curve_is_sane_doc(doc: Any) -> bool:
    """:func:`anchor_curve_is_sane` on an artifact already parsed."""
    if not isinstance(doc, dict):
        return False
    points = []
    for entry in doc.get("sweep") or []:
        try:
            batch, decode_ms = int(entry["batch"]), float(entry["decode_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if decode_ms <= 0.0:
            return False
        points.append((batch, decode_ms))
    if not points:
        return False
    # A single-batch anchor is narrow, not invalid: there is no monotonicity to
    # check, and it still calibrates the one operating point it measured.
    points.sort()
    peak = points[0][1]
    for _, decode_ms in points[1:]:
        if decode_ms < peak * (1.0 - _ANCHOR_MONOTONIC_TOLERANCE):
            return False
        peak = max(peak, decode_ms)
    return True


def _anchor_is_served(path: str) -> bool:
    """True when an anchor was measured against a real server, not offline vLLM.

    The distinction is not cosmetic. Given identical flags the offline ``LLM()``
    entrypoint and ``vllm serve`` resolve different attention and MoE kernels,
    and the served decode step ran 1.9x the offline one at concurrency 8 and
    5.5x at 128. Calibrating a served target from an offline anchor scored no
    better than not calibrating at all, where a served anchor scored 2.2%.
    """
    try:
        with open(path) as fh:
            meta = (json.load(fh) or {}).get("meta") or {}
    except (OSError, ValueError):
        return False
    return "serving" in str(meta.get("derived_from") or "").lower()


def _runtime_conflicts(path: str, runtime: dict[str, str]) -> bool:
    """True when an anchor recorded an image or engine version ``runtime`` contradicts.

    Only measured anchors record one. An axis either side leaves blank is not
    a conflict.
    """
    if not runtime:
        return False
    try:
        with open(path) as fh:
            recorded = ((json.load(fh) or {}).get("meta") or {}).get("runtime") or {}
    except (OSError, ValueError):
        return False
    return any(recorded.get(k) and v and str(recorded[k]) != str(v) for k, v in runtime.items())


def _anchor_is_real_weights(path: str) -> bool:
    """True when an anchor artifact was measured with real checkpoint weights."""
    try:
        with open(path) as fh:
            meta = (json.load(fh) or {}).get("meta") or {}
    except (OSError, ValueError):
        return False
    if meta.get("real_weights") is not None:
        return bool(meta["real_weights"])
    return str(meta.get("load_format") or "").lower() not in ("dummy", "")


def _resolve_workload_and_env(spec: ServingSpec) -> tuple[str, dict[str, str]]:
    """Return (workload_yaml_path, extra_env) for the projection.

    Precedence: an explicit ``HYPERLOOM_INFERASIM_WORKLOAD`` wins; otherwise a
    resolved preset name is fed to the bundled env-driven template via
    ``INFERASIM_MODEL``.
    """
    extra_env: dict[str, str] = {}
    explicit = os.environ.get(ENV_WORKLOAD)
    if explicit and Path(explicit).is_file():
        return str(Path(explicit).resolve()), extra_env

    preset = resolve_preset(spec.model_path)
    if not preset:
        raise InferasimBridgeError(
            f"could not resolve an InferaSim model preset for model={spec.model_path!r}; "
            f"set {ENV_MODEL}=<preset> or {ENV_WORKLOAD}=<workload.yaml>"
        )
    if not _TEMPLATE_WORKLOAD.is_file():
        raise InferasimBridgeError(f"bundled workload template missing: {_TEMPLATE_WORKLOAD}")
    # The template reads INFERASIM_MODEL/TP/PP/EP; parallelism is *also* forced via
    # CLI overrides below so an explicit workload YAML is honored too.
    extra_env["INFERASIM_MODEL"] = preset
    return str(_TEMPLATE_WORKLOAD), extra_env


def _purge_foreign_infera(root: str) -> None:
    """Drop cached ``infera*`` modules not originating from ``root``.

    Another ``infera`` checkout may already be importable on the default path
    (and may lack the ``projection`` subpackage). Once imported it is cached in
    ``sys.modules``, so a later ``sys.path`` insert cannot override the
    top-level package. Purge any cached ``infera`` whose file is outside our
    root so the re-import resolves against ``HYPERLOOM_INFERASIM_ROOT``.
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
    """Make ``infera.projection`` importable, honoring HYPERLOOM_INFERASIM_ROOT.

    When ``HYPERLOOM_INFERASIM_ROOT`` is set it takes precedence over any other
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
    except Exception as exc:
        raise InferasimBridgeError(
            f"cannot import Infera 'infera.projection' (set {ENV_ROOT} to the Infera "
            f"checkout or pip install amd-infera[projection]): {exc}"
        ) from exc


_DES_MIN_REQUESTS = 64
_DES_MAX_REQUESTS = 4000
_DES_REQUESTS_PER_CLIENT = 10
# Engines that schedule a prefill batch alone while the resident decodes wait.
_EXCLUSIVE_PREFILL_ENGINES = ("sglang", "atom")


def _des_argv(spec: ServingSpec) -> list[str]:
    """Replay the benchmark client: ``conc`` clients, each resubmitting on completion."""
    requests = spec.num_prompts or _DES_REQUESTS_PER_CLIENT * spec.conc
    requests = max(_DES_MIN_REQUESTS, 2 * spec.conc, min(_DES_MAX_REQUESTS, requests))
    argv = [
        "--des-closed-loop",
        "--des-num-requests",
        str(requests),
        "--des-seed",
        "0",
        # The client reports over every measured request; its warmup is a
        # separate burst the replay does not see.
        "--des-warmup-frac",
        "0",
    ]
    chunk = _parse_server_arg_int(spec.extra_server_args, "--chunked-prefill-size", "--max-num-batched-tokens")
    if chunk > 0:
        argv += ["--chunked-prefill-size", str(chunk), "--max-num-batched-tokens", str(chunk)]
    max_seqs = _parse_server_arg_int(spec.extra_server_args, "--max-num-seqs", "--max-running-requests")
    if max_seqs > 0:
        argv += ["--max-num-seqs", str(max_seqs)]
    if any(family in (spec.framework or "").lower() for family in _EXCLUSIVE_PREFILL_ENGINES):
        argv.append("--des-exclusive-prefill")
    return argv


def _build_argv(
    spec: ServingSpec,
    workload: str,
    anchor: AnchorChoice | None = None,
    *,
    mode: str | None = None,
    estimator_name: str | None = None,
) -> list[str]:
    """Build the ``inferasim inference`` argv for this serving spec."""
    mode = mode or projection_mode()
    arch = gpu_arch()
    hbm_gb = os.environ.get(ENV_HBM_GB) or _ARCH_HBM_GB.get(arch)
    serving_model = str(os.environ.get(ENV_SERVING_MODEL) or "continuous").lower()

    argv: list[str] = [
        "inference",
        "--config",
        workload,
        "--inference-mode",
        "both",
        "--profiling-mode",
        "simulate",
        "--serving-model",
        serving_model,
        "--input-len",
        str(spec.isl),
        "--output-len",
        str(spec.osl),
        "--inference-batch-size",
        str(spec.conc),
        "--max-concurrency",
        str(spec.conc),
        "--weight-dtype",
        spec.weight_dtype,
        "--kv-cache-dtype",
        spec.kv_cache_dtype,
        "--gpu-arch",
        arch,
    ]
    if (estimator_name or estimator()) == ESTIMATOR_DES:
        argv += _des_argv(spec)
    # Name the engine so InferaSim refuses a cross-engine anchor on its own,
    # rather than leaving ``select_anchor`` as the only thing standing between
    # a vLLM measurement and an SGLang candidate. Simulate mode is analytical
    # and returns the same number whichever engine is named, so this only ever
    # gates calibration -- which is what ``--load-benchmark`` below asks for.
    if spec.framework:
        argv += ["--serving-engine", spec.framework]
    if hbm_gb:
        argv += ["--hbm-capacity-gb", str(hbm_gb)]

    if mode == MODE_BENCHMARK:
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


_HARVEST_SCRIPT_REL = "infera/projection/core/projection/inference_projection/benchmark_vllm.py"
_DEFAULT_HARVEST_GPUS = 4
_DEFAULT_HARVEST_TIMEOUT_SEC = 1800
# Parallelism is chosen by the harvest itself (``--tp`` / ``--benchmark-gpus``);
# a width pinned in the deployment's server args would fight it.
_PARALLEL_FLAGS = frozenset(
    {
        "--tensor-parallel-size",
        "-tp",
        "--tp",
        "--tp-size",
        "--pipeline-parallel-size",
        "-pp",
        "--pp-size",
        "--data-parallel-size",
        "-dp",
        "--dp-size",
    }
)


def harvest_enabled() -> bool:
    """Harvest on a miss unless ``HYPERLOOM_INFERASIM_HARVEST`` turns it off."""
    raw = str(os.environ.get(ENV_HARVEST) or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _harvest_engine(framework: str) -> str | None:
    """The ``--serving-backend`` Infera's harvest launches for ``framework``.

    The harvest only knows vLLM, SGLang and ATOM; a build of one of them
    ("mori-sglang") is launched as its family.
    """
    fw = (framework or "").lower()
    for family in ("sglang", "atom", "vllm"):
        if family in fw:
            return family
    return None


def _strip_parallel_flags(server_args: str) -> str:
    toks = (server_args or "").split()
    out: list[str] = []
    skip = False
    for tok in toks:
        if skip:
            skip = False
            continue
        flag = tok.split("=", 1)[0]
        if flag in _PARALLEL_FLAGS:
            skip = "=" not in tok
            continue
        out.append(tok)
    return " ".join(out)


def _regime_key(spec: ServingSpec) -> str:
    """One harvest per regime and model; concurrent misses on it share the lock."""
    recipe = recipe_from_spec(spec)
    axes = {k: recipe.get(k) for k in ("model", "engine", "weight_dtype", "kv_cache_dtype", "attention_backend")}
    axes["speculative"] = recipe.get("speculative")
    return hashlib.sha256(json.dumps(axes, sort_keys=True).encode()).hexdigest()[:16]


@contextlib.contextmanager
def _regime_lock(store_root: str, key: str) -> Iterator[None]:
    lock_path = Path(store_root) / f".harvest-{key}.lock"
    with lock_path.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def harvest_command(spec: ServingSpec, save_path: str) -> list[str] | None:
    """The served-anchor harvest for ``spec``'s regime, or None if it cannot run."""
    root = os.environ.get(ENV_ROOT, "")
    script = Path(root) / _HARVEST_SCRIPT_REL
    engine = _harvest_engine(spec.framework)
    if not root or not script.is_file() or not engine or not spec.model_path:
        return None
    gpus = _as_int(os.environ.get(ENV_HARVEST_GPUS), 0) or _DEFAULT_HARVEST_GPUS
    gpus = max(1, min(gpus, spec.tp * spec.pp))
    python = os.environ.get(ENV_HARVEST_PYTHON) or sys.executable or "python3"
    cmd = [
        python,
        str(script),
        "--model",
        spec.model_path,
        "--tp",
        str(spec.tp),
        "--pp",
        str(spec.pp),
        "--benchmark-gpus",
        str(gpus),
        "--input-len",
        str(spec.isl),
        "--output-len",
        str(spec.osl),
        # Capture mode: the sweep follows the engine's CUDA-graph sizes up to
        # here, so the concurrency variants of this regime interpolate rather
        # than extrapolate from it.
        "--concurrency",
        str(min(1024, max(64, 2 * spec.conc))),
        # Real weights: dummy routing flattens the MoE decode curve, the
        # difference between ~2% and ~30% TPOT error on gpt-oss-120b.
        "--load-format",
        "auto",
        "--routing-dist",
        "none",
        "--serving-backend",
        engine,
        "--save",
        save_path,
    ]
    if spec.ep > 1:
        cmd.append("--enable-expert-parallel")
    server_args = _strip_parallel_flags(spec.extra_server_args)
    if server_args:
        cmd.append("--server-args=" + server_args)
    return cmd


def harvest_anchor(spec: ServingSpec) -> AnchorChoice | None:
    """Measure a served anchor for ``spec``'s regime, index it, and select it.

    Only runs in benchmark mode with an anchor store to write into. Holds a
    per-regime lock so concurrent misses on one regime boot one server, and
    re-checks the store under it in case another caller already harvested.
    Returns None whenever the harvest cannot run or does not produce an
    in-regime anchor; the caller then fails closed.
    """
    store_root = os.environ.get(ENV_ANCHOR_STORE)
    if not store_root or not harvest_enabled():
        return None
    Path(store_root).mkdir(parents=True, exist_ok=True)
    key = _regime_key(spec)
    with _regime_lock(store_root, key):
        existing = select_anchor(spec)
        if existing is not None and existing.regime_distance == 0:
            return existing
        save_path = str(Path(store_root) / "harvested" / f"{key}-tp{spec.tp}-c{spec.conc}.json")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = harvest_command(spec, save_path)
        if cmd is None:
            log.warning(
                "inferasim: cannot harvest an anchor for %s on %s (needs %s, a vllm/sglang/atom engine and a model path)",
                spec.model_path,
                spec.framework,
                ENV_ROOT,
            )
            return None
        timeout = _as_int(os.environ.get(ENV_HARVEST_TIMEOUT), 0) or _DEFAULT_HARVEST_TIMEOUT_SEC
        log.info("inferasim: no in-regime anchor for %s; harvesting one: %s", spec.model_path, " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("inferasim: anchor harvest did not run (%s)", exc)
            return None
        if proc.returncode != 0 or not Path(save_path).is_file():
            log.warning(
                "inferasim: anchor harvest failed rc=%s: %s", proc.returncode, (proc.stderr or proc.stdout or "")[-600:]
            )
            return None
        if not anchor_curve_is_sane(save_path):
            log.warning("inferasim: harvested anchor %s failed the decode-curve sanity check", save_path)
            return None
        _index_artifact(store_root, save_path)
        choice = select_anchor(spec)
    if choice is None or choice.regime_distance != 0:
        log.warning("inferasim: harvested anchor %s is not selected in-regime for %s", save_path, spec.model_path)
        return None
    return choice


def resolve_anchor(spec: ServingSpec, *, harvest: bool = True) -> AnchorChoice:
    """The in-regime anchor benchmark mode calibrates ``spec`` against.

    Harvests one on a miss when ``harvest``. Raises when none can be had:
    benchmark mode never falls back to an analytical number while still being
    labelled calibrated.
    """
    anchor = select_anchor(spec)
    if anchor is not None and anchor.regime_distance == 0:
        return anchor
    harvested = harvest_anchor(spec) if harvest else None
    if harvested is not None:
        return harvested
    raise InferasimBridgeError(
        f"benchmark mode but no in-regime anchor for {spec.model_path!r} "
        f"(engine={spec.framework}) could be found{' or harvested' if harvest else ''}"
    )


MEASURED_DERIVED_FROM = "serving benchmark (Hyperloom EXPLORE decision round; mean TPOT)"


def _measured_group_key(spec: ServingSpec, gpu: str) -> str:
    """One artifact per launch config; its concurrencies become one ladder."""
    ident = {
        "model": spec.model_path,
        "engine": spec.framework,
        "tp": spec.tp,
        "ep": spec.ep,
        "pp": spec.pp,
        "isl": spec.isl,
        "osl": spec.osl,
        "weight_dtype": spec.weight_dtype,
        "kv_cache_dtype": spec.kv_cache_dtype,
        "server_args": " ".join(sorted((spec.extra_server_args or "").split())),
        "gpu": gpu,
        "runtime": dict(sorted((spec.runtime or {}).items())),
    }
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:20]


def record_measured_anchor(
    spec: ServingSpec,
    *,
    tpot_mean_ms: float | None,
    gpu: str | None = None,
    source: dict[str, Any] | None = None,
) -> str | None:
    """Index a real decision round as a served anchor; return its path.

    The point is the run's mean TPOT at the batch it decoded at, which is how
    Infera's own served harvest derives a decode step. Prefill is left to the
    model: a TTFT under a full closed loop is mostly queue. Measurements of the
    same launch config at different concurrencies extend one artifact into a
    ladder; a re-measured concurrency replaces its point. No-op without an
    anchor store, or without a usable TPOT.
    """
    store_root = os.environ.get(ENV_ANCHOR_STORE)
    if not store_root or not tpot_mean_ms or tpot_mean_ms <= 0 or not spec.model_path:
        return None
    arch = gpu_arch(gpu)
    key = _measured_group_key(spec, arch)
    path = Path(store_root) / "measured" / f"{key}.json"
    method, k = parse_speculative(spec.extra_server_args)
    attn = _parse_server_arg_str(spec.extra_server_args, "--attention-backend")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _regime_lock(store_root, "measured-" + key):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                doc = {}
            points = {int(p["batch"]): p for p in (doc.get("sweep") or []) if p.get("batch")}
            points[spec.conc] = {"batch": spec.conc, "decode_ms": float(tpot_mean_ms)}
            sweep = [points[b] for b in sorted(points)]
            ref = sweep[-1]
            sources = list((doc.get("meta") or {}).get("sources") or [])
            if source:
                sources = (sources + [{"batch": spec.conc, **source}])[-32:]
            doc = {
                "backend": spec.framework,
                "client": "hyperloom-explore",
                "measured": {"model": {"decode_ms": ref["decode_ms"]}},
                "sweep": sweep,
                "meta": {
                    "model": spec.model_path,
                    "batch": ref["batch"],
                    "concurrency": ref["batch"],
                    "input_len": spec.isl,
                    "output_len": spec.osl,
                    "tp": spec.tp,
                    "ep": spec.ep,
                    "pp": spec.pp,
                    "target_tp": spec.tp,
                    "target_pp": spec.pp,
                    "benchmark_gpus": spec.tp * spec.pp,
                    "weight_dtype": spec.weight_dtype,
                    "kv_cache_dtype": spec.kv_cache_dtype,
                    "attention_backend": attn,
                    "enforce_eager": "--enforce-eager" in (spec.extra_server_args or "").split(),
                    "speculative_method": method or "",
                    "speculative_num_tokens": k or None,
                    "server_args": spec.extra_server_args,
                    "load_format": "auto",
                    "real_weights": True,
                    "gpu_arch": arch,
                    "runtime": dict(spec.runtime or {}),
                    "derived_from": MEASURED_DERIVED_FROM,
                    "sources": sources,
                },
            }
            if not anchor_curve_is_sane_doc(doc):
                log.info("inferasim: not indexing %s; its decode ladder is not monotonic", path)
                return None
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
            tmp.replace(path)
        _index_artifact(store_root, str(path))
    except OSError as exc:
        log.warning("inferasim: could not record a measured anchor under %s (%s)", store_root, exc)
        return None
    return str(path)


def _index_artifact(store_root: str, path: str) -> None:
    """Add ``path`` to the store's index; a store that cannot be opened finds it later."""
    try:
        _ensure_infera_importable()
        from infera.projection.core.projection.inference_projection.search.anchor_store import AnchorStore
    except Exception:  # noqa: BLE001 - AnchorStore.discover() indexes it on next open
        return
    with _regime_lock(store_root, "index"):
        AnchorStore(store_root, discover=False).add_artifact(path)


def project(spec: ServingSpec, *, mode: str | None = None) -> ProjMetrics:
    """Run the InferaSim projection for ``spec`` and return mapped metrics.

    ``mode`` defaults to :func:`resolve_mode` for ``spec``. Benchmark mode
    harvests on a miss only when it was asked for explicitly; ``auto`` never
    boots a server.
    """
    _ensure_infera_importable()
    workload, extra_env = _resolve_workload_and_env(spec)

    from infera.projection.cli import build_parser
    from infera.projection.core.projection.inference_projection import (
        launch_projection_from_cli,
    )

    configured = projection_mode()
    mode = mode or resolve_mode(spec)
    if mode == MODE_AUTO:
        mode = resolve_mode(spec)
    est = estimator()
    anchor = resolve_anchor(spec, harvest=configured == MODE_BENCHMARK) if mode == MODE_BENCHMARK else None
    argv = _build_argv(spec, workload, anchor, mode=mode, estimator_name=est)
    # Template reads INFERASIM_* env; also expose TP/PP/EP for template default
    # interpolation (overrides above still win for explicit workloads).
    prev_env: dict[str, str | None] = {}
    inject = dict(extra_env)
    inject.setdefault("INFERASIM_TP", str(spec.tp))
    inject.setdefault("INFERASIM_PP", str(spec.pp))
    inject.setdefault("INFERASIM_EP", str(spec.ep))
    for key, val in inject.items():
        prev_env[key] = os.environ.get(key)
        os.environ[key] = val
    try:
        args, overrides = build_parser().parse_known_args(argv)
        results = launch_projection_from_cli(args, overrides)
    except InferasimBridgeError:
        raise
    except Exception as exc:
        raise InferasimBridgeError(f"InferaSim projection failed: {exc}") from exc
    finally:
        for key, val in prev_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    perf = results.get("performance")
    if perf is None:
        raise InferasimBridgeError("InferaSim returned no performance projection")
    mem = results.get("memory")
    des = results.get("des")
    if est == ESTIMATOR_DES and not (des or {}).get("point"):
        # Ranking a replayed stack against a closed-form variant would compare
        # two estimators, not two configurations.
        raise InferasimBridgeError("InferaSim ran no closed-loop replay")

    return _metrics_from_results(spec, perf, mem, anchor, mode, des=des if est == ESTIMATOR_DES else None)


def _des_mean(stats: Any) -> float:
    return float((stats or {}).get("mean") or 0.0) if isinstance(stats, dict) else 0.0


def _metrics_from_results(
    spec: ServingSpec,
    perf: Any,
    mem: Any,
    anchor: AnchorChoice | None = None,
    mode: str = MODE_SIMULATE,
    *,
    des: dict[str, Any] | None = None,
) -> ProjMetrics:
    """Map InferaSim result objects onto benchmark measurement fields."""
    point = (des or {}).get("point")
    output_tps = float(getattr(perf, "decode_throughput_tps", 0.0) or 0.0)
    ttft_ms = float(getattr(perf, "ttft_ms", 0.0) or 0.0)
    tpot_ms = float(getattr(perf, "itl_ms", 0.0) or 0.0)
    itl_ms = tpot_ms
    e2el_ms = float(getattr(perf, "request_latency_ms", 0.0) or 0.0)
    if point is not None:
        output_tps = float(getattr(point, "system_throughput_tps", 0.0) or 0.0)
        # TTFT from when the client sent the request: under a closed loop the
        # server's queue is time the client is already counting.
        ttft_ms = _des_mean(getattr(point, "ttft_arrival", None)) or _des_mean(getattr(point, "ttft", None)) or ttft_ms
        # Per-request TPOT, which is what the benchmark client reports.
        tpot_ms = _des_mean(getattr(point, "tpot", None)) or tpot_ms
        itl_ms = _des_mean(getattr(point, "itl", None)) or tpot_ms
        e2el_ms = _des_mean(getattr(point, "e2e", None)) or e2el_ms
    osl = max(1, spec.osl)
    isl = max(1, spec.isl)
    request_tps = output_tps / osl if osl else 0.0
    total_tps = output_tps * (isl + osl) / osl if osl else output_tps

    mem_gb = 0.0
    if mem is not None:
        total_bytes = float(getattr(mem, "total_bytes", 0) or 0)
        mem_gb = total_bytes / (1024.0**3)
    extras = dict(getattr(perf, "extras", {}) or {})
    max_conc = int(extras.get("concurrency_used", 0) or extras.get("concurrency", 0) or spec.conc)
    extras["projection_mode"] = mode
    extras["estimator"] = ESTIMATOR_DES if point is not None else ESTIMATOR_ANALYTICAL
    if anchor is not None:
        # Provenance so a session can audit which warmup anchor served this
        # candidate and whether it stayed inside the anchor's regime.
        extras["anchor_path"] = anchor.path
        extras["anchor_regime_distance"] = anchor.regime_distance
        extras["anchor_needs_warmup"] = anchor.needs_warmup
        extras["anchor_real_weights"] = anchor.real_weights
        extras["anchor_served"] = anchor.served

    replica_gpus = int(getattr(perf, "replica_gpus", 0) or 0)
    extras["extrapolation"] = extrapolation_notes(spec, replica_gpus, bool(extras.get("benchmark_calibrated", 0.0)))

    return ProjMetrics(
        output_throughput=output_tps,
        request_throughput=request_tps,
        total_token_throughput=total_tps,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        itl_ms=itl_ms,
        e2el_ms=e2el_ms,
        decode_tps_per_gpu=float(getattr(perf, "decode_throughput_tps_per_gpu", 0.0) or 0.0),
        memory_per_gpu_gb=mem_gb,
        max_concurrency=max_conc,
        calibrated=bool(extras.get("benchmark_calibrated", 0.0)),
        replica_gpus=replica_gpus,
        extras=extras,
    )
