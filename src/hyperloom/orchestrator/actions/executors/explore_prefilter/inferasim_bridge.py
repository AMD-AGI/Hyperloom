# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferaSim projection bridge.

Maps a Hyperloom benchmark spec (the ``benchmark`` block of a materialized
Magpie YAML: framework/model/precision + TP/CONC/ISL/OSL envs) onto Infera's
``inferasim`` serving projection and returns projected throughput/latency.

The only consumer is EXPLORE's projection router (:mod:`_projection`),
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
* The entry point (:func:`inferasim_anchors.project`), anchor selection,
  harvesting and the measured anchors EXPLORE records live
  in :mod:`inferasim_anchors`.
* The projection is priced for the session's GPU (``ServingSpec.gpu``);
  ``HYPERLOOM_INFERASIM_GPU_ARCH`` only names the part when that is unknown.
* Throughput and latency come from Infera's discrete-event replay of a
  fixed-concurrency client (``HYPERLOOM_INFERASIM_ESTIMATOR=des``, the
  default), which ranked held-out configurations far better than the closed
  form did; ``analytical`` selects the closed form.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.framework_registry import server_args_env_name
from hyperloom.inference_optimizer.roofline_ceiling import HW_SPECS

from . import _reuse as reuse

# Env knobs (all optional unless noted). Documented in the module docstring and
# the runner --help.
ENV_ROOT = "HYPERLOOM_INFERASIM_ROOT"  # path to the Infera checkout (added to sys.path)
ENV_WORKLOAD = "HYPERLOOM_INFERASIM_WORKLOAD"  # explicit InferaSim workload YAML
ENV_MODEL = "HYPERLOOM_INFERASIM_MODEL"  # InferaSim model preset name (e.g. gpt_oss_120B)
ENV_GPU_ARCH = "HYPERLOOM_INFERASIM_GPU_ARCH"  # part to price when the session GPU is unknown
ENV_HBM_GB = "HYPERLOOM_INFERASIM_HBM_GB"  # per-GPU HBM capacity, GB
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
DTYPE_BF16 = "bf16"
DTYPE_FP8 = "fp8"
_DEFAULT_CONC = 64
_DEFAULT_SEQ_LEN = 1024
_DEFAULT_FRAMEWORK = "sglang"
_MAX_SEQS_FLAGS = ("--max-num-seqs", "--max-running-requests")
ATTENTION_BACKEND_FLAG = "--attention-backend"
_MINIMAX_M25 = "minimax_m2.5"

# Best-effort HF-path/name substring -> InferaSim megatron preset. The longest
# match wins; extend freely. Override any time with HYPERLOOM_INFERASIM_MODEL.
_MODEL_HEURISTICS: tuple[tuple[str, str], ...] = (
    ("gpt-oss-120b", "gpt_oss_120B"),
    ("gpt-oss-20b", "gpt_oss_20B"),
    ("gpt_oss_120b", "gpt_oss_120B"),
    ("gpt_oss_20b", "gpt_oss_20B"),
    ("minimax-m2.5", _MINIMAX_M25),
    ("minimax_m2.5", _MINIMAX_M25),
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
    ("deepseek-r1", "deepseek_v3"),
    ("deepseek-v3", "deepseek_v3"),
    ("deepseek-v2", "deepseek_v2"),
    ("minimax-m2", _MINIMAX_M25),
    ("mixtral-8x22b", "mixtral_8x22B_v0.1"),
    ("mixtral-8x7b", "mixtral_8x7B_v0.1"),
)

# Bundled env-driven workload template used when only a preset name is known.
_TEMPLATE_WORKLOAD = (
    Path(__file__).resolve().parents[4] / "inference_optimizer" / "assets" / "inferasim" / "inferasim_workload.yaml"
)


class InferasimBridgeError(RuntimeError):
    """Raised for any recoverable bridge failure (bad config, import, etc.)."""


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


@dataclass
class ServingSpec:
    """Normalized serving request extracted from a Hyperloom benchmark block."""

    framework: str
    model_path: str
    tp: int = 1
    ep: int = 1
    pp: int = 1
    conc: int = _DEFAULT_CONC
    isl: int = _DEFAULT_SEQ_LEN
    osl: int = _DEFAULT_SEQ_LEN
    weight_dtype: str = DTYPE_BF16
    kv_cache_dtype: str = DTYPE_BF16
    extra_server_args: str = ""
    num_prompts: int = 0
    # Image digest / engine version of the deployment, when known. A measured
    # anchor that recorded a different one is not used.
    runtime: dict[str, str] = field(default_factory=dict)
    # The session's GPU. Empty falls back to HYPERLOOM_INFERASIM_GPU_ARCH, then mi355x.
    gpu: str = ""
    # Digest of the launch levers the projection does not price (envs, kernel
    # overlays). A measured anchor recorded under another digest is not used.
    launch: str = ""


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
    if p in (DTYPE_FP8, "e4m3", "e5m2", "fp8_hybrid", "hybrid"):
        return DTYPE_FP8
    if p in ("mxfp4", "fp4"):
        return "mxfp4"
    return DTYPE_BF16


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
    if v.startswith(DTYPE_FP8):
        return DTYPE_FP8
    if v in (DTYPE_BF16, "bfloat16"):
        return DTYPE_BF16
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

    ``ambient=False`` reads the block alone, without the process env. That is what a caller comparing two
    materialized configs needs, since an exported ``CONC`` would otherwise
    project every variant at the same concurrency.
    """
    bench = bench or {}

    def override(name: str) -> str:
        return (os.environ.get(name) or "") if ambient else ""

    envs = dict(bench.get("envs") or {})
    framework = str(bench.get("framework") or _DEFAULT_FRAMEWORK).lower()
    model_path = str(bench.get("model") or override("MODEL"))
    # Infera names the engine as a free string, so a build ("mori-sglang") is a
    # regime of its own and reaches us as one. Resolving the args env through the
    # registry rather than an exact table means such a build still has its server
    # args read -- the hand-rolled table returned nothing for it, which silently
    # dropped the attention backend, KV dtype and speculative flags that decide
    # the regime.
    extra_args = str(_first_env_or(envs, server_args_env_name(framework), "", ambient=ambient))

    tp = _as_int(_first_env_or(envs, "TP", 1, ambient=ambient), 1)
    # EP/PP: from the server args, else 1.
    ep = _parse_server_arg_int(extra_args, "--ep-size", "--expert-parallel-size", "--moe-ep-size") or 1
    pp = _parse_server_arg_int(extra_args, "--pp-size", "--pipeline-parallel-size") or 1

    weight_dtype = _precision_to_weight_dtype(str(bench.get("precision") or DTYPE_BF16))
    # KV dtype: the server arg the variant sets, else bf16. Reading the arg matters because an explore grid changes the KV
    # dtype by passing this flag and nothing else -- and the projection prices KV
    # dtype perfectly well, so ignoring the flag made a lever the model *can*
    # see look like one it cannot, and projected an fp8 candidate as bf16.
    kv_dtype = str(_parse_kv_cache_dtype(extra_args) or DTYPE_BF16).lower()

    conc = max(1, _as_int(_first_env_or(envs, "CONC", _DEFAULT_CONC, ambient=ambient), _DEFAULT_CONC))
    # A running batch cannot exceed the scheduler's cap on concurrent sequences,
    # so a variant that lowers it lowers the batch the decode step actually runs.
    max_seqs = _parse_server_arg_int(extra_args, *_MAX_SEQS_FLAGS)
    if max_seqs and max_seqs > 0:
        conc = min(conc, max_seqs)

    return ServingSpec(
        framework=framework,
        model_path=model_path,
        tp=max(1, tp),
        ep=max(1, ep),
        pp=max(1, pp),
        conc=conc,
        isl=max(1, _as_int(_first_env_or(envs, "ISL", _DEFAULT_SEQ_LEN, ambient=ambient), _DEFAULT_SEQ_LEN)),
        osl=max(1, _as_int(_first_env_or(envs, "OSL", _DEFAULT_SEQ_LEN, ambient=ambient), _DEFAULT_SEQ_LEN)),
        weight_dtype=weight_dtype,
        kv_cache_dtype=kv_dtype,
        extra_server_args=extra_args,
        num_prompts=max(0, _as_int(_first_env_or(envs, "NUM_PROMPTS", 0, ambient=ambient), 0)),
        launch=reuse.launch_digest(envs, reuse.HARNESS_ENVS | {server_args_env_name(framework)}),
    )


def resolve_preset(model_path: str) -> str | None:
    """Best-effort map a model path/name to an InferaSim preset name."""
    explicit = os.environ.get(ENV_MODEL)
    if explicit and explicit.strip():
        return explicit.strip()
    key = re.sub(r"[^a-z0-9.]+", "-", (model_path or "").lower())
    # A distilled checkpoint names its teacher first; its architecture is the student.
    if "distill" in key:
        key = key.split("distill", 1)[1]
    hits = [(needle, preset) for needle, preset in _MODEL_HEURISTICS if needle in key]
    return max(hits, key=lambda hit: len(hit[0]))[1] if hits else None


def recipe_from_spec(spec: ServingSpec) -> dict[str, Any]:
    """Canonical InferaSim recipe dict for a Hyperloom serving spec."""
    attn = _parse_server_arg_str(spec.extra_server_args, ATTENTION_BACKEND_FLAG)
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
        "gpu_arch": gpu_arch(spec.gpu),
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
_EXCLUSIVE_PREFILL_ENGINES = (_DEFAULT_FRAMEWORK, "atom")


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
    max_seqs = _parse_server_arg_int(spec.extra_server_args, *_MAX_SEQS_FLAGS)
    if max_seqs > 0:
        argv += [_MAX_SEQS_FLAGS[0], str(max_seqs)]
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
    arch = gpu_arch(spec.gpu)
    hbm_gb = os.environ.get(ENV_HBM_GB) or (HW_SPECS.get(arch) or {}).get("hbm_gb")
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


def _des_mean(stats: Any) -> float:
    return float((stats or {}).get("mean") or 0.0) if isinstance(stats, dict) else 0.0


def _metrics_from_results(
    spec: ServingSpec, results: dict[str, Any], anchor: AnchorChoice | None = None, mode: str = MODE_SIMULATE
) -> ProjMetrics:
    """Map InferaSim's ``performance`` / ``memory`` / ``des`` results onto benchmark measurement fields."""
    perf, mem = results.get("performance"), results.get("memory")
    point = (results.get("des") or {}).get("point")
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
