#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ci/optimize_submit.py — Hyperloom CI variant of SaFE optimize_submit.

Submits SaFE inference optimization tasks. Reuses the same SaFE bearer token
as the rest of Hyperloom CI (CLAW_API_KEY).

Tracks the SaFE script's API contract (Primus-SaFE/scripts/optimize_submit.py
as of 2026-05-06):
  POST /api/v1/playground/models   body = {source, workspace, target.volume}
  GET  /api/v1/playground/models/{id}
  POST /api/v1/optimization/tasks  body = {modelId, mode=local, framework, ...}

Notes on tools / mode:
  - SaFE backend hard-codes Claw Tools=[16,18] for optimization tasks
    (apiserver/.../optimization/handler.go), so the client never sends a
    tools field. This is independent of the [67] used by Hyperloom's existing
    Claw-direct CI (ci-config.yaml) — different code path.
  - mode=local (default): prompt tells the agent "SandboxImage: ..." and the
    agent runs benchmarks directly in the sandbox.
  - mode=claw: prompt warns the agent it cannot reach /shared_nfs directly
    and must go through Claw (RayJob fan-out).

Usage:
  # Auto mode — single model
  python3 optimize_submit.py --model Qwen/Qwen3-8B

  # Auto mode — multiple models
  python3 optimize_submit.py --model Qwen/Qwen3-8B meta-llama/Llama-3.1-70B-Instruct

  # Auto mode — top-N from HuggingFace, filtered by size
  python3 optimize_submit.py --hf-top 10 --min-params 7

  # Dry run + write manifest for CI artifact
  python3 optimize_submit.py --hf-top 5 --dry-run --output-dir submit-output

Env vars (all optional, CLI flags take precedence):
  CLAW_API_KEY | SAFE_API_KEY        bearer token (ak-xxx)
  SAFE_BASE_URL | SAFE_API_URL       base URL (default: https://core42.example-internal-host.invalid)
  HARBOR_PREFIX                      image registry prefix
  HF_TOKEN                           HuggingFace token (gated models)
  SAFE_OPTIMIZE_WORKSPACE            override default 'core42-hyperloom'
  SAFE_OPTIMIZE_VOLUME               override default '/wekafs'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

log = logging.getLogger("optimize-submit")


# ── Defaults ────────────────────────────────────────────────────────────────────

DEFAULT_API_URL = "https://core42.example-internal-host.invalid"
# Two-workspace split due to conflicting K8s constraints: register =
# core42-hyperloom (has /wekafs RWX for downloads; shared volume is readable by
# all sandboxes); submit = core42-sandbox (only Sandbox-scoped workspaces can
# host the optimization task; core42-hyperloom rejects with Primus.00003).
# Needs SaFE selectLocalPath path-accessible fallback when submit != register,
# else submit_task 400s; workaround: set --submit-workspace == --register-workspace.
DEFAULT_REGISTER_WORKSPACE = "core42-hyperloom"
DEFAULT_SUBMIT_WORKSPACE = "core42-sandbox"
DEFAULT_VOLUME = "/wekafs"
DEFAULT_PROXY = "harbor.core42.example-internal-host.invalid/proxy"
# core42 is MI300X; override the Claw prompt-builder's wrong MI355X default so
# the prompt and TP policy use the right arch. Tool source paths are
# deliberately NOT pinned: install.sh clones writable per-session copies, and
# pinning the shared read-only /wekafs/hyperloom/InferenceX caused
# "Read-only file system" errors on first session. OOB/TraceLens are opt-in too.
DEFAULT_GPU_TYPE = "MI300X"
DEFAULT_GPU_PROFILE = "mi300x"
DEFAULT_KERNEL_BACKENDS = ["GEAK", "Claude Code", "Codex"]
DEFAULT_MAX_HOURS = 12.0
DEFAULT_TARGET_GAIN = 100.0
DEFAULT_RESULTS_PATH = "$RESULT_DIR"
DEFAULT_CONTEXT_RESERVE_TOKENS = 16

# Hardware facts from AMD Instinct datasheets. tp_thresholds_b is CI policy:
# MI300X baseline (32/128/256B) scaled by per-GPU HBM capacity.
GPU_PROFILES = {
    "mi300x": {
        "gpu_type": "MI300X",
        "llvm_target": "gfx942",
        "hbm_gb": 192,
        "hbm_bandwidth_tb_s": 5.3,
        "tp_thresholds_b": (32, 128, 256),
    },
    "mi325x": {
        "gpu_type": "MI325X",
        "llvm_target": "gfx942",
        "hbm_gb": 256,
        "hbm_bandwidth_tb_s": 6.0,
        "tp_thresholds_b": (43, 171, 341),
    },
    "mi355x": {
        "gpu_type": "MI355X",
        "llvm_target": "gfx950",
        "hbm_gb": 288,
        "hbm_bandwidth_tb_s": 8.0,
        "tp_thresholds_b": (48, 192, 384),
    },
}

_KERNEL_BACKEND_ALIASES = {
    "geak": "GEAK",
    "claude": "Claude Code",
    "claude-code": "Claude Code",
    "claude code": "Claude Code",
    "codex": "Codex",
    "cursor": "Cursor",
}


def normalize_gpu_profile(gpu_type: str | None, *, warn: bool = True) -> str | None:
    """Return the GPU profile key when ``gpu_type`` maps to a known CI profile."""
    raw = (gpu_type or "").strip()
    compact = re.sub(r"[^a-z0-9]", "", raw.lower())
    for key, profile in GPU_PROFILES.items():
        profile_type = str(profile["gpu_type"])
        aliases = {
            re.sub(r"[^a-z0-9]", "", key.lower()),
            re.sub(r"[^a-z0-9]", "", profile_type.lower()),
        }
        if compact in aliases or compact.endswith(key):
            return key
    if warn and raw:
        log.warning("unknown gpu_type=%r; using %s for TP policy only",
                    raw, GPU_PROFILES[DEFAULT_GPU_PROFILE]["gpu_type"])
    return None


def canonical_gpu_type(gpu_type: str | None) -> str:
    """Resolve a GPU type string to its canonical form.

    Args:
        gpu_type: Free-form GPU type/alias, or ``None``.

    Returns:
        The canonical GPU type from the matching profile, or the trimmed input
        (falling back to ``DEFAULT_GPU_TYPE``) when no profile matches.
    """
    profile_key = normalize_gpu_profile(gpu_type, warn=False)
    if profile_key:
        return str(GPU_PROFILES[profile_key]["gpu_type"])
    return (gpu_type or DEFAULT_GPU_TYPE).strip() or DEFAULT_GPU_TYPE

# Canonical prompt prefix (ci/prompt_prefix.txt) — single source of truth,
# also read by .github/workflows/optimize-submit.yml and used as the argparse
# default so direct CLI invocations get the same prefix.
_PROMPT_PREFIX_FILE = Path(__file__).resolve().parent / "prompt_prefix.txt"


def _load_default_prompt_prefix() -> str:
    """Resolve the default ``--prompt-prefix``: $SAFE_OPTIMIZE_PROMPT_PREFIX,
    else ci/prompt_prefix.txt, else empty string."""

    env_value = os.environ.get("SAFE_OPTIMIZE_PROMPT_PREFIX", "")
    if env_value:
        return env_value
    try:
        if _PROMPT_PREFIX_FILE.is_file():
            return _PROMPT_PREFIX_FILE.read_text(encoding="utf-8")
    except OSError:
        pass
    return ""


# RDMA bnxt_re tar package staged on WekaFS — same path the validated
# ci-config.yaml multi-node entries hand the agent. Kept as a module constant
# so the orchestrator (Claw-direct) and this SaFE path stay in sync.
_MULTINODE_BNXT_TAR = "/wekafs/primus/data/libbnxt/libbnxt_re-234.0.154.0.tar.gz"


def _multinode_prompt_suffix(nodes: int, rayjob_image: str) -> str:
    """RayJob topology block injected into the prompt when nodes > 1.

    The SaFE tasks body has no node-count field, so multi-node is expressed the
    same way the Claw-direct CI does: tell the agent to fan out to an N-node
    RayJob with the exact topology. Returns "" for single-node (unchanged path).
    """
    if nodes <= 1:
        return ""
    return (
        f"\n\nMulti-node run ({nodes} nodes): this model does NOT fit on a single "
        f"8-GPU sandbox node. You CANNOT benchmark it directly in the sandbox -- "
        f"launch it as a SaFE RayJob (Claw fan-out) and export NODES={nodes} "
        f"before starting `inference_optimizer optimize` so the Coordinator sizes "
        f"the cluster for {nodes}x8 GPUs (TP spans all nodes).\n"
        f"Task submission ({nodes}-node):\n"
        f"RayJob image: {rayjob_image}\n"
        f"RayJob resource per node: CPU=96, GPU=8, memory=1024Gi, ephemeralStorage=400Gi\n"
        f"RayJob node count: {nodes}\n"
        f"env:\n"
        f"- NODES={nodes}\n"
        f"- PATH_TO_BNXT_TAR_PACKAGE={_MULTINODE_BNXT_TAR}\n"
    )


def parse_kernel_backends(raw: str | None) -> list[str]:
    """Normalize user-facing kernel backend names for SaFE's API payload.

    Splits on commas/semicolons, lowercases, and maps each alias to its
    canonical SaFE name, de-duplicating while preserving order.

    Args:
        raw (str | None): Comma/semicolon-separated backend names, or None.

    Returns:
        list[str]: Canonical backend names, or the default list when ``raw`` is
            empty or yields nothing.

    Raises:
        ValueError: If a token is not a recognised backend alias.
    """

    if not raw:
        return list(DEFAULT_KERNEL_BACKENDS)
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        key = item.lower()
        normalized = _KERNEL_BACKEND_ALIASES.get(key)
        if normalized is None:
            raise ValueError(
                f"unknown kernel backend {item!r}; expected one of "
                "geak, claude, codex, cursor"
            )
        if normalized not in out:
            out.append(normalized)
    return out or list(DEFAULT_KERNEL_BACKENDS)

# Architectures well-supported by SGLang on ROCm 7.x.
SGLANG_ARCHS: set[str] = {
    "LlamaForCausalLM", "LlamaForCausalLMWithVisualEncoder",
    "Qwen2ForCausalLM", "Qwen3ForCausalLM",
    "Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM",
    "MistralForCausalLM", "MixtralForCausalLM",
    "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM",
    "GemmaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM",
    "InternLM2ForCausalLM", "InternLM3ForCausalLM",
    "Phi3ForCausalLM", "PhiForCausalLM",
    "GPTBigCodeForCausalLM", "FalconForCausalLM", "ChatGLMModel",
}

# Architectures that require vLLM (Lightning Attention, sparse, or special quant).
VLLM_REQUIRED_ARCHS: set[str] = {
    "MiniMaxText01ForCausalLM",
    "KimiForConditionalGeneration",
    "KimiK25ForConditionalGeneration",
}

# Quantization types that require vLLM.
VLLM_QUANT_TYPES: set[str] = {"mxfp4", "nvfp4", "int4", "gptq", "awq"}

# Arch suffixes marking a generative LM; used to filter out embedding/encoder/
# classifier models that sglang/vllm can't serve.
GENERATIVE_ARCH_SUFFIXES: tuple[str, ...] = (
    "ForCausalLM",
    "ForConditionalGeneration",
    "LMHeadModel",
    "ForSeq2SeqLM",
)


def is_generative_arch(arch: str) -> bool:
    """True if the HF arch is causal-LM-style inference; False for empty/unknown
    (better to skip than waste a sandbox slot)."""
    if not arch:
        return False
    return any(arch.endswith(s) for s in GENERATIVE_ARCH_SUFFIXES)


def _proxy() -> str:
    """Return the container registry proxy prefix.

    Returns:
        str: ``$HARBOR_PREFIX`` when set, otherwise the default proxy prefix.
    """
    return os.environ.get("HARBOR_PREFIX", DEFAULT_PROXY)


def _default_sglang_image() -> str:
    """Return the default SGLang server image.

    Returns:
        The pinned ``profilerfix`` SGLang image whose patched
        libamdhip64/libroctracer let rocprofiler capture kernels under
        ``HipGraphLaunch`` (issue #352).
    """
    # profilerfix: patched libamdhip64/libroctracer so rocprofiler captures
    # kernels under HipGraphLaunch (issue #352). Pre-profilerfix image (revert):
    # lmsysorg/sglang:v0.5.11-rocm720-mi30x
    return "primussafe/sglang:v0.5.11-rocm720-mi30x-profilerfix"


def _default_vllm_image() -> str:
    """Return the default vLLM server image.

    Returns:
        The proxy-qualified vLLM image (v0.19.0, one minor ahead of the
        InferenceX baseline to avoid v0.20 breakage).
    """
    # v0.19.0: one minor ahead of InferenceX baseline v0.17.0, avoiding v0.20 breakage.
    return f"{_proxy()}/vllm/vllm-openai-rocm:v0.19.0"


# ── HuggingFace client ──────────────────────────────────────────────────────────

class HuggingFaceClient:
    """Minimal HF API client for model metadata + top-models discovery."""

    BASE = "https://huggingface.co"

    def __init__(self, token: str = "", timeout: int = 15):
        """Initialise the HF client session.

        Args:
            token (str): Optional HuggingFace token for gated-model access.
            timeout (int): Per-request timeout in seconds.
        """
        self.timeout = timeout
        self._sess = requests.Session()
        self._sess.headers["User-Agent"] = "hyperloom-optimize-submit/1.0"
        if token:
            self._sess.headers["Authorization"] = f"Bearer {token}"

    def _get(self, path: str) -> dict | list:
        """GET a HuggingFace API path and return the parsed JSON body.

        Args:
            path (str): Path appended to the HF base URL.

        Returns:
            dict | list: The decoded JSON response.

        Raises:
            requests.HTTPError: If the response status indicates an error.
        """
        resp = self._sess.get(f"{self.BASE}{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def model_info(self, repo_id: str) -> dict:
        """Fetch a repo's model metadata from the HF API.

        Args:
            repo_id (str): HuggingFace repo id (e.g. ``Qwen/Qwen3-8B``).

        Returns:
            dict: The model-info JSON (pipeline_tag, safetensors, tags, etc.).
        """
        return self._get(f"/api/models/{repo_id}")  # type: ignore[return-value]

    def model_config(self, repo_id: str) -> dict:
        """Fetch a repo's ``config.json`` from the HF resolve endpoint.

        Args:
            repo_id (str): HuggingFace repo id.

        Returns:
            dict: The parsed ``config.json`` contents.
        """
        return self._get(f"/{repo_id}/resolve/main/config.json")  # type: ignore[return-value]

    def top_models(self, limit: int, min_params_b: float = 0.0) -> list[str]:
        """Return top-N text-generation repos by downloads, optionally size-filtered.

        Pool-then-filter: the listing API matches on tags only, so re-validate
        per-repo on pipeline_tag == "text-generation" AND a generative
        architectures[0] suffix; either failing (or a gated 401) → skip.
        """
        pool_size = max(limit * 10, 100)
        listing = self._get(
            f"/api/models?sort=downloads&direction=-1"
            f"&limit={pool_size}&filter=text-generation"
        )
        repos: list[str] = []
        for m in listing:  # type: ignore[union-attr]
            if len(repos) >= limit:
                break
            repo = m.get("modelId") or m.get("id", "")
            if not repo or "/" not in repo:
                continue

            try:
                info = self.model_info(repo)
            except Exception:
                continue  # gated / network error

            pipeline_tag = (info.get("pipeline_tag") or "").strip()
            if pipeline_tag and pipeline_tag != "text-generation":
                log.info("skip %s: pipeline_tag=%s (not text-generation)",
                         repo, pipeline_tag)
                continue

            if min_params_b > 0:
                total = (info.get("safetensors") or {}).get("total", 0)
                if (total / 1e9) < min_params_b:
                    continue

            # Final gate: config.json reachable AND architectures[0] generative.
            # Skip gated (401/403) or non-generative repos so the pool refills.
            try:
                cfg = self.model_config(repo)
            except Exception as e:
                log.info("skip %s: config.json unreachable (%s) — "
                         "likely a gated repo your HF_TOKEN hasn't been granted",
                         repo, e)
                continue
            arch = (cfg.get("architectures") or [""])[0]
            if not is_generative_arch(arch):
                log.info("skip %s: arch=%s is non-generative", repo, arch)
                continue

            repos.append(repo)
        return repos


# ── Auto-detection ──────────────────────────────────────────────────────────────

@dataclass
class DetectedConfig:
    """Auto-detected launch configuration for a model.

    Attributes:
        arch (str): HF ``architectures[0]`` class name.
        framework (str): Chosen serving framework (``sglang`` / ``vllm``).
        precision (str): Detected precision tag (e.g. ``FP8`` / ``INT4``).
        tp (int): Tensor-parallel size.
        concurrency (int): Benchmark concurrency.
        image (str): Container image to run.
        params_b (float): Parameter count in billions.
    """
    arch: str
    framework: str
    precision: str
    tp: int
    concurrency: int
    image: str
    params_b: float
    max_context_tokens: int


def _quant_type(config: dict) -> str:
    """Read the quantization tag from HF config.json (vendors disagree on the
    field name). Priority: quant_algo (NVIDIA modelopt — quant_method there is
    just the tool name) > quant_type > quantization_type > quant_method
    (current de-facto standard) > method. First non-empty wins, lowercased."""
    quant = config.get("quantization_config") or {}
    raw = (
        quant.get("quant_algo")
        or quant.get("quant_type")
        or quant.get("quantization_type")
        or quant.get("quant_method")
        or quant.get("method")
        or ""
    )
    return raw.lower() if isinstance(raw, str) else ""


def detect_framework(config: dict) -> str:
    """Choose the serving framework for a model from its config.

    vLLM is selected for architectures that require it or for quantization
    types it handles better; SGLang is used for known-good architectures;
    otherwise vLLM is the broader-support fallback.

    Args:
        config (dict): A HF ``config.json`` dict.

    Returns:
        str: ``"vllm"`` or ``"sglang"``.
    """
    arch = (config.get("architectures") or [""])[0]
    qt = _quant_type(config)
    if arch in VLLM_REQUIRED_ARCHS:
        return "vllm"
    if any(q in qt for q in VLLM_QUANT_TYPES):
        return "vllm"
    if arch in SGLANG_ARCHS:
        return "sglang"
    log.warning("unknown architecture %r — defaulting to vllm (broader support)", arch)
    return "vllm"


def detect_precision(config: dict) -> str:
    """Detect the serving precision from a model's quantization tag.

    Args:
        config (dict): A HF ``config.json`` dict.

    Returns:
        str: One of ``FP8`` / ``FP4`` / ``INT4``, defaulting to ``FP8`` for
            unquantized models on MI300X.
    """
    qt = _quant_type(config)
    if "fp8" in qt:   return "FP8"
    if "mxfp4" in qt: return "FP4"
    if "nvfp4" in qt: return "FP4"
    if "int4" in qt:  return "INT4"
    if "gptq" in qt:  return "INT4"
    if "awq" in qt:   return "INT4"
    return "FP8"  # unquantized default for MI300X


def detect_param_count(hf_info: dict, config: dict) -> float:
    """Estimate a model's parameter count in billions.

    Prefers the exact ``safetensors.total`` count from HF metadata; otherwise
    approximates from hidden size, layer count, and vocab size.

    Args:
        hf_info (dict): The HF model-info JSON.
        config (dict): The HF ``config.json`` dict.

    Returns:
        float: Parameter count in billions, or 0.0 when it cannot be estimated.
    """
    total = (hf_info.get("safetensors") or {}).get("total", 0)
    if total:
        return total / 1e9
    h = config.get("hidden_size", 0)
    n = config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)
    if h and n:
        return (12 * h * h * n + vocab * h) / 1e9
    return 0.0


def detect_max_context_tokens(config: dict) -> int:
    """Return the model context length from HF config.json when present."""
    candidates = []
    for key in ("max_position_embeddings", "max_sequence_length", "n_positions", "seq_length"):
        value = config.get(key)
        if isinstance(value, (int, float)) and value > 0:
            candidates.append(int(value))
    return min(candidates) if candidates else 0


def context_too_short(
    max_context_tokens: int,
    isl: int,
    osl: int,
    reserve_tokens: int = DEFAULT_CONTEXT_RESERVE_TOKENS,
) -> bool:
    """Return whether the model context cannot fit the requested workload.

    Args:
        max_context_tokens: Model's maximum context length; ``<= 0`` means
            unknown, in which case the check is skipped.
        isl: Input sequence length.
        osl: Output sequence length.
        reserve_tokens: Headroom kept free beyond input + output.

    Returns:
        ``True`` when the context is known and smaller than
        ``isl + osl + reserve_tokens``; otherwise ``False``.
    """
    if max_context_tokens <= 0:
        return False
    return max_context_tokens < (isl + osl + reserve_tokens)


def detect_tp(params_b: float, precision: str = "BF16",
              gpu_type: str | None = None) -> int:
    """Pick tensor parallelism from param count and GPU profile. precision is
    kept for API compatibility but unused — CI policy stays predictable across
    precision variants of the same model family."""
    if params_b <= 0:
        return 1
    profile_key = normalize_gpu_profile(gpu_type) or DEFAULT_GPU_PROFILE
    profile = GPU_PROFILES[profile_key]
    tp1_max, tp2_max, tp4_max = profile["tp_thresholds_b"]
    if params_b <= tp1_max:  return 1
    if params_b <= tp2_max:  return 2
    if params_b <= tp4_max:  return 4
    return 8


def detect_concurrency(tp: int, framework: str) -> int:
    """Pick a benchmark concurrency from tensor-parallel size and framework.

    Args:
        tp (int): Tensor-parallel size.
        framework (str): Serving framework (``vllm`` / ``sglang``).

    Returns:
        int: The chosen concurrency level.
    """
    if framework == "vllm":
        return 64 if tp <= 4 else 16
    return 64 if tp == 1 else 32 if tp <= 4 else 64


def _sglang_image_for(repo_id: str = "") -> str:
    """Pick the sglang image, honoring per-model baseline-arch needs.

    Default is the profilerfix image. MiMo-V2.x is the exception: the undated
    v0.5.11 profilerfix base does NOT register ``MiMoV2ForCausalLM`` (the
    server dies at model-loader registration, three baseline attempts in a
    row -> ``baseline_failed``), so it needs the image that carries the dated
    20260508 sglang arch. That image is profilerfix's two patched ROCm libs
    (libamdhip64/libroctracer, issue #352) layered onto the dated 20260508
    build, so rocprofiler kernel capture under HipGraphLaunch still works.
    Must be paired with ``--attention-backend triton`` (injected in
    ``_workload_envs.materialize_config_with_envs``) to dodge the aiter
    attention CUDA-graph-capture SIGABRT. Matched on the repo basename so it
    fires for the HF repo id (the /wekafs/<org>-<repo> local path is derived
    downstream from this same id).
    """
    basename = (repo_id or "").split("/")[-1].lower()
    if "mimo-v2" in basename:
        return "primussafe/sglang:v0.5.11-rocm720-mi30x-mimo-profilerfix"
    return _default_sglang_image()


def detect_image(framework: str, repo_id: str = "") -> str:
    """Select the server image for a framework and model.

    Args:
        framework: Serving framework (``vllm`` / ``sglang``).
        repo_id: Model repo id, used to honor per-model image overrides.

    Returns:
        The default vLLM image for ``vllm``; otherwise the SGLang image chosen
        by :func:`_sglang_image_for`.
    """
    return _default_vllm_image() if framework == "vllm" else _sglang_image_for(repo_id)


def auto_detect(hf: HuggingFaceClient, repo_id: str,
                gpu_type: str | None = None) -> DetectedConfig | None:
    """Derive a benchmark configuration from a model's HF metadata.

    Fetches model info and ``config.json`` and infers framework, precision,
    tensor parallelism, concurrency, image, and context limits.

    Args:
        hf: Hugging Face client used to fetch metadata.
        repo_id: Model repo id to inspect.
        gpu_type: Target GPU type, used for TP/profile selection.

    Returns:
        A :class:`DetectedConfig`, or ``None`` when the HF metadata cannot be
        fetched.
    """
    log.info("[%s] fetching HF metadata", repo_id)
    try:
        info = hf.model_info(repo_id)
        config = hf.model_config(repo_id)
    except Exception as e:
        log.error("[%s] HF fetch failed: %s", repo_id, e)
        return None

    arch = (config.get("architectures") or ["unknown"])[0]

    # Refuse non-generative repos even with explicit --model: sglang/vllm won't
    # start a server for them and the task would just burn a sandbox slot.
    if not is_generative_arch(arch):
        log.error("[%s] arch=%s is not a generative LM "
                  "(expected ForCausalLM / ForConditionalGeneration / LMHeadModel / ForSeq2SeqLM "
                  "suffix). Skipping — pass an actual causal-LM repo, or override "
                  "with --manual --framework vllm if you really want to try.",
                  repo_id, arch)
        return None
    pipeline_tag = (info.get("pipeline_tag") or "").strip()
    if pipeline_tag and pipeline_tag != "text-generation":
        log.error("[%s] pipeline_tag=%s is not 'text-generation' — skipping",
                  repo_id, pipeline_tag)
        return None

    framework = detect_framework(config)
    precision = detect_precision(config)
    params_b = detect_param_count(info, config)
    max_context_tokens = detect_max_context_tokens(config)
    tp = detect_tp(params_b, precision, gpu_type)
    conc = detect_concurrency(tp, framework)
    image = detect_image(framework, repo_id)

    cfg = DetectedConfig(
        arch=arch, framework=framework, precision=precision,
        tp=tp, concurrency=conc, image=image, params_b=params_b,
        max_context_tokens=max_context_tokens,
    )
    log.info("[%s] arch=%s params=%.1fB context=%d framework=%s precision=%s gpu=%s tp=%d conc=%d",
             repo_id, arch, params_b, max_context_tokens, framework, precision,
             canonical_gpu_type(gpu_type), tp, conc)
    return cfg


# ── SaFE client ─────────────────────────────────────────────────────────────────

class SafeOptimizeClient:
    """Thin wrapper for SaFE playground/optimization endpoints.

    Reuses the same bearer token as the rest of Hyperloom CI. The API contract
    here mirrors SaFE/scripts/optimize_submit.py (2026-05-06), in particular
    the ``target.volume`` field added to /api/v1/playground/models.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        register_workspace: str,
        submit_workspace: str,
        volume: str,
        timeout: int = 30,
        submit_workspaces_pool: list[str] | None = None,
    ):
        """Initialise the SaFE client and its authenticated HTTP session.

        Args:
            base_url (str): SaFE base URL (trailing slash trimmed).
            token (str): Bearer token for the Authorization header.
            register_workspace (str): Workspace where models are registered and
                downloaded (must allow RW writes to ``volume``).
            submit_workspace (str): Workspace where optimization tasks run (must
                allow the Sandbox scope).
            volume (str): Wekafs volume mounted RW in ``register_workspace``.
            timeout (int): Per-request timeout in seconds.
            submit_workspaces_pool (list[str] | None): Optional round-robin pool
                of submit workspaces; when set, each submit cycles through it.
        """
        self.base_url = base_url.rstrip("/")
        # Where the model registers + downloads (needs RW to the volume).
        self.register_workspace = register_workspace
        # Where the task is created (needs Sandbox scope); may equal register.
        self.submit_workspace = submit_workspace
        # Optional round-robin pool: each submit_task picks the next workspace,
        # letting a batch span multiple workspaces without manual splitting.
        self.submit_workspaces_pool = [
            w.strip() for w in (submit_workspaces_pool or []) if w and w.strip()
        ] or None
        self._submit_ws_counter = 0
        self.volume = volume
        self.timeout = timeout
        self._sess = requests.Session()
        self._sess.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        # Honor CA bundle env so corp proxies don't break HTTPS.
        self._sess.verify = os.environ.get(
            "SSL_CERT_FILE", os.environ.get("REQUESTS_CA_BUNDLE", True))

    def _request(self, method: str, path: str, body: dict | None = None,
                 timeout: float | None = None) -> dict:
        """Send an HTTP request to the SaFE API and return the parsed JSON.

        Args:
            method: HTTP method (e.g. ``GET``, ``POST``).
            path: API path appended to ``base_url``.
            body: Optional JSON request body.
            timeout: Optional per-request timeout; falls back to the client
                default.

        Returns:
            The decoded JSON response, or an empty dict when there is no body.

        Raises:
            RuntimeError: If the response status code is ``>= 400``.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self._sess.request(method, url, json=body,
                                  timeout=timeout or self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.content else {}

    def find_model(self, repo_id: str) -> dict | None:
        """Look up an existing SaFE Model by HF source URL, scoped to
        register_workspace (where the canonical Model CR + LocalPaths live)."""
        hf_url = f"https://huggingface.co/{repo_id}".rstrip("/")
        from urllib.parse import quote
        try:
            data = self._request(
                "GET",
                f"api/v1/playground/models?limit=200&workspace={quote(self.register_workspace)}",
            )
        except Exception as e:
            log.warning("list models failed: %s", e)
            return None
        for m in data.get("items", []):
            if (m.get("sourceURL") or "").rstrip("/") == hf_url:
                return m
        return None

    def register_model(
        self, repo_id: str, hf_token: str = "", local_path: str = "",
    ) -> str:
        """Register a model record with SaFE so submit_task has a model_id.

        local_path set → accessMode=local_path: SaFE skips its Download Job
        (phase=Ready immediately) since files are already on disk (prewarm path).
        local_path empty → accessMode=local: SaFE downloads from HF (slow
        fallback when prewarm can't run).
        """
        if local_path:
            # local_path mode bypasses SaFE's HF metadata fetch, so we MUST
            # provide displayName. SaFE feeds it into GenerateName → K8s
            # metadata.name, which must satisfy RFC 1123 (lowercase [a-z0-9-.],
            # 1-63 chars); sanitize here since the backend doesn't.
            import re
            raw = repo_id.split("/")[-1] or repo_id
            cleaned = re.sub(r"[^a-z0-9.-]+", "-",
                             raw.lower()).strip(".-") or "model"
            # Trim to 50 to leave headroom for GenerateName's -xxxxx suffix (max 63).
            display_name = cleaned[:50].rstrip(".-") or "model"
            body = {
                "displayName": display_name,
                "source": {
                    "url": repo_id,
                    "accessMode": "local_path",
                    "localPath": local_path,
                },
                "workspace": self.register_workspace,
            }
            log.info("[%s] register (local_path mode): workspace=%s "
                     "displayName=%s localPath=%s",
                     repo_id, self.register_workspace, display_name, local_path)
        else:
            body = {
                "source": {
                    "url": repo_id,
                    "accessMode": "local",
                    **({"token": hf_token} if hf_token else {}),
                },
                "workspace": self.register_workspace,
                "target": {"volume": self.volume},
            }
            log.info("[%s] register (local mode — SaFE will download): "
                     "workspace=%s volume=%s",
                     repo_id, self.register_workspace, self.volume)
        result = self._request("POST", "api/v1/playground/models", body)
        return result.get("id", "")

    def wait_ready(
        self, model_id: str, timeout_min: int = 480, poll_s: int = 30,
    ) -> bool:
        """Poll a SaFE model until it reaches the Ready phase.

        Args:
            model_id (str): SaFE model id to poll.
            timeout_min (int): Maximum minutes to wait before giving up.
            poll_s (int): Seconds between polls.

        Returns:
            bool: True once the model is Ready; False if it reaches Failed or
                the timeout elapses.
        """
        log.info("waiting for model %s to be Ready (timeout=%dm)", model_id, timeout_min)
        deadline = time.time() + timeout_min * 60
        last_phase = ""
        while time.time() < deadline:
            try:
                m = self._request("GET", f"api/v1/playground/models/{model_id}")
                phase = m.get("phase", "")
                if phase != last_phase:
                    log.info("model %s phase: %s", model_id, phase or "(empty)")
                    last_phase = phase
                if phase == "Ready":
                    return True
                if phase == "Failed":
                    log.error("model %s Failed: %s", model_id, m.get("message", ""))
                    return False
            except Exception as e:
                log.debug("phase poll error (will retry): %s", e)
            time.sleep(poll_s)
        log.error("model %s wait timed out after %dm", model_id, timeout_min)
        return False

    def submit_task(
        self,
        model_id: str,
        display_name: str,
        framework: str,
        precision: str,
        tp: int,
        concurrency: int,
        isl: int,
        osl: int,
        image: str | None,
        mode: str = "local",
        gpu_type: str | None = None,
        inferencex_path: str | None = None,
        oob_path: str | None = None,
        tracelens_root: str | None = None,
        prompt_prefix: str | None = None,
        prompt_suffix: str | None = None,
        kernel_backends: list[str] | None = None,
        max_hours: float | None = None,
        target_gain: float | None = None,
        results_path: str | None = None,
    ) -> dict:
        """Submit an optimization task to SaFE and return the API response.

        Builds the task request body from the model and benchmark parameters,
        choosing a target workspace (single or round-robin across the pool).

        Args:
            model_id: Registered SaFE model id.
            display_name: Human-readable task name.
            framework: Serving framework (``vllm`` / ``sglang``).
            precision: Model precision (e.g. ``BF16``, ``FP8``).
            tp: Tensor-parallel size.
            concurrency: Benchmark concurrency level.
            isl: Input sequence length.
            osl: Output sequence length.
            image: Server image to run, or ``None`` for the framework default.
            mode: Submission mode (e.g. ``local``).
            gpu_type: Target GPU type.
            inferencex_path: Optional InferenceX checkout path.
            oob_path: Optional out-of-box baseline path.
            tracelens_root: Optional TraceLens root path.
            prompt_prefix: Optional prompt prefix override.
            prompt_suffix: Optional prompt suffix override.
            kernel_backends: Optional list of kernel backends to enable.
            max_hours: Optional wall-clock budget for the task.
            target_gain: Optional target performance gain.
            results_path: Optional path where results should be written.

        Returns:
            The decoded API response for the submitted task.
        """
        # Pick the workspace: single submit_workspace, or round-robin across the
        # pool. Counter is per-instance, not thread-safe — fine since submit_task
        # runs serially (only wait_and_collect is parallel, after submit returns).
        if self.submit_workspaces_pool:
            chosen_ws = self.submit_workspaces_pool[
                self._submit_ws_counter % len(self.submit_workspaces_pool)
            ]
            self._submit_ws_counter += 1
            log.info("[submit] round-robin chose workspace=%s "
                     "(pool=%s, idx=%d)",
                     chosen_ws, ",".join(self.submit_workspaces_pool),
                     self._submit_ws_counter - 1)
        else:
            chosen_ws = self.submit_workspace
        body = {
            "displayName": display_name,
            "modelId": model_id,
            "workspace": chosen_ws,
            "mode": mode,
            "framework": framework,
            "precision": precision,
            "tp": tp,
            "ep": 1,
            "isl": isl,
            "osl": osl,
            "concurrency": concurrency,
            "kernelBackends": list(kernel_backends or DEFAULT_KERNEL_BACKENDS),
        }
        if max_hours and max_hours > 0:
            body["maxHours"] = max_hours
        if target_gain and target_gain > 0:
            body["targetGain"] = target_gain
        if results_path:
            body["resultsPath"] = results_path
        if image:
            body["image"] = image
        # Override SaFE's wrong-for-core42 MI355X default (see DEFAULT_GPU_TYPE).
        if gpu_type:
            body["gpuType"] = gpu_type
        # Always send inferencexPath (even empty) to suppress SaFE's Zod default
        # "/hyperloom/InferenceX"; empty lets install.sh clone a writable copy.
        body["inferencexPath"] = inferencex_path or ""
        if oob_path:
            body["oobPath"] = oob_path
        if tracelens_root:
            body["tracelensRoot"] = tracelens_root
        # Optional prefix/suffix forwarded to BuildHyperloomPrompt on the SaFE side.
        if prompt_prefix:
            body["promptPrefix"] = prompt_prefix
        if prompt_suffix:
            body["promptSuffix"] = prompt_suffix
        attempts = 8
        for attempt in range(1, attempts + 1):
            try:
                # The submit POST can be slow when the core42 apiserver is
                # under load from many parallel daily jobs. Give it a generous
                # read timeout so a busy-but-alive backend doesn't trip the
                # default 30s and get misreported as a hard submit failure.
                return self._request(
                    "POST", "api/v1/optimization/tasks", body, timeout=120)
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                transient = (
                    "HTTP 500" in msg
                    or "HTTP 502" in msg
                    or "HTTP 503" in msg
                    or "HTTP 504" in msg
                    or "timed out" in low      # requests ReadTimeout/ConnectTimeout
                    or "timeout" in low
                    or "connection" in low     # ConnectionError / HTTPSConnectionPool
                )
                if not transient or attempt >= attempts:
                    raise
                delay = random.uniform(10, 60)
                log.warning(
                    "[submit] transient SaFE/Claw submit failure "
                    "(attempt %d/%d, workspace=%s); retrying in %.1fs: %s",
                    attempt,
                    attempts,
                    chosen_ws,
                    delay,
                    msg,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable submit retry loop exit")

    # ── Task lifecycle ──

    # Lifecycle states from SaFE types.go OptimizationTaskStatus.
    TERMINAL_TASK_STATUSES = {"Succeeded", "Failed", "Interrupted"}

    def get_task(self, task_id: str) -> dict:
        """Fetch the current state of an optimization task.

        Args:
            task_id (str): SaFE optimization task id.

        Returns:
            dict: The task record JSON.
        """
        return self._request("GET", f"api/v1/optimization/tasks/{task_id}")

    def wait_task_done(
        self, task_id: str, timeout_min: int = 480, poll_s: int = 60,
    ) -> tuple[str, dict]:
        """Wait until the task reaches a terminal status. Returns (status, last_task).

        Prefer the Claw SSE stream (SaFE's task status lags Claw by minutes), and
        fall back to SaFE polling when no clawSessionId exists or SSE fails.
        Returns ('Timeout', {}) if neither sees a terminal status by the deadline.
        """
        log.info("[task %s] waiting for completion (timeout=%dm, poll=%ds)",
                 task_id, timeout_min, poll_s)
        deadline = time.time() + timeout_min * 60

        # Wait briefly (cap 60s) for clawSessionId to materialize, else fall
        # through to polling.
        sid = None
        for _ in range(12):
            sid = self._claw_session_id_for(task_id)
            if sid:
                break
            time.sleep(5)

        sse_used = False
        if sid:
            sse_used = True
            log.info("[task %s] using SSE on clawSessionId=%s", task_id, sid[:8])
            sse_reason = self._sse_wait_until_done(sid, deadline)
            log.info("[task %s] SSE finished: reason=%s", task_id, sse_reason)
            try:
                last_task = self.get_task(task_id)
            except Exception:
                last_task = {}
            sf_status = last_task.get("status", "") if last_task else ""
            if sf_status in self.TERMINAL_TASK_STATUSES:
                return sf_status, last_task
            # Stopped = sandbox pod exited (real end-of-task). SaFE's controller
            # lags shutdown by 10-180s, so short-poll up to 5min for its verdict.
            if sse_reason == "Stopped":
                log.info("[task %s] sandbox stopped — short-polling SaFE "
                         "for terminal status (up to 5min)", task_id)
                for _ in range(30):
                    time.sleep(10)
                    if time.time() > deadline:
                        break
                    try:
                        last_task = self.get_task(task_id)
                    except Exception:
                        continue
                    sf_status = last_task.get("status", "") if last_task else ""
                    if sf_status in self.TERMINAL_TASK_STATUSES:
                        log.info("[task %s] SaFE settled on %s after sandbox stop",
                                 task_id, sf_status)
                        return sf_status, last_task
                # Sandbox gone but SaFE hasn't settled — treat as Succeeded so
                # collect_artifacts can still read whatever the agent wrote.
                log.info("[task %s] SaFE never settled within 5min after "
                         "sandbox stop — returning Succeeded (collect "
                         "step will read ci_metrics.json directly)",
                         task_id)
                return "Succeeded", last_task
            if sse_reason == "deadline":
                return "Timeout", last_task
            # idle_timeout / stream_error are inconclusive (the SSE stream goes
            # quiet during long tool calls), so fall through to SaFE polling.
            log.info("[task %s] SSE inconclusive (reason=%s, sf_status=%s) — "
                     "falling back to SaFE polling for terminal status",
                     task_id, sse_reason, sf_status or "?")

        # Fallback / continuation: SaFE optimization-API polling.
        if not sse_used:
            log.info("[task %s] no clawSessionId yet — using SaFE polling", task_id)
        last_status = ""
        last_phase = -1
        last_task: dict = {}
        while time.time() < deadline:
            try:
                t = self.get_task(task_id)
                last_task = t
                status = t.get("status", "")
                phase = t.get("currentPhase", -1)
                if status != last_status or phase != last_phase:
                    log.info("[task %s] status=%s phase=%s message=%s",
                             task_id, status or "?", phase, (t.get("message") or "")[:120])
                    last_status, last_phase = status, phase
                if status in self.TERMINAL_TASK_STATUSES:
                    return status, t
            except Exception as e:
                log.debug("[task %s] poll error (will retry): %s", task_id, e)
            time.sleep(poll_s)
        log.warning("[task %s] wait timed out after %dm", task_id, timeout_min)
        return "Timeout", last_task

    def _sse_wait_until_done(self, session_id: str, deadline: float) -> str:
        """Subscribe to the Claw session SSE stream, return when the agent ends.

        IMPORTANT: do NOT return on ResultMessage — it fires at the end of EVERY
        agent turn (dozens over 1-3h), so the first turn was being mistaken for
        completion. The only reliable signal is sandboxStatus
        phase=Stopped/Terminated/Failed (sandbox pod actually exits).

        Returns: "Stopped" | "idle_timeout" (no events >10min) | "deadline"
        (per-task wall clock) | "stream_error" (caller falls back).
        """
        url = f"{self.base_url}/claw-api/v1/chat/sessions/{session_id}/messages"
        last_evt = time.time()
        idle_grace_s = 600  # 10 min of pure keepalive after the last event
        try:
            with self._sess.get(url, stream=True, timeout=(10, 60)) as r:
                if not r.ok:
                    log.warning("SSE stream HTTP %d for session %s",
                                r.status_code, session_id[:8])
                    return "stream_error"
                current_event = None
                for raw in r.iter_lines(decode_unicode=True):
                    now = time.time()
                    if now > deadline:
                        return "deadline"
                    if raw is None:
                        continue
                    if raw == "" or raw.startswith(":"):
                        # Blank-line separator or `: keepalive` heartbeat.
                        if now - last_evt > idle_grace_s:
                            return "idle_timeout"
                        continue
                    if raw.startswith("id:"):
                        continue
                    if raw.startswith("event:"):
                        current_event = raw[6:].strip()
                        continue
                    if not raw.startswith("data:"):
                        continue
                    payload = raw[5:].strip()
                    try:
                        d = json.loads(payload)
                    except Exception:
                        continue
                    last_evt = now
                    et = d.get("type") or current_event
                    # ResultMessage is intentionally not a return signal (see
                    # docstring); it still refreshes last_evt above.
                    if et == "sandboxStatus":
                        ph = (d.get("phase") or "").lower()
                        if ph in ("stopped", "terminated", "failed"):
                            return "Stopped"
        except requests.exceptions.RequestException as e:
            log.warning("SSE stream error for session %s: %s",
                        session_id[:8], type(e).__name__)
            return "stream_error"
        except Exception as e:
            log.warning("SSE stream unexpected error for session %s: %s",
                        session_id[:8], e)
            return "stream_error"
        # Stream closed cleanly without a Stopped signal — treat as idle.
        return "idle_timeout"

    def _claw_session_id_for(self, task_id: str) -> str | None:
        """Resolve clawSessionId for a task (per-instance cached). None when
        SaFE has no session attached (e.g. task failed before session creation)."""
        if not hasattr(self, "_claw_session_cache"):
            self._claw_session_cache = {}
        if task_id in self._claw_session_cache:
            return self._claw_session_cache[task_id]
        try:
            t = self.get_task(task_id)
        except Exception as e:
            log.warning("[task %s] get_task failed while resolving clawSessionId: %s",
                        task_id, e)
            return None
        sid = (t.get("clawSessionId") or "").strip()
        self._claw_session_cache[task_id] = sid or None
        return sid or None

    def list_artifacts(self, task_id: str) -> list[dict]:
        """List task artifacts via the SaFE standard endpoint.

        Returns items shaped {path, size, lastModified, downloadPath}.
        downloadPath is server-relative; download_artifact() uses it when present,
        else builds the /artifacts/download?path= URL from path.
        """
        try:
            data = self._request(
                "GET", f"api/v1/optimization/tasks/{task_id}/artifacts")
        except Exception as e:
            log.warning("[task %s] list_artifacts failed: %s", task_id, e)
            return []
        items = (data.get("items") or data.get("data")) if isinstance(data, dict) else data
        if isinstance(items, dict) and isinstance(items.get("items"), list):
            items = items["items"]
        if not isinstance(items, list):
            return []
        return items

    def download_artifact(self, task_id: str, path_or_item: "str | dict") -> bytes:
        """Download a single task artifact. Accepts a string path or a
        list_artifacts item dict; prefers the item's downloadPath when present."""
        if isinstance(path_or_item, dict):
            download_path = (path_or_item.get("downloadPath") or "").strip()
            if download_path:
                url = f"{self.base_url}/{download_path.lstrip('/')}"
            else:
                path = path_or_item.get("path", "")
                encoded = requests.utils.quote(path, safe="")
                url = (f"{self.base_url}/api/v1/optimization/tasks/{task_id}"
                       f"/artifacts/download?path={encoded}")
        else:
            encoded = requests.utils.quote(path_or_item, safe="")
            url = (f"{self.base_url}/api/v1/optimization/tasks/{task_id}"
                   f"/artifacts/download?path={encoded}")
        resp = self._sess.get(url, timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"SaFE artifact download -> HTTP {resp.status_code}: "
                f"{resp.text[:200]}")
        return resp.content

    def download_artifact_to(
        self, task_id: str, path_or_item: "str | dict", local_path: str,
    ) -> int:
        """Download an artifact and write it to a local file.

        Creates parent directories as needed.

        Args:
            task_id (str): SaFE optimization task id.
            path_or_item (str | dict): Artifact path string or item dict.
            local_path (str): Destination file path.

        Returns:
            int: Number of bytes written.
        """
        data = self.download_artifact(task_id, path_or_item)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        return len(data)


# ── Per-model record ────────────────────────────────────────────────────────────

@dataclass
class SubmissionRecord:
    """Per-model record tracking submission, completion, and CI delivery.

    Accumulates state across the submit → wait → collect pipeline and is
    serialised into the submission manifest.

    Attributes:
        model (str): HuggingFace repo id for this entry.
        status (str): Local submit stage (submitted/dry-run/skipped/failed).
        task_id (str | None): SaFE optimization task id once submitted.
        claw_session_id (str | None): Claw session UUID SaFE created.
        display_name (str | None): Task display name.
        model_path (str | None): Resolved model path reported by SaFE.
        safe_user_id (str | None): SaFE user id owning the task.
        safe_started_at (str | None): SaFE task start timestamp.
        safe_finished_at (str | None): SaFE task finish timestamp.
        detected (dict | None): Auto-detected config as a dict.
        overrides (dict): User/CLI overrides applied.
        pool (dict): Production-pool audit metadata.
        error (str | None): Error message when submit failed/skipped.
        category (str | None): Coarse model shape (moe/dense/"").
        sandbox_duration_seconds (float | None): SaFE wallclock duration.
        final_status (str | None): SaFE terminal status.
        final_phase (int | None): Current phase at the terminal moment.
        final_message (str | None): SaFE task message.
        ci_status (str | None): CI delivery status (separate from SaFE status).
        ci_success (bool): Whether usable artifacts were delivered.
        delivery_reason (str | None): Explanation for the CI delivery status.
        artifacts_dir (str | None): Local directory artifacts landed in.
        artifact_count (int): Number of collected artifacts.
        artifact_files (list[str]): Collected artifact file paths.
        artifact_sources (list[dict]): Provenance entries per artifact.
    """
    model: str
    status: str = "pending"            # local stage: submitted/dry-run/skipped/failed
    task_id: str | None = None
    # Claw session UUID SaFE creates at submit; used to correlate ci_metrics.json
    # under /wekafs/users/<uid>/<session>/ with the task (set in wait_and_collect_one).
    claw_session_id: str | None = None
    display_name: str | None = None
    model_path: str | None = None
    safe_user_id: str | None = None
    safe_started_at: str | None = None
    safe_finished_at: str | None = None
    detected: dict | None = None
    overrides: dict = field(default_factory=dict)
    pool: dict = field(default_factory=dict)
    error: str | None = None
    # Audit fields so each persisted artifact is self-describing.
    category: str | None = None             # moe / dense / "" — from detected.arch
    sandbox_duration_seconds: float | None = None  # SaFE startedAt -> finishedAt
    final_status: str | None = None    # SaFE: Succeeded/Failed/Interrupted/Timeout
    final_phase: int | None = None     # currentPhase at terminal moment
    final_message: str | None = None   # task.Message
    # CI delivery status is separate from SaFE final_status: a SaFE timeout may
    # still have written a useful session_breakdown worth publishing.
    ci_status: str | None = None        # Delivered / Missing artifacts / ...
    ci_success: bool = False
    delivery_reason: str | None = None
    artifacts_dir: str | None = None   # local dir where artifacts landed
    artifact_count: int = 0
    artifact_files: list[str] = field(default_factory=list)
    artifact_sources: list[dict] = field(default_factory=list)


# ── Per-model flow ──────────────────────────────────────────────────────────────

def process_model(
    repo_id: str,
    hf: HuggingFaceClient,
    safe: SafeOptimizeClient,
    overrides: dict,
    isl: int,
    osl: int,
    dry_run: bool,
    hf_token: str,
    manual_mode: bool,
    mode: str,
    gpu_type: str | None = None,
    inferencex_path: str | None = None,
    oob_path: str | None = None,
    tracelens_root: str | None = None,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    kernel_backends: list[str] | None = None,
    max_hours: float | None = None,
    target_gain: float | None = None,
    results_path: str | None = None,
    pool_metadata: dict | None = None,
) -> SubmissionRecord:
    """Run the full submit flow for one model: detect, register, submit.

    Auto-detects (or uses manual overrides for) the launch config, ensures the
    model is registered and Ready in SaFE (preferring prewarmed local_path
    mode), then submits the optimization task. Short-circuits for dry-run and
    for skip/failure conditions.

    Args:
        repo_id (str): HuggingFace repo id.
        hf (HuggingFaceClient): Client for HF metadata lookups.
        safe (SafeOptimizeClient): Client for SaFE register/submit calls.
        overrides (dict): Field overrides (framework/precision/tp/...).
        isl (int): Input sequence length.
        osl (int): Output sequence length.
        dry_run (bool): When True, plan only without registering/submitting.
        hf_token (str): HuggingFace token for gated downloads.
        manual_mode (bool): Skip auto-detect; requires ``framework`` override.
        mode (str): Execution mode passed to SaFE (``local`` / ``claw``).
        gpu_type (str | None): GPU type override for the prompt.
        inferencex_path (str | None): InferenceX checkout path override.
        oob_path (str | None): OOB checkout path override.
        tracelens_root (str | None): TraceLens checkout path override.
        prompt_prefix (str | None): Prompt prefix forwarded to SaFE.
        prompt_suffix (str | None): Prompt suffix forwarded to SaFE.
        kernel_backends (list[str] | None): Kernel optimization backends.
        max_hours (float | None): Max optimization wall-clock hours.
        target_gain (float | None): Target gain percentage.
        results_path (str | None): Results path passed to the prompt builder.
        pool_metadata (dict | None): Production-pool audit metadata.

    Returns:
        SubmissionRecord: The record describing the outcome of this model.
    """
    rec = SubmissionRecord(
        model=repo_id,
        overrides={k: v for k, v in overrides.items() if v is not None},
        pool={k: v for k, v in (pool_metadata or {}).items() if v not in (None, "")},
    )
    gpu_type = canonical_gpu_type(gpu_type)
    gpu_profile = normalize_gpu_profile(gpu_type, warn=False) or DEFAULT_GPU_PROFILE

    detected = None if manual_mode else auto_detect(hf, repo_id, gpu_type=gpu_type)
    if not detected and not manual_mode:
        rec.status = "skipped"
        rec.error = "auto-detect failed"
        return rec
    if manual_mode and not overrides.get("framework"):
        rec.status = "skipped"
        rec.error = "manual mode requires --framework"
        return rec
    if detected:
        rec.detected = asdict(detected)
        rec.category = _category_from_arch(rec.detected.get("arch", ""))
        max_context = int(rec.detected.get("max_context_tokens") or 0)
        if context_too_short(max_context, isl, osl):
            required = isl + osl + DEFAULT_CONTEXT_RESERVE_TOKENS
            rec.status = "skipped"
            rec.error = (
                "context_too_short: "
                f"max_context_tokens={max_context} < required={required} "
                f"(isl={isl}, osl={osl}, reserve={DEFAULT_CONTEXT_RESERVE_TOKENS})"
            )
            log.warning("[%s] skipping: %s", repo_id, rec.error)
            return rec

    framework = overrides.get("framework") or (detected.framework if detected else "")
    precision = overrides.get("precision") or (detected.precision if detected else "FP8")
    tp        = overrides.get("tp")        or (detected.tp if detected else 1)
    conc      = overrides.get("concurrency") or (detected.concurrency if detected else 64)
    image     = overrides.get("image") or (detected.image if detected else detect_image(framework, repo_id))

    log.info("[%s] => mode=%s framework=%s precision=%s tp=%d conc=%d image=%s",
             repo_id, mode, framework, precision, tp, conc, image)

    display_name = f"{repo_id.split('/')[-1]}-{precision.lower()}-{framework}-{gpu_profile}"
    rec.display_name = display_name
    rec.overrides["mode"] = mode
    if kernel_backends:
        rec.overrides["kernel_backends"] = kernel_backends
    if max_hours:
        rec.overrides["max_hours"] = max_hours
    if target_gain:
        rec.overrides["target_gain"] = target_gain
    if results_path:
        rec.overrides["results_path"] = results_path

    if dry_run:
        rec.status = "dry-run"
        return rec

    # If prewarm already populated /wekafs/models/<slug>/, use local_path mode so
    # SaFE sets phase=Ready immediately without re-downloading over our files.
    nfs_root = os.environ.get("NFS_ROOT", "/wekafs")
    target_slug = repo_id.replace("/", "-")
    target_dir = f"{nfs_root}/models/{target_slug}"
    use_local_path = False
    try:
        if os.path.isdir(target_dir):
            # Heuristic floor: any real HF repo has >=5 files (config + tokenizer
            # + weight shard).
            n_files = sum(1 for _ in os.scandir(target_dir))
            if n_files >= 5:
                use_local_path = True
                log.info("[%s] prewarm complete (%d files at %s) — registering "
                         "via local_path mode (skips SaFE Download Job)",
                         repo_id, n_files, target_dir)
            else:
                log.info("[%s] %s has only %d entries — falling back to SaFE "
                         "download via accessMode=local", repo_id, target_dir, n_files)
    except OSError as e:
        log.warning("[%s] could not probe %s: %s — falling back to SaFE download",
                    repo_id, target_dir, e)

    # Find existing SaFE record OR register fresh. Stale phase=Failed records
    # (aborted prior download) would make wait_ready return False forever, so we
    # re-register: SaFE issues a new model_id or resets to Pending, and the
    # Download Job now sees prewarmed files and finishes in seconds.
    safe_model = safe.find_model(repo_id)
    if safe_model and safe_model.get("phase") != "Failed":
        model_id = safe_model["id"]
        phase = safe_model.get("phase", "")
        log.info("[%s] found in SaFE: id=%s phase=%s", repo_id, model_id, phase)
        if phase != "Ready" and not safe.wait_ready(model_id):
            rec.status = "failed"
            rec.error = "model never reached Ready"
            return rec
    else:
        if safe_model:
            log.info("[%s] existing model %s is %s — re-registering "
                     "(prewarm should have populated /wekafs/models/ already)",
                     repo_id, safe_model.get("id"), safe_model.get("phase"))
        try:
            model_id = safe.register_model(
                repo_id, hf_token,
                local_path=target_dir if use_local_path else "",
            )
        except Exception as e:
            rec.status = "failed"
            rec.error = f"register: {e}"
            return rec
        if not model_id:
            rec.status = "failed"
            rec.error = "register returned empty id"
            return rec
        if safe_model and model_id == safe_model.get("id"):
            log.warning("[%s] SaFE returned the same id %s as the existing "
                        "Failed record — backend deduped by sourceURL and did "
                        "not reset phase. DELETE the record manually and rerun.",
                        repo_id, model_id)
        if not safe.wait_ready(model_id):
            rec.status = "failed"
            rec.error = "model never reached Ready"
            return rec

    try:
        task_prompt_prefix = prompt_prefix
        result = safe.submit_task(
            model_id, display_name, framework, precision, tp, conc, isl, osl, image,
            mode=mode, gpu_type=gpu_type, inferencex_path=inferencex_path,
            oob_path=oob_path, tracelens_root=tracelens_root,
            prompt_prefix=task_prompt_prefix, prompt_suffix=prompt_suffix,
            kernel_backends=kernel_backends, max_hours=max_hours,
            target_gain=target_gain, results_path=results_path)
    except Exception as e:
        rec.status = "failed"
        rec.error = f"submit_task: {e}"
        return rec

    rec.status = "submitted"
    rec.task_id = result.get("id", "?")
    log.info("[%s] OK — task_id=%s display=%s", repo_id, rec.task_id, display_name)
    return rec


# ── Post-submission: wait for completion + collect artifacts ───────────────────

# Default artifact filter — files SaFE has the agent copy to /workspace at end of
# Phase 10, plus variants seen in production.
DEFAULT_ARTIFACT_PATTERNS = (
    "optimization_report",   # optimization_report.md / *-optimization_report.md / ...
    "ci_metrics.json",
    # Key result preferred by claw-stats-service / V2 dashboard over ci_metrics;
    # part of _KEY_RESULT_SUFFIXES, so missing it triggers the NFS fallback.
    "session_breakdown.json",
    "baseline_summary.json",
    "sweep_results.csv",
    "sweep_results.txt",
    "kernel_candidates.json",
    "kernel_results.json",
    "run_context.env",
    "gpu_timeline.csv",
    "ci_summary.json",
    "ci_report.md",
)


def _is_wanted_artifact(path: str, all_artifacts: bool) -> bool:
    """Decide whether an artifact path should be downloaded.

    Args:
        path (str): The remote artifact path.
        all_artifacts (bool): When True, accept every artifact.

    Returns:
        bool: True when ``all_artifacts`` is set or the path matches a default
            artifact pattern.
    """
    if all_artifacts:
        return True
    p = path.lower()
    return any(pat in p for pat in DEFAULT_ARTIFACT_PATTERNS)


def _safe_local_path(artifacts_dir: Path, task_id: str, remote_path: str) -> Path:
    """Map a session-relative remote path to a local file path, stripping leading
    slashes / '..' so we never escape the artifacts dir."""
    rel = remote_path.lstrip("/").replace("\\", "/")
    parts = [seg for seg in rel.split("/") if seg and seg != ".." and seg != "."]
    return artifacts_dir / task_id / Path(*parts) if parts else artifacts_dir / task_id / "artifact.bin"


def _record_artifact_source(
    rec: SubmissionRecord,
    local_path: Path,
    source_type: str,
    *,
    remote_path: str | None = None,
    source_path: str | None = None,
    session_dir: str | None = None,
) -> None:
    """Append a provenance entry describing where an artifact came from.

    Args:
        rec (SubmissionRecord): Record whose ``artifact_sources`` is appended.
        local_path (Path): Local path the artifact was written to.
        source_type (str): Origin label (e.g. ``safe_artifact_api``, ``nfs_*``).
        remote_path (str | None): Remote artifact path, when applicable.
        source_path (str | None): Source filesystem path, when applicable.
        session_dir (str | None): Originating session directory, when known.
    """
    entry = {
        "source_type": source_type,
        "local_path": str(local_path).replace("\\", "/"),
        "file_name": local_path.name,
    }
    if remote_path:
        entry["remote_path"] = remote_path
    if source_path:
        entry["source_path"] = source_path
    if session_dir:
        entry["session_dir"] = session_dir
    rec.artifact_sources.append(entry)


def _write_artifact_sources(task_dir: Path, rec: SubmissionRecord) -> None:
    """Write the per-task ``artifact_sources.json`` provenance file.

    No-op when the record has no recorded artifact sources.

    Args:
        task_dir (Path): Per-task directory the file is written into.
        rec (SubmissionRecord): Record carrying the artifact source entries.
    """
    if not rec.artifact_sources:
        return
    payload = {
        "task_id": rec.task_id,
        "model": rec.model,
        "claw_session_id": rec.claw_session_id,
        "artifact_dir": str(task_dir).replace("\\", "/"),
        "files": rec.artifact_sources,
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "artifact_sources.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


# Files required for a run to count as "delivered". The NFS legacy fallback
# scans these suffixes; a miss triggers the wekafs rescue path.
_KEY_RESULT_SUFFIXES: tuple[str, ...] = (
    "optimization_report.md",
    "ci_metrics.json",
    "session_breakdown.json",
)


def _norm_token(s: str) -> str:
    """Aggressively normalise a string for fuzzy equality comparison.

    Lowercases and strips dashes, underscores, dots, slashes, and spaces.

    Args:
        s (str): String to normalise.

    Returns:
        str: The normalised token.
    """
    return (s or "").lower().replace("-", "").replace("_", "") \
        .replace(".", "").replace("/", "").replace(" ", "")


def _slug_token(s: str) -> str:
    """Convert a string to a lowercase dash-separated slug.

    Collapses any run of non-alphanumeric characters to a single dash and
    trims leading/trailing dashes.

    Args:
        s (str): String to slugify.

    Returns:
        str: The slugified token.
    """
    out = []
    prev_dash = False
    for ch in (s or "").lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-")


def _metrics_have_positive_throughput(path: str) -> bool:
    """Report whether a ci_metrics.json file has real, non-zero throughput.

    Args:
        path (str): Path to a ``ci_metrics.json`` file.

    Returns:
        bool: True when both baseline and optimized throughput parse to values
            greater than zero; False on any read/parse error.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    baseline = data.get("baseline_throughput") or data.get("tok_per_gpu_baseline")
    optimized = data.get("optimized_throughput") or data.get("tok_per_gpu_optimized")
    try:
        return float(baseline) > 0 and float(optimized) > 0
    except (TypeError, ValueError):
        return False


def _json_has_any_number(value) -> bool:
    """Recursively test whether a JSON value contains any real number.

    Booleans are not counted as numbers; lists are scanned up to the first 100
    elements.

    Args:
        value: Any JSON-decoded value.

    Returns:
        bool: True when an int/float (non-bool) is found anywhere within.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_json_has_any_number(v) for v in value.values())
    if isinstance(value, list):
        return any(_json_has_any_number(v) for v in value[:100])
    return False


def _breakdown_has_basic_data(path: Path) -> bool:
    """True when a session_breakdown JSON carries usable audit/perf payload. The
    delivery contract is "has structured data", not "has positive gain" (aborted
    runs may legitimately be zero)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    keys = (
        "baseline", "best", "final", "optimized", "steps", "actions",
        "phases", "session", "session_meta", "workload",
        "baseline_throughput", "optimized_throughput",
        "tok_per_gpu_baseline", "tok_per_gpu_optimized", "gain_pct",
        "cumulative_gain_pct",
    )
    if any(data.get(k) not in (None, {}, [], "") for k in keys):
        return True
    return _json_has_any_number(data)


def _mark_record_delivery(rec: SubmissionRecord) -> None:
    """Set CI-level delivery status from collected artifacts.

    Scans the record's artifact files/dir for a publishable
    ``session_breakdown`` JSON and updates ``ci_success``, ``ci_status``, and
    ``delivery_reason`` accordingly.

    Args:
        rec (SubmissionRecord): The submission record to mutate in place.
    """
    candidates: list[Path] = []
    for raw in rec.artifact_files:
        p = Path(raw)
        if p.is_file() and p.name.startswith("session_breakdown") and p.suffix == ".json":
            candidates.append(p)
    if rec.artifacts_dir:
        root = Path(rec.artifacts_dir)
        if root.is_dir():
            candidates.extend(
                p for p in root.glob("**/session_breakdown*.json")
                if p.is_file()
            )
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique = []
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    for p in unique:
        if _breakdown_has_basic_data(p):
            rec.ci_success = True
            rec.ci_status = "Delivered"
            rec.delivery_reason = f"publishable session_breakdown: {p.name}"
            return

    if rec.artifact_count:
        rec.ci_status = "Artifacts incomplete"
        rec.delivery_reason = "artifacts collected but no usable session_breakdown"
    else:
        rec.ci_status = "Missing artifacts"
        rec.delivery_reason = "no artifacts collected"


def _timestamp_hint_variants(value: str) -> set[str]:
    """Return path-matchable variants for skill session timestamps.

    Args:
        value (str): A session timestamp string (e.g. ``"20260512T010203Z"``).

    Returns:
        set[str]: Case- and separator-normalized variants for substring
            matching against paths; empty if ``value`` is blank.
    """
    raw = value.strip()
    if not raw:
        return set()
    compact = raw.replace("T", "").replace("t", "").replace("Z", "").replace("z", "")
    variants = {raw, raw.lower()}
    if compact and compact != raw:
        variants.update({compact, compact.lower()})
    return variants


def _session_hints_from_artifact_items(items: list[dict]) -> set[str]:
    """Collect session-timestamp hints from artifact item metadata.

    Args:
        items (list[dict]): Artifact item dicts with ``path``/``downloadPath``/
            ``name`` fields.

    Returns:
        set[str]: Timestamp hint variants discovered across the items.
    """
    hints: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        text = " ".join(
            str(item.get(key) or "")
            for key in ("path", "downloadPath", "name")
        )
        for match in re.findall(r"\b\d{8}T\d{6}Z\b", text, flags=re.IGNORECASE):
            hints.update(_timestamp_hint_variants(match))
    return hints


def _path_has_session_hint(path: str, hints: set[str]) -> bool:
    """Report whether a path contains any of the session timestamp hints.

    Args:
        path (str): The path to test.
        hints (set[str]): Timestamp hint variants to look for.

    Returns:
        bool: True if any hint appears in the normalized path.
    """
    if not hints:
        return False
    norm_path = _norm_token(path)
    return any(_norm_token(hint) in norm_path for hint in hints)


def _parse_safe_timestamp(value: str | None) -> datetime | None:
    """Parse a SaFE ISO-8601 timestamp into a UTC datetime.

    Args:
        value (str | None): An ISO-8601 string (``Z`` suffix accepted).

    Returns:
        datetime | None: A timezone-aware UTC datetime, or ``None`` if blank
            or unparseable.
    """
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_session_timestamp(value: str) -> datetime | None:
    """Parse a compact ``YYYYMMDDTHHMMSSZ`` session timestamp.

    Args:
        value (str): The compact session timestamp string.

    Returns:
        datetime | None: A UTC datetime, or ``None`` if blank or unparseable.
    """
    raw = value.strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw.upper(), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _session_timestamp_from_path(path: str) -> str:
    """Extract the last ``YYYYMMDDTHHMMSSZ`` timestamp found in a path.

    Args:
        path (str): The path to scan.

    Returns:
        str: The matched timestamp (upper-cased), or ``""`` if none found.
    """
    matches = re.findall(r"\b\d{8}T\d{6}Z\b", path, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else ""


def _timestamp_in_task_window(timestamp: str, rec: SubmissionRecord, margin_hours: int = 2) -> bool:
    """Check whether a session timestamp falls within the task's run window.

    Args:
        timestamp (str): A compact session timestamp string.
        rec (SubmissionRecord): The record providing SaFE start/finish times.
        margin_hours (int): Slack added on each side of the window.

    Returns:
        bool: True if the timestamp lies within the (padded) task window.
    """
    ts = _parse_session_timestamp(timestamp)
    start = _parse_safe_timestamp(rec.safe_started_at)
    end = _parse_safe_timestamp(rec.safe_finished_at)
    if ts is None or start is None:
        return False
    if end is None:
        end = start + timedelta(hours=24)
    return (start - timedelta(hours=margin_hours)) <= ts <= (end + timedelta(hours=margin_hours))


def _record_has_task_window(rec: SubmissionRecord) -> bool:
    """Report whether a record has a usable SaFE start timestamp.

    Args:
        rec (SubmissionRecord): The submission record to inspect.

    Returns:
        bool: True if ``safe_started_at`` parses into a timestamp.
    """
    return _parse_safe_timestamp(rec.safe_started_at) is not None


def _env_float(name: str, default: float) -> float:
    """Read a float environment setting with a safe fallback."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a float; using %.1f", name, raw, default)
        return default


def _read_session_state(session_dir: str | Path) -> dict:
    """Best-effort read of a session's state.json."""
    path = Path(session_dir) / "state.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _session_has_terminal_marker(session_dir: str | Path) -> bool:
    """True when a session has reached a state where CI should collect now."""
    root = Path(session_dir)
    if (root / "session_breakdown.json").is_file():
        return True
    if (root / "complete").is_file():
        return True
    state = _read_session_state(root)
    if state.get("close_sequence_done") is True:
        return True
    return False


def _session_activity_mtime(session_dir: str | Path) -> float:
    """Return a bounded best-effort activity timestamp for a session.

    ``state.json`` can be quiet while long Magpie subprocesses append logs or
    traces, so include the runtime subtrees CI relies on. The file cap avoids
    expensive walks for very large sessions.
    """
    root = Path(session_dir)
    mtimes: list[float] = []
    for rel in (
        "state.json",
        "session_breakdown.json",
        "complete",
        "reports/final.md",
        "reports/final.json",
    ):
        p = root / rel
        try:
            if p.exists():
                mtimes.append(p.stat().st_mtime)
        except OSError:
            continue

    seen = 0
    for sub in ("optimizer_runs", "runs", "reports"):
        base = root / sub
        if not base.exists():
            continue
        for walk_root, _dirs, files in os.walk(base):
            for name in files:
                seen += 1
                if seen > 5000:
                    return max(mtimes) if mtimes else 0.0
                if not name.endswith((".log", ".json", ".txt", ".md", ".csv", ".gz")):
                    continue
                try:
                    mtimes.append((Path(walk_root) / name).stat().st_mtime)
                except OSError:
                    continue
    return max(mtimes) if mtimes else 0.0


def _find_nfs_state_session_dir(
    rec: SubmissionRecord,
    current_session_hints: set[str] | None = None,
) -> str | None:
    """Locate the current NFS session using state.json, not breakdown files."""
    nfs_root = os.environ.get("NFS_ROOT", "/wekafs")
    users_root = Path(nfs_root) / "users"
    if not rec.safe_user_id or not users_root.is_dir():
        return None
    uid_path = users_root / rec.safe_user_id
    if not uid_path.is_dir():
        return None

    hints = set(current_session_hints or set())
    candidates: list[tuple[int, float, str]] = []
    for model_dir_name in _candidate_model_dir_names(rec):
        model_dir = uid_path / model_dir_name
        if not model_dir.is_dir():
            continue
        try:
            ts_entries = sorted(os.listdir(model_dir), reverse=True)
        except OSError:
            continue
        for ts_entry in ts_entries:
            session_dir = model_dir / ts_entry
            if not session_dir.is_dir():
                continue
            state_path = session_dir / "state.json"
            if not state_path.is_file():
                continue
            if hints:
                if not _path_has_session_hint(str(session_dir), hints):
                    continue
                score = 40
            else:
                ts = _session_timestamp_from_path(ts_entry)
                if not _timestamp_in_task_window(ts, rec):
                    continue
                score = 30
            state = _read_session_state(session_dir)
            workload = state.get("workload") if isinstance(state.get("workload"), dict) else {}
            model_field = str(
                state.get("model")
                or state.get("model_name")
                or workload.get("model_name")
                or ""
            )
            if model_field and not _record_model_field_matches(rec, model_field):
                continue
            if model_field:
                score += 100
            try:
                mtime = state_path.stat().st_mtime
            except OSError:
                continue
            candidates.append((score, mtime, str(session_dir)))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def _wait_for_nfs_session_delivery(
    rec: SubmissionRecord,
    current_session_hints: set[str] | None = None,
    poll_s: int = 60,
    grace_min: float | None = None,
    idle_min: float | None = None,
) -> str | None:
    """After SaFE early terminal, wait while the NFS session is still active."""
    grace_min = _env_float("SAFE_OPTIMIZE_NFS_LIVE_GRACE_MIN", 180.0) \
        if grace_min is None else grace_min
    idle_min = _env_float("SAFE_OPTIMIZE_NFS_IDLE_GRACE_MIN", 20.0) \
        if idle_min is None else idle_min
    if grace_min <= 0 or idle_min <= 0:
        return None

    session_dir = _find_nfs_state_session_dir(rec, current_session_hints)
    if not session_dir:
        return None

    now = time.time()
    activity = _session_activity_mtime(session_dir)
    if not activity:
        return session_dir
    if now - activity > idle_min * 60 and not _session_has_terminal_marker(session_dir):
        log.info(
            "[task %s] NFS session %s found but inactive for %.1fmin; "
            "collecting without grace wait",
            rec.task_id, session_dir, (now - activity) / 60,
        )
        return session_dir

    deadline = now + grace_min * 60
    idle_deadline = activity + idle_min * 60
    log.warning(
        "[task %s] SaFE status=%s but NFS session still appears active: %s; "
        "waiting up to %.1fmin (idle %.1fmin) for delivery contract files",
        rec.task_id, rec.final_status, session_dir, grace_min, idle_min,
    )
    while time.time() < deadline:
        if _session_has_terminal_marker(session_dir):
            log.info(
                "[task %s] NFS session reached terminal/delivery marker: %s",
                rec.task_id, session_dir,
            )
            return session_dir
        latest = _session_activity_mtime(session_dir)
        if latest > activity:
            activity = latest
            idle_deadline = latest + idle_min * 60
            log.info(
                "[task %s] NFS session still active (last activity %s)",
                rec.task_id,
                datetime.fromtimestamp(latest, tz=timezone.utc).isoformat(),
            )
        if time.time() > idle_deadline:
            log.warning(
                "[task %s] NFS session idle for %.1fmin without delivery "
                "marker; proceeding to collect",
                rec.task_id, idle_min,
            )
            return session_dir
        time.sleep(max(1, min(poll_s, 60)))

    log.warning(
        "[task %s] NFS live-session grace wait expired after %.1fmin; "
        "proceeding to collect",
        rec.task_id, grace_min,
    )
    return session_dir


def _category_from_arch(arch: str | None) -> str:
    """Coarse model-shape classification: "moe" if arch contains "moe", else
    "dense"; "" when unknown so downstream JSON stays "n/a"."""
    if not arch:
        return ""
    return "moe" if "moe" in arch.lower() else "dense"


def _sandbox_duration_seconds(last_task: dict) -> float | None:
    """SaFE-side sandbox wallclock = finishedAt - startedAt (from the task API).
    None when either field is missing/unparseable so we don't fabricate a duration."""
    from datetime import datetime
    start = (last_task or {}).get("startedAt") or ""
    end = (last_task or {}).get("finishedAt") or ""
    if not start or not end:
        return None
    try:
        # SaFE serializes UTC with trailing 'Z'; fromisoformat needs '+00:00'.
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except Exception:
        return None
    delta = (e - s).total_seconds()
    return round(delta, 1) if delta >= 0 else None


def _find_hyperloom_commit_sha(start: Path) -> str:
    """Resolve the Hyperloom git SHA the sandbox cloned (for audit fields).

    First hit wins: (1) hyperloom_source_commit.txt written by the agent (depth
    varies by which fallback collected it), then (2) the CI runner env
    (HYPERLOOM_SOURCE_REF, else GITHUB_SHA).
    """
    candidates = [
        start.parent / "hyperloom_source_commit.txt",
        start.parent / "session" / "hyperloom_source_commit.txt",
        start.parent.parent / "hyperloom_source_commit.txt",
    ]
    for sha_path in candidates:
        if not sha_path.exists():
            continue
        try:
            sha = sha_path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        # Accept only SHA-shaped strings; never trust a corrupted file.
        if 7 <= len(sha) <= 80 and all(c in "0123456789abcdef" for c in sha.lower()):
            return sha

    # Fallback: CI runner env. HYPERLOOM_SOURCE_REF is the pinned commit
    # (preferred); GITHUB_SHA is the unconditional fallback.
    for env_var in ("HYPERLOOM_SOURCE_REF", "GITHUB_SHA"):
        env_sha = (os.environ.get(env_var) or "").strip()
        if 7 <= len(env_sha) <= 80 and all(c in "0123456789abcdef" for c in env_sha.lower()):
            return env_sha
    return ""


def _backfill_ci_metrics_file(path: Path, rec: SubmissionRecord) -> None:
    """Backfill task metadata (model, image, hyperloom_commit, category,
    sandbox_duration_seconds) into ci_metrics.json / session_breakdown.json /
    manifest.json so each artifact is self-describing.

    Writes the right shape per filename: ci_metrics.json → flat top-level;
    session_breakdown.json → under session_meta; manifest.json → flat top-level
    (V2 cli schema; category/duration are extra keys V2 ignores on re-read).
    """
    if path.name not in ("ci_metrics.json", "session_breakdown.json", "manifest.json"):
        return
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict):
        return

    changed = False
    detected = rec.detected or {}
    image = detected.get("image") or ""
    hyperloom_sha = _find_hyperloom_commit_sha(path)
    image_tag = image.split("/")[-1] if image else ""

    if path.name == "ci_metrics.json":
        for key, value in {
            "model": rec.model,
            "task_id": rec.task_id,
            "claw_session_id": rec.claw_session_id,
        }.items():
            if value and not data.get(key):
                data[key] = value
                changed = True
        for key in ("framework", "tp"):
            if detected.get(key) is not None and data.get(key) is None:
                data[key] = detected.get(key)
                changed = True
        if image and not data.get("image"):
            data["image"] = image
            changed = True
        if hyperloom_sha and not data.get("hyperloom_commit"):
            data["hyperloom_commit"] = hyperloom_sha
            changed = True
        if rec.category and not data.get("category"):
            data["category"] = rec.category
            changed = True
        if rec.sandbox_duration_seconds is not None and not data.get("sandbox_duration_seconds"):
            data["sandbox_duration_seconds"] = rec.sandbox_duration_seconds
            changed = True

    elif path.name == "session_breakdown.json":
        # Keyed by a `session_meta` sub-dict; only write empty fields so we don't
        # overwrite what the V2 collectors filled in.
        meta = data.get("session_meta")
        if not isinstance(meta, dict):
            meta = {}
            data["session_meta"] = meta
        if image and not meta.get("image"):
            meta["image"] = image
            changed = True
        if image_tag and not meta.get("image_id"):
            meta["image_id"] = image_tag
            changed = True
        if hyperloom_sha and not meta.get("code_revision"):
            meta["code_revision"] = hyperloom_sha
            changed = True
        if rec.sandbox_duration_seconds is not None and not meta.get("session_duration_seconds"):
            meta["session_duration_seconds"] = rec.sandbox_duration_seconds
            changed = True
        # `category` isn't in the schema but unknown fields are tolerated.
        if rec.category and not meta.get("category"):
            meta["category"] = rec.category
            changed = True

    elif path.name == "manifest.json":
        # V2 cli schema: flat top-level keys, often null when the sandbox didn't
        # set HYPERLOOM_IMAGE or git rev-parse failed; backfill from the CI side.
        for key, value in {
            "model_name": rec.model,
            "claw_session_id": rec.claw_session_id,
        }.items():
            if value and not data.get(key):
                data[key] = value
                changed = True
        for key in ("framework", "tp"):
            if detected.get(key) is not None and not data.get(key):
                data[key] = detected.get(key)
                changed = True
        if image and not data.get("image"):
            data["image"] = image
            changed = True
        if hyperloom_sha and not data.get("code_revision"):
            data["code_revision"] = hyperloom_sha
            changed = True
        if rec.category and not data.get("category"):
            data["category"] = rec.category
            changed = True
        if rec.sandbox_duration_seconds is not None and not data.get("sandbox_duration_seconds"):
            data["sandbox_duration_seconds"] = rec.sandbox_duration_seconds
            changed = True

    if changed:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _backfill_wekafs_in_place(rec: SubmissionRecord) -> int:
    """Reverse-write audit fields back into the wekafs SOURCE files so operators
    see them under /wekafs/users/<uid>/<sess>/ without GHA artifact zips
    (image/category/duration are SaFE-side facts the agent/V2 cli never had).

    Match (mirrors Stage B): exact `model` field, else conservative session-dir
    match; only sessions modified in the last 24h. Updates ci_metrics.json,
    manifest.json, session_breakdown[_v2].json across subdirs via
    _backfill_ci_metrics_file. No-op when wekafs isn't mounted.
    """
    nfs_root = os.environ.get("NFS_ROOT", "/wekafs")
    users_root = os.path.join(nfs_root, "users")
    if not os.path.isdir(users_root):
        return 0
    target = _norm_token((rec.model or "").split("/")[-1])
    if not target:
        return 0

    fresh_cutoff = time.time() - 24 * 3600
    targets = ("ci_metrics.json", "manifest.json",
               "session_breakdown.json", "session_breakdown_v2.json")
    subdirs = ("", "phase10_report", "results", "v2_session")

    n = 0

    def _session_has_matching_json(sess_path: str) -> bool:
        """Return whether a session dir holds JSON matching the target model.

        Args:
            sess_path: Session directory to scan.

        Returns:
            ``True`` if any target JSON file under the known subdirs has a
            model field matching ``rec``; otherwise ``False``.
        """
        for sub in subdirs:
            base = os.path.join(sess_path, sub) if sub else sess_path
            if not os.path.isdir(base):
                continue
            for fn in targets:
                p = Path(base) / fn
                if not p.is_file():
                    continue
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                mf = _json_model_field(d) if isinstance(d, dict) else ""
                if mf and _record_model_field_matches(rec, mf):
                    return True
        return False

    def _backfill_files(sess_path: str) -> int:
        """Backfill the model field into target JSON files in a session dir.

        Args:
            sess_path: Session directory whose target JSON files should be
                updated.

        Returns:
            The number of files updated.
        """
        updated = 0
        for sub in subdirs:
            base = os.path.join(sess_path, sub) if sub else sess_path
            if not os.path.isdir(base):
                continue
            for fn in targets:
                p = Path(base) / fn
                if not p.is_file():
                    continue
                try:
                    before = p.read_bytes()
                    _backfill_ci_metrics_file(p, rec)
                    if p.read_bytes() != before:
                        updated += 1
                        log.info("[task %s] wekafs backfill: %s",
                                 rec.task_id, p)
                except Exception as e:
                    log.warning("[task %s] wekafs backfill failed for %s: %s",
                                rec.task_id, p, e)
        return updated

    for uid_dir in os.listdir(users_root):
        uid_path = os.path.join(users_root, uid_dir)
        if not os.path.isdir(uid_path):
            continue
        for sess in os.listdir(uid_path):
            sess_path = os.path.join(uid_path, sess)
            if not os.path.isdir(sess_path):
                continue
            try:
                if os.path.getmtime(sess_path) < fresh_cutoff:
                    continue
            except OSError:
                continue
            # Confirm session ownership: `model` field match (strongest), else
            # the conservative session-dir-name heuristic.
            matched = False
            for sub in subdirs:
                ci = os.path.join(sess_path, sub, "ci_metrics.json") if sub \
                    else os.path.join(sess_path, "ci_metrics.json")
                if not os.path.isfile(ci):
                    continue
                try:
                    d = json.loads(Path(ci).read_text(encoding="utf-8"))
                except Exception:
                    continue
                mf = str(d.get("model") or d.get("model_name") or "")
                if mf and _record_model_field_matches(rec, mf):
                    matched = True
                    break
            if not matched and _session_has_matching_json(sess_path):
                matched = True
            if not matched and _record_matches_session_dir(rec, sess):
                matched = True
            if not matched:
                continue
            n += _backfill_files(sess_path)

        # Current layout: /wekafs/users/<uid>/<model-basename>/<YYYYmmddTHHMMSSZ>/.
        # A deleted ci_metrics.json must not block manifest.json backfill.
        if rec.safe_user_id and uid_dir != rec.safe_user_id:
            continue
        for model_dir_name in _candidate_model_dir_names(rec):
            model_dir = os.path.join(uid_path, model_dir_name)
            if not os.path.isdir(model_dir):
                continue
            try:
                ts_entries = sorted(os.listdir(model_dir), reverse=True)
            except Exception:
                continue
            for ts_entry in ts_entries:
                sess_path = os.path.join(model_dir, ts_entry)
                if not os.path.isdir(sess_path):
                    continue
                try:
                    if os.path.getmtime(sess_path) < fresh_cutoff:
                        continue
                except OSError:
                    continue
                ts = _session_timestamp_from_path(ts_entry)
                matched = (
                    _record_has_task_window(rec)
                    and _timestamp_in_task_window(ts, rec)
                )
                if not matched and _session_has_matching_json(sess_path):
                    matched = True
                if not matched:
                    continue
                n += _backfill_files(sess_path)
    return n


def _record_matches_session_dir(rec: SubmissionRecord, sess_name: str) -> bool:
    """Conservative directory-name match for the /wekafs/users fallback. Prefer
    the displayName slug; fall back to basename with strict-term guards to avoid
    cross-wiring adjacent repos (Qwen2.5 vs -AWQ, Nano vs Super, bnb, etc.)."""
    sess_slug = _slug_token(sess_name)
    sess_norm = _norm_token(sess_name)
    display = rec.display_name or ""
    if display and _norm_token(display) in sess_norm:
        return True

    model = rec.model or ""
    base = model.split("/")[-1]
    base_norm = _norm_token(base)
    if not base_norm or base_norm not in sess_norm:
        return False

    identity = f"{model} {display}".lower()
    lower_dir = sess_name.lower()
    strict_terms = ("awq", "gptq", "bnb", "4bit", "abliterated",
                    "geneticlemonade", "nano", "super")
    for term in strict_terms:
        in_identity = term in identity
        in_dir = term in lower_dir
        if in_identity != in_dir:
            return False
    return True


def _record_model_field_matches(rec: SubmissionRecord, model_field: str) -> bool:
    """Compare a JSON model field with a SubmissionRecord conservatively."""
    observed = _norm_token(str(model_field or ""))
    if not observed:
        return False
    allowed = {
        _norm_token((rec.model or "").split("/")[-1]),
        _norm_token((rec.model or "").replace("/", "-")),
        _norm_token((rec.model_path or "").rstrip("/\\").split("/")[-1]),
        _norm_token(rec.display_name or ""),
    }
    allowed.discard("")
    return observed in allowed or _norm_token(str(model_field).split("/")[-1]) in allowed


def _candidate_model_dir_names(rec: SubmissionRecord) -> list[str]:
    """Derive plausible per-model directory basenames for a record.

    Args:
        rec (SubmissionRecord): The record supplying model path / id / display
            name candidates.

    Returns:
        list[str]: De-duplicated basename candidates, in priority order.
    """
    names: list[str] = []
    for value in (
        rec.model_path or "",
        (rec.model or "").replace("/", "-"),
        (rec.model or "").split("/")[-1],
        rec.display_name or "",
    ):
        name = str(value or "").strip().rstrip("/\\").split("/")[-1]
        if name and name not in names:
            names.append(name)
    return names


def _json_positive_perf(data: dict) -> bool:
    """Report whether a breakdown JSON has positive baseline AND optimized perf.

    Args:
        data (dict): A parsed session-breakdown/metrics JSON object.

    Returns:
        bool: True only if at least one positive baseline throughput and one
            positive optimized throughput are present.
    """
    def positive(value: object) -> bool:
        """Return True for a positive, non-bool numeric value.

        Args:
            value (object): Candidate value to test.

        Returns:
            bool: True if ``value`` is an ``int``/``float`` (not ``bool``)
                greater than zero.
        """
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0

    baseline = data.get("baseline") if isinstance(data.get("baseline"), dict) else {}
    final = data.get("final") if isinstance(data.get("final"), dict) else {}
    best = data.get("best") if isinstance(data.get("best"), dict) else {}
    base_values = (
        data.get("baseline_tput"),
        data.get("baseline_throughput"),
        data.get("tok_per_gpu_baseline"),
        baseline.get("throughput_tok_s_per_gpu"),
        baseline.get("output_throughput"),
    )
    opt_values = (
        data.get("best_tput"),
        data.get("optimized_throughput"),
        data.get("tok_per_gpu_optimized"),
        final.get("throughput_tok_s_per_gpu"),
        final.get("output_throughput"),
        best.get("throughput_tok_s_per_gpu"),
        best.get("output_throughput"),
    )
    return any(positive(v) for v in base_values) and any(positive(v) for v in opt_values)


def _json_model_field(data: dict) -> str:
    """Extract the model id from a breakdown/metrics JSON, trying known keys.

    Args:
        data (dict): A parsed session-breakdown/metrics JSON object.

    Returns:
        str: The first non-empty model field found, or ``""``.
    """
    workload = data.get("workload") if isinstance(data.get("workload"), dict) else {}
    session = data.get("session") if isinstance(data.get("session"), dict) else {}
    meta = data.get("session_meta") if isinstance(data.get("session_meta"), dict) else {}
    for value in (
        data.get("model"),
        data.get("model_name"),
        workload.get("model_name"),
        workload.get("model"),
        session.get("model"),
        meta.get("model"),
    ):
        if value not in (None, ""):
            return str(value)
    return ""


def _json_claw_session_id(data: dict) -> str:
    """Extract the Claw session id from a breakdown/metrics JSON.

    Args:
        data (dict): A parsed session-breakdown/metrics JSON object.

    Returns:
        str: The first non-empty ``claw_session_id`` found, or ``""``.
    """
    session = data.get("session") if isinstance(data.get("session"), dict) else {}
    meta = data.get("session_meta") if isinstance(data.get("session_meta"), dict) else {}
    for value in (
        data.get("claw_session_id"),
        session.get("claw_session_id"),
        meta.get("claw_session_id"),
    ):
        if value not in (None, ""):
            return str(value)
    return ""


def _env_truthy(name: str) -> bool:
    """Report whether an environment variable is set to a truthy value.

    Args:
        name (str): The environment variable name.

    Returns:
        bool: True if the value is one of 1/true/yes/y/on (case-insensitive).
    """
    return (os.environ.get(name) or "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }


def _copy_session_tree(src_dir: str, dst_dir: Path) -> int:
    """Copy an entire persisted session directory into ``dst_dir`` (existing files
    untouched). Returns the number of files copied."""
    copied = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for root, dirnames, filenames in os.walk(src_dir):
        dirnames[:] = [
            d for d in dirnames
            if d not in {".git", "__pycache__", ".pytest_cache"}
        ]
        rel_root = os.path.relpath(root, src_dir)
        out_root = dst_dir if rel_root == "." else dst_dir / rel_root
        out_root.mkdir(parents=True, exist_ok=True)
        for fname in filenames:
            src = os.path.join(root, fname)
            dst = out_root / fname
            if dst.exists():
                continue
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                log.warning("full-session copy failed %s -> %s: %s", src, dst, e)
                continue
            copied += 1
    return copied


def _nfs_fallback_collect(
    rec: SubmissionRecord,
    artifacts_dir: Path,
    copy_full_session: bool = False,
    current_session_hints: set[str] | None = None,
) -> int:
    """Scan NFS result directories for files matching this model, used when the
    SaFE artifact API returns nothing (V2 writes under /wekafs/users/<uid>/...).

    Two stages: A. legacy canonical CI dirs, matched by dir name; B. per-user
    session dirs, matched by each ci_metrics.json's `model` field (more accurate
    than dir-name fuzzy match). Mirrors ci/orchestrator.py's fallback.
    Returns count of files copied.
    """
    current_session_hints = set(current_session_hints or set())
    nfs_root = os.environ.get("NFS_ROOT", "/wekafs")
    model_basename = (rec.model or "").split("/")[-1]
    model_short = model_basename.lower().replace("-", "").replace("_", "").replace(".", "")
    if not model_short or not rec.task_id:
        return 0

    task_dir = artifacts_dir / rec.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    has_task_scoped_model_dir = bool(rec.safe_user_id and _record_has_task_window(rec))
    if not current_session_hints and not has_task_scoped_model_dir:
        log.info("[task %s] NFS fallback skipped: no current-session "
                 "timestamp hints from SaFE artifacts or task time window",
                 rec.task_id)
        return copied

    # ── Stage A: legacy canonical dirs (matched by current timestamp) ───────
    legacy_scan_dirs = [
        f"{nfs_root}/hyperloom-results",
        f"{nfs_root}/results/ci",
        f"{nfs_root}/inference-optimization/results",
    ]
    for scan_dir in legacy_scan_dirs:
        if not os.path.isdir(scan_dir):
            continue
        try:
            entries = sorted(os.listdir(scan_dir), reverse=True)
        except Exception:
            continue
        for entry in entries:
            if not _path_has_session_hint(entry, current_session_hints):
                continue
            entry_clean = entry.lower().replace("-", "").replace("_", "").replace(".", "")
            if model_short not in entry_clean:
                continue
            candidate = os.path.join(scan_dir, entry)
            if not os.path.isdir(candidate):
                continue
            for suffix in _KEY_RESULT_SUFFIXES:
                for root, _, fnames in os.walk(candidate):
                    for fn in fnames:
                        if not fn.endswith(suffix):
                            continue
                        src = os.path.join(root, fn)
                        dst = task_dir / fn
                        if dst.exists():
                            continue
                        try:
                            shutil.copy2(src, dst)
                            _backfill_ci_metrics_file(dst, rec)
                            _record_artifact_source(
                                rec,
                                dst,
                                "nfs_legacy",
                                source_path=src,
                                session_dir=candidate,
                            )
                        except Exception as e:
                            log.warning("[task %s] NFS legacy copy %s -> %s failed: %s",
                                        rec.task_id, src, dst, e)
                            continue
                        rec.artifact_files.append(str(dst))
                        rec.artifact_count += 1
                        copied += 1
                        log.info("[task %s] NFS legacy: copied %s -> %s",
                                 rec.task_id, src, dst)
            if copied:
                break
        if copied:
            break
    if copied:
        return copied

    # ── Stage B: /wekafs/users/<uid>/<session>/...  matched by timestamp ────
    users_root = f"{nfs_root}/users"
    if not os.path.isdir(users_root):
        return copied

    # Primary match: JSON model fields when present. Secondary: conservative
    # session-dir match (needed for runs that persist JSON without a `model` field).
    target = _norm_token(model_basename)
    if not target:
        return copied

    def _model_field_matches(model_field: str) -> bool:
        """Check a JSON ``model`` field against the record's expected names.

        Args:
            model_field (str): The model id read from a candidate result file.

        Returns:
            bool: True if it normalizes to one of the record's allowed names.
        """
        observed = _norm_token(model_field)
        allowed = {
            target,
            _norm_token((rec.model or "").replace("/", "-")),
            _norm_token((rec.model_path or "").rstrip("/\\").split("/")[-1]),
        }
        allowed.discard("")
        return observed in allowed or _norm_token(model_field.split("/")[-1]) in allowed

    def _consider_result_file(path: str, session_dir: str, score_base: int) -> None:
        """Score a candidate result file and append it to ``candidates``.

        Validates the file's model/session/claw fields against the record and,
        when it matches, records a ``(score, mtime, path, session_dir)`` tuple
        for later best-match selection.

        Args:
            path (str): Path to a candidate result JSON file.
            session_dir (str): The session directory containing the file.
            score_base (int): Base score reflecting match-source confidence.
        """
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        if not isinstance(data, dict):
            return

        model_field = _json_model_field(data)
        if model_field:
            if not _model_field_matches(model_field):
                return
            score = score_base + 100
        else:
            session_name = os.path.basename(session_dir.rstrip("/"))
            parent_name = os.path.basename(os.path.dirname(session_dir.rstrip("/")))
            if not (
                _record_matches_session_dir(rec, session_name)
                or _record_matches_session_dir(rec, parent_name)
            ):
                return
            score = score_base + 60

        claw = _json_claw_session_id(data)
        if rec.claw_session_id and claw:
            if rec.claw_session_id != claw:
                return
            score += 30
        if _json_positive_perf(data):
            score += 10
        else:
            log.info("[task %s] candidate %s matched but has no positive "
                     "throughput (will use only if no better match)",
                     rec.task_id, path)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        candidates.append((score, mtime, path, session_dir))

    candidates: list[tuple[int, float, str, str]] = []  # (score, mtime, marker_path, session_dir)
    uid_dirs = [rec.safe_user_id] if rec.safe_user_id else os.listdir(users_root)
    for uid_dir in uid_dirs:
        if not uid_dir:
            continue
        uid_path = os.path.join(users_root, uid_dir)
        if not os.path.isdir(uid_path):
            continue
        for sess in os.listdir(uid_path):
            sess_path = os.path.join(uid_path, sess)
            if not os.path.isdir(sess_path):
                continue
            if not _path_has_session_hint(sess_path, current_session_hints):
                continue
            for sub in ("", "phase10_report", "results"):
                ci_path = os.path.join(sess_path, sub, "ci_metrics.json") \
                    if sub else os.path.join(sess_path, "ci_metrics.json")
                _consider_result_file(ci_path, sess_path, 0)

        # Current layout: /wekafs/users/<uid>/<model-basename>/<YYYYmmddTHHMMSSZ>/.
        # With no artifact-derived hint, accept timestamp dirs only under this
        # exact user id + model dir and inside the task's startedAt/finishedAt window.
        if rec.safe_user_id and uid_dir == rec.safe_user_id:
            for model_dir_name in _candidate_model_dir_names(rec):
                model_dir = os.path.join(uid_path, model_dir_name)
                if not os.path.isdir(model_dir):
                    continue
                try:
                    ts_entries = sorted(os.listdir(model_dir), reverse=True)
                except Exception:
                    continue
                for ts_entry in ts_entries:
                    session_dir = os.path.join(model_dir, ts_entry)
                    if not os.path.isdir(session_dir):
                        continue
                    ts = _session_timestamp_from_path(ts_entry)
                    if current_session_hints:
                        if not _path_has_session_hint(session_dir, current_session_hints):
                            continue
                        score_base = 40
                    else:
                        if not _timestamp_in_task_window(ts, rec):
                            continue
                        score_base = 30
                    for sub, filename in (
                        ("", "ci_metrics.json"),
                        ("phase10_report", "ci_metrics.json"),
                        ("results", "ci_metrics.json"),
                        ("", "session_breakdown.json"),
                        ("phase10_report", "session_breakdown.json"),
                        ("v2_session", "session_breakdown.json"),
                    ):
                        result_path = os.path.join(session_dir, sub, filename) if sub \
                            else os.path.join(session_dir, filename)
                        _consider_result_file(result_path, session_dir, score_base)

    if not candidates:
        log.info("[task %s] no /wekafs/users candidate matched model=%s "
                 "(user_id=%s, hints=%s, task_window=%s)",
                 rec.task_id, model_basename, rec.safe_user_id or "?",
                 ",".join(sorted(current_session_hints)) or "-",
                 "yes" if _record_has_task_window(rec) else "no")
        return copied

    # Highest-confidence, then freshest (same model may be re-run).
    candidates.sort(reverse=True)
    _score, _mtime, best_ci, best_sess = candidates[0]
    log.info("[task %s] NFS user-session match: %s (from %d candidate(s), "
             "session=%s)", rec.task_id, best_ci, len(candidates), best_sess)

    # Copy ci_metrics + any optimization_report.md flat under task_dir/ (the
    # shape build_summary.py expects).
    targets = [best_ci]
    for cand in [
        os.path.join(best_sess, "optimization_report.md"),
        os.path.join(best_sess, "phase10_report", "optimization_report.md"),
        os.path.join(best_sess, "reports", "final.md"),
    ]:
        if os.path.isfile(cand):
            targets.append(cand)
            break
    # Optional audit artifact, when the agent emitted it.
    for cand in [
        os.path.join(best_sess, "session_breakdown.json"),
        os.path.join(best_sess, "phase10_report", "session_breakdown.json"),
    ]:
        if os.path.isfile(cand):
            targets.append(cand)
            break

    for src in targets:
        dst_name = ("optimization_report.md"
                    if src.endswith("final.md") else os.path.basename(src))
        dst = task_dir / dst_name
        if dst.exists():
            continue
        try:
            shutil.copy2(src, dst)
            _backfill_ci_metrics_file(dst, rec)
            _record_artifact_source(
                rec,
                dst,
                "nfs_user_session",
                source_path=src,
                session_dir=best_sess,
            )
        except Exception as e:
            log.warning("[task %s] NFS user-session copy %s -> %s failed: %s",
                        rec.task_id, src, dst, e)
            continue
        rec.artifact_files.append(str(dst))
        rec.artifact_count += 1
        copied += 1
        log.info("[task %s] NFS user-session: copied %s -> %s",
                 rec.task_id, src, dst)

    if copy_full_session:
        session_dst = task_dir / "session"
        n_full = _copy_session_tree(best_sess, session_dst)
        if n_full:
            copied += n_full
            rec.artifact_count += n_full
            rec.artifact_files.append(str(session_dst))
            log.info("[task %s] NFS full-session: copied %d file(s) %s -> %s",
                     rec.task_id, n_full, best_sess, session_dst)
        else:
            log.info("[task %s] NFS full-session: no new files copied from %s",
                     rec.task_id, best_sess)

    return copied


def wait_and_collect_one(
    safe: SafeOptimizeClient,
    rec: SubmissionRecord,
    artifacts_dir: Path,
    task_timeout_min: int,
    poll_s: int,
    collect: bool,
    all_artifacts: bool,
) -> SubmissionRecord:
    """Wait for one task to finish, then optionally download its artifacts:
    (1) SaFE artifacts API, then (2) NFS fallback when session_breakdown.json
    (the CI delivery contract) is still missing."""
    if not rec.task_id:
        return rec  # nothing to wait for (skipped/failed during submit)

    final_status, last_task = safe.wait_task_done(
        rec.task_id, timeout_min=task_timeout_min, poll_s=poll_s)
    rec.final_status = final_status
    rec.final_phase = last_task.get("currentPhase")
    rec.final_message = (last_task.get("message") or "")[:500] or None
    rec.claw_session_id = (last_task.get("clawSessionId") or "").strip() or None
    rec.model_path = (last_task.get("modelPath") or "").strip() or rec.model_path
    rec.safe_user_id = (last_task.get("userId") or "").strip() or rec.safe_user_id
    rec.safe_started_at = (last_task.get("startedAt") or "").strip() or rec.safe_started_at
    rec.safe_finished_at = (last_task.get("finishedAt") or "").strip() or rec.safe_finished_at
    rec.sandbox_duration_seconds = _sandbox_duration_seconds(last_task)
    if rec.claw_session_id:
        log.info("[task %s] clawSessionId=%s duration=%ss",
                 rec.task_id, rec.claw_session_id,
                 rec.sandbox_duration_seconds if rec.sandbox_duration_seconds is not None else "?")

    if not collect:
        rec.ci_status = "Not collected"
        rec.delivery_reason = "artifact collection disabled"
        return rec

    # Stage 1: SaFE artifacts API (most reliable when the agent copied files to
    # /workspace/hyperloom). Retry — terminal status can beat Claw's file index.
    items = []
    wanted = []
    for attempt in range(3):
        try:
            items = safe.list_artifacts(rec.task_id)
        except Exception as e:
            log.warning("[task %s] list_artifacts attempt %d failed: %s",
                        rec.task_id, attempt + 1, e)
            items = []
        wanted = [it for it in items
                  if _is_wanted_artifact(it.get("path", ""), all_artifacts)]
        wanted_paths = [it.get("path", "").lower() for it in wanted]
        # session_breakdown.json is the CI delivery contract (cli.finally always
        # writes it, even on abort), so retry only until it shows up.
        has_safe_breakdown = any(
            p.endswith("session_breakdown.json") for p in wanted_paths
        )
        if has_safe_breakdown:
            break
        if attempt < 2:
            log.info("[task %s] safe artifacts missing session_breakdown.json on "
                     "attempt %d; retrying", rec.task_id, attempt + 1)
            time.sleep(15)
    log.info("[task %s] safe artifacts: %d total, %d to download",
             rec.task_id, len(items), len(wanted))
    current_session_hints = _session_hints_from_artifact_items(items)
    if current_session_hints:
        log.info("[task %s] current session timestamp hints from artifacts: %s",
                 rec.task_id, ", ".join(sorted(current_session_hints)))

    if not has_safe_breakdown:
        waited_session = _wait_for_nfs_session_delivery(
            rec,
            current_session_hints=current_session_hints,
            poll_s=poll_s,
        )
        if waited_session:
            # The SaFE artifact index may lag behind the agent's final writes.
            # Re-list once after the grace wait before falling back to NFS.
            try:
                items = safe.list_artifacts(rec.task_id)
                wanted = [
                    it for it in items
                    if _is_wanted_artifact(it.get("path", ""), all_artifacts)
                ]
                wanted_paths = [it.get("path", "").lower() for it in wanted]
                has_safe_breakdown = any(
                    p.endswith("session_breakdown.json") for p in wanted_paths
                )
                current_session_hints.update(_session_hints_from_artifact_items(items))
                log.info("[task %s] safe artifacts after NFS grace wait: "
                         "%d total, %d to download",
                         rec.task_id, len(items), len(wanted))
            except Exception as e:
                log.warning("[task %s] post-grace list_artifacts failed: %s",
                            rec.task_id, e)

    task_dir = artifacts_dir / rec.task_id
    rec.artifacts_dir = str(task_dir)
    for it in wanted:
        path = it.get("path", "")
        if not path:
            continue
        local = _safe_local_path(artifacts_dir, rec.task_id, path)
        try:
            n = safe.download_artifact_to(rec.task_id, it, str(local))
            _backfill_ci_metrics_file(local, rec)
            _record_artifact_source(
                rec,
                local,
                "safe_artifact_api",
                remote_path=path,
            )
            rec.artifact_files.append(str(local))
            rec.artifact_count += 1
            log.info("[task %s] saved %s (%d bytes)", rec.task_id, path, n)
        except Exception as e:
            log.warning("[task %s] failed to download %s: %s", rec.task_id, path, e)

    # Stage 2: NFS fallback when Stage 1 didn't deliver session_breakdown.json
    # (the CI delivery contract). ci_metrics.json / optimization_report.md no
    # longer gate the fallback.
    has_breakdown = any(
        p.endswith("session_breakdown.json") for p in rec.artifact_files
    )
    if not has_breakdown:
        log.info("[task %s] missing session_breakdown.json — trying NFS fallback",
                 rec.task_id)
        copy_full_session = all_artifacts or _env_truthy("SAFE_OPTIMIZE_COPY_FULL_SESSION")
        n_added = _nfs_fallback_collect(
            rec,
            artifacts_dir,
            copy_full_session=copy_full_session,
            current_session_hints=current_session_hints,
        )
        if n_added:
            log.info("[task %s] NFS fallback added %d files", rec.task_id, n_added)
        else:
            log.info("[task %s] NFS fallback found nothing", rec.task_id)

    _mark_record_delivery(rec)
    if rec.ci_status:
        log.info("[task %s] CI delivery status: %s (%s)",
                 rec.task_id, rec.ci_status, rec.delivery_reason or "-")

    # Stage 3: reverse-backfill audit fields into the wekafs SOURCE files so
    # operators see them without the GHA artifact zip. No-op when wekafs unmounted.
    if rec.artifact_count:
        try:
            n_wkfs = _backfill_wekafs_in_place(rec)
            if n_wkfs:
                log.info("[task %s] wekafs in-place backfill updated %d file(s)",
                         rec.task_id, n_wkfs)
        except Exception as e:
            log.warning("[task %s] wekafs in-place backfill skipped due to %s: %s",
                        rec.task_id, type(e).__name__, e)
    else:
        log.info("[task %s] wekafs in-place backfill skipped: no artifacts collected",
                 rec.task_id)
    try:
        _write_artifact_sources(task_dir, rec)
    except Exception as e:
        log.warning("[task %s] artifact source metadata write skipped: %s",
                    rec.task_id, e)
    return rec


def process_completion(
    safe: SafeOptimizeClient,
    records: list[SubmissionRecord],
    artifacts_dir: Path,
    task_timeout_min: int,
    poll_s: int,
    collect: bool,
    all_artifacts: bool,
    parallel: int,
) -> None:
    """Wait + collect for all submitted records, in parallel up to ``parallel``.

    Args:
        safe (SafeOptimizeClient): Client used to poll and download artifacts.
        records (list[SubmissionRecord]): All submission records; only
            ``submitted`` ones with a task id are awaited.
        artifacts_dir (Path): Destination root for downloaded artifacts.
        task_timeout_min (int): Max minutes to wait per task.
        poll_s (int): Polling interval in seconds.
        collect (bool): When True, collect artifacts after each task finishes.
        all_artifacts (bool): When True, also copy full session trees.
        parallel (int): Max concurrent wait/collect workers (<=1 runs serially).
    """
    pending = [r for r in records if r.status == "submitted" and r.task_id]
    if not pending:
        log.info("no submitted tasks to wait for")
        return

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log.info("waiting for %d task(s) to finish (parallel=%d, timeout=%dm each)",
             len(pending), parallel, task_timeout_min)

    if parallel <= 1:
        for rec in pending:
            try:
                wait_and_collect_one(safe, rec, artifacts_dir,
                                     task_timeout_min, poll_s, collect, all_artifacts)
            except Exception as e:
                log.exception("[task %s] unexpected wait/collect error", rec.task_id)
                rec.final_status = rec.final_status or "Error"
                rec.final_message = (rec.final_message or "") + f" | wait error: {e}"
            finally:
                if not rec.ci_status:
                    _mark_record_delivery(rec)
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {
            ex.submit(wait_and_collect_one, safe, rec, artifacts_dir,
                      task_timeout_min, poll_s, collect, all_artifacts): rec
            for rec in pending
        }
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                fut.result()  # mutates rec in place
            except Exception as e:
                log.exception("[task %s] unexpected wait/collect error", rec.task_id)
                rec.final_status = rec.final_status or "Error"
                rec.final_message = (rec.final_message or "") + f" | wait error: {e}"
            finally:
                if not rec.ci_status:
                    _mark_record_delivery(rec)


# ── Manifest ────────────────────────────────────────────────────────────────────

def write_manifest(
    out_dir: Path,
    records: list[SubmissionRecord],
    base_url: str,
    register_workspace: str,
    submit_workspace: str,
    volume: str,
) -> None:
    """Write the submission manifest as JSON and a markdown summary table.

    Args:
        out_dir (Path): Output directory (created if absent).
        records (list[SubmissionRecord]): The submission records to serialize.
        base_url (str): SaFE API base URL recorded in the manifest.
        register_workspace (str): Workspace used for registration.
        submit_workspace (str): Workspace used for submission.
        volume (str): Storage volume name recorded in the manifest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api_url": base_url,
        "register_workspace": register_workspace,
        "submit_workspace": submit_workspace,
        "volume": volume,
        "records": [asdict(r) for r in records],
    }
    (out_dir / "submission_manifest.json").write_text(json.dumps(payload, indent=2))

    md = [
        "# SaFE Optimization Submission Manifest",
        f"- API: `{base_url}`",
        f"- Register workspace: `{register_workspace}`",
        f"- Submit workspace: `{submit_workspace}`",
        f"- Volume: `{volume}`",
        f"- Submitted at: {payload['submitted_at']}",
        "",
        "| Pool | Model | Category | Image | Duration | Submit | Final | CI | Phase | Task ID | Display Name | Artifacts | Note |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        # final_status is only meaningful when --wait-for-completion was on.
        final = r.final_status or ("-" if r.status == "submitted" else "")
        phase = "-" if r.final_phase is None else str(r.final_phase)
        artifacts_cell = (
            f"{r.artifact_count} files in `{r.artifacts_dir}`"
            if r.artifact_count else "-"
        )
        note_parts = []
        if r.error:
            note_parts.append(r.error)
        if r.final_message:
            note_parts.append(r.final_message)
        note = " \\| ".join(note_parts).replace("|", "\\|")[:200]
        pool_cell = "-"
        if r.pool:
            pool_id = r.pool.get("pool_id") or "-"
            pool_idx = r.pool.get("pool_index")
            batch_idx = r.pool.get("batch_index")
            batch_size = r.pool.get("batch_size")
            pool_cell = (
                f"`{pool_id}`"
                f"<br/>idx={pool_idx if pool_idx not in (None, '') else '-'}"
                f"<br/>batch={batch_idx if batch_idx not in (None, '') else '-'}/"
                f"{batch_size if batch_size not in (None, '') else '-'}"
            )
        # Image cell: tag suffix only for readability; full path is in JSON.
        image_cell = "-"
        image_full = (r.detected or {}).get("image", "") if r.detected else ""
        if image_full:
            image_cell = "`" + image_full.split("/")[-1] + "`"
        # Duration cell: rounded minutes; ms-precision is in JSON.
        duration_cell = "-"
        if r.sandbox_duration_seconds is not None:
            mins = r.sandbox_duration_seconds / 60.0
            duration_cell = f"{mins:.1f}m"
        category_cell = r.category or "-"
        ci_cell = r.ci_status or ("Succeeded" if r.final_status == "Succeeded" else "-")
        if r.ci_success and r.delivery_reason:
            ci_cell = f"{ci_cell}<br/>{r.delivery_reason}"
        md.append(
            f"| {pool_cell} | `{r.model}` | {category_cell} | {image_cell} | {duration_cell} | "
            f"{r.status} | {final or '-'} | {ci_cell} | {phase} | "
            f"`{r.task_id or '-'}` | {r.display_name or '-'} | {artifacts_cell} | {note} |"
        )
    (out_dir / "submission_manifest.md").write_text("\n".join(md) + "\n")
    log.info("manifest written to %s", out_dir)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the optimize-submit CLI.

    Returns:
        argparse.ArgumentParser: The configured parser with model selection,
            override, SaFE connection, and collection options.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", nargs="+", metavar="HF_REPO",
                     help="HuggingFace repo IDs, e.g. Qwen/Qwen3-8B")
    src.add_argument("--hf-top", type=int, metavar="N",
                     help="Auto-select top-N text-gen models from HuggingFace by downloads")
    parser.add_argument("--min-params", type=float, default=0.0, metavar="B",
                        help="Filter HF top-N to models with >=B billion params")

    parser.add_argument("--manual", action="store_true",
                        help="Manual mode: skip auto-detect; --framework is required")
    parser.add_argument("--framework", choices=["sglang", "vllm"],
                        help="Override detected framework")
    parser.add_argument("--precision", choices=["FP8", "FP4", "BF16", "INT4"],
                        help="Override detected precision")
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8, 16, 32],
                        help="Override detected tensor parallel size. Values >8 "
                             "require --nodes>1 (multi-node RayJob); tp must be "
                             "<= nodes*8.")
    parser.add_argument("--concurrency", type=int,
                        help="Override detected concurrency")
    parser.add_argument("--image", help="Override container image")
    parser.add_argument("--isl", type=int, default=1024)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--mode", choices=["local", "claw"], default="local",
                        help="Execution mode passed to SaFE (default: local — "
                             "agent runs in sandbox directly; 'claw' routes via RayJob)")
    parser.add_argument("--nodes", type=int, default=1, metavar="N",
                        help="Node count for the run. N>1 spreads the model across "
                             "an N-node RayJob (8 GPUs/node): forces --mode claw and "
                             "injects the RayJob topology (image, per-node resources, "
                             "NODES=N, bnxt tar) into the agent prompt — mirrors the "
                             "validated ci-config.yaml multi-node entries. Default 1.")
    parser.add_argument("--rayjob-image", default="",
                        help="Container image for the multi-node RayJob (used only "
                             "when --nodes>1). Falls back to --image when empty.")

    parser.add_argument("--api-url", default="",
                        help="SaFE base URL (defaults to $SAFE_BASE_URL or $SAFE_API_URL)")
    parser.add_argument("--api-key", default="",
                        help="SaFE bearer token (defaults to $CLAW_API_KEY or $SAFE_API_KEY)")
    parser.add_argument("--register-workspace", default="",
                        help=f"Workspace where models are registered + downloaded "
                             f"(defaults to $SAFE_OPTIMIZE_REGISTER_WORKSPACE "
                             f"then '{DEFAULT_REGISTER_WORKSPACE}')")
    parser.add_argument("--submit-workspace", default="",
                        help=f"Workspace where the optimization task runs "
                             f"(defaults to $SAFE_OPTIMIZE_SUBMIT_WORKSPACE "
                             f"then '{DEFAULT_SUBMIT_WORKSPACE}'). Used when "
                             f"--submit-workspaces is empty.")
    parser.add_argument("--submit-workspaces", default="",
                        help="Comma-separated list of submit workspaces for "
                             "round-robin task distribution (e.g. "
                             "'core42-sandbox,core42-hyperloom'). When set, "
                             "overrides --submit-workspace and spreads the "
                             "batch evenly across the listed workspaces. "
                             "Each must independently accept the same model "
                             "(register_workspace stays single). Defaults to "
                             "$SAFE_OPTIMIZE_SUBMIT_WORKSPACES.")
    parser.add_argument("--workspace", default="",
                        help="Shorthand: set both --register-workspace and "
                             "--submit-workspace to the same value (back-compat)")
    parser.add_argument("--volume", default="",
                        help=f"Wekafs volume mounted RW in --register-workspace "
                             f"(defaults to $SAFE_OPTIMIZE_VOLUME then '{DEFAULT_VOLUME}')")
    parser.add_argument("--gpu-type", default="",
                        help=f"GPU type tag for the prompt (defaults to "
                             f"$SAFE_OPTIMIZE_GPU_TYPE then '{DEFAULT_GPU_TYPE}'). "
                             f"Known profiles: {', '.join(GPU_PROFILES)}. "
                             f"SaFE backend default is MI355X — must override on core42.")
    parser.add_argument("--inferencex-path", default="",
                        help="Explicit InferenceX checkout path inside the "
                             "sandbox (dev override; also $SAFE_OPTIMIZE_"
                             "INFERENCEX_PATH). Leave empty (the default) so "
                             "install.sh clones a writable per-session copy "
                             "instead of pinning a shared read-only mount.")
    parser.add_argument("--oob-path", default="",
                        help="Optional OOB checkout override inside the sandbox. "
                             "Default is unset: sandbox-side install.sh prepares "
                             "and exports OOB paths.")
    parser.add_argument("--tracelens-root", default="",
                        help="TraceLens checkout path inside the sandbox. "
                             "Leave empty (the default) so install.sh clones "
                             "AMD-AGI/TraceLens into "
                             "$HYPERLOOM_RUNTIME_DIR/source-mirrors/TraceLens "
                             "and pins it to a fixed SHA. Override via "
                             "$SAFE_OPTIMIZE_TRACELENS_ROOT or this flag only "
                             "to point at an existing cluster checkout.")
    parser.add_argument("--prompt-prefix",
                        default=_load_default_prompt_prefix(),
                        help="Free-form prefix prepended to the SaFE-generated "
                             "Hyperloom prompt. Default resolves to "
                             "$SAFE_OPTIMIZE_PROMPT_PREFIX -> ci/prompt_prefix.txt "
                             "-> empty. Pass an empty string explicitly to suppress.")
    parser.add_argument("--prompt-suffix",
                        default=os.environ.get("SAFE_OPTIMIZE_PROMPT_SUFFIX", ""),
                        help="Optional free-form suffix appended to the SaFE-generated "
                             "Hyperloom prompt. (env: $SAFE_OPTIMIZE_PROMPT_SUFFIX)")
    parser.add_argument("--kernel-opt-backends",
                        default=os.environ.get("SAFE_OPTIMIZE_KERNEL_BACKENDS", ""),
                        help="Comma-separated kernel optimization backends to send "
                             "to SaFE's kernelBackends field. Aliases: geak, "
                             "claude, codex, cursor. Default: geak,claude,codex "
                             "(GEAK first).")
    parser.add_argument("--max-hours", type=float,
                        default=float(os.environ.get("SAFE_OPTIMIZE_MAX_HOURS", DEFAULT_MAX_HOURS)),
                        help="Max hours passed to the Hyperloom optimizer prompt "
                             "(default: 12).")
    parser.add_argument("--target-gain", type=float,
                        default=float(os.environ.get("SAFE_OPTIMIZE_TARGET_GAIN", DEFAULT_TARGET_GAIN)),
                        help="Target gain %% passed to the Hyperloom optimizer prompt "
                             "(default: 30).")
    parser.add_argument("--results-path",
                        default=os.environ.get("SAFE_OPTIMIZE_RESULTS_PATH", DEFAULT_RESULTS_PATH),
                        help="Results path passed to SaFE's prompt builder "
                             "(default: $RESULT_DIR so the prompt respects the "
                             "CI-selected persistent/ephemeral result root).")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                        help="HuggingFace token (or set $HF_TOKEN)")

    # Production-pool audit metadata: copied into submission_manifest.json (does
    # not affect submission) to trace which pool entry a task reran.
    parser.add_argument("--pool-id", default=os.environ.get("HYPERLOOM_POOL_ID", ""))
    parser.add_argument("--pool-index", default=os.environ.get("HYPERLOOM_POOL_INDEX", ""))
    parser.add_argument("--pool-batch-index", default=os.environ.get("HYPERLOOM_POOL_BATCH_INDEX", ""))
    parser.add_argument("--pool-batch-size", default=os.environ.get("HYPERLOOM_POOL_BATCH_SIZE", ""))
    parser.add_argument("--pool-source-task-id", default=os.environ.get("HYPERLOOM_POOL_SOURCE_TASK_ID", ""))

    parser.add_argument("--dry-run", action="store_true",
                        help="Auto-detect and print plan without registering or submitting")
    parser.add_argument("--output-dir", default="",
                        help="Write submission_manifest.{json,md} to this dir (for CI artifacts)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Post-submission: wait for tasks + collect artifacts.
    wait_group = parser.add_mutually_exclusive_group()
    wait_group.add_argument("--wait-for-completion", dest="wait_for_completion",
                            action="store_true", default=True,
                            help="(default) After submitting, poll each task until it reaches "
                                 "Succeeded/Failed/Interrupted/Timeout")
    wait_group.add_argument("--no-wait-for-completion", dest="wait_for_completion",
                            action="store_false",
                            help="Fire-and-forget: exit immediately after submitting")

    collect_group = parser.add_mutually_exclusive_group()
    collect_group.add_argument("--collect-artifacts", dest="collect_artifacts",
                               action="store_true", default=True,
                               help="(default) Download each finished task's artifacts to "
                                    "--artifacts-dir (implies --wait-for-completion)")
    collect_group.add_argument("--no-collect-artifacts", dest="collect_artifacts",
                               action="store_false",
                               help="Skip artifact download even when waiting")

    parser.add_argument("--all-artifacts", action="store_true",
                        help=f"Download every artifact (default keeps only files matching "
                             f"{', '.join(DEFAULT_ARTIFACT_PATTERNS)})")
    parser.add_argument("--artifacts-dir", default="task-artifacts",
                        help="Local directory where per-task artifacts land (default: ./task-artifacts)")
    parser.add_argument("--task-timeout-min", type=int, default=720,
                        help="Per-task wait timeout in minutes (default: 720 = 12h)")
    parser.add_argument("--poll-interval-s", type=int, default=60,
                        help="How often to poll task status, seconds (default: 60)")
    parser.add_argument("--wait-parallel", type=int, default=8,
                        help="How many tasks to wait for in parallel (default: 8)")
    parser.add_argument(
        "--submit-jitter-sec", type=int,
        default=int(os.environ.get("SAFE_OPTIMIZE_SUBMIT_JITTER_SEC", "0") or 0),
        help="Pre-submit random delay window in seconds. Each (parallel matrix) "
             "job sleeps random(0..N) before touching SaFE so register/submit "
             "calls de-sync instead of stampeding Claw-session creation all at "
             "once (the thundering herd that the backend answers with HTTP 500 "
             "'failed to create Claw session' / 504). 0 = off (default).")
    return parser


def main() -> int:
    """CLI entry point: register, submit, wait/collect, and write the manifest.

    Parses arguments, resolves the model set and SaFE connection, submits
    optimization tasks, optionally waits for completion and collects artifacts,
    and writes the submission manifest.

    Returns:
        int: Process exit code (0 on success, non-zero on fatal errors).
    """
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_url = (args.api_url
                or os.environ.get("SAFE_BASE_URL")
                or os.environ.get("SAFE_API_URL")
                or DEFAULT_API_URL)
    api_key = (args.api_key
               or os.environ.get("CLAW_API_KEY")
               or os.environ.get("SAFE_API_KEY")
               or "")
    # --workspace shorthand sets both to the same value (back-compat); explicit
    # --register-workspace / --submit-workspace override it.
    shared_ws = (args.workspace
                 or os.environ.get("SAFE_OPTIMIZE_WORKSPACE")
                 or "")
    register_workspace = (args.register_workspace
                          or os.environ.get("SAFE_OPTIMIZE_REGISTER_WORKSPACE")
                          or shared_ws
                          or DEFAULT_REGISTER_WORKSPACE)
    submit_workspace = (args.submit_workspace
                        or os.environ.get("SAFE_OPTIMIZE_SUBMIT_WORKSPACE")
                        or shared_ws
                        or DEFAULT_SUBMIT_WORKSPACE)
    # Round-robin pool: --submit-workspaces overrides single submit_workspace;
    # empty -> single-workspace mode.
    submit_workspaces_raw = (args.submit_workspaces
                             or os.environ.get("SAFE_OPTIMIZE_SUBMIT_WORKSPACES")
                             or "")
    submit_workspaces_pool = [w.strip() for w in submit_workspaces_raw.split(",")
                              if w and w.strip()]
    volume = (args.volume
              or os.environ.get("SAFE_OPTIMIZE_VOLUME")
              or DEFAULT_VOLUME)
    gpu_type_input = (args.gpu_type
                      or os.environ.get("SAFE_OPTIMIZE_GPU_TYPE")
                      or DEFAULT_GPU_TYPE)
    gpu_type = canonical_gpu_type(gpu_type_input)
    gpu_profile = normalize_gpu_profile(gpu_type, warn=False) or DEFAULT_GPU_PROFILE
    # Unset by default (install.sh clones a writable per-session copy); only an
    # explicit path pins one. Empty -> inferencexPath="" suppresses SaFE's default.
    inferencex_path = (args.inferencex_path
                       or os.environ.get("SAFE_OPTIMIZE_INFERENCEX_PATH")
                       or "")
    oob_path = (args.oob_path
                or os.environ.get("SAFE_OPTIMIZE_OOB_PATH")
                or "")
    tracelens_root = (args.tracelens_root
                      or os.environ.get("SAFE_OPTIMIZE_TRACELENS_ROOT", ""))
    try:
        kernel_backends = parse_kernel_backends(args.kernel_opt_backends)
    except ValueError as e:
        log.error("%s", e)
        return 2

    if not api_key and not args.dry_run:
        log.error("no API key set (CLAW_API_KEY / SAFE_API_KEY / --api-key)")
        return 2

    log.info("SaFE base_url=%s register_workspace=%s submit_workspace=%s volume=%s",
             base_url, register_workspace, submit_workspace, volume)
    log.info("Cluster prompt fields: gpu_type=%s gpu_profile=%s inferencex_path=%s oob_path=%s tracelens_root=%s",
             gpu_type, gpu_profile, inferencex_path, oob_path, tracelens_root)
    log.info("Kernel backends: %s", ", ".join(kernel_backends))
    if submit_workspaces_pool:
        log.info("submit round-robin pool: %s (overrides --submit-workspace)",
                 ",".join(submit_workspaces_pool))
    if register_workspace != submit_workspace and not submit_workspaces_pool:
        log.info("cross-workspace mode — needs SaFE selectLocalPath path-accessible "
                 "fallback to be deployed; will 400 on submit_task otherwise")

    hf = HuggingFaceClient(args.hf_token)
    # Dry-run never hits SaFE, so a placeholder token is fine.
    safe = SafeOptimizeClient(
        base_url, api_key or "dry-run",
        register_workspace=register_workspace,
        submit_workspace=submit_workspace,
        volume=volume,
        submit_workspaces_pool=submit_workspaces_pool or None,
    )
    if submit_workspaces_pool and args.pool_index:
        try:
            safe._submit_ws_counter = max(int(args.pool_index), 0)
            log.info(
                "submit round-robin offset seeded from pool_index=%s",
                args.pool_index,
            )
        except ValueError:
            log.warning("invalid pool_index=%r; round-robin starts at 0", args.pool_index)

    if args.hf_top:
        log.info("fetching HF top-%d (>=%.1fB)", args.hf_top, args.min_params)
        try:
            repos = hf.top_models(args.hf_top, min_params_b=args.min_params)
        except Exception as e:
            log.error("HF top-N fetch failed: %s", e)
            return 1
        log.info("selected %d models: %s", len(repos), repos)
    else:
        repos = list(args.model or [])

    if not repos:
        log.error("no models to process")
        return 1

    overrides = {
        "framework": args.framework,
        "precision": args.precision,
        "tp": args.tp,
        "concurrency": args.concurrency,
        "image": args.image,
    }
    pool_metadata = {
        "pool_id": args.pool_id,
        "pool_index": args.pool_index,
        "batch_index": args.pool_batch_index,
        "batch_size": args.pool_batch_size,
        "source_task_id": args.pool_source_task_id,
    }

    # ── Multi-node resolution ──────────────────────────────────────────────
    # --nodes>1 spans an N-node RayJob. The SaFE task body has no node count, so
    # force mode=claw and append the RayJob topology to the prompt suffix
    # (mirrors the Claw-direct CI). --nodes is global but effectively per-model.
    effective_mode = args.mode
    effective_prompt_suffix = args.prompt_suffix or None
    _nodes = args.nodes or 1
    # tp must fit nodes*8 GPUs (enforced for single-node too) so a stray --tp 16
    # is rejected at submit time, not at runtime on an 8-GPU sandbox.
    if args.tp and args.tp > _nodes * 8:
        log.error("--tp %d exceeds --nodes %d * 8 GPUs = %d; lower --tp or raise "
                  "--nodes (tp>8 requires multi-node).",
                  args.tp, _nodes, _nodes * 8)
        return 2
    if _nodes > 1:
        if effective_mode != "claw":
            log.warning("--nodes %d > 1 needs RayJob fan-out; forcing --mode claw "
                        "(was %r)", _nodes, effective_mode)
            effective_mode = "claw"
        rayjob_image = (args.rayjob_image or args.image or "").strip()
        if not rayjob_image:
            log.warning("--nodes %d > 1 but no --rayjob-image/--image set; the agent "
                        "must pick a RayJob image itself", args.nodes)
        effective_prompt_suffix = (
            (args.prompt_suffix or "")
            + _multinode_prompt_suffix(args.nodes, rayjob_image)
        ) or None
        log.info("multi-node: nodes=%d tp=%s mode=%s rayjob_image=%s",
                 args.nodes, args.tp, effective_mode,
                 rayjob_image or "(agent-chosen)")

    # Pre-submit jitter. When a large matrix fans out, every job otherwise hits
    # register/submit (and Claw-session creation) in the same instant — the
    # backend then sheds load with HTTP 500 "failed to create Claw session" /
    # 504. A per-process random(0..N) sleep here spreads the herd across an
    # N-second window so the backend sees a trickle rather than a spike. The
    # submit_task retry loop still backstops any residual collision.
    jitter = max(0, args.submit_jitter_sec)
    if jitter > 0 and not args.dry_run:
        d = random.uniform(0, jitter)
        log.info("submit jitter: sleeping %.1fs (window 0-%ds) to de-sync from "
                 "other parallel jobs before hitting SaFE", d, jitter)
        time.sleep(d)

    records: list[SubmissionRecord] = []
    for repo in repos:
        log.info("=" * 60)
        log.info("Model: %s", repo)
        rec = process_model(
            repo, hf, safe, overrides,
            args.isl, args.osl, args.dry_run, args.hf_token,
            manual_mode=args.manual,
            mode=effective_mode,
            gpu_type=gpu_type,
            inferencex_path=inferencex_path,
            oob_path=oob_path,
            tracelens_root=tracelens_root,
            prompt_prefix=args.prompt_prefix or None,
            prompt_suffix=effective_prompt_suffix,
            kernel_backends=kernel_backends,
            max_hours=args.max_hours,
            target_gain=args.target_gain,
            results_path=args.results_path,
            pool_metadata=pool_metadata,
        )
        records.append(rec)

    submitted = sum(1 for r in records if r.status == "submitted")
    submit_failed = [r for r in records if r.status == "failed"]
    log.info("=" * 60)
    log.info("Submitted: %d ok, %d failed, %d total",
             submitted, len(submit_failed), len(records))
    for r in submit_failed:
        log.warning("  submit failed: %s — %s", r.model, r.error)

    # Wait + collect (default on); skip on dry-run.
    if not args.dry_run and submitted > 0 and args.wait_for_completion:
        process_completion(
            safe, records,
            artifacts_dir=Path(args.artifacts_dir),
            task_timeout_min=args.task_timeout_min,
            poll_s=args.poll_interval_s,
            collect=args.collect_artifacts,
            all_artifacts=args.all_artifacts,
            parallel=args.wait_parallel,
        )

        from collections import Counter
        final_counts = Counter(r.final_status or "Pending"
                               for r in records if r.task_id)
        delivery_counts = Counter(r.ci_status or "Unknown"
                                  for r in records if r.task_id)
        log.info("=" * 60)
        log.info("Final task statuses: %s",
                 ", ".join(f"{k}={v}" for k, v in sorted(final_counts.items())))
        log.info("CI delivery statuses: %s",
                 ", ".join(f"{k}={v}" for k, v in sorted(delivery_counts.items())))
        delivered_non_success = [
            r for r in records
            if r.task_id and r.final_status != "Succeeded" and r.ci_success
        ]
        if delivered_non_success:
            log.warning(
                "SaFE terminal status was non-success, but CI artifacts were delivered: %s",
                ", ".join(
                    f"{r.model}:{r.final_status or 'Pending'}->{r.ci_status}"
                    for r in delivered_non_success
                ),
            )
        non_success = [
            r for r in records
            if r.task_id and r.final_status != "Succeeded" and not r.ci_success
        ]
    else:
        non_success = []

    # Manifest written after wait/collect so it captures final_status etc.
    if args.output_dir:
        write_manifest(Path(args.output_dir), records,
                       base_url, register_workspace, submit_workspace, volume)

    if args.dry_run:
        return 0
    if non_success:
        log.error("Non-success terminal statuses without deliverable artifacts: %s",
                  ", ".join(
                      f"{r.model}:{r.final_status or 'Pending'}:{r.ci_status or 'Unknown'}"
                      for r in non_success
                  ))
        return 2
    context_skipped = [
        r for r in records
        if r.status == "skipped" and (r.error or "").startswith("context_too_short:")
    ]
    if submitted == 0 and records and len(context_skipped) == len(records):
        log.info("All models skipped by policy: context_too_short")
        return 0
    return 0 if submitted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
