#!/usr/bin/env python3
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
  SAFE_BASE_URL | SAFE_API_URL       base URL (default: https://core42.primus-safe.amd.com)
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

DEFAULT_API_URL = "https://core42.primus-safe.amd.com"
# Two-workspace split — necessary because of conflicting K8s constraints:
#
#   register: core42-hyperloom + /wekafs
#     - core42-hyperloom workspace has /wekafs mounted ReadWriteMany so model
#       weights can actually be downloaded there.
#     - The path /wekafs/models/<repo> is on a shared weka volume that ALL
#       sandbox pods (including core42-sandbox) can read via their own
#       ReadOnlyMany /wekafs mount — i.e. it's effectively global storage.
#
#   submit: core42-sandbox
#     - Only workspaces whose spec.scopes includes 'Sandbox' can host the
#       Sandbox-typed K8s workload that SaFE creates for an optimization
#       task. core42-hyperloom intentionally excludes Sandbox (ops design)
#       and the admission webhook (vworkload.kb.io) rejects with
#       Primus.00003 if you try.
#
# Requires SaFE backend selectLocalPath to do path-accessible fallback when
# submit_workspace != register_workspace (see SaFE/.../optimization/
# model_helper.go — same pattern as sft.go:resolveModelLocalPathFromK8sModel).
# Until that lands, submit_task will 400; --submit-workspace can be set equal
# to --register-workspace as a workaround.
DEFAULT_REGISTER_WORKSPACE = "core42-hyperloom"
DEFAULT_SUBMIT_WORKSPACE = "core42-sandbox"
DEFAULT_VOLUME = "/wekafs"
DEFAULT_PROXY = "harbor.core42.primus-safe.amd.com/proxy"
# Cluster-aware defaults: SaFE backend's NormalizePromptConfig uses MI355X /
# /hyperloom/InferenceX which are wrong for core42 (it's MI300X and the
# canonical hyperloom-managed InferenceX checkout lives at
# /wekafs/hyperloom/InferenceX). Without overriding here the generated prompt
# sends the agent on a 5-10 min wild goose chase looking for
# /hyperloom/InferenceX, and it picks GPU-architecture-wrong heuristics later.
#
# /wekafs/hyperloom/InferenceX is the same priority path that:
#   - inference_optimizer/cli.py:1586          uses as the V2 skill default
#   - inference_optimizer/scripts/install.sh   bootstraps into
#   - .github/workflows/inference-optimization-ci.yml lists FIRST after
#     ${NFS_ROOT}/InferenceX in the config-file probe loop
# Keeping this aligned avoids the agent landing on a stale /wekafs/InferenceX
# checkout (left over from earlier non-hyperloom layouts on some sandboxes).
DEFAULT_GPU_TYPE = "MI300X"
DEFAULT_INFERENCEX_PATH = "/wekafs/hyperloom/InferenceX"
# OOB + TraceLens live next to InferenceX on the same hyperloom-managed mount.
# Like DEFAULT_INFERENCEX_PATH these are core42 fallbacks; --oob-path /
# --tracelens-root (or SAFE_OPTIMIZE_* env) override per-cluster.
DEFAULT_OOB_PATH = "/wekafs/hyperloom/OOB"
DEFAULT_TRACELENS_ROOT = "/wekafs/hyperloom/TraceLens-internal"
DEFAULT_KERNEL_BACKENDS = ["GEAK", "Claude Code", "Codex"]
DEFAULT_MAX_HOURS = 12.0
DEFAULT_TARGET_GAIN = 30.0
DEFAULT_RESULTS_PATH = "$RESULT_DIR"

_KERNEL_BACKEND_ALIASES = {
    "geak": "GEAK",
    "claude": "Claude Code",
    "claude-code": "Claude Code",
    "claude code": "Claude Code",
    "codex": "Codex",
    "cursor": "Cursor",
}

# Canonical prompt prefix lives in ci/prompt_prefix.txt next to this script.
# Single source of truth — same file is read by the GitHub workflow Submit
# step (.github/workflows/optimize-submit.yml) as the schedule-trigger
# fallback, and also serves as the argparse default here so any direct
# CLI invocation (manual debugging, peer scripts, etc.) gets the same
# prefix without having to set $SAFE_OPTIMIZE_PROMPT_PREFIX every time.
_PROMPT_PREFIX_FILE = Path(__file__).resolve().parent / "prompt_prefix.txt"


def _load_default_prompt_prefix() -> str:
    """Resolve the default prompt prefix for ``--prompt-prefix``.

    Resolution order:
      1. ``$SAFE_OPTIMIZE_PROMPT_PREFIX`` (lets ops override per-run without
         editing the file or argparse call)
      2. ``ci/prompt_prefix.txt`` next to this script (canonical content)
      3. empty string (caller is responsible — submit then refuses to ship
         an empty prefix)
    """

    env_value = os.environ.get("SAFE_OPTIMIZE_PROMPT_PREFIX", "")
    if env_value:
        return env_value
    try:
        if _PROMPT_PREFIX_FILE.is_file():
            return _PROMPT_PREFIX_FILE.read_text(encoding="utf-8")
    except OSError:
        pass
    return ""


def parse_kernel_backends(raw: str | None) -> list[str]:
    """Normalize user-facing kernel backend names for SaFE's API payload."""

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

# Architecture name suffixes that indicate the model is a generative LM (i.e.
# something we can run inference benchmarks on). Used to filter out embedding
# models, encoders, classifiers, etc. — for those, sglang/vllm won't even
# start a server.
GENERATIVE_ARCH_SUFFIXES: tuple[str, ...] = (
    "ForCausalLM",
    "ForConditionalGeneration",
    "LMHeadModel",
    "ForSeq2SeqLM",
)


def is_generative_arch(arch: str) -> bool:
    """True if the HF model architecture is suitable for causal-LM-style
    inference. Falls back to False for empty / unknown arch — better to
    skip than to waste a sandbox slot on something that won't run.
    """
    if not arch:
        return False
    return any(arch.endswith(s) for s in GENERATIVE_ARCH_SUFFIXES)


def _proxy() -> str:
    return os.environ.get("HARBOR_PREFIX", DEFAULT_PROXY)


def _default_sglang_image() -> str:
    # v0.5.11 (2026-05-05): Spec V2 by default + DFLASH on ROCm + all-reduce/RMSNorm fusion.
    # Confirmed available at harbor.core42.primus-safe.amd.com/proxy/lmsysorg/sglang.
    return f"{_proxy()}/lmsysorg/sglang:v0.5.11-rocm720-mi30x"


def _default_vllm_image() -> str:
    # v0.19.0 (skip-listed v0.20.0 to stay one minor ahead of InferenceX baseline v0.17.0
    # while avoiding any v0.20 breakage; bump to v0.20 once stability is confirmed).
    return f"{_proxy()}/vllm/vllm-openai-rocm:v0.19.0"


# ── HuggingFace client ──────────────────────────────────────────────────────────

class HuggingFaceClient:
    """Minimal HF API client for model metadata + top-models discovery."""

    BASE = "https://huggingface.co"

    def __init__(self, token: str = "", timeout: int = 15):
        self.timeout = timeout
        self._sess = requests.Session()
        self._sess.headers["User-Agent"] = "hyperloom-optimize-submit/1.0"
        if token:
            self._sess.headers["Authorization"] = f"Bearer {token}"

    def _get(self, path: str) -> dict | list:
        resp = self._sess.get(f"{self.BASE}{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def model_info(self, repo_id: str) -> dict:
        return self._get(f"/api/models/{repo_id}")  # type: ignore[return-value]

    def model_config(self, repo_id: str) -> dict:
        return self._get(f"/{repo_id}/resolve/main/config.json")  # type: ignore[return-value]

    def top_models(self, limit: int, min_params_b: float = 0.0) -> list[str]:
        """Return top-N text-generation repos by downloads, optionally filtered by size.

        Pool-then-filter: the listing API's ``filter=text-generation`` matches
        on tags only and lets through embedding / sentence-similarity / classifier
        models that happen to carry that tag. We re-validate per-repo against:

          1. ``pipeline_tag == "text-generation"`` (model card classification)
          2. ``architectures[0]`` ends in a generative suffix (ForCausalLM /
             ForConditionalGeneration / LMHeadModel / ForSeq2SeqLM)

        Either signal failing → skip. Saves a sandbox slot per garbage candidate.
        Gated repos that 401 on metadata also get silently skipped.
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

            # We need model_info() for both pipeline_tag and (optionally)
            # safetensors.total — fetch once.
            try:
                info = self.model_info(repo)
            except Exception:
                # Gated / network error → skip silently
                continue

            pipeline_tag = (info.get("pipeline_tag") or "").strip()
            if pipeline_tag and pipeline_tag != "text-generation":
                log.info("skip %s: pipeline_tag=%s (not text-generation)",
                         repo, pipeline_tag)
                continue

            if min_params_b > 0:
                total = (info.get("safetensors") or {}).get("total", 0)
                if (total / 1e9) < min_params_b:
                    continue

            # Final gate: config.json must be reachable AND architectures[0]
            # must be a generative LM. If config.json 401/403s (gated repo
            # whose model card is public but the actual files aren't grant-ed
            # to our HF token), skip the candidate so the pool auto-replaces
            # it with the next eligible repo. Same treatment for non-generative
            # architectures (BertModel / Qwen3Model / classifier / etc.).
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
    arch: str
    framework: str
    precision: str
    tp: int
    concurrency: int
    image: str
    params_b: float


def _quant_type(config: dict) -> str:
    """Read the quantization tag from a HF config.json.

    HF doesn't standardize on a single field name across vendors:
      - quant_algo           : NVIDIA modelopt (NVFP4 / FP8 / W4A8_AWQ / INT8_SQ).
                               Quant_method on these is "modelopt" — that's the
                               *tool name* not the precision, so we have to
                               look at quant_algo. Highest priority.
      - quant_type           : transformers built-in (older / GPTQ flow).
      - quantization_type    : legacy variant seen on a few older repos.
      - quant_method         : the de-facto current standard for most other
                               vendors (gpt-oss mxfp4, AWQ via autoawq,
                               DeepSeek-V3.x fp8, ...).
      - method               : occasional corner case.
    Try them in order; first non-empty wins. Always lowercase so callers can
    just do string contains/equality checks against {fp8, mxfp4, nvfp4, awq, ...}.
    """
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
    qt = _quant_type(config)
    if "fp8" in qt:   return "FP8"
    if "mxfp4" in qt: return "FP4"
    if "nvfp4" in qt: return "FP4"
    if "int4" in qt:  return "INT4"
    if "gptq" in qt:  return "INT4"
    if "awq" in qt:   return "INT4"
    return "FP8"  # unquantized default for MI300X


def detect_param_count(hf_info: dict, config: dict) -> float:
    total = (hf_info.get("safetensors") or {}).get("total", 0)
    if total:
        return total / 1e9
    h = config.get("hidden_size", 0)
    n = config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)
    if h and n:
        return (12 * h * h * n + vocab * h) / 1e9
    return 0.0


def detect_tp(params_b: float, precision: str = "BF16") -> int:
    """Pick a tensor-parallel size based on quantization-aware weight footprint.

    The old logic only looked at params_b and hit two real bugs in production:
      - gpt-oss-20b   (21.5B FP4 → ~11 GB weights) was given TP=4 even though
        it fits comfortably on 1× MI300X (192 GB). TP=4 then failed the
        baseline benchmark (vllm + EP=1 + non-divisible shape on a 21B MoE).
      - 30B FP8 models (e.g. Qwen3-Coder-30B-A3B) similarly got TP=4 when
        TP=1 would have been fine.

    Bytes per param by precision:
      FP4 / INT4 / NVFP4   = 0.5
      FP8                  = 1.0
      BF16 / FP16 (default)= 2.0

    Snap-to thresholds (after ~30% headroom for KV cache + activations):
      weight_gb < 50   → TP=1   (single-GPU comfortable)
      weight_gb < 280  → TP=4   (single GPU technically fits but multi-GPU
                                 gives better throughput; matches MI300X
                                 memory budget for 1024/1024 ISL/OSL @ conc=64)
      weight_gb ≥ 280  → TP=8   (must shard across the full node)
    """
    if params_b <= 0:
        return 1
    p = (precision or "").upper()
    if p in ("FP4", "INT4", "NVFP4", "MXFP4"):
        bytes_per_param = 0.5
    elif p == "FP8":
        bytes_per_param = 1.0
    else:                       # BF16 / FP16 / unknown
        bytes_per_param = 2.0
    weight_gb = params_b * bytes_per_param
    if weight_gb < 50:   return 1
    if weight_gb < 280:  return 4
    return 8


def detect_concurrency(tp: int, framework: str) -> int:
    if framework == "vllm":
        return 64 if tp <= 4 else 16
    return 64 if tp == 1 else 32 if tp <= 4 else 64


def detect_image(framework: str) -> str:
    return _default_vllm_image() if framework == "vllm" else _default_sglang_image()


def auto_detect(hf: HuggingFaceClient, repo_id: str) -> DetectedConfig | None:
    log.info("[%s] fetching HF metadata", repo_id)
    try:
        info = hf.model_info(repo_id)
        config = hf.model_config(repo_id)
    except Exception as e:
        log.error("[%s] HF fetch failed: %s", repo_id, e)
        return None

    arch = (config.get("architectures") or ["unknown"])[0]

    # Defensive: even when the user passes --model X explicitly, refuse to
    # submit non-generative repos. sglang/vllm won't start a server for these
    # (Qwen3-Embedding-8B, BertModel, etc.) and the task would just burn a
    # sandbox slot before failing in phase 0.
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
    tp = detect_tp(params_b, precision)
    conc = detect_concurrency(tp, framework)
    image = detect_image(framework)

    cfg = DetectedConfig(
        arch=arch, framework=framework, precision=precision,
        tp=tp, concurrency=conc, image=image, params_b=params_b,
    )
    log.info("[%s] arch=%s params=%.1fB framework=%s precision=%s tp=%d conc=%d",
             repo_id, arch, params_b, framework, precision, tp, conc)
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
        self.base_url = base_url.rstrip("/")
        # Where the model gets registered + downloaded (must allow RW writes
        # to the configured volume).
        self.register_workspace = register_workspace
        # Where the optimization task is created (must allow Sandbox scope).
        # Can equal register_workspace when both constraints are satisfied
        # by a single workspace (rare in practice on core42).
        self.submit_workspace = submit_workspace
        # Optional round-robin pool: when set, each submit_task picks the
        # next workspace from the list instead of always using
        # self.submit_workspace. Lets a large batch span both
        # core42-sandbox (128 GPU) and core42-hyperloom (256 GPU)
        # without manually splitting the model list.
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

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self._sess.request(method, url, json=body, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.content else {}

    def find_model(self, repo_id: str) -> dict | None:
        """Look up an existing SaFE Model by HF source URL, scoped to register_workspace.

        We filter by the *register* workspace because that's where the
        canonical Model CR + LocalPaths live. Submitting to a different
        submit_workspace later relies on selectLocalPath's path-accessible
        fallback to find the file via shared storage.
        """
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

        Two flavors:
          * ``local_path`` set → accessMode=local_path. SaFE skips its own
            Download Job entirely (model_controller.go:97 sets phase=Ready
            directly) because we promise the files are already on disk at
            ``local_path``. This is the path the prewarm step writes to.
          * ``local_path`` empty → accessMode=local. SaFE creates a K8s
            Download Job and pulls from HuggingFace itself (slow / fragile
            on big repos). Kept as a fallback when prewarm cannot run.
        """
        if local_path:
            # local_path mode bypasses SaFE's HF metadata fetch, so the
            # caller MUST provide displayName (validated by
            # handlers/model-handlers/models.go::createModelFromLocalPath:380).
            # We mirror the convention SaFE uses for HF-downloaded models —
            # the trailing path segment of the repo ID (e.g. "Qwen3-Coder-Next").
            #
            # Sanitization: SaFE feeds displayName straight into
            # commonutils.GenerateName which becomes the K8s Model.amd.com
            # metadata.name. That field must satisfy RFC 1123 (lowercase
            # [a-z0-9-.], 1-63 chars, start/end alphanumeric). Without this
            # sanitization the K8s API rejects displayName="Qwen3-Coder-Next"
            # with HTTP 500 "Model.amd.com \"Qwen3-Coder-Next-xxxxx\" is
            # invalid: metadata.name: Invalid value ...". SaFE should ideally
            # sanitize on the backend, but for now we hand-deliver a clean
            # name. Keep the original repo basename in the source.url field
            # so the dashboard can still show pretty-cased text via metadata
            # lookups.
            import re
            raw = repo_id.split("/")[-1] or repo_id
            cleaned = re.sub(r"[^a-z0-9.-]+", "-",
                             raw.lower()).strip(".-") or "model"
            # Trim to 50 chars to leave headroom for the -xxxxx suffix
            # GenerateName appends (K8s metadata.name max is 63).
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
        # Pick the workspace for this submit. Default = self.submit_workspace
        # (single-workspace mode). When a round-robin pool is configured,
        # cycle through it so a batch of N tasks spreads across the pool
        # evenly. Counter is per-instance and not thread-safe — fine because
        # process_model calls submit_task serially in the dispatch loop;
        # only wait_and_collect is run in parallel and that's after
        # submit_task already returned.
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
        # Override SaFE backend's wrong-for-core42 defaults (MI355X /
        # /hyperloom/InferenceX). See DEFAULT_GPU_TYPE/_INFERENCEX_PATH above.
        if gpu_type:
            body["gpuType"] = gpu_type
        if inferencex_path:
            body["inferencexPath"] = inferencex_path
        if oob_path:
            body["oobPath"] = oob_path
        if tracelens_root:
            body["tracelensRoot"] = tracelens_root
        # Optional prefix/suffix forwarded to BuildHyperloomPrompt on the
        # SaFE side. We use this to point the skill at the alternate
        # inference_optimizer SKILL.md the batch lives in, before the
        # auto-generated body kicks in.
        if prompt_prefix:
            body["promptPrefix"] = prompt_prefix
        if prompt_suffix:
            body["promptSuffix"] = prompt_suffix
        return self._request("POST", "api/v1/optimization/tasks", body)

    # ── Task lifecycle ──

    # Lifecycle states observed in SaFE (see types.go OptimizationTaskStatus).
    TERMINAL_TASK_STATUSES = {"Succeeded", "Failed", "Interrupted"}

    def get_task(self, task_id: str) -> dict:
        return self._request("GET", f"api/v1/optimization/tasks/{task_id}")

    def wait_task_done(
        self, task_id: str, timeout_min: int = 480, poll_s: int = 60,
    ) -> tuple[str, dict]:
        """Wait until the task reaches a terminal status. Returns (status, last_task_dict).

        Strategy: prefer Claw SSE stream (sees the agent's `ResultMessage` event the
        instant phase 11 finishes). Fall back to SaFE optimization-API polling if
        no clawSessionId is available yet, or if the SSE stream fails.

        Why SSE: SaFE's optimization-task `status` field is updated by a
        background controller and frequently lags Claw by minutes (often shows
        Timeout while the agent is still running phase 9/10/11). SSE on the
        underlying Claw session is the source of truth for actual completion.

        Returns ('Timeout', {}) if neither stream nor polling sees a terminal
        status before the deadline.
        """
        log.info("[task %s] waiting for completion (timeout=%dm, poll=%ds)",
                 task_id, timeout_min, poll_s)
        deadline = time.time() + timeout_min * 60

        # Step 1: wait briefly for clawSessionId to materialize. SaFE creates the
        # Claw session as part of submit_task, so this almost always returns
        # immediately; cap the wait at 60s to fall through to polling if not.
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
            # If SaFE already shows terminal, trust it.
            if sf_status in self.TERMINAL_TASK_STATUSES:
                return sf_status, last_task
            # Stopped = sandbox pod exited. That's our real end-of-task
            # signal. SaFE's controller usually lags the sandbox shutdown
            # by 10-180s before flipping the task to Succeeded/Failed
            # (writes the optimization report check, etc.), so short-poll
            # SaFE for up to 5 minutes to pick up its verdict.
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
                # Sandbox is gone but SaFE controller hasn't settled. Treat
                # as Succeeded so build_summary still has a chance to read
                # whatever ci_metrics.json the agent wrote — Stage A/B of
                # collect_artifacts will pull from /wekafs/users/... and
                # the actual gain numbers tell the real story.
                log.info("[task %s] SaFE never settled within 5min after "
                         "sandbox stop — returning Succeeded (collect "
                         "step will read ci_metrics.json directly)",
                         task_id)
                return "Succeeded", last_task
            # deadline = we've burned the per-task wall clock.
            if sse_reason == "deadline":
                return "Timeout", last_task
            # Anything else (idle_timeout, stream_error) is *inconclusive*:
            # Claw's `/chat/sessions/.../messages` stream sometimes goes quiet
            # (only `: keepalive` heartbeats) for many minutes while the agent
            # is busy in tool calls, and we can't tell that apart from the agent
            # having actually finished. Don't trust SSE alone — fall through to
            # SaFE polling and wait for a real terminal status.
            log.info("[task %s] SSE inconclusive (reason=%s, sf_status=%s) — "
                     "falling back to SaFE polling for terminal status",
                     task_id, sse_reason, sf_status or "?")

        # Step 2 (fallback / continuation): SaFE optimization-API polling.
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

        Stream protocol (text/event-stream over /claw-api/v1/chat/sessions/<id>/messages):
          - Each event is `id:`/`event:`/`data:` lines + blank-line separator.
          - Historical events are replayed from the beginning, then the stream
            stays open with `: keepalive` heartbeats.

        End-of-task signal — IMPORTANT — we used to return on the first
        ``ResultMessage``, but that's emitted at the end of EVERY agent
        turn (V2 skill runs phase 0-10 over 1-3h with dozens of turns, and
        each one fires a ResultMessage). The first turn's ResultMessage
        was being mistaken for task completion, leaving phase=0 tasks
        marked Succeeded with artifact_count=0.

        The only reliable end-of-task signal is the sandbox lifecycle
        event ``sandboxStatus phase=Stopped/Terminated/Failed`` — that
        fires when the sandbox pod actually exits, well after phase 10
        finishes (or when SaFE's controller kills it).

        Returns one of:
          - "Stopped":       sandboxStatus phase reached Stopped/Terminated/Failed
          - "idle_timeout":  no events for >10min after replay finished
          - "deadline":      hit the per-task wall-clock deadline
          - "stream_error":  HTTP error or socket exception (caller will fall back)
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
                    # NOTE: ResultMessage is intentionally not a return
                    # signal — see _sse_wait_until_done docstring. It still
                    # refreshes last_evt above so the idle-timeout window
                    # only starts ticking when the agent goes silent.
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
        # Stream closed cleanly without ResultMessage — treat as idle.
        return "idle_timeout"

    # _claw_session_id_for() is kept because _sse_wait_until_done() still needs
    # the Claw session id to subscribe to the SSE stream for real-time completion
    # detection. It is no longer used by list_artifacts / download_artifact.

    def _claw_session_id_for(self, task_id: str) -> str | None:
        """Resolve clawSessionId for a task, with a small per-instance cache.

        Calls GET /api/v1/optimization/tasks/<id> once and caches the result.
        Returns None if SaFE doesn't have a session attached (e.g. task that
        failed before Claw session creation).
        """
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

        The bug that returned only 1 file ("ListSessionFiles") was fixed in
        SaFE backend (2026-05-12); the endpoint now returns the full artifact
        tree, including files under sandbox subdirectories like
        ``hyperloom/ci_metrics.json``. Verified by
        .github/workflows/verify-safe-api.yml (run 25709271562).

        Returned items shape:
          ``{"path": "...", "size": ..., "lastModified": "...", "downloadPath": "..."}``
        ``downloadPath`` is server-relative; download_artifact() uses it
        directly when present, otherwise falls back to constructing the
        ``/artifacts/download?path=`` URL from ``path``.
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
        """Download a single task artifact via the SaFE standard endpoint.

        Accepts either a string path (e.g. ``"hyperloom/ci_metrics.json"``) or
        an item dict returned by list_artifacts. When given an item dict and
        the dict has a ``downloadPath`` field, that URL is used directly
        (preferred — SaFE may change query-string format).
        """
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
        data = self.download_artifact(task_id, path_or_item)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        return len(data)


# ── Per-model record ────────────────────────────────────────────────────────────

@dataclass
class SubmissionRecord:
    model: str
    status: str = "pending"            # local stage: submitted/dry-run/skipped/failed
    task_id: str | None = None
    # Claw session UUID (e.g. 1c11a036-9ef5-47d1-8f52-ca2398d05078) that
    # SaFE creates when submit_task runs. Used by the dashboard to deep-link
    # back to the chat transcript + by us to correlate ci_metrics.json on
    # /wekafs/users/<uid>/<session>/ with the SaFE task. Populated from
    # last_task.clawSessionId in wait_and_collect_one.
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
    # Audit fields surfaced through manifest / ci_metrics / session_breakdown
    # so each persisted artifact is self-describing.
    category: str | None = None             # moe / dense / "" — from detected.arch
    sandbox_duration_seconds: float | None = None  # SaFE startedAt -> finishedAt
    # Filled in by process_completion when --wait-for-completion is on.
    final_status: str | None = None    # SaFE: Succeeded/Failed/Interrupted/Timeout
    final_phase: int | None = None     # currentPhase at terminal moment
    final_message: str | None = None   # task.Message
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
    rec = SubmissionRecord(
        model=repo_id,
        overrides={k: v for k, v in overrides.items() if v is not None},
        pool={k: v for k, v in (pool_metadata or {}).items() if v not in (None, "")},
    )

    detected = None if manual_mode else auto_detect(hf, repo_id)
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

    framework = overrides.get("framework") or (detected.framework if detected else "")
    precision = overrides.get("precision") or (detected.precision if detected else "FP8")
    tp        = overrides.get("tp")        or (detected.tp if detected else 1)
    conc      = overrides.get("concurrency") or (detected.concurrency if detected else 64)
    image     = overrides.get("image") or (detected.image if detected else detect_image(framework))

    log.info("[%s] => mode=%s framework=%s precision=%s tp=%d conc=%d image=%s",
             repo_id, mode, framework, precision, tp, conc, image)

    display_name = f"{repo_id.split('/')[-1]}-{precision.lower()}-{framework}-mi300x"
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

    # Detect whether the prewarm step has already populated /wekafs/models/<slug>/
    # with the model files. When it has, we use SaFE's `local_path` accessMode
    # which makes model_controller.go set phase=Ready immediately without
    # creating any Download Job (no second-pass HF pull). This is the whole
    # point of prewarm — without local_path mode, SaFE re-downloads on top of
    # our files and we get nothing for the prewarm work.
    nfs_root = os.environ.get("NFS_ROOT", "/wekafs")
    target_slug = repo_id.replace("/", "-")
    target_dir = f"{nfs_root}/models/{target_slug}"
    use_local_path = False
    try:
        if os.path.isdir(target_dir):
            # Heuristic: any HF repo has at least config.json + tokenizer.* +
            # one weight shard. 5 files is well under that floor while
            # tolerating sparse model layouts.
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

    # Model resolution: find existing SaFE record OR register fresh.
    #
    # Stale Failed records are the common foot-gun: a previous run's download
    # job aborted (HF rate limit, transient NFS hang, etc.) and left a model
    # in phase=Failed. Without intervention, wait_ready sees Failed and
    # returns False instantly, so submit can never succeed even though our
    # prewarm step has since written the real files into /wekafs/models/.
    # When that happens we re-register: SaFE's POST /api/v1/playground/models
    # either issues a new model_id (preferred) or resets the existing one to
    # Pending and re-triggers the Download Job, which now sees prewarmed
    # files on /wekafs and finishes in seconds.
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
        result = safe.submit_task(
            model_id, display_name, framework, precision, tp, conc, isl, osl, image,
            mode=mode, gpu_type=gpu_type, inferencex_path=inferencex_path,
            oob_path=oob_path, tracelens_root=tracelens_root,
            prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix,
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

# Default artifact filter — matches what SaFE prompt_builder.go forces the agent
# to copy at end of Phase 10 (cp .../optimization_report.md .../ci_metrics.json
# /workspace/), plus reasonable variants observed in production.
DEFAULT_ARTIFACT_PATTERNS = (
    "optimization_report",   # matches optimization_report.md / *-optimization_report.md / etc.
    "ci_metrics.json",
    # Promoted to a key result alongside optimization_report.md +
    # ci_metrics.json in 2026-05 — claw-stats-service / V2 dashboard prefer
    # this over ci_metrics.json. Now part of ``_KEY_RESULT_SUFFIXES``, so
    # missing it WILL trigger the wekafs NFS fallback (same as the other
    # two contract files).
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
    if all_artifacts:
        return True
    p = path.lower()
    return any(pat in p for pat in DEFAULT_ARTIFACT_PATTERNS)


def _safe_local_path(artifacts_dir: Path, task_id: str, remote_path: str) -> Path:
    """Map a session-relative remote path to a local file path.

    Strips leading slashes and normalizes separators so we never escape the
    artifacts dir even if the remote returns absolute or '../' paths.
    """
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


# Files that MUST be present in the task directory for the run to count as
# "delivered". session_breakdown.json was promoted from optional/audit to a
# key result in 2026-05 because claw-stats-service / the V2 dashboard prefer
# it over ci_metrics.json. NFS legacy fallback scans for any of these
# suffixes; missing any one of them is what triggers the wekafs PERSIST_DIR
# rescue path in `_nfs_user_session_fallback`.
_KEY_RESULT_SUFFIXES: tuple[str, ...] = (
    "optimization_report.md",
    "ci_metrics.json",
    "session_breakdown.json",
)


def _norm_token(s: str) -> str:
    return (s or "").lower().replace("-", "").replace("_", "") \
        .replace(".", "").replace("/", "").replace(" ", "")


def _slug_token(s: str) -> str:
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
    """True when ``ci_metrics.json`` carries real, non-zero throughput."""
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


def _timestamp_hint_variants(value: str) -> set[str]:
    """Return path-matchable variants for skill session timestamps."""
    raw = value.strip()
    if not raw:
        return set()
    compact = raw.replace("T", "").replace("t", "").replace("Z", "").replace("z", "")
    variants = {raw, raw.lower()}
    if compact and compact != raw:
        variants.update({compact, compact.lower()})
    return variants


def _session_hints_from_artifact_items(items: list[dict]) -> set[str]:
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
    if not hints:
        return False
    norm_path = _norm_token(path)
    return any(_norm_token(hint) in norm_path for hint in hints)


def _parse_safe_timestamp(value: str | None) -> datetime | None:
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
    raw = value.strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw.upper(), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _session_timestamp_from_path(path: str) -> str:
    matches = re.findall(r"\b\d{8}T\d{6}Z\b", path, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else ""


def _timestamp_in_task_window(timestamp: str, rec: SubmissionRecord, margin_hours: int = 2) -> bool:
    ts = _parse_session_timestamp(timestamp)
    start = _parse_safe_timestamp(rec.safe_started_at)
    end = _parse_safe_timestamp(rec.safe_finished_at)
    if ts is None or start is None:
        return False
    if end is None:
        end = start + timedelta(hours=24)
    return (start - timedelta(hours=margin_hours)) <= ts <= (end + timedelta(hours=margin_hours))


def _record_has_task_window(rec: SubmissionRecord) -> bool:
    return _parse_safe_timestamp(rec.safe_started_at) is not None


def _category_from_arch(arch: str | None) -> str:
    """Coarse model-shape classification used for cron pool reporting.

    HF `architectures[0]` follows a stable naming convention: anything with
    "Moe" in the class name is a Mixture-of-Experts variant; everything
    else (Llama/Qwen/Mistral/Gemma/...) is a dense transformer. Returns
    ``""`` when arch is unknown so downstream JSON stays "n/a" rather than
    a misleading "dense".
    """
    if not arch:
        return ""
    return "moe" if "moe" in arch.lower() else "dense"


def _sandbox_duration_seconds(last_task: dict) -> float | None:
    """SaFE-side sandbox wallclock = finishedAt - startedAt.

    Both timestamps come from SaFE's optimization-task API
    (``startedAt`` is when the K8s sandbox pod actually transitioned to
    Running; ``finishedAt`` is when the agent process exited and SaFE
    sealed the task). Returns None when either field is missing or
    unparseable so we don't fabricate a duration.
    """
    from datetime import datetime
    start = (last_task or {}).get("startedAt") or ""
    end = (last_task or {}).get("finishedAt") or ""
    if not start or not end:
        return None
    try:
        # SaFE serializes UTC with trailing 'Z' — Python's fromisoformat
        # only accepts '+00:00', so normalise first.
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except Exception:
        return None
    delta = (e - s).total_seconds()
    return round(delta, 1) if delta >= 0 else None


def _find_hyperloom_commit_sha(start: Path) -> str:
    """Resolve the Hyperloom git SHA the sandbox cloned (for audit fields).

    Strategy (tried in order, first hit wins):

      1. ``hyperloom_source_commit.txt`` written by the agent inside the
         sandbox. The PERSIST step in ``ci/prompt_prefix.txt`` writes it
         at ``$RESULT_DIR/hyperloom_source_commit.txt`` and inside the V2
         session dir, so by the time NFS Stage B pulls things back it can
         show up at ``task_dir/hyperloom_source_commit.txt`` or
         ``task_dir/session/hyperloom_source_commit.txt`` (depth varies
         by which fallback collected the artifacts).

      2. CI runner environment. The agent doesn't always write the txt
         file (e.g. when it exits before the PERSIST snippet runs, or
         when V2 cli's session_dir eats the file), but the runner that
         dispatched the workflow ALWAYS knows the commit it pinned the
         sandbox to: it's ``HYPERLOOM_SOURCE_REF`` (set by ``optimize-
         submit.yml`` step 'Submit + wait + collect'), or ``GITHUB_SHA``
         (the head of the branch the workflow ran on, identical to
         ``HYPERLOOM_SOURCE_REF`` for any push-triggered run).
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
        # accept anything that looks like a SHA (or short SHA) — never trust
        # garbage from an empty / corrupted file.
        if 7 <= len(sha) <= 80 and all(c in "0123456789abcdef" for c in sha.lower()):
            return sha

    # Fallback: ask the CI runner environment. HYPERLOOM_SOURCE_REF is the
    # commit we explicitly pinned the sandbox to (preferred over GITHUB_SHA
    # when set, because HYPERLOOM_SOURCE_REF survives even dispatch with a
    # custom ref input). GITHUB_SHA is the unconditional fallback — every
    # GitHub Actions step has it.
    for env_var in ("HYPERLOOM_SOURCE_REF", "GITHUB_SHA"):
        env_sha = (os.environ.get(env_var) or "").strip()
        if 7 <= len(env_sha) <= 80 and all(c in "0123456789abcdef" for c in env_sha.lower()):
            return env_sha
    return ""


def _backfill_ci_metrics_file(path: Path, rec: SubmissionRecord) -> None:
    """Backfill task metadata into the three persisted-artifact shapes
    (ci_metrics.json / session_breakdown.json / manifest.json) so each is
    self-describing without cross-referencing GitHub Actions logs.

    Originally added because missing ``model`` blocked correlation with
    /wekafs/users entries (Run 25813519878). Extended in 2026-05 to also
    cover audit fields the sandbox / V2 cli collectors don't always reach:

    * ``image``                     — registry-qualified container image
                                      (auto-detected at register time)
    * ``hyperloom_commit``          — git SHA of the Hyperloom source tree
                                      the agent actually cloned
    * ``category``                  — moe / dense (from detected arch)
    * ``sandbox_duration_seconds``  — SaFE startedAt → finishedAt wallclock

    The function sniffs the filename and writes the right shape:
    * ``ci_metrics.json``       → flat top-level keys
    * ``session_breakdown.json``→ under ``session_meta`` sub-dict
    * ``manifest.json``         → flat top-level (matches the V2 cli
                                  schema v3 used by ``inference_optimizer/
                                  manifest.py``); ``category`` /
                                  ``sandbox_duration_seconds`` are extra
                                  keys we add — V2 cli ignores unknown
                                  fields on re-read.
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
        # The breakdown is keyed by a `session_meta` sub-dict (see
        # inference_optimizer/breakdown/schema.py::SessionMeta). Only write
        # when the field is currently empty so we don't overwrite anything
        # the V2 collectors did fill in.
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
        # session_duration_seconds is already declared in the v1.1 schema,
        # we just need to set it when the V2 collector left it blank.
        if rec.sandbox_duration_seconds is not None and not meta.get("session_duration_seconds"):
            meta["session_duration_seconds"] = rec.sandbox_duration_seconds
            changed = True
        # `category` isn't in the schema today but unknown fields are
        # tolerated; dashboards can pick it up without a schema bump.
        if rec.category and not meta.get("category"):
            meta["category"] = rec.category
            changed = True

    elif path.name == "manifest.json":
        # V2 cli schema (inference_optimizer/manifest.py): flat top-level
        # keys for image / code_revision / framework / tp / model_name.
        # These often ship as null when the sandbox didn't set the
        # HYPERLOOM_IMAGE env or when cwd-based `git rev-parse` failed;
        # we authoritatively backfill them from the CI side.
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
    """Reverse-write the audit fields back to the wekafs SOURCE files so
    operators ssh-ing into /wekafs/users/<uid>/<sess>/ see them directly
    without going through GHA artifact zips.

    Without this, the audit fields landed only in CI-runner-local
    artifacts/<task_id>/ci_metrics.json. wekafs originals stayed bare
    because they were written by the sandbox agent / V2 cli, neither of
    which knows the image/category/duration (those are SaFE-side facts
    only the CI runner sees after wait_task_done).

    Match strategy mirrors _nfs_user_session_fallback (Stage B):
      * exact `model` field match in the candidate ci_metrics.json, or
      * conservative session-dir-name match via _record_matches_session_dir.
    Only sessions modified in the last 24h are considered, to avoid
    backfilling stale runs on the same uid.

    Files updated per matching session: ci_metrics.json, manifest.json,
    session_breakdown.json, session_breakdown_v2.json (also under
    phase10_report/, results/, v2_session/ subdirs). Each is fed through
    _backfill_ci_metrics_file which already knows the right shape per
    filename — and which no-ops on files whose fields are already set.

    No-op when wekafs is not mounted (e.g. CI runner doesn't have NFS).
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
            # Confirm this session belongs to our task: a `model` field
            # match in any ci_metrics.json is the strongest signal; else
            # fall back to the conservative session-dir-name heuristic.
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
                if mf and _norm_token(mf.split("/")[-1]) == target:
                    matched = True
                    break
            if not matched and _record_matches_session_dir(rec, sess):
                matched = True
            if not matched:
                continue
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
                            n += 1
                            log.info("[task %s] wekafs backfill: %s",
                                     rec.task_id, p)
                    except Exception as e:
                        log.warning("[task %s] wekafs backfill failed for %s: %s",
                                    rec.task_id, p, e)
    return n


def _record_matches_session_dir(rec: SubmissionRecord, sess_name: str) -> bool:
    """Conservative directory-name match for /wekafs/users fallback.

    Prefer exact displayName slug because SaFE constructs it from model +
    precision + framework + gpu. Fall back to basename only with hard guards
    to avoid cross-wiring adjacent repos (Qwen2.5 vs Qwen2.5-AWQ, Nano vs
    Super, bnb vs non-bnb, etc.).
    """
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


def _candidate_model_dir_names(rec: SubmissionRecord) -> list[str]:
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
    def positive(value: object) -> bool:
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
    return (os.environ.get(name) or "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }


def _copy_session_tree(src_dir: str, dst_dir: Path) -> int:
    """Copy an entire persisted session directory into ``dst_dir``.

    Returns the number of files copied. Existing files are left untouched so
    previously downloaded SaFE artifacts keep their timestamp/content.
    """
    copied = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for root, dirnames, filenames in os.walk(src_dir):
        # Avoid recursively copying accidental nested CI artifact dirs.
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
    """Scan NFS result directories for files matching this model.

    Used when Claw API list_artifacts returned nothing useful (the very common
    case where the HyperloomV2 orchestrator writes outputs under its own
    session directory under /wekafs/users/<uid>/<arbitrary_dir>/, which SaFE's
    artifact API doesn't know about). Two-stage scan:

      A. Canonical CI dirs (legacy, kept for back-compat):
            /wekafs/hyperloom-results, /wekafs/results/ci,
            /wekafs/inference-optimization/results
         Match by directory name containing the model basename.

      B. Per-user session dirs (the one that actually matters for V2 today):
            /wekafs/users/<uid>/<session_name>/{,phase10_report,results}/ci_metrics.json
         Match by reading every ci_metrics.json under the tree and comparing
         its `model` JSON field to rec.model basename — much more accurate
         than directory-name fuzzy match (agent picks arbitrary names like
         `qwen25-coder-14b-awq-opt` vs canonical `Qwen2.5-Coder-14B-Instruct-AWQ`).

    Stage B is required for the 2026-05-12 batch onward where the runner now
    mounts /wekafs RW. Mirrors the fallback in ci/orchestrator.py (3a → 3b)
    so optimize_submit is consistent with the main CI pipeline.

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

    # Primary match: exact equality via JSON model fields when the agent wrote
    # them. Secondary match: conservative session-dir match. The secondary path
    # is needed for HyperloomV2 runs that persist result JSON without a `model`
    # field in the JSON (observed in Run 25813519878).
    target = _norm_token(model_basename)
    if not target:
        return copied

    def _model_field_matches(model_field: str) -> bool:
        observed = _norm_token(model_field)
        allowed = {
            target,
            _norm_token((rec.model or "").replace("/", "-")),
            _norm_token((rec.model_path or "").rstrip("/\\").split("/")[-1]),
        }
        allowed.discard("")
        return observed in allowed or _norm_token(model_field.split("/")[-1]) in allowed

    def _consider_result_file(path: str, session_dir: str, score_base: int) -> None:
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

    # rglob with depth 4/5 is enough for the legacy layouts we've seen:
    #   /wekafs/users/<uid>/<session>/ci_metrics.json                (depth 4)
    #   /wekafs/users/<uid>/<session>/phase10_report/ci_metrics.json (depth 5)
    #   /wekafs/users/<uid>/<session>/results/ci_metrics.json        (depth 5)
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

        # Current HyperloomV2 layout on core42:
        #   /wekafs/users/<uid>/<model-path-basename>/<YYYYmmddTHHMMSSZ>/
        #     session_breakdown.json
        #     reports/final.md
        #
        # SaFE's artifact API may list only workspace source files for these
        # runs, so there is no artifact-derived timestamp hint. In that case,
        # accept timestamp directories only when they are under this exact
        # SaFE user id + exact model dir and fall inside the task's
        # startedAt/finishedAt window.
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

    # Pick highest-confidence, then freshest match — same model can be re-run
    # several times.
    candidates.sort(reverse=True)
    _score, _mtime, best_ci, best_sess = candidates[0]
    log.info("[task %s] NFS user-session match: %s (from %d candidate(s), "
             "session=%s)", rec.task_id, best_ci, len(candidates), best_sess)

    # Copy the ci_metrics + any optimization_report.md sitting alongside or
    # under phase10_report/. Everything lands flat under task_dir/, same shape
    # build_summary.py expects.
    targets = [best_ci]
    for cand in [
        os.path.join(best_sess, "optimization_report.md"),
        os.path.join(best_sess, "phase10_report", "optimization_report.md"),
        os.path.join(best_sess, "reports", "final.md"),
    ]:
        if os.path.isfile(cand):
            targets.append(cand)
            break
    # Optional audit artifact — pull it back when the agent emitted it.
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
    """Wait for one task to finish, then optionally download its artifacts.

    Two-stage collection:
      1. SaFE API: list_artifacts(task_id) -> download each via /artifacts/download
      2. NFS fallback: if we still don't have ci_metrics.json + optimization_report.md,
         walk the canonical NFS result directories for a matching model dir
    """
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
        return rec

    # Stage 1: SaFE artifacts API. This is the most reliable path when the
    # agent copied final files to /workspace/hyperloom (SaFE exposes them as
    # hyperloom/ci_metrics.json and hyperloom/optimization_report.md). Retry a
    # few times because task terminal status can beat Claw's file index by
    # seconds.
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
        has_safe_metrics = any(p.endswith("ci_metrics.json") for p in wanted_paths)
        has_safe_report = any(p.endswith("optimization_report.md") for p in wanted_paths)
        if has_safe_metrics and has_safe_report:
            break
        if attempt < 2:
            log.info("[task %s] safe artifacts missing key files on attempt %d "
                     "(metrics=%s report=%s); retrying",
                     rec.task_id, attempt + 1, has_safe_metrics, has_safe_report)
            time.sleep(15)
    log.info("[task %s] safe artifacts: %d total, %d to download",
             rec.task_id, len(items), len(wanted))
    current_session_hints = _session_hints_from_artifact_items(items)
    if current_session_hints:
        log.info("[task %s] current session timestamp hints from artifacts: %s",
                 rec.task_id, ", ".join(sorted(current_session_hints)))

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

    # Stage 2: NFS fallback if we didn't get the key result files via Claw.
    has_metrics = any(p.endswith("ci_metrics.json") for p in rec.artifact_files)
    has_report = any(p.endswith("optimization_report.md") for p in rec.artifact_files)
    if not (has_metrics and has_report):
        log.info("[task %s] missing key files (metrics=%s report=%s) — trying NFS fallback",
                 rec.task_id, has_metrics, has_report)
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

    # Stage 3: reverse-backfill the audit fields directly into the wekafs
    # SOURCE files so operators ssh-ing into /wekafs/users/<uid>/<sess>/
    # see image/hyperloom_commit/category/sandbox_duration_seconds without
    # downloading the GHA artifact zip. No-op when wekafs isn't mounted.
    if rec.final_status == "Succeeded":
        try:
            n_wkfs = _backfill_wekafs_in_place(rec)
            if n_wkfs:
                log.info("[task %s] wekafs in-place backfill updated %d file(s)",
                         rec.task_id, n_wkfs)
        except Exception as e:
            log.warning("[task %s] wekafs in-place backfill skipped due to %s: %s",
                        rec.task_id, type(e).__name__, e)
    else:
        log.info("[task %s] wekafs in-place backfill skipped for final_status=%s",
                 rec.task_id, rec.final_status or "?")
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
    """Wait + collect for all submitted records, in parallel up to ``parallel``."""
    pending = [r for r in records if r.status == "submitted" and r.task_id]
    if not pending:
        log.info("no submitted tasks to wait for")
        return

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log.info("waiting for %d task(s) to finish (parallel=%d, timeout=%dm each)",
             len(pending), parallel, task_timeout_min)

    if parallel <= 1:
        for rec in pending:
            wait_and_collect_one(safe, rec, artifacts_dir,
                                 task_timeout_min, poll_s, collect, all_artifacts)
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


# ── Manifest ────────────────────────────────────────────────────────────────────

def write_manifest(
    out_dir: Path,
    records: list[SubmissionRecord],
    base_url: str,
    register_workspace: str,
    submit_workspace: str,
    volume: str,
) -> None:
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
        "| Pool | Model | Category | Image | Duration | Submit | Final | Phase | Task ID | Display Name | Artifacts | Note |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        # Final status only meaningful when --wait-for-completion was on.
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
        # Image cell: keep just the tag suffix (e.g. `sglang:v0.5.11-rocm720-mi30x`)
        # so the manifest is readable; full registry path is in JSON.
        image_cell = "-"
        image_full = (r.detected or {}).get("image", "") if r.detected else ""
        if image_full:
            image_cell = "`" + image_full.split("/")[-1] + "`"
        # Duration cell: rounded minutes for readability, ms-precision in JSON.
        duration_cell = "-"
        if r.sandbox_duration_seconds is not None:
            mins = r.sandbox_duration_seconds / 60.0
            duration_cell = f"{mins:.1f}m"
        category_cell = r.category or "-"
        md.append(
            f"| {pool_cell} | `{r.model}` | {category_cell} | {image_cell} | {duration_cell} | "
            f"{r.status} | {final or '-'} | {phase} | "
            f"`{r.task_id or '-'}` | {r.display_name or '-'} | {artifacts_cell} | {note} |"
        )
    (out_dir / "submission_manifest.md").write_text("\n".join(md) + "\n")
    log.info("manifest written to %s", out_dir)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8],
                        help="Override detected tensor parallel size")
    parser.add_argument("--concurrency", type=int,
                        help="Override detected concurrency")
    parser.add_argument("--image", help="Override container image")
    parser.add_argument("--isl", type=int, default=1024)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--mode", choices=["local", "claw"], default="local",
                        help="Execution mode passed to SaFE (default: local — "
                             "agent runs in sandbox directly; 'claw' routes via RayJob)")

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
                             f"SaFE backend default is MI355X — must override on core42.")
    parser.add_argument("--inferencex-path", default="",
                        help=f"InferenceX checkout path inside the sandbox "
                             f"(defaults to $SAFE_OPTIMIZE_INFERENCEX_PATH then "
                             f"'{DEFAULT_INFERENCEX_PATH}'). SaFE backend default is "
                             f"/hyperloom/InferenceX which doesn't exist on core42.")
    parser.add_argument("--oob-path", default="",
                        help=f"OOB checkout path inside the sandbox (defaults to "
                             f"$SAFE_OPTIMIZE_OOB_PATH then '{DEFAULT_OOB_PATH}').")
    parser.add_argument("--tracelens-root", default="",
                        help=f"TraceLens checkout path inside the sandbox (defaults "
                             f"to $SAFE_OPTIMIZE_TRACELENS_ROOT then "
                             f"'{DEFAULT_TRACELENS_ROOT}').")
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

    # Production-pool audit metadata. These fields do not affect submission;
    # they are copied into submission_manifest.json so we can answer "which
    # fixed leaderboard pool entry did this task rerun?" after the fact.
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
    return parser


def main() -> int:
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
    # --workspace is shorthand for "set both to the same value" (back-compat
    # with single-workspace usage from before the split). Explicit
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
    # Round-robin pool: --submit-workspaces overrides single submit_workspace.
    # Empty -> single workspace mode (back-compat with existing dispatches).
    submit_workspaces_raw = (args.submit_workspaces
                             or os.environ.get("SAFE_OPTIMIZE_SUBMIT_WORKSPACES")
                             or "")
    submit_workspaces_pool = [w.strip() for w in submit_workspaces_raw.split(",")
                              if w and w.strip()]
    volume = (args.volume
              or os.environ.get("SAFE_OPTIMIZE_VOLUME")
              or DEFAULT_VOLUME)
    gpu_type = (args.gpu_type
                or os.environ.get("SAFE_OPTIMIZE_GPU_TYPE")
                or DEFAULT_GPU_TYPE)
    inferencex_path = (args.inferencex_path
                       or os.environ.get("SAFE_OPTIMIZE_INFERENCEX_PATH")
                       or DEFAULT_INFERENCEX_PATH)
    oob_path = (args.oob_path
                or os.environ.get("SAFE_OPTIMIZE_OOB_PATH")
                or DEFAULT_OOB_PATH)
    tracelens_root = (args.tracelens_root
                      or os.environ.get("SAFE_OPTIMIZE_TRACELENS_ROOT")
                      or DEFAULT_TRACELENS_ROOT)
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
    log.info("Cluster prompt fields: gpu_type=%s inferencex_path=%s oob_path=%s tracelens_root=%s",
             gpu_type, inferencex_path, oob_path, tracelens_root)
    log.info("Kernel backends: %s", ", ".join(kernel_backends))
    if submit_workspaces_pool:
        log.info("submit round-robin pool: %s (overrides --submit-workspace)",
                 ",".join(submit_workspaces_pool))
    if register_workspace != submit_workspace and not submit_workspaces_pool:
        log.info("cross-workspace mode — needs SaFE selectLocalPath path-accessible "
                 "fallback to be deployed; will 400 on submit_task otherwise")

    hf = HuggingFaceClient(args.hf_token)
    # Dry-run never hits SaFE; pass an empty token so callers don't need a real one.
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

    records: list[SubmissionRecord] = []
    for repo in repos:
        log.info("=" * 60)
        log.info("Model: %s", repo)
        rec = process_model(
            repo, hf, safe, overrides,
            args.isl, args.osl, args.dry_run, args.hf_token,
            manual_mode=args.manual,
            mode=args.mode,
            gpu_type=gpu_type,
            inferencex_path=inferencex_path,
            oob_path=oob_path,
            tracelens_root=tracelens_root,
            prompt_prefix=args.prompt_prefix or None,
            prompt_suffix=args.prompt_suffix or None,
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

    # Wait + collect (default on). Skip on dry-run since nothing was submitted.
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

        # Per-status summary after completion.
        from collections import Counter
        final_counts = Counter(r.final_status or "Pending"
                               for r in records if r.task_id)
        log.info("=" * 60)
        log.info("Final task statuses: %s",
                 ", ".join(f"{k}={v}" for k, v in sorted(final_counts.items())))
        non_success = [r for r in records if r.task_id and r.final_status != "Succeeded"]
    else:
        non_success = []

    # Manifest is written *after* wait/collect so it captures final_status etc.
    if args.output_dir:
        write_manifest(Path(args.output_dir), records,
                       base_url, register_workspace, submit_workspace, volume)

    if args.dry_run:
        return 0
    if non_success:
        log.error("Non-success terminal statuses: %s",
                  ", ".join(f"{r.model}:{r.final_status or 'Pending'}" for r in non_success))
        return 2
    return 0 if submitted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
