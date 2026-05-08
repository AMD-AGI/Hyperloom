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
import sys
import time
from dataclasses import asdict, dataclass, field
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
# InferenceX checkout actually lives on /wekafs). Without overriding here the
# generated prompt sends the agent on a 5-10 min wild goose chase looking for
# /hyperloom/InferenceX, and it picks GPU-architecture-wrong heuristics later.
DEFAULT_GPU_TYPE = "MI300X"
DEFAULT_INFERENCEX_PATH = "/wekafs/InferenceX"

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
    return f"{_proxy()}/lmsysorg/sglang:v0.5.10-rocm720-mi30x"


def _default_vllm_image() -> str:
    return f"{_proxy()}/vllm/vllm-openai-rocm:v0.18.0"


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

            # Belt-and-suspenders: also validate architectures[0] looks generative.
            # config.json is a separate fetch; do it last so we don't waste it on
            # candidates already rejected above.
            try:
                cfg = self.model_config(repo)
            except Exception:
                # Some repos hide config.json behind gating even when info is public —
                # don't kill the candidate over it.
                cfg = {}
            arch = (cfg.get("architectures") or [""])[0]
            if arch and not is_generative_arch(arch):
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
    ):
        self.base_url = base_url.rstrip("/")
        # Where the model gets registered + downloaded (must allow RW writes
        # to the configured volume).
        self.register_workspace = register_workspace
        # Where the optimization task is created (must allow Sandbox scope).
        # Can equal register_workspace when both constraints are satisfied
        # by a single workspace (rare in practice on core42).
        self.submit_workspace = submit_workspace
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

    def register_model(self, repo_id: str, hf_token: str = "") -> str:
        body = {
            "source": {
                "url": repo_id,
                "accessMode": "local",
                **({"token": hf_token} if hf_token else {}),
            },
            "workspace": self.register_workspace,
            "target": {"volume": self.volume},
        }
        log.info("[%s] register: workspace=%s volume=%s",
                 repo_id, self.register_workspace, self.volume)
        result = self._request("POST", "api/v1/playground/models", body)
        return result.get("id", "")

    def wait_ready(
        self, model_id: str, timeout_min: int = 120, poll_s: int = 30,
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
    ) -> dict:
        body = {
            "displayName": display_name,
            "modelId": model_id,
            "workspace": self.submit_workspace,
            "mode": mode,
            "framework": framework,
            "precision": precision,
            "tp": tp,
            "ep": 1,
            "isl": isl,
            "osl": osl,
            "concurrency": concurrency,
            "kernelBackends": ["Claude Code"],
        }
        if image:
            body["image"] = image
        # Override SaFE backend's wrong-for-core42 defaults (MI355X /
        # /hyperloom/InferenceX). See DEFAULT_GPU_TYPE/_INFERENCEX_PATH above.
        if gpu_type:
            body["gpuType"] = gpu_type
        if inferencex_path:
            body["inferencexPath"] = inferencex_path
        return self._request("POST", "api/v1/optimization/tasks", body)

    # ── Task lifecycle ──

    # Lifecycle states observed in SaFE (see types.go OptimizationTaskStatus).
    TERMINAL_TASK_STATUSES = {"Succeeded", "Failed", "Interrupted"}

    def get_task(self, task_id: str) -> dict:
        return self._request("GET", f"api/v1/optimization/tasks/{task_id}")

    def wait_task_done(
        self, task_id: str, timeout_min: int = 480, poll_s: int = 60,
    ) -> tuple[str, dict]:
        """Poll task until it reaches a terminal status. Returns (status, last_task_dict).

        Returns ('Timeout', {}) if the deadline elapses while still Pending/Running.
        Transient HTTP errors are logged and retried — they don't abort the wait.
        """
        log.info("[task %s] waiting for completion (timeout=%dm, poll=%ds)",
                 task_id, timeout_min, poll_s)
        deadline = time.time() + timeout_min * 60
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

    def list_artifacts(self, task_id: str) -> list[dict]:
        try:
            data = self._request("GET", f"api/v1/optimization/tasks/{task_id}/artifacts")
        except Exception as e:
            log.warning("[task %s] list_artifacts failed: %s", task_id, e)
            return []
        return data.get("items", [])

    def download_artifact(self, task_id: str, path: str) -> bytes:
        url = (f"{self.base_url}/api/v1/optimization/tasks/{task_id}/artifacts/download"
               f"?path={requests.utils.quote(path, safe='')}")
        resp = self._sess.get(url, timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"download {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.content

    def download_artifact_to(self, task_id: str, path: str, local_path: str) -> int:
        data = self.download_artifact(task_id, path)
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
    display_name: str | None = None
    detected: dict | None = None
    overrides: dict = field(default_factory=dict)
    error: str | None = None
    # Filled in by process_completion when --wait-for-completion is on.
    final_status: str | None = None    # SaFE: Succeeded/Failed/Interrupted/Timeout
    final_phase: int | None = None     # currentPhase at terminal moment
    final_message: str | None = None   # task.Message
    artifacts_dir: str | None = None   # local dir where artifacts landed
    artifact_count: int = 0
    artifact_files: list[str] = field(default_factory=list)


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
) -> SubmissionRecord:
    rec = SubmissionRecord(
        model=repo_id,
        overrides={k: v for k, v in overrides.items() if v is not None},
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

    if dry_run:
        rec.status = "dry-run"
        return rec

    safe_model = safe.find_model(repo_id)
    if safe_model:
        model_id = safe_model["id"]
        phase = safe_model.get("phase", "")
        log.info("[%s] found in SaFE: id=%s phase=%s", repo_id, model_id, phase)
        if phase != "Ready" and not safe.wait_ready(model_id):
            rec.status = "failed"
            rec.error = "model never reached Ready"
            return rec
    else:
        try:
            model_id = safe.register_model(repo_id, hf_token)
        except Exception as e:
            rec.status = "failed"
            rec.error = f"register: {e}"
            return rec
        if not model_id:
            rec.status = "failed"
            rec.error = "register returned empty id"
            return rec
        if not safe.wait_ready(model_id):
            rec.status = "failed"
            rec.error = "model never reached Ready"
            return rec

    try:
        result = safe.submit_task(
            model_id, display_name, framework, precision, tp, conc, isl, osl, image,
            mode=mode, gpu_type=gpu_type, inferencex_path=inferencex_path)
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


def wait_and_collect_one(
    safe: SafeOptimizeClient,
    rec: SubmissionRecord,
    artifacts_dir: Path,
    task_timeout_min: int,
    poll_s: int,
    collect: bool,
    all_artifacts: bool,
) -> SubmissionRecord:
    """Wait for one task to finish, then optionally download its artifacts."""
    if not rec.task_id:
        return rec  # nothing to wait for (skipped/failed during submit)

    final_status, last_task = safe.wait_task_done(
        rec.task_id, timeout_min=task_timeout_min, poll_s=poll_s)
    rec.final_status = final_status
    rec.final_phase = last_task.get("currentPhase")
    rec.final_message = (last_task.get("message") or "")[:500] or None

    if not collect:
        return rec

    items = safe.list_artifacts(rec.task_id)
    wanted = [it for it in items if _is_wanted_artifact(it.get("path", ""), all_artifacts)]
    log.info("[task %s] artifacts: %d total, %d to download",
             rec.task_id, len(items), len(wanted))

    task_dir = artifacts_dir / rec.task_id
    rec.artifacts_dir = str(task_dir)
    for it in wanted:
        path = it.get("path", "")
        if not path:
            continue
        local = _safe_local_path(artifacts_dir, rec.task_id, path)
        try:
            n = safe.download_artifact_to(rec.task_id, path, str(local))
            rec.artifact_files.append(str(local))
            rec.artifact_count += 1
            log.info("[task %s] saved %s (%d bytes)", rec.task_id, path, n)
        except Exception as e:
            log.warning("[task %s] failed to download %s: %s", rec.task_id, path, e)
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
        "| Model | Submit | Final | Phase | Task ID | Display Name | Artifacts | Note |",
        "|---|---|---|---|---|---|---|---|",
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
        md.append(
            f"| `{r.model}` | {r.status} | {final or '-'} | {phase} | "
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
                             f"then '{DEFAULT_SUBMIT_WORKSPACE}')")
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
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                        help="HuggingFace token (or set $HF_TOKEN)")

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
    parser.add_argument("--task-timeout-min", type=int, default=480,
                        help="Per-task wait timeout in minutes (default: 480 = 8h)")
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
    volume = (args.volume
              or os.environ.get("SAFE_OPTIMIZE_VOLUME")
              or DEFAULT_VOLUME)
    gpu_type = (args.gpu_type
                or os.environ.get("SAFE_OPTIMIZE_GPU_TYPE")
                or DEFAULT_GPU_TYPE)
    inferencex_path = (args.inferencex_path
                       or os.environ.get("SAFE_OPTIMIZE_INFERENCEX_PATH")
                       or DEFAULT_INFERENCEX_PATH)

    if not api_key and not args.dry_run:
        log.error("no API key set (CLAW_API_KEY / SAFE_API_KEY / --api-key)")
        return 2

    log.info("SaFE base_url=%s register_workspace=%s submit_workspace=%s volume=%s",
             base_url, register_workspace, submit_workspace, volume)
    log.info("Cluster prompt fields: gpu_type=%s inferencex_path=%s",
             gpu_type, inferencex_path)
    if register_workspace != submit_workspace:
        log.info("cross-workspace mode — needs SaFE selectLocalPath path-accessible "
                 "fallback to be deployed; will 400 on submit_task otherwise")

    hf = HuggingFaceClient(args.hf_token)
    # Dry-run never hits SaFE; pass an empty token so callers don't need a real one.
    safe = SafeOptimizeClient(
        base_url, api_key or "dry-run",
        register_workspace=register_workspace,
        submit_workspace=submit_workspace,
        volume=volume,
    )

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

    # Manifest is written *after* wait/collect so it captures final_status etc.
    if args.output_dir:
        write_manifest(Path(args.output_dir), records,
                       base_url, register_workspace, submit_workspace, volume)

    if args.dry_run:
        return 0
    # Non-zero only when nothing was submitted at all (per user request:
    # "完成就算成功" — partial task failures don't fail the workflow).
    return 0 if submitted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
