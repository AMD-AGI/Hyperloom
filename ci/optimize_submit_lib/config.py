from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

log = logging.getLogger("optimize-submit")

# ── Defaults ────────────────────────────────────────────────────────────────────

DEFAULT_API_URL = ""
# Live SaFE/Claw submits are deployment-specific. Public defaults stay empty so
# forks must opt in with their own service URL, workspaces, and shared volume.
DEFAULT_REGISTER_WORKSPACE = ""
DEFAULT_SUBMIT_WORKSPACE = ""
DEFAULT_VOLUME = ""
# Default to MI300X policy for AMD public CI examples. Tool source paths are
# intentionally not pinned so install.sh can clone writable copies.
DEFAULT_GPU_TYPE = "MI300X"
DEFAULT_GPU_PROFILE = "mi300x"
DEFAULT_KERNEL_BACKENDS = ["GEAK"]
DEFAULT_MAX_HOURS = 6.0
DEFAULT_TARGET_GAIN = 500.0
DEFAULT_RESULTS_PATH = "$RESULT_DIR"
DEFAULT_CONTEXT_RESERVE_TOKENS = 16
# Models whose config.json max_position_embeddings is at or below this are
# skipped outright: their context window is too small to be worth a sandbox slot.
MIN_MAX_POSITION_EMBEDDINGS = 2048

# Hardware facts from AMD Instinct datasheets. tp_thresholds_b is CI policy
# scaled by per-GPU HBM capacity.
GPU_PROFILES = {
    "mi300x": {
        "gpu_type": "MI300X",
        "llvm_target": "gfx942",
        "hbm_gb": 192,
        "hbm_bandwidth_tb_s": 5.3,
        "tp_thresholds_b": (80, 128, 256),
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
    """Return the GPU profile key when ``gpu_type`` maps to a known CI profile.

    Args:
        gpu_type (str | None): Free-form GPU type/alias.
        warn (bool): Log a warning when a non-empty ``gpu_type`` is unknown.

    Returns:
        str | None: The matching :data:`GPU_PROFILES` key, or ``None`` when no
        profile matches.
    """
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
        log.warning(
            "unknown gpu_type=%r; using %s for TP policy only", raw, GPU_PROFILES[DEFAULT_GPU_PROFILE]["gpu_type"]
        )
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


# Canonical prompt prefix file (ci/prompt_prefix.txt), used as the argparse default.
_PROMPT_PREFIX_FILE = Path(__file__).resolve().parent / "prompt_prefix.txt"


def _load_default_prompt_prefix() -> str:
    """Resolve the default ``--prompt-prefix``: $SAFE_OPTIMIZE_PROMPT_PREFIX,
    else ci/prompt_prefix.txt, else empty string.

    Returns:
        str: The resolved prompt prefix, or ``""`` when no source supplies one.
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


# Optional RDMA bnxt_re tar package. Deployments that require it should set the
# env var; public defaults do not expose private storage paths.
_MULTINODE_BNXT_TAR = os.environ.get("SAFE_OPTIMIZE_BNXT_TAR", "").strip()


def _multinode_prompt_suffix(nodes: int, rayjob_image: str) -> str:
    """RayJob topology block injected into the prompt when nodes > 1.

    Returns "" for single-node.

    Args:
        nodes (int): Node count for the run; ``<= 1`` produces no suffix.
        rayjob_image (str): Container image to advertise for the RayJob.

    Returns:
        str: The multi-node topology prompt block, or ``""`` for single-node.
    """
    if nodes <= 1:
        return ""
    lines = [
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
    ]
    if _MULTINODE_BNXT_TAR:
        lines.append(f"- PATH_TO_BNXT_TAR_PACKAGE={_MULTINODE_BNXT_TAR}\n")
    return "".join(lines)


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
            raise ValueError(f"unknown kernel backend {item!r}; expected one of geak, claude, codex, cursor")
        if normalized not in out:
            out.append(normalized)
    return out or list(DEFAULT_KERNEL_BACKENDS)


# Architectures well-supported by SGLang on ROCm 7.x.
SGLANG_ARCHS: set[str] = {
    "LlamaForCausalLM",
    "LlamaForCausalLMWithVisualEncoder",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3MoeForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "InternLM2ForCausalLM",
    "InternLM3ForCausalLM",
    "Phi3ForCausalLM",
    "PhiForCausalLM",
    "GPTBigCodeForCausalLM",
    "FalconForCausalLM",
    "ChatGLMModel",
    # Architectures natively supported by sglang v0.5.12 (transformers 5.x).
    "Gemma4ForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Mistral3ForConditionalGeneration",
    "NemotronHForCausalLM",
    "Glm4ForCausalLM",
    "Glm4MoeForCausalLM",
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
# classifier models.
GENERATIVE_ARCH_SUFFIXES: tuple[str, ...] = (
    "ForCausalLM",
    "ForConditionalGeneration",
    "LMHeadModel",
    "ForSeq2SeqLM",
)


def is_generative_arch(arch: str) -> bool:
    """True if the HF arch is causal-LM-style inference; False for empty/unknown
    (better to skip than waste a sandbox slot).

    Args:
        arch (str): HF ``architectures[0]`` class name.

    Returns:
        bool: True when ``arch`` ends with a known generative suffix.
    """
    if not arch:
        return False
    return any(arch.endswith(s) for s in GENERATIVE_ARCH_SUFFIXES)


def _default_sglang_image() -> str:
    """Return the default SGLang server image.

    Returns:
        The configured SGLang image, or a public image placeholder suitable
        for dry runs.
    """
    return os.environ.get("SAFE_OPTIMIZE_SGLANG_IMAGE", "lmsysorg/sglang:latest")


def _default_vllm_image() -> str:
    """Return the default vLLM server image.

    Returns:
        The configured vLLM image, or a public image placeholder suitable
        for dry runs.
    """
    return os.environ.get("SAFE_OPTIMIZE_VLLM_IMAGE", "vllm/vllm-openai:latest")


# Shared env parser used by the SaFE client and artifact fallback logic.
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
