# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared environment-variable safety helpers.

The helpers in this module are deliberately stdlib-only so they can be used by
CLI preflight, benchmark executors, and subprocess dispatchers without creating
package import cycles.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SERVER_ARGS_RE = re.compile(r"^EXTRA_[A-Z0-9_]+_ARGS$")

BLOCKED_UNTRUSTED_ENV_NAMES: frozenset[str] = frozenset(
    {
        "BASH_ENV",
        "CDPATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "ENV",
        "GCONV_PATH",
        "IFS",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PATH",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "RUBYOPT",
        "SHELLOPTS",
    }
)

BLOCKED_CHILD_ENV_NAMES: frozenset[str] = frozenset(
    {
        "BASH_ENV",
        "DYLD_INSERT_LIBRARIES",
        "ENV",
        "GCONV_PATH",
        "LD_AUDIT",
        "LD_PRELOAD",
        "PYTHONINSPECT",
        "PYTHONSTARTUP",
        "RUBYOPT",
        "SHELLOPTS",
    }
)

_SECRET_KEY_RE = re.compile(r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)(?:_|$)", re.IGNORECASE)
_NON_SECRET_TOKEN_ENV_NAMES: frozenset[str] = frozenset(
    {
        # SGLang tuning switch; TOKEN describes quantization granularity, not a credential.
        "SGLANG_USE_AITER_FP8_PER_TOKEN",
    }
)

WORKLOAD_ENV_EXACT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "CONC",
        "EP",
        "GPU_METRICS_CSV",
        "HIP_VISIBLE_DEVICES",
        "HSA_FORCE_FINE_GRAIN_PCIE",
        "ISL",
        "MAX_MODEL_LEN",
        "MODEL_PATH",
        "NUM_PROMPTS",
        "NUM_WARMUPS",
        "OSL",
        "PORT",
        "PROFILE",
        "PROFILE_EXTRA_BODY",
        "PROFILE_OSL",
        "RANDOM_RANGE_RATIO",
        "ROCR_VISIBLE_DEVICES",
        "RUN_EVAL",
        "SERVER_LOG",
        "TP",
    }
)

WORKLOAD_ENV_PREFIX_ALLOWLIST: tuple[str, ...] = (
    "AITER_",
    "ATOM_",
    "BENCH_",
    "CUDA_",
    "FLASHINFER_",
    "HIP_",
    "HSA_",
    "HYPERLOOM_PROFILE_",
    "MAGPIE_",
    "MORI_",
    "NCCL_",
    "PROFILE_",
    "PYTORCH_",
    "RCCL_",
    "ROCBLAS_",
    "ROCM_",
    "ROCR_",
    "SGLANG_",
    "TORCH_",
    "TRITON_",
    "VLLM_",
    "XDIT_",
)

DOTENV_EXACT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_MODEL",
        "CODEX_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "FRAMEWORK",
        "GEAK_CLAUDE_MODEL",
        "HIP_PATH",
        "HYPERLOOM_BENCHMARK_BACKEND",
        "HYPERLOOM_DOCKER_TARGET_HOST",
        "HYPERLOOM_FRAMEWORK_ENV",
        "HYPERLOOM_LLM_MODE",
        "HYPERLOOM_RUN_MODE",
        "HYPERLOOM_RUNTIME_DIR",
        "HYPERLOOM_SKILL_PATH",
        "HYPERLOOM_WHEEL_REPO",
        "HYPERLOOM_WHEEL_TAG",
        "INFERENCE_OPTIMIZER_FORCE_PYTHON",
        "KERNEL_AGENT_ENV",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ROCM_PATH",
        "SAFE_API_KEY",
        "SGLANG_ROCM_EXTRA",
        "SGLANG_ROCM_INDEX_URL",
        "SGLANG_ROCM_PYPI_VERSION",
        "USER_DATA_PATH",
        "VIRTUAL_ENV",
        "VLLM_PYTHON",
        "VLLM_VENV_ROOT",
    }
)

DOTENV_PREFIX_ALLOWLIST: tuple[str, ...] = (
    "AITER_",
    "GEAK_",
    "HYPERLOOM_",
    "INFERENCE_OPTIMIZER_",
    "SGLANG_",
    "VLLM_",
)

KERNEL_AGENT_ENV_EXACT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "GEAK_CLAUDE_BIN",
        "GEAK_CLAUDE_MODEL",
        "GEAK_E2E_RUNNER",
        "GEAK_MAX_BENCHMARK_SHAPES",
        "GEAK_ROOT",
        "GEAK_RUN_MODE",
        "GEAK_SCORE_TARGET",
        "GEAK_SKIP_PROFILE",
        "HYPERLOOM_KERNEL_AGENT_ROOT",
        "HYPERLOOM_ROOT",
        "HYPERLOOM_RUNTIME_DIR",
        "INFERENCEX_PATH",
        "KERNEL_AGENT_ENV",
        "KERNEL_AGENT_LOG_LEVEL",
        "KERNEL_AGENT_ROOT",
        "MAGPIE_PATH",
        "MAGPIE_PYTHON",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "TRACELENS_INTERNAL_ROOT",
        "TRACELENS_ROOT",
        "USER_DATA_PATH",
    }
)


def valid_env_key(key: object) -> bool:
    return bool(_ENV_KEY_RE.fullmatch(str(key or "")))


def is_blocked_untrusted_env_key(key: object) -> bool:
    name = str(key or "").strip()
    upper = name.upper()
    return (
        not valid_env_key(name)
        or upper in BLOCKED_UNTRUSTED_ENV_NAMES
        or (upper not in _NON_SECRET_TOKEN_ENV_NAMES and bool(_SECRET_KEY_RE.search(upper)))
    )


def is_allowed_workload_env_key(key: object) -> bool:
    name = str(key or "").strip()
    upper = name.upper()
    if is_blocked_untrusted_env_key(upper):
        return False
    return (
        upper in WORKLOAD_ENV_EXACT_ALLOWLIST
        or _SERVER_ARGS_RE.fullmatch(upper) is not None
        or any(upper.startswith(prefix) for prefix in WORKLOAD_ENV_PREFIX_ALLOWLIST)
    )


def is_allowed_dotenv_key(key: object) -> bool:
    name = str(key or "").strip()
    upper = name.upper()
    if not valid_env_key(upper) or upper in BLOCKED_UNTRUSTED_ENV_NAMES:
        return False
    return upper in DOTENV_EXACT_ALLOWLIST or any(
        upper.startswith(prefix) for prefix in DOTENV_PREFIX_ALLOWLIST
    )


def is_allowed_kernel_agent_env_key(key: object) -> bool:
    name = str(key or "").strip()
    upper = name.upper()
    return valid_env_key(upper) and upper in KERNEL_AGENT_ENV_EXACT_ALLOWLIST


def filter_untrusted_env_mapping(
    envs: Mapping[str, object] | None,
    *,
    allow_predicate,
) -> tuple[dict[str, str], dict[str, str]]:
    """Filter env overrides from state, LLM output, or shared env files.

    Returns ``(allowed, dropped)`` where ``dropped`` maps key to a short reason.
    """
    allowed: dict[str, str] = {}
    dropped: dict[str, str] = {}
    for key, value in (envs or {}).items():
        name = str(key or "").strip()
        if not valid_env_key(name):
            dropped[name or "<empty>"] = "invalid_env_key"
            continue
        upper = name.upper()
        if not allow_predicate(upper):
            dropped[name] = "not_allowed"
            continue
        allowed[upper] = str(value)
    return allowed, dropped


def scrub_child_process_env(env: dict[str, str]) -> dict[str, str]:
    """Remove startup/preload hooks from a child-process environment in place."""
    for name in BLOCKED_CHILD_ENV_NAMES:
        env.pop(name, None)
    return env


__all__ = [
    "BLOCKED_CHILD_ENV_NAMES",
    "BLOCKED_UNTRUSTED_ENV_NAMES",
    "filter_untrusted_env_mapping",
    "is_allowed_dotenv_key",
    "is_allowed_kernel_agent_env_key",
    "is_allowed_workload_env_key",
    "is_blocked_untrusted_env_key",
    "scrub_child_process_env",
    "valid_env_key",
]
