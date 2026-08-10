# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared environment-variable safety helpers.

The helpers in this module are deliberately stdlib-only so they can be used by
CLI preflight, benchmark executors, and subprocess dispatchers without creating
package import cycles.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_PYTHON_PACKAGE_ROOT_BASENAMES: frozenset[str] = frozenset({"site-packages", "dist-packages"})

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

BENCHMARK_SECRET_ENV_NAMES: frozenset[str] = frozenset(
    {
        "AMD_API_KEY",
        "AMD_LLM_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_CUSTOM_HEADERS",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAW_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEAK_API_KEY",
        "LLM_API_KEY",
        "LLM_GATEWAY_KEY",
        "LLM_PROXY_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_CUSTOM_HEADERS",
        # Legacy: not consumed anymore, still scrubbed if present.
        "SAFE_API_KEY",
    }
)

_SECRET_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-_.=]{8,}"), r"\1[REDACTED]"),
    (re.compile(r"\b((?:ak|sk|pk)-(?:lf-)?)[A-Za-z0-9\-_]{6,}"), r"\1[REDACTED]"),
    (re.compile(r"\b(gh[pousr]_|github_pat_)[A-Za-z0-9_]{10,}"), r"\1[REDACTED]"),
    (
        re.compile(
            r"(?i)\b([A-Z0-9_]*"
            r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)"
            r"[A-Z0-9_]*\s*[=:]\s*)"
            r"[^\s,;'\"]+"
        ),
        r"\1[REDACTED]",
    ),
)

DOTENV_EXACT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        # Gateway auth header. Setup writes this one into .env (the AMD APIM
        # subscription key), so the loader has to read it back or a
        # header-authenticated gateway silently loses its credential whenever the
        # shell has not exported it already. Its OpenAI-side counterpart below is
        # operator-written only.
        "ANTHROPIC_CUSTOM_HEADERS",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_MODEL",
        "CODEX_MODEL",
        # Retired provider variables, still readable so a pre-migration .env can
        # be normalized by hyperloom.common.llm_config.deepseek_compat_env.
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "FORGE_PATH",
        "FRAMEWORK",
        "GEAK_CLAUDE_MODEL",
        "HIP_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
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
        "KERNEL_AGENT_ROOT",
        "KERNEL_OPT_BACKEND_ORDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_CUSTOM_HEADERS",
        "NO_PROXY",
        "ROCM_PATH",
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
    "FORGE_",
    "GEAK_",
    "HF_",
    "HYPERLOOM_",
    "INFERENCE_OPTIMIZER_",
    "KERNEL_OPT_",
    "SGLANG_",
    "VLLM_",
)

KERNEL_AGENT_ENV_EXACT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "FORGE_PATH",
        "GEAK_CLAUDE_BIN",
        "GEAK_CLAUDE_MODEL",
        "GEAK_E2E_RUNNER",
        "GEAK_MAX_BENCHMARK_SHAPES",
        "GEAK_ROOT",
        "GEAK_RUN_MODE",
        "GEAK_SCORE_TARGET",
        "GEAK_SKIP_PROFILE",
        "HYPERLOOM_FORGE_REWRITE_BY_FLYDSL",
        "HYPERLOOM_KERNEL_AGENT_ROOT",
        "HYPERLOOM_ROOT",
        "HYPERLOOM_RUNTIME_DIR",
        "HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV",
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        "INFERENCEX_PATH",
        "KERNEL_AGENT_ENV",
        "KERNEL_AGENT_LOG_LEVEL",
        "KERNEL_AGENT_ROOT",
        "KERNEL_OPT_BACKEND_ORDER",
        "MAGPIE_PATH",
        "MAGPIE_PYTHON",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "TRACELENS_INTERNAL_ROOT",
        "TRACELENS_ROOT",
        "USER_DATA_PATH",
    }
)


def is_python_package_root(path: object) -> bool:
    """True when ``path`` is a ``site-packages``/``dist-packages`` dir.

    Already on the import path, so keeping it off PYTHONPATH avoids shadowing an
    isolated venv's packages; a source checkout root returns False.
    """
    name = str(path or "").strip().rstrip("/")
    if not name:
        return False
    return name.rsplit("/", 1)[-1] in _PYTHON_PACKAGE_ROOT_BASENAMES


def valid_env_key(key: object) -> bool:
    return bool(_ENV_KEY_RE.fullmatch(str(key or "")))


def is_allowed_dotenv_key(key: object) -> bool:
    name = str(key or "").strip()
    upper = name.upper()
    if not valid_env_key(upper) or upper in BLOCKED_UNTRUSTED_ENV_NAMES:
        return False
    return upper in DOTENV_EXACT_ALLOWLIST or any(upper.startswith(prefix) for prefix in DOTENV_PREFIX_ALLOWLIST)


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
        allowed[name] = str(value)
    return allowed, dropped


def scrub_child_process_env(env: dict[str, str]) -> dict[str, str]:
    """Remove startup/preload hooks from a child-process environment in place."""
    for name in BLOCKED_CHILD_ENV_NAMES:
        env.pop(name, None)
    return env


def scrub_benchmark_process_env(env: dict[str, str]) -> dict[str, str]:
    """Remove control-plane credentials from a benchmark environment in place.

    Serving benchmarks target the local model server and do not need LLM-agent
    credentials. Keeping those values out of the process tree also prevents
    shell tracing and lm-eval command/result serialization from persisting them.
    """
    scrub_child_process_env(env)
    for name in BENCHMARK_SECRET_ENV_NAMES:
        env.pop(name, None)
    return env


def filter_benchmark_env_mapping(envs: Mapping[str, object] | None) -> dict[str, str]:
    """Return benchmark env overrides without control-plane credentials."""
    return {
        str(name): str(value)
        for name, value in (envs or {}).items()
        if str(name).strip().upper() not in BENCHMARK_SECRET_ENV_NAMES
    }


def redact_secret_values(text: str) -> str:
    """Mask recognizable credential values before diagnostic text is persisted."""
    out = str(text or "")
    for pattern, replacement in _SECRET_REDACTION_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


__all__ = [
    "BENCHMARK_SECRET_ENV_NAMES",
    "BLOCKED_CHILD_ENV_NAMES",
    "BLOCKED_UNTRUSTED_ENV_NAMES",
    "filter_benchmark_env_mapping",
    "filter_untrusted_env_mapping",
    "is_allowed_dotenv_key",
    "is_allowed_kernel_agent_env_key",
    "is_python_package_root",
    "redact_secret_values",
    "scrub_benchmark_process_env",
    "scrub_child_process_env",
    "valid_env_key",
]
