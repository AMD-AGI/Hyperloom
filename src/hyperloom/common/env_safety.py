# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared environment-variable safety helpers.

The helpers in this module are deliberately stdlib-only so they can be used by
CLI preflight, benchmark executors, and subprocess dispatchers without creating
package import cycles.
"""

from __future__ import annotations

from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES as _GPU_MASK_ENV_NAMES

import os
import re
from collections.abc import Mapping

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_PYTHON_PACKAGE_ROOT_BASENAMES: frozenset[str] = frozenset({"site-packages", "dist-packages"})

BLOCKED_UNTRUSTED_ENV_NAMES: frozenset[str] = frozenset(
    {
        "BASH_ENV",
        "CDPATH",
        "ENV",
        "GCONV_PATH",
        "GIT_SSH_COMMAND",
        "IFS",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NODE_OPTIONS",
        "PATH",
        "PERL5OPT",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        # site.py adds its site-packages to sys.path, so it loads arbitrary code.
        "PYTHONUSERBASE",
        "SHELLOPTS",
    }
)

BLOCKED_CHILD_ENV_NAMES: frozenset[str] = frozenset(
    {
        "BASH_ENV",
        "ENV",
        "GCONV_PATH",
        "LD_AUDIT",
        "LD_PRELOAD",
        "PYTHONINSPECT",
        "PYTHONSTARTUP",
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
        # ANTHROPIC_AUTH_TOKEN is deliberately absent: no installer ever writes
        # it (they persist the API-key spelling and actively remove this one),
        # so anything read back here is a hand-written leftover -- and since it
        # outranks a subscription token, reading it would silently move the run
        # onto API billing.
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
        # install.sh persists this next to the Anthropic URL and key, so the
        # reader has to accept it or a header-authenticated gateway loses its
        # credential on the way back in.
        "ANTHROPIC_CUSTOM_HEADERS",
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
        # install.sh never writes the OpenAI side itself, but the URL and key are
        # already accepted from an operator-supplied env file; the header that
        # authenticates the same endpoint is read on the same terms.
        "OPENAI_CUSTOM_HEADERS",
        "TRACELENS_INTERNAL_ROOT",
        "TRACELENS_ROOT",
        "USER_DATA_PATH",
    }
)

# GPU visibility masks: setting one selects the hardware rather than tuning it.
# Single definition in ``hyperloom.common.visible_devices``.
GPU_MASK_ENV_NAMES = _GPU_MASK_ENV_NAMES

# Env names an untrusted external source (reference recipe, framework-switch
# manifest) may never set: shell-unsafe vars plus the workload/benchmark keys the
# optimizer's CLI flags own — setting one retargets the benchmark instead of
# toggling a knob.
BLOCKED_EXTERNAL_ENV_NAMES: frozenset[str] = (
    BLOCKED_UNTRUSTED_ENV_NAMES
    | BENCHMARK_SECRET_ENV_NAMES
    | GPU_MASK_ENV_NAMES
    | frozenset(
        {
            "HOME",
            "MODEL",
            "MODEL_PATH",
            "TP",
            "EP",
            "CONC",
            "ISL",
            "OSL",
            "MAX_MODEL_LEN",
            "PRECISION",
            "PORT",
            "NUM_PROMPTS",
            "NUM_WARMUPS",
            "RANDOM_RANGE_RATIO",
            "RUN_EVAL",
            "PROFILE",
            "RESULT_DIR",
            "RESULT_FILENAME",
            # Reroute traffic, model downloads or TLS trust. Kept out of
            # BLOCKED_UNTRUSTED_ENV_NAMES because a local .env may set the proxies.
            "CURL_CA_BUNDLE",
            "HF_ENDPOINT",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "REQUESTS_CA_BUNDLE",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TMPDIR",
        }
    )
)

# Env names a per-variant override may never set. Workload pins stay allowed:
# the sweep and shape-capture grids set them from code.
BLOCKED_VARIANT_ENV_NAMES: frozenset[str] = BLOCKED_UNTRUSTED_ENV_NAMES | BENCHMARK_SECRET_ENV_NAMES

# Credential-shaped name fragments, so an unlisted secret cannot be persisted
# into a session YAML by name alone.
_SECRET_NAME_FRAGMENTS: tuple[str, ...] = ("APIKEY", "API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

# Masked out before matching, not exempted whole, so TOKENIZER_API_KEY still reads
# as a credential while TOKENIZERS_PARALLELISM does not.
_SECRET_FRAGMENT_EXEMPTIONS: tuple[str, ...] = ("TOKENIZER",)


def is_secret_shaped_env_name(key: object) -> bool:
    """True when a name looks like a credential rather than a tuning knob."""
    upper = str(key or "").strip().upper()
    for exempt in _SECRET_FRAGMENT_EXEMPTIONS:
        upper = upper.replace(exempt, "")
    return any(fragment in upper for fragment in _SECRET_NAME_FRAGMENTS)


def is_allowed_external_env_key(key: object) -> bool:
    """True when an env export from an untrusted external source is safe to carry."""
    upper = str(key or "").strip().upper()
    if not valid_env_key(upper) or upper in BLOCKED_EXTERNAL_ENV_NAMES:
        return False
    return not is_secret_shaped_env_name(upper)


def is_allowed_variant_env_key(key: object) -> bool:
    """True when a per-variant env override is safe to hand a benchmark subprocess."""
    upper = str(key or "").strip().upper()
    return valid_env_key(upper) and upper not in BLOCKED_VARIANT_ENV_NAMES


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


def build_benchmark_env(*layers: Mapping[str, object] | None) -> dict[str, str]:
    """Build a benchmark subprocess env: parent env under each layer, later winning.

    Keys are upper-cased and values stringified for the YAML's plain scalars.
    """
    env = os.environ.copy()
    for layer in layers:
        env.update({str(key).upper(): str(value) for key, value in (layer or {}).items()})
    return scrub_benchmark_process_env(env)


def redact_secret_values(text: str) -> str:
    """Mask recognizable credential values before diagnostic text is persisted."""
    out = str(text or "")
    for pattern, replacement in _SECRET_REDACTION_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


__all__ = [
    "BENCHMARK_SECRET_ENV_NAMES",
    "BLOCKED_CHILD_ENV_NAMES",
    "BLOCKED_EXTERNAL_ENV_NAMES",
    "BLOCKED_UNTRUSTED_ENV_NAMES",
    "BLOCKED_VARIANT_ENV_NAMES",
    "GPU_MASK_ENV_NAMES",
    "build_benchmark_env",
    "filter_untrusted_env_mapping",
    "is_allowed_dotenv_key",
    "is_allowed_external_env_key",
    "is_allowed_kernel_agent_env_key",
    "is_allowed_variant_env_key",
    "is_python_package_root",
    "is_secret_shaped_env_name",
    "redact_secret_values",
    "scrub_benchmark_process_env",
    "scrub_child_process_env",
    "valid_env_key",
]
