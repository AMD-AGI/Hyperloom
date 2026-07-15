"""Validate env keys forwarded over SSH to multi-node inference pods."""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Exact-match denylist of the only env keys unsafe to forward into a shell-launched
# pod process: dynamic-loader hijack (LD_*), python import hijack (PYTHON*), binary
# hijack (PATH), and shell-startup / field-splitting injection (BASH_ENV/ENV/IFS).
# Everything else (tuning knobs like NCCL_*/MC_*/SGLANG_*/PYTORCH_CUDA_ALLOC_CONF,
# etc.) is forwarded verbatim. No prefix/substring matching.
_DENY_KEYS: frozenset[str] = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "PYTHONPATH",
        "PYTHONHOME",
        "PATH",
        "BASH_ENV",
        "ENV",
        "IFS",
    }
)


def is_valid_env_key(key: str) -> bool:
    """Return True when ``key`` is a safe POSIX-style env var name.

    Args:
        key: Candidate environment variable name.

    Returns:
        bool: True when the name matches the allowed character set.
    """
    return bool(_ENV_KEY_RE.match(key))


def is_forward_env_key_allowed(key: str) -> bool:
    """Return True when ``key`` may be forwarded over SSH to pod processes.

    Default-allow: any POSIX-shaped key is forwarded unless it is an exact
    member of :data:`_DENY_KEYS` (the loader / python / PATH / shell injection
    vectors). No prefix or substring matching, so tuning knobs are never dropped.

    Args:
        key: Environment variable name to evaluate.

    Returns:
        bool: True when the key is allowed for SSH forwarding.
    """
    if not is_valid_env_key(key):
        return False
    return key not in _DENY_KEYS


def filter_forward_env(
    env: dict[str, str],
    *,
    warn_on_drop: bool = True,
) -> dict[str, str]:
    """Drop disallowed keys from an env dict destined for SSH forwarding.

    Args:
        env: Raw key/value pairs to filter.
        warn_on_drop: When True, log a warning for each dropped key.

    Returns:
        dict[str, str]: The filtered env mapping.
    """
    out: dict[str, str] = {}
    for raw_key, raw_val in env.items():
        key = str(raw_key)
        if is_forward_env_key_allowed(key):
            out[key] = str(raw_val)
        elif warn_on_drop:
            log.warning("dropping disallowed multi-node forward env key %r", key)
    return out


def assert_env_key_shapes(env: dict[str, str]) -> None:
    """Raise ValueError when any env key is not a valid POSIX identifier.

    Used for credential-bearing SSH stdin scripts where values are quoted and
    only key-shape injection (F002.1) must be blocked.

    Args:
        env: Key/value pairs about to be injected into a remote shell.

    Raises:
        ValueError: When one or more keys fail :func:`is_valid_env_key`.
    """
    bad = [str(k) for k in env if not is_valid_env_key(str(k))]
    if bad:
        raise ValueError(f"invalid SSH env key names: {bad!r}")


def assert_forward_env_keys(env: dict[str, str]) -> None:
    """Raise ValueError when any env key is not allowed for SSH forwarding.

    Args:
        env: Key/value pairs about to be injected into a remote shell.

    Raises:
        ValueError: When one or more keys fail :func:`is_forward_env_key_allowed`.
    """
    bad = [str(k) for k in env if not is_forward_env_key_allowed(str(k))]
    if bad:
        raise ValueError(f"disallowed SSH forward env keys: {bad!r}")
