"""Validate env keys forwarded over SSH to multi-node inference pods."""

from __future__ import annotations

import logging
import re

from hyperloom.common.env_safety import BLOCKED_UNTRUSTED_ENV_NAMES

log = logging.getLogger(__name__)

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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

    Default-allow: any POSIX-shaped key is forwarded unless it is a member of
    :data:`hyperloom.common.env_safety.BLOCKED_UNTRUSTED_ENV_NAMES`.  No prefix
    or substring matching, so tuning knobs are never dropped.

    Args:
        key: Environment variable name to evaluate.

    Returns:
        bool: True when the key is allowed for SSH forwarding.
    """
    if not is_valid_env_key(key):
        return False
    return key not in BLOCKED_UNTRUSTED_ENV_NAMES


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
