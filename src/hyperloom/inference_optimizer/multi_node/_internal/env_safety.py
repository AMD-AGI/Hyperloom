"""Validate env keys forwarded over SSH to multi-node inference pods."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping

from hyperloom.common.env_safety import BLOCKED_UNTRUSTED_ENV_NAMES, valid_env_key

from .log import warn

log = logging.getLogger(__name__)


def is_forward_env_key_allowed(key: str) -> bool:
    """Return True when ``key`` may be forwarded over SSH to pod processes."""
    if not valid_env_key(key):
        return False
    return key not in BLOCKED_UNTRUSTED_ENV_NAMES


def filter_forward_env(
    env: dict[str, str],
    *,
    warn_on_drop: bool = True,
) -> dict[str, str]:
    """Drop disallowed keys from an env dict destined for SSH forwarding."""
    out: dict[str, str] = {}
    for raw_key, raw_val in env.items():
        key = str(raw_key)
        if is_forward_env_key_allowed(key):
            out[key] = str(raw_val)
        elif warn_on_drop:
            log.warning("dropping disallowed multi-node forward env key %r", key)
    return out


def parse_forward_env(source: Mapping[str, str]) -> tuple[dict[str, str], tuple[str, ...]]:
    """Parse variant set/unset controls once, preserving string value coercion."""
    extra: dict[str, str] = {}
    raw = source.get("HYPERLOOM_MN_EXTRA_FWD_ENV", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            warn("HYPERLOOM_MN_EXTRA_FWD_ENV is not valid JSON; skipping per-variant env forwarding")
        else:
            if isinstance(parsed, dict):
                extra = filter_forward_env(parsed)
            else:
                warn("HYPERLOOM_MN_EXTRA_FWD_ENV is not a JSON object; skipping per-variant env forwarding")
    unset: set[str] = set()
    raw = source.get("HYPERLOOM_MN_UNSET_FWD_ENV", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            warn("HYPERLOOM_MN_UNSET_FWD_ENV is not valid JSON; skipping per-variant env unsets")
        else:
            if isinstance(parsed, list):
                unset = {str(key) for key in parsed if is_forward_env_key_allowed(str(key))}
    return extra, tuple(sorted(unset - extra.keys()))


def assert_env_key_shapes(env: dict[str, str]) -> None:
    """Raise ValueError when any env key is not a valid POSIX identifier."""
    bad = [str(k) for k in env if not valid_env_key(str(k))]
    if bad:
        raise ValueError(f"invalid SSH env key names: {bad!r}")


def assert_forward_env_keys(env: dict[str, str]) -> None:
    """Raise ValueError when any env key is not allowed for SSH forwarding."""
    bad = [str(k) for k in env if not is_forward_env_key_allowed(str(k))]
    if bad:
        raise ValueError(f"disallowed SSH forward env keys: {bad!r}")
