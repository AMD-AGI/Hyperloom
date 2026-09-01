# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sanitize knowledge before it crosses into the shared KB Store."""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Mapping
from typing import Any

from hyperloom.common.env_safety import (
    BENCHMARK_SECRET_ENV_NAMES,
    GPU_MASK_ENV_NAMES,
    redact_secret_values,
    valid_env_key,
)

_DROP = object()

#: The one key whose subtree may carry absolute paths from the producing host.
#: Deliberately unambiguous rather than a word like ``origin`` that another
#: producer could introduce for something else and silently widen the exemption.
HOST_ORIGIN_KEY = "host_origin"

_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_PATH_KEY_RE = re.compile(
    r"(?:^|_)(?:path|paths|dir|directory|file|filename|root|workspace|home)(?:$|_)",
    re.IGNORECASE,
)
_EMBEDDED_UNIX_PATH_RE = re.compile(
    r"(^|[\s=(\[,])(?:file://)?/(?:[^\s,;)\]}'\"]+)",
)
_EMBEDDED_WINDOWS_PATH_RE = re.compile(
    r"(^|[\s=(\[,])[A-Za-z]:[\\/](?:[^\s,;)\]}'\"]+)",
)

# Shared replay should carry optimization knobs, not arbitrary process state.
# Prefixes are deliberately narrow and can be extended when a new producer
# introduces a reviewed, replayable environment family.
PUBLISH_ENV_EXACT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "GPU_MAX_HW_QUEUES",
        "NUM_PROMPTS",
        "NUM_WARMUPS",
        "OMP_NUM_THREADS",
    }
)
PUBLISH_ENV_PREFIX_ALLOWLIST: tuple[str, ...] = (
    "AITER_",
    "CK_",
    "FLASHINFER_",
    "FLASH_ATTENTION_",
    "HIP_",
    "HIPBLASLT_",
    "HSA_",
    "MIOPEN_",
    "NCCL_",
    "PYTORCH_TUNABLEOP_",
    "RCCL_",
    "ROCBLAS_",
    "ROCM_",
    "SGLANG_",
    "TORCHINDUCTOR_",
    "TRITON_",
    "VLLM_",
)


def _key_parts(value: str) -> tuple[list[str], str]:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    parts = [part for part in re.split(r"[^a-z0-9]+", snake_case.lower()) if part]
    return parts, "".join(parts)


def _is_secret_key(value: str) -> bool:
    upper = str(value or "").strip().upper()
    if upper in BENCHMARK_SECRET_ENV_NAMES:
        return True
    parts, normalized = _key_parts(value)
    return "token" in parts or any(
        marker in normalized
        for marker in (
            "authorization",
            "password",
            "passwd",
            "secret",
            "apikey",
            "privatekey",
            "accesskey",
            "sshkey",
            "credential",
        )
    )


def _is_secret_option(value: str) -> bool:
    parts, normalized = _key_parts(value)
    if any(
        marker in normalized
        for marker in (
            "authorization",
            "password",
            "passwd",
            "secret",
            "apikey",
            "privatekey",
            "accesskey",
            "sshkey",
            "credential",
        )
    ):
        return True
    if not parts or "token" not in parts:
        return False
    return (
        len(parts) == 1
        or parts[-1] == "token"
        or bool({"api", "auth", "bearer", "access", "refresh", "hf", "gbrain"} & set(parts))
    )


def _is_absolute_path_text(value: str) -> bool:
    text = str(value or "").strip()
    return text.startswith(("/", "~/", "file://")) or bool(_WINDOWS_ABSOLUTE_RE.match(text))


def _redact_embedded_paths(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix = match.group(1)
        return f"{prefix}[LOCAL_PATH]"

    text = _EMBEDDED_UNIX_PATH_RE.sub(replace, value)
    return _EMBEDDED_WINDOWS_PATH_RE.sub(replace, text)


def _is_publish_env_key(value: str) -> bool:
    upper = str(value or "").strip().upper()
    if (
        not valid_env_key(upper)
        or _is_secret_key(upper)
        or upper in BENCHMARK_SECRET_ENV_NAMES
        or upper in GPU_MASK_ENV_NAMES
    ):
        return False
    return upper in PUBLISH_ENV_EXACT_ALLOWLIST or any(
        upper.startswith(prefix) for prefix in PUBLISH_ENV_PREFIX_ALLOWLIST
    )


def sanitize_publish_env_mapping(
    envs: Mapping[str, object] | None,
) -> dict[str, str]:
    """Return only reviewed, secret-free, host-independent replay envs."""
    safe: dict[str, str] = {}
    for raw_name, raw_value in (envs or {}).items():
        name = str(raw_name or "").strip()
        if not _is_publish_env_key(name):
            continue
        value = os.fspath(raw_value) if isinstance(raw_value, os.PathLike) else str(raw_value)
        if _is_absolute_path_text(value):
            continue
        redacted = redact_secret_values(value)
        if redacted != value:
            continue
        safe[name] = _redact_embedded_paths(value)
    return safe


def sanitize_publish_server_args(value: str) -> str:
    """Remove credential options and host-local path operands from argv text."""
    try:
        tokens = shlex.split(str(value or ""))
    except ValueError as exc:
        raise ValueError(f"extra_server_args is not shell-tokenizable: {exc}") from exc
    safe: list[str] = []
    skip_value = False
    for token in tokens:
        if skip_value:
            skip_value = False
            continue
        option, separator, operand = token.partition("=")
        normalized_option = option.lstrip("-")
        if _is_secret_option(normalized_option):
            skip_value = not separator
            continue
        path_operand = operand if separator else token
        if _is_absolute_path_text(path_operand):
            if not separator and safe and safe[-1].startswith("-") and "=" not in safe[-1]:
                safe.pop()
            continue
        redacted = redact_secret_values(token)
        if redacted != token or "[LOCAL_PATH]" in _redact_embedded_paths(token):
            if not separator and safe and safe[-1].startswith("-") and "=" not in safe[-1]:
                safe.pop()
            continue
        safe.append(token)
    whitespace_tokens = [token for token in safe if any(ch.isspace() for ch in token)]
    if whitespace_tokens:
        raise ValueError(
            "extra_server_args contains a whitespace-bearing value unsupported "
            "by Magpie's unquoted environment expansion"
        )
    from ...actions.executors._grid_server_args import _reserialize_json_blobs

    return _reserialize_json_blobs(" ".join(safe))


def _sanitize_value(value: Any, *, key: str = "", allow_absolute: bool = False) -> Any:
    if key in {
        "source_file",
        "source_files",
        "target_file",
        "target_files",
        "target_path",
    }:
        return _DROP
    if _is_secret_key(key):
        return _DROP
    if key == "extra_server_args":
        return sanitize_publish_server_args(str(value or ""))
    if key == "extra_envs":
        return sanitize_publish_env_mapping(value if isinstance(value, Mapping) else None)
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for raw_key, nested in value.items():
            name = str(raw_key)
            cleaned = _sanitize_value(
                nested,
                key=name,
                allow_absolute=allow_absolute or name == HOST_ORIGIN_KEY,
            )
            if cleaned is not _DROP:
                safe[name] = cleaned
        return safe
    if isinstance(value, (list, tuple, set)):
        safe_items = []
        for item in value:
            cleaned = _sanitize_value(item, key=key, allow_absolute=allow_absolute)
            if cleaned is not _DROP:
                safe_items.append(cleaned)
        return safe_items
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if isinstance(value, str):
        if _is_absolute_path_text(value):
            # Everywhere but the host-origin subtree, an absolute path is a leak
            # of the producing machine. There it is the payload: a later session
            # cannot look for the checkout a KEEP was taken from without being
            # told where that was.
            return value if allow_absolute else _DROP
        if _PATH_KEY_RE.search(key) and ("/" in value or "\\" in value):
            # Relative bundle references remain replayable; absolute values were
            # removed above.
            return value
        return _redact_embedded_paths(redact_secret_values(value))
    return value


def sanitize_shared_knowledge(knowledge: Mapping[str, Any]) -> dict[str, Any]:
    """Return a publishable copy with secrets and host paths removed.

    Host paths survive in exactly one place, the :data:`HOST_ORIGIN_KEY` subtree,
    because a KEEP that cannot say which checkout it came from cannot be replayed
    on an image that lays that checkout out somewhere else.

    The exemption is scoped precisely: inside the subtree an absolute-path string
    is returned verbatim, which means the value-level scrubbing every other
    string gets -- ``redact_secret_values`` and embedded-path redaction -- is
    bypassed for it (that is the whole point: the path must survive intact).
    Key-level dropping is *not* relaxed there: a secret-named key is still
    removed inside the subtree exactly as it is everywhere else. So do not put a
    credential-bearing value under this key expecting the path exemption to also
    launder its content -- it will not.
    """
    cleaned = _sanitize_value(dict(knowledge))
    return cleaned if isinstance(cleaned, dict) else {}


__all__ = [
    "HOST_ORIGIN_KEY",
    "PUBLISH_ENV_EXACT_ALLOWLIST",
    "PUBLISH_ENV_PREFIX_ALLOWLIST",
    "sanitize_publish_env_mapping",
    "sanitize_publish_server_args",
    "sanitize_shared_knowledge",
]
