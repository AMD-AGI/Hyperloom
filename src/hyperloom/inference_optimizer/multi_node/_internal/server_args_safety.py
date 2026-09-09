"""Denylist validation for framework server CLI flags forwarded to multi-node pods."""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath

# Explicit flags that must not reach pod launchers (path / revision / model injection).
_DENIED_CLI_FLAGS: frozenset[str] = frozenset(
    {
        "--adapter-model-path",
        "--adapter-path",
        "--allowed-local-media-path",
        "--chat-template",
        "--code-revision",
        "--config",
        "--download-dir",
        "--hf-overrides",
        "--lora-dirs",
        "--lora-modules",
        "--lora-path",
        "--lora-paths",
        "--model",
        "--model-id",
        "--model-path",
        "--quantization-param-path",
        "--revision",
        "--tokenizer",
        "--tokenizer-path",
        "--tokenizer-revision",
    }
)

# Suffixes that usually denote filesystem or download injection vectors.
_DENIED_FLAG_SUFFIXES: tuple[str, ...] = (
    "-dir",
    "-file",
    "-path",
)

# Legitimate tuning flags ending in a denied suffix. Explicit denies still win, and only the name is exempt: values
# remain subject to _unsafe_path_value_reason so this cannot reopen filesystem or model injection.
_SUFFIX_EXEMPT_CLI_FLAGS: frozenset[str] = frozenset(
    {
        "--speculative-draft-model-path",
    }
)


def is_denied_server_flag(flag: str) -> bool:
    """Return whether a single CLI flag token is denied at the fan-out boundary."""
    name = (flag or "").strip()
    if not name.startswith("--"):
        return False
    # Explicit deny always wins (defense in depth over the exemption list).
    if name in _DENIED_CLI_FLAGS:
        return True
    # Allow known-safe tuning flags before applying the broad suffix guard.
    if name in _SUFFIX_EXEMPT_CLI_FLAGS:
        return False
    return any(name.endswith(suffix) for suffix in _DENIED_FLAG_SUFFIXES)


def _unsafe_path_value_reason(value: str | None) -> str:
    """Return why an exempt flag's path value is unsafe (\"\" when acceptable)."""
    val = (value or "").strip()
    if not val:
        return "missing value"
    if not val.startswith("/"):
        # Subsumes remote URIs (``http://``, ``s3://``, ``hf://``) and bare HF repo ids, either of which would make
        # every pod run its own uncontrolled download instead of reading the shared filesystem.
        return "must be an absolute path, not a repo id or URI"
    if ".." in PurePosixPath(val).parts:
        return "must not traverse with '..'"
    return ""


def _flag_value_pairs(tokens: list[str]) -> list[tuple[str, str | None]]:
    """Return ``(flag, value)`` pairs for both ``--flag=value`` and ``--flag value``."""
    pairs: list[tuple[str, str | None]] = []
    for idx, tok in enumerate(tokens):
        if not tok.startswith("--"):
            continue
        if "=" in tok:
            name, _, val = tok.partition("=")
            pairs.append((name, val))
            continue
        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
        pairs.append((tok, None if (nxt is None or nxt.startswith("--")) else nxt))
    return pairs


def find_unsafe_flag_values(raw: str) -> list[str]:
    """Return ``\"flag: reason\"`` entries for exempt flags carrying unsafe values."""
    text = (raw or "").strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text)
    except ValueError:
        return []
    out: list[str] = []
    for flag, value in _flag_value_pairs(tokens):
        if flag not in _SUFFIX_EXEMPT_CLI_FLAGS:
            continue
        reason = _unsafe_path_value_reason(value)
        entry = f"{flag}: {reason}"
        if reason and entry not in out:
            out.append(entry)
    return out


class ServerArgsRejected(ValueError):
    """Raised when ``extra_server_args`` contains a denied CLI flag."""


def find_denied_flags(raw: str) -> list[str]:
    """Return denied flag tokens present in a shell-style server-args string."""
    text = (raw or "").strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text)
    except ValueError:
        return ["<unparseable>"]
    denied: list[str] = []
    for tok in tokens:
        flag = tok.split("=", 1)[0]
        if is_denied_server_flag(flag) and flag not in denied:
            denied.append(flag)
    return denied


def validate_server_args(raw: str, *, context: str = "") -> None:
    """Raise :class:`ServerArgsRejected` on denied flags or unsafe flag values."""
    where = f" ({context})" if context else ""
    denied = find_denied_flags(raw)
    if denied:
        raise ServerArgsRejected(f"denied server flags {denied!r}{where}")
    unsafe = find_unsafe_flag_values(raw)
    if unsafe:
        raise ServerArgsRejected(f"unsafe server flag values {unsafe!r}{where}")


def prepare_shell_safe_extra_args(raw: str, *, context: str = "") -> str:
    """Validate ``raw`` and return a shell-safe extra-args string for fan-out."""
    validate_server_args(raw, context=context)
    return shell_safe_extra_args(raw, context=context)


def shell_safe_extra_args(raw: str, *, context: str = "") -> str:
    """Return ``raw`` re-quoted per shell token so it can be spliced after ``--``."""
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        where = f" ({context})" if context else ""
        raise ServerArgsRejected(f"extra_args is not shell-tokenizable: {exc}{where}") from exc
    return " ".join(shlex.quote(tok) for tok in tokens)
