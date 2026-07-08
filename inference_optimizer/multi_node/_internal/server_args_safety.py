"""Denylist validation for framework server CLI flags forwarded to multi-node pods."""

from __future__ import annotations

import shlex

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


def is_denied_server_flag(flag: str) -> bool:
    """Return whether a single CLI flag token is denied at the fan-out boundary.

    Args:
        flag: A ``--flag`` token (``flag=value`` callers must split first).

    Returns:
        bool: True when the flag is an explicit deny or matches a denied suffix.
    """
    name = (flag or "").strip()
    if not name.startswith("--"):
        return False
    if name in _DENIED_CLI_FLAGS:
        return True
    return any(name.endswith(suffix) for suffix in _DENIED_FLAG_SUFFIXES)


class ServerArgsRejected(ValueError):
    """Raised when ``extra_server_args`` contains a denied CLI flag."""


def find_denied_flags(raw: str) -> list[str]:
    """Return denied flag tokens present in a shell-style server-args string.

    Args:
        raw: Whitespace-separated server CLI flags.

    Returns:
        list[str]: Denied flag names found (empty when clean or blank).
    """
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
    """Raise :class:`ServerArgsRejected` when ``raw`` contains denied flags.

    Args:
        raw: Whitespace-separated server CLI flags.
        context: Optional label for error messages.

    Raises:
        ServerArgsRejected: When one or more denied flags are present.
    """
    denied = find_denied_flags(raw)
    if denied:
        where = f" ({context})" if context else ""
        raise ServerArgsRejected(f"denied server flags {denied!r}{where}")
