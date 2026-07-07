"""Denylist validation for framework server CLI flags forwarded to multi-node pods."""

from __future__ import annotations

import shlex

# Flags that must not reach pod launchers (path / revision injection vectors).
_DENIED_CLI_FLAGS: frozenset[str] = frozenset(
    {
        "--allowed-local-media-path",
        "--download-dir",
        "--revision",
        "--code-revision",
        "--tokenizer-path",
    }
)


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
        if flag in _DENIED_CLI_FLAGS and flag not in denied:
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
