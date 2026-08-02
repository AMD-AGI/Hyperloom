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

# Legitimate optimization knobs that happen to end with a denied suffix.
# These are exempt from the suffix heuristic but stay subject to the explicit
# deny list above, so a hard-blocked flag can never be re-enabled here. The
# suffix rule is a broad guard against filesystem/model injection; flags listed
# here are known tuning parameters (e.g. the speculative-decoding draft model)
# that the optimizer must be allowed to sweep. The exemption covers the flag
# name only -- values stay constrained by _unsafe_path_value_reason below.
_SUFFIX_EXEMPT_CLI_FLAGS: frozenset[str] = frozenset(
    {
        "--speculative-draft-model-path",
    }
)


def is_denied_server_flag(flag: str) -> bool:
    """Return whether a single CLI flag token is denied at the fan-out boundary.

    Args:
        flag: A ``--flag`` token (``flag=value`` callers must split first).

    Returns:
        bool: True when the flag is an explicit deny or matches a denied suffix
        without being an allowlisted exemption.
    """
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
    """Return why an exempt flag's path value is unsafe ("" when acceptable).

    Every exempt flag ends in a denied suffix by construction, so its value is a
    filesystem path. Exempting the name alone would re-open the vector the suffix
    guard closes, hence these shape rules.

    Args:
        value: Token following the flag, or None when the flag carried none.

    Returns:
        str: Human-readable reason, or "" when the value passes every rule.
    """
    val = (value or "").strip()
    if not val:
        return "missing value"
    if not val.startswith("/"):
        # Subsumes remote URIs (``http://``, ``s3://``, ``hf://``) and bare HF
        # repo ids, either of which would make every pod run its own
        # uncontrolled download instead of reading the shared filesystem.
        return "must be an absolute path, not a repo id or URI"
    if ".." in PurePosixPath(val).parts:
        return "must not traverse with '..'"
    return ""


def _flag_value_pairs(tokens: list[str]) -> list[tuple[str, str | None]]:
    """Return ``(flag, value)`` pairs for both ``--flag=value`` and ``--flag value``.

    Args:
        tokens: Shell-split server-arg tokens.

    Returns:
        list[tuple[str, str | None]]: Flag names with their values. A value is
        None when the flag ends the token list or is followed by another ``--``
        flag, so a dangling path flag can never swallow the next flag as a value.
    """
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
    """Return ``"flag: reason"`` entries for exempt flags carrying unsafe values.

    The name-level exemption only decides that a flag may appear; this decides
    what it is allowed to point at.

    Args:
        raw: Whitespace-separated server CLI flags.

    Returns:
        list[str]: Violations found; empty when clean, blank, or unparseable
        (an unparseable string is already reported by :func:`find_denied_flags`).
    """
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
    """Raise :class:`ServerArgsRejected` on denied flags or unsafe flag values.

    Args:
        raw: Whitespace-separated server CLI flags.
        context: Optional label for error messages.

    Raises:
        ServerArgsRejected: When a denied flag is present, or when a suffix-exempt
            flag points at a value outside the allowed path shape.
    """
    where = f" ({context})" if context else ""
    denied = find_denied_flags(raw)
    if denied:
        raise ServerArgsRejected(f"denied server flags {denied!r}{where}")
    unsafe = find_unsafe_flag_values(raw)
    if unsafe:
        raise ServerArgsRejected(f"unsafe server flag values {unsafe!r}{where}")


def prepare_shell_safe_extra_args(raw: str, *, context: str = "") -> str:
    """Validate ``raw`` and return a shell-safe extra-args string for fan-out.

    Args:
        raw: Whitespace-separated server CLI flags (may be empty).
        context: Optional label for error messages.

    Returns:
        str: The re-quoted, shell-safe token string (empty when ``raw`` blank).

    Raises:
        ServerArgsRejected: When ``raw`` is denied or not shell-tokenizable.
    """
    validate_server_args(raw, context=context)
    return shell_safe_extra_args(raw, context=context)


def shell_safe_extra_args(raw: str, *, context: str = "") -> str:
    """Return ``raw`` re-quoted per shell token so it can be spliced after ``--``.

    ``extra_args`` is forwarded verbatim after a ``--`` separator into a Ray
    Dashboard shell entrypoint and word-split by the pod launcher into argv. A
    raw splice lets a value like ``--foo 1; touch x`` inject a second shell
    command. This tokenises with shlex (the same word-splitting the pod applies)
    and re-quotes each token, so multi-token flag/value semantics are preserved
    while any shell metacharacter (``;`` ``|`` ``$()`` …) stays inside a single
    quoted argv token and can no longer act as shell control syntax.

    Args:
        raw: Whitespace-separated server CLI flags (may be empty).
        context: Optional label for error messages.

    Returns:
        str: The re-quoted, shell-safe token string (empty when ``raw`` blank).

    Raises:
        ServerArgsRejected: When ``raw`` is not shell-tokenizable (e.g. an
            unbalanced quote), which a raw splice would carry through unchecked.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        where = f" ({context})" if context else ""
        raise ServerArgsRejected(f"extra_args is not shell-tokenizable: {exc}{where}") from exc
    return " ".join(shlex.quote(tok) for tok in tokens)
