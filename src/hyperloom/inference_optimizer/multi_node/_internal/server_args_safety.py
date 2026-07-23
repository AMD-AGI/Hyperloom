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

# Legitimate serving flags that end in a denied suffix but are NOT injection
# vectors in the optimizer context — exempted from the suffix catch-all so a
# valid user/LLM param is not rejected (the explicit ``_DENIED_CLI_FLAGS`` deny
# still wins over this allowlist). Keep this list tight: only add flags that are
# real serving/tuning knobs whose value is expected to be a benign path. Extend
# here when a new legitimate ``--*-path/-dir/-file`` flag is needed.
_ALLOWED_PATH_FLAGS: frozenset[str] = frozenset(
    {
        # Draft/EAGLE speculative decoding needs a draft model path; without this
        # the suffix rule would block draft-model spec-decode variants entirely.
        "--speculative-draft-model-path",
    }
)


def is_denied_server_flag(flag: str) -> bool:
    """Return whether a single CLI flag token is denied at the fan-out boundary.

    Precedence: explicit deny wins, then the known-safe allowlist exempts a flag
    from the suffix catch-all, then the ``-dir/-file/-path`` suffix rule applies.

    Args:
        flag: A ``--flag`` token (``flag=value`` callers must split first).

    Returns:
        bool: True when the flag is an explicit deny or matches a denied suffix
        and is not on the known-safe allowlist.
    """
    name = (flag or "").strip()
    if not name.startswith("--"):
        return False
    if name in _DENIED_CLI_FLAGS:
        return True
    if name in _ALLOWED_PATH_FLAGS:
        return False
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
