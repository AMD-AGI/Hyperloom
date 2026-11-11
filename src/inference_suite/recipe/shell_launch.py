# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lift a static launch from a shell launch script: the flags of the one command that starts the server, and the
literal exports.

The script is read, never run. What its text alone cannot fix -- a flag whose value names a shell variable, an
export whose value expands one -- is listed apart rather than guessed, and the caller decides whether that is an
error. Which exported names a launch may carry is the caller's policy too, so :attr:`ShellLaunch.env` is unfiltered.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from inference_suite.argv import is_flag

#: Flags the runner sets for every round, so a script's own values are taken out.
_RUNNER_OWNED_FLAGS = frozenset({"--port", "--host", "--served-model-name", "--result-dir", "--result-filename"})
_RUNNER_OWNED_PREFIXES = ("--result-", "--served-model")
_MODEL_FLAGS = frozenset({"--model", "--model-path"})
#: ``python3 -m <module>`` entrypoints, by module prefix, and the frameworks whose CLI runs ``<framework> serve``.
_ENTRY_MODULES = {"sglang": "sglang.launch_server", "atom": "atom.entrypoints"}
_SERVE_CLIS = frozenset({"vllm", "sglang"})
_EXPORT = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
# ``${FOO:-1}`` / ``${FOO-1}``, capturing the name and the default.
_SELF_DEFAULT = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*):?-(.*)\}$")


class ShellLaunchError(ValueError):
    """A script's text does not fix one launch."""


@dataclass(frozen=True)
class ShellLaunch:
    """What a launch script fixes in its text.

    ``args`` are the entrypoint's flags with literal values, in script order. ``unresolved_flags`` name the flags
    whose value is a shell variable and ``stripped`` the runner-owned ones; neither is in ``args``. ``model`` is the
    literal model the entrypoint names. ``env`` holds every literal export, a ``${NAME:-default}`` of its own name
    resolved to the default; ``unresolved_env`` names the exports whose value only a shell could produce.
    """

    has_entrypoint: bool
    args: tuple[str, ...] = ()
    unresolved_flags: tuple[str, ...] = ()
    stripped: tuple[str, ...] = ()
    model: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    unresolved_env: tuple[str, ...] = ()


def _entrypoint_end(tokens: Sequence[str], framework: str) -> int | None:
    """The index just past ``framework``'s serving entrypoint when ``tokens`` run it, else None.

    Matched on words, so a comment, or an ``echo`` whose quoted text names the entrypoint, runs no server.
    """
    module = _ENTRY_MODULES.get(framework)
    for i, token in enumerate(tokens[:-1]):
        following = tokens[i + 1]
        if token == "-m" and module is not None and following.startswith(module):
            return i + 2
        if framework in _SERVE_CLIS and Path(token).name == framework and following == "serve":
            return i + 2
    return None


def parse_shell_launch(text: str, *, framework: str) -> ShellLaunch:
    """Lift ``framework``'s launch from the shell script ``text``.

    Raises:
        ShellLaunchError: When the script starts more than one server, or a line naming the entrypoint does not parse
            as shell words.
    """
    env, unresolved_env = _exports(text)
    launch: list[str] | None = None
    for line in _logical_lines(text):
        if framework not in line:
            continue
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError as error:
            raise ShellLaunchError(f"a line naming {framework} does not parse as shell words: {error}") from error
        end = _entrypoint_end(tokens, framework)
        if end is None:
            continue
        if launch is not None:
            raise ShellLaunchError(f"it starts more than one {framework} server")
        launch = tokens[end:]
    if launch is None:
        return ShellLaunch(has_entrypoint=False, env=env, unresolved_env=unresolved_env)
    return _lift_entrypoint(launch, env=env, unresolved_env=unresolved_env)


def _logical_lines(text: str) -> Iterator[str]:
    """Join backslash line continuations into single lines."""
    pending = ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            pending += line.rstrip()[:-1] + " "
            continue
        yield pending + line
        pending = ""
    if pending:
        yield pending


def _without_redirection(tokens: list[str]) -> list[str]:
    """Drop the shell redirection and backgrounding tail (``> log 2>&1 &``)."""
    kept: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("&", ";"):
            break
        if token.startswith((">", "<", "2>")) or "2>&1" in token:
            i += 2 if token in (">", "<", "2>") else 1
            continue
        kept.append(token)
        i += 1
    return kept


def _flag_value(tokens: list[str], i: int) -> tuple[str | None, int]:
    """The value of the flag at ``tokens[i]`` (None when bare) and how many tokens the flag spans."""
    token = tokens[i]
    if "=" in token:
        return token.split("=", 1)[1], 1
    if i + 1 < len(tokens) and not is_flag(tokens[i + 1]):
        return tokens[i + 1], 2
    return None, 1


def _runner_owned(name: str) -> bool:
    return name in _RUNNER_OWNED_FLAGS or name.startswith(_RUNNER_OWNED_PREFIXES)


def _lift_entrypoint(tokens: list[str], *, env: dict[str, str], unresolved_env: tuple[str, ...]) -> ShellLaunch:
    tokens = _without_redirection(tokens)
    model: str | None = None
    args: list[str] = []
    unresolved: list[str] = []
    stripped: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not is_flag(token):
            if model is None and "$" not in token:
                model = token
            i += 1
            continue
        name = token.split("=", 1)[0]
        value, span = _flag_value(tokens, i)
        if name in _MODEL_FLAGS and value is not None:
            model = model if "$" in value else value
        elif _runner_owned(name):
            stripped.append(name)
        elif value is not None and "$" in value:
            unresolved.append(name)
        else:
            args.extend(tokens[i : i + span])
        i += span
    return ShellLaunch(
        has_entrypoint=True,
        args=tuple(args),
        unresolved_flags=tuple(unresolved),
        stripped=tuple(stripped),
        model=model,
        env=env,
        unresolved_env=unresolved_env,
    )


def _exports(text: str) -> tuple[dict[str, str], tuple[str, ...]]:
    env: dict[str, str] = {}
    unresolved: list[str] = []
    lines = iter(text.splitlines(keepends=True))
    for line in lines:
        match = _EXPORT.match(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2)
        while True:
            try:
                tokens = shlex.split(value)
                break
            except ValueError:
                continuation = next(lines, None)
                if continuation is None:
                    tokens = []
                    break
                value += continuation
        literal = _literal_export(name, value, tokens)
        if literal is None:
            unresolved.append(name)
        else:
            env[name] = literal
    return env, tuple(unresolved)


def _literal_export(name: str, value: str, tokens: list[str]) -> str | None:
    if len(tokens) != 1 and value.strip():
        return None
    literal = tokens[0] if tokens else ""
    if not _has_shell_expansion(value):
        return literal
    match = _SELF_DEFAULT.match(literal)
    if not match or match.group(1) != name or _has_shell_expansion(match.group(2)):
        return None
    return match.group(2)


def _has_shell_expansion(value: str) -> bool:
    """Recognize dynamic shell syntax while preserving quoted literal values."""
    quote = ""
    escaped = False
    for char in value.strip():
        if escaped:
            escaped = False
        elif quote == "'":
            if char == "'":
                quote = ""
        elif char == "\\":
            escaped = True
        elif char == quote:
            quote = ""
        elif not quote and char in ("'", '"'):
            quote = char
        elif char in ("$", "`") or (not quote and char in ";&|<>()"):
            return True
    return False


__all__ = ["ShellLaunch", "ShellLaunchError", "parse_shell_launch"]
