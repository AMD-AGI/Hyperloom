# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a failed test's logs say went wrong, in the log's own words.

The signature table of :mod:`hyperloom.common.failure_signature` knows the failures an enablement run acts on. A
measurement also meets failures it does not know: the server refuses its own command line, the evaluation client is
answered ``400``, a worker dies on a Python exception. Each of them names its cause in one line of the server log or of
the benchmark's stderr, and that line is what the specialist needs, so each reader here returns it. A failure the
signature table does know (an import that failed, a refused flag) quotes its line the same way.

Every reader takes the joined log text and returns the last line that names the cause, shortened to one bounded line, or
``None``.
"""

from __future__ import annotations

import re

#: A cause line longer than this is cut, so a reason stays one readable line.
CAUSE_CHARS = 300

_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")

#: ``<prog>: error: <message>``, the one shape argparse prints a refused command line in, with the messages argparse
#: itself defines (``argparse.py``).
_ARGPARSE = re.compile(
    r": error: (?:ambiguous option|unrecognized arguments?|the following arguments are required"
    r"|argument \S+:|expected (?:one|at most one|at least one|\d+) arguments?|invalid choice"
    r"|invalid \w+ value|ignored explicit argument|one of the arguments)"
)
#: A compiler's ``file.c:3:1: error: ...`` shares the ``: error:`` with argparse and names a source location first.
_COMPILER_DIAGNOSTIC = re.compile(r":\d+(?::\d+)?: error:")

#: A ``400`` the server answered or the evaluation client reported: the access-log line, an ``aiohttp``/``requests``
#: error, or the JSON body the server sent back.
_BAD_REQUEST = re.compile(
    r"\b400 Bad Request|HTTPError: 400|ClientResponseError: 400|\b400 Client Error|BadRequestError"
    r'|"code"\s*:\s*400\b|[Ss]tatus(?: code)?[:=]\s*400\b|HTTP/1\.[01]"\s+400\b'
)
#: The lm_eval client's own report of a refused request carries the server's message, which names the cause.
_REFUSAL_MESSAGE = re.compile(r"API request failed|[\"']message[\"']\s*:")

#: The exception a failed import raised, ``ImportError: message`` or ``ModuleNotFoundError: message`` after an optional
#: log prefix. A server that skips a model whose imports fail logs ``Ignore import error when loading <module>: ...``
#: and carries on; that line is not an exception and does not match.
_IMPORT_FAILURE = re.compile(r"(?<![\w.])(?:[A-Za-z_]\w*\.)*(?:ImportError|ModuleNotFoundError):\s*\S")

#: ``SomeError: message`` after an optional log prefix, or a bare ``SomeError`` on a line of its own.
_EXCEPTION_WITH_MESSAGE = re.compile(r"(?<![\w.])((?:[A-Za-z_]\w*\.)*(?:[A-Z]\w*)?(?:Error|Exception)):\s*\S")
_EXCEPTION_ALONE = re.compile(r"^\s*(?:[A-Za-z_]\w*\.)*(?:[A-Z]\w*)?(?:Error|Exception)\s*$")


def one_line(text: str) -> str:
    """``text`` as one line of at most :data:`CAUSE_CHARS` characters."""
    flat = _CONTROL.sub(" ", text).strip()
    return flat if len(flat) <= CAUSE_CHARS else flat[: CAUSE_CHARS - 3] + "..."


def last_line(text: str) -> str:
    """The last non-blank line of ``text`` as one bounded line; ``""`` when there is none."""
    for line in reversed(text.splitlines()):
        if line.strip():
            return one_line(line)
    return ""


def _last_matching(text: str, matches) -> str | None:
    for line in reversed(text.splitlines()):
        if matches(line):
            return one_line(line)
    return None


def argparse_error(text: str) -> str | None:
    """The command-line error a server printed when it refused its own arguments (an ambiguous or unknown flag)."""
    return _last_matching(text, lambda line: bool(_ARGPARSE.search(line)) and not _COMPILER_DIAGNOSTIC.search(line))


def bad_request(text: str) -> str | None:
    """The ``400`` an evaluation request was answered with; the line that carries the server's message when there is one."""
    lines = [line for line in text.splitlines() if _BAD_REQUEST.search(line)]
    with_message = [line for line in lines if _REFUSAL_MESSAGE.search(line)]
    chosen = (with_message or lines or [None])[-1]
    return None if chosen is None else one_line(chosen)


def import_error(text: str) -> str | None:
    """The ``ImportError``/``ModuleNotFoundError`` line a failed import ended the server with."""
    return _last_matching(text, lambda line: bool(_IMPORT_FAILURE.search(line)))


def python_exception(text: str) -> str | None:
    """The last ``SomeError: message`` line of a traceback or of a logged exception."""
    return _last_matching(text, lambda line: bool(_EXCEPTION_WITH_MESSAGE.search(line) or _EXCEPTION_ALONE.match(line)))
