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

from hyperloom.common.failure_signature import IMPORT_FAILURE_LINE
from hyperloom.common.outside_facts import (
    ARGPARSE_INVALID_CHOICE,
    ARGPARSE_UNRECOGNIZED_ARGUMENTS_RE,
    HIP_ERROR_PREFIX,
    HIP_NO_BINARY_FOR_GPU,
    HIP_NO_KERNEL_IMAGE,
    HIP_UNDEFINED_SYMBOL,
    HSA_STATUS_ERROR,
    HUB_RATE_LIMITED,
)

#: A cause line longer than this is cut, so a reason stays one readable line.
CAUSE_CHARS = 300

_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")

#: ``<prog>: error: <message>``, the one shape argparse prints a refused command line in, with the messages argparse
#: itself defines (``argparse.py``).
_ARGPARSE = re.compile(
    r": error: (?:ambiguous option|" + ARGPARSE_UNRECOGNIZED_ARGUMENTS_RE + r"|the following arguments are required"
    r"|argument \S+:|expected (?:one|at most one|at least one|\d+) arguments?|"
    + ARGPARSE_INVALID_CHOICE
    + r"|invalid \w+ value|ignored explicit argument|one of the arguments)"
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

#: What a missing compiled kernel prints: an extension that references a symbol nothing defines, or a GPU binary built for
#: another device. Nothing else a HIP call reports belongs here, nor a failed HSA call: the ROCr queue abort that names an
#: ``HSA_STATUS_ERROR_*`` status is a kernel that ran and faulted.
_MISSING_KERNEL = re.compile("|".join((HIP_UNDEFINED_SYMBOL, HIP_NO_KERNEL_IMAGE, HIP_NO_BINARY_FOR_GPU)))
#: The exception a failed HIP call raised (``torch.AcceleratorError: HIP error: an illegal memory access was
#: encountered``, ``RuntimeError: HIP runtime error: operation failed ...``), which states the fault in words. The
#: ``HIP warning:`` a stream prints for the same fault is not it.
#: A write the file system refused for want of room: ``ENOSPC`` (errno 28) or ``EDQUOT`` (errno 122), as Python and the C
#: library print them.
_DISK_FULL = re.compile(r"\[Errno (?:28|122)\]|No space left on device|Disk quota exceeded", re.IGNORECASE)
_WATCHDOG_TIMEOUT = re.compile(r"watchdog timeout", re.IGNORECASE)
#: A ``git`` command the benchmark client ran to fetch its own dependencies (a per-test ``submodule update`` of
#: the replay client) that timed out: ``subprocess.TimeoutExpired: Command ['git', ...] timed out after N seconds``.
_GIT_FETCH_TIMEOUT = re.compile(r"TimeoutExpired: Command \[['\"]git['\"],.*\] timed out after")
_HUB_RATE_LIMITED = re.compile(HUB_RATE_LIMITED)
#: A frame of aiter's JIT build in a stack dumped at the watchdog: py-spy writes ``build_module (aiter/jit/core.py:N)``, a
#: Python traceback ``File ".../aiter/jit/core.py", line N, in build_module``.
_JIT_BUILD_FRAME = re.compile(r"\bbuild_module\b.*\baiter[/.]jit[/.]core\b|\baiter[/.]jit[/.]core\b.*\bbuild_module\b")
#: The exception a file the process may not open (``EACCES``) raised, as Python prints it: ``PermissionError: [Errno 13]
#: Permission denied: '<path>'``. A logger that reports the same refusal and carries on (``huggingface_hub``'s "Could not
#: cache non-existence of file. Will ignore error and continue. Error: [Errno 13] ...") names no ``OSError`` and does not
#: match.
_PERMISSION_DENIED = re.compile(
    r"(?<![\w.])(?:[A-Za-z_]\w*\.)*(?:PermissionError|OSError|IOError):\s*"
    r"\[Errno 13\] Permission denied: (['\"])(?P<path>[^'\"]+)\1"
)
#: A path in the Hugging Face Hub cache's own layout, under a repository's directory ``models--<org>--<name>`` (or
#: ``datasets--``, ``spaces--``), its blobs and snapshots and its lock in ``.locks``, wherever ``HF_HUB_CACHE`` puts it.
_HUB_CACHE_ENTRY = re.compile(r"/(?:models|datasets|spaces)--[^/]+(?:/|$)")
#: The locks ``datasets`` takes beside a prepared dataset's directory, ``<dataset>/<config>/<version>/<hash>``, while it
#: prepares it (``<hash>_builder.lock``) and while it writes its info (``<hash>_info.lock``); ``<dataset>`` is
#: ``<org>___<name>`` for a Hub repository, the bare name for one without an organisation or a packaged builder.
_DATASETS_BUILD_LOCK = re.compile(r"/[^/]+/[^/]+/\d+\.\d+\.\d+/[0-9a-f]{16,}_(?:builder|info)\.lock$")
_HIP_EXCEPTION = re.compile(r"\bHIP (?:runtime )?error: ")
#: The name of the failed call, on its own line when the exception text was cut (``Search for `hipErrorIllegalAddress'``).
_HIP_ERROR_TOKEN = re.compile(HIP_ERROR_PREFIX + r"[A-Za-z]+")
#: The ROCr runtime's report of a GPU fault in a running kernel, which kills the process before Python raises anything:
#: ``:0:rocdevice.cpp :N: ... Callback: Queue 0x... aborting with error : HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: ...``.
_HSA_STATUS_TOKEN = re.compile(HSA_STATUS_ERROR + r"[A-Z_]*")

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
    return _last_matching(text, lambda line: bool(IMPORT_FAILURE_LINE.search(line)))


def python_exception(text: str) -> str | None:
    """The last ``SomeError: message`` line of a traceback or of a logged exception."""
    return _last_matching(text, lambda line: bool(_EXCEPTION_WITH_MESSAGE.search(line) or _EXCEPTION_ALONE.match(line)))


def git_fetch_timeout(text: str) -> str | None:
    """The line that says a ``git`` command of the benchmark's own set-up timed out, a stall of the network or of the
    remote that no candidate can cause."""
    return _last_matching(text, lambda line: bool(_GIT_FETCH_TIMEOUT.search(line)))


def disk_full(text: str) -> str | None:
    """The line that says a write failed because the disk or the quota was full."""
    return _last_matching(text, lambda line: bool(_DISK_FULL.search(line)))


def missing_hip_kernel(text: str) -> str | None:
    """The line that says a compiled kernel or a symbol of it is absent (an undefined symbol, no image for this GPU)."""
    return _last_matching(text, lambda line: bool(_MISSING_KERNEL.search(line)))


def hip_error(text: str) -> str | None:
    """The line that states a HIP or HSA call failed at run time: the exception that carries the fault's words, else the
    line that names the failed call or the HSA status the runtime aborted the queue with."""
    return _last_matching(text, lambda line: bool(_HIP_EXCEPTION.search(line))) or _last_matching(
        text, lambda line: bool(_HIP_ERROR_TOKEN.search(line) or _HSA_STATUS_TOKEN.search(line))
    )


def jit_build_timeout(text: str) -> str | None:
    """The watchdog's line when a server's scheduler watchdog fired while a stack it dumped was still in aiter's JIT build
    (a module rebuilt at the first prefill, past the watchdog); ``None`` when either is missing."""
    if not _JIT_BUILD_FRAME.search(text):
        return None
    return _last_matching(text, lambda line: bool(_WATCHDOG_TIMEOUT.search(line)))


def hub_rate_limited(text: str) -> str | None:
    """The line that says the Hugging Face Hub refused a download for the rate limit, once the client's retries were spent:
    a limit of the egress address, which no candidate can cause."""
    return _last_matching(text, lambda line: bool(_HUB_RATE_LIMITED.search(line)))


def terminal_error(text: str) -> str | None:
    """The last exception line of ``text`` (:func:`python_exception`'s line, uncut): the error a log ended on, the final line
    of its last traceback when nothing failed after it."""
    for line in reversed(text.splitlines()):
        if _EXCEPTION_WITH_MESSAGE.search(line) or _EXCEPTION_ALONE.match(line):
            return line
    return None


def hf_cache_permission(line: str) -> str | None:
    """The path of a Hugging Face cache whose refusal (``PermissionError: [Errno 13] Permission denied``) ``line``
    raised: a path with ``huggingface`` in it, one in the Hub cache's layout, or a lock ``datasets`` takes for a dataset it
    prepares. The cache is shared by every sandbox that mounts it, so no candidate causes the refusal; any other path
    refused stays what it was.

    Each is known by its own shape, never by ``HF_HOME`` and its kin: the process that reads the log is not always the one
    that ran the test, and a variable naming a broad directory would take a candidate's own files for the cache.
    ``line`` is one log's :func:`terminal_error`: a refusal the process logged and survived is not what ended it.
    """
    for found in reversed(list(_PERMISSION_DENIED.finditer(line))):
        path = found.group("path")
        if "huggingface" in path.lower() or _HUB_CACHE_ENTRY.search(path) or _is_datasets_lock(path):
            return one_line(path)
    return None


def _is_datasets_lock(path: str) -> bool:
    """Whether ``path`` is a lock ``datasets`` takes for a dataset it prepares, whatever ``HF_DATASETS_CACHE`` calls the
    cache: the one in the cache directory, named after the dataset's directory below it with every ``/`` turned into ``_``
    (``<cache>/_<cache with _>_openai___gsm8k_main_0.0.0_<hash>.lock``), or one beside that directory
    (``<cache>/openai___gsm8k/main/0.0.0/<hash>_builder.lock``, ``..._info.lock``)."""
    directory, _, name = path.rpartition("/")
    root_lock = bool(directory) and name.endswith(".lock") and name.startswith(directory.replace("/", "_") + "_")
    return root_lock or bool(_DATASETS_BUILD_LOCK.search(path))
