# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""llama.cpp's ``test-backend-ops`` as a microbenchmark of exact shapes: the case file it reads, the times it prints,
and its check of a backend against the CPU.

The cases are written to a file (``--test-file``) rather than picked from the tool's own perf list, whose shapes are not
a model's. A line names the operation and the types by the numbers of ggml's enums, which are read from the header of
the tree the executable was built from (:func:`ggml_enum`), never assumed. Each case is named, and the name is what the
output is read back by. How the tool is driven and read is the pack's (``LLAMACPP_BACKEND_OPS``).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.outside_facts import LLAMACPP_BACKEND_OPS

__all__ = [
    "CHECK_OK",
    "BackendOpsError",
    "MulMatCase",
    "case_file",
    "check_cases",
    "ggml_enum",
    "time_cases",
]

_TEST_MODE = "test"
_ENUM_ENTRY = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*(?P<value>[^,]+?))?\s*,?\s*$")
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_PERF_LINE = re.compile(r"^\s*\w+\(name=(?P<name>[^,)]+)[^\n]*?:\s+(?P<runs>\d+) runs -\s+(?P<us>[0-9.]+) us/run")
_CHECK_LINE = re.compile(r"^\s*\w+\(name=(?P<name>[^,)]+)[^\n]*?:\s+(?P<verdict>.*)$")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
#: The verdict :func:`check_cases` gives a case that matches the CPU backend.
CHECK_OK = "OK"
_TAIL_CHARS = 2000


class BackendOpsError(RuntimeError):
    """The tool could not time or check the cases: it failed, a case was not supported, or its output lacks one."""


@dataclass(frozen=True)
class MulMatCase:
    """``n`` columns of activations times an ``m`` x ``k`` weight of type ``type_a`` (a ggml type name without its
    prefix, ``Q8_0``), named ``name`` (no blank, comma or parenthesis), in ``copies`` independent products of one
    launch (ggml's third dimension), so the weights read are ``copies`` times one weight."""

    name: str
    type_a: str
    m: int
    k: int
    n: int
    copies: int = 1


def ggml_enum(header: str, enum: str) -> dict[str, int]:
    """The values of ``enum <enum> { ... }`` in the C header text ``header``: explicit ones as written, the others one
    more than the entry before.

    Raises:
        BackendOpsError: The header holds no such enum, or an entry's value is not an integer or an earlier entry.
    """
    found = re.search(rf"enum\s+{re.escape(enum)}\s*\{{(?P<body>.*?)\}}", _COMMENT.sub("", header), re.DOTALL)
    if found is None:
        raise BackendOpsError(f"the header holds no enum {enum}")
    values: dict[str, int] = {}
    following = 0
    for part in found.group("body").split(","):
        entry = _ENUM_ENTRY.match(part)
        if entry is None or not part.strip():
            continue
        written = entry.group("value")
        if written is None:
            value = following
        elif written.strip() in values:
            value = values[written.strip()]
        else:
            try:
                value = int(written.strip(), 0)
            except ValueError:
                raise BackendOpsError(f"enum {enum}: {entry.group('name')} = {written!r} is not a number") from None
        values[entry.group("name")] = value
        following = value + 1
    return values


def case_file(cases: Sequence[MulMatCase], header: str) -> str:
    """The ``--test-file`` text of ``cases``, numbered by the enums of ``header`` (the tree's ggml header).

    Raises:
        BackendOpsError: The header lacks the operation or a type.
    """
    facts = LLAMACPP_BACKEND_OPS
    ops, types = ggml_enum(header, facts["op_enum"]), ggml_enum(header, facts["type_enum"])

    def number(table: Mapping[str, int], prefix: str, name: str) -> int:
        if f"{prefix}{name}" not in table:
            raise BackendOpsError(f"the tree's {facts['enum_header']} has no {prefix}{name}")
        return table[f"{prefix}{name}"]

    op = number(ops, facts["op_prefix"], "MUL_MAT")
    f32 = number(types, facts["type_prefix"], facts["activation_type"])
    lines = []
    for case in cases:
        weight = number(types, facts["type_prefix"], case.type_a)
        contiguous = "0 0 0 0"
        c = case.copies
        lines.append(
            f"{op} {f32} {case.m} {case.n} {c} 1 0 2 "
            f"{weight} {case.k} {case.m} {c} 1 {contiguous} {f32} {case.k} {case.n} {c} 1 {contiguous} {case.name}"
        )
    return "\n".join(lines) + "\n"


def _run(executable: Path, mode: str, cases: Path, env: Mapping[str, str], timeout: float) -> tuple[int, str]:
    facts = LLAMACPP_BACKEND_OPS
    command = [str(executable), mode, "-b", facts["backend"], "--test-file", str(cases)]
    try:
        done = subprocess.run(command, capture_output=True, text=True, env=dict(env), timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BackendOpsError(f"{executable.name} {mode} did not run to its end: {exc}") from None
    return done.returncode, _ANSI.sub("", done.stdout + done.stderr)


def _every(names: Sequence[str], found: Mapping[str, object], output: str) -> None:
    missing = [name for name in names if name not in found]
    if missing:
        raise BackendOpsError(f"the output names no result of {missing}: {output[-_TAIL_CHARS:]}")


def time_cases(
    executable: Path, cases: Path, names: Sequence[str], env: Mapping[str, str], timeout: float
) -> dict[str, float]:
    """One run of the perf mode over the case file ``cases``: the microseconds per run of each case, by name.

    Raises:
        BackendOpsError: The tool failed, or its output lacks a time of one of ``names`` (a case it does not support).
    """
    code, output = _run(executable, LLAMACPP_BACKEND_OPS["mode"], cases, env, timeout)
    if code != 0:
        raise BackendOpsError(f"{executable.name} exited {code}: {output[-_TAIL_CHARS:]}")
    times = {match["name"]: float(match["us"]) for match in map(_PERF_LINE.match, output.splitlines()) if match}
    _every(names, times, output)
    return {name: times[name] for name in names}


def check_cases(
    executable: Path, cases: Path, names: Sequence[str], env: Mapping[str, str], timeout: float
) -> dict[str, str]:
    """The test mode over ``cases``: each case's result against the CPU backend, ``OK`` when it matched, else what the
    tool said.

    The tool exits non-zero when a case does not match; that is a result, read like the others.

    Raises:
        BackendOpsError: The tool could not run, or its output lacks a result of one of ``names``.
    """
    _, output = _run(executable, _TEST_MODE, cases, env, timeout)
    verdicts = {
        match["name"]: match["verdict"].strip() for match in map(_CHECK_LINE.match, output.splitlines()) if match
    }
    _every(names, verdicts, output)
    return {name: CHECK_OK if verdicts[name].startswith(CHECK_OK) else verdicts[name] for name in names}
