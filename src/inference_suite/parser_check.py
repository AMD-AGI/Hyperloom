# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hold a composed launch to the installed framework's own argument parser (D7).

The parser runs in the serving interpreter with the launch environment, after every patch on that path has taken
effect, so a flag a patch adds is known. A flag the parser still does not have is refused, never dropped, and the
refusal names the layer that set it: a candidate's flag costs the candidate, a base or user flag stops the line.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from inference_suite.contracts.launch import Launch, LaunchArg, LayerSource
from inference_suite.contracts.report import FailedStep
from inference_suite.framework_probe import run_probe
from inference_suite.frameworks import Framework, framework_facts

_DETAIL_CHARS = 600

# ``parse_args``, not ``parse_known_args``: vLLM's FlexibleArgumentParser expands
# ``--<group>-config.<field> <value>`` into the group's JSON inside ``parse_args`` only, so the
# ``parse_known_args`` entry point reports every dotted flag as unrecognised. The leftover tokens are recovered
# from the intercepted ``unrecognized arguments`` message instead. Required-ness is cleared first: the argv under
# test is the tail after the entrypoint, not a whole invocation.
_PARSE_PROGRAM = (
    "import json, re, sys\n"
    "\n"
    "\n"
    "def _emit(status, message='', unknown=()):\n"
    "    print(json.dumps({'status': status, 'message': str(message)[-4000:], 'unknown': list(unknown)}))\n"
    "    raise SystemExit(0)\n"
    "\n"
    "\n"
    "class _Rejected(Exception):\n"
    "    pass\n"
    "\n"
    "\n"
    "def _reject(message, *_a, **_k):\n"
    "    raise _Rejected(str(message))\n"
    "\n"
    "\n"
    "argv = json.loads(sys.argv[1])\n"
    "parser = _build_parser()\n"
    "for action in getattr(parser, '_actions', ()):\n"
    "    action.required = False\n"
    "for group in getattr(parser, '_mutually_exclusive_groups', ()):\n"
    "    group.required = False\n"
    "parser.error = _reject\n"
    "parser.exit = _reject\n"
    "try:\n"
    "    parser.parse_args(argv)\n"
    "except _Rejected as exc:\n"
    "    leftover = re.search(r'unrecognized arguments:\\s*(.*)', str(exc), re.S)\n"
    "    _emit('invalid', exc, leftover.group(1).split() if leftover else ())\n"
    "_emit('ok')\n"
)

#: How argparse names the argument whose value it rejected: ``argument --tp/--tensor-parallel-size: ...``.
_REJECTED_ARGUMENT = re.compile(r"argument (-[^\s:]+):")


class ParserCheckError(RuntimeError):
    """The parser check could not reach a verdict: no parser for the framework, or an answer it cannot read."""


class ParseStatus(str, Enum):
    ACCEPTED = "ok"
    REJECTED = "invalid"


@dataclass(frozen=True)
class ParserProbe:
    """The program and arguments that run a framework's parser over one argv in the serving interpreter."""

    program: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class ParserAnswer:
    """What the installed parser said about one argv.

    ``unknown_flags`` are the flags it does not have; ``rejected_option`` is every spelling of the one argument
    whose value it refused, empty when it named none.
    """

    status: ParseStatus
    detail: str
    unknown_flags: tuple[str, ...] = ()
    rejected_option: tuple[str, ...] = ()


@dataclass(frozen=True)
class RefusedFlag:
    flag: str
    source: LayerSource


@dataclass(frozen=True)
class FlagRefusal:
    """Why the installed parser refused the launch, and the layers whose flags it refused.

    ``flags`` is empty when the parser refused the argv without naming a flag the launch carries.
    """

    flags: tuple[RefusedFlag, ...]
    detail: str
    step: FailedStep = FailedStep.FLAG_UNKNOWN

    @property
    def source(self) -> LayerSource | None:
        """The earliest layer among the refused flags; ``None`` when the parser named none."""
        order = list(LayerSource)
        return min((flag.source for flag in self.flags), key=order.index, default=None)


def parser_probe(framework: Framework, argv: Sequence[str]) -> ParserProbe | None:
    """The probe that runs ``framework``'s parser over ``argv``; ``None`` when the framework has no parser."""
    source = framework_facts(framework).parser_source
    if source is None:
        return None
    return ParserProbe(program=source + "\n" + _PARSE_PROGRAM, args=(json.dumps(list(argv)),))


def _flag_names(tokens: Sequence[object]) -> tuple[str, ...]:
    names: list[str] = []
    for token in tokens:
        text = str(token).strip()
        name = text.split("=", 1)[0]
        if text.startswith("-") and name not in names:
            names.append(name)
    return tuple(names)


def read_answer(answer: Mapping[str, object]) -> ParserAnswer:
    """The parser's verdict from the object the parse program printed."""
    status = str(answer.get("status", ""))
    message = str(answer.get("message", ""))
    detail = message.strip()[-_DETAIL_CHARS:]
    if status not in {item.value for item in ParseStatus}:
        raise ParserCheckError(detail or f"parser probe reported {status!r}")
    unknown = answer.get("unknown")
    rejected = _REJECTED_ARGUMENT.search(message)
    return ParserAnswer(
        status=ParseStatus(status),
        detail=detail,
        unknown_flags=_flag_names(unknown if isinstance(unknown, list) else ()),
        rejected_option=tuple(rejected.group(1).split("/")) if rejected else (),
    )


def _carrier(launch: Launch, flag: str) -> LaunchArg:
    for arg in launch.args:
        if flag in _flag_names(arg.tokens()):
            return arg
    raise ParserCheckError(f"the parser refused {flag}, which the launch does not carry")


def _refused_flags(launch: Launch, answer: ParserAnswer) -> tuple[RefusedFlag, ...]:
    if answer.unknown_flags:
        return tuple(RefusedFlag(flag, _carrier(launch, flag).source) for flag in answer.unknown_flags)
    return tuple(RefusedFlag(arg.flag, arg.source) for arg in launch.args if arg.flag in answer.rejected_option)


def check(launch: Launch, python: str, env: Mapping[str, str], *, cwd: Path) -> FlagRefusal | None:
    """Run the installed parser over ``launch``'s server flags in ``python`` with ``env``, started in ``cwd``; ``None``
    when it accepts.

    Raises:
        ParserCheckError: The framework has no parser, or the parser's answer cannot be read.
        FrameworkProbeError: The serving interpreter could not run the parser.
    """
    framework = launch.framework.name
    probe = parser_probe(framework, [token for arg in launch.args for token in arg.tokens()])
    if probe is None:
        raise ParserCheckError(f"{framework.value} has no server parser to hold the launch to")
    answer = read_answer(run_probe(python, probe.program, probe.args, env, cwd=cwd))
    if answer.status is ParseStatus.ACCEPTED:
        return None
    return FlagRefusal(flags=_refused_flags(launch, answer), detail=answer.detail)


__all__ = [
    "FlagRefusal",
    "ParseStatus",
    "ParserAnswer",
    "ParserCheckError",
    "ParserProbe",
    "RefusedFlag",
    "check",
    "parser_probe",
    "read_answer",
]
