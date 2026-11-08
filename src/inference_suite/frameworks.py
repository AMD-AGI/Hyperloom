# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The inference frameworks a round can run, and the launch facts each one fixes.

These are the only copy of the facts: Hyperloom's ``framework_registry`` derives its kind, package and
expert-parallel answers from here and keeps only what is Hyperloom's own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Framework(str, Enum):
    SGLANG = "sglang"
    VLLM = "vllm"
    ATOM = "atom"
    XDIT = "xdit"
    #: An operator's own workload.
    CUSTOM = "custom"


class FrameworkKind(str, Enum):
    #: Runs an OpenAI-compatible server that a load client drives.
    SERVING = "serving"
    #: Runs a script with no server.
    SCRIPTABLE = "scriptable"


@dataclass(frozen=True)
class ExpertParallelFlag:
    """The server option that turns on expert parallelism.

    ``aliases`` is every argparse spelling of it. vLLM and ATOM take a switch (``takes_size`` false) and derive the
    size from TP x DP.
    """

    flag: str
    aliases: tuple[str, ...]
    takes_size: bool


@dataclass(frozen=True)
class ServerFlags:
    """The flags that carry a round's runtime facts to the server; ``None`` when the server takes no such flag.

    ``context_length`` sets the context the server serves; without it the server serves the model's native one.
    """

    model: str
    host: str
    port: str
    served_model_name: str | None
    context_length: str


@dataclass(frozen=True)
class FrameworkFacts:
    """What one framework fixes about a launch.

    ``python_package`` is the importable package it installs as, ``None`` when it ships none. A scriptable
    framework has no ``server_flags``; a framework with no ``expert_parallel`` option runs no expert parallelism.
    ``parser_source`` is Python source defining ``_build_parser()``, run in the serving interpreter so an argv is
    judged by the parser that would refuse it; ``None`` when no server parser can be reached. ``metrics_flag`` is the
    switch without which the server serves no ``/metrics``; ``None`` when it serves them by default.
    """

    kind: FrameworkKind
    python_package: str | None
    server_flags: ServerFlags | None = None
    expert_parallel: ExpertParallelFlag | None = None
    parser_source: str | None = None
    metrics_flag: str | None = None


# vLLM 0.30 moved the serve parser from ``entrypoints.openai`` to ``entrypoints.launchers``.
_VLLM_PARSER_SOURCE = (
    "def _build_parser():\n"
    "    try:\n"
    "        from vllm.entrypoints.launchers.cli_args import make_arg_parser\n"
    "    except ModuleNotFoundError as exc:\n"
    "        if exc.name not in ('vllm.entrypoints.launchers', 'vllm.entrypoints.launchers.cli_args'):\n"
    "            raise\n"
    "        from vllm.entrypoints.openai.cli_args import make_arg_parser\n"
    "    try:\n"
    "        from vllm.utils.argparse_utils import FlexibleArgumentParser\n"
    "    except ImportError:\n"
    "        from vllm.utils import FlexibleArgumentParser\n"
    "    return make_arg_parser(FlexibleArgumentParser())\n"
)

_SGLANG_PARSER_SOURCE = (
    "def _build_parser():\n"
    "    import argparse\n"
    "    from sglang.srt.server_args import ServerArgs\n"
    "    parser = argparse.ArgumentParser()\n"
    "    ServerArgs.add_cli_args(parser)\n"
    "    return parser\n"
)

# ATOM v0.1.5's server uses plain argparse; newer releases provide a native parser.
_ATOM_PARSER_SOURCE = (
    "def _build_parser():\n"
    "    import argparse\n"
    "    try:\n"
    "        from atom.utils.arg_parser import FlexibleArgumentParser as parser_cls\n"
    "    except ModuleNotFoundError as exc:\n"
    "        if exc.name not in ('atom.utils', 'atom.utils.arg_parser'):\n"
    "            raise\n"
    "        parser_cls = argparse.ArgumentParser\n"
    "    from atom.model_engine.arg_utils import EngineArgs\n"
    "    parser = parser_cls()\n"
    "    EngineArgs.add_cli_args(parser)\n"
    "    return parser\n"
)


_HOST_FLAG = "--host"
#: The expert-parallel switch vLLM and ATOM share.
_VLLM_STYLE_EP_SWITCH = "--enable-expert-parallel"

_FACTS: dict[Framework, FrameworkFacts] = {
    Framework.SGLANG: FrameworkFacts(
        kind=FrameworkKind.SERVING,
        python_package="sglang",
        server_flags=ServerFlags("--model-path", _HOST_FLAG, "--port", "--served-model-name", "--context-length"),
        expert_parallel=ExpertParallelFlag("--ep-size", ("--ep-size", "--expert-parallel-size", "--ep"), True),
        parser_source=_SGLANG_PARSER_SOURCE,
        metrics_flag="--enable-metrics",
    ),
    Framework.VLLM: FrameworkFacts(
        kind=FrameworkKind.SERVING,
        python_package="vllm",
        server_flags=ServerFlags("--model", _HOST_FLAG, "--port", "--served-model-name", "--max-model-len"),
        expert_parallel=ExpertParallelFlag(_VLLM_STYLE_EP_SWITCH, (_VLLM_STYLE_EP_SWITCH, "-ep"), False),
        parser_source=_VLLM_PARSER_SOURCE,
    ),
    Framework.ATOM: FrameworkFacts(
        kind=FrameworkKind.SERVING,
        python_package="atom",
        server_flags=ServerFlags("--model", _HOST_FLAG, "--server-port", None, "--max-model-len"),
        expert_parallel=ExpertParallelFlag(_VLLM_STYLE_EP_SWITCH, (_VLLM_STYLE_EP_SWITCH,), False),
        parser_source=_ATOM_PARSER_SOURCE,
    ),
    Framework.XDIT: FrameworkFacts(kind=FrameworkKind.SCRIPTABLE, python_package="xfuser"),
    Framework.CUSTOM: FrameworkFacts(kind=FrameworkKind.SCRIPTABLE, python_package=None),
}


def framework_facts(framework: Framework) -> FrameworkFacts:
    """The launch facts of ``framework``."""
    return _FACTS[framework]


__all__ = [
    "ExpertParallelFlag",
    "Framework",
    "FrameworkFacts",
    "FrameworkKind",
    "ServerFlags",
    "framework_facts",
]
