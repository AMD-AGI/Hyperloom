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
    """

    kind: FrameworkKind
    python_package: str | None
    server_flags: ServerFlags | None = None
    expert_parallel: ExpertParallelFlag | None = None


_FACTS: dict[Framework, FrameworkFacts] = {
    Framework.SGLANG: FrameworkFacts(
        kind=FrameworkKind.SERVING,
        python_package="sglang",
        server_flags=ServerFlags("--model-path", "--host", "--port", "--served-model-name", "--context-length"),
        expert_parallel=ExpertParallelFlag("--ep-size", ("--ep-size", "--expert-parallel-size", "--ep"), True),
    ),
    Framework.VLLM: FrameworkFacts(
        kind=FrameworkKind.SERVING,
        python_package="vllm",
        server_flags=ServerFlags("--model", "--host", "--port", "--served-model-name", "--max-model-len"),
        expert_parallel=ExpertParallelFlag("--enable-expert-parallel", ("--enable-expert-parallel", "-ep"), False),
    ),
    Framework.ATOM: FrameworkFacts(
        kind=FrameworkKind.SERVING,
        python_package="atom",
        server_flags=ServerFlags("--model", "--host", "--server-port", None, "--max-model-len"),
        expert_parallel=ExpertParallelFlag("--enable-expert-parallel", ("--enable-expert-parallel",), False),
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
