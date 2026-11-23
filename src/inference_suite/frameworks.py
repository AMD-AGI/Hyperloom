# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The inference frameworks a round can run, and the launch facts each one fixes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, last_alias_value


class Framework(str, Enum):
    SGLANG = "sglang"
    VLLM = "vllm"
    ATOM = "atom"
    XDIT = "xdit"
    LLAMACPP = "llamacpp"
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
class DataParallelFlag:
    """The server option that sets the data-parallel size, every argparse spelling of it.

    Each replica takes ``tp`` cards of its own unless a ``shares_tp`` switch is on: then the replicas split the TP
    ranks' cards (sglang's DP attention).
    """

    aliases: tuple[str, ...]
    shares_tp: tuple[str, ...] = ()

    def replicas(self, args: Sequence[str]) -> int:
        """How many sets of ``tp`` cards ``args`` occupy."""
        occurrences = argv_flag_occurrences(args)
        if {name for name, _ in occurrences} & set(map(canonical_flag_name, self.shares_tp)):
            return 1
        return int(last_alias_value(occurrences, frozenset(map(canonical_flag_name, self.aliases))) or 1)


@dataclass(frozen=True)
class RequestSlotsFlag:
    """The server option that sets how many requests it runs at once, every argparse spelling of it; a request past
    that many waits for a slot to free."""

    aliases: tuple[str, ...]

    def slots(self, args: Sequence[str]) -> int | None:
        """The slots ``args`` set; ``None`` when they set none or a non-positive count, which leaves it to the
        server."""
        value = last_alias_value(argv_flag_occurrences(args), frozenset(map(canonical_flag_name, self.aliases)))
        try:
            count = int(value or 0)
        except ValueError:
            return None
        return count if count > 0 else None


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

    def guarded(self) -> tuple[str, ...]:
        """The flags the runner's guard sets on every round: the model, the address and the served name."""
        return tuple(flag for flag in (self.model, self.host, self.port, self.served_model_name) if flag)


@dataclass(frozen=True)
class ServerCommand:
    """How a framework's server starts.

    ``entrypoint`` is the command the recipes spell, resolved through ``PATH``, which a base records as provenance.
    ``module`` runs that same server under a named interpreter, so the server runs in the install the version probe
    and the parser check read; ``None`` for a native server, which runs as its entrypoint and is probed as one.
    """

    entrypoint: tuple[str, ...]
    module: tuple[str, ...] | None

    @property
    def native(self) -> bool:
        """Whether the server is a native program rather than a module of the serving interpreter."""
        return self.module is None

    def argv(self, python: str) -> tuple[str, ...]:
        """The server command under ``python``; a native server runs as its entrypoint whatever ``python`` is."""
        return self.entrypoint if self.module is None else (python, *self.module)


class ModelFileError(ValueError):
    """A model directory does not hold exactly one model file the server loads."""


@dataclass(frozen=True)
class FrameworkFacts:
    """What one framework fixes about a launch.

    ``python_package`` is the importable package it installs as, ``None`` when it ships none. A scriptable
    framework has no ``server_flags``; a framework with no ``expert_parallel`` option runs no expert parallelism,
    and one with no ``data_parallel`` option runs one replica.
    ``parser_source`` is Python source defining ``_build_parser()``, run in the serving interpreter so an argv is
    judged by the parser that would refuse it; ``None`` when no server parser can be reached. ``metrics_flag`` is the
    switch without which the server serves no ``/metrics``; ``None`` when it serves them by default.
    ``ready_timeout_env`` is the variable, in seconds, after which the server gives up on its own start and exits;
    ``None`` when it waits for its start without a limit of its own. ``model_file`` is the suffix of the one file of a
    model directory the server loads, ``None`` when it loads the directory. ``request_slots`` is the option capping
    the requests the server runs at once, ``None`` when it has none that a load past it only queues behind.
    """

    kind: FrameworkKind
    python_package: str | None
    server_flags: ServerFlags | None = None
    server_command: ServerCommand | None = None
    expert_parallel: ExpertParallelFlag | None = None
    data_parallel: DataParallelFlag | None = None
    parser_source: str | None = None
    metrics_flag: str | None = None
    ready_timeout_env: str | None = None
    model_file: str | None = None
    request_slots: RequestSlotsFlag | None = None

    def served_model(self, model_path: str) -> str:
        """What the server's model flag names for the model at ``model_path``: the path itself, or the one
        ``model_file`` in it when ``model_path`` is a directory.

        Raises:
            ModelFileError: When the directory holds no such file, or more than one model.
        """
        directory = Path(model_path)
        if self.model_file is None or not directory.is_dir():
            return model_path
        files = sorted(path.name for path in directory.iterdir() if path.name.endswith(self.model_file))
        if len(files) != 1:
            found = ", ".join(files) or "none"
            raise ModelFileError(f"{model_path} must hold one *{self.model_file} model to serve, found {found}")
        return str(directory / files[0])

    def cards(self, tp: int, args: Sequence[str]) -> int:
        """The GPUs a launch at ``tp`` running ``args`` occupies: ``tp`` for each data-parallel replica that holds
        cards of its own."""
        return tp * (1 if self.data_parallel is None else self.data_parallel.replicas(args))


# vLLM 0.30 moved the serve parser from ``entrypoints.openai`` to ``entrypoints.launchers``, and 0.32 nightlies moved
# it on to ``entrypoints.common``; newest first, so an install that keeps an older module beside it reads the one
# ``vllm serve`` imports.
_VLLM_PARSER_MODULES = (
    "vllm.entrypoints.common.cli_args",
    "vllm.entrypoints.launchers.cli_args",
    "vllm.entrypoints.openai.cli_args",
)
_VLLM_PARSER_SOURCE = (
    "def _build_parser():\n"
    "    import importlib\n"
    f"    for module in {_VLLM_PARSER_MODULES!r}:\n"
    "        try:\n"
    "            make_arg_parser = importlib.import_module(module).make_arg_parser\n"
    "        except ModuleNotFoundError as exc:\n"
    "            if exc.name not in (module, module.rpartition('.')[0]):\n"
    "                raise\n"
    "            missing = exc\n"
    "            continue\n"
    "        break\n"
    "    else:\n"
    "        raise missing\n"
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

# ATOM's server builds its parser inside ``main()``: the engine's arguments, then its own (``--host``,
# ``--server-port`` and more), then it parses and starts. No function returns that parser, so ``main()`` runs until
# it parses and the parser is taken there, before anything starts.
_ATOM_PARSER_SOURCE = (
    "def _build_parser():\n"
    "    import argparse\n"
    "\n"
    "    class _Built(Exception):\n"
    "        def __init__(self, parser):\n"
    "            super().__init__('the ATOM server parser is built')\n"
    "            self.parser = parser\n"
    "\n"
    "    def _take(self, *_args, **_kwargs):\n"
    "        raise _Built(self)\n"
    "\n"
    "    parse_known_args = argparse.ArgumentParser.parse_known_args\n"
    "    argparse.ArgumentParser.parse_known_args = _take\n"
    "    try:\n"
    "        from atom.entrypoints.openai_server import main\n"
    "        main()\n"
    "    except _Built as built:\n"
    "        return built.parser\n"
    "    finally:\n"
    "        argparse.ArgumentParser.parse_known_args = parse_known_args\n"
    "    raise RuntimeError('the ATOM server returned without parsing its arguments')\n"
)


_MODEL_FLAG = "--model"
_HOST_FLAG = "--host"
_PORT_FLAG = "--port"
_SERVED_MODEL_NAME_FLAG = "--served-model-name"
#: The expert-parallel switch vLLM and ATOM share.
_VLLM_STYLE_EP_SWITCH = "--enable-expert-parallel"

_FACTS: dict[Framework, FrameworkFacts] = {
    Framework.SGLANG: FrameworkFacts(
        kind=FrameworkKind.SERVING,
        python_package=Framework.SGLANG.value,
        server_flags=ServerFlags("--model-path", _HOST_FLAG, _PORT_FLAG, _SERVED_MODEL_NAME_FLAG, "--context-length"),
        server_command=ServerCommand(("python3", "-m", "sglang.launch_server"), ("-m", "sglang.launch_server")),
        expert_parallel=ExpertParallelFlag("--ep-size", ("--ep-size", "--expert-parallel-size", "--ep"), True),
        data_parallel=DataParallelFlag(("--data-parallel-size", "--dp-size"), shares_tp=("--enable-dp-attention",)),
        parser_source=_SGLANG_PARSER_SOURCE,
        metrics_flag="--enable-metrics",
        # The warmup request it sends once its HTTP server is up; past it the server kills itself before it reports
        # ready. Unset (-1) means 600 s.
        ready_timeout_env="SGLANG_WARMUP_TIMEOUT",
    ),
    Framework.VLLM: FrameworkFacts(
        kind=FrameworkKind.SERVING,
        python_package=Framework.VLLM.value,
        server_flags=ServerFlags(_MODEL_FLAG, _HOST_FLAG, _PORT_FLAG, _SERVED_MODEL_NAME_FLAG, "--max-model-len"),
        # The ``vllm`` console script is ``vllm.entrypoints.cli.main:main``.
        server_command=ServerCommand(("vllm", "serve"), ("-m", "vllm.entrypoints.cli.main", "serve")),
        expert_parallel=ExpertParallelFlag(_VLLM_STYLE_EP_SWITCH, (_VLLM_STYLE_EP_SWITCH, "-ep"), False),
        data_parallel=DataParallelFlag(("--data-parallel-size", "-dp")),
        parser_source=_VLLM_PARSER_SOURCE,
        # How long the frontend waits for the engine cores to report ready (600 s unset), the Rust frontend's
        # ``engine_ready_timeout_secs`` included.
        ready_timeout_env="VLLM_ENGINE_READY_TIMEOUT_S",
    ),
    Framework.ATOM: FrameworkFacts(
        kind=FrameworkKind.SERVING,
        python_package=Framework.ATOM.value,
        server_flags=ServerFlags(_MODEL_FLAG, _HOST_FLAG, "--server-port", _SERVED_MODEL_NAME_FLAG, "--max-model-len"),
        server_command=ServerCommand(
            ("python3", "-m", "atom.entrypoints.openai_server"), ("-m", "atom.entrypoints.openai_server")
        ),
        expert_parallel=ExpertParallelFlag(_VLLM_STYLE_EP_SWITCH, (_VLLM_STYLE_EP_SWITCH,), False),
        parser_source=_ATOM_PARSER_SOURCE,
    ),
    Framework.LLAMACPP: FrameworkFacts(
        kind=FrameworkKind.SERVING,
        python_package=None,
        server_flags=ServerFlags(_MODEL_FLAG, _HOST_FLAG, _PORT_FLAG, "--alias", "--ctx-size"),
        server_command=ServerCommand(("llama-server",), None),
        metrics_flag="--metrics",
        model_file=".gguf",
        request_slots=RequestSlotsFlag(("-np", "--parallel")),
    ),
    Framework.XDIT: FrameworkFacts(kind=FrameworkKind.SCRIPTABLE, python_package="xfuser"),
    Framework.CUSTOM: FrameworkFacts(kind=FrameworkKind.SCRIPTABLE, python_package=None),
}


def framework_facts(framework: Framework) -> FrameworkFacts:
    """The launch facts of ``framework``."""
    return _FACTS[framework]


__all__ = [
    "DataParallelFlag",
    "ExpertParallelFlag",
    "Framework",
    "FrameworkFacts",
    "FrameworkKind",
    "ModelFileError",
    "RequestSlotsFlag",
    "ServerCommand",
    "ServerFlags",
    "framework_facts",
]
