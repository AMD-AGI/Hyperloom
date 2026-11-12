# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The benchmark client the rule gate measures every test with, for an agent to measure its own A/Bs alike.

On an InferenceX workload it is inference_suite's fixed client on the canonical measurement's workload, as the
toolbox's rendered canonical configuration names it, run from the staged InferenceX the gate's round runs: the warmup
pass on its own seed, then the measured pass, each built by the function a round builds its passes with. A client of
the agent's own sends other prompts, which route MoE tokens otherwise, so its A/Bs can rank kernels the other way round.

On an AgentX workload it is the gate's round itself, the trace replay of the canonical point on the session's frozen
base and staged tools, run by ``python -m inference_suite run`` against the agent's server, which serves a launch of
its own (:func:`~hyperloom.orchestrator.actions.executors.external_agentx_config`). Its ``report.json`` is the gate's
reading of that server, and the agent needs no client or trace of its own.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import yaml

import inference_suite
from hyperloom.common.benchmark_config import is_agentx
from hyperloom.orchestrator.actions.executors import external_agentx_config, fixed_workload, staged_tool_cache
from hyperloom_common.shared_storage import make_shared_dirs, write_shared_text
from inference_suite import (
    ClientPass,
    ClientTarget,
    ClientTool,
    client_argv,
    fixed_client_env,
    inferencex_root,
    warmup_pass,
)
from inference_suite.contracts import FixedWorkload, SuiteConfig

#: What the gate's fixed client runs under: the image's interpreter.
PYTHON = "python3"
#: The file the measured pass saves its result in, under the result directory.
RESULT_FILE = "inferencex_result.json"
#: The host the gate's client reaches an agent's server on.
HOST = "0.0.0.0"
#: Beside the run directory an AgentX client writes into, the config it runs from: a run directory starts empty.
CONFIG_SUFFIX = ".config.yaml"


class GateClient(Protocol):
    """The gate's client for an agent."""

    @property
    def seconds(self) -> float:
        """How long one measurement may take; ``0`` when the agent's own cap bounds it."""

    def stage(self, port: int, result_dir: Path) -> str:
        """Write what measuring a server on ``port`` as the gate does needs, and return the shell command that measures
        it, writing its result in ``result_dir``."""


@dataclass(frozen=True)
class FixedGateClient:
    """The workload the gate's fixed client sends: the staged InferenceX root holding the ``infx`` package it runs
    from, the model and the fixed load."""

    package_root: str
    model: str
    workload: FixedWorkload

    @property
    def seconds(self) -> float:
        return 0.0

    def stage(self, port: int, result_dir: Path) -> str:
        """The shell command; it needs nothing written."""

        tool = ClientTool(PYTHON, self.package_root)
        target = ClientTarget(f"http://{HOST}:{port}", self.model, self.model)
        measured = ClientPass(self.workload.seed, self.workload.num_prompts, result_dir / RESULT_FILE)
        env = [f"{name}={value}" for name, value in fixed_client_env(tool).items()]
        passes = [client_pass for client_pass in (warmup_pass(self.workload), measured) if client_pass is not None]
        return " && ".join(shlex.join([*env, *client_argv(tool, target, self.workload, p)]) for p in passes)


@dataclass(frozen=True)
class AgentxGateClient:
    """The gate's AgentX round as a client: ``config``, its replay against a server on :data:`HOST`, whose port each
    agent's command sets."""

    config: SuiteConfig

    @property
    def seconds(self) -> float:
        return self.config.deadlines.round_sec

    def stage(self, port: int, result_dir: Path) -> str:
        """Write the round's config for a server on ``port`` beside ``result_dir``, the round's run directory, and
        return the command that runs it from the interpreter and packages this process runs."""

        config = replace(
            self.config,
            server=replace(self.config.server, port=port),
            paths=replace(self.config.paths, run_dir=str(result_dir)),
        )
        path = result_dir.with_name(f"{result_dir.name}{CONFIG_SUFFIX}")
        make_shared_dirs(path.parent)
        write_shared_text(path, yaml.safe_dump(config.to_document(), sort_keys=False))
        packages = Path(inference_suite.__file__).resolve().parents[1]
        argv = [sys.executable, "-m", "inference_suite", "run", "--config", str(path), "--run-dir", str(result_dir)]
        return shlex.join([f"PYTHONPATH={packages}", *argv])


def read_gate_client(canonical: Path, session_dir: Path, timeout_sec: float) -> GateClient:
    """The gate's client of the rendered canonical configuration at ``canonical``: its fixed client, or for an AgentX
    one its replay on the base the session at ``session_dir`` froze, bounded by ``timeout_sec``, its server's port and
    run directory left for each agent's command to set.

    Raises:
        SuiteRoundRefused: For an AgentX one, when the session froze no base, staged no tools, or the replay cannot
            be built from the canonical point.
    """

    document = yaml.safe_load(canonical.read_text(encoding="utf-8"))
    bench = document["benchmark"]
    if not is_agentx(bench):
        return FixedGateClient(str(inferencex_root(staged_tool_cache())), str(bench["model"]), fixed_workload(document))
    url = f"http://{HOST}"
    return AgentxGateClient(external_agentx_config(session_dir, document, url, session_dir, timeout_sec))


__all__ = ["AgentxGateClient", "FixedGateClient", "GateClient", "read_gate_client"]
