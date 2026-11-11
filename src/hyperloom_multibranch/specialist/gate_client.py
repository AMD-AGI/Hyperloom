# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The benchmark client the rule gate measures every InferenceX test with, for an agent to measure its own A/Bs alike.

It is inference_suite's fixed client on the canonical measurement's workload, as the toolbox's rendered canonical
configuration names it, run from the staged InferenceX the gate's round runs: the warmup pass on its own seed, then the
measured pass, each built by the function a round builds its passes with. A client of the agent's own sends other
prompts, which route MoE tokens otherwise, so its A/Bs can rank kernels the other way round.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

import yaml

from hyperloom.common.benchmark_config import is_agentx
from hyperloom.orchestrator.actions.executors import fixed_workload, staged_tool_cache
from inference_suite import (
    ClientPass,
    ClientTarget,
    ClientTool,
    client_argv,
    fixed_client_env,
    inferencex_root,
    warmup_pass,
)
from inference_suite.contracts import FixedWorkload

#: What the gate's fixed client runs under: the image's interpreter.
PYTHON = "python3"
#: The file the measured pass saves its result in, under the result directory.
RESULT_FILE = "inferencex_result.json"
#: The host the gate's client reaches an agent's server on.
HOST = "0.0.0.0"


@dataclass(frozen=True)
class FixedGateClient:
    """The workload the gate's fixed client sends: the staged InferenceX root holding the ``infx`` package it runs
    from, the model and the fixed load."""

    package_root: str
    model: str
    workload: FixedWorkload

    def stage(self, port: int, result_dir: Path) -> str:
        """The shell command that measures a server on ``port`` as the gate does, writing its result in
        ``result_dir``; it needs nothing written."""

        tool = ClientTool(PYTHON, self.package_root)
        target = ClientTarget(f"http://{HOST}:{port}", self.model, self.model)
        measured = ClientPass(self.workload.seed, self.workload.num_prompts, result_dir / RESULT_FILE)
        env = [f"{name}={value}" for name, value in fixed_client_env(tool).items()]
        passes = [client_pass for client_pass in (warmup_pass(self.workload), measured) if client_pass is not None]
        return " && ".join(shlex.join([*env, *client_argv(tool, target, self.workload, p)]) for p in passes)


def read_gate_client(canonical: Path) -> FixedGateClient | None:
    """The gate's fixed client of the rendered canonical configuration at ``canonical``; ``None`` for an AgentX one,
    which replays a trace instead."""

    document = yaml.safe_load(canonical.read_text(encoding="utf-8"))
    bench = document["benchmark"]
    if is_agentx(bench):
        return None
    return FixedGateClient(str(inferencex_root(staged_tool_cache())), str(bench["model"]), fixed_workload(document))


__all__ = ["FixedGateClient", "read_gate_client"]
