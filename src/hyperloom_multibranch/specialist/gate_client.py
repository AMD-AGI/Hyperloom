# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The benchmark client the rule gate measures every InferenceX test with, for an agent to measure its own A/Bs alike.

It is InferenceX's ``benchmark_serving.py`` on the canonical measurement's workload, as the toolbox's rendered canonical
configuration names it, built by the same function the single-lineage flow builds its client with. A client of the
agent's own sends other prompts, which route MoE tokens otherwise, so its A/Bs can rank kernels the other way round.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

import yaml

from hyperloom.common.outside_facts import ENV_CONC, ENV_ISL, ENV_NUM_PROMPTS, ENV_OSL, ENV_RANDOM_RANGE_RATIO
from hyperloom.orchestrator.actions.executors.bypass_engine import ClientRun, build_client_command

#: What the gate's client runs under: the image's interpreter, as the InferenceX benchmark library calls it.
PYTHON = "python3"
#: The canonical variables that make the benchmark scripts pass ``--trust-remote-code`` to the client.
TRUST_REMOTE_CODE_ENVS = ("MAGPIE_TRUST_REMOTE_CODE", "BENCH_TRUST_REMOTE_CODE")


@dataclass(frozen=True)
class GateClient:
    """The workload the gate's client sends: the InferenceX checkout it runs from, the model, the request shape, the
    concurrency and the prompt count (``None``: the client's default for the concurrency)."""

    inferencex_root: str
    model: str
    isl: int
    osl: int
    conc: int
    random_range_ratio: float
    num_prompts: int | None
    trust_remote_code: bool

    @classmethod
    def read(cls, canonical: Path) -> GateClient | None:
        """The client of the rendered canonical configuration at ``canonical``; ``None`` for an AgentX one, which
        replays a trace instead."""

        bench = yaml.safe_load(canonical.read_text(encoding="utf-8"))["benchmark"]
        envs = bench["envs"]
        if ENV_ISL not in envs:
            return None
        prompts = envs.get(ENV_NUM_PROMPTS)
        return cls(
            inferencex_root=str(bench["inferencex_path"]),
            model=str(bench["model"]),
            isl=int(envs[ENV_ISL]),
            osl=int(envs[ENV_OSL]),
            conc=int(envs[ENV_CONC]),
            random_range_ratio=float(envs[ENV_RANDOM_RANGE_RATIO]),
            num_prompts=None if prompts is None else int(prompts),
            trust_remote_code=any(str(envs.get(name)) == "1" for name in TRUST_REMOTE_CODE_ENVS),
        )

    def command(self, port: int, result_dir: Path) -> str:
        """The shell command that measures a server on ``port`` as the gate does, writing its result in
        ``result_dir``; its warmups are the client's default, twice the concurrency, as the gate's are."""

        run = ClientRun(
            inferencex_root=self.inferencex_root,
            python_exe=PYTHON,
            model=self.model,
            base_url=f"http://0.0.0.0:{port}",
            isl=self.isl,
            osl=self.osl,
            conc=self.conc,
            random_range_ratio=self.random_range_ratio,
            result_dir=str(result_dir),
            num_prompts=self.num_prompts,
            trust_remote_code=self.trust_remote_code,
        )
        return shlex.join(build_client_command(run))


__all__ = ["GateClient"]
