# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every option of one forge-fuse run, grouped by concern, with no command line attached."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HarnessNoiseProbe:
    """The hidden diagnostic that repeats one kernel-validation harness and reports its variance."""

    harness_noise_path: str
    harness_noise_repeat: int
    harness_noise_env: tuple[str, ...]


@dataclass(frozen=True)
class DecodeSampling:
    """The decode batch shapes are taken at, and the decode steps the trace captured."""

    decode_batch: int
    decode_steps: int


@dataclass(frozen=True)
class DiscoveryChoice:
    """How recipes are discovered, and the kernel an anchored discovery fuses around."""

    discover_mode: str
    fuse_kernel: str


@dataclass(frozen=True)
class RunStages:
    """Which stages run: dry run, authoring, validation, and whether every confirmed pattern is fused at once."""

    dry_run: bool
    author: bool
    validate: bool
    fuse_all_confirmed: bool


@dataclass(frozen=True)
class FuseAgentOptions:
    """The agent provider, sandbox, model and turn cap for discovery and authoring."""

    agent_backend: str
    agent_sandbox_mode: str
    llm_model: Optional[str]
    max_turns: int


@dataclass(frozen=True)
class SmokeOptions:
    """The A/B decode lengths, and the extra serving args the smoke launch needs."""

    ab_isl: int
    ab_osl: int
    server_extra: str


@dataclass(frozen=True)
class ServingLayout:
    """How the session shards and sizes the model: TP, attention TP, KV block size and max model length."""

    tp: int
    attn_tp_size: int
    block_size: int
    max_model_len: int


@dataclass(frozen=True)
class FuseRequest:
    """Every option of one ``forge-fuse`` invocation, grouped by concern."""

    trace_path: str
    model_path: str
    framework: str
    output_dir: str
    noise: HarnessNoiseProbe
    framework_root: str
    repo_scope: bool
    decode: DecodeSampling
    discovery: DiscoveryChoice
    stages: RunStages
    gpu: str
    agent: FuseAgentOptions
    max_recipes: int
    smoke: SmokeOptions
    gpu_arch: str
    layout: ServingLayout
    verbose: bool
