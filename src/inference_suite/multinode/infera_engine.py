# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``infera.engine.sglang`` command one GPU host runs, shared by the PD launch and the pod launcher.

Standard library only: the pod launcher ships this file beside it and imports it as ``infera_engine``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

INFERA_ENGINE = "infera.engine.sglang"
#: The vLLM-only switch SGLang does not take; expert parallelism comes from ``--ep-size``.
VLLM_ONLY_EP_SWITCH = "--enable-expert-parallel"
_BIND_ADDRESS = "0.0.0.0"  # nosec B104 - the engine must bind for pod-to-pod traffic.
_DP_ENABLE_FLAGS = frozenset({"enable-dp-attention", "enable-dp-lm-head"})


@dataclass(frozen=True)
class EngineHost:
    """Where one host's engine sits: the model and its TP and EP sizes, the port it serves, the address it advertises,
    and its rank in a worker of ``nnodes`` hosts joined at ``leader``. ``ep`` at or below one runs no expert
    parallelism."""

    model: str
    tp: int
    port: int
    advertise_host: str
    nnodes: int = 1
    node_rank: int = 0
    leader: str = ""
    dist_init_port: int = 5000
    ep: int = 1


def _flag_name(token: str) -> str:
    """The canonical name of a flag token: no leading dashes, lowercase, underscores as dashes."""
    return token.partition("=")[0].lstrip("-").lower().replace("_", "-")


def _flag_names(tokens: Iterable[str]) -> frozenset[str]:
    """The canonical names of the flag tokens among ``tokens``."""
    return frozenset(_flag_name(token) for token in tokens if token.startswith("-") and token.strip("-"))


def sglang_argv(host: EngineHost, extra: Sequence[str]) -> list[str]:
    """The ``infera.engine.sglang`` argv of ``host``, with ``extra`` (the role's own flags, PD flags and context
    length) after the launcher's flags.

    The launcher's rendezvous, expert-parallel and data-parallel flags are added only where ``extra`` does not already
    set them. A PD leg, one that names a disaggregation mode, skips the server warmup unless ``extra`` skips it.
    """
    names = _flag_names(extra)
    argv = ["python3", "-m", INFERA_ENGINE, "--model-path", host.model, "--tp-size", str(host.tp)]
    if "trust-remote-code" not in names:
        argv.append("--trust-remote-code")
    argv += ["--host", _BIND_ADDRESS, "--port", str(host.port), "--discovery-backend", "kubernetes"]
    argv += ["--advertise-host", host.advertise_host, "--request-transport", "nats"]
    if host.nnodes > 1:
        argv += ["--nnodes", str(host.nnodes), "--node-rank", str(host.node_rank)]
        argv += ["--dist-init-addr", f"{host.leader}:{host.dist_init_port}"]
    if host.ep > 1 and "ep-size" not in names:
        argv += ["--ep-size", str(host.ep)]
    argv += [token for token in extra if token != VLLM_ONLY_EP_SWITCH]
    if names & _DP_ENABLE_FLAGS and "dp-size" not in names and host.tp > 1:
        argv += ["--dp-size", str(host.tp)]
    if "disaggregation-mode" in names and "skip-server-warmup" not in names:
        argv.append("--skip-server-warmup")
    return argv
