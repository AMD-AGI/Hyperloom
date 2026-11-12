# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a PD session asks resolution for: the model, framework, board, workload and each role's hosts."""

from __future__ import annotations

from dataclasses import dataclass

from inference_suite.contracts.base_launch import Workload
from inference_suite.frameworks import Framework


class PdQueryError(ValueError):
    """A PD query breaks its contract."""


@dataclass(frozen=True)
class RoleAsk:
    """One role's hosts, one worker per ``nodes_per_worker`` of them, and the topology the operator fixed (``None``
    where the operator left it to the source)."""

    hosts: tuple[str, ...]
    tp: int | None = None
    ep: int | None = None
    dp_attention: bool | None = None

    def __post_init__(self) -> None:
        if not self.hosts or not all(isinstance(host, str) and host for host in self.hosts):
            raise PdQueryError(f"a role needs at least one host, got {self.hosts!r}")
        for name in ("tp", "ep"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise PdQueryError(f"a role's {name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class PdQuery:
    """A PD session: ``model_id`` as InferenceX names it, ``model_path`` the engines load, the board and GPUs per
    node of every host, the workload (with its sequence shape when fixed), its concurrencies, both roles and the
    router host (empty when the platform runs the router)."""

    model_id: str
    model_path: str
    framework: Framework
    board: str
    gpus_per_node: int
    workload: Workload
    concurrencies: tuple[int, ...]
    prefill: RoleAsk
    decode: RoleAsk
    router_host: str = ""
    isl: int | None = None
    osl: int | None = None
    transfer_backend: str = ""

    def __post_init__(self) -> None:
        if self.framework not in (Framework.SGLANG, Framework.ATOM):
            raise PdQueryError(f"a PD session runs sglang or atom, not {self.framework.value}")
        if self.workload not in (Workload.FIXED, Workload.AGENTX):
            raise PdQueryError(f"a PD session runs a fixed or AgentX workload, not {self.workload.value}")
        if not self.model_id or not self.model_path or not self.board:
            raise PdQueryError("a PD session names its model id, model path and board")
        if not isinstance(self.gpus_per_node, int) or self.gpus_per_node < 1:
            raise PdQueryError(f"gpus_per_node must be a positive integer, got {self.gpus_per_node!r}")
        if not self.concurrencies or not all(isinstance(c, int) and c >= 1 for c in self.concurrencies):
            raise PdQueryError(f"a PD session needs positive concurrencies, got {self.concurrencies!r}")
        fixed = self.workload is Workload.FIXED
        if (self.isl is not None and self.osl is not None) != fixed:
            raise PdQueryError("a fixed workload names its isl and osl, and an AgentX one neither")

    @property
    def model_name(self) -> str:
        """The model's directory name, as the PD default table keys its rows."""
        return self.model_path.rstrip("/").rsplit("/", 1)[-1]

    @property
    def target_concurrency(self) -> int:
        return max(self.concurrencies)


__all__ = ["PdQuery", "PdQueryError", "RoleAsk"]
