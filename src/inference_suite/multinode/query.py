# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a multi-node session asks resolution for: the model, framework, board, workload, mode and each role's hosts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from inference_suite.contracts.base_launch import Workload
from inference_suite.frameworks import Framework
from inference_suite.multinode.record import FrontendType, PdMode, Role

#: The router each framework serves through when the session names none.
DEFAULT_FRONTEND = {Framework.SGLANG: FrontendType.SGLANG_ROUTER, Framework.ATOM: FrontendType.ATOMESH}
#: The frontends a session may name, by framework.
SERVED_FRONTENDS = {
    Framework.SGLANG: (FrontendType.SGLANG_ROUTER, FrontendType.ATOMESH, FrontendType.INFERA),
    Framework.ATOM: (FrontendType.SGLANG_ROUTER, FrontendType.ATOMESH),
}


class PdQueryError(ValueError):
    """A PD query breaks its contract."""


@dataclass(frozen=True)
class RoleAsk:
    """One role's hosts, one worker per ``nodes_per_worker`` of them, and the topology the operator fixed (``None``
    where the operator left it to the source). The node list boundary validates them."""

    hosts: tuple[str, ...]
    tp: int | None = None
    ep: int | None = None
    dp_attention: bool | None = None


@dataclass(frozen=True)
class PdQuery:
    """A multi-node session: ``model_id`` as InferenceX names it, ``model_path`` the engines load, the board and GPUs
    per node of every host, the workload (with its sequence shape when fixed), its concurrencies, the ask of each role
    of its mode, the router host (empty when the platform runs the router) and the router it runs (``None`` for the
    framework's)."""

    model_id: str
    model_path: str
    framework: Framework
    board: str
    gpus_per_node: int
    workload: Workload
    concurrencies: tuple[int, ...]
    roles: Mapping[Role, RoleAsk]
    router_host: str = ""
    isl: int | None = None
    osl: int | None = None
    transfer_backend: str = ""
    frontend: FrontendType | None = None
    mode: PdMode = PdMode.DISAGGREGATED

    def __post_init__(self) -> None:
        if self.workload not in (Workload.FIXED, Workload.AGENTX):
            raise PdQueryError(f"a PD session runs a fixed or AgentX workload, not {self.workload.value}")

    @property
    def model_name(self) -> str:
        """The model's directory name, as the PD default table keys its rows."""
        return self.model_path.rstrip("/").rsplit("/", 1)[-1]

    @property
    def served_frontend(self) -> FrontendType:
        """The router the session runs: the one it names, else the framework's."""
        return self.frontend or DEFAULT_FRONTEND[self.framework]

    @property
    def target_concurrency(self) -> int:
        return max(self.concurrencies)


__all__ = ["DEFAULT_FRONTEND", "SERVED_FRONTENDS", "PdQuery", "PdQueryError", "RoleAsk"]
