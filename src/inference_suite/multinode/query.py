# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a multi-node session asks resolution for: the model, framework, board, workload, mode and each role's hosts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from inference_suite.contracts.base_launch import Workload
from inference_suite.frameworks import Framework
from inference_suite.multinode.record import MODE_ROLES, FrontendType, PdMode, Role

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
        if self.frontend not in (None, *SERVED_FRONTENDS[self.framework]):
            served = ", ".join(frontend.value for frontend in SERVED_FRONTENDS[self.framework])
            raise PdQueryError(f"a {self.framework.value} session serves through {served}, not {self.frontend.value}")
        if set(self.roles) != set(MODE_ROLES[self.mode]):
            raise PdQueryError(f"a {self.mode.value} session asks for {[role.value for role in MODE_ROLES[self.mode]]}")
        fixed = self.workload is Workload.FIXED
        if (self.isl is not None and self.osl is not None) != fixed:
            raise PdQueryError("a fixed workload names its isl and osl, and an AgentX one neither")

    @property
    def prefill(self) -> RoleAsk:
        """The prefill ask of a disaggregated query."""
        return self.roles[Role.PREFILL]

    @property
    def decode(self) -> RoleAsk:
        """The decode ask of a disaggregated query."""
        return self.roles[Role.DECODE]

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
