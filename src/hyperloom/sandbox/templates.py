# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local sandbox templates and the names the platform owns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, get_args

from hyperloom.common.outside_facts import ROCR_VISIBLE_DEVICES_ENV
from hyperloom.sandbox.socket import Role

__all__ = ["RESERVED_ENV_NAMES", "LocalTemplate"]

#: Environment names only the platform sets; a caller that passes one is refused.
RESERVED_ENV_NAMES = frozenset(
    {
        "NODE_NAME",
        "E2B_SANDBOX_ID",
        ROCR_VISIBLE_DEVICES_ENV,
        "HYPERLOOM_SIGNING_KEY_FILE",
        "HYPERLOOM_SANDBOX_ROOT",
    }
)

_CONFIG_KEYS = frozenset({"role", "command", "gpus", "sealed", "signing", "envs"})


@dataclass(frozen=True)
class LocalTemplate:
    """One kind of sandbox the local provider can create.

    Attributes:
        name: The template name callers pass to ``create``.
        role: The only role allowed to create it.
        command: Startup argv launched at create as its own process group
            (empty: nothing is started).
        gpus: How many cards each sandbox leases from the pool.
        sealed: Callers may only write ``/in/`` and read ``/out/``, never run commands.
        signing: Generate a per-sandbox signing identity (sealed templates only).
        envs: Fixed environment for the startup command and commands.
    """

    name: str
    role: Role
    command: tuple[str, ...] = ()
    gpus: int = 0
    sealed: bool = False
    signing: bool = False
    envs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in get_args(Role):
            raise ValueError(f"template {self.name!r}: unknown role {self.role!r}")
        if self.role == "measurement" and not self.sealed:
            raise ValueError(f"template {self.name!r}: a measurement template must be sealed")
        if self.signing and not self.sealed:
            raise ValueError(f"template {self.name!r}: only a sealed template may carry a signing identity")
        if self.gpus < 0:
            raise ValueError(f"template {self.name!r}: gpus must be >= 0")
        reserved = RESERVED_ENV_NAMES & set(self.envs)
        if reserved:
            raise ValueError(f"template {self.name!r}: envs name platform variables {sorted(reserved)}")

    @classmethod
    def from_mapping(cls, name: str, config: Mapping[str, Any]) -> LocalTemplate:
        """Build a template from its entry in the socket's ``config.templates``.

        Args:
            name: The template name.
            config: Keys ``role`` (required), ``command``, ``gpus``, ``sealed``,
                ``signing``, ``envs``.

        Returns:
            LocalTemplate: The validated template.

        Raises:
            ValueError: On an unknown key, a missing role or an invalid combination.
        """
        unknown = set(config) - _CONFIG_KEYS
        if unknown:
            raise ValueError(f"template {name!r}: unknown keys {sorted(unknown)}")
        if "role" not in config:
            raise ValueError(f"template {name!r}: role is required")
        command: Sequence[str] = config.get("command", ())
        if isinstance(command, str):
            raise ValueError(f"template {name!r}: command is an argv list, not a string")
        return cls(
            name=name,
            role=config["role"],
            command=tuple(command),
            gpus=int(config.get("gpus", 0)),
            sealed=bool(config.get("sealed", False)),
            signing=bool(config.get("signing", False)),
            envs=dict(config.get("envs", {})),
        )
