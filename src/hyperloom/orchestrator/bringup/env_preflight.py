# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Name the fault in the host that no patch to the model could repair."""

from __future__ import annotations

import os
from dataclasses import dataclass

from hyperloom.inference_optimizer.breakdown.stop_reasons import ENV_FAULT
from hyperloom.orchestrator.bringup.argv_preflight import OK

#: Reserved for a condition no round could change: a checkpoint path that does
#: not exist on this host. :data:`OK` is the argv preflight's, shared so the two
#: read alike.
FAULT = "fault"

#: The checkpoint the config names is a filesystem path that resolves to nothing.
CHECKPOINT_UNRESOLVED = "checkpoint_unresolved"


@dataclass(frozen=True)
class EnvVerdict:
    """What the host said about the round that is about to launch.

    Attributes:
        status: :data:`OK` or :data:`FAULT`.
        fault: Which named fault was found.
        detail: Filesystem evidence.
    """

    status: str
    fault: str = ""
    detail: str = ""

    @property
    def terminal(self) -> bool:
        """bool: True when this verdict must stop the round rather than launch."""
        return self.status == FAULT


def check_environment(*, model: str) -> EnvVerdict:
    """Decide whether the host holds the checkpoint this round would serve.

    Args:
        model: Model path or repository id the round would serve.

    Returns:
        EnvVerdict: A fault when the path does not resolve, else :data:`OK`.
    """
    raw = model.strip()
    if not raw:
        return EnvVerdict(status=OK)
    expanded = os.path.expanduser(raw)
    if not (os.path.isabs(expanded) or raw.startswith(("." + os.sep, ".." + os.sep, "~"))):
        return EnvVerdict(status=OK)
    if os.path.exists(expanded):
        return EnvVerdict(status=OK)
    return EnvVerdict(
        status=FAULT,
        fault=CHECKPOINT_UNRESOLVED,
        detail=f"the checkpoint path {expanded} does not exist on this host",
    )


__all__ = [
    "CHECKPOINT_UNRESOLVED",
    "ENV_FAULT",
    "FAULT",
    "OK",
    "EnvVerdict",
    "check_environment",
]
