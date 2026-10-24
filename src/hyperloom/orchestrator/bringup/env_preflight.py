# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Name the fault in the host that no patch to the model could repair."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.bringup import BootObservation, Excerpt, LadderStage, redact
from hyperloom.inference_optimizer.breakdown.stop_reasons import ENV_FAULT
from hyperloom.orchestrator.bringup.argv_preflight import OK

#: Names this module's observations in downstream artifacts.
PRODUCER = "preflight.environment"

#: The stream name recorded on the excerpt, so a reader can tell a preflight
#: record from a server log.
STREAM = "env_preflight"

#: Reserved for a condition no round could change: a checkpoint path that does
#: not exist on this host. :data:`OK` is the argv preflight's, shared so the two
#: read alike.
FAULT = "fault"

#: The checkpoint the config names is a filesystem path that resolves to nothing.
CHECKPOINT_UNRESOLVED = "checkpoint_unresolved"

#: Where each fault sits on the boot ladder, so an environment record and a real
#: boot observation of the same wall are comparable.
_FAULT_STAGE: Mapping[str, LadderStage] = {
    CHECKPOINT_UNRESOLVED: LadderStage.CONFIG_VALIDATE,
}


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


def env_fault_observation(verdict: EnvVerdict, *, session_dir: Path | None = None) -> BootObservation:
    """Return the boot observation for a host that cannot run this round.

    Args:
        verdict: The faulting verdict.
        session_dir: Session root, redacted out of the recorded text.

    Returns:
        BootObservation: The observation to persist.
    """
    roots = (str(session_dir),) if session_dir else ()
    stage = _FAULT_STAGE.get(verdict.fault, LadderStage.PROCESS_START)
    body = "\n".join(part for part in (f"environment fault ({verdict.fault})", verdict.detail) if part)
    excerpt = Excerpt(
        text=redact(body, roots=roots),
        stream=STREAM,
        byte_start=0,
        byte_end=len(body.encode("utf-8", "replace")),
    )
    return BootObservation(
        producer=PRODUCER,
        stage_reached=stage,
        stage_failed=stage,
        matched_marker=verdict.fault,
        failure_text=excerpt.text,
        excerpt=excerpt,
        evidence_ref=STREAM,
        env_fault=verdict.fault,
    )


def is_env_fault(observation: BootObservation | None) -> bool:
    """True when ``observation`` names one of the faults that end a run."""
    return observation is not None and observation.env_fault in _FAULT_STAGE


__all__ = [
    "CHECKPOINT_UNRESOLVED",
    "ENV_FAULT",
    "FAULT",
    "OK",
    "PRODUCER",
    "EnvVerdict",
    "check_environment",
    "env_fault_observation",
    "is_env_fault",
]
