# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The single entry point that turns bring-up streams into a verdict."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.bringup import BootObservation, failure_digest
from hyperloom.orchestrator.bringup.persist import LoadedObservation, load_boot_observation

if TYPE_CHECKING:
    from hyperloom.agents.framework.enablement import FailureSignature


def session_root(owner: Any) -> Path | None:
    """Return the owning coordinator's session root, when it has one."""
    root = getattr(owner, "session_dir", None)
    return Path(root) if root else None


@dataclass(frozen=True)
class BringupVerdict:
    """One classification of one bring-up, in the two shapes callers need."""

    observation: "BootObservation"
    signature: "FailureSignature"


def observe_bringup(
    *,
    server_log: str = "",
    server_elapsed_sec: float = 0.0,
    wrapper_stderr: str = "",
    wrapper_stdout: str = "",
    session_dir: Path | None = None,
) -> BringupVerdict:
    """Classify one bring-up's streams into a single verdict.

    Args:
        server_log: The server child's own log text, consulted first.
        server_elapsed_sec: Seconds the server child ran, on its own clock.
        wrapper_stderr: Launcher stderr, read only when the server log is empty.
        wrapper_stdout: Launcher stdout, the last stream tried.
        session_dir: Session root, redacted out of the excerpt and used to
            resolve the pinned trees frames are normalised against.

    Returns:
        BringupVerdict: The observation and the signature of the stream it
        selected.
    """
    from hyperloom.agents.framework.enablement import classify_failure

    from hyperloom.orchestrator.bringup.ladder import (
        SERVER_LOG,
        WRAPPER_STDERR,
        WRAPPER_STDOUT,
        classify,
    )
    from hyperloom.orchestrator.bringup.trees import read_trees

    observation = classify(
        server_log=server_log,
        server_elapsed_sec=server_elapsed_sec,
        wrapper_stderr=wrapper_stderr,
        wrapper_stdout=wrapper_stdout,
        trees=read_trees(session_dir) if session_dir is not None else (),
        session_root=str(session_dir) if session_dir is not None else "",
    )
    streams = {
        SERVER_LOG: server_log,
        WRAPPER_STDERR: wrapper_stderr,
        WRAPPER_STDOUT: wrapper_stdout,
    }
    # An unset ``evidence_ref`` means no stream carried anything; any other
    # value names one of the slots above.
    evidence = streams[observation.evidence_ref] if observation.evidence_ref else ""
    return BringupVerdict(observation=observation, signature=classify_failure(evidence))


def verdict_of(observation: "BootObservation") -> BringupVerdict:
    """Recover a verdict from an observation that was persisted earlier."""
    from hyperloom.agents.framework.enablement import classify_failure

    excerpt = observation.excerpt
    return BringupVerdict(
        observation=observation,
        signature=classify_failure(excerpt.text if excerpt is not None else ""),
    )


def recorded_verdict(
    observation_path: str | Path | None,
    *,
    wrapper_text: str = "",
    session_dir: Path | None = None,
) -> tuple[BringupVerdict, LoadedObservation]:
    """Recover a round's verdict from what it recorded, or from wrapper text.

    Args:
        observation_path: The artifact path the round recorded; may be empty.
        wrapper_text: Launcher-side text to classify when nothing was recorded.
        session_dir: Session root, for the second classification.
    """
    loaded = load_boot_observation(observation_path)
    if loaded.observation is not None:
        return verdict_of(loaded.observation), loaded
    return observe_bringup(wrapper_stderr=wrapper_text, session_dir=session_dir), loaded


def stage_of(observation: BootObservation | None) -> int:
    """Return how far up the ladder ``observation`` got, as a stage value."""
    if observation is None:
        return 0
    failed = observation.stage_failed
    return int(max(observation.stage_reached, failed if failed is not None else observation.stage_reached))


def digest_of(observation: BootObservation | None) -> str:
    """Return the failure digest of the wall ``observation`` hit."""
    if observation is None or observation.stage_failed is None:
        return ""
    return failure_digest(observation)


def round_advanced(before: BootObservation | None, after: BootObservation | None) -> bool:
    """Whether the boot after a patch is worth keeping the patch for.

    Args:
        before: The observation the previous round recorded.
        after: The observation this round recorded.

    Returns:
        bool: ``True`` when the boot reached a deeper stage, or stopped at the
        same one for a digest ``before`` did not carry. A boot that got less far
        is never an advance.
    """
    if after is None:
        return False
    reached, previous = stage_of(after), stage_of(before)
    if reached != previous:
        return reached > previous
    digest = digest_of(after)
    return bool(digest) and digest != digest_of(before)


__all__ = [
    "BringupVerdict",
    "digest_of",
    "observe_bringup",
    "recorded_verdict",
    "round_advanced",
    "session_root",
    "stage_of",
    "verdict_of",
]
