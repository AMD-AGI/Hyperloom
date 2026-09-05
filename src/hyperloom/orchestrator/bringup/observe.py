# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The single entry point that turns bring-up streams into a verdict.

Every caller classifies through :func:`observe_bringup` so two attempts are
compared on the same reading. :func:`verdict_of` recovers the same verdict from
an observation persisted earlier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.orchestrator.bringup.persist import LoadedObservation, load_boot_observation

if TYPE_CHECKING:
    from hyperloom.agents.framework.enablement import FailureSignature
    from hyperloom.common.bringup import BootObservation


def session_root(owner: Any) -> Path | None:
    """Return the owning coordinator's session root, when it has one.

    ``owner`` is duck-typed and need not carry a session at all; ``None`` then
    means nothing is redacted and no trees are pinned.
    """
    root = getattr(owner, "session_dir", None)
    return Path(root) if root else None


@dataclass(frozen=True)
class BringupVerdict:
    """One classification of one bring-up, in the two shapes callers need.

    ``observation`` carries the ladder stage, terminal frame and redacted
    excerpt that get recorded and digested; ``signature`` is the enablement rule
    signature the bridge search and the runnable gate consume.
    """

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
    """Recover a verdict from an observation that was persisted earlier.

    The signature is re-derived from the observation's own excerpt, which was
    materialised at capture time.
    """
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

    Wrapper text classifies to a different digest than the recorded observation
    would for the same failure, so the returned load result says which reading
    was used and, when nothing was recorded, why.

    Args:
        observation_path: The artifact path the round recorded; may be empty.
        wrapper_text: Launcher-side text to classify when nothing was recorded.
        session_dir: Session root, for the second classification.
    """
    loaded = load_boot_observation(observation_path)
    if loaded.observation is not None:
        return verdict_of(loaded.observation), loaded
    return observe_bringup(wrapper_stderr=wrapper_text, session_dir=session_dir), loaded


__all__ = [
    "BringupVerdict",
    "observe_bringup",
    "recorded_verdict",
    "session_root",
    "verdict_of",
]
