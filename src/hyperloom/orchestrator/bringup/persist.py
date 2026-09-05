# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write one bring-up attempt's observation to the session, and read it back."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.bringup import BootObservation
from hyperloom.inference_optimizer.session.session_paths import reports_dir
from hyperloom.orchestrator.bringup.ladder import observation_summary
from hyperloom.orchestrator.bringup.trees import path_slug

log = logging.getLogger(__name__)

#: No path was recorded for this side of the comparison.
DEGRADED_NO_PATH = "no_observation_path"

#: A path was recorded but nothing could be read back from it.
DEGRADED_UNREADABLE = "observation_unreadable"


@dataclass(frozen=True)
class LoadedObservation:
    """The result of asking for a persisted observation.

    Attributes:
        observation: The observation, or ``None`` when it could not be loaded.
        path: The path that was asked for.
        degraded: Empty when ``observation`` is set; otherwise
            :data:`DEGRADED_NO_PATH` or :data:`DEGRADED_UNREADABLE`.
    """

    observation: BootObservation | None
    path: str = ""
    degraded: str = ""


def _observation_path(session_dir: Path, output_dir: Path, attempt: int) -> Path:
    """Return ``<session_dir>/reports/bringup/<slot>-<digest>-<attempt>.json``.

    ``output_dir`` is the per-round workspace slot the attempt ran in and
    ``attempt`` its index within that slot; the slug's digest separates two
    slots that share a name.
    """
    slug = path_slug(str(output_dir), fallback="round")
    return reports_dir(session_dir) / "bringup" / f"{slug}-{attempt:03d}.json"


def write_boot_observation(
    observation: BootObservation,
    *,
    session_dir: Path,
    output_dir: Path,
    attempt: int,
) -> str:
    """Persist one attempt's boot observation, returning the artifact path.

    Args:
        observation: The ``BootObservation`` to record.
        session_dir: The session root directory.
        output_dir: The per-round workspace slot the attempt ran in.
        attempt: The attempt's index within that slot.

    Returns:
        str: The path the observation was written to.

    Raises:
        OSError: When the artifact cannot be written.
    """
    from hyperloom.common.io import atomic_write_json

    target = _observation_path(session_dir, output_dir, attempt)
    atomic_write_json(target, observation_summary(observation), trailing_newline=True)
    return str(target)


def load_boot_observation(path: str | Path | None) -> LoadedObservation:
    """Read back an observation written by :func:`write_boot_observation`.

    Args:
        path: The artifact path recorded alongside the attempt; empty when the
            attempt recorded none.

    Returns:
        LoadedObservation: The observation, or a named degraded outcome.
    """
    text = "" if path is None else str(path).strip()
    if not text:
        return LoadedObservation(observation=None, degraded=DEGRADED_NO_PATH)
    try:
        raw = json.loads(Path(text).read_text(encoding="utf-8"))
        observation = BootObservation.from_dict(raw)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        log.warning("bringup: could not read boot observation %s (%s)", text, exc)
        return LoadedObservation(observation=None, path=text, degraded=DEGRADED_UNREADABLE)
    return LoadedObservation(observation=observation, path=text)


__all__ = [
    "DEGRADED_NO_PATH",
    "DEGRADED_UNREADABLE",
    "LoadedObservation",
    "load_boot_observation",
    "write_boot_observation",
]
