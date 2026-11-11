# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Record a round whose server argv the installed parser refused as a boot observation at the parse rung."""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.bringup import BootObservation, Excerpt, LadderStage, redact
from hyperloom.inference_optimizer.breakdown.stop_reasons import ARGV_INVALID

#: Names the refusal's observations in downstream artifacts; archived observations carry the same value.
PRODUCER = "preflight.argv"

#: The stream recorded on the excerpt, so a reader can tell parser output from a server log.
STREAM = "argv_preflight"


def argv_invalid_observation(detail: str, *, session_dir: Path | None = None) -> BootObservation:
    """Return the boot observation of a round the installed parser refused before launch.

    Args:
        detail: What the parser said, as the round's report recorded it.
        session_dir: Session root, redacted out of the recorded text.
    """
    roots = (str(session_dir),) if session_dir else ()
    body = "\n".join(part for part in ("server argv rejected by the installed parser", detail.strip()) if part)
    excerpt = Excerpt(
        text=redact(body, roots=roots),
        stream=STREAM,
        byte_start=0,
        byte_end=len(body.encode("utf-8", "replace")),
    )
    return BootObservation(
        producer=PRODUCER,
        stage_reached=LadderStage.ARGV_PARSE,
        stage_failed=LadderStage.ARGV_PARSE,
        matched_marker=ARGV_INVALID,
        failure_text=excerpt.text,
        excerpt=excerpt,
        evidence_ref=STREAM,
    )


def is_argv_invalid(observation: BootObservation | None) -> bool:
    """True when ``observation`` records a server argv the installed parser refused."""
    return observation is not None and observation.producer == PRODUCER and observation.matched_marker == ARGV_INVALID


__all__ = ["ARGV_INVALID", "PRODUCER", "argv_invalid_observation", "is_argv_invalid"]
