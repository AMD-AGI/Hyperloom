# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Breakdown recorder: author-time capture of ``session_breakdown.json`` data.

Producers record facts where they are born (see :func:`.recorder.get_recorder`);
the exporter assembles them at finalize (see :func:`assemble_parts`). Each
section has a single owning producer, so there is no cross-producer write
contention.

The session is bound once, at startup, so no entry point below takes a path::

    from hyperloom.inference_optimizer.session.session_binding import bind_session

    bind_session(session_dir)     # coordinator startup, the only place

Write side::

    from hyperloom.inference_optimizer.breakdown.recorder.recorder import get_recorder

    rec = get_recorder(producer="coordinator")
    rec.record_singleton("session", session_row)          # one final blob

Read side::

    from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts, has_parts

    sections = assemble_parts(session_dir)   # {section: list | dict}

For SBD v6 timeline events there is a second surface on top of that primitive
one, so the rules governing ids, keys and ordering live in one place instead of
being restated per event type: :mod:`.event_ids` builds the two id forms,
:mod:`.event_sink` writes a row into whichever event its caller decided it
belongs to, :mod:`.event_rows` filters/orders/groups rows at assembly, and
:mod:`.event_timeline` owns the two timeline writes an event makes and the
residual states a killed session leaves behind.

Recording is best-effort by design, so a fact that never arrived leaves nothing
behind to explain itself. Set ``HYPERLOOM_BREAKDOWN_TRACE=1`` to log every
write, naming its call site and, when a write merges into an existing fragment,
the fields whose values it changed (see :mod:`.trace`).
"""

from __future__ import annotations

from .assembler import assemble_parts, has_parts
from .outcome_stage import record_stage_reached
from .session_metadata import record_metadata_identity, record_metadata_langfuse

__all__ = [
    "assemble_parts",
    "has_parts",
    "record_metadata_identity",
    "record_metadata_langfuse",
    "record_stage_reached",
]
