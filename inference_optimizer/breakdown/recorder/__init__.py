# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Breakdown recorder: author-time capture of ``session_breakdown.json`` data.

Producers record facts where they are born (see :func:`get_recorder`); the
exporter assembles them at finalize (see :func:`assemble_parts`). Each section
has a single owning producer, so there is no cross-producer write contention.

Write side::

    from inference_optimizer.breakdown.recorder import get_recorder

    rec = get_recorder(session_dir, producer="sweep")
    rec.record_singleton("sweep", sweep_payload)          # one final blob
    rec.record_item("phase_timeline", event, key=task_id)  # event stream

Read side::

    from inference_optimizer.breakdown.recorder import assemble_parts, has_parts

    sections = assemble_parts(session_dir)   # {section: list | dict}
"""

from __future__ import annotations

from . import instrument
from .assembler import assemble_parts, has_parts, parts_dir
from .instrument import (
    record_critic_iteration,
    record_kernel_backend_result,
    record_kernel_discovery,
    record_kernel_dispatch,
    record_kernel_e2e,
    record_kernel_invocations,
    record_phase_event,
    record_robustness_signal,
    record_singleton_section,
    record_specialist_round,
    record_tool_version,
    snapshot_state_sections,
)
from .recorder import Recorder, get_recorder
from .sections import (
    DERIVED_SECTIONS,
    SECTION_SHAPES,
    SectionShape,
    section_shape,
)

__all__ = [
    "DERIVED_SECTIONS",
    "SECTION_SHAPES",
    "Recorder",
    "SectionShape",
    "assemble_parts",
    "get_recorder",
    "has_parts",
    "instrument",
    "parts_dir",
    "record_critic_iteration",
    "record_kernel_backend_result",
    "record_kernel_discovery",
    "record_kernel_dispatch",
    "record_kernel_e2e",
    "record_kernel_invocations",
    "record_phase_event",
    "record_robustness_signal",
    "record_singleton_section",
    "record_specialist_round",
    "record_tool_version",
    "section_shape",
    "snapshot_state_sections",
]
