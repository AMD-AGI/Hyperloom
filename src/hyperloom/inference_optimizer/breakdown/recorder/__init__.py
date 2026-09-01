# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Breakdown recorder: author-time capture of ``session_breakdown.json`` data.

Producers record facts where they are born (see :func:`get_recorder`); the
exporter assembles them at finalize (see :func:`assemble_parts`). Each section
has a single owning producer, so there is no cross-producer write contention.

Write side::

    from hyperloom.inference_optimizer.breakdown.recorder import get_recorder

    rec = get_recorder(session_dir, producer="sweep")
    rec.record_singleton("sweep", sweep_payload)          # one final blob
    rec.record_item("phase_timeline", event, key=task_id)  # event stream

Read side::

    from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts, has_parts

    sections = assemble_parts(session_dir)   # {section: list | dict}

Recording is best-effort by design, so a fact that never arrived leaves nothing
behind to explain itself. Set ``HYPERLOOM_BREAKDOWN_TRACE=1`` to log every
write, naming its call site and, when a write merges into an existing fragment,
the fields whose values it changed (see :mod:`.trace`).
"""

from __future__ import annotations

from . import instrument
from .assembler import assemble_parts, has_parts, parts_dir
from .trace import TRACE, TRACE_ENV, enable_trace, trace_enabled
from .instrument import (
    record_action_operation,
    record_adoption,
    record_artifact,
    record_critic_iteration,
    record_kernel_backend_result,
    record_kernel_discovery,
    record_kernel_dispatch,
    record_kernel_e2e,
    record_kernel_invocations,
    record_kernel_strategy_selection,
    record_native_kernel_run_start,
    record_native_kernel_run_result,
    record_geak_e2e_attempt,
    record_geak_operation,
    record_gemm_tuning_operation,
    record_measurement,
    record_operation,
    record_phase_event,
    record_phase_transition,
    record_robustness_signal,
    record_run_snapshot,
    record_singleton_section,
    record_specialist_round,
    record_subject,
    record_tool_version,
    record_trace_event,
    snapshot_state_sections,
)
from .recorder import (
    DERIVED_SECTIONS,
    SECTION_SHAPES,
    Recorder,
    SectionShape,
    get_recorder,
    section_shape,
)

__all__ = [
    "DERIVED_SECTIONS",
    "SECTION_SHAPES",
    "Recorder",
    "SectionShape",
    "TRACE",
    "TRACE_ENV",
    "assemble_parts",
    "enable_trace",
    "get_recorder",
    "has_parts",
    "instrument",
    "parts_dir",
    "record_action_operation",
    "record_adoption",
    "record_artifact",
    "record_critic_iteration",
    "record_kernel_backend_result",
    "record_kernel_discovery",
    "record_kernel_dispatch",
    "record_kernel_e2e",
    "record_kernel_invocations",
    "record_kernel_strategy_selection",
    "record_native_kernel_run_start",
    "record_native_kernel_run_result",
    "record_geak_e2e_attempt",
    "record_geak_operation",
    "record_gemm_tuning_operation",
    "record_measurement",
    "record_operation",
    "record_phase_event",
    "record_phase_transition",
    "record_robustness_signal",
    "record_run_snapshot",
    "record_singleton_section",
    "record_specialist_round",
    "record_subject",
    "record_tool_version",
    "record_trace_event",
    "snapshot_state_sections",
    "section_shape",
    "trace_enabled",
]
