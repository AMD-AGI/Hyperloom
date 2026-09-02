# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Breakdown recorder: author-time capture of ``session_breakdown.json`` data.

Producers record facts where they are born (see :func:`get_recorder`); the
exporter assembles them at finalize (see :func:`assemble_parts`). Each section
has a single owning producer, so there is no cross-producer write contention.

The session is bound once, at startup, so no entry point below takes a path::

    from hyperloom.inference_optimizer.session.session_binding import bind_session

    bind_session(session_dir)     # coordinator startup, the only place

Write side::

    from hyperloom.inference_optimizer.breakdown.recorder import get_recorder

    rec = get_recorder(producer="sweep")
    rec.record_singleton("sweep", sweep_payload)          # one final blob
    rec.record_item("phase_timeline", event, key=task_id)  # event stream

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

from . import instrument
from .assembler import (
    KERNEL_EVENT_SECTIONS,
    assemble_parts,
    has_parts,
    kernel_event_parts,
    parts_dir,
)
from .event_ids import EVENT_ID_SEPARATOR, EventId, event_id, fragment_key, parse_event_id
from .event_rows import (
    EVENT_ID_FIELD,
    SCOPE_FIELDS,
    group_rows,
    rows_for_event,
    sort_rows,
    wire_row,
    wire_rows,
)
from .event_sink import EventSink, RecordSink, make_sink
from .event_timeline import (
    EVENT_STATUS_INTERRUPTED,
    EVENT_STATUS_RUNNING,
    RESIDUAL_NO_EVENT,
    RESIDUAL_RUNNING,
    TERMINAL_EVENT_STATUSES,
    TIMELINE_SEQUENCE_FIELD,
    ResidualEvent,
    build_envelope,
    finish_event,
    open_event,
    residual_events,
)
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
    recorder_for,
    section_shape,
)

__all__ = [
    "DERIVED_SECTIONS",
    "EVENT_ID_FIELD",
    "EVENT_ID_SEPARATOR",
    "EVENT_STATUS_INTERRUPTED",
    "EVENT_STATUS_RUNNING",
    "KERNEL_EVENT_SECTIONS",
    "RESIDUAL_NO_EVENT",
    "RESIDUAL_RUNNING",
    "SCOPE_FIELDS",
    "SECTION_SHAPES",
    "TERMINAL_EVENT_STATUSES",
    "TIMELINE_SEQUENCE_FIELD",
    "EventId",
    "EventSink",
    "RecordSink",
    "Recorder",
    "ResidualEvent",
    "SectionShape",
    "TRACE",
    "TRACE_ENV",
    "assemble_parts",
    "build_envelope",
    "enable_trace",
    "event_id",
    "finish_event",
    "fragment_key",
    "get_recorder",
    "group_rows",
    "has_parts",
    "instrument",
    "kernel_event_parts",
    "make_sink",
    "open_event",
    "parse_event_id",
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
    "recorder_for",
    "residual_events",
    "rows_for_event",
    "snapshot_state_sections",
    "section_shape",
    "sort_rows",
    "trace_enabled",
    "wire_row",
    "wire_rows",
]
