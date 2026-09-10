# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The names of the v6 spool sections, and which event each one belongs to.

A leaf module on purpose: it imports nothing from the recorder, so both the
assembler and the per-event modules that read through the assembler can name
these without an import cycle. That cycle is why the tuples used to be spelled
out twice, once here and once in the module that owns the event -- two lists
that had to be edited together and could only be checked by eye.
"""

from __future__ import annotations

import re

_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(value: str) -> str:
    """Filesystem-safe token; empty input collapses to ``unknown``."""
    s = _SANITIZE.sub("-", str(value or "").strip())
    return s.strip("-.") or "unknown"


def section_glob(section: str) -> str:
    """The glob matching every fragment of one section.

    Every fragment is named ``<section>__<producer>...``, so a reader after one
    section can ask the filesystem for it instead of parsing the whole spool
    and discarding what it did not want. On a long run the spool reaches tens
    of thousands of files and the parsing, not the listing, is the cost.
    """
    return f"{slug(section)}__*.json"


#: The tail of an appended fragment's name: ``<pid>-<seq>``. Appended rows have
#: no key to be named after, so their names cannot say which event they belong
#: to and they are always read.
_APPENDED_FRAGMENT = re.compile(r"__\d+-\d{6}\.json$")


def may_hold_event(name: str, event_slug: str) -> bool:
    """Whether the fragment file ``name`` could hold a row of that event.

    A keyed row is named after its fragment key, and every such key starts
    with the event id, so its name always contains the event's slug. An
    appended row is named after the write instead, so nothing in its name
    identifies the event and it has to be read to find out.

    Deliberately one-sided: it may say yes to a file that turns out to belong
    to another event, which costs one parse, and callers filter the rows they
    get by event anyway. Saying no to a file that does hold the event would
    silently drop a fact, so nothing here may guess in that direction.
    """
    return event_slug in name or bool(_APPENDED_FRAGMENT.search(name))


KERNEL_EVENT_SECTIONS: tuple[str, ...] = (
    "kernel_event",
    "kernel_lane_run",
    "kernel_rebench_attempt",
    "kernel_trace_analyze",
    "kernel_geak_attempt",
    "kernel_geak_discovery",
    "kernel_geak_acceptance",
    "kernel_discovered",
    "kernel_integrate",
)

#: The roofline substreams. Rows belong to whichever event tagged them: the
#: roofline's own when dispatched, the enclosing phase's when called inline.
ROOFLINE_EVENT_SECTIONS: tuple[str, ...] = (
    "roofline_event",
    "roofline_action",
    "roofline_profile_run",
    "roofline_analysis_run",
    "roofline_kernel",
)

BASELINE_EVENT_SECTIONS: tuple[str, ...] = (
    "baseline_event",
    "baseline_action",
    "baseline_run",
    "baseline_round",
)

CONC_SWEEP_EVENT_SECTIONS: tuple[str, ...] = (
    "conc_sweep_event",
    "conc_sweep_action",
    "conc_sweep_arm",
    "conc_sweep_variant",
    "conc_sweep_pair",
)

ENABLEMENT_EVENT_SECTIONS: tuple[str, ...] = (
    "enablement_event",
    "enablement_attempt",
    "enablement_build",
    "enablement_revalidation",
    "enablement_human_review",
)

PHASE_EVENT_SECTIONS: tuple[str, ...] = (
    "phase_event",
    "phase_segment",
    "phase_action",
    "phase_marker",
    "phase_proposal",
)

STACK_EVENT_SECTIONS: tuple[str, ...] = (
    "stack_event",
    "stack_adoption",
    "stack_validation",
)

WARM_REPLAY_EVENT_SECTIONS: tuple[str, ...] = (
    "warm_replay_event",
    "warm_replay_gate",
)

WARM_START_EVENT_SECTIONS: tuple[str, ...] = (
    "warm_start_event",
    "warm_start_read",
)

FRAMEWORK_EVENT_SECTIONS: tuple[str, ...] = (
    "framework_event",
    "framework_plateau",
    "framework_run",
    "framework_proposal",
    "framework_proposal_step",
    "framework_attempt",
    "framework_attempt_gate",
)

#: Every section holding v6 event rows. Consumed by the timeline rather than
#: the breakdown envelope, so assembly pops them out of the wire shape.
EVENT_SECTIONS: tuple[str, ...] = (
    KERNEL_EVENT_SECTIONS
    + ROOFLINE_EVENT_SECTIONS
    + BASELINE_EVENT_SECTIONS
    + CONC_SWEEP_EVENT_SECTIONS
    + ENABLEMENT_EVENT_SECTIONS
    + PHASE_EVENT_SECTIONS
    + STACK_EVENT_SECTIONS
    + WARM_REPLAY_EVENT_SECTIONS
    + WARM_START_EVENT_SECTIONS
    + FRAMEWORK_EVENT_SECTIONS
)

__all__ = [
    "BASELINE_EVENT_SECTIONS",
    "CONC_SWEEP_EVENT_SECTIONS",
    "ENABLEMENT_EVENT_SECTIONS",
    "EVENT_SECTIONS",
    "FRAMEWORK_EVENT_SECTIONS",
    "KERNEL_EVENT_SECTIONS",
    "PHASE_EVENT_SECTIONS",
    "ROOFLINE_EVENT_SECTIONS",
    "STACK_EVENT_SECTIONS",
    "WARM_REPLAY_EVENT_SECTIONS",
    "WARM_START_EVENT_SECTIONS",
    "may_hold_event",
    "section_glob",
    "slug",
]
