# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The names of the v6 spool sections, and which event each one belongs to.

A leaf module on purpose: it imports nothing from the recorder, so both the
assembler and the per-event modules that read through the assembler can name
these without an import cycle, and each tuple is spelled out once rather than
also in the module that owns the event.
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
    "kernel_ave_attempt",
)

#: The roofline substreams. Rows belong to whichever event tagged them: the
#: roofline's own when dispatched, the enclosing phase's when called inline;
#: only the rows' event id differs.
ROOFLINE_EVENT_SECTIONS: tuple[str, ...] = (
    "roofline_event",
    "roofline_action",
    "roofline_profile_run",
    "roofline_analysis_run",
    "roofline_kernel",
)

#: Runs and rounds are separate because the executor retries at both levels:
#: a pass the budget refused before it booted anything has a run and no round,
#: and a flat list would drop it.
BASELINE_EVENT_SECTIONS: tuple[str, ...] = (
    "baseline_event",
    "baseline_action",
    "baseline_run",
    "baseline_round",
)

#: An arm is a whole ladder run under one set of server args, a variant one
#: rung of it including rungs that only ever attempted to boot, a pair the two
#: arms joined at one concurrency.
CONC_SWEEP_EVENT_SECTIONS: tuple[str, ...] = (
    "conc_sweep_event",
    "conc_sweep_action",
    "conc_sweep_arm",
    "conc_sweep_variant",
    "conc_sweep_pair",
)

#: One event per session: the lane's trigger and the round that settles it are
#: recorded from different phases and must land on the same event.
ENABLEMENT_EVENT_SECTIONS: tuple[str, ...] = (
    "enablement_event",
    "enablement_attempt",
    "enablement_build",
    "enablement_revalidation",
    "enablement_human_review",
)

#: One event per (phase, macro_cycle), so a re-entry is another segment row
#: rather than a second event. Proposals live here rather than on the framework
#: event because a proposal exists in every phase, and a refused one is never
#: dispatched and so never gets an action row.
PHASE_EVENT_SECTIONS: tuple[str, ...] = (
    "phase_event",
    "phase_segment",
    "phase_action",
    "phase_marker",
    "phase_proposal",
    "phase_denial",
)

#: One event per session: adoptions arrive from four phases into one ordered
#: chain, and reconciliation covers the whole.
STACK_EVENT_SECTIONS: tuple[str, ...] = (
    "stack_event",
    "stack_adoption",
    "stack_validation",
)

#: Applies are rows rather than a tally on the event: the plan's counts already
#: say how many landed, and what a reader needs from a replay that lost is which
#: item it was that did not.
WARM_REPLAY_EVENT_SECTIONS: tuple[str, ...] = (
    "warm_replay_event",
    "warm_replay_gate",
    "warm_replay_apply",
)

#: Reads are rows rather than a tally, which cannot be confined to T0: the
#: session's audit log also holds writes and mid-session amendment reads.
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
