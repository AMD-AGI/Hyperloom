# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a specialist of a server built from its checkout finds gains: one section, told to a single-branch and a
multi-branch specialist alike.

The method is the same for every such server: a time budget of one decode step from the timing lines of a measured
round's log, which the specialist is handed since a sandbox shows it no log, then the levers the launch does not use
yet (an option of the server, an auxiliary model file, a source change in the GPU
backend), each bounded by the share of the step it touches and stated with what would prove it wrong. The text names no
server: where a server's options are defined, the flag that names its auxiliary model, the words of its timing lines
and the host side of its decode loop are the rule pack's, by component, and so are the files of its GPU kernels.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from hyperloom.common.outside_facts import NATIVE_AUX_MODEL_FLAGS, NATIVE_SERVING, kernel_knobs
from hyperloom.common.server_timing import TIMING_LINES_KEPT

__all__ = ["gpu_kernel_dirs", "native_discovery"]

_EVIDENCE = """\
Finding gains in {component}: time evidence first, then the levers this launch does not use yet, in the order of the \
time each can touch. When a request ends, the server logs its decode time and tokens on the line holding \
"{request_timing}" and, when it drafts, the draft tokens the target accepted and the tokens a verify step yielded on \
the line holding "{speculative_stats}". {handed} Before the first candidate you choose with a measured round to read, \
write down a time budget of one decode step from those lines of it, not from a prior or from memory: the step's wall \
time (the time per token times the tokens per step); the least the target's pass and the drafts need, which is the \
bytes each reads per step over the card's memory bandwidth; and the rest, which is kernels running below that bound \
and host work between GPU launches while the GPU idles. A \
lever gains at most the share of the step it touches, so the budget ranks the levers and bounds each one. Where the \
session takes no GPU profile of this server (a profile test is refused, no roofline is recorded), those lines and the \
source are the evidence there is: do not wait for a profile."""
_HANDED = """\
Those lines are handed to you, the last {kept} of each kind a server.log holds, read from the log for you: {carried} \
The logs themselves are {logs}; a sandbox does not show them to you, and the handed lines are then the ones you have."""
_LEVERS = """\
Three kinds of lever a launch can lack:
(a) An option of the server that the launch does not set. Take the options from the checkout, never from memory: each \
is one {option_entry} entry of {option_table}, with its spellings, its help text and the field it sets ({help_command} \
prints those the server takes). Read the whole table once. For each option \
that moves work off the part of the step your budget shows largest, or removes it, read the path it switches on \
({host_loop}) and make it a candidate: it is a code path this session has not measured, not a value in the settings \
sweep, and it costs no build. No option is ruled out by a label in its help text: each is judged by the measurement \
and the accuracy gate.
(b) An auxiliary model file. {aux_model_flag} names the draft model or speculative head. List the directory of every \
model file the launch names, following symlinks, for other builds of that file. \
The served model's weights stay as the session has them; a draft the target verifies is not those weights, so which \
of the files already there drafts is a launch choice, decided by the acceptance on the "{speculative_stats}" line and \
the accuracy gate. Its ceiling is the drafts' share of the step.
(c) A source change in the GPU backend{kernel_dirs}: a launch constant fixed per architecture \
that was chosen for another shape or card than this workload's, or two kernels \
that can be one (a small kernel feeding a larger one, two kernels reading the same input). Its ceiling is the part of \
those kernels' share that lies above their bandwidth bound.
Order candidates by expected gain, the share touched times the part of it the lever removes; at equal gain a launch \
change goes before a build. With each one state the budget line it acts on, the gain you expect and its upper bound, \
and the observation that would prove it wrong (the timing line that must move, the acceptance that must hold), so a \
flat result names the assumption that failed."""
_KERNEL_DIRS = " ({dirs})"


def gpu_kernel_dirs(component: str) -> tuple[str, ...]:
    """The directories of ``component``'s GPU kernels, which are those of the files its kernel knobs are set in."""
    files = (knob.file for knob in kernel_knobs().values() if knob.component == component)
    return tuple(sorted({f"{PurePosixPath(file).parent}/" for file in files}))


def native_discovery(component: str, *, carried: str, logs: str) -> str:
    """The section for a specialist of ``component``'s server; ``carried`` is the sentence that says what carries the
    timing lines it is handed, and ``logs`` where the logs they were read from are. Empty for a component the rule pack
    states no serving facts of."""
    facts = NATIVE_SERVING.get(component)
    if facts is None:
        return ""
    dirs = gpu_kernel_dirs(component)
    handed = _HANDED.format(kept=TIMING_LINES_KEPT, carried=carried, logs=logs)
    evidence = _EVIDENCE.format(component=component, handed=handed, **facts)
    flag, *aliases = NATIVE_AUX_MODEL_FLAGS[component]
    levers = _LEVERS.format(
        kernel_dirs=_KERNEL_DIRS.format(dirs=", ".join(dirs)) if dirs else "",
        aux_model_flag=f"{flag} ({', '.join(aliases)})" if aliases else flag,
        **facts,
    )
    return f"{evidence}\n{levers}"
