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

A session that wrote the list of its server's options from the server's own help hands the specialist that list
(:class:`OptionList`) in place of asking it to read the option table: asked to read the table, a specialist takes the
options from memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hyperloom.common.outside_facts import NATIVE_AUX_MODEL_FLAGS, NATIVE_SERVING, kernel_knobs
from hyperloom.common.server_timing import TIMING_LINES_KEPT

__all__ = ["DISCOVERY_TEXTS", "OptionList", "gpu_kernel_dirs", "native_discovery"]

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
prints those the server takes). {whole_list} For each option \
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
_WHOLE_TABLE = "Read the whole table once."
_OPTION_LIST = """\
The session ran {help_command} on the server it built and wrote every option it takes, with its spellings, its help \
text and whether the launch sets it, to {file}: {total} options, {unset} of them not set by the launch. Go through all \
of the unset ones, not the ones you remember."""

#: The texts the section is rendered from, for the version of a prompt that carries it.
DISCOVERY_TEXTS = (_EVIDENCE, _HANDED, _LEVERS, _KERNEL_DIRS, _WHOLE_TABLE, _OPTION_LIST)


@dataclass(frozen=True)
class OptionList:
    """The list a session wrote of the options its server takes, from the server's own help: the file, how many
    options it holds, and how many of them the launch leaves unset."""

    file: Path
    total: int
    unset: int


def gpu_kernel_dirs(component: str) -> tuple[str, ...]:
    """The directories of ``component``'s GPU kernels, which are those of the files its kernel knobs are set in."""
    files = (knob.file for knob in kernel_knobs().values() if knob.component == component)
    return tuple(sorted({f"{PurePosixPath(file).parent}/" for file in files}))


def native_discovery(component: str, *, carried: str, logs: str, options: OptionList | None = None) -> str:
    """The section for a specialist of ``component``'s server; ``carried`` is the sentence that says what carries the
    timing lines it is handed, ``logs`` where the logs they were read from are, and ``options`` the option list the
    session wrote (none: the specialist is asked to read the option table). Empty for a component the rule pack states
    no serving facts of."""
    facts = NATIVE_SERVING.get(component)
    if facts is None:
        return ""
    dirs = gpu_kernel_dirs(component)
    handed = _HANDED.format(kept=TIMING_LINES_KEPT, carried=carried, logs=logs)
    evidence = _EVIDENCE.format(component=component, handed=handed, **facts)
    flag, *aliases = NATIVE_AUX_MODEL_FLAGS[component]
    whole_list = _WHOLE_TABLE
    if options is not None:
        whole_list = _OPTION_LIST.format(
            help_command=facts["help_command"], file=options.file, total=options.total, unset=options.unset
        )
    levers = _LEVERS.format(
        kernel_dirs=_KERNEL_DIRS.format(dirs=", ".join(dirs)) if dirs else "",
        aux_model_flag=f"{flag} ({', '.join(aliases)})" if aliases else flag,
        whole_list=whole_list,
        **facts,
    )
    return f"{evidence}\n{levers}"
