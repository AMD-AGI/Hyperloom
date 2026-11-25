# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The review of its server's options a research specialist returns, in a session that keeps the option list of the
server it built.

Such a session keeps what its server's own ``--help`` prints, with the options its base launch leaves unset
(:func:`~hyperloom.inference_optimizer.native_builds.server_options`). A research specialist is asked for one verdict
on each of those (``options_review`` in its done file), so that an option it does not propose is one it looked at, not
one it never read. What it returns is held against the list: the result's notes say whether the review covers it, and
one that does not is said at WARNING. The proposals are admitted either way.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.native_builds import OptionVerdict, server_options
from hyperloom.inference_optimizer.session.paths import session_dir

from .profile import MODE_RESEARCH
from .prompt_inputs import SpecialistPromptInputs

__all__ = ["review_notes", "review_request"]

log = logging.getLogger(__name__)

#: The done-file field that carries the review.
_OPTIONS_REVIEW = "options_review"
_NAMED = 10


def review_request(inp: SpecialistPromptInputs) -> list[str]:
    """The done-file contract lines that ask a research specialist for the review; none for a patch specialist, or in
    a session that keeps no option list."""
    kept = server_options(session_dir(), inp.framework) if inp.mode == MODE_RESEARCH else None
    if kept is None:
        return []
    unset = kept.coverage(()).unset
    verdicts = "|".join(verdict.value for verdict in OptionVerdict)
    return [
        f"- ``{_OPTIONS_REVIEW}`` is REQUIRED in this session: one entry for each of the **{unset}** options",
        f'  that ``{kept.file}`` marks ``"set_by_base": false``. That file is the server\'s own',
        "  ``--help``, entry by entry, with whether the base launch sets each option; read it rather than",
        "  recalling the options. An entry is",
        f'  ``{{"option": "<any spelling the file lists for it>", "verdict": "{verdicts}",',
        '  "note": "<one line of evidence>"}``.',
        f"  ``{OptionVerdict.CANDIDATE.value}``: worth a measured round, so the ones you rank highest are your",
        f"  ``proposal_set``; ``{OptionVerdict.NO_EFFECT_ON_STEP.value}``: touches no part of a decode step of this",
        f"  workload; ``{OptionVerdict.DEFAULT_ALREADY.value}``: its default is the value you would set;",
        f"  ``{OptionVerdict.CHANGES_WORKLOAD_OR_WEIGHTS.value}``: changes what is served or measured (the workload,",
        f"  the served model's weights or its output), not how fast; ``{OptionVerdict.NOT_USABLE_HERE.value}``: needs",
        "  hardware, a build or a file this session does not have. The note says what you read that settles it:",
        "  the help text, the source the option switches on, or the timing line it would move.",
        f"  The app checks your entries against the file and records how many of the {unset} they cover.",
    ]


def _reviewed(block: object) -> list[str]:
    """The spellings the entries of ``block`` give a verdict on; an entry of another shape reviews nothing."""
    verdicts = {verdict.value for verdict in OptionVerdict}
    entries = block if isinstance(block, list) else []
    return [
        entry["option"].strip()
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("option"), str) and entry.get("verdict") in verdicts
    ]


def review_notes(session: Path | None, task: Any, mode: str, payload: Mapping[str, Any]) -> list[str]:
    """The notes the review in ``payload``, the done file of ``task``'s specialist of ``mode``, leaves on its result:
    whether it covers the options the base launch of the session at ``session`` leaves unset, and how many spellings
    it names that the server lists no option for. A review that falls short is said at WARNING. None for a patch
    specialist, or in a session that keeps no option list."""
    framework = str((task.params or {}).get("framework") or "")
    kept = server_options(session, framework) if mode == MODE_RESEARCH else None
    if kept is None:
        return []
    coverage = kept.coverage(_reviewed(payload.get(_OPTIONS_REVIEW)))
    if not coverage.missing and not coverage.unknown:
        return [f"options_review_complete:{coverage.unset}"]
    log.warning(
        "specialist %s: its options_review misses %d of the %d options the base launch leaves unset (%s) and names "
        "%d the server does not list (%s)",
        task.task_id,
        len(coverage.missing),
        coverage.unset,
        ", ".join(coverage.missing[:_NAMED]),
        len(coverage.unknown),
        ", ".join(coverage.unknown[:_NAMED]),
    )
    covered = (
        f"options_review_incomplete:{len(coverage.missing)}/{coverage.unset}"
        if coverage.missing
        else f"options_review_complete:{coverage.unset}"
    )
    return [covered, *([f"options_review_unknown:{len(coverage.unknown)}"] if coverage.unknown else [])]
