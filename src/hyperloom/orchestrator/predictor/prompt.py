# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The predictor's part of orchestration's untested-proposal section.

The state layer says which proposals still wait for a bench; this says what
orchestration does with the predictor's: copy its rows into the next grid
verbatim, leave the ones a queued grid already holds, and dispatch its
source-change mandate by id. The specialists' rows follow under their own
header.
"""

from __future__ import annotations

from collections.abc import Collection

from hyperloom.common.prompt_safety import flatten_for_prompt
from hyperloom.orchestrator.actions.executors._accuracy_gate import ACCURACY_THRESHOLD
from hyperloom.orchestrator.predictor.mandate import open_mandate
from hyperloom.orchestrator.predictor.rows import PROVENANCE
from hyperloom.orchestrator.predictor.view import SessionView
from hyperloom.orchestrator.state._shared_state.render import untested_proposal_line

_HEADER = (
    "Predictor proposals from this cycle that no explore round has benched, most-voted first. The",
    "Coordinator benches the head of the queue, these rows first, only while no explore is queued or",
    "running, which rarely happens while your grids hold the lane. These rows are the exception to",
    "dispatching only variants not listed here: put the ones worth a slot into your next `explore` grid",
    "verbatim, keeping each row's name, fields and `provenance: primatune`. The predictor is re-asked only",
    "when the stack moves, so a row you pass over now is very likely never measured; if you skip one, say why.",
)


def _accuracy_gate(state: SessionView) -> list[str]:
    """The accuracy gate every explore KEEP passes; nothing while the session measures no accuracy."""
    baseline = float(state.baseline_accuracy or 0.0)
    if baseline <= 0:
        return []
    return [
        f"Every variant runs the accuracy eval, and a KEEP needs accuracy no more than {ACCURACY_THRESHOLD:g} below",
        f"the baseline's {baseline:.3f}, so a row that changes numerics cannot keep on throughput alone.",
    ]


def queue_block(state: SessionView, *, in_grids: Collection[str] = frozenset(), max_entries: int = 12) -> str:
    """The predictor's rows still waiting for a bench, the ones a queued grid holds, and its open mandate.

    Args:
        in_grids: Content fingerprints of the variants in queued or running explore grids. A row among
            them is named rather than offered, since orchestration copies offered rows.
        max_entries: Rows rendered in full; the rest are counted.
    """
    rows = [row for row in state.untested_proposal_rows() if row["provenance"] == PROVENANCE]
    offered = [row for row in rows if row["fingerprint"] not in in_grids]
    held = [row["name"] for row in rows if row["fingerprint"] in in_grids]
    mandate = open_mandate(state)
    out: list[str] = []
    if offered:
        out.extend((*_HEADER, *_accuracy_gate(state), ""))
        # The row was cut to its rationale budget when it was filed, so its reason is shown whole.
        out.extend(untested_proposal_line(row, why_chars=None) for row in offered[:max_entries])
        if len(offered) > max_entries:
            out.append(f"(+{len(offered) - max_entries} more not shown)")
    if held:
        if out:
            out.append("")
        out.append(
            flatten_for_prompt(
                "Predictor rows already in a queued or running explore grid, not to be copied again: " + ", ".join(held)
            )
        )
    if mandate:
        if out:
            out.append("")
        out.extend(
            [
                "Predictor source-change mandate (prose, not a variant). To act on it, dispatch",
                "`delegate{action_name='specialist', params={scope:'freeform', mode:'patch', "
                f"primatune_mandate_id:'{mandate['mandate_id']}', task_description:'<one line>'}}}}`.",
                "The Coordinator substitutes the mandate's own text for task_description; carry the id verbatim.",
                flatten_for_prompt(f"• {mandate['mandate_id']} why={mandate['mandate'][:160]}"),
            ]
        )
    return "\n".join(out)
