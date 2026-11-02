# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Slot selector answers for the tree tests, every field present as a schema-constrained provider returns them."""

from __future__ import annotations

from kernelforge.workflows.forge_loop.roles.slot_selector import SlotSelectionAnswer

_SLOT_FIELDS = ("direction", "hint", "source", "gains")


def _selection(slots=(), *, remeasure=(), refresh_analysis=(), direction_status=()) -> SlotSelectionAnswer:
    """An answer naming ``slots``, each with the fields its type does not use left empty."""
    return SlotSelectionAnswer.model_validate(
        {
            "rationale": "",
            "slots": [{**dict.fromkeys(_SLOT_FIELDS, ""), **slot} for slot in slots],
            "remeasure": list(remeasure),
            "refresh_analysis": list(refresh_analysis),
            "direction_status": list(direction_status),
        }
    )
