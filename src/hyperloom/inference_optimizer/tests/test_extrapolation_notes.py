"""The projection has to say when it is working outside its evidence.

A wrong number that announces itself is a different class of problem from a
wrong number that does not, and the search consumes these silently. Context is
the case that matters: Hyperloom sweeps it to 65,536 while nothing validated the
model past about 1,500, so crossing the boundary is routine rather than exotic.
"""
from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors.infersim_bridge import (
    SINGLE_NODE_GPUS,
    VALIDATED_CONTEXT_TOKENS,
    ServingSpec,
    extrapolation_notes,
)


def spec(isl: int = 1024, osl: int = 128, **kw) -> ServingSpec:
    return ServingSpec(framework="vllm", model_path="/models/gpt-oss-120b",
                       isl=isl, osl=osl, **kw)


def test_inside_the_validated_box_says_nothing():
    notes = extrapolation_notes(spec(isl=1024, osl=128), replica_gpus=8,
                                calibrated=True)
    assert notes == []


def test_context_past_the_validated_range_is_flagged():
    notes = extrapolation_notes(spec(isl=65536, osl=4096), replica_gpus=8,
                                calibrated=True)
    assert any("context" in n for n in notes)
    assert any(str(65536 + 4096) in n for n in notes)


def test_the_boundary_itself_is_not_flagged():
    """Exactly at the validated edge is still inside it."""
    notes = extrapolation_notes(spec(isl=VALIDATED_CONTEXT_TOKENS, osl=0),
                                replica_gpus=8, calibrated=True)
    assert not any("context" in n for n in notes)


def test_context_counts_what_the_run_will_hold_not_just_the_prompt():
    """A short prompt decoding for a long time still ends up at long context."""
    notes = extrapolation_notes(spec(isl=512, osl=131072), replica_gpus=8,
                                calibrated=True)
    assert any("context" in n for n in notes)


def test_crossing_a_node_boundary_is_flagged():
    notes = extrapolation_notes(spec(), replica_gpus=SINGLE_NODE_GPUS * 2,
                                calibrated=True)
    assert any("nodes" in n for n in notes)


def test_uncalibrated_is_flagged():
    notes = extrapolation_notes(spec(), replica_gpus=8, calibrated=False)
    assert any("simulation" in n for n in notes)


def test_notes_accumulate_rather_than_shadowing_each_other():
    notes = extrapolation_notes(spec(isl=131072, osl=1024), replica_gpus=32,
                                calibrated=False)
    assert len(notes) == 3


@pytest.mark.parametrize("isl,osl", [(0, 0), (None, None)])
def test_missing_lengths_do_not_raise(isl, osl):
    assert extrapolation_notes(spec(isl=isl, osl=osl), replica_gpus=8,
                               calibrated=True) == []
