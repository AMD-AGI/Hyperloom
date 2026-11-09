# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract types cross the boundary as JSON and read back unchanged; anything else is refused."""

from __future__ import annotations

import json
import math

import pytest

from hyperloom.brain_api import (
    Attempt,
    Candidate,
    ConfigDelta,
    Gap,
    Outcome,
    PatchFile,
    Proposal,
    SearchState,
    WireError,
    Workload,
    from_wire,
    json_value,
    to_wire,
)


def _state() -> SearchState:
    candidate = Candidate("graph-bs", ConfigDelta("--cuda-graph-max-bs 256"), reason="fewer eager steps")
    return SearchState(
        workload=Workload("Qwen3-32B", "sglang", tp=8, model_info={"layers": 64}),
        baseline_tput=100.0,
        current_tput=105,
        keep_threshold_pct=1.0,
        launch=ConfigDelta(extra_envs={"A": "1"}),
        stack=(candidate,),
        attempts=(Attempt(candidate, Outcome.KEEP, 5.0),),
        gaps=(Gap("attn.decode", domain="attention", evidence={"pct": 41.0}),),
    )


def test_a_state_reads_back_unchanged_through_json() -> None:
    state = _state()
    wire = json.loads(json.dumps(to_wire(state)))
    assert from_wire(SearchState, wire) == state
    assert wire["current_tput"] == 105
    assert isinstance(from_wire(SearchState, wire).current_tput, float)


def test_patches_and_optional_fields_round_trip() -> None:
    proposal = Proposal(
        candidates=(Candidate("p", patches=(PatchFile("python/x.py", "--- a\n+++ b\n"),), predicted_gain_pct=None),),
        confidence=0.4,
    )
    assert from_wire(Proposal, json.loads(json.dumps(to_wire(proposal)))) == proposal


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda w: w.update(extra=1), "unknown keys"),
        (lambda w: w.pop("baseline_tput"), "missing keys"),
        (lambda w: w.update(baseline_tput="fast"), "expected float"),
        (lambda w: w["attempts"][0].update(outcome="maybe"), "not a Outcome"),
        (lambda w: w.update(gaps={}), "expected a list"),
        (lambda w: w.update(workload=None), "workload: expected an object"),
        (lambda w: w["attempts"][0].update(gain_pct="x"), r"attempts\[0\].gain_pct: expected float"),
        (lambda w: w["workload"].update(tp=True), "expected int"),
        (lambda w: w["launch"].update(args_mode="merge"), "args_mode"),
    ],
)
def test_a_malformed_object_is_refused_with_its_path(mutate, match) -> None:
    wire = to_wire(_state())
    mutate(wire)
    with pytest.raises(WireError, match=match):
        from_wire(SearchState, wire)


def test_construction_checks_hold_without_the_wire() -> None:
    with pytest.raises(WireError, match="changes nothing"):
        Candidate("noop")
    with pytest.raises(WireError, match="relative"):
        PatchFile("/etc/passwd", "x")
    with pytest.raises(WireError, match="relative"):
        PatchFile("a/../../b", "x")
    with pytest.raises(WireError, match="strings"):
        ConfigDelta(extra_envs={"A": 1})
    with pytest.raises(WireError, match="model_name"):
        Workload("", "sglang")


@pytest.mark.parametrize("value", [math.nan, {1: "a"}, object(), {"a": [math.inf]}])
def test_json_value_refuses_what_would_not_survive_json(value) -> None:
    with pytest.raises(WireError):
        json_value(value, "here")


def test_mappings_are_copied_so_the_caller_cannot_change_a_frozen_value() -> None:
    envs = {"A": "1"}
    delta = ConfigDelta(extra_envs=envs)
    envs["B"] = "2"
    assert delta.extra_envs == {"A": "1"}
    assert ConfigDelta().empty and not delta.empty
