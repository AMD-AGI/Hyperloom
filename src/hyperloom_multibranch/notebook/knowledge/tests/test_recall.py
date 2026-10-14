# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recall sorts what other sessions published into priors and leads to re-verify, by their conditions."""

from __future__ import annotations

import pytest
from hyperloom_kb import RemoteReadResult

from hyperloom_multibranch.notebook.knowledge.recall import DECISION, RecallTier, recall_experiences
from hyperloom_multibranch.notebook.knowledge.tests.fakes import (
    FakeExperienceReader,
    completed_read,
    experience_summary,
)
from hyperloom_multibranch.notebook.tests.factories import profile

SAME = {"gpu": "mi355x", "framework": "vllm", "bottleneck": "attention", "isl": 1024, "osl": 1024}


def test_an_experience_measured_under_every_queried_condition_is_a_prior_and_others_are_to_reverify() -> None:
    reader = FakeExperienceReader(
        completed_read(
            experience_summary("exp-same", SAME),
            experience_summary("exp-other-gpu", {**SAME, "gpu": "mi300x", "isl": 2048}, decision="revert"),
            experience_summary(
                "exp-no-bottleneck",
                {key: value for key, value in SAME.items() if key != "bottleneck"} | {"isl": 1024.0},
            ),
        )
    )

    recall = recall_experiences(reader, profile(), 1000.0, bottleneck="attention", branch="s1", limit=5)

    tiers = {item.experience_id: (item.tier, item.differing) for item in recall.experiences}
    assert tiers == {
        "exp-same": (RecallTier.PRIOR, ()),
        "exp-other-gpu": (RecallTier.REVERIFY, ("gpu", "isl")),
        "exp-no-bottleneck": (RecallTier.REVERIFY, ("bottleneck",)),
    }
    assert recall.rendered_refs == ("exp-same", "exp-other-gpu", "exp-no-bottleneck")
    assert recall.experiences[0].gain_pct == pytest.approx(10.0)
    assert [item.experience_id for item in recall.of(RecallTier.PRIOR)] == ["exp-same"]


def test_the_read_carries_the_sessions_identity_workload_baseline_and_bottleneck() -> None:
    reader = FakeExperienceReader(completed_read())

    recall_experiences(reader, profile(), 1000.0, bottleneck="attention", branch="s1", intent="fp8 kv cache", limit=5)

    ((decision, context, limit),) = reader.calls
    assert (decision, limit) == (DECISION, 5)
    assert context["identity"]["gpu"] == "MI355X" and "isl" not in context["identity"]
    assert context["workload"] == {"tp": 8, "conc": 64, "isl": 1024, "osl": 1024}
    assert context["benchmark_baseline"]["throughput"] == 1000.0
    assert context["observations"] == {"bottleneck": "attention", "branch": "s1", "intent": "fp8 kv cache"}


def test_without_a_kb_or_with_a_failed_read_the_recall_says_so_and_still_renders() -> None:
    disabled = recall_experiences(None, profile(), None, bottleneck="attention")
    failed = recall_experiences(
        FakeExperienceReader(RemoteReadResult("", "unavailable", "", (), ("planner_timeout",))),
        profile(),
        None,
        bottleneck="x",
    )
    nothing = recall_experiences(FakeExperienceReader(completed_read()), profile(), None, bottleneck="x")

    assert disabled.render() == ["- unavailable (disabled)"]
    assert failed.render() == ["- unavailable (unavailable: planner_timeout)"]
    assert nothing.render()[-1] == "- nothing matched"


def test_the_rendering_groups_the_tiers_and_names_what_differs() -> None:
    no_values = {**experience_summary("exp-bare", {**SAME, "gpu": "mi300x"}), "baseline_value": None}
    reader = FakeExperienceReader(completed_read(experience_summary("exp-same", SAME), no_values))

    lines = recall_experiences(reader, profile(hardware="mi355x"), 1000.0, bottleneck="attention").render()

    assert lines[1:] == [
        "priors, measured under this session's conditions:",
        "- --kv-cache-dtype fp8: keep +10.00% [exp-same]",
        "to re-verify, measured under other conditions:",
        "- --kv-cache-dtype fp8: keep n/a (gpu differ) [exp-bare]",
    ]
