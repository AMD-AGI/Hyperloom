# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recall sorts what other sessions published into priors and leads to re-verify, by their conditions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from hyperloom_kb import RemoteReadResult, RenderedRef

from hyperloom_multibranch.knowledge.recall import DECISION, RecallTier, recall_experiences
from hyperloom_multibranch.notebook.tests.factories import profile

SAME = {"gpu": "mi355x", "framework": "vllm", "bottleneck": "attention", "isl": 1024, "osl": 1024}


def _summary(experience_id: str, identity: Mapping[str, Any], decision: str = "keep") -> dict[str, Any]:
    return {
        "experience_id": experience_id,
        "source_run_id": "run-old",
        "identity": dict(identity),
        "change_summary": "--kv-cache-dtype fp8",
        "decision": decision,
        "baseline_value": 1000.0,
        "outcome_value": 1100.0,
    }


class FakeReader:
    def __init__(self, result: RemoteReadResult) -> None:
        self.result = result
        self.calls: list[tuple[str, Mapping[str, Any], int | None]] = []

    def read(self, decision: str, context: Mapping[str, Any], *, limit: int | None = None) -> RemoteReadResult:
        self.calls.append((decision, context, limit))
        return self.result


def _completed(*experiences: dict[str, Any]) -> RemoteReadResult:
    refs = tuple(RenderedRef(item["experience_id"], "representative") for item in experiences)
    return RemoteReadResult("read-1", "completed", "=== block ===", refs, (), experiences)


def test_an_experience_measured_under_every_queried_condition_is_a_prior_and_others_are_to_reverify() -> None:
    reader = FakeReader(
        _completed(
            _summary("exp-same", SAME),
            _summary("exp-other-gpu", {**SAME, "gpu": "mi300x", "isl": 2048}, decision="revert"),
            _summary(
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
    reader = FakeReader(_completed())

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
        FakeReader(RemoteReadResult("", "unavailable", "", (), ("planner_timeout",))), profile(), None, bottleneck="x"
    )
    nothing = recall_experiences(FakeReader(_completed()), profile(), None, bottleneck="x")

    assert disabled.render() == ["- unavailable (disabled)"]
    assert failed.render() == ["- unavailable (unavailable: planner_timeout)"]
    assert nothing.render()[-1] == "- nothing matched"


def test_the_rendering_groups_the_tiers_and_names_what_differs() -> None:
    no_values = {**_summary("exp-bare", {**SAME, "gpu": "mi300x"}), "baseline_value": None}
    reader = FakeReader(_completed(_summary("exp-same", SAME), no_values))

    lines = recall_experiences(reader, profile(hardware="mi355x"), 1000.0, bottleneck="attention").render()

    assert lines[1:] == [
        "priors, measured under this session's conditions:",
        "- --kv-cache-dtype fp8: keep +10.00% [exp-same]",
        "to re-verify, measured under other conditions:",
        "- --kv-cache-dtype fp8: keep n/a (gpu differ) [exp-bare]",
    ]
