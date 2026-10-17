# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recall sorts what other sessions published into priors and leads to re-verify, by their conditions."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from hyperloom_kb import RemoteReadResult

from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.notebook.knowledge.recall import (
    CONDITION_KEYS,
    DECISION,
    IdentityIndex,
    RecallQuery,
    RecallTier,
    recall_experiences,
)
from hyperloom_multibranch.notebook.knowledge.tests.fakes import (
    FakeExperienceReader,
    FakeExport,
    completed_read,
    experience_summary,
)
from hyperloom_multibranch.notebook.tests.factories import profile

SAME = {
    "model": "deepseek-ai/DeepSeek-R1",
    "gpu": "mi355x",
    "framework": "vllm",
    "tp": 8,
    "conc": 64,
    "bottleneck": "attention",
    "isl": 1024,
    "osl": 1024,
}
ATTENTION = RecallQuery("attention")


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

    recall = recall_experiences(reader, profile(), 1000.0, RecallQuery("attention", "s1", limit=5))

    tiers = {item.experience_id: (item.tier, item.differing) for item in recall.experiences}
    assert tiers == {
        "exp-same": (RecallTier.PRIOR, ()),
        "exp-other-gpu": (RecallTier.REVERIFY, ("gpu", "isl")),
        "exp-no-bottleneck": (RecallTier.REVERIFY, ("bottleneck",)),
    }
    assert recall.rendered_refs == ("exp-same", "exp-other-gpu", "exp-no-bottleneck")
    assert recall.experiences[0].gain_pct == pytest.approx(10.0)
    assert [item.experience_id for item in recall.of(RecallTier.PRIOR)] == ["exp-same"]


def test_a_summary_without_an_identity_is_one_to_reverify_never_a_prior() -> None:
    bare = {key: value for key, value in experience_summary("exp-bare", SAME).items() if key != "identity"}
    reader = FakeExperienceReader(completed_read(bare))

    (item,) = recall_experiences(reader, profile(), 1000.0, ATTENTION).experiences

    assert (item.tier, item.differing) == (RecallTier.REVERIFY, CONDITION_KEYS)


def test_another_model_tp_or_concurrency_makes_an_experience_one_to_reverify() -> None:
    reader = FakeExperienceReader(
        completed_read(
            experience_summary("exp-other-model", {**SAME, "model": "Qwen/Qwen3-8B"}),
            experience_summary("exp-other-tp-conc", {**SAME, "tp": 4, "conc": 128.0}),
            experience_summary("exp-same-conc-as-float", {**SAME, "conc": 64.0}),
        )
    )

    tiers = {
        item.experience_id: (item.tier, item.differing)
        for item in recall_experiences(reader, profile(), 1000.0, ATTENTION).experiences
    }

    assert tiers == {
        "exp-other-model": (RecallTier.REVERIFY, ("model",)),
        "exp-other-tp-conc": (RecallTier.REVERIFY, ("tp", "conc")),
        "exp-same-conc-as-float": (RecallTier.PRIOR, ()),
    }


def _bare(experience_id: str) -> dict[str, object]:
    return {key: value for key, value in experience_summary(experience_id, SAME).items() if key != "identity"}


def test_a_summary_without_an_identity_is_sorted_by_the_identity_the_export_holds() -> None:
    export = FakeExport(
        [
            {"id": "exp-a", "identity": SAME},
            {"id": "exp-b", "identity": {**SAME, "gpu": "mi300x"}},
            {"id": "exp-c", "identity": SAME},
        ]
    )
    index = IdentityIndex(export, "schema:sha256:" + "0" * 64)
    reader = FakeExperienceReader(completed_read(_bare("exp-a"), _bare("exp-b")))

    first = recall_experiences(reader, profile(), 1000.0, ATTENTION, identities=index)
    export.records.append({"id": "exp-d", "identity": SAME})
    reader.result = completed_read(_bare("exp-c"), _bare("exp-d"))
    later = recall_experiences(reader, profile(), 1000.0, ATTENTION, identities=index)

    assert [(item.experience_id, item.tier, item.differing) for item in first.experiences] == [
        ("exp-a", RecallTier.PRIOR, ()),
        ("exp-b", RecallTier.REVERIFY, ("gpu",)),
    ]
    assert [item.tier for item in later.experiences] == [RecallTier.PRIOR, RecallTier.PRIOR]
    # The whole export once, two records a page; then only from where it stopped, for exp-d.
    assert export.afters == [0, 2, 3]


def test_a_recalled_experience_names_the_layer_of_its_change_family_from_the_summary_or_the_export() -> None:
    export = FakeExport(
        [
            {"id": "exp-config", "identity": SAME, "change": {"change_family": "config_variant"}},
            {"id": "exp-patch", "identity": SAME, "change": {"change_family": "source_patch"}},
            {"id": "exp-unknown", "identity": SAME},
        ]
    )
    named = {**experience_summary("exp-named", SAME), "change_family": "source_patch"}
    reader = FakeExperienceReader(
        completed_read(
            experience_summary("exp-config", SAME),
            experience_summary("exp-patch", SAME),
            experience_summary("exp-unknown", SAME),
            named,
        )
    )

    recall = recall_experiences(reader, profile(), 1000.0, ATTENTION, identities=IdentityIndex(export))

    assert {item.experience_id: item.layer for item in recall.experiences} == {
        "exp-config": IntentLayer.CONFIG,
        "exp-patch": IntentLayer.SOURCE,
        "exp-unknown": None,
        "exp-named": IntentLayer.SOURCE,
    }


def test_an_export_that_fails_leaves_the_experience_one_to_reverify(caplog: pytest.LogCaptureFixture) -> None:
    index = IdentityIndex(FakeExport(failing=True))
    reader = FakeExperienceReader(completed_read(_bare("exp-a")))

    with caplog.at_level(logging.WARNING, logger="hyperloom_multibranch.notebook.knowledge.recall"):
        (item,) = recall_experiences(reader, profile(), 1000.0, ATTENTION, identities=index).experiences

    assert (item.tier, item.differing) == (RecallTier.REVERIFY, CONDITION_KEYS)
    assert "unreachable" in caplog.text


def test_an_index_is_built_only_on_a_reader_whose_service_exports() -> None:
    remote = SimpleNamespace(client=FakeExport(), schema_ref="schema:sha256:" + "1" * 64)

    assert isinstance(IdentityIndex.of(remote), IdentityIndex)
    assert IdentityIndex.of(FakeExperienceReader(completed_read())) is None
    assert IdentityIndex.of(None) is None


def test_the_read_carries_the_sessions_identity_workload_baseline_and_bottleneck() -> None:
    reader = FakeExperienceReader(completed_read())

    recall_experiences(reader, profile(), 1000.0, RecallQuery("attention", "s1", "fp8 kv cache", limit=5))

    ((decision, context, limit),) = reader.calls
    assert (decision, limit) == (DECISION, 5)
    assert context["identity"]["gpu"] == "MI355X" and "isl" not in context["identity"]
    assert context["workload"] == {"tp": 8, "conc": 64, "isl": 1024, "osl": 1024}
    assert context["benchmark_baseline"]["throughput"] == 1000.0
    assert context["observations"] == {"bottleneck": "attention", "branch": "s1", "intent": "fp8 kv cache"}


def test_without_a_kb_or_with_a_failed_read_the_recall_says_so_and_still_renders() -> None:
    disabled = recall_experiences(None, profile(), None, ATTENTION)
    failed = recall_experiences(
        FakeExperienceReader(RemoteReadResult("", "unavailable", "", (), ("planner_timeout",))),
        profile(),
        None,
        RecallQuery("x"),
    )
    nothing = recall_experiences(FakeExperienceReader(completed_read()), profile(), None, RecallQuery("x"))

    assert disabled.render() == ["- unavailable (disabled)"]
    assert failed.render() == ["- unavailable (unavailable: planner_timeout)"]
    assert nothing.render()[-1] == "- nothing matched"


def test_the_rendering_groups_the_tiers_and_names_what_differs() -> None:
    no_values = {**experience_summary("exp-bare", {**SAME, "gpu": "mi300x"}), "baseline_value": None}
    reader = FakeExperienceReader(completed_read(experience_summary("exp-same", SAME), no_values))

    lines = recall_experiences(reader, profile(hardware="mi355x"), 1000.0, ATTENTION).render()

    assert lines[1:] == [
        "priors, measured under this session's conditions:",
        "- --kv-cache-dtype fp8: keep +10.00% [exp-same]",
        "to re-verify, measured under other conditions:",
        "- --kv-cache-dtype fp8: keep n/a (gpu differ) [exp-bare]",
    ]
