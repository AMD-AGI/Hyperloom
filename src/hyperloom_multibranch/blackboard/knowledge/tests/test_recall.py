# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recall sorts what other sessions published into priors and leads to re-verify, by their conditions."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from hyperloom_kb import RemoteReadResult

from hyperloom.common.perf_metric import TOTAL_TOKEN_THROUGHPUT
from hyperloom_multibranch.contracts.blackboard.peers import IntentLayer
from hyperloom_multibranch.blackboard.knowledge.recall import (
    CONDITION_KEYS,
    INTERACTIVITY_OBJECTIVE,
    RECALL_LIMIT,
    THROUGHPUT_OBJECTIVE,
    TOTAL_THROUGHPUT_OBJECTIVE,
    IdentityIndex,
    RecallQuery,
    RecallTier,
    recall_experiences,
)
from hyperloom_multibranch.blackboard.knowledge.tests.fakes import (
    FakeExperienceReader,
    FakeExport,
    completed_read,
    experience_summary,
)
from hyperloom_multibranch.blackboard.tests.factories import profile

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

    recall = recall_experiences(reader, profile(), ATTENTION)

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

    (item,) = recall_experiences(reader, profile(), ATTENTION).experiences

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
        for item in recall_experiences(reader, profile(), ATTENTION).experiences
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
            {"id": "exp-a", "identity": SAME, "objective": THROUGHPUT_OBJECTIVE},
            {"id": "exp-b", "identity": {**SAME, "gpu": "mi300x"}, "objective": THROUGHPUT_OBJECTIVE},
            {"id": "exp-c", "identity": SAME, "objective": THROUGHPUT_OBJECTIVE},
        ]
    )
    index = IdentityIndex(export, "schema:sha256:" + "0" * 64)
    reader = FakeExperienceReader(completed_read(_bare("exp-a"), _bare("exp-b")))

    first = recall_experiences(reader, profile(), ATTENTION, identities=index)
    export.records.append({"id": "exp-d", "identity": SAME, "objective": THROUGHPUT_OBJECTIVE})
    reader.result = completed_read(_bare("exp-c"), _bare("exp-d"))
    later = recall_experiences(reader, profile(), ATTENTION, identities=index)

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
            {"id": "exp-kernel", "identity": SAME, "change": {"change_family": "kernel_artifact"}},
            {"id": "exp-unknown", "identity": SAME},
        ]
    )
    named = {**experience_summary("exp-named", SAME), "change_family": "source_patch"}
    reader = FakeExperienceReader(
        completed_read(
            experience_summary("exp-config", SAME),
            experience_summary("exp-patch", SAME),
            experience_summary("exp-kernel", SAME),
            experience_summary("exp-unknown", SAME),
            named,
        )
    )

    recall = recall_experiences(reader, profile(), ATTENTION, identities=IdentityIndex(export))

    assert {item.experience_id: item.layer for item in recall.experiences} == {
        "exp-config": IntentLayer.CONFIG,
        "exp-patch": IntentLayer.SOURCE,
        "exp-kernel": IntentLayer.KERNEL,
        "exp-unknown": None,
        "exp-named": IntentLayer.SOURCE,
    }


def test_an_export_that_fails_leaves_the_experience_one_to_reverify(caplog: pytest.LogCaptureFixture) -> None:
    index = IdentityIndex(FakeExport(failing=True))
    reader = FakeExperienceReader(completed_read(_bare("exp-a")))

    with caplog.at_level(logging.WARNING, logger="hyperloom_multibranch.blackboard.knowledge.recall"):
        (item,) = recall_experiences(reader, profile(), ATTENTION, identities=index).experiences

    assert (item.tier, item.differing) == (RecallTier.REVERIFY, (*CONDITION_KEYS, "objective"))
    assert "unreachable" in caplog.text


def test_an_experience_on_another_objective_is_to_reverify_after_the_rest_and_names_its_objective() -> None:
    export = FakeExport(
        [
            {"id": "exp-throughput", "identity": SAME, "objective": THROUGHPUT_OBJECTIVE},
            {"id": "exp-interactivity", "identity": SAME, "objective": INTERACTIVITY_OBJECTIVE},
        ]
    )
    names = ("exp-throughput", "exp-interactivity", "exp-elsewhere")
    reader = FakeExperienceReader(completed_read(*(experience_summary(name, SAME) for name in names)))

    inferencex = recall_experiences(reader, profile(), ATTENTION, identities=IdentityIndex(export))
    agentx = recall_experiences(reader, profile(isl=None, osl=None), ATTENTION, identities=IdentityIndex(export))

    assert [(item.experience_id, item.tier, item.differing) for item in inferencex.experiences] == [
        ("exp-throughput", RecallTier.PRIOR, ()),
        ("exp-elsewhere", RecallTier.REVERIFY, ("objective",)),
        ("exp-interactivity", RecallTier.REVERIFY, ("objective",)),
    ]
    assert [(item.experience_id, item.tier) for item in agentx.experiences] == [
        ("exp-interactivity", RecallTier.PRIOR),
        ("exp-elsewhere", RecallTier.REVERIFY),
        ("exp-throughput", RecallTier.REVERIFY),
    ]
    assert agentx.experiences[-1].describe() == (
        "--kv-cache-dtype fp8: keep +10.00% on e2e_throughput@v1 (objective differ) [exp-throughput]"
    )


def test_an_index_is_built_only_on_a_reader_whose_service_exports() -> None:
    remote = SimpleNamespace(client=FakeExport(), schema_ref="schema:sha256:" + "1" * 64)

    assert isinstance(IdentityIndex.of(remote), IdentityIndex)
    assert IdentityIndex.of(FakeExperienceReader(completed_read())) is None
    assert IdentityIndex.of(None) is None


def _sent(reader: FakeExperienceReader) -> list[dict[str, object]]:
    """What each read asked for: the value of every exact signal by its field, and the texts."""

    return [
        {
            **{signal.field: signal.value for signal in signals if signal.field},
            "text": [s.text for s in signals if s.text],
        }
        for signals, _ in reader.calls
    ]


def test_the_read_carries_the_sessions_conditions_objective_and_bottleneck_weighed_as_the_framework_phase() -> None:
    reader = FakeExperienceReader(completed_read())

    recall_experiences(reader, profile(), ATTENTION)

    ((signals, limit),) = reader.calls
    (sent,) = _sent(reader)
    assert limit == RECALL_LIMIT
    assert sent["identity.gpu"] == "MI355X" and sent["objective"] == THROUGHPUT_OBJECTIVE
    assert {field: sent[field] for field in ("identity.tp", "identity.conc", "identity.isl", "identity.osl")} == {
        "identity.tp": 8,
        "identity.conc": 64,
        "identity.isl": 1024,
        "identity.osl": 1024,
    }
    assert sent["text"] == ["attention"]
    weights = {signal.field or signal.text: signal.weight for signal in signals}
    assert (weights["identity.model"], weights["identity.tp"], weights["attention"]) == (1.0, 0.4, 0.6)


def test_an_agentx_session_asks_for_the_objective_it_is_graded_on() -> None:
    reader = FakeExperienceReader(completed_read())

    recall_experiences(reader, profile(isl=None, osl=None), ATTENTION)
    recall_experiences(reader, profile(isl=None, osl=None), RecallQuery("attention", metric=TOTAL_TOKEN_THROUGHPUT))
    recall_experiences(reader, profile(), ATTENTION)

    assert [sent["objective"] for sent in _sent(reader)] == [
        INTERACTIVITY_OBJECTIVE,
        TOTAL_THROUGHPUT_OBJECTIVE,
        THROUGHPUT_OBJECTIVE,
    ]


def test_without_a_kb_or_with_a_failed_read_the_recall_says_so_and_still_renders() -> None:
    disabled = recall_experiences(None, profile(), ATTENTION)
    failed = recall_experiences(
        FakeExperienceReader(RemoteReadResult("", "unavailable", "", (), ("timed out",))),
        profile(),
        RecallQuery("x"),
    )
    nothing = recall_experiences(FakeExperienceReader(completed_read()), profile(), RecallQuery("x"))

    assert disabled.render() == ["- unavailable (disabled)"]
    assert failed.render() == ["- unavailable (unavailable: timed out)"]
    assert nothing.render()[-1] == "- nothing matched"


def test_the_rendering_groups_the_tiers_and_names_what_differs() -> None:
    no_values = {**experience_summary("exp-bare", {**SAME, "gpu": "mi300x"}), "baseline_value": None}
    reader = FakeExperienceReader(completed_read(experience_summary("exp-same", SAME), no_values))

    lines = recall_experiences(reader, profile(hardware="mi355x"), ATTENTION).render()

    assert lines[1:] == [
        "priors, measured under this session's conditions:",
        "- --kv-cache-dtype fp8: keep +10.00% [exp-same]",
        "to re-verify, measured under other conditions:",
        "- --kv-cache-dtype fp8: keep n/a (gpu differ) [exp-bare]",
    ]
