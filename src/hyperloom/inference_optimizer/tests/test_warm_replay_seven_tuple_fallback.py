# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Remote seven-tuple warm-start fallback policy."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.knowledge.recipe_kb_t0 import (
    _cascade_warm_start_search,
    _donor_is_trustworthy,
    _framework_version_is_compatible,
    _hardware_is_compatible,
    _rank_warm_candidates,
    _topology_suffix,
)
from hyperloom.orchestrator.knowledge.recipe_kb.local_store import (
    _matches_labels,
)


TARGET = {
    "model": "checkpoint-a",
    "hardware": "mi300x_ws16_pd1p1d_tp8_ep2_rayjob",
    "framework_name": "sglang",
    "model_type": "qwen",
    "architectures": "qwen3forcausallm",
    "framework_version": "1.2.5",
    "precision": "bf16",
}
TARGET_CID = "inference:" + ":".join(TARGET.values())
TARGET_IDENTITY = {**TARGET, "canonical_id": TARGET_CID, "gfx_arch": "gfx942"}
TARGET_SHAPE = {"conc": 64, "isl": 128, "osl": 128}


def _candidate(
    model: str,
    *,
    hardware: str = TARGET["hardware"],
    framework_name: str = TARGET["framework_name"],
    model_type: str = TARGET["model_type"],
    architectures: str = TARGET["architectures"],
    framework_version: str = TARGET["framework_version"],
    precision: str = TARGET["precision"],
    gain: float = 10.0,
    updated_at: str = "2026-08-13T00:00:00Z",
    **extra: Any,
) -> dict[str, Any]:
    dimensions = {
        "model": model,
        "hardware": hardware,
        "framework_name": framework_name,
        "model_type": model_type,
        "architectures": architectures,
        "framework_version": framework_version,
        "precision": precision,
    }
    return {
        "canonical_id": "inference:" + ":".join(dimensions.values()),
        **dimensions,
        "conc": 64,
        "isl": 128,
        "osl": 128,
        "replayable": True,
        "view_source": "current",
        "replay_material_available": True,
        "validated_gain_pct": gain,
        "updated_at": updated_at,
        **extra,
    }


class _RemoteKB:
    mode = "remote"

    def __init__(
        self,
        searches: list[list[dict[str, Any]]],
        exact: dict[str, Any] | None = None,
    ) -> None:
        self.searches = list(searches)
        self.exact = exact
        self.search_calls: list[dict[str, Any]] = []
        self.selected: list[str] = []

    def get_recipe(self, *, canonical_id: str) -> dict[str, Any] | None:
        return self.exact

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.search_calls.append(dict(kwargs))
        return self.searches.pop(0) if self.searches else []

    def select_candidate(self, row: dict[str, Any]) -> bool:
        self.selected.append(str(row["canonical_id"]))
        return True


def _cascade(kb: _RemoteKB) -> tuple[dict[str, Any], str, float]:
    return _cascade_warm_start_search(
        kb,  # type: ignore[arg-type]
        cid=TARGET_CID,
        identity=TARGET_IDENTITY,
        shape=TARGET_SHAPE,
    )


@pytest.mark.parametrize(
    ("hardware", "suffix"),
    [
        ("mi300x", ""),
        ("mi325x_ws16_pd1p1d_tp8_ep2_rayjob", "_ws16_pd1p1d_tp8_ep2_rayjob"),
        ("mi355x_ws8_tp8_infera", "_ws8_tp8_infera"),
        ("mi300x_ep8_cpx", "_ep8_cpx"),
        ("mi300x_cpx", "_cpx"),
        ("radeon8060s", ""),
        # A single node never encodes tp, so this tail is part of the label.
        ("mi300x_tp8", ""),
    ],
)
def test_topology_suffix_is_what_the_slug_ends_in(hardware: str, suffix: str) -> None:
    assert _topology_suffix(hardware) == suffix


def _row(hardware: str, gfx_arch: str = "") -> dict[str, str]:
    return {"hardware": hardware, "gfx_arch": gfx_arch}


def test_same_slug_is_compatible_whatever_the_rows_record() -> None:
    assert _hardware_is_compatible(_row(TARGET["hardware"]), _row(TARGET["hardware"]))
    assert _hardware_is_compatible(_row("MI300X", "gfx942"), _row("mi300x"))
    assert not _hardware_is_compatible(_row(""), _row(""))


def test_another_label_is_compatible_only_on_the_same_recorded_arch_and_topology() -> None:
    target = _row(TARGET["hardware"], "gfx942")
    assert _hardware_is_compatible(target, _row("mi325x_ws16_pd1p1d_tp8_ep2_rayjob", "gfx942"))
    assert _hardware_is_compatible(target, _row("mi300a_ws16_pd1p1d_tp8_ep2_rayjob", "GFX942"))
    assert not _hardware_is_compatible(target, _row("mi325x_ws8_pd1p1d_tp8_ep2_rayjob", "gfx942"))
    assert not _hardware_is_compatible(target, _row("mi355x_ws16_pd1p1d_tp8_ep2_rayjob", "gfx950"))
    assert _hardware_is_compatible(_row("mi300x_tp8", "gfx942"), _row("mi325x", "gfx942"))
    assert not _hardware_is_compatible(_row("mi300x_tp8", "gfx942"), _row("mi325x_cpx", "gfx942"))


def test_a_row_without_a_recorded_arch_matches_its_own_slug_only() -> None:
    assert not _hardware_is_compatible(_row("mi300x", "gfx942"), _row("mi325x"))
    assert not _hardware_is_compatible(_row("mi300x"), _row("mi325x", "gfx942"))
    assert not _hardware_is_compatible(_row("mi300x"), _row("mi325x"))


def test_candidate_hardware_falls_back_to_its_canonical_id() -> None:
    candidate = _candidate("alias-a", hardware="mi325x_ws16_pd1p1d_tp8_ep2_rayjob", gfx_arch="gfx942")
    candidate.pop("hardware")
    assert _hardware_is_compatible(TARGET_IDENTITY, candidate)


@pytest.mark.parametrize(
    ("target", "candidate", "compatible"),
    [
        ("1.2.5", "1.2.5", True),
        ("1.2.5", "1.2.4", True),
        ("1.2.5", "1.2.6", False),
        ("1.2.5", "1.1.99", False),
        ("1.2.5", "2.2.1", False),
        ("0.4.6.post5", "0.4.6.post4", True),
        ("0.11.0", "0.11.0rc2", True),
        ("0.10.1.dev2+gabc", "0.10.1.dev1+gdef", True),
        ("unknown_version", "1.2.4", False),
    ],
)
def test_framework_semver_direction_and_minor(
    target: str,
    candidate: str,
    compatible: bool,
) -> None:
    assert _framework_version_is_compatible(target, candidate) is compatible


def test_local_framework_name_filter_matches_canonical_rows() -> None:
    assert _matches_labels(
        {"framework_name": "vllm"},
        {"framework_name": "vllm"},
    )


def test_borrowed_donor_requires_matching_workload_shape() -> None:
    donor = _candidate("shape-donor")
    assert _donor_is_trustworthy(donor, TARGET, TARGET_SHAPE)
    donor.pop("conc")
    assert not _donor_is_trustworthy(donor, TARGET, TARGET_SHAPE)


def test_exact_and_each_fallback_confidence_order() -> None:
    exact = _candidate(TARGET["model"])
    exact_kb = _RemoteKB([[_candidate("must-not-run")]], exact=exact)
    assert _cascade(exact_kb)[1:] == ("exact", 1.0)
    assert exact_kb.search_calls == []

    cases = [
        ([[_candidate("alias-a")]], ("same_arch_class", 0.95)),
        (
            [[], [_candidate("alias-a", hardware="mi325x_ws16_pd1p1d_tp8_ep2_rayjob", gfx_arch="gfx942")]],
            ("same_gpu_isa", 0.85),
        ),
        (
            [
                [],
                [],
                [
                    _candidate(
                        "alias-a",
                        hardware="mi325x_ws16_pd1p1d_tp8_ep2_rayjob",
                        framework_version="1.2.4",
                        gfx_arch="gfx942",
                    )
                ],
            ],
            ("compatible_framework_version", 0.72),
        ),
    ]
    for searches, expected in cases:
        assert _cascade(_RemoteKB(searches))[1:] == expected


def test_empty_exact_material_falls_back_before_runtime() -> None:
    exact = {
        **_candidate(TARGET["model"]),
        "replay_material_available": False,
        "what_worked": [{"name": "exact-prior"}],
    }
    donor = _candidate("alias-a")
    row, tier, confidence = _cascade(_RemoteKB([[donor]], exact=exact))
    assert row["canonical_id"] == donor["canonical_id"]
    assert row["exact_history"]["what_worked"] == [{"name": "exact-prior"}]
    assert (tier, confidence) == ("same_arch_class", 0.95)


def test_relaxed_model_allowed_but_unrelated_structure_rejected() -> None:
    unrelated = _candidate(
        "unrelated",
        model_type="llama",
        architectures="llamaforcausallm",
        gain=99.0,
    )
    compatible = _candidate(
        "checkpoint-a-mirror",
        gain=8.0,
    )
    row, tier, _confidence = _cascade(_RemoteKB([[unrelated, compatible]]))
    assert tier == "same_arch_class"
    assert row["canonical_id"] == compatible["canonical_id"]


def test_ranking_ignores_unverified_alias_metadata() -> None:
    rows = [
        _candidate("other", framework_version="1.2.4", gain=90.0),
        _candidate(
            "mirror",
            framework_version="1.2.3",
            gain=100.0,
            model_alias=TARGET["model"],
        ),
        _candidate(
            "mirror-newer",
            framework_version="1.2.4",
            gain=5.0,
            model_alias=TARGET["model"],
            updated_at="2026-08-12T00:00:00Z",
        ),
        _candidate(
            "mirror-newest",
            framework_version="1.2.4",
            gain=5.0,
            model_alias=TARGET["model"],
            updated_at="2026-08-13T00:00:00Z",
        ),
    ]
    ranked = _rank_warm_candidates(
        rows,
        target_framework_version=TARGET["framework_version"],
    )
    assert [row["model"] for row in ranked] == [
        "other",
        "mirror-newest",
        "mirror-newer",
        "mirror",
    ]


def test_final_tier_rejects_newer_and_wrong_minor_then_chooses_nearest() -> None:
    valid_old = _candidate("old", framework_version="1.2.3", gain=50.0)
    valid_near = _candidate("near", framework_version="1.2.4", gain=2.0)
    newer = _candidate("newer", framework_version="1.2.6", gain=100.0)
    wrong_minor = _candidate("minor", framework_version="1.1.99", gain=100.0)
    row, tier, confidence = _cascade(_RemoteKB([[], [], [valid_old, newer, wrong_minor, valid_near]]))
    assert row["canonical_id"] == valid_near["canonical_id"]
    assert (tier, confidence) == ("compatible_framework_version", 0.72)


def test_selected_tier_never_searches_lower_runtime_fallbacks() -> None:
    kb = _RemoteKB(
        [
            [_candidate("selected")],
            [_candidate("must-not-be-read", hardware="mi325x")],
        ]
    )
    row, tier, confidence = _cascade(kb)
    assert row["model"] == "selected"
    assert (tier, confidence) == ("same_arch_class", 0.95)
    assert len(kb.search_calls) == 1
    assert kb.selected == [row["canonical_id"]]


def test_relaxed_tiers_search_every_hardware_and_filter_on_the_recorded_arch() -> None:
    kb = _RemoteKB([[], [], []])
    assert _cascade(kb)[1:] == ("miss", 0.0)
    assert kb.search_calls[0]["label_match"]["hardware"] == TARGET["hardware"]
    for call in kb.search_calls[1:]:
        assert set(call) == {"label_match", "limit"}
        assert "hardware" not in call["label_match"]


def test_relaxed_tier_skips_rows_on_another_or_unrecorded_arch() -> None:
    sibling = "mi325x_ws16_pd1p1d_tp8_ep2_rayjob"
    unrecorded = _candidate("unrecorded", hardware=sibling, gain=90.0)
    other_arch = _candidate("other-arch", hardware=sibling, gfx_arch="gfx950", gain=80.0)
    same_arch = _candidate("same-arch", hardware=sibling, gfx_arch="gfx942", gain=5.0)
    row, tier, _confidence = _cascade(_RemoteKB([[], [unrecorded, other_arch, same_arch]]))
    assert row["model"] == "same-arch"
    assert tier == "same_gpu_isa"


def test_a_session_without_a_card_borrows_nothing_across_labels() -> None:
    sibling = _candidate("alias-a", hardware="mi325x_ws16_pd1p1d_tp8_ep2_rayjob", gfx_arch="gfx942")
    row, tier, _confidence = _cascade_warm_start_search(
        _RemoteKB([[], [sibling], [sibling]]),  # type: ignore[arg-type]
        cid=TARGET_CID,
        identity={**TARGET_IDENTITY, "gfx_arch": ""},
        shape=TARGET_SHAPE,
    )
    assert (row, tier) == ({}, "miss")
