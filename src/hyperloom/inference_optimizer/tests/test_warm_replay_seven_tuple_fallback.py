# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Remote seven-tuple warm-start fallback policy."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.knowledge.recipe_kb_t0 import (
    _cascade_warm_start_search,
    _framework_version_is_compatible,
    _hardware_fallback_values,
    _hardware_is_compatible,
    _parse_hardware_topology,
    _precision_is_compatible,
    _rank_warm_candidates,
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

    def get_recipe(self, **_kwargs: Any) -> dict[str, Any] | None:
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
        hw=TARGET["hardware"],
        framework=TARGET["framework_name"],
        model_type_val=TARGET["model_type"],
        architectures_val=["Qwen3ForCausalLM"],
        arch_slug=TARGET["architectures"],
        fw_version=TARGET["framework_version"],
        precision=TARGET["precision"],
        warm_prefer=None,
        target_conc=64,
        target_isl=128,
        target_osl=128,
    )


@pytest.mark.parametrize(
    ("hardware", "expected"),
    [
        ("mi300x", ("mi300x", "gfx942", "")),
        (
            "MI325X_ws16_pd1p1d_tp8_ep2_rayjob",
            (
                "mi325x",
                "gfx942",
                "_ws16_pd1p1d_tp8_ep2_rayjob",
            ),
        ),
        ("mi355x_ws8_tp8_infera", ("mi355x", "gfx950", "_ws8_tp8_infera")),
        ("h100", None),
        ("mi300x_tp8", None),
    ],
)
def test_hardware_topology_parser(
    hardware: str,
    expected: tuple[str, str, str] | None,
) -> None:
    assert _parse_hardware_topology(hardware) == expected


def test_hardware_fallback_preserves_topology_and_isa() -> None:
    values = _hardware_fallback_values(TARGET["hardware"])
    assert values == [
        "mi300x_ws16_pd1p1d_tp8_ep2_rayjob",
        "mi308x_ws16_pd1p1d_tp8_ep2_rayjob",
        "mi325x_ws16_pd1p1d_tp8_ep2_rayjob",
    ]
    assert _hardware_is_compatible(
        TARGET["hardware"],
        "mi325x_ws16_pd1p1d_tp8_ep2_rayjob",
    )
    assert not _hardware_is_compatible(
        TARGET["hardware"],
        "mi325x_ws8_pd1p1d_tp8_ep2_rayjob",
    )
    assert not _hardware_is_compatible(TARGET["hardware"], "mi355x_ws16_pd1p1d_tp8_ep2_rayjob")
    assert not _hardware_is_compatible("h100", "h200")


@pytest.mark.parametrize(
    ("target", "candidate", "compatible"),
    [
        ("bf16", "bf16", True),
        ("bf16", "fp16", True),
        ("fp16", "bf16", True),
        ("fp8", "fp8", True),
        ("fp8", "bf16", False),
        ("mxfp4", "mxfp8", False),
        ("int8", "awq", False),
        ("gptq", "awq", False),
        ("unknown_precision", "bf16", False),
    ],
)
def test_precision_whitelist(
    target: str,
    candidate: str,
    compatible: bool,
) -> None:
    assert _precision_is_compatible(target, candidate) is compatible


@pytest.mark.parametrize(
    ("target", "candidate", "compatible"),
    [
        ("1.2.5", "1.2.5", True),
        ("1.2.5", "1.2.4", True),
        ("1.2.5", "1.2.6", False),
        ("1.2.5", "1.1.99", False),
        ("1.2.5", "2.2.1", False),
        ("unknown_version", "1.2.4", False),
    ],
)
def test_framework_semver_direction_and_minor(
    target: str,
    candidate: str,
    compatible: bool,
) -> None:
    assert _framework_version_is_compatible(target, candidate) is compatible


def test_exact_and_each_fallback_confidence_order() -> None:
    exact = _candidate(TARGET["model"])
    exact_kb = _RemoteKB([[_candidate("must-not-run")]], exact=exact)
    assert _cascade(exact_kb)[1:] == ("exact", 1.0)
    assert exact_kb.search_calls == []

    cases = [
        ([[_candidate("alias-a")]], ("same_arch_class", 0.95)),
        (
            [[], [_candidate("alias-a", hardware="mi325x_ws16_pd1p1d_tp8_ep2_rayjob")]],
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
                        precision="fp16",
                    )
                ],
            ],
            ("compatible_precision", 0.78),
        ),
        (
            [
                [],
                [],
                [],
                [
                    _candidate(
                        "alias-a",
                        hardware="mi325x_ws16_pd1p1d_tp8_ep2_rayjob",
                        precision="fp16",
                        framework_version="1.2.4",
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
    assert row["exact_history"]["what_worked"] == [
        {"name": "exact-prior"}
    ]
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
    row, tier, _confidence = _cascade(
        _RemoteKB([[unrelated, compatible]])
    )
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
    row, tier, confidence = _cascade(
        _RemoteKB([[], [], [], [valid_old, newer, wrong_minor, valid_near]])
    )
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


def test_hardware_search_passes_exact_same_isa_values() -> None:
    kb = _RemoteKB([[], [], [], []])
    assert _cascade(kb)[1:] == ("miss", 0.0)
    assert "hardware_in" not in kb.search_calls[0]
    for call in kb.search_calls[1:]:
        assert call["hardware_in"] == _hardware_fallback_values(
            TARGET["hardware"]
        )
