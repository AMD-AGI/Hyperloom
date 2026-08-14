# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Warm-replay BORROWED config-donor trustworthiness gate.

Covers the donor acceptance gate that prevents cross-model warm-replay from
borrowing configs that are evidence-free (zero validated gain), cross/unknown
architecture, or workload-shape incompatible — the empirical root causes of
neutral/negative warm-replay gains.
"""

from __future__ import annotations

from typing import Any

from hyperloom.orchestrator.knowledge.recipe_kb_t0 import (
    _build_warm_start_context,
    _cascade_warm_start_search,
    _donor_is_trustworthy,
    _find_config_donor,
)


def _donor(
    *,
    canonical_id: str = "donor-cid",
    arch: list[str] | None = None,
    model_type: str = "qwen2",
    gain: float = 12.5,
    hardware: str = "mi300x",
    framework_version: str = "1.2.5",
    precision: str = "bf16",
    conc: Any = 64,
    isl: Any = 128,
    osl: Any = 128,
    with_config: bool = True,
) -> dict[str, Any]:
    """Build a minimal donor recipe row for gate tests."""
    row: dict[str, Any] = {
        "canonical_id": canonical_id,
        "architectures": ["Qwen2ForCausalLM"] if arch is None else arch,
        "model_type": model_type,
        "hardware": hardware,
        "framework_version": framework_version,
        "precision": precision,
        "validated_gain_pct": gain,
        "conc": conc,
        "isl": isl,
        "osl": osl,
    }
    if with_config:
        row["best_config"] = {"extra_server_args": "--enable-foo", "extra_envs": {"X": "1"}}
    return row


_TARGET = {
    "target_arch_slug": "qwen2forcausallm",
    "target_model_type": "qwen2",
    "target_conc": 64,
    "target_isl": 128,
    "target_osl": 128,
}


def test_trustworthy_donor_accepted() -> None:
    assert _donor_is_trustworthy(_donor(), **_TARGET) is True


class _StubKB:
    """A search-only KB stub returning a fixed donor row list."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def search(self, *, label_match: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:  # noqa: ARG002
        return list(self._rows)


class _BatchKB:
    """Return one result batch per degradation tier."""

    def __init__(self, batches: list[list[dict[str, Any]]]) -> None:
        self._batches = list(batches)

    def search(
        self,
        *,
        label_match: dict[str, Any],
        limit: int = 100,
    ) -> list[dict[str, Any]]:  # noqa: ARG002
        return self._batches.pop(0) if self._batches else []


def _find_kwargs(**over: Any) -> dict[str, Any]:
    base = {
        "cid": "self-cid",
        "hardware": "mi300x",
        "framework": "sglang",
        "model_type": "qwen2",
        "arch_slug": "qwen2forcausallm",
        "framework_version": "v1",
        "precision": "bf16",
        "target_conc": 64,
        "target_isl": 128,
        "target_osl": 128,
    }
    base.update(over)
    return base


def test_find_config_donor_skips_untrustworthy() -> None:
    kb = _StubKB(
        [
            _donor(canonical_id="untrusted", gain=0.0),
            _donor(canonical_id="trusted", gain=20.0),
        ]
    )
    donor, tier, conf = _find_config_donor(kb, **_find_kwargs())
    assert donor is not None
    assert donor["validated_gain_pct"] == 20.0
    assert tier == "same_arch_class"
    assert conf == 0.95


def test_find_config_donor_returns_none_when_all_untrustworthy() -> None:
    kb = _StubKB([_donor(gain=0.0), _donor(arch=["LlamaForCausalLM"], gain=30.0)])
    donor, tier, conf = _find_config_donor(kb, **_find_kwargs())
    assert donor is None
    assert tier == ""
    assert conf == 0.0


def test_find_config_donor_skips_self_cid() -> None:
    kb = _StubKB([_donor(gain=20.0)])
    donor, _tier, _conf = _find_config_donor(kb, **_find_kwargs(cid="donor-cid"))
    assert donor is None


def test_find_config_donor_uses_framework_version_fallback() -> None:
    compatible = _donor(
        hardware="mi325x",
        precision="fp16",
        framework_version="1.2.4",
        gain=20.0,
    )
    kb = _BatchKB([[], [], [], [compatible]])

    donor, tier, conf = _find_config_donor(
        kb,
        **_find_kwargs(
            hardware="mi300x",
            framework_version="1.2.5",
        ),
    )

    assert donor is compatible
    assert (tier, conf) == ("compatible_framework_version", 0.72)


def test_remote_cascade_skips_unproven_and_wrong_structure() -> None:
    exact = {
        "canonical_id": "self-cid",
        "replayable": False,
        "what_worked": [{"name": "target-prior"}],
        "sessions": [{"gain_pct": 80.0}],
        "validated_gain_pct": 80.0,
    }
    unproven = {
        **_donor(gain=0.0),
        "canonical_id": "unproven",
        "replayable": True,
        "view_source": "current",
    }
    empty = {
        **_donor(gain=25.0, with_config=False),
        "canonical_id": "empty",
        "replayable": True,
        "view_source": "current",
        "replay_config_available": False,
    }
    wrong_structure = {
        **_donor(arch=["LlamaForCausalLM"], gain=30.0),
        "canonical_id": "wrong-structure",
        "replayable": True,
        "view_source": "current",
    }
    selected = {
        **_donor(gain=12.0),
        "canonical_id": "selected",
        "replayable": True,
        "view_source": "current",
        "sessions": [{"gain_pct": 12.0}],
    }

    class _RemoteKB:
        def __init__(self) -> None:
            self.selected: list[str] = []

        def get_recipe(self, **_kwargs: Any) -> dict[str, Any]:
            return exact

        def search(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return [empty, unproven, wrong_structure, selected]

        def select_candidate(self, row: dict[str, Any]) -> bool:
            self.selected.append(str(row["canonical_id"]))
            return True

    kb = _RemoteKB()
    row, tier, confidence = _cascade_warm_start_search(
        kb,  # type: ignore[arg-type]
        cid="self-cid",
        hw="mi300x",
        framework="sglang",
        model_type_val="qwen2",
        architectures_val=["Qwen2ForCausalLM"],
        arch_slug="qwen2forcausallm",
        fw_version="v1",
        precision="bf16",
        warm_prefer=None,
        target_conc=64,
        target_isl=128,
        target_osl=128,
    )

    assert row["canonical_id"] == "selected"
    assert row["validated_gain_pct"] == 12.0
    assert row["sessions"][0]["gain_pct"] == 12.0
    assert row["exact_history"]["validated_gain_pct"] == 80.0
    assert row["exact_history"]["sessions"][0]["gain_pct"] == 80.0
    assert tier == "same_arch_class"
    assert confidence == 0.95
    assert kb.selected == ["selected"]
    context = _build_warm_start_context(
        status="hit",
        tier=tier,
        confidence=confidence,
        canonical_id="self-cid",
        source="kb-store",
        recipe=row,
    )
    assert context["proven_prior"] == [{"name": "target-prior"}]
