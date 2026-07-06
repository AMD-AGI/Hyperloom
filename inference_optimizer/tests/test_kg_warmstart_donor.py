# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""KG-native cross-model warm-start donor synthesis (single_top).

Covers ``generate_warmstart_donor_graph_guided`` (kg_client) and the
``_kg_native_config_donor`` cortex_t0 wiring: the strongest positive-gain,
non-reverted ``KNOB_IMPROVES`` edge for the target arch+precision is adopted as
a recipe-shaped donor; reverted / zero-gain / config-less / empty cases yield
``None`` so warm-start falls back to the recipe-KB sibling search.
"""

from __future__ import annotations

from typing import Any

from inference_optimizer.recipe_kb.kg_client import (
    Fact,
    generate_warmstart_donor_graph_guided,
)


def _knob_fact(
    *,
    fp: str,
    obj: str = "qwen2forcausallm+fp8",
    gain: str = "+20%",
    args: str = "--enable-foo",
    envs_json: str = '{"X": "1"}',
    name: str = "foo",
    keep_n: str = "5",
    predicate: str = "KNOB_IMPROVES",
) -> Fact:
    """Build a KNOB_IMPROVES/KNOB_REVERTED_ON fact for the fake KG."""
    return Fact(
        subject=fp,
        predicate=predicate,
        object=obj,
        properties={
            "gain": gain,
            "args": args,
            "envs": envs_json,
            "name": name,
            "keep_n": keep_n,
        },
    )


class _FakeKG:
    """Duck-typed KG exposing only query_facts_safe (what the generator uses)."""

    def __init__(self, improves: list[Fact], reverted: list[Fact] | None = None) -> None:
        self._improves = improves
        self._reverted = reverted or []

    def query_facts_safe(self, **kwargs: Any) -> list[Fact]:
        preds = kwargs.get("predicate") or []
        if "KNOB_REVERTED_ON" in preds:
            return self._reverted
        if "KNOB_IMPROVES" in preds:
            return self._improves
        return []


_ARCHS = ["Qwen2ForCausalLM"]
_KW = {"architectures": _ARCHS, "precision": "fp8", "hardware": "mi300x", "framework": "sglang"}


def test_single_top_donor_picks_highest_gain() -> None:
    kg = _FakeKG([
        _knob_fact(fp="low", gain="+5%", args="--low"),
        _knob_fact(fp="high", gain="+40%", args="--high", envs_json='{"Y": "2"}'),
    ])
    donor = generate_warmstart_donor_graph_guided(kg, model_type="qwen2", **_KW)
    assert donor is not None
    assert donor["best_config"]["extra_server_args"] == "--high"
    assert donor["best_config"]["extra_envs"] == {"Y": "2"}
    assert donor["validated_gain_pct"] == 40.0
    assert donor["canonical_id"].startswith("kg-synth:qwen2forcausallm+fp8:")
    assert donor["provenance"]["source"] == "kg_native_warmstart"


def test_reverted_knob_excluded() -> None:
    # The only improving knob is also reverted → no donor.
    kg = _FakeKG(
        improves=[_knob_fact(fp="r1", gain="+30%")],
        reverted=[_knob_fact(fp="r1", predicate="KNOB_REVERTED_ON")],
    )
    assert generate_warmstart_donor_graph_guided(kg, **_KW) is None


def test_zero_gain_yields_none() -> None:
    kg = _FakeKG([_knob_fact(fp="z", gain="0%")])
    assert generate_warmstart_donor_graph_guided(kg, **_KW) is None


def test_configless_knob_yields_none() -> None:
    # Positive gain but neither args nor envs → not replayable.
    kg = _FakeKG([_knob_fact(fp="c", gain="+15%", args="", envs_json="")])
    assert generate_warmstart_donor_graph_guided(kg, **_KW) is None


def test_empty_kg_yields_none() -> None:
    assert generate_warmstart_donor_graph_guided(_FakeKG([]), **_KW) is None


def test_no_architectures_yields_none() -> None:
    kg = _FakeKG([_knob_fact(fp="x", gain="+20%")])
    out = generate_warmstart_donor_graph_guided(
        kg, architectures=[], precision="fp8", hardware="mi300x", framework="sglang"
    )
    assert out is None


def test_cortex_helper_requires_native_kg(monkeypatch) -> None:
    # _kg_native_config_donor must NOT borrow from a non-native (sim) client.
    from hyperloom.orchestrator import cortex_t0

    class _SimKG:
        _native = False

        def is_available(self) -> bool:
            return True

        def query_facts_safe(self, **kwargs: Any) -> list[Fact]:
            return [_knob_fact(fp="x", gain="+20%")]

    monkeypatch.setattr(
        "inference_optimizer.recipe_kb.kg_client.get_kg_client", lambda: _SimKG()
    )
    out = cortex_t0._kg_native_config_donor(
        architectures=_ARCHS, precision="fp8", hardware="mi300x", framework="sglang", model_type="qwen2"
    )
    assert out is None


def test_cortex_helper_returns_native_donor(monkeypatch) -> None:
    from hyperloom.orchestrator import cortex_t0

    class _NativeKG:
        _native = True

        def is_available(self) -> bool:
            return True

        def query_facts_safe(self, **kwargs: Any) -> list[Fact]:
            preds = kwargs.get("predicate") or []
            if "KNOB_IMPROVES" in preds:
                return [_knob_fact(fp="x", gain="+25%", args="--win")]
            return []

    monkeypatch.setattr(
        "inference_optimizer.recipe_kb.kg_client.get_kg_client", lambda: _NativeKG()
    )
    out = cortex_t0._kg_native_config_donor(
        architectures=_ARCHS, precision="fp8", hardware="mi300x", framework="sglang", model_type="qwen2"
    )
    assert out is not None
    assert out["best_config"]["extra_server_args"] == "--win"
    assert out["validated_gain_pct"] == 25.0
