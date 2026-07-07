# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for KG-enhanced warm-start context in cortex_t0."""

from __future__ import annotations

from typing import Any

from hyperloom.orchestrator.knowledge.cortex_t0 import _build_warm_start_context
from hyperloom.inference_optimizer.recipe_kb.kg_client import KGClient

_FACTS_PAGE = """# KG

## Facts
- qwen3moeforcausallm VARIANT_OF qwen2forcausallm (distance: 1)
- bad_patch REVERTED_ON qwen3moeforcausallm (loss: -5%, error: oom)
- fp8_kernel_patch REVERTED_ON qwen2forcausallm (loss: -4.2%, error: perf_regression)
- aiter_backend IMPROVES qwen3moeforcausallm (gain: +30%, confidence: 0.95, hw: mi300x, fw: sglang)
"""


class _FakeMcp:
    """Fake MCP returning a single facts page for all searches."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages

    def call(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "list_pages":
            return [{"slug": s} for s in self.pages]
        if tool == "search":
            return [{"slug": s} for s in self.pages]
        if tool == "get_page":
            return {"content": self.pages.get(args.get("slug"), "")}
        return {}


def _kg() -> KGClient:
    return KGClient(_FakeMcp({"kg": "---\ntype: recipe\n---\n\n" + _FACTS_PAGE}))


def _ctx() -> dict[str, Any]:
    return _build_warm_start_context(
        status="hit",
        tier="exact",
        confidence=0.9,
        canonical_id="inference:qwen3:mi300x:sglang:qwen3:qwen3moeforcausallm:0.5.11:fp8",
        source="local",
        recipe={"prs_tested": []},
        model_architectures=["Qwen3MoeForCausalLM"],
        hardware="mi300x",
        framework="sglang",
        kg_client=_kg(),
    )


def test_kg_hard_block_same_arch() -> None:
    ctx = _ctx()
    hard = [b for b in ctx.get("blocked_patches", []) if b.get("source") == "kg"]
    assert any(b["patch_file"] == "bad_patch" and b["block_type"] == "hard" for b in hard)


def test_kg_advisory_block_related_arch() -> None:
    ctx = _ctx()
    adv = ctx.get("advisory_blocked_patches", [])
    fp8 = [b for b in adv if b["patch_file"] == "fp8_kernel_patch"]
    assert len(fp8) == 1
    # confidence decayed for indirect (related-arch) inference
    assert fp8[0]["confidence"] < 0.95
    assert fp8[0]["block_type"] == "advisory"


def test_kg_recommended_knobs_positive() -> None:
    ctx = _ctx()
    recs = ctx.get("recommended_knobs", [])
    assert any(r["knob"] == "aiter_backend" and r["expected_gain"] > 0 for r in recs)


_KNOB_FACTS_PAGE = """# KG

## Facts
- aiter_backend IMPROVES qwen3moeforcausallm (gain: +30%, confidence: 0.95, hw: mi300x, fw: sglang)
- abc123 KNOB_IMPROVES qwen3moeforcausallm+fp8 (gain: +12%, args: --moe-runner-backend aiter, name: moe-aiter, keep_n: 3, hw: mi300x, fw: sglang)
"""


def _kg_knob() -> KGClient:
    return KGClient(_FakeMcp({"kg": "---\ntype: recipe\n---\n\n" + _KNOB_FACTS_PAGE}))


def _ctx_knob() -> dict[str, Any]:
    return _build_warm_start_context(
        status="hit",
        tier="exact",
        confidence=0.9,
        canonical_id="inference:qwen3:mi300x:sglang:qwen3:qwen3moeforcausallm:0.5.11:fp8",
        source="local",
        recipe={"prs_tested": []},
        model_architectures=["Qwen3MoeForCausalLM"],
        hardware="mi300x",
        framework="sglang",
        precision="fp8",
        kg_client=_kg_knob(),
    )


def test_graph_guided_knobs_disabled_by_default(monkeypatch: Any) -> None:
    monkeypatch.delenv("GBRAIN_KG_GUIDED", raising=False)
    ctx = _ctx_knob()
    assert "graph_guided_knobs" not in ctx


def test_graph_guided_knobs_enabled_by_flag(monkeypatch: Any) -> None:
    monkeypatch.setenv("GBRAIN_KG_GUIDED", "1")
    ctx = _ctx_knob()
    knobs = ctx.get("graph_guided_knobs", [])
    assert any(
        k["knob"] == "abc123" and k["args"] == "--moe-runner-backend aiter" and k["source"] == "kg_knob"
        for k in knobs
    )


def test_filter_warm_patches_advisory_and_expiry() -> None:
    from types import SimpleNamespace

    from hyperloom.orchestrator.loop.coordinator import Coordinator

    patches = [
        {"patch_file": "good.py", "measured_gain_pct": 10},
        {"patch_file": "expired.py", "measured_gain_pct": 8, "expired": True},
        {"patch_file": "risky.py", "measured_gain_pct": 5},
    ]
    advisory = [{"patch_file": "risky.py", "confidence": 0.8}]
    state = SimpleNamespace(gpu_type="mi300x", framework="sglang")
    kept = Coordinator._filter_warm_patches_with_kg(
        SimpleNamespace(), patches, advisory, state
    )
    files = {p["patch_file"] for p in kept}
    assert files == {"good.py"}


def test_filter_warm_patches_keeps_low_confidence_advisory() -> None:
    from types import SimpleNamespace

    from hyperloom.orchestrator.loop.coordinator import Coordinator

    patches = [{"patch_file": "maybe.py", "measured_gain_pct": 5}]
    advisory = [{"patch_file": "maybe.py", "confidence": 0.5}]  # below 0.75 threshold
    kept = Coordinator._filter_warm_patches_with_kg(
        SimpleNamespace(), patches, advisory, SimpleNamespace()
    )
    assert [p["patch_file"] for p in kept] == ["maybe.py"]


def test_specialist_prompt_renders_kg_recommended_knobs() -> None:
    from hyperloom.orchestrator.specialists.domains import get_domain
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
        SpecialistPromptInputs,
        build_specialist_prompts,
    )

    domain = get_domain("serving_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        task_id="t-kg",
        domain=domain,
        max_turns=4,
        gap_canonical_id="gap.kg",
        gap_layer=domain.layer,
        kg_recommended_knobs=[
            {"knob": "aiter_backend", "expected_gain": 30.0, "confidence": 0.95, "source": "kg_graph"},
        ],
    )
    _, user = build_specialist_prompts(inp)
    assert "5d. GRAPH-RECOMMENDED KNOBS" in user
    assert "aiter_backend" in user
    assert "gain=+30.0%" in user
    assert "conf=0.95" in user


def test_specialist_prompt_renders_kg_guided_knobs() -> None:
    from hyperloom.orchestrator.specialists.domains import get_domain
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
        SpecialistPromptInputs,
        build_specialist_prompts,
    )

    domain = get_domain("serving_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        task_id="t-kg-guided",
        domain=domain,
        max_turns=4,
        gap_canonical_id="gap.kg",
        gap_layer=domain.layer,
        kg_guided_knobs=[
            {
                "knob": "abc123",
                "name": "moe-aiter",
                "args": "--moe-runner-backend aiter",
                "envs": {"VLLM_USE_AITER": "1"},
                "expected_gain": 12.0,
                "evidence_count": 3,
                "source": "kg_knob",
            },
        ],
    )
    _, user = build_specialist_prompts(inp)
    assert "5e. GRAPH-GUIDED CONFIG KNOBS" in user
    assert "moe-aiter" in user
    assert "--moe-runner-backend aiter" in user
    assert "VLLM_USE_AITER=1" in user
    assert "gain=+12.0%" in user
    assert "kept=3x" in user


def test_specialist_prompt_kg_section_placeholder_when_empty() -> None:
    from hyperloom.orchestrator.specialists.domains import get_domain
    from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
        SpecialistPromptInputs,
        build_specialist_prompts,
    )

    domain = get_domain("serving_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        task_id="t-kg-empty", domain=domain, max_turns=4, gap_canonical_id="gap.kg", gap_layer=domain.layer
    )
    _, user = build_specialist_prompts(inp)
    assert "5d. GRAPH-RECOMMENDED KNOBS" in user  # section always present


class _RecordingKG:
    """Fake native KG client recording emit_fact_safe calls."""

    def __init__(self, *, native: bool = True, available: bool = True) -> None:
        self._native = native
        self._available = available
        self.calls: list[dict[str, Any]] = []

    def is_available(self) -> bool:
        return self._available

    def emit_fact_safe(self, **kwargs: Any) -> bool:
        self.calls.append(kwargs)
        return True


def _emit_decision(monkeypatch: Any, kg: Any, **kwargs: Any) -> None:
    from types import SimpleNamespace

    import hyperloom.inference_optimizer.recipe_kb.kg_client as kgmod
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    monkeypatch.setattr(kgmod, "get_kg_client", lambda: kg)
    selfish = SimpleNamespace(shared_state=SimpleNamespace(gpu_type="mi300x", framework="sglang"))
    Coordinator._emit_kg_decision(selfish, **kwargs)


def test_emit_kg_decision_keep_emits_improves(monkeypatch: Any) -> None:
    kg = _RecordingKG()
    _emit_decision(
        monkeypatch, kg,
        patch_file="aiter_backend", outcome="KEEP", gain_pct=12.0, error_class="",
        archs=["Qwen2ForCausalLM"],
    )
    assert len(kg.calls) == 1
    call = kg.calls[0]
    assert call["subject"] == "aiter_backend"
    assert call["predicate"] == "IMPROVES"
    assert call["object"] == "Qwen2ForCausalLM"
    assert call["properties"]["gain"] == "+12.0%"
    assert call["properties"]["hw"] == "mi300x"
    assert call["properties"]["fw"] == "sglang"


def test_emit_kg_decision_revert_emits_reverted_on(monkeypatch: Any) -> None:
    kg = _RecordingKG()
    _emit_decision(
        monkeypatch, kg,
        patch_file="fp8_patch", outcome="REVERT", gain_pct=-4.2, error_class="perf",
        archs=["Qwen2ForCausalLM", "LlamaForCausalLM"],
    )
    assert {c["predicate"] for c in kg.calls} == {"REVERTED_ON"}
    assert {c["object"] for c in kg.calls} == {"Qwen2ForCausalLM", "LlamaForCausalLM"}
    assert kg.calls[0]["properties"]["error"] == "perf"


def test_emit_kg_decision_skips_non_native(monkeypatch: Any) -> None:
    # Non-native client would write a discarded fence; write-back must skip it.
    kg = _RecordingKG(native=False)
    _emit_decision(
        monkeypatch, kg,
        patch_file="p", outcome="KEEP", gain_pct=5.0, error_class="", archs=["A"],
    )
    assert kg.calls == []


def test_emit_kg_decision_keep_zero_gain_no_edge(monkeypatch: Any) -> None:
    # A KEEP with non-positive gain is not an IMPROVES claim.
    kg = _RecordingKG()
    _emit_decision(
        monkeypatch, kg,
        patch_file="p", outcome="KEEP", gain_pct=0.0, error_class="", archs=["A"],
    )
    assert kg.calls == []


def test_kg_disabled_when_no_client() -> None:
    # No kg_client and env unset -> get_kg_client() returns None -> no KG keys.
    ctx = _build_warm_start_context(
        status="hit",
        tier="exact",
        confidence=0.9,
        canonical_id="inference:x:mi300x:sglang:llama:llamaforcausallm:0.5.11:fp8",
        source="local",
        recipe={"prs_tested": []},
        model_architectures=["LlamaForCausalLM"],
        hardware="mi300x",
        framework="sglang",
    )
    assert "advisory_blocked_patches" not in ctx
    assert "recommended_knobs" not in ctx
