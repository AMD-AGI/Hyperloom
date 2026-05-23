"""Roofline-v2 N22: analysis.md keyword -> variant advisory tests.

Bridges the gap N20-A left open: the LLM has the catalogue + a soft
prompt hint to include all relevant variants, but empirically still
under-includes (Qwen3-30B-A3B N20c session: analysis said 'torch.compile',
LLM picked everything cuda-graph related but skipped torch_compile_on).

N22 adds a non-blocking advisory at the proposal layer:
* coordinator._record_keyword_implied_advice() inspects every backends/
  params propose with a variants subset
* compares against analysis.md keyword map (curated in
  _analysis_keyword_map.py)
* writes an advisory into shared_state.last_proposal_advice when
  the LLM omitted any keyword-implied variant
* the next-tick orchestration prompt renders the advisory so the LLM
  self-corrects on the follow-up propose

Properties this test suite locks:
* advisory triggers when an implied variant is missing
* advisory does NOT trigger when all implied variants are included
* advisory does NOT block the proposal (no PolicyDenied)
* unknown framework variants are silently narrowed out (vLLM run
  doesn't get advised to add SGLang-only variants)
* empty analysis.md / empty variants list -> no advisory (graceful)
* FIFO cap (5 most recent) keeps shared_state from growing unbounded
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator._analysis_keyword_map import (
    ANALYSIS_KEYWORD_TO_VARIANTS,
    extract_required_variants_from_analysis,
    format_missing_variants_advice,
)
from inference_optimizer.orchestrator.action_executors.params import (
    DEFAULT_PARAMS_GRID,
)
from inference_optimizer.orchestrator.action_executors.backends import (
    DEFAULT_BACKENDS_GRID,
)
from inference_optimizer.orchestrator.backends import (
    MockBackend,
    MockCriticBackend,
    MockKernelBackend,
    MockRobustnessBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import PolicyDenied
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import target_baseline_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel":        MockKernelBackend(),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }


def _write_baseline_marker(sd: Path) -> Path:
    p = target_baseline_json(sd)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    return p


def _seed_post_roofline(coord: Coordinator, *, analysis_md: str) -> None:
    """Bring the SharedState to a "baseline + roofline_1 done" state with
    a custom analysis.md text so we can test keyword extraction against
    realistic content."""
    _write_baseline_marker(coord.session_dir)
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
    coord.shared_state.last_trace_analyze = {
        "trace_input": "/tmp/profile.trace.json.gz",
        "analysis_md_text": analysis_md,
        "roofline_snapshot_id": 1,
    }
    coord.shared_state.discovered_flags_at_last_snapshot = {}


# Mirror of the real Qwen3-30B-A3B N20c analysis.md Executive Summary +
# P1 (used here so the test directly reproduces the empirical miss this
# whole feature was designed to catch).
_QWEN3_HOST_BOUND_ANALYSIS = """
# Qwen3 (vocab=151936) - MI300X Analysis

## Executive Summary

The dominant signal is severe GPU underutilization: across a 1751.23 ms
window, computation accounts for only ~0.09% of wall time, with no
recorded GPU kernels in the cpu_idle stream and only ~0.01% memcpy /
0% collective communication.

## System-Level Optimizations

### P1: Investigate Severe GPU Underutilization

Insight: The GPU is idle 99.9% of the captured window.
Action: Verify the trace captures an active steady-state window.
If representative, remove host-side stalls (blocking sync, host
preprocessing, Python overhead), raise effective batch/concurrency,
and capture repetitive submission sequences with GPU graphs or
torch.compile to amortize per-launch CPU cost.
"""


# ===========================================================================
# Pure-function tests on the keyword map module
# ===========================================================================
class TestKeywordExtraction:
    def test_empty_text_returns_empty(self):
        required, matches = extract_required_variants_from_analysis(
            "", available_variants=[v.name for v in DEFAULT_PARAMS_GRID],
        )
        assert required == []
        assert matches == []

    def test_torch_compile_keyword_triggers_torch_compile_on(self):
        """The empirical Qwen3 miss the entire feature is built to
        catch — analysis says 'torch.compile', LLM should include
        torch_compile_on."""
        required, matches = extract_required_variants_from_analysis(
            _QWEN3_HOST_BOUND_ANALYSIS,
            available_variants=[v.name for v in DEFAULT_PARAMS_GRID],
        )
        assert "torch_compile_on" in required
        # Validate the match record references the trigger keyword
        triggers = {key for key, _ in matches}
        assert "torch.compile" in triggers

    def test_cuda_graph_keyword_pulls_full_family(self):
        """A keyword that maps to a 4-variant family should include
        all 4 (after narrowing to available)."""
        required, _ = extract_required_variants_from_analysis(
            "Use cuda graphs to amortize per-launch CPU cost.",
            available_variants=[v.name for v in DEFAULT_PARAMS_GRID],
        )
        cuda_graph_variants = {v for v in required if v.startswith("cuda_graph_max_bs")}
        assert cuda_graph_variants >= {
            "cuda_graph_max_bs_8", "cuda_graph_max_bs_16",
            "cuda_graph_max_bs_32", "cuda_graph_max_bs_64",
        }

    def test_case_insensitive_match(self):
        required, _ = extract_required_variants_from_analysis(
            "TORCH.COMPILE is essential here.",
            available_variants=[v.name for v in DEFAULT_PARAMS_GRID],
        )
        assert "torch_compile_on" in required

    def test_substring_match_inside_word(self):
        """`torch.compile()` and `torch.compile)` should still match
        the bare `torch.compile` key."""
        required, _ = extract_required_variants_from_analysis(
            "We recommend torch.compile() with mode='reduce-overhead'.",
            available_variants=[v.name for v in DEFAULT_PARAMS_GRID],
        )
        assert "torch_compile_on" in required

    def test_unavailable_variants_filtered_out(self):
        """Keyword may imply 4 cuda_graph_* variants but if only 2
        are in available, only those 2 are returned."""
        required, _ = extract_required_variants_from_analysis(
            "cuda graphs",
            available_variants=["cuda_graph_max_bs_32", "cuda_graph_max_bs_64"],
        )
        assert set(required) == {"cuda_graph_max_bs_32", "cuda_graph_max_bs_64"}

    def test_no_matching_keyword_returns_empty(self):
        required, matches = extract_required_variants_from_analysis(
            "## Executive Summary\nNo issues detected; baseline at peak.",
            available_variants=[v.name for v in DEFAULT_PARAMS_GRID],
        )
        assert required == []
        assert matches == []

    def test_allreduce_keyword_pulls_custom_ar(self):
        required, _ = extract_required_variants_from_analysis(
            "AllReduce dominates 30% of GPU time on this rank.",
            available_variants=[v.name for v in DEFAULT_BACKENDS_GRID],
        )
        assert "custom_ar" in required

    def test_moe_keyword_pulls_moe_aiter(self):
        required, _ = extract_required_variants_from_analysis(
            "MoE expert routing accounts for 22% of decode time.",
            available_variants=[v.name for v in DEFAULT_BACKENDS_GRID],
        )
        assert "moe_aiter" in required
        assert "enable_fused_moe" in required


# ===========================================================================
# format_missing_variants_advice
# ===========================================================================
class TestFormatAdvice:
    def test_returns_none_when_all_present(self):
        msg = format_missing_variants_advice(
            proposed_variants=["torch_compile_on", "cuda_graph_max_bs_64"],
            required_variants=["torch_compile_on", "cuda_graph_max_bs_64"],
            matches=[("torch.compile", ("torch_compile_on",)),
                     ("cuda graph", ("cuda_graph_max_bs_64",))],
            action_name="params",
        )
        assert msg is None

    def test_returns_none_when_no_required(self):
        msg = format_missing_variants_advice(
            proposed_variants=["x"],
            required_variants=[],
            matches=[],
            action_name="params",
        )
        assert msg is None

    def test_lists_missing_variants_and_triggers(self):
        msg = format_missing_variants_advice(
            proposed_variants=["cuda_graph_max_bs_64"],
            required_variants=["torch_compile_on", "cuda_graph_max_bs_64"],
            matches=[("torch.compile", ("torch_compile_on",)),
                     ("cuda graph", ("cuda_graph_max_bs_64",))],
            action_name="params",
        )
        assert msg is not None
        assert "[N22 advisory]" in msg
        assert "torch.compile" in msg
        assert "torch_compile_on" in msg
        # Already-included variants must NOT be reported missing
        assert "cuda_graph_max_bs_64" not in msg.split("missing variant(s):")[1]
        # Action name surfaced for audit
        assert "action='params'" in msg


# ===========================================================================
# End-to-end coordinator integration
# ===========================================================================
class TestCoordinatorIntegration:
    def test_params_propose_with_missing_torch_compile_writes_advice(
        self, session_dir, monkeypatch,
    ):
        """The motivating empirical case: analysis says torch.compile,
        LLM proposes params with cuda_graph_* but NO torch_compile_on
        -> coordinator records advice into shared_state."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_roofline(coord, analysis_md=_QWEN3_HOST_BOUND_ANALYSIS)

        denied = coord._sequence_denial_for_action(
            "params",
            proposed_params={"variants": [
                "cuda_graph_max_bs_64", "cuda_graph_max_bs_32",
                "decode_steps_16", "decode_steps_32",
                "max_running_requests_256",
            ]},
        )
        # Advisory must NOT block
        assert denied is None or "torch_compile_on" not in str(denied)
        # Advisory must be recorded
        advisories = coord.shared_state.last_proposal_advice
        assert len(advisories) == 1
        text = advisories[0]
        assert "torch_compile_on" in text
        assert "torch.compile" in text
        assert "action='params'" in text

    def test_complete_variant_set_writes_no_advice(self, session_dir, monkeypatch):
        """When the LLM includes torch_compile_on and the cuda_graph
        family + decode_steps + max_running_requests, no advisory
        triggers."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_roofline(coord, analysis_md=_QWEN3_HOST_BOUND_ANALYSIS)

        coord._sequence_denial_for_action(
            "params",
            proposed_params={"variants": [
                "torch_compile_on",
                "cuda_graph_max_bs_8", "cuda_graph_max_bs_16",
                "cuda_graph_max_bs_32", "cuda_graph_max_bs_64",
                "decode_steps_8", "decode_steps_16", "decode_steps_32",
                "max_running_requests_128", "max_running_requests_256",
            ]},
        )
        assert coord.shared_state.last_proposal_advice == []

    def test_advisory_does_not_block_proposal(self, session_dir, monkeypatch):
        """The N22 contract: missing-variant advice is non-blocking.
        The propose must return None (allowed) even when advisory fires."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", raising=False)
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_roofline(coord, analysis_md=_QWEN3_HOST_BOUND_ANALYSIS)

        denied = coord._sequence_denial_for_action(
            "params",
            proposed_params={"variants": ["cuda_graph_max_bs_64"]},
        )
        # Must NOT return PolicyDenied just because torch_compile_on is
        # missing. (Other pre-existing gates could fire — we only care
        # that the N22 check itself doesn't insert a denial.)
        if denied is not None:
            assert "[N22 advisory]" not in str(denied)
            assert "torch_compile_on" not in str(denied)
        # And the advisory IS recorded
        assert coord.shared_state.last_proposal_advice

    def test_no_variants_field_skips_advisory(self, session_dir, monkeypatch):
        """LLM submitted no variants subset (defaulted to full grid) ->
        no advisory needed (the default grid already includes all the
        keyword-implied variants by construction)."""
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_roofline(coord, analysis_md=_QWEN3_HOST_BOUND_ANALYSIS)

        coord._sequence_denial_for_action(
            "params",
            proposed_params={},
        )
        assert coord.shared_state.last_proposal_advice == []

    def test_no_analysis_md_skips_advisory(self, session_dir, monkeypatch):
        """Before the first roofline (no analysis.md cached yet),
        N22 should be a no-op even if variants are passed."""
        coord = Coordinator(session_dir, backends=_silent_backends())
        # NO _seed_post_roofline call -> no last_trace_analyze
        coord.shared_state.baseline_tput = 100.0
        _write_baseline_marker(coord.session_dir)

        coord._sequence_denial_for_action(
            "params",
            proposed_params={"variants": ["cuda_graph_max_bs_64"]},
        )
        assert coord.shared_state.last_proposal_advice == []

    def test_backends_propose_with_missing_custom_ar(self, session_dir, monkeypatch):
        """Same logic applies to backends action — analysis says
        AllReduce hot, LLM proposes only attn_aiter -> advisory
        flags missing custom_ar."""
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_roofline(coord, analysis_md=(
            "## Executive Summary\nAllReduce is 25% of decode time."
        ))

        coord._sequence_denial_for_action(
            "backends",
            proposed_params={"variants": ["attn_aiter"]},
        )
        advisories = coord.shared_state.last_proposal_advice
        assert len(advisories) == 1
        assert "custom_ar" in advisories[0]
        assert "AllReduce" in advisories[0] or "allreduce" in advisories[0].lower()

    def test_other_actions_not_affected(self, session_dir, monkeypatch):
        """N22 only inspects backends/params proposes. Other actions
        (sweep, baseline, etc.) must not trigger it."""
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_roofline(coord, analysis_md=_QWEN3_HOST_BOUND_ANALYSIS)

        for action in ("baseline", "sweep", "report"):
            coord._sequence_denial_for_action(
                action,
                proposed_params={"variants": ["cuda_graph_max_bs_64"]},
            )
        # Even with the variants field set, none of these should trigger
        # N22 (it only listens for backends/params)
        assert coord.shared_state.last_proposal_advice == []


class TestAdvisoryFifoCap:
    def test_fifo_caps_at_five(self, session_dir, monkeypatch):
        """Long-running session shouldn't grow last_proposal_advice
        unbounded; keep the most recent 5."""
        coord = Coordinator(session_dir, backends=_silent_backends())
        _seed_post_roofline(coord, analysis_md=_QWEN3_HOST_BOUND_ANALYSIS)

        # Fire 7 advisories (each missing torch_compile_on)
        for i in range(7):
            coord._sequence_denial_for_action(
                "params",
                proposed_params={"variants": [f"cuda_graph_max_bs_{8 * (i + 1)}"]
                                  if (8 * (i + 1)) in (8, 16, 32, 64) else
                                  ["cuda_graph_max_bs_64"]},
            )
        advisories = coord.shared_state.last_proposal_advice
        assert len(advisories) == 5  # capped


class TestKeywordMapVariantsExist:
    """Defensive: every variant the map references should exist in
    one of the registered grids. Catches map / grid drift."""

    def test_all_mapped_variants_exist_in_registered_grids(self):
        all_known_names = (
            {v.name for v in DEFAULT_PARAMS_GRID}
            | {v.name for v in DEFAULT_BACKENDS_GRID}
        )
        unknown = set()
        for keyword, variants in ANALYSIS_KEYWORD_TO_VARIANTS.items():
            for v in variants:
                if v not in all_known_names:
                    unknown.add(v)
        assert unknown == set(), (
            f"_analysis_keyword_map.py references variant(s) not in any "
            f"registered grid: {sorted(unknown)} — either typo the map "
            f"entry or add the variant to DEFAULT_*_GRID."
        )
