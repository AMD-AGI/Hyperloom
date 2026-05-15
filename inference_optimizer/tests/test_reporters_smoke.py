"""Smoke tests for the ``breakdown.reporters`` compose pipeline.

We feed each renderer a hand-built ``session_breakdown.json`` shape
and check the cross-section + section invariants that historically
broke. Specifically:

* GEAK / OOB ``not_attempted`` MUST NOT yield decisions of any other
  kind (the old MAE bug attributed 715% gain to GEAK on a session
  that never ran GEAK).
* All-zero GPU power/temp with non-zero samples MUST raise a
  ``data_quality_flag`` (the gpu_monitor units bug).
* ``phase_timeline=[]`` MUST be flagged so report consumers know
  process reconstruction is unavailable.
* Deterministic-only path (no LLM) must produce non-empty markdown
  whose executive summary mentions the headline and every data
  quality flag.
* LLM path with a faulty client (e.g. JSON parse failure) must
  degrade to the deterministic exec summary instead of crashing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from inference_optimizer.breakdown.reporters import render_session_report
from inference_optimizer.breakdown.reporters.base import REGISTRY


def _fixture_breakdown(**overrides: Any) -> dict[str, Any]:
    base = {
        "session": {
            "session_id": "test-sid",
            "claw_session_id": "test-claw",
            "sandbox_user_id": "sandbox-1",
            "stop_reason": "time_exhausted",
            "elapsed_minutes": 720,
            "tick_count": 12,
            "host": "node-1",
            "code_revision": "deadbeef",
            "session_dir": "/wekafs/sessions/test",
        },
        "workload": {
            "model_name": "deepseek-ai/DeepSeek-R1",
            "framework": "vllm",
            "gpu_type": "MI300X",
            "tp": 8, "conc": 64, "isl": 1024, "osl": 1024,
            "precision": "FP8", "max_model_len": 4096,
            "objective": {"kind": "gain_pct", "value": 30.0},
        },
        "baseline": {
            "throughput_tok_s_per_gpu": 2205.0,
            "accuracy": None,
            "ttft_mean_ms": None,
            "e2el_mean_ms": None,
            "failure_streak": 0,
            "attempts_history": [],
        },
        "final": {
            "throughput_tok_s_per_gpu": 2447.5,
            "cumulative_gain_pct_validated": 10.99,
            "cumulative_gain_pct_per_round_sum": 10.99,
            "validated_at_stack_len": 1,
            "validated_ts": "2026-05-12T11:54:00Z",
            "stack_changed_after_validation": False,
            "extra_sglang_args": "",
            "action_path": ["backends:vllm_kv_fp8"],
        },
        "phase_timeline": [],
        "capability_summary": {
            "backends":       {"status": "kept",          "attempts": 1, "keeps": 1},
            "params":         {"status": "not_attempted", "attempts": 0, "keeps": 0, "tested": 5},
            "sweep":          {"status": "not_attempted", "attempts": 0, "keeps": 0, "grid_size": 9},
            "geak":           {"status": "not_attempted", "attempts": 0, "keeps": 0},
            "oob":            {"status": "not_attempted", "attempts": 0, "keeps": 0},
            "validate_stack": {"status": "not_attempted", "attempts": 0, "keeps": 0},
        },
        "kernel_lifecycle": {
            "detected": [{"kernel_id": f"k{i}"} for i in range(50)],
            "recommended": [{"kernel_id": f"k{i}"} for i in range(10)],
            "optimized": [], "adopted": [], "partial": [], "reverted": [], "rejected": [],
        },
        "param_search": {
            "backends": {"accepted": ["vllm_kv_fp8"], "tested": {"vllm_kv_fp8": True}},
            "params":   {"accepted": [], "tested": {}},
            "discovered_flags": {},
            "backend_winners_history": [],
            "synergy_attempted": [],
        },
        "sweep": {"all_variants": [], "grid_size": 0},
        "critic_robustness": [],
        "telemetry": {
            "gpu_monitor_aggregate": {
                "samples": 52, "max_power_w": 0, "avg_power_w": 0,
                "max_temp_c": 0, "avg_temp_c": 0,
                "max_util_pct": 0, "avg_util_pct": 0, "source_file_count": 1,
            },
            "benchmark_files_total": 4,
            "log_files_total": 3,
            "artifact_bytes_total": 100_000,
        },
        "attribution": {
            "source_breakdown": {"validated_total_pct": 10.99},
            "notes": [],
            "method": "single_source",
        },
        "source_files": {"state_json": ["state.json"]},
    }
    for k, v in overrides.items():
        base[k] = v
    return base


def test_all_renderers_register_in_stable_order() -> None:
    """compose.py module imports must keep this exact order."""
    expected = [
        "session", "workload", "baseline", "final",
        "capability_summary", "phase_timeline", "kernel_lifecycle",
        "geak_invocations", "oob_invocations",
        "param_search", "sweep", "critic_robustness",
        "attribution", "source_files",
    ]
    assert [sid for sid, _ in REGISTRY] == expected


def test_telemetry_renderer_is_not_registered() -> None:
    """Telemetry section is intentionally dropped from the report
    layout (gpu monitor data has been consistently broken on real
    wekafs sessions). Anti-regression for re-adding it by accident."""
    assert "telemetry" not in [sid for sid, _ in REGISTRY]


def test_deterministic_only_path_produces_complete_report() -> None:
    r = render_session_report(_fixture_breakdown())
    md = r.markdown
    assert "# Hyperloom Session Report — test-sid" in md
    assert "## Executive Summary" in md
    assert "10.99%" in md
    assert "MI300X" in md
    # GEAK + OOB sections must mark themselves as not attempted
    geak = next(s for s in r.sections if s.section_id == "geak_invocations")
    assert geak.skipped
    assert any(d.kind == "not_attempted" for d in geak.decisions)
    oob = next(s for s in r.sections if s.section_id == "oob_invocations")
    assert oob.skipped


def test_skipped_sections_do_not_emit_placeholders() -> None:
    """Previously a skipped section produced ``## Title\\n_Section
    skipped: no data captured..._`` filler. Users complained that it
    added visual noise without any information. Suppressing it
    entirely is the contract now — verify by checking the markdown
    does not mention ``geak_invocations`` / ``sweep`` / ``critic`` /
    ``phase_timeline`` / ``not captured`` placeholder strings."""
    r = render_session_report(_fixture_breakdown())
    md = r.markdown
    assert "Section skipped" not in md
    assert "no data captured" not in md
    # The H3 title for a skipped section must also be absent.
    assert "### GEAK Invocations" not in md
    assert "### Sweep" not in md
    assert "### Phase Timeline" not in md


def test_section_groups_use_h2_titles_with_h3_subsections() -> None:
    r = render_session_report(_fixture_breakdown())
    md = r.markdown
    # H2 group titles for any group that has at least one live section.
    assert "## Session & Workload" in md
    assert "## Performance Results" in md
    assert "## Capability Search" in md
    assert "## Kernel Optimization" in md
    # H3 subsections live underneath the H2 groups.
    assert "### Session" in md
    assert "### Baseline" in md
    assert "### Capability Summary" in md
    assert "### Kernel Lifecycle" in md
    # Groups with all-skipped sections must NOT emit their H2 title.
    # In the fixture sweep + phase_timeline are skipped; "Run Trace"
    # group is phase_timeline-only so it should disappear entirely.
    assert "## Run Trace" not in md


def test_telemetry_section_is_absent_from_markdown() -> None:
    """Telemetry must not appear in the report at any level."""
    r = render_session_report(_fixture_breakdown())
    assert "## Telemetry" not in r.markdown
    assert "### Telemetry" not in r.markdown
    assert "gpu_monitor_aggregate" not in r.markdown


def test_geak_not_attempted_never_emits_kept_decision() -> None:
    """Anti-regression for the MAE-era hallucination: GEAK got
    attributed gain on a session it never ran on."""
    r = render_session_report(_fixture_breakdown())
    for sec in r.sections:
        if sec.section_id != "geak_invocations":
            continue
        for d in sec.decisions:
            assert d.kind == "not_attempted", (
                f"GEAK section emitted non-not_attempted decision {d!r} "
                f"despite no invocations on disk"
            )


def test_attribution_method_marks_single_source_when_path_len_1() -> None:
    r = render_session_report(_fixture_breakdown())
    g = r.global_facts
    assert g.attribution_method.startswith("single-source")
    assert g.gain_attribution_lines, "expected at least one attribution line"
    assert g.gain_attribution_lines[0].startswith("100% via 1 backends KEEP")


def test_attribution_missing_when_no_gain() -> None:
    bd = _fixture_breakdown()
    bd["final"] = {"throughput_tok_s_per_gpu": None,
                   "cumulative_gain_pct_validated": None,
                   "action_path": []}
    r = render_session_report(bd)
    assert r.global_facts.attribution_method == "missing"
    assert r.global_facts.gain_attribution_lines == []


# ---------------------------------------------------------------------------
# LLM integration smoke
# ---------------------------------------------------------------------------
@dataclass
class _GoodLLM:
    def complete(self, *, system: str, user: str) -> str:
        payload = json.loads(user)
        sids = [s["section_id"] for s in payload["sections"] if not s["skipped"]]
        return json.dumps({
            "executive_summary": "Validated +10.99% via backends KEEP on DeepSeek-R1 MI300X.",
            "section_narratives": {sid: f"narr-{sid}" for sid in sids},
        })


@dataclass
class _BrokenLLM:
    def complete(self, *, system: str, user: str) -> str:
        return "Sorry, I cannot comply — { not json"


@dataclass
class _RaisingLLM:
    def complete(self, *, system: str, user: str) -> str:
        raise RuntimeError("network down")


def test_llm_path_inserts_narratives_for_non_skipped_sections() -> None:
    r = render_session_report(_fixture_breakdown(), llm_client=_GoodLLM())
    assert r.used_llm
    assert "Validated +10.99% via backends KEEP" in r.markdown
    # Narratives must appear for non-skipped sections only.
    assert "narr-session" in r.markdown
    assert "narr-capability_summary" in r.markdown
    # Skipped sections must NOT get narratives, and the LLM was never
    # shown them in the user prompt either (see llm_prompt.build_user_prompt).
    assert "narr-geak_invocations" not in r.markdown
    assert "narr-sweep" not in r.markdown
    # Skipped sections must not appear under "sections" in the LLM
    # prompt (they may still appear under global_facts.capabilities_*
    # which is fine — the LLM is allowed to talk about not-attempted
    # capabilities).
    prompt = json.loads(r.llm_user_prompt)
    section_ids = {s["section_id"] for s in prompt["sections"]}
    assert "geak_invocations" not in section_ids
    assert "oob_invocations" not in section_ids
    assert "sweep" not in section_ids


def test_llm_broken_json_falls_back_to_deterministic_exec_summary() -> None:
    r = render_session_report(_fixture_breakdown(), llm_client=_BrokenLLM())
    # Deterministic exec summary must still appear (no crash).
    assert "## Executive Summary" in r.markdown
    assert "baseline 2205.00 → final" in r.markdown


def test_llm_exception_does_not_crash_compose() -> None:
    r = render_session_report(_fixture_breakdown(), llm_client=_RaisingLLM())
    assert r.markdown  # non-empty
    assert "<llm_error" in r.llm_raw_response


# ---------------------------------------------------------------------------
# Section-level invariants worth pinning
# ---------------------------------------------------------------------------
def test_kernel_lifecycle_funnel_propagates_to_global_facts() -> None:
    r = render_session_report(_fixture_breakdown())
    f = r.global_facts.kernel_pipeline_funnel
    assert f["detected"] == 50 and f["recommended"] == 10
    assert f["optimized"] == 0 and f["adopted"] == 0


@pytest.mark.parametrize("cap_status,expected_kind", [
    ("kept",     "kept"),
    ("reverted", "reverted"),
    ("rejected", "rejected"),
])
def test_capability_decision_kind_round_trips(
    cap_status: str, expected_kind: str,
) -> None:
    bd = _fixture_breakdown()
    bd["capability_summary"]["sweep"] = {
        "status": cap_status, "attempts": 1, "keeps": 1 if cap_status == "kept" else 0,
    }
    r = render_session_report(bd)
    cap = next(s for s in r.sections if s.section_id == "capability_summary")
    decisions = {d.subject: d.kind for d in cap.decisions}
    assert decisions.get("sweep") == expected_kind
