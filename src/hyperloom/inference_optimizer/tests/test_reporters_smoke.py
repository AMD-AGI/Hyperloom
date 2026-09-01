# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests for the ``breakdown.reporters`` compose pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.reporters import render_session_report
from hyperloom.inference_optimizer.breakdown.reporters.base import REGISTRY


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
            "session_dir": "/path/sessions/test",
        },
        "workload": {
            "model_name": "deepseek-ai/DeepSeek-R1",
            "framework": "vllm",
            "gpu_type": "MI300X",
            "tp": 8,
            "conc": 64,
            "isl": 1024,
            "osl": 1024,
            "precision": "FP8",
            "max_model_len": 4096,
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
            "validated_at_stack_len": 1,
            "validated_ts": "2026-05-12T11:54:00Z",
            "stack_changed_after_validation": False,
            "extra_server_args": "",
            "action_path": ["explore:vllm_kv_fp8"],
        },
        "phase_timeline": [],
        "capability_summary": {
            "explore": {"status": "kept", "attempts": 1, "keeps": 1},
            "backends": {"status": "not_attempted", "attempts": 0, "keeps": 0},
            "params": {"status": "not_attempted", "attempts": 0, "keeps": 0},
            "sweep": {"status": "not_attempted", "attempts": 0, "keeps": 0, "grid_size": 9},
            "geak": {"status": "not_attempted", "attempts": 0, "keeps": 0},
            "validate_stack": {"status": "not_attempted", "attempts": 0, "keeps": 0},
        },
        "kernel_lifecycle": {
            "detected": [{"kernel_id": f"k{i}"} for i in range(50)],
            "recommended": [{"kernel_id": f"k{i}"} for i in range(10)],
            "optimized": [],
            "adopted": [],
            "partial": [],
            "reverted": [],
            "rejected": [],
        },
        "param_search": {
            "explore": {"accepted": ["vllm_kv_fp8"], "tested": {"vllm_kv_fp8": True}},
            "backends": {"accepted": ["vllm_kv_fp8"], "tested": {"vllm_kv_fp8": True}},
            "params": {"accepted": [], "tested": {}},
            "discovered_flags": {},
            "synergy_attempted": [],
        },
        "sweep": {"all_variants": [], "grid_size": 0},
        "critic_robustness": [],
        "telemetry": {
            "gpu_monitor_aggregate": {
                "samples": 52,
                "max_power_w": 0,
                "avg_power_w": 0,
                "max_temp_c": 0,
                "avg_temp_c": 0,
                "max_util_pct": 0,
                "avg_util_pct": 0,
                "source_file_count": 1,
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
        "session",
        "workload",
        "baseline",
        "final",
        "capability_summary",
        "phase_timeline",
        "kernel_lifecycle",
        "kernel_profiling",
        "kernel_decision_path",
        "roofline",
        "geak_invocations",
        "forge_invocations",
        "param_search",
        "decision_journal",
        "sweep",
        "critic_robustness",
        "attribution",
        "optimizations",
        "source_files",
        "data_provenance",
    ]
    assert [sid for sid, _ in REGISTRY] == expected


def test_telemetry_renderer_is_not_registered() -> None:
    """Telemetry section is intentionally dropped from the report layout."""
    assert "telemetry" not in [sid for sid, _ in REGISTRY]


def test_deterministic_only_path_produces_complete_report() -> None:
    r = render_session_report(_fixture_breakdown())
    md = r.markdown
    assert "# Hyperloom Session Report — test-sid" in md
    assert "## Executive Summary" in md
    assert "10.99%" in md
    assert "MI300X" in md
    geak = next(s for s in r.sections if s.section_id == "geak_invocations")
    assert geak.skipped
    assert any(d.kind == "not_attempted" for d in geak.decisions)


def test_skipped_sections_do_not_emit_placeholders() -> None:
    """Skipped sections must emit no placeholder filler or H3 titles."""
    r = render_session_report(_fixture_breakdown())
    md = r.markdown
    assert "Section skipped" not in md
    assert "no data captured" not in md
    assert "### GEAK Invocations" not in md
    assert "### Sweep" not in md
    assert "### Phase Timeline" not in md


def test_section_groups_use_h2_titles_with_h3_subsections() -> None:
    r = render_session_report(_fixture_breakdown())
    md = r.markdown
    assert "## Session & Workload" in md
    assert "## Performance Results" in md
    assert "## Capability Search" in md
    assert "## Kernel Optimization" in md
    assert "### Session" in md
    assert "### Baseline" in md
    assert "### Capability Summary" in md
    assert "### Kernel Lifecycle" in md
    assert "## Run Trace" not in md


def test_telemetry_section_is_absent_from_markdown() -> None:
    """Telemetry must not appear in the report at any level."""
    r = render_session_report(_fixture_breakdown())
    assert "## Telemetry" not in r.markdown
    assert "### Telemetry" not in r.markdown
    assert "gpu_monitor_aggregate" not in r.markdown


def test_geak_not_attempted_never_emits_kept_decision() -> None:
    """GEAK must not be attributed gain on a session it never ran on."""
    r = render_session_report(_fixture_breakdown())
    for sec in r.sections:
        if sec.section_id != "geak_invocations":
            continue
        for d in sec.decisions:
            assert d.kind == "not_attempted", (
                f"GEAK section emitted non-not_attempted decision {d!r} despite no invocations on disk"
            )


def test_attribution_unattributed_when_no_validated_split_path_len_1() -> None:
    # With no validated source_breakdown, a single action_path entry must NOT
    # be stamped "100% via 1 KEEP" (it may be a seeded/warm-replayed entry).
    r = render_session_report(_fixture_breakdown())
    g = r.global_facts
    assert g.attribution_method.startswith("unattributed")
    assert g.gain_attribution_lines, "expected at least one attribution line"
    line = g.gain_attribution_lines[0]
    assert "unattributed" in line
    assert "KEEP" not in line
    assert "explore" in line


def test_legacy_backends_action_path_reported_as_unattributed() -> None:
    bd = _fixture_breakdown()
    bd["final"]["action_path"] = ["backends:vllm_kv_fp8"]

    r = render_session_report(bd)

    line = r.global_facts.gain_attribution_lines[0]
    assert "unattributed" in line
    assert "KEEP" not in line
    assert "backends:vllm_kv_fp8" in line


def test_legacy_source_buckets_survive_alongside_explore() -> None:
    bd = _fixture_breakdown()
    bd["attribution"] = {
        "method": "reconstructed",
        "source_breakdown": {
            "validated_total_pct": 14.5,
            "explore_pct_of_total": 10.0,
            "backends_pct_of_total": 4.5,
            "params_pct_of_total": 0.0,
            "geak_pct_of_total": 0.0,
            "sweep_pct_of_total": 0.0,
        },
        "notes": [],
    }

    r = render_session_report(bd)

    assert any(line.startswith("explore: 10.00% of total") for line in r.global_facts.gain_attribution_lines)
    assert any(line.startswith("backends: 4.50% of total") for line in r.global_facts.gain_attribution_lines)


def test_attribution_missing_when_no_gain() -> None:
    bd = _fixture_breakdown()
    bd["final"] = {"throughput_tok_s_per_gpu": None, "cumulative_gain_pct_validated": None, "action_path": []}
    r = render_session_report(bd)
    assert r.global_facts.attribution_method == "missing"
    assert r.global_facts.gain_attribution_lines == []


@dataclass
class _GoodLLM:
    def complete(self, *, system: str, user: str) -> str:
        payload = json.loads(user)
        sids = [s["section_id"] for s in payload["sections"] if not s["skipped"]]
        return json.dumps(
            {
                "executive_summary": "Validated +10.99% via explore KEEP on DeepSeek-R1 MI300X.",
                "section_narratives": {sid: f"narr-{sid}" for sid in sids},
            }
        )


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
    assert "Validated +10.99% via explore KEEP" in r.markdown
    assert "narr-session" in r.markdown
    assert "narr-capability_summary" in r.markdown
    assert "narr-geak_invocations" not in r.markdown
    assert "narr-sweep" not in r.markdown
    prompt = json.loads(r.llm_user_prompt)
    section_ids = {s["section_id"] for s in prompt["sections"]}
    assert "geak_invocations" not in section_ids
    assert "sweep" not in section_ids


def test_llm_broken_json_falls_back_to_deterministic_exec_summary() -> None:
    r = render_session_report(_fixture_breakdown(), llm_client=_BrokenLLM())
    assert "## Executive Summary" in r.markdown
    assert "baseline 2205.00 tok/s/GPU → final" in r.markdown


def test_llm_exception_does_not_crash_compose() -> None:
    r = render_session_report(_fixture_breakdown(), llm_client=_RaisingLLM())
    assert r.markdown
    assert "<llm_error" in r.llm_raw_response


def test_kernel_lifecycle_funnel_propagates_to_global_facts() -> None:
    r = render_session_report(_fixture_breakdown())
    f = r.global_facts.kernel_pipeline_funnel
    assert f["detected"] == 50 and f["recommended"] == 10
    assert f["optimized"] == 0 and f["adopted"] == 0


@pytest.mark.parametrize(
    "cap_status,expected_kind",
    [
        ("kept", "kept"),
        ("reverted", "reverted"),
        ("rejected", "rejected"),
    ],
)
def test_capability_decision_kind_round_trips(
    cap_status: str,
    expected_kind: str,
) -> None:
    bd = _fixture_breakdown()
    bd["capability_summary"]["sweep"] = {
        "status": cap_status,
        "attempts": 1,
        "keeps": 1 if cap_status == "kept" else 0,
    }
    r = render_session_report(bd)
    cap = next(s for s in r.sections if s.section_id == "capability_summary")
    decisions = {d.subject: d.kind for d in cap.decisions}
    assert decisions.get("sweep") == expected_kind


def test_gain_that_belongs_to_nobody_gets_its_own_row() -> None:
    """Shares are taken against what the session moved, so the rest must show.

    Without a row for it, the sources' shares quietly fail to reach 100% and
    the reader is left to work out what the missing slice was.
    """
    bd = _fixture_breakdown()
    bd.pop("attribution", None)
    bd["optimizations"] = {
        "entries": [{"id": "opt-1", "source": "explore", "gain_pct": 9.0, "validated": True}],
        "summary_by_source": {"explore": {"keeps": 1, "total_gain_pct": 9.0}},
        "validation": {
            "method": "recorded_adoptions",
            "validated_total_gain_pct": 10.0,
            "attributed_total_gain_pct": 9.0,
            "unattributed_gain_pct": 1.0,
        },
    }

    r = render_session_report(bd)
    sec = next(s for s in r.sections if s.section_id == "attribution")

    assert "unattributed (between adopted steps)" in sec.markdown_block
    # 9 of 10 and 1 of 10: the shares close.
    assert any("90.0" in fact and "explore" in fact for fact in sec.key_facts), sec.key_facts
    # The residue is not a contributor and must not reach the leaderboard.
    assert not any(d.subject.startswith("attribution:unattributed") for d in sec.decisions)


def test_v5_attribution_method_and_notes_render_from_validation() -> None:
    bd = _fixture_breakdown()
    bd.pop("attribution", None)
    bd["optimizations"] = {
        "entries": [
            {
                "id": "opt-1",
                "source": "explore",
                "gain_pct": 10.99,
                "validated": True,
            }
        ],
        "summary_by_source": {
            "explore": {"keeps": 1, "total_gain_pct": 10.99},
        },
        "validation": {
            "method": "reconstructed",
            "validated_total_gain_pct": 10.99,
            "notes": ["gain ledger reconstructed from throughput"],
        },
    }

    r = render_session_report(bd)

    assert r.global_facts.attribution_method == "reconstructed"
    assert any("gain ledger reconstructed from throughput" in flag for flag in r.global_facts.data_quality_flags)
    sec = next(s for s in r.sections if s.section_id == "attribution")
    assert any("reconstructed" in fact for fact in sec.key_facts)
    assert any("gain ledger reconstructed from throughput" in fact for fact in sec.key_facts)


def test_invocation_section_renders_when_present() -> None:
    """Baseline/final renderers surface an ``### Invocation`` block; secret-shaped envs are filtered out."""
    bd = _fixture_breakdown()
    bd["session"]["image"] = "registry.example/hyperloom:abc123"
    bd["baseline"]["invocation"] = {
        "framework_args": "python -m sglang.launch_server --model /weka/m --tp 8",
        "extra_envs": {"TP": "8", "VLLM_FLASH_ATTN": "1"},
        "config_path": "runs/baseline/h1/baseline_config.with_envs.yaml",
        "server_log_path": "runs/baseline/h1/benchmark_001/server.log",
    }
    r = render_session_report(bd)
    base = next(s for s in r.sections if s.section_id == "baseline")
    md = base.markdown_block
    assert "### Invocation" in md
    assert "sglang.launch_server" in md
    assert "registry.example/hyperloom:abc123" in md
    assert "TP=8" in md
    assert "VLLM_FLASH_ATTN=1" in md
    assert "OPENAI_API_KEY" not in md
    prompt = json.loads(r.llm_user_prompt)
    user_text = json.dumps(prompt)
    assert "sglang.launch_server" not in user_text, "framework_args leaked into LLM prompt"


def test_invocation_renders_framework_args_source() -> None:
    """When ``invocation.framework_args_source`` is set, the renderer surfaces the lineage label under the command line."""
    bd = _fixture_breakdown()
    bd["session"]["image"] = "registry.example/hyperloom:src"
    bd["baseline"]["invocation"] = {
        "framework_args": "python -m sglang.launch_server --tp 4",
        "framework_args_source": "yaml_cmd",
        "extra_envs": {"TP": "4"},
        "config_path": "runs/baseline/h1/baseline_config.with_envs.yaml",
        "server_log_path": "runs/baseline/h1/benchmark_001/server.log",
    }
    r = render_session_report(bd)
    base = next(s for s in r.sections if s.section_id == "baseline")
    md = base.markdown_block
    assert "### Invocation" in md
    assert "yaml_cmd" in md, md
    assert "**source**" in md, md


# ---- one bad section must not cost the report ------------------------------
@pytest.fixture
def restore_registry():
    """Undo renderer registrations made inside a test."""
    saved = list(REGISTRY)
    yield
    REGISTRY[:] = saved


def test_a_raising_renderer_does_not_lose_the_other_sections(restore_registry) -> None:
    from hyperloom.inference_optimizer.breakdown.reporters.base import register_renderer

    @register_renderer("session")
    def _boom(_breakdown):  # noqa: ANN001, ANN202
        raise TypeError("drifted shape")

    r = render_session_report(_fixture_breakdown())

    assert r.markdown
    assert "## Performance Results" in r.markdown
    broken = next(s for s in r.sections if s.section_id == "session")
    assert broken.warnings == ["section could not be rendered: TypeError: drifted shape"]
    assert len(r.sections) == len(REGISTRY)


def test_a_renderer_failure_is_reported_not_swallowed(restore_registry) -> None:
    from hyperloom.inference_optimizer.breakdown.reporters.base import register_renderer

    @register_renderer("baseline")
    def _boom(_breakdown):  # noqa: ANN001, ANN202
        raise ValueError("bad row")

    r = render_session_report(_fixture_breakdown())

    assert any("[baseline] section could not be rendered" in f for f in r.global_facts.data_quality_flags)
    assert "section could not be rendered: ValueError: bad row" in r.markdown


@pytest.mark.parametrize(
    "section",
    ["session", "workload", "baseline", "final", "attribution", "kernel_lifecycle", "capability_summary"],
)
def test_a_section_that_drifted_to_a_string_still_yields_a_report(section: str) -> None:
    """Producers are not schema-checked, so a drifted shape must cost one section."""
    r = render_session_report({"session": {"session_id": "s"}, section: "drifted"})

    assert r.markdown.startswith("# Hyperloom Session Report")


def test_every_section_drifting_at_once_still_yields_a_report() -> None:
    bd = {sid: "drifted" for sid, _ in REGISTRY}
    bd["session"] = "drifted"

    r = render_session_report(bd)

    assert r.markdown.startswith("# Hyperloom Session Report")
    assert r.global_facts.kernel_pipeline_funnel["detected"] == 0


def test_a_drifted_section_yields_neutral_facts() -> None:
    r = render_session_report(
        {
            "session": {"session_id": "s", "stop_reason": "target_reached"},
            "kernel_lifecycle": ["not", "a", "dict"],
        }
    )

    facts = r.global_facts
    assert facts.kernel_pipeline_funnel["detected"] == 0
    # Facts that do not depend on the drifted section keep their values.
    assert facts.stop_reason == "target_reached"


def test_numeric_metrics_recorded_as_strings_still_produce_a_headline() -> None:
    r = render_session_report(
        {
            "session": {"session_id": "s"},
            "baseline": {"throughput_tok_s_per_gpu": "2205"},
            "final": {"throughput_tok_s_per_gpu": "2447", "cumulative_gain_pct_validated": "10.99"},
        }
    )

    assert "+10.99% validated gain" in r.global_facts.headline
