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


def _baseline_event(**action: Any) -> dict[str, Any]:
    """The ``baseline`` event holding the session's anchoring measurement."""
    return {
        "type": "baseline",
        "ext": {
            "actions": [
                {
                    "task_id": "b-1",
                    "status": "succeeded",
                    "start_time": "2026-05-12T10:00:00Z",
                    "end_time": "2026-05-12T10:04:00Z",
                    "request": {"establishes_quality_ref": True, "failure_streak_before": 0},
                    "measurement": {"output_throughput": 2205.0, "throughput_unit": "tok/s"},
                    **action,
                }
            ]
        },
    }


def _kernel_event(detected: int = 50, selected: int = 10) -> dict[str, Any]:
    """A ``kernel`` event whose analysis nominated ``selected`` of ``detected``."""
    return {
        "type": "kernel",
        "ext": {
            "forge": {
                "discovered_kernels": [
                    {"kernel_id": f"k{i}", "name": f"k{i}", "gpu_pct": float(detected - i), "selected": i < selected}
                    for i in range(detected)
                ]
            }
        },
    }


def _framework_event(outcome: str = "KEEP") -> dict[str, Any]:
    """A ``framework_agent`` event holding one measured configuration variant."""
    return {
        "type": "framework_agent",
        "ext": {
            "attempts": [
                {
                    "attempt_id": "t1:r1:fp",
                    "arm": "config",
                    "round_id": "r1",
                    "variant_name": "vllm_kv_fp8",
                    "outcome": outcome,
                    "benched": True,
                    "adopted": outcome == "KEEP",
                    "measurement": {"after_tput": 2447.5, "gain_pct": 10.99},
                }
            ]
        },
    }


def _fixture_breakdown(**overrides: Any) -> dict[str, Any]:
    # ``outcome.baseline`` is the anchoring measurement as the export projects
    # it; the baseline event on the timeline is the same measurement as its own
    # recorder wrote it. A real export carries both, so the fixture does too.
    base = {
        "metadata": {
            "session": {
                "session_id": "test-sid",
                "claw_session_id": "test-claw",
                "sandbox_user_id": "sandbox-1",
                "elapsed_minutes": 720,
                "tick_count": 12,
                "host": "node-1",
                "code_revision": "deadbeef",
                "session_dir": "/path/sessions/test",
            },
            "task_config": {
                "model_name": "deepseek-ai/DeepSeek-R1",
                "framework_name": "vllm",
                "gpu_type": "MI300X",
                "tp": 8,
                "conc": 64,
                "isl": 1024,
                "osl": 1024,
                "precision": "FP8",
                "max_model_len": 4096,
                "objective": {"kind": "gain_pct", "value": 30.0},
            },
        },
        "outcome": {
            "stop_reason": "time_exhausted",
            "baseline": {
                "output_throughput": 2205.0,
                "accuracy": None,
                "ttft_mean_ms": None,
                "e2el_mean_ms": None,
            },
            "final": {
                "output_throughput": 2447.5,
                "gain_pct": 10.99,
                "extra_server_args": "",
                "action_path": ["explore:vllm_kv_fp8"],
            },
            # A ledger with no measurable per-source contribution: the session
            # validated a total but nothing in it can be credited to a source.
            "validation": {
                "validated_at_stack_len": 1,
                "validated_ts": "2026-05-12T11:54:00Z",
                "stack_changed_after_validation": False,
                "attribution": {"available": True, "by_source": {}},
                "notes": [],
            },
        },
        "timeline": [_baseline_event(), _kernel_event(), _framework_event()],
        "critic": {
            "iterations": [
                {
                    "iteration_id": "critic-iteration:1:abc",
                    "iter": 1,
                    "ts": "2026-05-12T11:20:00Z",
                    "topic": "backends:vllm_kv_fp8",
                    "verdict": "1 approve",
                    "verdict_counts": {"approve": 1},
                    "summary": "the flag pays for itself",
                    "phase": "FRAMEWORK_AGENT",
                    "macro_cycle": 0,
                    "review_path": "critic-workdir/000001/review.json",
                    "framework_reviews": [
                        {
                            "proposal_msg_id": "p1",
                            "arm": "config",
                            "verdict": "approve",
                            "effective_verdict": "approve",
                            "reasoning": "cheap to try, easy to revert",
                        }
                    ],
                },
                # A pass that only spoke: it must still be reported.
                {
                    "iteration_id": "critic-iteration:2:def",
                    "iter": 2,
                    "ts": "2026-05-12T11:40:00Z",
                    "topic": "heartbeat",
                    "verdict": "",
                    "verdict_counts": {},
                    "summary": "ok (critic)",
                    "phase": "SWEEP",
                },
            ]
        },
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
        "enablement",
        "capability_summary",
        "phase_timeline",
        "kernel_lifecycle",
        "roofline",
        "param_search",
        "attribution",
        "optimizations",
        "critic",
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
    lifecycle = next(s for s in r.sections if s.section_id == "kernel_lifecycle")
    assert any(d.kind == "not_attempted" for d in lifecycle.decisions)


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
    # With no validated source_breakdown, a single action_path entry must NOT be stamped "100% via 1 KEEP" (it may be
    # a seeded/warm-replayed entry).
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
    bd["outcome"]["final"]["action_path"] = ["backends:vllm_kv_fp8"]

    r = render_session_report(bd)

    line = r.global_facts.gain_attribution_lines[0]
    assert "unattributed" in line
    assert "KEEP" not in line
    assert "backends:vllm_kv_fp8" in line


def test_every_source_the_ledger_credits_gets_its_own_attribution_line() -> None:
    # A source that contributed nothing measurable is not a line: printing it
    # as 0.00% reads as a source that ran and failed rather than one the ledger
    # could not credit.
    bd = _fixture_breakdown()
    bd["outcome"]["validation"]["validated_total_gain_pct"] = 14.5
    bd["outcome"]["validation"]["attribution"] = {
        "available": True,
        "by_source": {
            "framework_agent": {"keep_count": 1, "total_gain_pct": 10.0},
            "kernel": {"keep_count": 1, "total_gain_pct": 4.5},
            "warm_replay": {"keep_count": 0, "total_gain_pct": 0.0},
        },
    }

    lines = render_session_report(bd).global_facts.gain_attribution_lines

    assert any(line.startswith("framework_agent: 10.00% of total") for line in lines)
    assert any(line.startswith("kernel: 4.50% of total") for line in lines)
    assert not any(line.startswith("warm_replay") for line in lines)


def test_attribution_missing_when_no_gain() -> None:
    bd = _fixture_breakdown()
    bd["outcome"]["final"] = {"output_throughput": None, "gain_pct": None, "action_path": []}
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
    assert "baseline 2205.00 tok/s → final" in r.markdown


def test_llm_exception_does_not_crash_compose() -> None:
    r = render_session_report(_fixture_breakdown(), llm_client=_RaisingLLM())
    assert r.markdown
    assert "<llm_error" in r.llm_raw_response


def test_kernel_lifecycle_funnel_propagates_to_global_facts() -> None:
    r = render_session_report(_fixture_breakdown())
    f = r.global_facts.kernel_pipeline_funnel
    assert f["detected"] == 50 and f["recommended"] == 10
    assert f["attempted"] == 0 and f["adopted"] == 0


def test_the_funnel_counts_a_kernel_every_lane_rejected_as_attempted() -> None:
    bd = _fixture_breakdown()
    bd["timeline"] = [
        _baseline_event(),
        {
            "type": "kernel",
            "ext": {
                "attempts": [
                    {"kernel_id": "k_a", "route": "geak", "outcome": "rejected", "source_kind": "geak_authored_kernel"}
                ]
            },
        },
    ]
    r = render_session_report(bd)

    assert (r.global_facts.kernel_pipeline_funnel["attempted"], r.global_facts.kernel_pipeline_funnel["rejected"]) == (
        1,
        1,
    )
    assert "recommended=1 → attempted=1 → adopted=0 (reverted=0, rejected=1, partial=0)." in r.markdown


def test_variant_funnel_reaches_the_key_facts_and_the_summary() -> None:
    """A breakdown exported before ``outcome.variants`` existed is counted off its timeline."""
    md = render_session_report(_fixture_breakdown()).markdown

    line = "1 bench run over 1 distinct change → kept 1 (100.0%) · reverted 0 · failed 0 → validated +10.99%"
    assert f"- **Variants**: {line}\n" in md
    assert f"- Variants: {line}.\n" in md


def test_variant_funnel_reads_the_exported_block() -> None:
    bd = _fixture_breakdown()
    bd["outcome"]["variants"] = {
        "bench_runs": 12,
        "distinct": 7,
        "kept": 2,
        "reverted": 10,
        "failed": 0,
        "keep_rate_pct": 16.67,
    }
    bd["outcome"]["validation"]["adoption_count"] = 2

    r = render_session_report(bd)

    assert (
        "- **Variants**: 12 bench runs over 7 distinct changes → kept 2 (16.7%) · reverted 10 · failed 0"
        " → final stack 2 changes, validated +10.99%\n"
    ) in r.markdown
    assert json.loads(r.llm_user_prompt)["global_facts"]["variant_funnel"]["bench_runs"] == 12


def test_a_session_that_benched_nothing_says_so() -> None:
    bd = _fixture_breakdown()
    bd["timeline"] = [_baseline_event(), _kernel_event()]

    assert "- **Variants**: none benched\n" in render_session_report(bd).markdown


def test_a_drifted_variants_block_still_renders() -> None:
    bd = _fixture_breakdown()
    bd["outcome"]["variants"] = {"bench_runs": "3", "kept": "x", "keep_rate_pct": "n/a"}

    funnel = render_session_report(bd).global_facts.variant_funnel

    assert (funnel["bench_runs"], funnel["kept"], funnel["keep_rate_pct"]) == (3, 0, None)


@pytest.mark.parametrize(
    "outcome,expected_kind",
    [
        ("KEEP", "kept"),
        ("REVERT", "tried"),
        ("FAILED", "tried"),
    ],
)
def test_capability_decision_kind_follows_what_the_arm_kept(
    outcome: str,
    expected_kind: str,
) -> None:
    """A capability that ran and kept nothing is ``tried``, not absent.

    The three verdicts a variant can end on collapse to two standings, because
    what the report asks of a capability is whether anything it produced
    survived -- not how the one variant that did not survive failed.
    """
    bd = _fixture_breakdown()
    bd["timeline"] = [_baseline_event(), _kernel_event(), _framework_event(outcome=outcome)]

    r = render_session_report(bd)
    cap = next(s for s in r.sections if s.section_id == "capability_summary")

    assert {d.subject: d.kind for d in cap.decisions}.get("explore") == expected_kind


def test_a_capability_that_never_ran_emits_no_decision() -> None:
    """``not_attempted`` is an absence, and crediting it as a verdict would
    put a capability nobody invoked on the same footing as one that lost."""
    bd = _fixture_breakdown()
    bd["timeline"] = [_baseline_event()]

    r = render_session_report(bd)
    cap = next(s for s in r.sections if s.section_id == "capability_summary")

    assert cap.decisions == []
    assert "explore" in r.global_facts.capabilities_not_attempted


def test_gain_that_belongs_to_nobody_gets_its_own_row() -> None:
    """Shares are taken against what the session moved, so the rest must show."""
    bd = _fixture_breakdown()
    bd["outcome"]["validation"].update(
        {
            "validated_total_gain_pct": 10.0,
            "attributed_gain_pct": 9.0,
            "unattributed_gain_pct": 1.0,
            "attribution": {
                "available": True,
                "by_source": {"framework_agent": {"keep_count": 1, "total_gain_pct": 9.0}},
            },
        }
    )

    r = render_session_report(bd)
    sec = next(s for s in r.sections if s.section_id == "attribution")

    assert "unattributed (between adopted steps)" in sec.markdown_block
    # 9 of 10 and 1 of 10: the shares close.
    assert any("90.0" in fact and "framework_agent" in fact for fact in sec.key_facts), sec.key_facts
    # The residue is not a contributor and must not reach the leaderboard.
    assert not any(d.subject.startswith("attribution:unattributed") for d in sec.decisions)


def test_the_ledger_is_the_only_attribution_method_and_carries_its_findings() -> None:
    # The section used to have to name which of several reconstructions it had
    # managed; a split read off the ledger has one provenance, so the label is
    # a constant and the notes are findings rather than an explanation of how
    # the figures were assembled.
    note = "the last whole-stack validation predates the final adoptions"
    bd = _fixture_breakdown()
    bd["outcome"]["validation"].update(
        {
            "validated_total_gain_pct": 10.99,
            "attributed_gain_pct": 10.99,
            "attribution": {
                "available": True,
                "by_source": {"framework_agent": {"keep_count": 1, "total_gain_pct": 10.99}},
            },
            "notes": [note],
        }
    )

    r = render_session_report(bd)

    assert r.global_facts.attribution_method == "stack_ledger"
    assert any(note in flag for flag in r.global_facts.data_quality_flags)
    sec = next(s for s in r.sections if s.section_id == "attribution")
    assert any(note in fact for fact in sec.key_facts)


def test_invocation_section_renders_when_present() -> None:
    """Baseline/final renderers surface an ``### Invocation`` block; secret-shaped envs are filtered out."""
    bd = _fixture_breakdown()
    bd["metadata"]["session"]["image"] = "registry.example/hyperloom:abc123"
    bd["timeline"] = [
        _baseline_event(
            invocation={
                "framework_args": "python -m sglang.launch_server --model /weka/m --tp 8",
                "extra_envs": {"TP": "8", "VLLM_FLASH_ATTN": "1"},
                "config_path": "runs/baseline/h1/baseline_config.with_envs.yaml",
                "server_log_path": "runs/baseline/h1/benchmark_001/server.log",
            }
        ),
        _kernel_event(),
        _framework_event(),
    ]
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
    bd["metadata"]["session"]["image"] = "registry.example/hyperloom:src"
    bd["timeline"] = [
        _baseline_event(
            invocation={
                "framework_args": "python -m sglang.launch_server --tp 4",
                "framework_args_source": "yaml_cmd",
                "extra_envs": {"TP": "4"},
                "config_path": "runs/baseline/h1/baseline_config.with_envs.yaml",
                "server_log_path": "runs/baseline/h1/benchmark_001/server.log",
            }
        ),
        _kernel_event(),
        _framework_event(),
    ]
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
    def _boom(_breakdown):
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
    def _boom(_breakdown):
        raise ValueError("bad row")

    r = render_session_report(_fixture_breakdown())

    assert any("[baseline] section could not be rendered" in f for f in r.global_facts.data_quality_flags)
    assert "section could not be rendered: ValueError: bad row" in r.markdown


@pytest.mark.parametrize(
    "section",
    ["metadata", "baseline", "final", "attribution", "kernel_lifecycle", "capability_summary"],
)
def test_a_section_that_drifted_to_a_string_still_yields_a_report(section: str) -> None:
    """Producers are not schema-checked, so a drifted shape must cost one section."""
    r = render_session_report({"metadata": {"session": {"session_id": "s"}}, section: "drifted"})

    assert r.markdown.startswith("# Hyperloom Session Report")


def test_every_section_drifting_at_once_still_yields_a_report() -> None:
    bd = {sid: "drifted" for sid, _ in REGISTRY}
    bd["metadata"] = "drifted"

    r = render_session_report(bd)

    assert r.markdown.startswith("# Hyperloom Session Report")
    assert r.global_facts.kernel_pipeline_funnel["detected"] == 0


def test_a_drifted_section_yields_neutral_facts() -> None:
    r = render_session_report(
        {
            "metadata": {"session": {"session_id": "s"}},
            "outcome": {"stop_reason": "target_reached"},
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
            "metadata": {"session": {"session_id": "s"}},
            "outcome": {
                "baseline": {"output_throughput": "2205"},
                "final": {"output_throughput": "2447", "gain_pct": "10.99"},
            },
        }
    )

    assert "+10.99% validated gain" in r.global_facts.headline


#: One AgentX round's axes, as the recorder stamps them onto ``perf``.
_AGENTX_PERF = {
    "e2e_norm_intvty_p50": 41.8,
    "e2e_norm_intvty_p90": 22.4,
    "output_tput_per_gpu": 275.6,
    "total_throughput": 26500.0,
    "input_throughput": 24296.0,
    "ttft_p50_ms": 110.0,
    "ttft_p90_ms": 240.0,
    "tpot_p50_ms": 18.0,
    "tpot_p90_ms": 34.0,
    "duration_seconds": 3600.0,
    "request_error_rate": 0.0,
}


def _agentx_breakdown() -> dict[str, Any]:
    """A breakdown whose baseline and validation rounds both carry graded axes."""
    b = _fixture_breakdown()
    b["outcome"]["baseline"]["perf"] = dict(_AGENTX_PERF)
    b["outcome"]["baseline"]["submission_valid"] = True
    b["outcome"]["validation"]["perf"] = dict(_AGENTX_PERF)
    b["timeline"][0]["ext"]["actions"][0]["measurement"]["perf"] = dict(_AGENTX_PERF)
    b["timeline"][0]["ext"]["actions"][0]["measurement"]["submission_valid"] = True
    return b


def test_the_report_shows_the_axes_the_session_was_graded_on() -> None:
    """The ``perf`` block reached the artifact from the first V6 recorder and no renderer read it.

    An AgentX session is ranked on median interactivity with the tail and per-GPU output as guards, so a report
    that shows only ``output_throughput`` states a figure the verdict was not taken on.
    """
    md = render_session_report(_agentx_breakdown()).markdown

    for key in ("e2e_norm_intvty_p50", "e2e_norm_intvty_p90", "output_tput_per_gpu"):
        assert key in md, f"{key} is graded but never rendered"
    for key in ("ttft_p50_ms", "ttft_p90_ms", "tpot_p50_ms", "tpot_p90_ms"):
        assert key in md, f"{key} was asked for in the detail view"
    # The inputs a pair is refused on. Without them a REVERT on the objective and one on a drifted window read the
    # same in the report.
    for key in ("duration_seconds", "request_error_rate"):
        assert key in md, f"{key} decides comparability and must be auditable"
    assert "submission_valid" in md


def test_a_refused_round_says_why_upstream_refused_it() -> None:
    """``submission_valid: false`` alone leaves a reader with a rejected round and no reason for it."""
    b = _agentx_breakdown()
    measurement = b["timeline"][0]["ext"]["actions"][0]["measurement"]
    measurement["submission_valid"] = False
    measurement["submission_invalid_reasons"] = ["duration=120s(canonical 3600s)"]

    md = render_session_report(b).markdown

    assert "submission_invalid_reasons" in md
    assert "duration=120s(canonical 3600s)" in md


def test_an_accepted_round_carries_no_refusal_reasons() -> None:
    """The reasons list is empty on every accepted round; rendering the key would be noise on all of them."""
    md = render_session_report(_agentx_breakdown()).markdown

    assert "submission_valid" in md
    assert "submission_invalid_reasons" not in md


def test_a_round_with_no_graded_axes_renders_no_axis_block() -> None:
    """A synthetic session measures none of them, and eleven nulls would claim it was graded on them."""
    md = render_session_report(_fixture_breakdown()).markdown

    # Anchored on a positive assertion first: both checks below would also pass on an empty report, and
    # reporters/base.py substitutes a warning section for a renderer that raises.
    assert "- **output_throughput**:" in md
    assert "AgentX graded axes" not in md
    assert "e2e_norm_intvty_p50" not in md


def test_only_the_interactivity_pair_marks_a_round_agentx_graded() -> None:
    """Every other published axis is filled for an ordinary measurement too.

    ``_merge_raw_result`` fills duration from raw ``duration``, the latency percentiles from ``median_ttft_ms`` and
    friends, and the error rate whenever the raw result has it, none of them gated on the workload;
    ``_promote_baseline`` stamps per-GPU output unconditionally. Only the interactivity pair has a single producer,
    ``agentx/mapping.py``, so only it proves the agentic mapper ran.
    """
    from hyperloom.inference_optimizer.breakdown.reporters._renderers._agentx_metrics import has_graded_axes

    for key, value in (
        ("total_throughput", 26500.0),
        ("input_throughput", 24296.0),
        ("output_tput_per_gpu", 275.6),
        ("duration_seconds", 3600.0),
        ("request_error_rate", 0.0),
        ("ttft_p50_ms", 110.0),
        ("ttft_p90_ms", 240.0),
        ("tpot_p50_ms", 18.0),
        ("tpot_p90_ms", 34.0),
    ):
        assert not has_graded_axes({key: value}), f"{key} is measured off AgentX too and cannot gate the block"
    assert has_graded_axes({"e2e_norm_intvty_p50": 41.8})
    assert has_graded_axes({"e2e_norm_intvty_p90": 22.4})


def test_a_synthetic_round_that_reports_a_duration_heads_no_agentx_section() -> None:
    """The reported trigger: an ordinary SGLang round whose report carries a duration, a median TTFT and a zero
    error rate, and which never measured interactivity, must not be headed as AgentX-graded."""
    b = _fixture_breakdown()
    synthetic_perf = {
        "e2e_norm_intvty_p50": None,
        "e2e_norm_intvty_p90": None,
        "duration_seconds": 3600.0,
        "ttft_p50_ms": 110.0,
        "tpot_p50_ms": 18.0,
        "request_error_rate": 0.0,
        "total_throughput": 26500.0,
    }
    b["outcome"]["baseline"]["perf"] = dict(synthetic_perf)
    b["outcome"]["validation"]["perf"] = dict(synthetic_perf)
    b["timeline"][0]["ext"]["actions"][0]["measurement"]["perf"] = dict(synthetic_perf)

    md = render_session_report(b).markdown

    assert "- **output_throughput**:" in md
    assert "AgentX graded axes" not in md


def test_a_non_finite_axis_reads_the_same_in_the_facts_as_in_the_table() -> None:
    """``_md_cell`` renders NaN as an em dash; a key fact saying "nan" would be a second answer."""
    from hyperloom.inference_optimizer.breakdown.reporters._renderers._agentx_metrics import graded_axes_facts

    assert graded_axes_facts({"e2e_norm_intvty_p50": float("nan")}, label="X") == []
    assert graded_axes_facts({"e2e_norm_intvty_p50": float("inf")}, label="X") == []
    assert graded_axes_facts({"e2e_norm_intvty_p50": 41.8}, label="X") == ["X e2e_norm_intvty_p50: 41.8."]


def _section(result: Any, section_id: str) -> Any:
    """The rendered section ``section_id`` of a ``render_session_report`` result."""
    return next(s for s in result.sections if s.section_id == section_id)


def _stack_event(**settled: Any) -> dict[str, Any]:
    """A ``stack`` event whose settled validation row produced the session's gain."""
    return {"type": "stack", "ext": {"validations": {"settled": settled}}}


def _graded_breakdown() -> dict[str, Any]:
    """An AgentX-graded session: p50 interactivity moved while the tok/s figures barely did."""
    b = _fixture_breakdown()
    b["outcome"]["final"]["graded_on"] = "e2e_norm_intvty_p50"
    b["outcome"]["final"]["gain_pct"] = 26.5457
    b["timeline"].append(
        _stack_event(
            baseline_tput=63.02819,
            validated_tput=79.75949,
            validated_gain_pct=26.5457,
            graded_objective="e2e_norm_intvty_p50",
        )
    )
    return b


def test_headline_on_an_interactivity_session_quotes_the_graded_axis() -> None:
    """Throughput in tok/s beside a gain measured on interactivity reads as one derived from the other."""
    result = render_session_report(_graded_breakdown())

    expected = "e2e_norm_intvty_p50 63.03 → 79.76 = +26.55% validated gain"
    assert result.global_facts.headline == expected
    assert f"- **Headline**: {expected}" in result.markdown.splitlines()


def test_headline_uses_the_settled_row_when_the_baseline_was_re_anchored() -> None:
    """``outcome.baseline`` holds the latest anchor; the gain was measured against the ledger's baseline."""
    b = _graded_breakdown()
    b["outcome"]["baseline"]["perf"] = {"e2e_norm_intvty_p50": 150.51}
    b["outcome"]["final"]["gain_pct"] = 44.52
    b["timeline"][-1] = _stack_event(
        baseline_tput=103.479,
        validated_tput=149.544,
        validated_gain_pct=44.5158,
        graded_objective="e2e_norm_intvty_p50",
    )

    headline = render_session_report(b).global_facts.headline

    assert headline == "e2e_norm_intvty_p50 103.48 → 149.54 = +44.52% validated gain"


def test_headline_names_only_the_gain_and_axis_without_a_settled_row() -> None:
    b = _graded_breakdown()
    b["timeline"] = [e for e in b["timeline"] if e["type"] != "stack"]

    headline = render_session_report(b).global_facts.headline

    assert headline == "+26.55% validated gain on e2e_norm_intvty_p50"


def test_headline_names_only_the_gain_and_axis_when_the_settled_row_is_on_another_axis() -> None:
    b = _graded_breakdown()
    b["timeline"][-1]["ext"]["validations"]["settled"]["graded_objective"] = "e2e_norm_intvty_p90"

    assert render_session_report(b).global_facts.headline == "+26.55% validated gain on e2e_norm_intvty_p50"


def test_headline_names_only_the_gain_and_axis_when_the_pair_does_not_reproduce_it() -> None:
    b = _graded_breakdown()
    b["timeline"][-1]["ext"]["validations"]["settled"]["validated_tput"] = 70.0

    assert render_session_report(b).global_facts.headline == "+26.55% validated gain on e2e_norm_intvty_p50"


def test_headline_reads_the_last_stack_event() -> None:
    b = _graded_breakdown()
    first = _stack_event(
        baseline_tput=1.0, validated_tput=9.0, validated_gain_pct=800.0, graded_objective="e2e_norm_intvty_p50"
    )
    b["timeline"].insert(0, first)

    assert (
        render_session_report(b).global_facts.headline == "e2e_norm_intvty_p50 63.03 → 79.76 = +26.55% validated gain"
    )


def test_headline_keeps_the_throughput_sentence_when_the_graded_axis_is_throughput() -> None:
    b = _graded_breakdown()
    b["outcome"]["final"]["graded_on"] = "output_throughput"

    headline = render_session_report(b).global_facts.headline

    assert headline == "baseline 2205.00 tok/s → final 2447.50 tok/s = +26.55% validated gain"


def test_final_gain_fact_names_the_axis_it_was_measured_on() -> None:
    facts = _section(render_session_report(_graded_breakdown()), "final").key_facts

    assert "Validated cumulative gain: +26.55% (e2e_norm_intvty_p50)." in facts


def test_final_gain_fact_is_unlabelled_when_graded_on_throughput() -> None:
    facts = _section(render_session_report(_fixture_breakdown()), "final").key_facts

    assert "Validated cumulative gain: +10.99%." in facts


def test_elapsed_reports_the_total_across_legs() -> None:
    b = _fixture_breakdown()
    b["metadata"]["session"]["total_elapsed_minutes"] = 2741.93
    result = render_session_report(b)

    assert result.global_facts.elapsed_minutes == 2741.93
    stop_line = next(line for line in result.markdown.splitlines() if line.startswith("- **Stop reason**"))
    assert " · elapsed(total) `2741.9 min` · " in stop_line
    assert "elapsed(total)=2742min" in result.markdown


def test_elapsed_falls_back_to_the_leg_value_with_a_plain_label() -> None:
    result = render_session_report(_fixture_breakdown())

    stop_line = next(line for line in result.markdown.splitlines() if line.startswith("- **Stop reason**"))
    assert " · elapsed `720.0 min` · " in stop_line
    assert "elapsed=720min" in result.markdown


def test_session_section_lists_the_leg_and_the_total_separately() -> None:
    b = _fixture_breakdown()
    b["metadata"]["session"]["total_elapsed_minutes"] = 2741.93
    section = _section(render_session_report(b), "session")

    assert "Wall-clock elapsed (this leg): 720.0 minutes." in section.key_facts
    assert "Total elapsed across all legs: 2741.9 minutes." in section.key_facts
    assert "- **elapsed_minutes**: 720" in section.markdown_block.splitlines()
    assert "- **total_elapsed_minutes**: 2741.93" in section.markdown_block.splitlines()


def test_summary_names_the_graded_axis_and_labels_the_stop_target() -> None:
    result = render_session_report(_graded_breakdown())

    stop_line = next(line for line in result.markdown.splitlines() if line.startswith("- **Stop reason**"))
    assert stop_line == (
        "- **Stop reason**: `time_exhausted` · elapsed `720.0 min` · "
        "graded on `e2e_norm_intvty_p50` · stop target `gain_pct>=30`"
    )
    assert "graded_on=e2e_norm_intvty_p50, stop target gain_pct>=30)." in result.markdown


def test_graded_axis_is_unset_without_a_recorded_verdict_axis() -> None:
    """The configured axis says what was asked for, not what the verdict was taken on."""
    b = _fixture_breakdown()
    b["metadata"]["grading"] = {"objective": "e2e_norm_intvty_p90"}
    result = render_session_report(b)

    assert result.global_facts.graded_on == ""
    stop_line = next(line for line in result.markdown.splitlines() if line.startswith("- **Stop reason**"))
    assert stop_line == (
        "- **Stop reason**: `time_exhausted` · elapsed `720.0 min` · graded on `unset` · stop target `gain_pct>=30`"
    )


def test_workload_labels_the_objective_as_the_stop_target() -> None:
    section = _section(render_session_report(_fixture_breakdown()), "workload")

    assert "Stop target: gain_pct>=30." in section.key_facts
    assert "- **stop_target**: gain_pct>=30" in section.markdown_block.splitlines()


def _scriptable_breakdown() -> dict[str, Any]:
    """A server-less session whose rounds recorded a zero TTFT because they measure none."""
    b = _fixture_breakdown()
    b["metadata"]["task_config"]["framework_name"] = "custom"
    action = b["timeline"][0]["ext"]["actions"][0]
    action["measurement"] = {"output_throughput": 9.13, "throughput_unit": "unit/s", "ttft_mean_ms": 0}
    b["outcome"]["baseline"]["output_throughput"] = 9.13
    b["outcome"]["final"].update({"output_throughput": 10.25, "ttft_mean_ms": 0, "gain_pct": 12.35})
    return b


def test_a_zero_ttft_is_unmeasured_and_not_printed() -> None:
    result = render_session_report(_scriptable_breakdown())
    baseline = _section(result, "baseline")
    final = _section(result, "final")

    assert baseline.markdown_block.splitlines()[:2] == [
        "- **output_throughput**: 9.13",
        "- **throughput_unit**: unit/s",
    ]
    assert not any(line.startswith("- **ttft_mean_ms**") for line in baseline.markdown_block.splitlines())
    assert not any(line.startswith("- **ttft_mean_ms**") for line in final.markdown_block.splitlines())
    assert not any(fact.startswith("Baseline TTFT") for fact in baseline.key_facts)
    assert baseline.warnings == []


def test_a_measured_ttft_still_prints() -> None:
    b = _scriptable_breakdown()
    b["timeline"][0]["ext"]["actions"][0]["measurement"]["ttft_mean_ms"] = 12.5
    baseline = _section(render_session_report(b), "baseline")

    assert "- **ttft_mean_ms**: 12.50" in baseline.markdown_block.splitlines()
    assert "Baseline TTFT mean: 12.5 ms." in baseline.key_facts


def test_final_throughput_unit_matches_the_throughput_value() -> None:
    final = _section(render_session_report(_scriptable_breakdown()), "final")

    lines = final.markdown_block.splitlines()
    assert lines[:2] == ["- **final_output_throughput**: 10.25", "- **throughput_unit**: unit/s"]
    assert "Final: 97.56 ms." in final.key_facts


def _unvalidated_breakdown() -> dict[str, Any]:
    """A session that adopted nothing and never ran a whole-stack validation."""
    b = _fixture_breakdown()
    b["outcome"]["final"].pop("gain_pct")
    b["outcome"]["validation"].update(
        {
            "validated_at_stack_len": None,
            "validated_ts": None,
            "adoption_count": 0,
            "stack_changed_after_validation": True,
        }
    )
    return b


_NO_VALIDATION = "No validated gain is recorded while baseline/final throughput are set — no validation run completed."


def test_no_validation_run_does_not_also_claim_a_stale_validation() -> None:
    final = _section(render_session_report(_unvalidated_breakdown()), "final")

    assert final.warnings == [_NO_VALIDATION]


def test_a_validation_that_predates_the_final_stack_is_still_reported() -> None:
    b = _fixture_breakdown()
    b["outcome"]["validation"]["stack_changed_after_validation"] = True
    final = _section(render_session_report(b), "final")

    assert final.warnings == [
        "The last whole-stack validation predates the final adoptions, so the validated cumulative gain "
        "describes a shorter stack than the one that shipped."
    ]


def _kernel_attempt_event(speedup: float) -> dict[str, Any]:
    """A ``kernel`` event with one GEAK attempt that the end-to-end gate kept."""
    attempt = {
        "kernel_id": "k_rms",
        "name": "k_rms",
        "route": "geak",
        "speedup": speedup,
        "outcome": "adopted",
        "e2e": {"decision": "KEEP"},
    }
    return {"type": "kernel", "ext": {"attempts": [attempt]}}


def _kernel_row(speedup: float) -> str:
    b = _fixture_breakdown()
    b["timeline"].append(_kernel_attempt_event(speedup))
    md = _section(render_session_report(b), "kernel_lifecycle").markdown_block
    return next(line for line in md.splitlines() if line.startswith("| k_rms |"))


def test_an_unmeasured_kernel_speedup_renders_as_missing() -> None:
    assert "| — (1 att) [KEEP] |" in _kernel_row(0.0)


def test_a_measured_kernel_speedup_still_renders() -> None:
    assert "| 1.50x (1 att) [KEEP] |" in _kernel_row(1.5)


def _roofline_event(*error_classes: str) -> dict[str, Any]:
    """A ``roofline`` event whose runs each failed with the matching error class."""
    actions = [
        {"failure": {"error_class": cls, "message": "boom", "stage": "x"}, "outcome": {}} for cls in error_classes
    ]
    return {"type": "roofline", "failure": None, "ext": {"actions": actions}}


def test_roofline_runs_that_all_failed_report_their_error_classes() -> None:
    b = _fixture_breakdown()
    b["timeline"] += [
        _roofline_event("trace_analyze_failed"),
        _roofline_event("trace_analyze_failed", "profile_failed"),
    ]
    result = render_session_report(b)

    roofline = _section(result, "roofline")
    assert roofline.skipped
    assert roofline.key_facts == ["3 run(s), all failed: trace_analyze_failed×2, profile_failed×1"]
    assert (
        "  - [roofline] skipped: 3 run(s), all failed: trace_analyze_failed×2, profile_failed×1"
        in result.markdown.splitlines()
    )


def test_a_session_without_roofline_events_keeps_the_no_data_wording() -> None:
    result = render_session_report(_fixture_breakdown())

    assert _section(result, "roofline").skipped
    assert "  - [roofline] skipped: no data recorded this session." in result.markdown.splitlines()


def test_attribution_prompt_names_the_methods_the_report_really_emits() -> None:
    from hyperloom.inference_optimizer.breakdown.reporters.llm_prompt import SYSTEM_PROMPT

    start = SYSTEM_PROMPT.index("- Reflect the ``attribution_method``")
    bullet = SYSTEM_PROMPT[start:].strip()

    assert bullet == (
        '- Reflect the ``attribution_method`` honestly: ``"stack_ledger"`` is the only method backed by the\n'
        '  stack ledger; any other value (``"unattributed"``, ``"unattributed (stack listed for reference,\n'
        '  not verified as KEEP)"`` or ``"missing"``) MUST be described in plain language as not validated.'
    )


def _stop_target_facts(objective: dict[str, Any]) -> tuple[str, str]:
    """The workload fact and the summary's stop target for ``objective``."""
    b = _fixture_breakdown()
    b["metadata"]["task_config"]["objective"] = objective
    result = render_session_report(b)
    fact = next(f for f in _section(result, "workload").key_facts if f.startswith("Stop target"))
    return fact, result.global_facts.stop_target


def test_several_stop_targets_are_all_rendered() -> None:
    gain = {"kind": "gain_pct", "value": 500.0}
    roofline = {"kind": "roofline_pct", "value": 80.0}

    fact, target = _stop_target_facts({**gain, "objectives": [gain, roofline]})

    assert target == "gain_pct>=500, roofline_pct>=80"
    assert fact == "Stop target: gain_pct>=500, roofline_pct>=80."


def test_a_baseline_directory_target_names_the_directory() -> None:
    fact, target = _stop_target_facts({"kind": "baseline", "value": "/runs/base"})

    assert target == "baseline=/runs/base"
    assert fact == "Stop target: baseline=/runs/base."


def test_a_time_only_target_has_no_value() -> None:
    fact, target = _stop_target_facts({"kind": "time_only", "value": None})

    assert target == "time_only"
    assert fact == "Stop target: time_only."


def test_a_numeric_throughput_target_keeps_the_threshold() -> None:
    assert _stop_target_facts({"kind": "tput", "value": 1500.5})[1] == "tput>=1500.5"
