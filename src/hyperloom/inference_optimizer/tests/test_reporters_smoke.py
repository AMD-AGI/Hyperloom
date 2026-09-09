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
                    "measurement": {"throughput_tok_s_per_gpu": 2205.0, "throughput_unit": "tok/s"},
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
                "throughput_tok_s_per_gpu": 2205.0,
                "accuracy": None,
                "ttft_mean_ms": None,
                "e2el_mean_ms": None,
            },
            "final": {
                "throughput_tok_s_per_gpu": 2447.5,
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
        "roofline",
        "param_search",
        "attribution",
        "optimizations",
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
    bd["outcome"]["final"] = {"throughput_tok_s_per_gpu": None, "gain_pct": None, "action_path": []}
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
    """Shares are taken against what the session moved, so the rest must show.

    Without a row for it, the sources' shares quietly fail to reach 100% and
    the reader is left to work out what the missing slice was.
    """
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
                "baseline": {"throughput_tok_s_per_gpu": "2205"},
                "final": {"throughput_tok_s_per_gpu": "2447", "gain_pct": "10.99"},
            },
        }
    )

    assert "+10.99% validated gain" in r.global_facts.headline
