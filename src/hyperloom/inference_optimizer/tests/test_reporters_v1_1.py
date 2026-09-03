# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression guards for retired V1.1 report sections and live renderers."""

from __future__ import annotations

import importlib.util

from hyperloom.inference_optimizer.breakdown.reporters import render_session_report
from hyperloom.inference_optimizer.breakdown.reporters.base import REGISTRY
from hyperloom.inference_optimizer.breakdown.reporters._renderers.invocations import render_forge, render_geak
from hyperloom.inference_optimizer.breakdown.reporters._renderers.phase_timeline import render as render_phase_timeline


_RETIRED = {
    "data_provenance",
    "decision_journal",
    "kernel_decision_path",
    "kernel_profiling",
}


def _base_breakdown(**overrides):
    base = {"session": {"session_id": "report-test", "session_dir": "/tmp/session"}}
    base.update(overrides)
    return base


def test_retired_renderer_modules_are_removed() -> None:
    package = "hyperloom.inference_optimizer.breakdown.reporters._renderers"
    assert all(importlib.util.find_spec(f"{package}.{name}") is None for name in _RETIRED)


def test_retired_renderer_ids_are_not_registered() -> None:
    registered = {section_id for section_id, _render in REGISTRY}
    assert registered.isdisjoint(_RETIRED)


def test_legacy_decision_journal_payload_is_ignored() -> None:
    md = render_session_report(_base_breakdown(decision_journal=[{"round_id": "dead-round"}])).markdown
    assert "dead-round" not in md
    assert "Decision Journal" not in md


def test_legacy_kernel_profiling_payload_is_ignored() -> None:
    md = render_session_report(
        _base_breakdown(kernel_profiling=[{"run_id": "dead-profile", "outputs": {"tool": "dead-tool"}}])
    ).markdown
    assert "dead-profile" not in md
    assert "dead-tool" not in md


def test_legacy_artifact_paths_are_never_opened(tmp_path) -> None:
    outside = tmp_path / "legacy-profile.log"
    outside.write_text("secret-tail-line\n", encoding="utf-8")
    md = render_session_report(
        _base_breakdown(
            kernel_profiling=[{"artifacts": {"tracelens_log": str(outside)}}],
            data_provenance=[{"section": "dead-provenance"}],
        )
    ).markdown
    assert "secret-tail-line" not in md
    assert "dead-provenance" not in md


def test_invocation_renderer_normalizes_and_caps_attempt_rows() -> None:
    attempts = [{"ts": f"t{i}", "kernel_id": f"k{i}", "decision": "REVERT"} for i in range(26)]
    attempts.extend(
        [
            "legacy-kernel-id",
            {
                "ts": "done",
                "kernel_name": "named-kernel",
                "decision": "KEEP",
                "micro_speedup": 1.25,
                "workspace_path": "/tmp/ws",
                "error": "x" * 100,
            },
            {"kernel_id": "failed-kernel", "decision": "FAILED"},
            {"kernel_id": "error-kernel", "decision": "ERROR"},
        ]
    )

    sec = render_geak({"geak_invocations": attempts})

    assert not sec.skipped
    assert any("30 invocation(s), 1 KEEP, 2 FAILED" in fact for fact in sec.key_facts)
    assert sec.decisions[0].kind == "kept"
    assert "_Showing last 25 of 30 attempts._" in sec.markdown_block
    assert "legacy-kernel-id" in sec.markdown_block
    assert "named-kernel" in sec.markdown_block
    assert "/tmp/ws" in sec.markdown_block
    assert "x" * 80 in sec.markdown_block
    assert "x" * 81 not in sec.markdown_block

    forge_sec = render_forge({"forge_invocations": attempts})

    assert not forge_sec.skipped
    assert forge_sec.section_id == "forge_invocations"
    assert any("30 invocation(s), 1 KEEP, 2 FAILED" in fact for fact in forge_sec.key_facts)
    assert "named-kernel" in forge_sec.markdown_block


def test_phase_timeline_renderer_renders_capped_histogram() -> None:
    events = ["bootstrap"]
    events.extend(
        {
            "ts": f"t{i}",
            "action": f"action-{i}",
            "decision": "KEEP" if i % 2 == 0 else "REVERT",
            "task_id": f"task-{i}",
            "error_class": "RuntimeError" if i == 30 else "",
        }
        for i in range(31)
    )

    sec = render_phase_timeline({"phase_timeline": events})

    assert not sec.skipped
    assert any("Recorded 32 phase event(s); newest = `action-30` (KEEP)." in fact for fact in sec.key_facts)
    assert any("KEEP=16" in fact and "REVERT=15" in fact and "(none)=1" in fact for fact in sec.key_facts)
    assert "_Showing last 30 of 32 events._" in sec.markdown_block
    assert "action-0" not in sec.markdown_block
    assert "action-30" in sec.markdown_block
    assert "RuntimeError" in sec.markdown_block


def test_compose_omits_retired_v1_1_sections() -> None:
    md = render_session_report(
        _base_breakdown(
            decision_journal=[{"round_id": "b-1"}],
            kernel_profiling=[{"run_id": "p1"}],
            kernel_decision_path=[{"kid": "k1"}],
            data_provenance=[{"section": "source"}],
        )
    ).markdown
    assert "### Decision Journal" not in md
    assert "### Kernel Profiling" not in md
    assert "### Kernel Decision Path" not in md
    assert "### Data Provenance" not in md
