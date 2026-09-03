# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression guards for the retired kernel-decision-path renderer."""

from __future__ import annotations

import importlib.util

from hyperloom.inference_optimizer.breakdown.reporters import render_session_report
from hyperloom.inference_optimizer.breakdown.reporters.base import REGISTRY
from hyperloom.inference_optimizer.breakdown.reporters.compose import SECTION_GROUPS


_MODULE = "hyperloom.inference_optimizer.breakdown.reporters._renderers.kernel_decision_path"


def _report(payload):
    return render_session_report({"session": {"session_id": "legacy"}, "kernel_decision_path": payload})


def test_fmt_duration_none():
    assert importlib.util.find_spec(_MODULE) is None


def test_fmt_duration_non_numeric():
    assert "kernel_decision_path" not in {section_id for section_id, _render in REGISTRY}


def test_fmt_duration_seconds():
    grouped = {section_id for _title, section_ids in SECTION_GROUPS for section_id in section_ids}
    assert "kernel_decision_path" not in grouped


def test_fmt_duration_minutes():
    assert "Kernel Decision Path" not in _report([]).markdown


def test_render_absent_field_skipped():
    assert all(section.section_id != "kernel_decision_path" for section in _report(None).sections)


def test_render_empty_entries_skipped():
    assert all(section.section_id != "kernel_decision_path" for section in _report([]).sections)


def test_render_with_entries():
    report = _report([{"kid": "dead-kernel", "steps": [{"step": "kernel_opt"}]}]).markdown
    assert "dead-kernel" not in report
    assert "kernel_opt" not in report


def test_render_truncates_steps_and_kids():
    payload = [{"kid": f"dead-{index}", "steps": [{"step": "dead-step"}] * 15} for index in range(10)]
    report = _report(payload).markdown
    assert "dead-step" not in report


def test_render_ignores_non_dict_entries():
    assert "bad" not in _report(["bad", 123]).markdown


def test_render_entry_without_steps():
    assert "dead-entry" not in _report([{"kid": "dead-entry", "steps": []}]).markdown
