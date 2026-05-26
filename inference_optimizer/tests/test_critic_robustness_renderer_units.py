"""Unit tests for the ``critic_robustness`` breakdown renderer.

Exercises the four observable shapes of the collector input: empty,
prompt-only V1 payloads, V2 dicts with empty fields, and fully-populated
entries with a truncated rationale.
"""

from __future__ import annotations

import pytest

from inference_optimizer.breakdown.reporters._renderers import critic_robustness as cr_mod
from inference_optimizer.breakdown.reporters.base import RenderedSection


def _render(payload):
    return cr_mod.render({"critic_robustness": payload})


class TestCriticRobustnessRenderer:
    def test_empty_returns_skipped(self):
        out = _render([])
        assert isinstance(out, RenderedSection)
        assert out.section_id == "critic_robustness"
        assert out.skipped is True
        assert any("no critic robustness" in s.lower() for s in out.key_facts)

    def test_prompt_only_v1_payload_is_skipped(self):
        out = _render(["raw prompt"])
        assert out.skipped is True
        # The renderer must surface the V1-shape warning so the operator
        # knows the collector ran on an old session.
        assert any("prompt-only" in w for w in out.warnings)

    def test_empty_payloads_v2_is_skipped(self):
        out = _render([
            {"prompt": "x", "response": None, "decision": "", "rationale": ""},
        ])
        assert out.skipped is True
        assert any("non-actionable" in w for w in out.warnings)

    def test_populated_payload_renders_markdown_table(self):
        out = _render([
            {
                "ts": "2026-05-13T01:01:01Z",
                "action": "kernel_opt",
                "decision": "KEEP",
                "pass_count": 3,
                "fail_count": 1,
                "rationale": "Improved attention kernel reduces decode latency by 4%.",
            },
            {
                "prompt": "raw fallback",
            },
        ])
        # Even though one entry is prompt-only, the populated one keeps
        # the section visible.
        assert out.skipped is False
        # Markdown table head wires up to our column names.
        assert "decision" in out.markdown_block
        assert "kernel_opt" in out.markdown_block

    def test_excess_rows_truncated_with_banner(self):
        rows = [
            {
                "decision": "KEEP",
                "pass_count": 1,
                "fail_count": 0,
                "ts": f"t{i}",
            }
            for i in range(cr_mod._MAX_ROWS + 5)
        ]
        out = _render(rows)
        assert out.skipped is False
        assert "Showing first" in out.markdown_block
