# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the ``param_search`` breakdown renderer."""

from __future__ import annotations


from hyperloom.inference_optimizer.breakdown.reporters._renderers import param_search as ps_mod
from hyperloom.inference_optimizer.breakdown.reporters.base import RenderedSection


def _render(*attempts):
    """Render a breakdown whose framework event holds ``attempts``."""
    return ps_mod.render({"timeline": [{"type": "framework_agent", "ext": {"attempts": list(attempts)}}]})


def _attempt(name: str, outcome: str, gain: float | None = None, *, arm: str = "config", round_id: str = "r1"):
    """One configuration-arm attempt row as the recorder writes it."""
    return {
        "attempt_id": f"t:{round_id}:{name}",
        "arm": arm,
        "round_id": round_id,
        "variant_name": name,
        "outcome": outcome,
        "measurement": {"after_tput": 2400.0, "gain_pct": gain},
    }


class TestParamSearchRenderer:
    def test_a_session_with_no_framework_event_is_skipped(self):
        out = ps_mod.render({})
        assert isinstance(out, RenderedSection)
        assert out.skipped is True
        assert out.markdown_block == ""

    def test_an_arm_that_measured_nothing_is_skipped_and_says_so(self):
        out = _render()
        assert out.skipped is True
        assert out.warnings == ["The configuration arm measured no variant this session."]

    def test_measured_variants_unskip_the_section(self):
        out = _render(_attempt("kv_fp8", "KEEP", 10.99), _attempt("chunked", "REVERT", -1.2))

        assert out.skipped is False
        assert "Explore Search" in out.markdown_block
        assert "kv_fp8" in out.markdown_block
        assert any("2 variant(s) measured" in fact and "1 kept" in fact for fact in out.key_facts)

    def test_the_source_arm_is_not_a_parameter_search(self):
        """The source arm authors patches; it has its own place in the report."""
        out = _render(_attempt("patch-1", "KEEP", 3.0, arm="source"))

        assert out.skipped is True

    def test_the_best_gain_is_reported_and_the_table_leads_with_it(self):
        out = _render(
            _attempt("small", "REVERT", 0.4),
            _attempt("big", "KEEP", 12.5),
        )

        assert any("Best measured variant gain: +12.50%" in fact for fact in out.key_facts)
        body = out.markdown_block
        assert body.index("big") < body.index("small")

    def test_a_withheld_winner_is_reported_apart_from_a_plain_revert(self):
        """``KEEP_UNSTABLE`` is a variant that won and did not reproduce, which
        is a different fact from one that lost on measurement."""
        out = _render(_attempt("flaky", "KEEP_UNSTABLE", 5.0))

        assert any("withheld" in fact for fact in out.key_facts)
        assert out.key_facts[0].endswith("0 kept.")
