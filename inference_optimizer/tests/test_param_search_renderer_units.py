# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the ``param_search`` breakdown renderer."""

from __future__ import annotations


from inference_optimizer.breakdown.reporters._renderers import param_search as ps_mod
from inference_optimizer.breakdown.reporters.base import RenderedSection


def _render(payload):
    return ps_mod.render({"param_search": payload})


class TestParamSearchRenderer:
    def test_empty_payload_is_skipped(self):
        out = _render({})
        assert isinstance(out, RenderedSection)
        assert out.skipped is True
        assert any("backends DFS" in f for f in out.key_facts)
        assert any("params DFS" in f for f in out.key_facts)

    def test_populated_explore_unskips_section(self):
        out = _render({
            "explore": {"accepted": ["a"], "tested": {"a": {}}, "cursor": 1,
                        "last_round": 2},
        })
        assert out.skipped is False
        assert "Explore Search" in out.markdown_block

    def test_populated_backends_unskips_section(self):
        out = _render({
            "backends": {"accepted": ["a"], "tested": {"a": {}}, "cursor": 1,
                         "last_round": 2},
            "params":   {"accepted": [], "tested": {}, "cursor": 0,
                         "last_round": 0},
        })
        assert out.skipped is False
        assert "Backends DFS" in out.markdown_block

    def test_discovered_flags_rendered_per_framework(self):
        out = _render({
            "discovered_flags": {
                "sglang": {
                    "backend_flags": ["a", "b"],
                    "param_flags": ["x"],
                    "source_path": "/tmp/sglang/server_args.py",
                },
            },
        })
        # Non-empty flags inventory keeps the section unskipped.
        assert any(
            "discovered_flags[sglang]" in f and "param=1" in f
            for f in out.key_facts
        )
        assert out.skipped is False

    def test_winners_history_truncated_to_last_five(self):
        winners = [
            {"round_id": i, "action": "backends", "base_tput": 100.0 + i,
             "best": {"name": f"v{i}", "gain_pct": 2.5}}
            for i in range(8)
        ]
        out = _render({"backend_winners_history": winners})
        assert "Backend winners history" in out.markdown_block
        assert "v7" in out.markdown_block
        assert "| v0 |" not in out.markdown_block

    def test_synergy_truncation_marker(self):
        synergy = [f"combo_{i}" for i in range(25)]
        out = _render({"synergy_attempted": synergy})
        assert "Synergy combos attempted" in out.markdown_block
        assert "(+5 more)" in out.markdown_block
