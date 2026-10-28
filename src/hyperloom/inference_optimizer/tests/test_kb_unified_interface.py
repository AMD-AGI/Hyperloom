"""Tests for the local Recipe KB warm-start interface."""

from __future__ import annotations

from hyperloom.orchestrator.knowledge.recipe_kb_t0 import (
    _build_warm_start_context,
    _recipe_is_actionable,
    WarmMatch,
)


def test_recipe_actionability_and_warm_context() -> None:
    assert _recipe_is_actionable({"best_config": {"extra_server_args": "--x 1"}})
    assert not _recipe_is_actionable({"canonical_id": "inference:m:h:f:mt:a:v:p", "best_config": {}})
    context = _build_warm_start_context(
        match=WarmMatch(
            {
                "canonical_id": "inference:m:h:f:mt:a:v:p",
                "best_config": {
                    "extra_server_args": "--x 1",
                    "extra_envs": {"A": "1"},
                },
                "best_throughput": 100.0,
            },
            "exact",
            1.0,
        ),
        canonical_id="inference:m:h:f:mt:a:v:p",
        source="recipe-kb",
    )
    assert context["status"] == "hit"
    assert context["recommended_replay"]["extra_server_args"] == "--x 1"
    assert context["recommended_replay"]["extra_envs"] == {"A": "1"}
