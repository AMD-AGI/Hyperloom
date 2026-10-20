# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The multi-branch ``session_breakdown.json``: every notebook fact, and the write-back outcome."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from hyperloom.contracts.recipe import LaunchConfig, Recipe
from hyperloom.orchestrator.knowledge.remote_recipe.models import RemoteWriteResult
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import LlmUsage, LlmUsageRecord, NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import UsageSource
from hyperloom_multibranch.notebook.knowledge.breakdown import (
    EXPORTER_VERSION,
    FILENAME,
    SCHEMA_VERSION,
    build_breakdown,
    write_breakdown,
)
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import FinalRecipe

FINAL_RECIPE = Recipe("rocm/vllm@sha256:" + "a" * 64, LaunchConfig(server_args="--tensor-parallel-size 8"))
NO_TOKENS = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cache_write_1h": 0}


def test_the_breakdown_lays_out_the_session_tree_tests_and_measurement(snapshot: NotebookSnapshot) -> None:
    written = RemoteWriteResult("written", "", "inference:x", "sess-1", "optimized_throughput", 1041.0)

    breakdown = build_breakdown(snapshot, kb_write_back=written)

    assert (breakdown["schema_version"], breakdown["exporter_version"]) == (SCHEMA_VERSION, EXPORTER_VERSION)
    assert breakdown["metadata"]["session_profile"]["session_id"] == "sess-1"
    assert [row["activity"] for row in breakdown["metadata"]["manifest_fingerprints"]] == ["act-1"]
    assert breakdown["baseline"]["values"] == {"output_tps": 1000.0}
    assert {node["commit_id"] for node in breakdown["lineage"]["commits"]} == {"c0", "c1", "i1"}
    assert {row["branch"]: row["llm_usd"] for row in breakdown["branches"]} == {
        INTEGRATION_BRANCH: 0.0,
        "s1": 1.25,
        "s2": 0.5,
    }
    assert breakdown["spend"] == {"gpu_hours": 1.0, "llm_usd": 2.0, "llm_unpriced_tokens": NO_TOKENS}
    assert breakdown["environment_changes"] == []
    assert [row["request"]["request_id"] for row in breakdown["tests"]] == ["r1", "r2", "r3", "r5", "r4", "m1"]
    assert [(row["node"], row["tests"], row["gpu_seconds"]) for row in breakdown["measurement_sandboxes"]] == [
        ("node-1", 5, 3000.0),
        ("node-2", 1, 600.0),
    ]
    assert breakdown["close"]["kb_write_back"]["status"] == "written"
    assert build_breakdown(snapshot)["close"] == {"final": None, "selection": {}, "kb_write_back": None}


def test_the_spend_counts_the_tokens_of_unpriced_llm_calls_beside_the_dollars(snapshot: NotebookSnapshot) -> None:
    tokens = TokenCounts(input=1_000, output=500, cache_read=20_000, cache_write_1h=300)
    unpriced = LlmUsageRecord("u1", snapshot.taken_at, LlmUsage("", UsageSource.JUDGE, None, tokens))

    spend = build_breakdown(dataclasses.replace(snapshot, llm_usage=(*snapshot.llm_usage, unpriced)))["spend"]

    assert spend["llm_usd"] == 2.0
    assert spend["llm_unpriced_tokens"] == {
        **NO_TOKENS,
        "input": 1_000,
        "output": 500,
        "cache_read": 20_000,
        "cache_write_1h": 300,
    }


def test_the_close_section_names_the_commit_final_selection_chose_and_how(snapshot: NotebookSnapshot) -> None:
    final = FinalRecipe("i1", FINAL_RECIPE, Path("files"), "output_tps")
    selection = {"candidates": ["i1", "c1"], "chosen": "i1", "reason": "the integration head held its gain"}

    close = build_breakdown(snapshot, final=final, selection=selection)["close"]

    assert close["final"] == {
        "commit_id": "i1",
        "recipe_digest": FINAL_RECIPE.digest(),
        "recipe": FINAL_RECIPE.to_document(),
    }
    assert close["selection"] == selection


def test_the_breakdown_is_written_as_sorted_indented_json(snapshot: NotebookSnapshot, tmp_path: Path) -> None:
    breakdown = build_breakdown(snapshot)

    path = write_breakdown(tmp_path / "export", breakdown)

    assert path == tmp_path / "export" / FILENAME
    assert json.loads(path.read_text(encoding="utf-8")) == breakdown
    assert path.read_text(encoding="utf-8").startswith('{\n  "baseline"')
