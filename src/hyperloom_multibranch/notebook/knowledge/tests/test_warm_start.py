# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE reads the Recipe KB champions of this session's task and scope as leads, whichever flow wrote them."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.recipe import LaunchConfig, Recipe
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom.orchestrator.knowledge.remote_recipe.values import (
    CURRENT_KNOWLEDGE_SCHEMA_VERSION,
    RECORD_KIND_HYPERLOOM_RECIPE,
)
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import FinalRecipe, knowledge_document, recipe_identity
from hyperloom_multibranch.notebook.knowledge.tests.fakes import FakeRecipeStore
from hyperloom_multibranch.notebook.knowledge.warm_start import identity_labels, warm_leads
from hyperloom_multibranch.notebook.tests.factories import profile
from hyperloom_multibranch.notebook.tests.sessions import METRIC

PROFILE = profile()
EXACT_ID, SCOPE = recipe_identity(PROFILE)
OLDER_ID, _ = recipe_identity(profile(framework_version="0.10.0"))
LABELS = identity_labels(PROFILE)


def _envelope(canonical_id: str, session_id: str, knowledge: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "canonical_id": canonical_id,
        "session_id": session_id,
        "record_id": f"record-{session_id}",
        "revision": 1,
        "knowledge": knowledge,
        "view": {"source": "current", "replayable": True},
        "scope": SCOPE.as_dict(),
        "artifacts": {"files": [], "file_count": 0},
        **overrides,
    }


def _single_lineage_knowledge() -> dict[str, Any]:
    return {
        "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
        "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
        "optimized_throughput": 1100.0,
        "validated_e2e_gain": 10.0,
        "workload_shape": {"tp": 8, "conc": 64, "isl": 1024, "osl": 1024},
        "value": {
            "config": {"extra_server_args": "--enable-expert-parallel", "extra_envs": {"VLLM_USE_AITER": "1"}},
            "patch": {"patches": [], "provenance": []},
            "kernel": {},
        },
        "lessons": ["expert parallelism pays at conc 64"],
        "pitfalls": ["fp8 kv cache fails accuracy"],
        "what_failed": ["block size 32 regressed"],
        "provenance": {"kernel_optimizer": "forge"},
    }


@pytest.fixture
def store(snapshot: NotebookSnapshot, tmp_path: Path) -> FakeRecipeStore:
    ours = knowledge_document(
        snapshot,
        PROFILE,
        FinalRecipe("i1", Recipe("sha256:" + "a" * 64, LaunchConfig("--max-num-seqs 256")), tmp_path, METRIC),
        SCOPE,
    )
    malformed = _envelope("inference:malformed", "bad", ours, schema_version=1)
    other_scope = _envelope("inference:other-scope", "wide", ours, scope={**SCOPE.as_dict(), "conc": 128})
    return FakeRecipeStore(
        identities=[
            {"canonical_id": OLDER_ID, "dimensions": {**LABELS, "framework_version": "0.10.0"}},
            {"canonical_id": EXACT_ID, "dimensions": LABELS},
            {"canonical_id": "agentx:elsewhere"},
            {"canonical_id": "inference:malformed"},
            {"canonical_id": "inference:other-scope"},
            {"canonical_id": "inference:nothing-in-scope"},
        ],
        views={
            EXACT_ID: _envelope(EXACT_ID, "sess-1", ours),
            OLDER_ID: _envelope(OLDER_ID, "older", _single_lineage_knowledge()),
            "inference:malformed": malformed,
            "inference:other-scope": other_scope,
        },
    )


def test_leads_come_from_either_flow_exact_identity_first(
    store: FakeRecipeStore, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="hyperloom_multibranch.notebook.knowledge.warm_start"):
        leads = warm_leads(RemoteRecipeClient(store), PROFILE)

    assert [lead.canonical_id for lead in leads] == [EXACT_ID, OLDER_ID]
    exact, older = leads
    assert (exact.session_id, exact.server_args, exact.replayable, exact.differing) == (
        "sess-1",
        "--max-num-seqs 256",
        True,
        {},
    )
    assert exact.gain_pct == pytest.approx(4.1)
    assert exact.what_failed[0] == "attention / split kv on c0: regressed at -2.00%"
    assert (older.gain_pct, older.best_throughput, dict(older.envs)) == (10.0, 1100.0, {"VLLM_USE_AITER": "1"})
    assert (older.lessons, older.pitfalls, older.what_failed) == (
        ("expert parallelism pays at conc 64",),
        ("fp8 kv cache fails accuracy",),
        ("block size 32 regressed",),
    )
    assert dict(older.differing) == {"framework_version": ("0.10.0", LABELS["framework_version"])}
    assert older.render().startswith(f"- {OLDER_ID}, framework_version 0.10.0 (here ")
    assert "inference:malformed" in caplog.text and "inference:other-scope" in caplog.text


def test_the_search_matches_six_dimensions_page_by_page(store: FakeRecipeStore) -> None:
    warm_leads(RemoteRecipeClient(store), PROFILE)

    assert [search["offset"] for search in store.searches] == [0, 2, 4]
    assert store.searches[0]["scheme"] == "inference"
    assert store.searches[0]["match"] == {key: value for key, value in LABELS.items() if key != "framework_version"}
    assert len(warm_leads(RemoteRecipeClient(store), PROFILE, limit=1)) == 1


def test_a_scope_the_kb_cannot_file_reads_nothing(store: FakeRecipeStore) -> None:
    assert warm_leads(RemoteRecipeClient(store), profile(kernel_optimizer="")) == ()
    assert store.searches == []


def test_a_lead_without_values_renders_what_it_has(store: FakeRecipeStore) -> None:
    store.views[OLDER_ID]["knowledge"]["value"]["config"] = {"extra_envs": "not a mapping"}
    store.views[OLDER_ID]["knowledge"].pop("validated_e2e_gain")

    older = warm_leads(RemoteRecipeClient(store), PROFILE)[1]

    assert (older.server_args, dict(older.envs)) == ("", {})
    assert older.gain_pct == 0.0 and older.render().endswith(": +0.00% [older]")
