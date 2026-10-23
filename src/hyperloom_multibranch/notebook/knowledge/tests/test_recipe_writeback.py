# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE writes the final recipe, its files and the session's lessons to the Recipe KB, only past the champion."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.perf_metric import GRADED_INTVTY_P50, GRADED_TOTAL
from hyperloom.contracts.recipe import (
    DIGEST_PREFIX,
    ComponentPatches,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeError,
    with_unexpressed,
)
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.orchestrator.knowledge.remote_recipe import KBStoreError
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom.orchestrator.knowledge.remote_recipe.values import knowledge_to_warm_recipe
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.repo import ResolvedPatch
from hyperloom_multibranch.notebook.knowledge.recipe_kb import from_recipe_kb_value, to_recipe_kb_value
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import (
    AGENTX_METRIC,
    AGENTX_OBJECTIVE_SCHEMA,
    KB_METRIC,
    OBJECTIVE_SCHEMA,
    FinalRecipe,
    publish_final_recipe as _publish_final_recipe,
    recipe_identity,
)
from hyperloom_multibranch.notebook.knowledge.tests.fakes import FakeRecipeStore
from hyperloom_multibranch.notebook.tests.factories import profile
from hyperloom_multibranch.notebook.tests.sessions import METRIC

PATCH = b"diff --git a/moe.py b/moe.py\n"
KERNEL = b'{"tile": [128, 128]}\n'
OVERLAY = "patch/overlays/000000/00-0001-moe.patch"
RULE_PACK = {"version": "1", "digest": "sha256:" + "c" * 64}


def publish_final_recipe(
    client: RemoteRecipeClient, snapshot: NotebookSnapshot, final: FinalRecipe, rule_pack=RULE_PACK
):
    return _publish_final_recipe(client, snapshot, final, rule_pack)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def final(tmp_path: Path) -> FinalRecipe:
    checkout = tmp_path / "checkout"
    (checkout / "patch" / "vllm").mkdir(parents=True)
    (checkout / "kernel" / "gemm").mkdir(parents=True)
    (checkout / "patch" / "vllm" / "0001-moe.patch").write_bytes(PATCH)
    (checkout / "kernel" / "gemm" / "moe.json").write_bytes(KERNEL)
    recipe = Recipe(
        base_image_digest="rocm/vllm@sha256:" + "a" * 64,
        launch=LaunchConfig(server_args="--max-num-seqs 256"),
        components=(
            ComponentPatches(
                "vllm",
                "https://github.com/vllm-project/vllm",
                "b" * 40,
                (PatchRef("patch/vllm/0001-moe.patch", _sha(PATCH)),),
            ),
        ),
        kernels=(KernelArtifact(KernelKind.GEMM, "moe_e8", "kernel/gemm/moe.json", _sha(KERNEL)),),
    )
    return FinalRecipe("i1", recipe, checkout, METRIC)


def _champion(value: float) -> dict[str, object]:
    return {"metric": KB_METRIC, "session_id": "older", "value": value}


def test_a_session_past_the_champion_writes_its_recipe_files_and_lessons(
    snapshot: NotebookSnapshot, final: FinalRecipe
) -> None:
    store = FakeRecipeStore(champion=_champion(1000.0))

    outcome = publish_final_recipe(RemoteRecipeClient(store), snapshot, final)

    assert (outcome.status, outcome.primary_metric, outcome.primary_value) == ("written", KB_METRIC, 1041.0)
    assert outcome.canonical_id.startswith("inference:") and outcome.session_id == "sess-1"
    assert store.uploads == {OVERLAY: PATCH, "kernel/gemm/moe.json": KERNEL}
    assert store.promoted == ("sess-1", KB_METRIC, 1041.0)
    assert store.objective_schema == OBJECTIVE_SCHEMA
    knowledge = store.knowledge
    assert knowledge is not None
    assert knowledge["value"] == to_recipe_kb_value(final.recipe)
    assert knowledge["validated_e2e_gain"] == pytest.approx(4.1)
    assert knowledge["workload_shape"] == {"tp": 8, "conc": 64, "isl": 1024, "osl": 1024}
    assert knowledge["what_worked"][0] == "merge of c1 on c0: +4.10%, merged as i1"
    assert knowledge["what_failed"][0] == "attention / split kv on c0: regressed at -2.00%"
    assert len(knowledge["pitfalls"]) == 2 and len(knowledge["lessons"]) == 2
    assert knowledge["remaining_gaps"] == ["output_tps: 1041.0 is 69% of the roofline bound 1500.0"]
    provenance = knowledge["provenance"]
    assert (provenance["commit_id"], provenance["recipe_digest"]) == ("i1", final.recipe.digest())
    assert provenance["rule_pack"] == RULE_PACK
    assert "warmup_round_seed" not in provenance and "synthesized_patches" not in provenance
    replayed = from_recipe_kb_value(knowledge["value"], base_image_digest=provenance["base_image_digest"])
    assert replayed == final.recipe
    warm = knowledge_to_warm_recipe({"knowledge": knowledge, "session_id": "sess-1"})
    assert (warm["best_throughput"], warm["replay_material_available"], warm["isl"]) == (1041.0, True, 1024)


def test_a_patch_a_model_wrote_during_a_merge_is_named_in_the_provenance(
    snapshot: NotebookSnapshot, final: FinalRecipe
) -> None:
    """The KB says which published patch no branch wrote, and leaves out a model's patch the recipe does not carry."""

    store = FakeRecipeStore(champion=_champion(1000.0))
    published = final.recipe.components[0].patches[0]
    merged = ResolvedPatch("vllm", "patch/vllm/0000-old.patch", published, "claude-x", "resolver-3", 2)
    dropped = ResolvedPatch(
        "vllm", "patch/vllm/0002-other.patch", PatchRef("patch/vllm/m.patch", "d" * 64), "m", "v", 1
    )

    publish_final_recipe(
        RemoteRecipeClient(store), dataclasses.replace(snapshot, resolved_patches=(merged, dropped)), final
    )

    assert store.knowledge is not None
    assert store.knowledge["provenance"]["synthesized_patches"] == [
        {
            "component": "vllm",
            "path": published.path,
            "sha256": published.sha256,
            "replaces": "patch/vllm/0000-old.patch",
            "model": "claude-x",
            "prompt_version": "resolver-3",
            "attempts": 2,
        }
    ]
    assert store.knowledge["value"] == to_recipe_kb_value(final.recipe)


def test_a_session_short_of_the_champion_writes_nothing(snapshot: NotebookSnapshot, final: FinalRecipe) -> None:
    store = FakeRecipeStore(champion=_champion(2000.0))

    outcome = publish_final_recipe(RemoteRecipeClient(store), snapshot, final)

    assert (outcome.status, outcome.reason) == ("skipped", "not_better_than_champion")
    assert (store.uploads, store.knowledge, store.promoted) == ({}, None, None)


class _ReadOnlyStore(FakeRecipeStore):
    def put_dir(self, canonical_id: str, session_id: str, *_: Any, **__: Any) -> dict[str, str]:
        raise KBStoreError(f"POST /v1/kb/{canonical_id}/sessions/{session_id}/files:grant -> HTTP 403: write_forbidden")


def test_a_write_the_kb_store_refuses_is_logged_as_an_error_and_raised(
    snapshot: NotebookSnapshot, final: FinalRecipe, caplog: pytest.LogCaptureFixture
) -> None:
    store = _ReadOnlyStore(champion=_champion(1000.0))

    with caplog.at_level(logging.ERROR), pytest.raises(KBStoreError, match="write_forbidden"):
        publish_final_recipe(RemoteRecipeClient(store), snapshot, final)

    (logged,) = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert logged.startswith("the Recipe KB refused the final recipe of sess-1 on inference:") and "403" in logged
    assert (store.knowledge, store.promoted) == (None, None)


def test_a_file_that_differs_from_its_pinned_hash_stops_the_write(
    snapshot: NotebookSnapshot, final: FinalRecipe
) -> None:
    (final.files_root / "kernel" / "gemm" / "moe.json").write_bytes(b"{}")
    store = FakeRecipeStore()

    with pytest.raises(RecipeError, match="the recipe pins"):
        publish_final_recipe(RemoteRecipeClient(store), snapshot, final)
    (final.files_root / "kernel" / "gemm" / "moe.json").unlink()
    with pytest.raises(RecipeError, match="not in the checkout"):
        publish_final_recipe(RemoteRecipeClient(store), snapshot, final)
    assert store.uploads == {}


def test_with_the_session_blob_store_the_files_come_from_it_by_hash(
    snapshot: NotebookSnapshot, final: FinalRecipe, tmp_path: Path
) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    blobs.put(PATCH)
    blobs.put(KERNEL)
    empty_checkout = tmp_path / "recipe-only"
    empty_checkout.mkdir()
    store = FakeRecipeStore(champion=_champion(1000.0))

    outcome = publish_final_recipe(
        RemoteRecipeClient(store), snapshot, dataclasses.replace(final, files_root=empty_checkout, blobs=blobs)
    )

    assert outcome.status == "written"
    assert store.uploads == {OVERLAY: PATCH, "kernel/gemm/moe.json": KERNEL}


def test_a_blob_missing_from_the_store_stops_the_write(
    snapshot: NotebookSnapshot, final: FinalRecipe, tmp_path: Path
) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    blobs.put(PATCH)
    store = FakeRecipeStore()

    with pytest.raises(RecipeError, match="not in the blob store"):
        publish_final_recipe(RemoteRecipeClient(store), snapshot, dataclasses.replace(final, blobs=blobs))
    assert store.uploads == {}


def test_the_final_commit_needs_a_recorded_value(snapshot: NotebookSnapshot, final: FinalRecipe) -> None:
    with pytest.raises(RecipeError, match="recorded output_tps"):
        publish_final_recipe(
            RemoteRecipeClient(FakeRecipeStore()), snapshot, dataclasses.replace(final, commit_id="c9")
        )


@pytest.mark.parametrize(
    ("session_profile", "reason"),
    [
        (None, "no_session_profile"),
        (profile(kernel_optimizer=""), "invalid_recipe_scope"),
    ],
)
def test_a_session_the_kb_cannot_file_is_skipped_with_the_reason(
    snapshot: NotebookSnapshot, final: FinalRecipe, session_profile: object, reason: str
) -> None:
    store = FakeRecipeStore()

    outcome = publish_final_recipe(
        RemoteRecipeClient(store), dataclasses.replace(snapshot, profile=session_profile), final
    )

    assert (outcome.status, outcome.reason) == ("skipped", reason)
    assert store.scopes == []


def _agentx(snapshot: NotebookSnapshot, baseline: dict[str, float], final: dict[str, float]) -> NotebookSnapshot:
    """The scripted session as an AgentX one: no ISL or OSL, and interactivity recorded at the root and at ``i1``."""

    root = dataclasses.replace(snapshot.baselines[-1], values=baseline)
    records = tuple(
        dataclasses.replace(record, judgement=dataclasses.replace(record.judgement, recorded_value=final))
        if record.commit_id == "i1" and record.judgement is not None
        else record
        for record in snapshot.records
    )
    return dataclasses.replace(
        snapshot, profile=profile(isl=None, osl=None), baselines=(*snapshot.baselines[:-1], root), records=records
    )


def test_an_agentx_session_writes_its_interactivity_gain_as_the_main_flow_does(
    snapshot: NotebookSnapshot, final: FinalRecipe
) -> None:
    store = FakeRecipeStore(champion={"metric": AGENTX_METRIC, "session_id": "older", "value": 5.0})
    agentx = _agentx(
        snapshot, {GRADED_INTVTY_P50: 10.0, GRADED_TOTAL: 2000.0}, {GRADED_INTVTY_P50: 11.0, GRADED_TOTAL: 2100.0}
    )

    outcome = publish_final_recipe(
        RemoteRecipeClient(store), agentx, dataclasses.replace(final, metric=GRADED_INTVTY_P50)
    )

    assert (outcome.status, outcome.primary_metric) == ("written", AGENTX_METRIC)
    assert outcome.canonical_id.startswith("agentx:")
    assert store.promoted == ("sess-1", AGENTX_METRIC, pytest.approx(10.0))
    assert store.objective_schema == AGENTX_OBJECTIVE_SCHEMA
    knowledge = store.knowledge
    assert knowledge is not None
    metrics = ("candidate_interactivity", "baseline_interactivity", GRADED_TOTAL, "baseline_total_throughput")
    assert [knowledge[key] for key in metrics] == [11.0, 10.0, 2100.0, 2000.0]
    assert KB_METRIC not in knowledge and knowledge["workload_shape"] == {"tp": 8, "conc": 64}
    warm = knowledge_to_warm_recipe({"knowledge": knowledge, "session_id": "sess-1"})
    assert warm["validated_gain_pct"] == pytest.approx(10.0)


def test_an_agentx_session_without_total_throughput_carries_interactivity_alone_and_meets_the_champion_on_it(
    snapshot: NotebookSnapshot, final: FinalRecipe
) -> None:
    agentx = _agentx(snapshot, {GRADED_INTVTY_P50: 10.0}, {GRADED_INTVTY_P50: 11.0})
    on_interactivity = dataclasses.replace(final, metric=GRADED_INTVTY_P50)
    written, kept = (
        FakeRecipeStore(champion={"metric": AGENTX_METRIC, "session_id": "older", "value": value})
        for value in (5.0, 20.0)
    )

    publish_final_recipe(RemoteRecipeClient(written), agentx, on_interactivity)
    outcome = publish_final_recipe(RemoteRecipeClient(kept), agentx, on_interactivity)

    assert written.knowledge is not None
    assert {key: written.knowledge.get(key) for key in (GRADED_TOTAL, "baseline_total_throughput")} == {
        GRADED_TOTAL: None,
        "baseline_total_throughput": None,
    }
    assert (outcome.status, outcome.reason, kept.promoted) == ("skipped", "not_better_than_champion", None)


@pytest.mark.parametrize(
    ("notes", "seed"),
    [
        (("run_grid_warmup_discarded_first", "warmup_round_seed:1"), "1"),
        (("warmup_round_seed:1", "run_grid_warmup_seed_unpatched"), "unpatched"),
    ],
)
def test_the_record_names_how_the_warmup_behind_the_final_commit_ran(
    snapshot: NotebookSnapshot, final: FinalRecipe, notes: tuple[str, ...], seed: str
) -> None:
    store = FakeRecipeStore(champion=_champion(1000.0))
    records = tuple(
        dataclasses.replace(record, result=dataclasses.replace(record.result, nonfatal_warnings=notes))
        if record.request_id == "m1" and record.result is not None
        else record
        for record in snapshot.records
    )

    publish_final_recipe(RemoteRecipeClient(store), dataclasses.replace(snapshot, records=records), final)

    assert store.knowledge is not None
    assert store.knowledge["provenance"]["warmup_round_seed"] == seed


def _recorded_with(snapshot: NotebookSnapshot, commit_id: str, digest: str) -> NotebookSnapshot:
    commits = tuple(
        dataclasses.replace(node, recipe_digest=digest) if node.commit_id == commit_id else node
        for node in snapshot.lineage.commits
    )
    return dataclasses.replace(snapshot, lineage=dataclasses.replace(snapshot.lineage, commits=commits))


def test_a_recipe_other_than_the_one_the_notebook_recorded_is_not_written(
    snapshot: NotebookSnapshot, final: FinalRecipe
) -> None:
    store = FakeRecipeStore(champion=_champion(1000.0))
    other = _recorded_with(snapshot, "i1", DIGEST_PREFIX + "0" * 64)
    same = _recorded_with(snapshot, "i1", final.recipe.digest())

    refused = publish_final_recipe(RemoteRecipeClient(store), other, final)
    written = publish_final_recipe(RemoteRecipeClient(store), same, final)

    assert (refused.status, refused.reason) == ("skipped", "recipe_digest_mismatch")
    assert written.status == "written"


def test_a_recipe_that_leaves_part_of_the_measured_environment_out_is_not_written(
    snapshot: NotebookSnapshot, final: FinalRecipe
) -> None:
    store = FakeRecipeStore(champion=_champion(1000.0))
    incomplete = dataclasses.replace(final, recipe=with_unexpressed(final.recipe, ["patch:site-packages/vllm/x.py"]))

    outcome = publish_final_recipe(RemoteRecipeClient(store), snapshot, incomplete)

    assert (outcome.status, outcome.reason) == ("skipped", "incomplete_recipe")
    assert (store.uploads, store.knowledge, store.promoted) == ({}, None, None)


def test_the_identity_follows_the_profile_and_agentx_has_its_own_scheme() -> None:
    inference_id, scope = recipe_identity(profile())
    agentx_id, agentx_scope = recipe_identity(profile(isl=None, osl=None))

    assert inference_id.startswith("inference:") and agentx_id.startswith("agentx:")
    assert scope.as_dict() == {"kernel_optimizer": "forge", "tp": 8, "conc": 64, "isl": 1024, "osl": 1024}
    assert agentx_scope.as_dict() == {"kernel_optimizer": "forge", "tp": 8, "conc": 64}
