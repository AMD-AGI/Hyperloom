# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE writes the final recipe, its files and the session's lessons to the Recipe KB, only past the champion."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

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
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom.orchestrator.knowledge.remote_recipe.values import knowledge_to_warm_recipe
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.knowledge.recipe_kb import from_recipe_kb_value, to_recipe_kb_value
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import (
    KB_METRIC,
    OBJECTIVE_SCHEMA,
    FinalRecipe,
    publish_final_recipe,
    recipe_identity,
)
from hyperloom_multibranch.notebook.knowledge.tests.fakes import FakeRecipeStore
from hyperloom_multibranch.notebook.tests.factories import profile
from hyperloom_multibranch.notebook.tests.sessions import METRIC

PATCH = b"diff --git a/moe.py b/moe.py\n"
KERNEL = b'{"tile": [128, 128]}\n'
OVERLAY = "patch/overlays/000000/00-0001-moe.patch"


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
    replayed = from_recipe_kb_value(knowledge["value"], base_image_digest=provenance["base_image_digest"])
    assert replayed == final.recipe
    warm = knowledge_to_warm_recipe({"knowledge": knowledge, "session_id": "sess-1"})
    assert (warm["best_throughput"], warm["replay_material_available"], warm["isl"]) == (1041.0, True, 1024)


def test_a_session_short_of_the_champion_writes_nothing(snapshot: NotebookSnapshot, final: FinalRecipe) -> None:
    store = FakeRecipeStore(champion=_champion(2000.0))

    outcome = publish_final_recipe(RemoteRecipeClient(store), snapshot, final)

    assert (outcome.status, outcome.reason) == ("skipped", "not_better_than_champion")
    assert (store.uploads, store.knowledge, store.promoted) == ({}, None, None)


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
        (profile(isl=None, osl=None), "agentx_objective_not_recorded"),
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
