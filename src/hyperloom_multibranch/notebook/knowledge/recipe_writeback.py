# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE: the final recipe and what the session learnt, written to the Recipe KB when it beats the champion.

The record carries the columns the single-lineage CLOSE writes: the identity and scope come from the session profile
PRELUDE wrote, ``value`` from :func:`to_recipe_kb_value`, the experience lists from :func:`session_lessons`. The base
image digest and runtime snapshot, which ``value`` leaves out, travel in the provenance. Patch and kernel files are
read from the session's blob store by the hash the recipe pins (the experiment repository holds recipes only), or
copied out of a checkout when no store is given, and staged under the refs ``value`` names; each is checked against
its pinned hash, so the KB stores exactly what was measured.

InferenceX sessions compare ``optimized_throughput`` with the champion of their partition. An AgentX champion compares
an interactivity gain graded on two axes the multi-branch gate does not record yet, so AgentX sessions are skipped,
with that reason, until it does. When the notebook recorded the final commit's recipe digest, a recipe with another
digest is skipped too (``recipe_digest_mismatch``): CLOSE was handed some other commit's recipe. A recipe that names
anything as unexpressed does not rebuild what was measured, so it is skipped as well (``incomplete_recipe``).
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hyperloom.contracts.recipe import Recipe, RecipeError
from hyperloom.inference_optimizer.recipe_snapshot_constants import recipe_canonical_id
from hyperloom.measure.compile import BlobStore
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom.orchestrator.knowledge.remote_recipe.models import (
    Artifact,
    KnowledgeBundle,
    RecipeScope,
    RemoteRecipeValidationError,
    RemoteWriteResult,
)
from hyperloom.orchestrator.knowledge.remote_recipe.sanitize import sanitize_shared_knowledge
from hyperloom.orchestrator.knowledge.remote_recipe.values import (
    CURRENT_KNOWLEDGE_SCHEMA_VERSION,
    RECORD_KIND_HYPERLOOM_RECIPE,
)
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot, SessionProfile
from hyperloom_multibranch.notebook.knowledge.lessons import session_lessons
from hyperloom_multibranch.notebook.knowledge.recipe_kb import overlay_ref, to_recipe_kb_value

KB_METRIC = "optimized_throughput"
GAIN_METRIC = "validated_e2e_gain"
OBJECTIVE_SCHEMA = "single_throughput"
PRODUCER = "hyperloom-multibranch"
SKIPPED = "skipped"
AGENTX = "agentx"


@dataclass(frozen=True)
class FinalRecipe:
    """The commit CLOSE publishes, its recipe, and where the patch and kernel files the recipe names are read from.

    ``metric`` is the recorded-value metric that is the session's output throughput. With ``blobs`` (the session's
    blob store) every file is read from it by its pinned sha256 and ``files_root`` is not read; without it the files
    are taken from the checkout at ``files_root``.
    """

    commit_id: str
    recipe: Recipe
    files_root: Path
    metric: str
    blobs: BlobStore | None = None


def recipe_identity(profile: SessionProfile) -> tuple[str, RecipeScope]:
    """The canonical id and scope a session's record is filed under; raises on a scope the KB cannot partition on."""

    scope = RecipeScope(profile.kernel_optimizer, profile.tp, profile.conc, profile.isl, profile.osl)
    scope.validate()
    canonical_id = recipe_canonical_id(
        model=profile.model,
        hardware=profile.hardware,
        framework_name=profile.framework,
        model_type=profile.model_type,
        architectures=list(profile.architectures),
        framework_version=profile.framework_version,
        precision=profile.precision,
        scheme=scope.identity_scheme,
    )
    return canonical_id, scope


def knowledge_document(
    snapshot: NotebookSnapshot, profile: SessionProfile, final: FinalRecipe, scope: RecipeScope
) -> dict[str, object]:
    """The record's knowledge: metrics, workload shape, ``value`` columns, experience lists and provenance."""

    value = snapshot.value_of(final.commit_id).get(final.metric)
    baseline = snapshot.baseline.values.get(final.metric) if snapshot.baseline else None
    if value is None or not baseline:
        raise RecipeError(f"{final.commit_id} and the baseline need a recorded {final.metric} to publish")
    lessons = session_lessons(snapshot, final.commit_id)
    return sanitize_shared_knowledge(
        {
            "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
            "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
            KB_METRIC: value,
            GAIN_METRIC: (value / baseline - 1.0) * 100.0,
            "workload_shape": {key: item for key, item in scope.as_dict().items() if key != "kernel_optimizer"},
            "value": to_recipe_kb_value(final.recipe),
            "what_worked": list(lessons.worked),
            "what_failed": list(lessons.failed),
            "remaining_gaps": list(lessons.gaps),
            "lessons": list(lessons.learned),
            "pitfalls": list(lessons.pitfalls),
            "provenance": {
                "producer": PRODUCER,
                "kernel_optimizer": scope.kernel_optimizer,
                "phase": "CLOSE",
                "session_id": profile.session_id,
                "commit_id": final.commit_id,
                "recipe_digest": final.recipe.digest(),
                "base_image_digest": final.recipe.base_image_digest,
                "runtime_snapshot": final.recipe.runtime_snapshot,
                "generated_at": datetime.fromtimestamp(snapshot.taken_at, timezone.utc).isoformat(),
            },
        }
    )


def stage_files(recipe: Recipe, files_root: Path, staging: Path, blobs: BlobStore | None = None) -> list[Artifact]:
    """Stage each patch and kernel file under the ref ``value`` names it by; raises on a file the recipe does not pin.

    The bytes come from ``blobs`` by sha256 when it is given, otherwise from the checkout at ``files_root``.
    """

    def stage(path: str, ref: str, sha256: str) -> Artifact:
        content = blobs.get(sha256) if blobs is not None else _checked_out(files_root / path)
        found = hashlib.sha256(content).hexdigest()
        if found != sha256:
            raise RecipeError(f"{path} has sha256 {found}; the recipe pins {sha256}")
        destination = staging / ref
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return Artifact(path=ref, source=destination, kind=ref.split("/", 1)[0])

    artifacts = [
        stage(patch.path, overlay_ref(stack, member, patch.path), patch.sha256)
        for stack, component in enumerate(recipe.components)
        for member, patch in enumerate(component.patches)
    ]
    artifacts.extend(stage(kernel.ref, kernel.ref, kernel.sha256) for kernel in recipe.kernels)
    return artifacts


def _checked_out(source: Path) -> bytes:
    if not source.is_file():
        raise RecipeError(f"{source} is not in the checkout")
    return source.read_bytes()


def publish_final_recipe(
    client: RemoteRecipeClient, snapshot: NotebookSnapshot, final: FinalRecipe
) -> RemoteWriteResult:
    """Write ``final`` when it beats the champion of its partition; the result says what was written or why not."""

    profile = snapshot.profile
    if profile is None:
        return RemoteWriteResult(SKIPPED, "no_session_profile")
    try:
        canonical_id, scope = recipe_identity(profile)
    except RemoteRecipeValidationError:
        return RemoteWriteResult(SKIPPED, "invalid_recipe_scope", session_id=profile.session_id)
    if scope.identity_scheme == AGENTX:
        return RemoteWriteResult(SKIPPED, "agentx_objective_not_recorded", canonical_id, profile.session_id)
    if not final.recipe.complete:
        return RemoteWriteResult(SKIPPED, "incomplete_recipe", canonical_id, profile.session_id)
    commit = snapshot.lineage.commit(final.commit_id)
    if commit is not None and commit.recipe_digest not in (None, final.recipe.digest()):
        return RemoteWriteResult(SKIPPED, "recipe_digest_mismatch", canonical_id, profile.session_id)
    knowledge = knowledge_document(snapshot, profile, final, scope)
    with tempfile.TemporaryDirectory(prefix="hyperloom-multibranch-recipe-") as temporary:
        staging = Path(temporary)
        bundle = KnowledgeBundle(knowledge, stage_files(final.recipe, final.files_root, staging, final.blobs))
        return client.write_if_better(
            canonical_id,
            profile.session_id,
            bundle,
            scope=scope,
            files_dir=staging,
            primary_metric=KB_METRIC,
            primary_value=float(knowledge[KB_METRIC]),
            objective_schema=OBJECTIVE_SCHEMA,
        )


__all__ = [
    "KB_METRIC",
    "OBJECTIVE_SCHEMA",
    "FinalRecipe",
    "knowledge_document",
    "publish_final_recipe",
    "recipe_identity",
    "stage_files",
]
