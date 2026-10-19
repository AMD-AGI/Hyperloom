# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE: the final recipe and what the session learnt, written to the Recipe KB when it beats the champion.

The record carries the columns the single-lineage CLOSE writes: the identity and scope come from the session profile
PRELUDE wrote, ``value`` from :func:`to_recipe_kb_value`, the experience lists from :func:`session_lessons`. The base
image digest and runtime snapshot, which ``value`` leaves out, travel in the provenance. Patch and kernel files are
read from the session's blob store by the hash the recipe pins (the experiment repository holds recipes only), or
copied out of a checkout when no store is given, and staged under the refs ``value`` names; each is checked against
its pinned hash, so the KB stores exactly what was measured.

InferenceX sessions compare ``optimized_throughput`` with the champion of their partition. An AgentX session compares,
as the main flow's AgentX records do, its gain in P50 interactivity over its own baseline (``interactivity_gain_pct``,
objective ``agentx_keep``), and carries both interactivities beside it, with both total throughputs when the gate
recorded them. When the notebook recorded the final commit's recipe digest, a recipe with another digest is skipped
(``recipe_digest_mismatch``): CLOSE was handed some other commit's recipe. A recipe that names anything as unexpressed
does not rebuild what was measured, so it is skipped as well (``incomplete_recipe``).

The provenance names how the warmup of the measurements behind the final commit ran (``warmup_round_seed``): the seed
run_grid noted, or ``unpatched`` when its patch did not apply. Results from before the warmup had a seed of its own
note neither, and the key is left out: their warmup replayed the measured prompts, which reads about 3% high (#56).
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hyperloom.common.perf_metric import GRADED_TOTAL
from hyperloom.contracts.recipe import Recipe, RecipeError
from hyperloom.inference_optimizer.recipe_snapshot_constants import recipe_canonical_id
from hyperloom.measure.compile import BlobStore
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom.orchestrator.knowledge.remote_recipe.models import (
    Artifact,
    KBSelectionProfile,
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
AGENTX_METRIC = "interactivity_gain_pct"
AGENTX_OBJECTIVE_SCHEMA = "agentx_keep"
WARMUP_SEED_NOTE = "warmup_round_seed:"
WARMUP_SEED_UNPATCHED = "run_grid_warmup_seed_unpatched"
PRODUCER = "hyperloom-multibranch"
SKIPPED = "skipped"
AGENTX = "agentx"


@dataclass(frozen=True)
class FinalRecipe:
    """The commit CLOSE publishes, its recipe, and where the patch and kernel files the recipe names are read from.

    ``metric`` is the recorded-value metric the session is graded on: output throughput for InferenceX, P50
    interactivity for AgentX. With ``blobs`` (the session's blob store) every file is read from it by its pinned sha256
    and ``files_root`` is not read; without it the files are taken from the checkout at ``files_root``.
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


def _metrics(snapshot: NotebookSnapshot, final: FinalRecipe, scope: RecipeScope) -> dict[str, float]:
    """The record's metrics, named as the main flow names them for the session's kind of workload."""

    current = snapshot.value_of(final.commit_id)
    anchor = snapshot.baseline.values if snapshot.baseline else {}
    value, baseline = current.get(final.metric), anchor.get(final.metric)
    if value is None or not baseline:
        raise RecipeError(f"{final.commit_id} and the baseline need a recorded {final.metric} to publish")
    gain = (value / baseline - 1.0) * 100.0
    if scope.identity_scheme != AGENTX:
        return {KB_METRIC: value, GAIN_METRIC: gain}
    metrics = {AGENTX_METRIC: gain, "candidate_interactivity": value, "baseline_interactivity": baseline}
    total, baseline_total = current.get(GRADED_TOTAL), anchor.get(GRADED_TOTAL)
    if total and baseline_total:
        metrics.update({GRADED_TOTAL: total, "baseline_total_throughput": baseline_total})
    return metrics


def _warmup_seed(snapshot: NotebookSnapshot, commit_id: str) -> dict[str, str]:
    """The ``warmup_round_seed`` the measurements behind ``commit_id``'s recorded value noted, as provenance."""

    made = next((record for record in snapshot.records if record.commit_id == commit_id), None)
    if made is not None and made.judgement is not None:
        request_ids = made.judgement.request_ids
    else:
        request_ids = snapshot.baseline.request_ids if snapshot.baseline else ()
    results = {record.request_id: record.result for record in snapshot.records if record.result is not None}
    notes = [
        note for request_id in request_ids if request_id in results for note in results[request_id].nonfatal_warnings
    ]
    seeds = sorted({note.removeprefix(WARMUP_SEED_NOTE) for note in notes if note.startswith(WARMUP_SEED_NOTE)})
    if WARMUP_SEED_UNPATCHED in notes:
        return {"warmup_round_seed": "unpatched"}
    return {"warmup_round_seed": ",".join(seeds)} if seeds else {}


def knowledge_document(
    snapshot: NotebookSnapshot, profile: SessionProfile, final: FinalRecipe, scope: RecipeScope
) -> dict[str, object]:
    """The record's knowledge: metrics, workload shape, ``value`` columns, experience lists and provenance."""

    metrics = _metrics(snapshot, final, scope)
    lessons = session_lessons(snapshot, final.commit_id)
    return sanitize_shared_knowledge(
        {
            "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
            "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
            **metrics,
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
                **_warmup_seed(snapshot, final.commit_id),
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
    if not final.recipe.complete:
        return RemoteWriteResult(SKIPPED, "incomplete_recipe", canonical_id, profile.session_id)
    commit = snapshot.lineage.commit(final.commit_id)
    if commit is not None and commit.recipe_digest not in (None, final.recipe.digest()):
        return RemoteWriteResult(SKIPPED, "recipe_digest_mismatch", canonical_id, profile.session_id)
    knowledge = knowledge_document(snapshot, profile, final, scope)
    agentx = scope.identity_scheme == AGENTX
    metric, objective = (AGENTX_METRIC, AGENTX_OBJECTIVE_SCHEMA) if agentx else (KB_METRIC, OBJECTIVE_SCHEMA)
    with tempfile.TemporaryDirectory(prefix="hyperloom-multibranch-recipe-") as temporary:
        staging = Path(temporary)
        bundle = KnowledgeBundle(knowledge, stage_files(final.recipe, final.files_root, staging, final.blobs))
        value = float(knowledge[metric])
        selection = KBSelectionProfile(scope, metric, value, objective, {metric: value})
        return client.write_if_better(canonical_id, profile.session_id, bundle, selection, files_dir=staging)


__all__ = [
    "AGENTX_METRIC",
    "AGENTX_OBJECTIVE_SCHEMA",
    "KB_METRIC",
    "OBJECTIVE_SCHEMA",
    "FinalRecipe",
    "knowledge_document",
    "publish_final_recipe",
    "recipe_identity",
    "stage_files",
]
