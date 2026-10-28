# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance of a test request: the checks of design 4.17 that refuse a request before it queues.

Steps: the request validates (3.7, done by its constructor, plus the toolbox's own: no preset ``request_id``, a
``retest_of`` that names an accepted request, a PROFILE that no live profile already covers); the parent recipe is read from the experiment repository; the delta
applies, changes an explore's parent recipe, and the environment compiles; the measurement surface matches the canonical one, and a test that carries the accuracy
evaluation serves a context that evaluation can run in. Compile, points, surface and the evaluation's context come from
hosted measurement (``hyperloom.measure``) through the one :class:`MeasurementPlanner` seam.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from typing import Protocol

from hyperloom.common.outside_facts import TUNABLEOP_ENABLED_ENV, TUNABLEOP_TABLE_ENV, TUNABLEOP_TUNING_ENV
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import Recipe, RecipeDelta, RecipeError, apply, diff
from hyperloom.common.outside_facts import AITER_META_WHEEL_DIR
from hyperloom.measure.compile import BaseImageMismatchError, BlobStore, CompiledEnvironment
from hyperloom.measure.eval_context import EvalContextError
from hyperloom.measure.jit.key import build_surface
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.fake_gain import (
    CONTEXT_COPY_REFUSAL,
    SPECULATIVE_DECODING_PINNED,
    launch_context_copy_refusal,
    pinned_speculative_refusal,
)
from hyperloom.measure.surface import SurfaceViolation
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected, WorkloadKind
from hyperloom_multibranch.toolbox.long_context import LongContextSetup
from hyperloom_multibranch.toolbox.long_context import decide as decide_long_context
from hyperloom_multibranch.toolbox.short_prompts import (
    ShortPromptSetup,
    blob_patch_files,
    decide,
    no_patch_files,
)


#: PyTorch tunes a GEMM shape the first time a server runs it when TunableOp is switched on (the value ``1``, nothing else
#: enables it) and tuning is not switched off (the value ``0``, nothing else disables it); the variables are named in
#: ``hyperloom.common.outside_facts``.
ONLINE_TUNING_CODE = "online_tuning_unsupported"
CONTEXT_BELOW_EVAL_CODE = "context_below_eval"
NO_EFFECTIVE_CHANGE_CODE = "no_effective_change"
DELTA_NOT_APPLICABLE_CODE = "delta_not_applicable"
#: The refusal of a candidate that patches aiter's ``csrc`` or ``hsa`` on a toolbox whose tests cannot build the modules
#: the patch touches; the reason the toolbox gives (``AcceptanceSetup.aiter_builds_missing``) follows it.
AITER_SOURCES_REASON = "aiter csrc/hsa patches need native builds; this toolbox has none configured"
ONLINE_TUNING_REASON = (
    f"{TUNABLEOP_ENABLED_ENV}=1 without {TUNABLEOP_TUNING_ENV}=0 tunes every GEMM shape the first time the server runs it: at "
    "boot, in the warmup, and for each new prompt length of the evaluation. The test then takes a time no candidate "
    "controls and ends at the attempt timeout, and the tuned results file is lost with the server. For tuned GEMMs "
    "put an offline-tuned table with put_kernel_artifact (kind gemm): an aiter table (env such as "
    f"AITER_CONFIG_GEMM_BF16), or a TunableOp table (env {TUNABLEOP_TABLE_ENV}, with {TUNABLEOP_ENABLED_ENV}=1 and "
    f"{TUNABLEOP_TUNING_ENV}=0 in the launch envs; the file is placed for every device ordinal TunableOp reads)"
)

# Every purpose whose result the InferenceX rule gate reads an eval score from: explore and confirm are compared,
# integration and final are checked (accuracy_passes), baseline sets the accuracy bar (write_inferencex_baseline).
_ACCURACY_PURPOSES = (Purpose.BASELINE, Purpose.EXPLORE, Purpose.CONFIRM, Purpose.INTEGRATION, Purpose.FINAL)


def backend_kind(kind: TestKind) -> str:
    """The kind a measurement backend is called with: a sweep is a test over several points."""
    return TestKind.TEST.value if kind is TestKind.SWEEP else kind.value


def check_kind(request: TestRequest, supported: Collection[str] | None) -> None:
    """Refuse a request whose kind the backend does not run, before anything is queued; ``None`` accepts every kind."""
    if supported is not None and backend_kind(request.kind) not in supported:
        raise RequestRejected(
            f"this toolbox's backend runs kinds {sorted(supported)}, not {request.kind.value!r}", "unsupported_kind"
        )


def _check_unmeasured(request: TestRequest) -> None:
    """A profile or repro test is not a measurement: it evaluates no accuracy (the contract refuses several
    concurrencies, and a repro test with no attachment)."""
    if request.kind in (TestKind.PROFILE, TestKind.REPRO) and request.accuracy:
        raise RequestRejected(f"a {request.kind.value} test does not evaluate accuracy")


def _check_explore_changes_something(request: TestRequest) -> None:
    """An explore measures a change; with an empty delta it would re-measure the parent and record the intent as tried."""
    if request.purpose is Purpose.EXPLORE and request.delta.fingerprint() == RecipeDelta().fingerprint():
        raise RequestRejected("an explore test must change something, and its delta is empty", "empty_delta")


def _restated_settings(delta: RecipeDelta) -> str:
    """The flags, environment variables, patches and kernels ``delta`` sets, in one line."""
    config = delta.config
    named = [
        *([f"flags {config.server_args!r}"] if config.server_args else []),
        *(f"env {name}={value}" for name, value in sorted(config.envs.items())),
        *(f"patch {patch.path}" for change in delta.components for patch in change.add_patches),
        *(f"kernel for {artifact.target}" for artifact in delta.add_kernels),
    ]
    return ", ".join(named) or "the settings it names"


def _check_explore_changes_the_parent(parent: Recipe, candidate: Recipe, request: TestRequest) -> None:
    """An explore whose delta applies to no effect would re-measure the parent: every setting it names is there already.

    The comparison is ``hyperloom.contracts.recipe.diff``, the one ``notebook.changes.same_recipe`` builds on (that
    module sits above the toolbox in the import layers). Confirm, integration, FINAL and baseline tests re-run a recipe
    on purpose and pass.
    """
    if request.purpose is Purpose.EXPLORE and diff(parent, candidate).is_empty:
        raise RequestRejected(
            f"the candidate recipe equals parent commit {request.parent_commit}'s: it already has "
            f"{_restated_settings(request.delta)}; change something the parent does not have",
            NO_EFFECTIVE_CHANGE_CODE,
        )


def _check_online_tuning(candidate: Recipe) -> None:
    """Refuse a candidate that tunes GEMMs while its test runs: its length follows the shapes the test happens to meet."""
    envs = candidate.launch.envs
    if envs.get(TUNABLEOP_ENABLED_ENV) == "1" and envs.get(TUNABLEOP_TUNING_ENV) != "0":
        raise RequestRejected(ONLINE_TUNING_REASON, ONLINE_TUNING_CODE)


def _check_aiter_sources(accepted: object, missing: str | None) -> None:
    """Refuse a candidate that patches aiter's ``csrc`` or ``hsa`` when this toolbox cannot build the modules it touches
    (``missing`` says why), unless its launch rebuilds every module itself (``AITER_REBUILD``): each attempt of its
    test would be refused by its runner, and the image's prebuilt modules are not the patch."""
    environment = getattr(accepted, "environment", None)
    if missing is None or not isinstance(environment, CompiledEnvironment):
        return
    paths = [
        entry.path for root in environment.roots if root.import_name == AITER_META_WHEEL_DIR for entry in root.entries
    ]
    if paths and not build_surface((), environment.recipe.launch.envs, {}).rebuild:
        raise RequestRejected(f"{paths[0]}: {AITER_SOURCES_REASON}: {missing}", DELTA_NOT_APPLICABLE_CODE)


def _check_upstream_label(parent: Recipe, request: TestRequest) -> None:
    """Refuse a delta that moves the upstream label of a component the parent already has.

    Every component is built from the installed package of the session's base image, so a different ``upstream_ref``
    or ``upstream_repo`` would label a measurement with source it never ran. A value equal to the parent's moves
    nothing and passes; a component the parent lacks is a new one, which ``apply`` requires both fields for.
    """
    for change in request.delta.components:
        pinned = parent.component(change.component)
        if pinned is not None and (
            change.upstream_ref not in (None, pinned.upstream_ref)
            or change.upstream_repo not in (None, pinned.upstream_repo)
        ):
            raise RequestRejected(
                f"{change.component} is built from the session's installed package; only patches, launch config and "
                "kernels may change, so drop upstream_ref and upstream_repo from its change",
                "upstream_label_unsupported",
            )


def _no_live_profile(request: TestRequest, recipe_digest: str) -> str | None:
    return None


class MeasurementPlanner(Protocol):
    """Compiles the test environment, generates the measurement points and checks the surface (acceptance steps 2 to 4).

    Raises ``RecipeError`` when it cannot: a ``SurfaceViolation`` means the measurement surface differs, a
    ``BaseImageMismatchError`` a recipe on another image, an ``EvalContextError`` a launch whose context the accuracy
    evaluation cannot run in, any other error a delta or patch that does not build.
    The returned object is handed to the executor with every attempt.
    """

    @property
    def blobs(self) -> BlobStore | None:
        """The blob store the planner compiles from, which the short-prompt check reads the files a recipe's patch
        changes from; ``None`` for a planner that compiles from none, whose every patch then counts as touching."""
        ...

    def plan(self, request: TestRequest, parent: Recipe) -> object: ...


@dataclass(frozen=True)
class AcceptanceSetup:
    """What acceptance judges a request against: the repository, the planner and the active image.

    ``workload_kind`` is the campaign's workload (default InferenceX, the MB1 scenario, design 7.3): in an InferenceX
    campaign every ``explore`` and ``confirm`` request is accepted with ``accuracy=True``; AgentX leaves requests as
    submitted. A recipe on an image other than ``active_image_digest`` is refused; ``None`` skips the check.
    ``supported_kinds`` are the kinds the backend runs (a sweep counts as ``test``); any other kind is refused at
    acceptance as ``unsupported_kind``, and ``None`` accepts every kind. ``short_prompts`` is the staged prompt set of the
    short-prompt consistency check: with it, a test whose change touches attention, the KV cache or prefill (the rule
    pack says which), whose purpose the pack always checks, or whose request forces it, runs the check; ``None`` runs none.
    ``long_context`` is the staged prompt set of the long-context check, which runs under the short-prompt check's lever
    list and always purposes, or when a request sets ``long_context_check``; ``None`` runs none.

    ``explore_accuracy`` off leaves the accuracy evaluation (GSM8K) out of a plain ``explore`` test (one that does not
    force the consistency check), whose accepted request then records ``accuracy=False``: the check was skipped, and the
    result carries no score that could read as a pass. Confirm, integration, FINAL and baseline tests, and an explore
    that forces the consistency check (a lead being adopted), still run it. It is on by default because the rule gate
    still reads an eval score from every explore (``rulegate.readings.eval_score``); once the gate judges an explore
    without one, the default turns off and this field goes.

    ``aiter_builds_missing`` says why this toolbox's tests cannot build the aiter modules a patch to aiter's ``csrc`` or
    ``hsa`` touches (its runners would refuse every attempt of such a test); with it such a candidate is refused at
    acceptance, naming the patched path. ``None`` (the default) accepts it.
    """

    repo: ExperimentRepo
    planner: MeasurementPlanner
    active_image_digest: str | None
    workload_kind: WorkloadKind = WorkloadKind.INFERENCEX
    supported_kinds: Collection[str] | None = None
    short_prompts: ShortPromptSetup | None = None
    long_context: LongContextSetup | None = None
    explore_accuracy: bool = True
    aiter_builds_missing: str | None = None


@dataclass(frozen=True)
class Prepared:
    """What acceptance learned: the compiled environment, handed to the executor with every attempt, and the digest of
    the recipe the request measures (the parent's with the delta applied; it names the image too)."""

    environment: object
    recipe_digest: str = ""


class Acceptance:
    def __init__(
        self,
        setup: AcceptanceSetup,
        *,
        is_accepted: Callable[[str], bool],
        live_profile: Callable[[TestRequest, str], str | None] = _no_live_profile,
    ) -> None:
        """``live_profile`` answers, for a PROFILE request and the digest of the recipe it measures, the id of a profile
        of the same recipe that is queued, running or measured, or ``None``; the toolbox reads it from its submit log."""
        self._repo = setup.repo
        self._planner = setup.planner
        self._image = setup.active_image_digest
        self._is_accepted = is_accepted
        self._live_profile = live_profile
        self._workload = setup.workload_kind
        self._short_prompts = setup.short_prompts
        blobs = setup.planner.blobs
        self._patch_files = no_patch_files if blobs is None else blob_patch_files(blobs)
        self._long_context = setup.long_context
        self._explore_accuracy = setup.explore_accuracy
        self._aiter_builds_missing = setup.aiter_builds_missing

    def with_required_accuracy(self, request: TestRequest) -> TestRequest:
        """The request an InferenceX campaign runs: ``baseline``, ``explore``, ``confirm``, ``integration`` and
        ``final`` tests always carry accuracy, except a plain explore when ``explore_accuracy`` is off, which never does.

        The rule gate refuses to judge an InferenceX result that has no score, so acceptance sets the flag (the
        accepted request records it) rather than refusing. AgentX campaigns and every other purpose are unchanged.
        """

        if self._workload is not WorkloadKind.INFERENCEX or request.purpose not in _ACCURACY_PURPOSES:
            return request
        plain_explore = request.purpose is Purpose.EXPLORE and not request.forces_consistency_check
        wanted = not (plain_explore and not self._explore_accuracy and request.kind is not TestKind.ACCURACY)
        return request if request.accuracy == wanted else replace(request, accuracy=wanted)

    async def prepare(self, request: TestRequest, *, replay: bool = False) -> Prepared:
        """Run the checks; ``replay`` re-prepares an already accepted request after a restart (it has its id)."""

        if not replay and request.request_id:
            raise RequestRejected("request_id is assigned by the toolbox; submit the request without one")
        _check_unmeasured(request)
        _check_explore_changes_something(request)
        if not replay and request.retest_of is not None and not self._is_accepted(request.retest_of):
            raise RequestRejected(f"retest_of {request.retest_of!r} names no accepted request", "unknown_retest")
        recipe = await self._read_parent(request)
        self._check_image(recipe)
        if not replay:  # a request an earlier toolbox accepted is not withdrawn by a later rule
            _check_upstream_label(recipe, request)
        candidate = self._check_delta(recipe, request)
        digest = candidate.digest()
        if not replay:
            self._check_profile_exists(request, digest)
        self._check_fake_gain(recipe, candidate)
        if not replay:
            _check_explore_changes_the_parent(recipe, candidate, request)
            _check_online_tuning(candidate)
        planned = self._plan(recipe, request)
        if not replay:
            _check_aiter_sources(planned, self._aiter_builds_missing)
        accepted = self._with_short_prompts(planned, recipe, candidate, request)
        return Prepared(self._with_long_context(accepted, recipe, candidate, request), digest)

    def _with_short_prompts(self, accepted: object, parent: Recipe, candidate: Recipe, request: TestRequest) -> object:
        """``accepted`` with the consistency check this request runs, when it runs one and ``accepted`` can carry it: on the
        test's own server after its rounds, and on no other."""
        plan = decide(request, parent, candidate, self._short_prompts, self._patch_files)
        return (
            replace(accepted, short_prompts=plan)
            if plan is not None and isinstance(accepted, AcceptedMeasurement)
            else accepted
        )

    def _with_long_context(self, accepted: object, parent: Recipe, candidate: Recipe, request: TestRequest) -> object:
        """``accepted`` with the long-context check this request runs, when it runs one and ``accepted`` can carry it."""
        plan = decide_long_context(request, parent, candidate, self._long_context, self._patch_files)
        return (
            replace(accepted, long_context=plan)
            if plan is not None and isinstance(accepted, AcceptedMeasurement)
            else accepted
        )

    def _check_profile_exists(self, request: TestRequest, recipe_digest: str) -> None:
        """Refuse a PROFILE of a recipe that already has a live one, and name that one.

        A profile is not a claimed test, so nothing else keeps branches from profiling one recipe over and over; each
        repeat holds a card for a trace the session already has or is about to have. The point is always the
        canonical one (a profile ignores ``concurrency``), so the recipe the request measures, whose digest names its
        image too, says whether it is the same measurement, whatever commit holds it: a merge commit whose recipe is
        its source's is the same profile (two such pairs were profiled twice in one session). The caller waits for the
        existing request instead.
        """

        if request.kind is not TestKind.PROFILE:
            return
        existing = self._live_profile(request, recipe_digest)
        if existing is not None:
            raise RequestRejected(
                f"the recipe of commit {request.parent_commit} already has a profile, request {existing}, queued, "
                "running or measured; wait_test on it, or read the commit's op shares from the notebook",
                "profile_exists",
                existing_request_id=existing,
            )

    async def _read_parent(self, request: TestRequest) -> Recipe:
        try:
            return await self._repo.read_recipe(request.parent_commit)
        except KeyError:
            raise RequestRejected(
                f"parent commit {request.parent_commit!r} is not in the experiment repository", "unknown_parent"
            ) from None

    def _check_image(self, recipe: Recipe) -> None:
        """Refuse a recipe on another base image. Only the base image is compared, as the compile does
        (:func:`~hyperloom.measure.compile.check_base_image`): a ``runtime_snapshot`` names no image to compare, and the
        compile refuses one beside native stacks (``RuntimeSourcesConflict``, ``delta_not_applicable``)."""
        if self._image is not None and recipe.base_image_digest != self._image:
            raise RequestRejected(
                f"recipe base image {recipe.base_image_digest} is not the active framework image {self._image}",
                "base_image_mismatch",
            )

    @staticmethod
    def _check_delta(recipe: Recipe, request: TestRequest) -> Recipe:
        try:
            return apply(recipe, request.delta)
        except RecipeError as error:
            raise RequestRejected(
                f"the delta does not fit the parent recipe: {error}", DELTA_NOT_APPLICABLE_CODE
            ) from error

    def _check_fake_gain(self, parent: Recipe, candidate: Recipe) -> None:
        """Refuse a candidate whose gain main refuses as fake: its launch is compared with its parent's.

        InferenceX runs a synthetic benchmark, where a context-copy draft is refused (#1904); AgentX runs real
        datasets, where a draft whose acceptance length the parent simulates stays pinned (#1885).
        """
        if self._workload is WorkloadKind.INFERENCEX:
            reason, code = launch_context_copy_refusal(parent.launch, candidate.launch), CONTEXT_COPY_REFUSAL
        else:
            reason, code = pinned_speculative_refusal(parent.launch, candidate.launch), SPECULATIVE_DECODING_PINNED
        if reason:
            raise RequestRejected(reason, code)

    def _plan(self, recipe: Recipe, request: TestRequest) -> object:
        try:
            return self._planner.plan(request, recipe)
        except SurfaceViolation as error:
            raise RequestRejected(
                f"measurement surface differs from the canonical one: {error}", "surface_mismatch"
            ) from error
        except BaseImageMismatchError as error:
            raise RequestRejected(str(error), "base_image_mismatch") from error
        except EvalContextError as error:
            raise RequestRejected(str(error), CONTEXT_BELOW_EVAL_CODE) from error
        except RecipeError as error:
            raise RequestRejected(f"the environment does not compile: {error}", DELTA_NOT_APPLICABLE_CODE) from error
