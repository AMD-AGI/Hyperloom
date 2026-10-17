# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance of a test request: the checks of design 4.17 that refuse a request before it queues.

Steps: the request validates (3.7, done by its constructor, plus the toolbox's own: no preset ``request_id``, a
``retest_of`` that names an accepted request); the parent recipe is read from the experiment repository; the delta
applies and the environment compiles; the measurement surface matches the canonical one. Compile, points and surface come from
hosted measurement (``hyperloom.measure``) through the one :class:`MeasurementPlanner` seam.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from typing import Protocol

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import Recipe, RecipeError, apply
from hyperloom.measure.compile import BaseImageMismatchError
from hyperloom.measure.fake_gain import (
    CONTEXT_COPY_REFUSAL,
    SPECULATIVE_DECODING_PINNED,
    launch_context_copy_refusal,
    pinned_speculative_refusal,
)
from hyperloom.measure.surface import SurfaceViolation
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected, WorkloadKind


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


def _check_profile(request: TestRequest) -> None:
    """A profile test is one profiled round: it evaluates no accuracy (several concurrencies the contract refuses)."""
    if request.kind is TestKind.PROFILE and request.accuracy:
        raise RequestRejected("a profile test does not evaluate accuracy")


class MeasurementPlanner(Protocol):
    """Compiles the test environment, generates the measurement points and checks the surface (acceptance steps 2 to 4).

    Raises ``RecipeError`` when it cannot: a ``SurfaceViolation`` means the measurement surface differs, a
    ``BaseImageMismatchError`` a recipe on another image, any other error a delta or patch that does not build.
    The returned object is handed to the executor with every attempt.
    """

    def plan(self, request: TestRequest, parent: Recipe) -> object: ...


@dataclass(frozen=True)
class Prepared:
    """What acceptance learned: the compiled environment, handed to the executor with every attempt."""

    environment: object


class Acceptance:
    def __init__(
        self,
        repo: ExperimentRepo,
        planner: MeasurementPlanner,
        *,
        active_image_digest: str | None,
        is_accepted: Callable[[str], bool],
        workload_kind: WorkloadKind = WorkloadKind.INFERENCEX,
    ) -> None:
        self._repo = repo
        self._planner = planner
        self._image = active_image_digest
        self._is_accepted = is_accepted
        self._workload = workload_kind

    def with_required_accuracy(self, request: TestRequest) -> TestRequest:
        """The request an InferenceX campaign runs: ``baseline``, ``explore``, ``confirm``, ``integration`` and
        ``final`` tests always carry accuracy.

        The rule gate refuses to judge an InferenceX result that has no score, so acceptance sets the flag (the
        accepted request records it) rather than refusing. AgentX campaigns and every other purpose are unchanged.
        """

        if self._workload is WorkloadKind.INFERENCEX and request.purpose in _ACCURACY_PURPOSES and not request.accuracy:
            return replace(request, accuracy=True)
        return request

    async def prepare(self, request: TestRequest, *, replay: bool = False) -> Prepared:
        """Run the checks; ``replay`` re-prepares an already accepted request after a restart (it has its id)."""

        if not replay and request.request_id:
            raise RequestRejected("request_id is assigned by the toolbox; submit the request without one")
        _check_profile(request)
        if not replay and request.retest_of is not None and not self._is_accepted(request.retest_of):
            raise RequestRejected(f"retest_of {request.retest_of!r} names no accepted request", "unknown_retest")
        recipe = await self._read_parent(request)
        self._check_image(recipe)
        candidate = self._check_delta(recipe, request)
        self._check_fake_gain(recipe, candidate)
        return Prepared(self._plan(recipe, request))

    async def _read_parent(self, request: TestRequest) -> Recipe:
        try:
            return await self._repo.read_recipe(request.parent_commit)
        except KeyError:
            raise RequestRejected(
                f"parent commit {request.parent_commit!r} is not in the experiment repository", "unknown_parent"
            ) from None

    def _check_image(self, recipe: Recipe) -> None:
        if self._image is None:
            return
        digests = {recipe.base_image_digest, recipe.runtime_snapshot or recipe.base_image_digest}
        if digests != {self._image}:
            raise RequestRejected(
                f"recipe image {sorted(digests)} is not the active framework image {self._image}", "base_image_mismatch"
            )

    @staticmethod
    def _check_delta(recipe: Recipe, request: TestRequest) -> Recipe:
        try:
            return apply(recipe, request.delta)
        except RecipeError as error:
            raise RequestRejected(
                f"the delta does not fit the parent recipe: {error}", "delta_not_applicable"
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
        except RecipeError as error:
            raise RequestRejected(f"the environment does not compile: {error}", "delta_not_applicable") from error
