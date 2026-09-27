# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Record what KTH would have said, and change nothing.

Shadow mode exists to answer one question before the gate is given authority:
on the patches this repository actually produces, how often would KTH have
refused, and would it have been right? Running the gate for real to find that
out is the wrong order of operations.

Three properties hold here, and each is enforced by construction rather than by
convention, because the failure mode is that shadow mode quietly becomes
enforcement.

1.  **It is enabled separately.** ``HYPERLOOM_KTH_SHADOW_ENABLE`` is its own
    flag. Turning the real gate on does not turn this on and vice versa, and
    enabling both is a configuration error rather than a silent precedence
    rule, so an operator cannot get enforcement while believing they asked to
    observe.

2.  **Its result cannot be enforced on.** :class:`KthShadowObservation`
    deliberately has no ``eligible`` attribute and no boolean the integration
    path could branch on. ``if not observation.eligible`` does not typecheck
    and does not run. The verdict is carried as free text under a name that
    cannot be mistaken for a decision.

3.  **It cannot fail the run.** Every error inside :meth:`observe` becomes a
    recorded observation. A shadow evaluation that crashes reverts nothing,
    blocks nothing and changes no benchmark.

The observation is written to its own ``kth_shadow`` field on the integration
result, never to ``kth``, so an artifact consumer cannot confuse an observation
with a validated qualification either.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hyperloom.common.env import env_bool, env_float, env_str

from .controller_publication import ControllerPatchPublication
from .kth_qualification import (
    KthConfigurationError,
    KthQualificationProvider,
    KthQualificationResult,
)

#: Written into every observation. A reader scanning artifacts for the string
#: "kth" must be able to tell at a glance that this one decided nothing.
SHADOW_LABEL = "OBSERVATION ONLY - this KTH result affected nothing"

#: The field name on the integration result. Kept distinct from ``kth`` so the
#: two can never be read as the same kind of record.
SHADOW_FIELD = "kth_shadow"


@dataclass(frozen=True)
class KthShadowObservation:
    """What KTH said about a patch that was integrated regardless.

    There is intentionally no ``eligible`` property here, and no other boolean
    summarising the outcome. The absence is the safety property: the
    enforcement path in ``controller_patch_integration`` reads
    ``qualified.eligible``, and this type does not have one, so an observation
    cannot be substituted for a qualification without the substitution failing
    loudly at the point it is written.
    """

    #: Always ``SHADOW_LABEL``. First key a human sees in the artifact.
    label: str = SHADOW_LABEL
    #: Always ``True``. A machine consumer checks this before reading anything else.
    recorded_only: bool = True
    #: KTH's status string, verbatim. Free text on purpose.
    observed_status: str = ""
    #: KTH's verdict string, verbatim.
    observed_verdict: str = ""
    observed_reason: str = ""
    kth_revision: str = ""
    plan_id: str = ""
    subject_digest: str = ""
    artifacts_dir: str = ""
    #: What the patch's fate actually was, so the two can be compared later
    #: without rejoining against another artifact.
    integration_outcome: str = ""
    #: Set when the shadow evaluation itself failed. An empty string means the
    #: evaluation ran; a non-empty one means nothing was observed and, as
    #: everywhere else here, nothing happened as a result.
    shadow_error: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "recorded_only": self.recorded_only,
            "observed_status": self.observed_status,
            "observed_verdict": self.observed_verdict,
            "observed_reason": self.observed_reason,
            "kth_revision": self.kth_revision,
            "plan_id": self.plan_id,
            "subject_digest": self.subject_digest,
            "artifacts_dir": self.artifacts_dir,
            "integration_outcome": self.integration_outcome,
            "shadow_error": self.shadow_error,
            "seconds": self.seconds,
        }


def _from_result(result: KthQualificationResult, *, seconds: float) -> KthShadowObservation:
    """Narrow a qualification into an observation, dropping every decision.

    ``eligible`` and ``performance_admitted`` are not carried across. What a
    shadow run may report is what KTH said, not what would have followed.
    """
    return KthShadowObservation(
        observed_status=str(result.status),
        observed_verdict=str(result.verdict),
        observed_reason=str(result.reason),
        kth_revision=str(result.kth_revision),
        plan_id=str(result.plan_id),
        subject_digest=str(result.subject_digest),
        artifacts_dir=str(result.artifacts_dir),
        seconds=round(seconds, 4),
    )


@dataclass(frozen=True)
class KthShadowObserver:
    """Runs a qualification for the record and returns something inert."""

    provider: KthQualificationProvider
    #: Wall-clock ceiling for one observation. Shorter than the enforcing
    #: gate's by default: an observation that nobody acts on should not be the
    #: reason an integration run takes twice as long.
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls) -> "KthShadowObserver | None":
        """The configured observer, or ``None`` when shadow mode is off."""
        if not env_bool("HYPERLOOM_KTH_SHADOW_ENABLE"):
            return None
        if env_bool("HYPERLOOM_KTH_ENABLE"):
            # Refusing is the point. Silently preferring one would leave an
            # operator who asked to observe running an enforcing gate, or an
            # operator who asked to enforce quietly not enforcing.
            raise KthConfigurationError(
                "HYPERLOOM_KTH_ENABLE and HYPERLOOM_KTH_SHADOW_ENABLE are both set. "
                "Shadow mode records what the gate would have said and enforces nothing; "
                "the enforcing gate already records its own result. Choose one."
            )
        timeout_s = env_float("HYPERLOOM_KTH_SHADOW_TIMEOUT_S", 120.0)
        if not timeout_s > 0:
            raise KthConfigurationError("HYPERLOOM_KTH_SHADOW_TIMEOUT_S must be positive")
        executable = env_str("HYPERLOOM_KTH_QUALIFY_EXECUTABLE") or "kth-qualify"
        return cls(
            provider=KthQualificationProvider(
                executable=executable,
                timeout_s=timeout_s,
                expected_revision=env_str("HYPERLOOM_KTH_EXPECTED_SHA").lower(),
                reviewed_plans={},
            ),
            timeout_s=timeout_s,
        )

    def observe(
        self,
        publication: ControllerPatchPublication,
        *,
        base_commit: str,
        patch: bytes,
        changed_paths: Sequence[str],
        artifacts_root: Path,
        integration_outcome: str = "",
        session_id: str = "",
    ) -> KthShadowObservation:
        """Qualify for the record. Never raises, never decides anything."""
        started = time.perf_counter()
        try:
            artifacts_root.mkdir(parents=True, exist_ok=True)
            result = self.provider.qualify(
                publication,
                base_commit=base_commit,
                patch=patch,
                changed_paths=tuple(changed_paths),
                artifacts_root=artifacts_root,
                session_id=session_id,
            )
        except BaseException as error:  # noqa: BLE001
            # Deliberately broad. A shadow observation is not worth failing an
            # integration run for, and narrowing this would eventually let
            # something through that is.
            return KthShadowObservation(
                integration_outcome=integration_outcome,
                shadow_error=f"{type(error).__name__}: {error}",
                seconds=round(time.perf_counter() - started, 4),
            )
        observation = _from_result(result, seconds=time.perf_counter() - started)
        return replace_outcome(observation, integration_outcome)


def replace_outcome(observation: KthShadowObservation, outcome: str) -> KthShadowObservation:
    """The same observation, told what actually happened to the patch."""
    return replace(observation, integration_outcome=outcome)


def shadow_record(observation: KthShadowObservation | None) -> Mapping[str, Any]:
    """The artifact payload, or an empty mapping when shadow mode is off."""
    return observation.to_dict() if observation is not None else {}


__all__ = [
    "SHADOW_FIELD",
    "SHADOW_LABEL",
    "KthShadowObservation",
    "KthShadowObserver",
    "shadow_record",
]
