# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed Kernel Trust Harness (KTH) qualification of an applied candidate.

KTH is an independent subprocess: Hyperloom sends the exact applied candidate,
KTH decides ``Eligible for performance evaluation``, ``Blocked`` or
``Inconclusive`` and returns an attestation, and Hyperloom only checks that the
attestation is complete and bound to the candidate it sent. Anything short of a
validated Eligible attestation keeps the candidate away from the performance
validator.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from hyperloom.common.env import env_bool, env_float, env_str
from hyperloom.common.env_safety import redact_secret_values, scrub_benchmark_process_env
from hyperloom.common.io import atomic_write_json, atomic_write_text

from .controller_publication import ControllerPatchPublication
from .kth_contract import (
    ADAPTIVE_BINDING_KEYS,
    ADAPTIVE_REQUEST_SCHEMA,
    REVIEWED_REQUEST_SCHEMA,
    REVISION,
    SAFE_ID,
    VERDICT_BLOCKED,
    VERDICT_ELIGIBLE,
    VERDICT_EXIT_CODES,
    VERDICT_INCONCLUSIVE,
    CandidateArtifact,
    adaptive_subject_digest,
    build_candidate_envelope,
    candidate_control_fields,
    envelope_digest,
    environment_identity,
    reviewed_subject_digest,
    sha256_digest,
)

KthStatus = Literal["eligible", "blocked", "inconclusive", "failed"]

_VERDICT_STATUS: dict[str, KthStatus] = {
    VERDICT_ELIGIBLE: "eligible",
    VERDICT_BLOCKED: "blocked",
    VERDICT_INCONCLUSIVE: "inconclusive",
}
_REVIEWED_REQUIRED = frozenset(
    {
        "schema_version",
        "request_id",
        "subject_digest",
        "kth_sha",
        "qualification_plan",
        "candidate_identity",
        "execution_binding",
        "execution_mode",
        "hardware_identity",
        "verdict",
        "primary_detector",
        "mandatory_oracle_coverage",
        "findings",
        "unexplored_regions",
        "duration_s",
        "replay",
        "repair_feedback",
    }
)
_ADAPTIVE_REQUIRED = frozenset(
    {
        "schema_version",
        "request_id",
        "subject_digest",
        "kth_revision",
        "envelope_digest",
        "autospec",
        "resolved_spec",
        "evidence_plan",
        "binding",
        "verdict",
        "findings",
    }
)
#: Scalar Controller manifest entries forwarded to KTH as exploratory GEAK observations, never as conclusions.
_GEAK_OBSERVATION_KEYS = ("mean_case_speedup", "best_wall_ms", "iteration", "correctness_passed")
_EXECUTION_MODES = frozenset({"real", "simulated"})
#: Captured subprocess output is diagnostics, not a transcript; a runaway child cannot fill the session.
_LOG_LIMIT_BYTES = 1 << 20


class KthConfigurationError(ValueError):
    """``HYPERLOOM_KTH_*`` enables the gate but does not describe a usable provider."""


class _Rejected(Exception):
    """KTH gave no answer that can qualify the candidate that was sent."""


@dataclass(frozen=True)
class KthQualificationResult:
    """What KTH decided about one candidate, as Hyperloom validated it."""

    status: KthStatus
    reason: str
    request_id: str = ""
    request_schema: str = ""
    plan_id: str = ""
    verdict: str = ""
    subject_digest: str = ""
    patch_digest: str = ""
    kth_revision: str = ""
    #: KTH's own label for how the candidate ran (``real`` or ``simulated``); reviewed-plan attestations only.
    execution_mode: str = ""
    primary_detector: str = ""
    artifacts_dir: str = ""
    repair_feedback: dict[str, Any] = field(default_factory=dict)
    performance_admitted: bool = False

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"


def _reviewed_plans(raw: str) -> dict[str, str]:
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise KthConfigurationError(f"HYPERLOOM_KTH_PLANS is not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise KthConfigurationError("HYPERLOOM_KTH_PLANS must map kernel paths to plan IDs")
    for kernel_path, plan_id in payload.items():
        if not isinstance(plan_id, str) or not SAFE_ID.fullmatch(plan_id):
            raise KthConfigurationError(f"HYPERLOOM_KTH_PLANS[{kernel_path!r}] is not a valid plan ID")
    return dict(payload)


@dataclass(frozen=True)
class KthQualificationProvider:
    """A host-configured ``kth-qualify`` executable and the plans the host has reviewed."""

    executable: str = "kth-qualify"
    timeout_s: float = 300.0
    expected_revision: str = ""
    #: Repo-relative kernel path to the host-owned KTH plan ID that covers it. Unmapped kernels are sent as an
    #: adaptive candidate envelope; a publication never selects its own plan.
    reviewed_plans: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> KthQualificationProvider | None:
        """The configured provider, or ``None`` when ``HYPERLOOM_KTH_ENABLE`` is off."""
        if not env_bool("HYPERLOOM_KTH_ENABLE"):
            return None
        timeout_s = env_float("HYPERLOOM_KTH_TIMEOUT_S", 300.0)
        if not timeout_s > 0:
            raise KthConfigurationError("HYPERLOOM_KTH_TIMEOUT_S must be positive")
        expected = env_str("HYPERLOOM_KTH_EXPECTED_SHA").lower()
        if expected and not REVISION.fullmatch(expected):
            raise KthConfigurationError("HYPERLOOM_KTH_EXPECTED_SHA must be a full hexadecimal revision")
        return cls(
            executable=env_str("HYPERLOOM_KTH_QUALIFY_EXECUTABLE") or "kth-qualify",
            timeout_s=timeout_s,
            expected_revision=expected,
            reviewed_plans=_reviewed_plans(env_str("HYPERLOOM_KTH_PLANS")),
        )

    def qualify(
        self,
        publication: ControllerPatchPublication,
        *,
        base_commit: str,
        patch: bytes,
        changed_paths: tuple[str, ...],
        artifacts_root: Path,
        session_id: str = "",
    ) -> KthQualificationResult:
        """Qualify ``patch`` as applied on ``base_commit``; every failure is a non-eligible result."""
        request_id = f"hyperloom-{uuid.uuid4().hex}"
        artifacts_dir = artifacts_root / request_id
        artifacts_dir.mkdir(parents=True)
        plan_id = self.reviewed_plans.get(publication.kernel_path, "")
        artifact = CandidateArtifact(
            candidate_id=_candidate_id(publication.operator_id),
            attempt_id=request_id,
            base_repository=str(publication.repo_root),
            base_commit=base_commit,
            patch=patch,
            changed_paths=changed_paths,
        )
        base = KthQualificationResult(
            status="failed",
            reason="",
            request_id=request_id,
            request_schema=REVIEWED_REQUEST_SCHEMA if plan_id else ADAPTIVE_REQUEST_SCHEMA,
            plan_id=plan_id,
            patch_digest=artifact.patch_digest,
            artifacts_dir=str(artifacts_dir),
        )
        steering = _publication_steering(publication)
        if steering:
            return self._persist(
                replace(base, reason=f"publication tries to steer KTH qualification through {', '.join(steering)}")
            )
        if not changed_paths:
            return self._persist(replace(base, reason="KTH cannot qualify an empty patch scope"))

        if plan_id:
            request = _reviewed_request(request_id, plan_id, artifact, publication.kernel_path)
        else:
            envelope = _envelope(publication, artifact, session_id=session_id)
            atomic_write_json(artifacts_dir / "envelope.json", envelope, trailing_newline=True)
            request = {"schema_version": ADAPTIVE_REQUEST_SCHEMA, "request_id": request_id, "envelope": envelope}
        try:
            exit_code, attestation = self._run(request, artifacts_dir)
        except _Rejected as failure:
            return self._persist(replace(base, reason=str(failure)))
        reported = replace(
            base,
            verdict=str(attestation.get("verdict") or ""),
            subject_digest=str(attestation.get("subject_digest") or ""),
            kth_revision=str(attestation.get("kth_sha") or attestation.get("kth_revision") or ""),
            execution_mode=str(attestation.get("execution_mode") or ""),
            primary_detector=str(attestation.get("primary_detector") or ""),
            repair_feedback=_mapping(attestation.get("repair_feedback")),
        )
        try:
            if plan_id:
                self._check_reviewed(attestation, artifact, request, exit_code)
            else:
                self._check_adaptive(attestation, request, exit_code)
        except _Rejected as rejection:
            return self._persist(replace(reported, reason=str(rejection)))
        verdict = str(attestation["verdict"])
        return self._persist(replace(reported, status=_VERDICT_STATUS[verdict], reason=_reason(attestation)))

    def mark_performance_admitted(self, result: KthQualificationResult) -> KthQualificationResult:
        return self._persist(replace(result, performance_admitted=True))

    def _run(self, request: dict[str, Any], artifacts_dir: Path) -> tuple[int, dict[str, Any]]:
        """Run ``kth-qualify`` on ``request``; return its verdict exit code and the attestation it wrote."""
        request_path = artifacts_dir / "request.json"
        attestation_path = artifacts_dir / "attestation.json"
        atomic_write_json(request_path, request, trailing_newline=True)
        try:
            process = subprocess.run(
                [self.executable, "--request", str(request_path), "--out", str(attestation_path)],
                cwd=artifacts_dir,
                env=scrub_benchmark_process_env(os.environ.copy()),
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self._write_logs(artifacts_dir, error.stdout, error.stderr)
            raise _Rejected(f"KTH timed out after {self.timeout_s:g}s") from error
        except OSError as error:
            raise _Rejected(f"KTH executable {self.executable!r} could not run: {error}") from error
        self._write_logs(artifacts_dir, process.stdout, process.stderr)
        if process.returncode not in VERDICT_EXIT_CODES.values():
            raise _Rejected(f"KTH infrastructure failure (exit {process.returncode})")
        try:
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _Rejected(f"KTH attestation is missing or malformed: {error}") from error
        if not isinstance(attestation, dict):
            raise _Rejected("KTH attestation is not a JSON object")
        return process.returncode, attestation

    def _check_common(self, attestation: dict[str, Any], request: dict[str, Any], exit_code: int) -> None:
        if attestation.get("schema_version") != request["schema_version"]:
            raise _Rejected(f"unsupported KTH attestation schema {attestation.get('schema_version')!r}")
        if attestation.get("request_id") != request["request_id"]:
            raise _Rejected("KTH attestation answers a different request (replayed or stale)")
        verdict = attestation.get("verdict")
        if verdict not in VERDICT_EXIT_CODES:
            raise _Rejected(f"KTH verdict {verdict!r} is not one of the three contract verdicts")
        if VERDICT_EXIT_CODES[verdict] != exit_code:
            raise _Rejected(f"KTH verdict {verdict!r} disagrees with exit code {exit_code}")
        if not isinstance(attestation.get("findings"), list):
            raise _Rejected("KTH findings are malformed")
        revision = str(attestation.get("kth_sha") or attestation.get("kth_revision") or "")
        # A run KTH stopped before binding names no revision; it cannot be Eligible, so there is nothing to pin.
        if not revision and verdict != VERDICT_ELIGIBLE:
            return
        if not REVISION.fullmatch(revision):
            raise _Rejected("KTH attestation does not name a full harness revision")
        if self.expected_revision and revision != self.expected_revision:
            raise _Rejected(f"KTH revision {revision} is not the configured {self.expected_revision}")

    def _check_reviewed(
        self,
        attestation: dict[str, Any],
        artifact: CandidateArtifact,
        request: dict[str, Any],
        exit_code: int,
    ) -> None:
        missing = sorted(_REVIEWED_REQUIRED - set(attestation))
        if missing:
            raise _Rejected(f"KTH attestation is missing {missing}")
        self._check_common(attestation, request, exit_code)
        candidate = request["candidate"]
        plan = _object(attestation, "qualification_plan")
        identity = _object(attestation, "candidate_identity")
        execution = _object(attestation, "execution_binding")
        coverage = _object(attestation, "mandatory_oracle_coverage")
        patch_sha = artifact.patch_digest.removeprefix("sha256:")
        if plan.get("plan_id") != request["plan_id"] or plan.get("kernel_path") != candidate["kernel_path"]:
            raise _Rejected("KTH qualified a different plan than the host selected")
        expected_identity = {
            "candidate_id": candidate["candidate_id"],
            "base_commit": candidate["base_commit"],
            "kernel_path": candidate["kernel_path"],
            "patch_sha256": patch_sha,
            "plan_id": request["plan_id"],
        }
        for key, value in expected_identity.items():
            if identity.get(key) != value:
                raise _Rejected(f"KTH candidate identity does not match the candidate sent: {key}")
        if execution.get("patch_sha256") != patch_sha or execution.get("base_commit") != candidate["base_commit"]:
            raise _Rejected("KTH execution is not bound to the submitted patch and base")
        if attestation["execution_mode"] not in _EXECUTION_MODES:
            raise _Rejected(f"KTH execution mode {attestation['execution_mode']!r} is not a contract value")
        subject = reviewed_subject_digest(
            base_commit=artifact.base_commit,
            patch=artifact.patch,
            kernel_path=candidate["kernel_path"],
            qualification_plan=plan,
        )
        if attestation["subject_digest"] != subject:
            raise _Rejected("KTH subject digest does not match the candidate sent")
        if attestation["verdict"] != VERDICT_ELIGIBLE:
            return
        outside = _outside_scope(artifact.changed_paths, plan)
        if outside:
            raise _Rejected(f"patch changes {outside}, which plan {request['plan_id']} does not cover")
        if not attestation["findings"]:
            raise _Rejected("KTH reported Eligible without findings")
        if (
            coverage.get("complete") is not True
            or coverage.get("missing_oracles")
            or not coverage.get("executed_cases")
        ):
            raise _Rejected("KTH reported Eligible without complete mandatory execution evidence")
        if plan.get("bind_source") is True and not execution.get("artifact_sha256"):
            raise _Rejected("KTH reported Eligible for a source-bound plan without an artifact digest")

    def _check_adaptive(self, attestation: dict[str, Any], request: dict[str, Any], exit_code: int) -> None:
        missing = sorted(_ADAPTIVE_REQUIRED - set(attestation))
        if missing:
            raise _Rejected(f"KTH attestation is missing {missing}")
        self._check_common(attestation, request, exit_code)
        autospec = _object(attestation, "autospec")
        resolved = _object(attestation, "resolved_spec")
        plan = _object(attestation, "evidence_plan")
        binding = _object(attestation, "binding")
        eligible = attestation["verdict"] == VERDICT_ELIGIBLE
        if not binding and not eligible:
            # KTH stops before binding when it cannot bind the artifact; that is already a non-eligible outcome.
            return
        envelope = request["envelope"]
        sent_digest = envelope_digest(envelope)
        if attestation["envelope_digest"] != sent_digest:
            raise _Rejected("KTH attestation covers a different candidate envelope")
        artifact = envelope["artifact"]
        expected = {
            "envelope_digest": sent_digest,
            "base_commit": envelope["base_commit"],
            "patch_digest": envelope["patch_digest"],
            "environment_identity": environment_identity(envelope["environment"]),
            "autospec_digest": autospec.get("digest"),
            "resolved_digest": resolved.get("digest"),
            "plan_digest": plan.get("digest"),
            "kth_revision": attestation["kth_revision"],
        }
        for key, declared in (
            ("binary_digest", artifact.get("binary_digest") or artifact.get("code_object_digest")),
            ("module_digest", artifact.get("module_digest")),
            ("pre_patch_module_digest", artifact.get("pre_patch_module_digest")),
        ):
            if declared:
                expected[key] = declared
        for key, value in expected.items():
            if binding.get(key) != value:
                raise _Rejected(f"KTH binding does not match the candidate sent: {key}")
        if any(not isinstance(binding.get(key), str) for key in ADAPTIVE_BINDING_KEYS):
            raise _Rejected("KTH binding is incomplete")
        subject = adaptive_subject_digest(binding)
        if binding.get("subject_digest") != subject or attestation["subject_digest"] != subject:
            raise _Rejected("KTH subject digest does not match the bound candidate")
        if not eligible:
            return
        if resolved.get("trust_class") != "fully_verified" or attestation.get("autospec_uncertainty") is not False:
            raise _Rejected("KTH reported Eligible without a fully verified specification")
        if plan.get("uncovered") or plan.get("required_unavailable") or not plan.get("selected"):
            raise _Rejected("KTH reported Eligible without complete mandatory execution evidence")
        if not attestation["findings"]:
            raise _Rejected("KTH reported Eligible without findings")

    def _write_logs(self, artifacts_dir: Path, stdout: bytes | str | None, stderr: bytes | str | None) -> None:
        for name, stream in (("stdout.log", stdout), ("stderr.log", stderr)):
            data = stream.encode("utf-8") if isinstance(stream, str) else (stream or b"")
            text = data[:_LOG_LIMIT_BYTES].decode("utf-8", errors="replace")
            atomic_write_text(artifacts_dir / name, redact_secret_values(text))

    def _persist(self, result: KthQualificationResult) -> KthQualificationResult:
        atomic_write_json(Path(result.artifacts_dir) / "result.json", asdict(result), trailing_newline=True)
        return result


def _candidate_id(operator_id: str) -> str:
    return operator_id if SAFE_ID.fullmatch(operator_id) else "operator-" + sha256_digest(operator_id.encode())[7:39]


def _publication_steering(publication: ControllerPatchPublication) -> list[str]:
    try:
        payload = json.loads(publication.publication_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"an unreadable publication ({error})"]
    return list(candidate_control_fields(payload))


def _reviewed_request(
    request_id: str,
    plan_id: str,
    artifact: CandidateArtifact,
    kernel_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": REVIEWED_REQUEST_SCHEMA,
        "request_id": request_id,
        "plan_id": plan_id,
        "candidate": {
            "candidate_id": artifact.candidate_id,
            "base_commit": artifact.base_commit,
            "kernel_path": kernel_path,
            "patch_base64": base64.b64encode(artifact.patch).decode("ascii"),
        },
    }


def _envelope(
    publication: ControllerPatchPublication,
    artifact: CandidateArtifact,
    *,
    session_id: str,
) -> dict[str, Any]:
    identity = publication.identity
    observations: dict[str, Any] = {"micro_validated": True}
    for key in _GEAK_OBSERVATION_KEYS:
        value = publication.manifest.get(key)
        if isinstance(value, (bool, int, float, str)):
            observations[key] = value
    declared = {"producer": "hyperloom_context", "trust_level": "hypothesized", "provenance": "controller_publication"}
    return build_candidate_envelope(
        artifact,
        environment={
            "hardware": identity["gpu"],
            "software": f"{identity['framework']}=={identity['framework_version']} {identity['backend']}",
            **declared,
        },
        hyperloom_context={
            "operator_id": publication.operator_id,
            "workload": publication.operator_name,
            "session_id": session_id,
            **declared,
        },
        geak_harness={
            "harness_id": identity["producer"],
            "observations": observations,
            "producer": "geak_harness",
            "trust_level": "hypothesized",
            "provenance": "controller_publication.manifest",
        },
        field_provenance={
            "patch": {
                "producer": "hyperloom_integration",
                "trust_level": "verified",
                "provenance": "git diff of the applied integration worktree",
                "digest": artifact.patch_digest,
            },
            "changed_paths": {
                "producer": "hyperloom_integration",
                "trust_level": "verified",
                "provenance": "git diff of the applied integration worktree",
            },
        },
    )


def _outside_scope(changed_paths: tuple[str, ...], plan: Mapping[str, Any]) -> list[str]:
    allowed = [str(item) for item in plan.get("allowed_paths") or []] or [str(plan.get("kernel_path") or "")]

    def covered(path: str) -> bool:
        return any(path == entry or (entry.endswith("/") and path.startswith(entry)) for entry in allowed)

    return [path for path in changed_paths if not covered(path)]


def _object(attestation: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = attestation.get(key)
    if not isinstance(value, dict):
        raise _Rejected(f"KTH attestation field {key} is malformed")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _reason(attestation: Mapping[str, Any]) -> str:
    feedback = attestation.get("repair_feedback")
    mechanism = feedback.get("primary_mechanism") if isinstance(feedback, dict) else None
    meaning = mechanism.get("meaning") if isinstance(mechanism, dict) else ""
    return str(meaning or attestation.get("reason") or attestation.get("primary_detector") or attestation["verdict"])


__all__ = [
    "KthConfigurationError",
    "KthQualificationProvider",
    "KthQualificationResult",
    "KthStatus",
]
