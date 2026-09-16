# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Typed subprocess bridge for Kernel Trust Harness qualification."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from hyperloom.common.io import atomic_write_json, atomic_write_text

from .controller_publication import ControllerPatchPublication

KthStatus = Literal["eligible", "kth_blocked", "kth_inconclusive", "needs_review"]
_KTH_SHA = re.compile(r"^[0-9a-f]{40,64}$")
_VERDICT_EXIT = {
    "Eligible for performance evaluation": 0,
    "Blocked": 2,
    "Inconclusive": 3,
}


@dataclass(frozen=True)
class KthQualificationResult:
    status: KthStatus
    reason: str
    verdict: str = ""
    request_id: str = ""
    subject_digest: str = ""
    primary_detector: str = ""
    artifacts_dir: str = ""
    repair_feedback: dict[str, Any] | None = None
    performance_reached: bool = False

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"


@dataclass(frozen=True)
class KthQualificationProvider:
    executable: str = "kth-qualify"
    timeout_s: float = 300.0
    expected_kth_sha: str | None = None

    def qualify(
        self,
        publication: ControllerPatchPublication,
        *,
        artifacts_root: Path,
    ) -> KthQualificationResult:
        patch_bytes = publication.patch_path.read_bytes()
        patch_sha = hashlib.sha256(patch_bytes).hexdigest()
        request_id = f"kth-{patch_sha[:24]}"
        artifacts_dir = artifacts_root / hashlib.sha256(publication.operator_id.encode("utf-8")).hexdigest()[:24]
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        request_path = artifacts_dir / "request.json"
        attestation_path = artifacts_dir / "attestation.json"
        request = {
            "schema_version": "1.0.0",
            "request_id": request_id,
            "plan_id": publication.kth_plan_id,
            "candidate": {
                "candidate_id": publication.operator_id,
                "base_commit": publication.base_commit,
                "kernel_path": publication.kernel_path,
                "patch_base64": base64.b64encode(patch_bytes).decode("ascii"),
            },
        }
        atomic_write_json(request_path, request, trailing_newline=True)
        command = [
            self.executable,
            "--request",
            str(request_path),
            "--out",
            str(attestation_path),
        ]
        try:
            process = subprocess.run(
                command,
                cwd=publication.repo_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except FileNotFoundError:
            return self._persist(
                KthQualificationResult(
                    status="needs_review",
                    reason=f"KTH executable not found: {self.executable}",
                    request_id=request_id,
                    artifacts_dir=str(artifacts_dir),
                )
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout if isinstance(error.stdout, str) else ""
            stderr = error.stderr if isinstance(error.stderr, str) else ""
            atomic_write_text(artifacts_dir / "stdout.log", stdout, make_parents=True)
            atomic_write_text(artifacts_dir / "stderr.log", stderr, make_parents=True)
            return self._persist(
                KthQualificationResult(
                    status="needs_review",
                    reason=f"KTH timed out after {self.timeout_s:g}s",
                    request_id=request_id,
                    artifacts_dir=str(artifacts_dir),
                )
            )

        atomic_write_text(artifacts_dir / "stdout.log", process.stdout)
        atomic_write_text(artifacts_dir / "stderr.log", process.stderr)
        if process.returncode not in {0, 2, 3}:
            return self._persist(
                KthQualificationResult(
                    status="needs_review",
                    reason=f"KTH infrastructure exit {process.returncode}",
                    request_id=request_id,
                    artifacts_dir=str(artifacts_dir),
                )
            )
        try:
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return self._persist(
                KthQualificationResult(
                    status="needs_review",
                    reason=f"KTH attestation is missing or malformed: {error}",
                    request_id=request_id,
                    artifacts_dir=str(artifacts_dir),
                )
            )
        invalid = self._validate_attestation(
            attestation,
            request=request,
            patch_bytes=patch_bytes,
            process_exit=process.returncode,
        )
        if invalid:
            return self._persist(
                KthQualificationResult(
                    status="needs_review",
                    reason=invalid,
                    verdict=str(attestation.get("verdict") or ""),
                    request_id=request_id,
                    subject_digest=str(attestation.get("subject_digest") or ""),
                    primary_detector=str(attestation.get("primary_detector") or ""),
                    artifacts_dir=str(artifacts_dir),
                    repair_feedback=_mapping(attestation.get("repair_feedback")),
                )
            )
        verdict = str(attestation["verdict"])
        status: KthStatus = {
            "Eligible for performance evaluation": "eligible",
            "Blocked": "kth_blocked",
            "Inconclusive": "kth_inconclusive",
        }[verdict]
        feedback = _mapping(attestation.get("repair_feedback"))
        primary = str(attestation.get("primary_detector") or "")
        reason = _feedback_reason(feedback, primary, verdict)
        return self._persist(
            KthQualificationResult(
                status=status,
                reason=reason,
                verdict=verdict,
                request_id=request_id,
                subject_digest=str(attestation["subject_digest"]),
                primary_detector=primary,
                artifacts_dir=str(artifacts_dir),
                repair_feedback=feedback,
            )
        )

    def mark_performance_reached(self, result: KthQualificationResult) -> KthQualificationResult:
        updated = replace(result, performance_reached=True)
        return self._persist(updated)

    def _persist(self, result: KthQualificationResult) -> KthQualificationResult:
        if result.artifacts_dir:
            directory = Path(result.artifacts_dir)
            atomic_write_json(
                directory / "result.json",
                asdict(result),
                trailing_newline=True,
            )
            atomic_write_json(
                directory / "repair_feedback.json",
                result.repair_feedback or {},
                trailing_newline=True,
            )
        return result

    def _validate_attestation(
        self,
        attestation: Any,
        *,
        request: dict[str, Any],
        patch_bytes: bytes,
        process_exit: int,
    ) -> str:
        if not isinstance(attestation, dict):
            return "KTH attestation must be a JSON object"
        required = {
            "schema_version",
            "request_id",
            "subject_digest",
            "kth_sha",
            "qualification_plan",
            "candidate_identity",
            "execution_mode",
            "hardware_identity",
            "verdict",
            "primary_detector",
            "mandatory_oracle_coverage",
            "findings",
            "unexplored_regions",
            "duration_s",
            "replay",
        }
        missing = sorted(required - set(attestation))
        if missing:
            return f"KTH attestation missing required fields: {missing}"
        if attestation["schema_version"] != "1.0.0":
            return "KTH attestation schema version is unsupported"
        if attestation["request_id"] != request["request_id"]:
            return "KTH attestation request identity is stale"
        verdict = str(attestation["verdict"])
        if verdict not in _VERDICT_EXIT or _VERDICT_EXIT[verdict] != process_exit:
            return "KTH verdict and subprocess exit do not match"
        kth_sha = str(attestation["kth_sha"])
        if not _KTH_SHA.fullmatch(kth_sha):
            return "KTH attestation has no full harness revision"
        if self.expected_kth_sha and kth_sha != self.expected_kth_sha:
            return "KTH harness revision does not match the configured revision"

        identity = attestation["candidate_identity"]
        plan = attestation["qualification_plan"]
        coverage = attestation["mandatory_oracle_coverage"]
        if not isinstance(identity, dict) or not isinstance(plan, dict):
            return "KTH candidate or plan identity is malformed"
        candidate = request["candidate"]
        expected_identity = {
            "candidate_id": candidate["candidate_id"],
            "base_commit": candidate["base_commit"],
            "kernel_path": candidate["kernel_path"],
            "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
            "plan_id": request["plan_id"],
        }
        for key, value in expected_identity.items():
            if identity.get(key) != value:
                return f"KTH candidate identity mismatch: {key}"
        if plan.get("plan_id") != request["plan_id"]:
            return "KTH qualification plan identity mismatch"
        if plan.get("kernel_path") != candidate["kernel_path"]:
            return "KTH qualification plan path mismatch"
        expected_digest = _subject_digest(
            candidate["base_commit"],
            patch_bytes,
            candidate["kernel_path"],
            plan,
        )
        if attestation["subject_digest"] != expected_digest:
            return "KTH subject digest mismatch"
        if not isinstance(coverage, dict) or coverage.get("complete") is not True:
            return "KTH mandatory evidence is incomplete"
        if not isinstance(attestation["findings"], list) or not attestation["findings"]:
            return "KTH findings are missing"
        if not isinstance(attestation["unexplored_regions"], list):
            return "KTH unexplored-region evidence is malformed"
        return ""


def _mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _feedback_reason(feedback: dict[str, Any] | None, primary: str, verdict: str) -> str:
    mechanism = (feedback or {}).get("primary_mechanism")
    meaning = mechanism.get("meaning") if isinstance(mechanism, dict) else ""
    return str(meaning or primary or verdict)


def _subject_digest(
    base_commit: str,
    patch_bytes: bytes,
    kernel_path: str,
    qualification_plan: dict[str, Any],
) -> str:
    parts = [
        ("base_commit", base_commit.encode("ascii")),
        ("patch", patch_bytes),
        ("kernel_path", kernel_path.encode("utf-8")),
        (
            "qualification_plan",
            json.dumps(
                qualification_plan,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        ),
    ]
    digest = hashlib.sha256()
    digest.update(b"kth-subject-v1\0")
    for name, value in parts:
        label = name.encode("utf-8")
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "KthQualificationProvider",
    "KthQualificationResult",
    "KthStatus",
]
