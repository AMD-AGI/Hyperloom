# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed subprocess bridge for independent KTH qualification."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field, replace
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
_VERDICT_LOG = {
    "Eligible for performance evaluation": "ELIGIBLE",
    "Blocked": "BLOCKED",
    "Inconclusive": "INCONCLUSIVE",
}


def _log(*lines: str) -> None:
    for line in lines:
        print(f"[HYPERLOOM] {line}", flush=True)


def _parse_map(raw: str) -> dict[str, str]:
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("KTH host map JSON must be an object")
        return {str(key): str(value) for key, value in payload.items() if str(key) and str(value)}
    mapping: dict[str, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, separator, value = part.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise ValueError(f"invalid KTH host map entry: {part!r}")
        mapping[key.strip()] = value.strip()
    return mapping


def _git_sha(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = completed.stdout.strip().lower()
    return sha if _KTH_SHA.fullmatch(sha) else None


def _working_tree_digest(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "diff", "HEAD"],
        capture_output=True,
        timeout=60,
        check=False,
    )
    digest = hashlib.sha256()
    digest.update(completed.stdout)
    return "sha256:" + digest.hexdigest()


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
    plan_id: str = ""
    patch_sha256: str = ""
    pre_change_tree: str = ""
    post_change_tree: str = ""

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"


@dataclass(frozen=True)
class KthQualificationProvider:
    executable: str = "kth-qualify"
    timeout_s: float = 300.0
    expected_kth_sha: str | None = None
    operation_plans: dict[str, str] = field(default_factory=dict)
    kernel_path_plans: dict[str, str] = field(default_factory=dict)
    allowed_path_prefixes: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> KthQualificationProvider:
        root = os.environ.get("HYPERLOOM_KTH_ROOT")
        expected = os.environ.get("HYPERLOOM_KTH_EXPECTED_SHA") or None
        if not expected and root:
            expected = _git_sha(Path(root))
        prefixes = tuple(
            item.strip()
            for item in os.environ.get("HYPERLOOM_KTH_ALLOWED_PATHS", "").split(",")
            if item.strip()
        )
        return cls(
            executable=os.environ.get("HYPERLOOM_KTH_QUALIFY_EXECUTABLE", "kth-qualify"),
            timeout_s=float(os.environ.get("HYPERLOOM_KTH_TIMEOUT_S", "300")),
            expected_kth_sha=expected,
            operation_plans=_parse_map(os.environ.get("HYPERLOOM_KTH_OPERATION_PLANS", "")),
            kernel_path_plans=_parse_map(os.environ.get("HYPERLOOM_KTH_KERNEL_PATH_PLANS", "")),
            allowed_path_prefixes=prefixes,
        )

    def plan_for(self, publication: ControllerPatchPublication) -> str | None:
        if publication.operator_name in self.operation_plans:
            return self.operation_plans[publication.operator_name]
        if publication.kernel_path in self.kernel_path_plans:
            return self.kernel_path_plans[publication.kernel_path]
        for path, plan_id in self.kernel_path_plans.items():
            if publication.kernel_path == path or publication.kernel_path.endswith("/" + path):
                return plan_id
        return None

    def resolve_plan(
        self,
        publication: ControllerPatchPublication,
        touched: list[str],
    ) -> tuple[str | None, str]:
        host_plan = self.plan_for(publication)
        declared = publication.kth_plan_id
        if host_plan:
            if declared and declared != host_plan:
                return None, (
                    f"publication plan {declared!r} does not match host-owned plan {host_plan!r}"
                )
            scope_error = self._scope_error(host_plan, publication, touched)
            if scope_error:
                return None, scope_error
            return host_plan, ""
        if declared:
            return None, (
                f"publication selected plan {declared!r}, but the host did not map this operation"
            )
        return None, ""

    def _scope_error(
        self,
        plan_id: str,
        publication: ControllerPatchPublication,
        touched: list[str],
    ) -> str:
        if not touched:
            return "KTH cannot bind an empty patch scope"
        allowed = self.allowed_path_prefixes or (publication.kernel_path,)
        illegal = [
            path
            for path in touched
            if not any(path == prefix or path.startswith(prefix.rstrip("/") + "/") or path.startswith(prefix) for prefix in allowed)
        ]
        if illegal:
            return f"patch paths {illegal} are outside the host-owned plan {plan_id} scope {list(allowed)}"
        return ""

    def qualify(
        self,
        publication: ControllerPatchPublication,
        *,
        artifacts_root: Path,
        plan_id: str,
        pre_change_tree: str = "",
        post_change_tree: str = "",
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
            "plan_id": plan_id,
            "candidate": {
                "candidate_id": publication.operator_id,
                "base_commit": publication.base_commit,
                "kernel_path": publication.kernel_path,
                "patch_base64": base64.b64encode(patch_bytes).decode("ascii"),
            },
        }
        atomic_write_json(request_path, request, trailing_newline=True)
        _log(f"Candidate ready: sha256:{patch_sha}")
        _log("Requesting independent KTH qualification")
        command = [
            self.executable,
            "--request",
            str(request_path),
            "--out",
            str(attestation_path),
        ]
        try:
            process = self._run_command(
                command,
                cwd=publication.repo_root,
                timeout_s=self.timeout_s,
            )
        except FileNotFoundError:
            return self._persist(
                KthQualificationResult(
                    status="needs_review",
                    reason=f"KTH executable not found: {self.executable}",
                    request_id=request_id,
                    artifacts_dir=str(artifacts_dir),
                    plan_id=plan_id,
                    patch_sha256=patch_sha,
                    pre_change_tree=pre_change_tree,
                    post_change_tree=post_change_tree,
                )
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout if isinstance(error.stdout, str) else ""
            stderr = error.stderr if isinstance(error.stderr, str) else ""
            atomic_write_text(artifacts_dir / "stdout.log", stdout or "", make_parents=True)
            atomic_write_text(artifacts_dir / "stderr.log", stderr or "", make_parents=True)
            return self._persist(
                KthQualificationResult(
                    status="needs_review",
                    reason=f"KTH timed out after {self.timeout_s:g}s",
                    request_id=request_id,
                    artifacts_dir=str(artifacts_dir),
                    plan_id=plan_id,
                    patch_sha256=patch_sha,
                    pre_change_tree=pre_change_tree,
                    post_change_tree=post_change_tree,
                )
            )

        atomic_write_text(artifacts_dir / "stdout.log", process.stdout or "")
        atomic_write_text(artifacts_dir / "stderr.log", process.stderr or "")
        if process.returncode not in {0, 2, 3}:
            return self._persist(
                KthQualificationResult(
                    status="needs_review",
                    reason=f"KTH infrastructure exit {process.returncode}",
                    request_id=request_id,
                    artifacts_dir=str(artifacts_dir),
                    plan_id=plan_id,
                    patch_sha256=patch_sha,
                    pre_change_tree=pre_change_tree,
                    post_change_tree=post_change_tree,
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
                    plan_id=plan_id,
                    patch_sha256=patch_sha,
                    pre_change_tree=pre_change_tree,
                    post_change_tree=post_change_tree,
                )
            )
        invalid = self._validate_attestation(
            attestation,
            request=request,
            patch_bytes=patch_bytes,
            process_exit=process.returncode,
            pre_change_tree=pre_change_tree,
            post_change_tree=post_change_tree,
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
                    plan_id=plan_id,
                    patch_sha256=patch_sha,
                    pre_change_tree=pre_change_tree,
                    post_change_tree=post_change_tree,
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
        _log(f"KTH verdict: {_VERDICT_LOG[verdict]}")
        if status != "eligible":
            _log("Performance benchmark: SKIPPED")
            _log("Candidate: REJECTED")
        else:
            _log("Performance benchmark: PERMITTED")
        _log(f"Evidence retained: {artifacts_dir}")
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
                plan_id=plan_id,
                patch_sha256=patch_sha,
                pre_change_tree=pre_change_tree,
                post_change_tree=post_change_tree,
            )
        )

    def mark_performance_reached(self, result: KthQualificationResult) -> KthQualificationResult:
        updated = replace(result, performance_reached=True)
        return self._persist(updated)

    def _run_command(
        self,
        command: list[str],
        *,
        cwd: Path,
        timeout_s: float,
    ) -> subprocess.CompletedProcess[str]:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            raise
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def _pump_stderr() -> None:
            if process.stderr is None:
                return
            for line in process.stderr:
                stderr_lines.append(line)
                sys.stderr.write(line)
                sys.stderr.flush()

        reader = threading.Thread(target=_pump_stderr, daemon=True)
        reader.start()
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    stdout_lines.append(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait(timeout=5)
            reader.join(timeout=2)
            raise subprocess.TimeoutExpired(
                command,
                timeout_s,
                output="".join(stdout_lines),
                stderr="".join(stderr_lines),
            ) from error
        reader.join(timeout=2)
        return subprocess.CompletedProcess(
            command,
            returncode,
            "".join(stdout_lines),
            "".join(stderr_lines),
        )

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
        pre_change_tree: str,
        post_change_tree: str,
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
            return "KTH harness revision does not match the host-pinned revision"

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
        if not isinstance(coverage, dict):
            return "KTH coverage evidence is malformed"
        if verdict == "Eligible for performance evaluation" and coverage.get("complete") is not True:
            return "KTH mandatory evidence is incomplete"
        if not isinstance(attestation["findings"], list) or not attestation["findings"]:
            return "KTH findings are missing"
        if not isinstance(attestation["unexplored_regions"], list):
            return "KTH unexplored-region evidence is malformed"
        if attestation.get("execution_mode") == "real":
            source_binding = attestation.get("source_binding")
            if not isinstance(source_binding, dict):
                return "KTH source binding is missing"
            if source_binding.get("base_commit") != candidate["base_commit"]:
                return "KTH source binding does not match the pre-change tree"
            if source_binding.get("patch_sha256") != expected_identity["patch_sha256"]:
                return "KTH source binding does not match the submitted patch"
            observed_tree = str(source_binding.get("post_change_tree") or "")
            if post_change_tree and observed_tree and observed_tree != post_change_tree:
                return "KTH source binding does not match the post-change tree"
            if verdict == "Eligible for performance evaluation":
                binary = attestation.get("binary_binding")
                if not isinstance(binary, dict) or not binary.get("sha256"):
                    return "KTH binary binding is missing"
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
    "_working_tree_digest",
]
