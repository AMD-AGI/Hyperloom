# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel.controller_publication import ControllerPatchPublication
from hyperloom.orchestrator.kernel.kth_qualification import (
    KthQualificationProvider,
    _subject_digest,
)


def _publication(tmp_path: Path) -> ControllerPatchPublication:
    repo = tmp_path / "repo"
    repo.mkdir()
    patch_path = tmp_path / "change.patch"
    patch_path.write_bytes(b"diff --git a/kernel.py b/kernel.py\n+fixed\n")
    report = tmp_path / "report.md"
    report.write_text("# report\n")
    metadata = tmp_path / "publication.json"
    metadata.write_text("{}")
    return ControllerPatchPublication(
        operator_id="kernel:forge-loop:rms:standalone:unknown:triton:mi300x",
        identity={},
        base_commit="a" * 40,
        best_commit="b" * 40,
        repo_root=repo,
        kernel_path="kernels/rmsnorm.py",
        operator_name="rms",
        manifest={},
        patch_path=patch_path,
        report_path=report,
        publication_path=metadata,
        kth_plan_id="host/rmsnorm-v1",
    )


def _attestation(request: dict, patch_bytes: bytes, verdict: str) -> dict:
    patch_sha = hashlib.sha256(patch_bytes).hexdigest()
    plan = {
        "plan_id": request["plan_id"],
        "plan_version": "1",
        "kernel_path": request["candidate"]["kernel_path"],
        "operation_spec": "rmsnorm",
        "operation_spec_version": "1",
        "candidate_registry_key": "host-rms",
        "reference_registry_key": "host-rms-reference",
        "budget": 8,
        "allow_fake_collectives": False,
        "mandatory_oracle_ids": ["REF", "ATTESTATION"],
    }
    return {
        "schema_version": "1.0.0",
        "request_id": request["request_id"],
        "subject_digest": _subject_digest(
            request["candidate"]["base_commit"],
            patch_bytes,
            request["candidate"]["kernel_path"],
            plan,
        ),
        "kth_sha": "c" * 40,
        "qualification_plan": plan,
        "candidate_identity": {
            "candidate_id": request["candidate"]["candidate_id"],
            "base_commit": request["candidate"]["base_commit"],
            "kernel_path": request["candidate"]["kernel_path"],
            "patch_sha256": patch_sha,
            "plan_id": request["plan_id"],
        },
        "execution_mode": "simulated",
        "hardware_identity": {"machine": "test"},
        "verdict": verdict,
        "primary_detector": "REF" if verdict == "Blocked" else None,
        "mandatory_oracle_coverage": {
            "complete": True,
            "expected_oracles": ["REF", "ATTESTATION"],
            "exercised_oracles": ["REF"],
            "missing_oracles": [],
            "expected_cases": ["one"],
            "executed_cases": ["one"],
        },
        "findings": [{"check_id": "REF", "status": "consistent"}],
        "unexplored_regions": ["other shapes"],
        "duration_s": 0.1,
        "replay": {"command": "host-owned replay"},
        "repair_feedback": {"primary_mechanism": {"id": "REF", "meaning": "reference mismatch"}},
    }


def _subprocess_result(
    verdict: str,
    *,
    mutate=None,
):
    code = {
        "Eligible for performance evaluation": 0,
        "Blocked": 2,
        "Inconclusive": 3,
    }[verdict]

    def run(command, **_kwargs):
        request_path = Path(command[command.index("--request") + 1])
        out_path = Path(command[command.index("--out") + 1])
        request = json.loads(request_path.read_text())
        patch = __import__("base64").b64decode(request["candidate"]["patch_base64"])
        attestation = _attestation(request, patch, verdict)
        if mutate:
            mutate(attestation)
        out_path.write_text(json.dumps(attestation))
        return subprocess.CompletedProcess(command, code, stdout="kth stdout", stderr="")

    return run


@pytest.mark.parametrize(
    ("verdict", "status"),
    [
        ("Eligible for performance evaluation", "eligible"),
        ("Blocked", "kth_blocked"),
        ("Inconclusive", "kth_inconclusive"),
    ],
)
def test_provider_maps_completed_outcomes_and_persists_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verdict: str,
    status: str,
) -> None:
    publication = _publication(tmp_path)
    monkeypatch.setattr(subprocess, "run", _subprocess_result(verdict))
    provider = KthQualificationProvider(executable="/trusted/kth-qualify")
    result = provider.qualify(
        publication,
        artifacts_root=tmp_path / "session" / "kth_qualification",
    )

    assert result.status == status
    artifact_dir = Path(result.artifacts_dir)
    assert (artifact_dir / "request.json").is_file()
    assert (artifact_dir / "attestation.json").is_file()
    assert (artifact_dir / "stdout.log").read_text() == "kth stdout"
    assert (artifact_dir / "stderr.log").is_file()
    assert json.loads((artifact_dir / "result.json").read_text())["performance_reached"] is False
    assert (artifact_dir / "repair_feedback.json").is_file()


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda attestation: attestation.update(request_id="stale"),
            "request identity is stale",
        ),
        (
            lambda attestation: attestation.update(subject_digest="sha256:" + "0" * 64),
            "subject digest mismatch",
        ),
        (
            lambda attestation: attestation["mandatory_oracle_coverage"].update(complete=False),
            "mandatory evidence is incomplete",
        ),
        (
            lambda attestation: attestation.update(kth_sha="unknown"),
            "no full harness revision",
        ),
    ],
)
def test_stale_digest_and_incomplete_evidence_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    reason: str,
) -> None:
    publication = _publication(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _subprocess_result("Eligible for performance evaluation", mutate=mutation),
    )
    result = KthQualificationProvider().qualify(
        publication,
        artifacts_root=tmp_path / "session" / "kth_qualification",
    )
    assert result.status == "needs_review"
    assert reason in result.reason
    assert result.performance_reached is False


def test_malformed_attestation_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publication = _publication(tmp_path)

    def run(command, **_kwargs):
        out_path = Path(command[command.index("--out") + 1])
        out_path.write_text("{not-json")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    result = KthQualificationProvider().qualify(
        publication,
        artifacts_root=tmp_path / "session" / "kth_qualification",
    )
    assert result.status == "needs_review"
    assert "malformed" in result.reason


def test_missing_executable_and_timeout_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publication = _publication(tmp_path)

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(subprocess, "run", missing)
    provider = KthQualificationProvider()
    missing_result = provider.qualify(
        publication,
        artifacts_root=tmp_path / "missing" / "kth_qualification",
    )
    assert missing_result.status == "needs_review"

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("kth-qualify", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    timeout_result = provider.qualify(
        publication,
        artifacts_root=tmp_path / "timeout" / "kth_qualification",
    )
    assert timeout_result.status == "needs_review"
    assert "timed out" in timeout_result.reason
