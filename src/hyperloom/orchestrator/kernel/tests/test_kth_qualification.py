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


def test_blocked_attestation_may_omit_unobserved_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _publication(tmp_path)

    def mutate(attestation):
        attestation["mandatory_oracle_coverage"]["complete"] = False
        attestation["mandatory_oracle_coverage"]["missing_oracles"] = ["PROVENANCE"]

    monkeypatch.setattr(subprocess, "run", _subprocess_result("Blocked", mutate=mutate))
    result = KthQualificationProvider().qualify(
        publication,
        artifacts_root=tmp_path / "session" / "kth_qualification",
    )
    assert result.status == "kth_blocked"
    assert result.performance_reached is False


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


def test_from_env_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPERLOOM_KTH_ENABLE", raising=False)
    monkeypatch.delenv("HYPERLOOM_KTH_ADAPTIVE", raising=False)
    provider = KthQualificationProvider.from_env()
    assert provider.enabled() is False
    assert provider.adaptive is False


def test_from_env_enable_turns_on_adaptive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_KTH_ENABLE", "1")
    provider = KthQualificationProvider.from_env()
    assert provider.enabled() is True
    assert provider.adaptive is True


def test_envelope_omits_forbidden_selection_fields(tmp_path: Path) -> None:
    from hyperloom.orchestrator.kernel.kth_qualification import build_candidate_envelope

    publication = _publication(tmp_path)
    envelope = build_candidate_envelope(publication, publication.patch_path.read_bytes())
    blob = json.dumps(envelope)
    assert "verdict" not in envelope
    assert "detector_ids" not in envelope
    assert "thresholds" not in envelope
    assert "command" not in blob
    assert envelope["schema_version"] == "1.0.0"
    assert envelope["candidate_id"] == publication.operator_id


def _adaptive_attestation(request: dict, verdict: str) -> dict:
    return {
        "schema_version": "2.0.0",
        "request_id": request["request_id"],
        "subject_digest": "sha256:" + "ab" * 32,
        "kth_revision": "c" * 40,
        "envelope_digest": "sha256:" + "de" * 32,
        "autospec": {
            "version": "1.0.0",
            "digest": "sha256:" + "11" * 32,
            "spec_trust_class": "fully_verified"
            if verdict == "Eligible for performance evaluation"
            else "partial_inferred",
            "unresolved": [],
            "conflicts": [],
        },
        "resolved_spec": {
            "name": "elementwise_binary",
            "version": "1.0",
            "source": "reviewed",
            "trust_class": "fully_verified"
            if verdict == "Eligible for performance evaluation"
            else "partial_inferred",
            "digest": "sha256:" + "22" * 32,
            "decisions": [],
        },
        "evidence_plan": {
            "plan_id": "adaptive/demo",
            "digest": "sha256:" + "33" * 32,
            "selected": [{"oracle_id": "generic.schema.v1", "check_id": "SCHEMA", "reason": "generic"}],
            "excluded": [{"oracle_id": "collective.topology.v1", "check_id": "CONCURRENCY", "reason": "irrelevant"}],
            "required_unavailable": [],
            "covered": ["contract.schema"],
            "uncovered": [],
            "cost_estimate": 3,
            "actual_cost": 0.1,
        },
        "verdict": verdict,
        "reason": "test",
        "findings": [{"check_id": "SCHEMA", "status": "consistent"}],
        "autospec_uncertainty": verdict != "Eligible for performance evaluation",
        "duration_s": 0.1,
    }


def _adaptive_subprocess(verdict: str, *, mutate=None):
    code = {
        "Eligible for performance evaluation": 0,
        "Blocked": 2,
        "Inconclusive": 3,
    }[verdict]

    def run(command, **_kwargs):
        request_path = Path(command[command.index("--request") + 1])
        out_path = Path(command[command.index("--out") + 1])
        request = json.loads(request_path.read_text())
        assert request["schema_version"] == "2.0.0"
        assert "envelope" in request
        attestation = _adaptive_attestation(request, verdict)
        if mutate:
            mutate(attestation)
        out_path.write_text(json.dumps(attestation))
        return subprocess.CompletedProcess(command, code, stdout="adaptive", stderr="")

    return run


def test_adaptive_blocked_never_maps_to_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = _publication(tmp_path)
    monkeypatch.setattr(subprocess, "run", _adaptive_subprocess("Blocked"))
    result = KthQualificationProvider(adaptive=True).qualify(
        publication, artifacts_root=tmp_path / "session" / "kth_qualification"
    )
    assert result.status == "kth_blocked"
    assert result.eligible is False
    request = json.loads(Path(result.artifacts_dir).joinpath("request.json").read_text())
    assert request["schema_version"] == "2.0.0"
    assert (Path(result.artifacts_dir) / "envelope.json").is_file()


def test_adaptive_inconclusive_never_maps_to_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = _publication(tmp_path)
    monkeypatch.setattr(subprocess, "run", _adaptive_subprocess("Inconclusive"))
    result = KthQualificationProvider(adaptive=True).qualify(
        publication, artifacts_root=tmp_path / "session" / "kth_qualification"
    )
    assert result.status == "kth_inconclusive"
    assert result.eligible is False


def test_adaptive_uncertain_eligible_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = _publication(tmp_path)

    def mutate(attestation):
        attestation["autospec_uncertainty"] = True
        attestation["resolved_spec"]["trust_class"] = "partial_inferred"

    monkeypatch.setattr(
        subprocess,
        "run",
        _adaptive_subprocess("Eligible for performance evaluation", mutate=mutate),
    )
    result = KthQualificationProvider(adaptive=True).qualify(
        publication, artifacts_root=tmp_path / "session" / "kth_qualification"
    )
    assert result.status == "needs_review"
    assert result.eligible is False


def test_replay_attestation_cannot_bind_a_different_request_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = _publication(tmp_path)
    captured: dict[str, str] = {}

    def first_run(command, **_kwargs):
        request_path = Path(command[command.index("--request") + 1])
        out_path = Path(command[command.index("--out") + 1])
        request = json.loads(request_path.read_text())
        captured["request_id"] = request["request_id"]
        captured["digest"] = hashlib.sha256(publication.patch_path.read_bytes()).hexdigest()
        attestation = _adaptive_attestation(request, "Blocked")
        out_path.write_text(json.dumps(attestation))
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", first_run)
    first = KthQualificationProvider(adaptive=True).qualify(
        publication, artifacts_root=tmp_path / "first"
    )
    replay = json.loads(Path(first.artifacts_dir).joinpath("attestation.json").read_text())

    def replay_on_stale(command, **_kwargs):
        out_path = Path(command[command.index("--out") + 1])
        stale = dict(replay)
        stale["request_id"] = "stale-other-artifact"
        out_path.write_text(json.dumps(stale))
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", replay_on_stale)
    second = KthQualificationProvider(adaptive=True).qualify(
        publication, artifacts_root=tmp_path / "second"
    )
    assert second.status == "needs_review"
    assert "stale" in second.reason
    assert first.subject_digest == replay["subject_digest"]

