# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller.paths import operator_directory_name
from hyperloom.common.env import EnvValueError
from hyperloom.orchestrator.kernel.controller_publication import (
    ControllerPatchPublication,
    load_controller_publication,
)
from hyperloom.orchestrator.kernel.kth_contract import envelope_digest, sha256_digest
from hyperloom.orchestrator.kernel.kth_qualification import (
    KthConfigurationError,
    KthQualificationProvider,
    KthQualificationResult,
)
from hyperloom.orchestrator.kernel.tests.kth_fakes import REVISION, FakeKth

ELIGIBLE = "Eligible for performance evaluation"
_PATCH = b"diff --git a/k.py b/k.py\n--- a/k.py\n+++ b/k.py\n@@ -1 +1 @@\n-X = 1\n+X = 2\n"
_BASE = "c" * 40


def _publication(tmp_path: Path, **extra: object) -> ControllerPatchPublication:
    operator_id = "kernel:forge-loop:k:standalone:unknown:triton:mi355x"
    patch_dir = tmp_path / "patches" / operator_directory_name(operator_id)
    patch_dir.mkdir(parents=True, exist_ok=True)
    (patch_dir / "change.patch").write_bytes(_PATCH)
    (patch_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    (patch_dir / "publication.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "operator_id": operator_id,
                "identity": {
                    "producer": "forge-loop",
                    "kernel_name": "k",
                    "framework": "standalone",
                    "framework_version": "unknown",
                    "backend": "triton",
                    "gpu": "mi355x",
                },
                "base_commit": _BASE,
                "best_commit": "b" * 40,
                "repo_root": str(tmp_path),
                "kernel_path": "k.py",
                "operator_name": "k",
                "micro_validated": True,
                "changed_files": ["k.py"],
                "manifest": {"changed_files": ["k.py"], "mean_case_speedup": 1.4},
                **extra,
            }
        ),
        encoding="utf-8",
    )
    return load_controller_publication(patch_dir)


def _qualify(
    provider: KthQualificationProvider,
    tmp_path: Path,
    *,
    publication: ControllerPatchPublication | None = None,
    patch: bytes = _PATCH,
) -> KthQualificationResult:
    return provider.qualify(
        publication or _publication(tmp_path),
        base_commit=_BASE,
        patch=patch,
        changed_paths=("k.py",),
        artifacts_root=tmp_path / "kth",
        session_id="session",
    )


@pytest.fixture
def fake(tmp_path: Path) -> FakeKth:
    return FakeKth(tmp_path / "fake")


@pytest.fixture
def reviewed(fake: FakeKth) -> KthQualificationProvider:
    return KthQualificationProvider(executable=str(fake.executable), timeout_s=30, reviewed_plans={"k.py": "host/k-v1"})


@pytest.fixture
def adaptive(fake: FakeKth) -> KthQualificationProvider:
    return KthQualificationProvider(executable=str(fake.executable), timeout_s=30)


@pytest.mark.parametrize(
    ("verdict", "status"),
    [(ELIGIBLE, "eligible"), ("Blocked", "blocked"), ("Inconclusive", "inconclusive")],
)
@pytest.mark.parametrize("mode", ["reviewed", "adaptive"])
def test_each_verdict_maps_to_its_status(request, tmp_path, fake, mode, verdict, status) -> None:
    provider = request.getfixturevalue(mode)
    fake.answer(verdict)

    result = _qualify(provider, tmp_path)

    assert result.status == status
    assert result.eligible is (status == "eligible")
    assert result.verdict == verdict
    assert result.kth_revision == REVISION
    assert result.patch_digest == sha256_digest(_PATCH)
    recorded = json.loads((Path(result.artifacts_dir) / "result.json").read_text(encoding="utf-8"))
    assert recorded["status"] == status


def test_reviewed_request_carries_only_the_host_plan_and_the_candidate(tmp_path, fake, reviewed) -> None:
    _qualify(reviewed, tmp_path)

    (sent,) = fake.requests
    assert set(sent) == {"schema_version", "request_id", "plan_id", "candidate"}
    assert sent["schema_version"] == "1.0.0"
    assert sent["plan_id"] == "host/k-v1"
    assert set(sent["candidate"]) == {"candidate_id", "base_commit", "kernel_path", "patch_base64"}


def test_unmapped_kernel_is_sent_as_an_adaptive_envelope(tmp_path, fake, adaptive) -> None:
    result = _qualify(adaptive, tmp_path)

    (sent,) = fake.requests
    assert set(sent) == {"schema_version", "request_id", "envelope"}
    envelope = sent["envelope"]
    assert envelope["base_commit"] == _BASE
    assert envelope["changed_paths"] == ["k.py"]
    assert envelope["geak_harness"]["producer"] == "geak_harness"
    assert envelope["geak_harness"]["observations"] == {"micro_validated": True, "mean_case_speedup": 1.4}
    persisted = json.loads((Path(result.artifacts_dir) / "envelope.json").read_text(encoding="utf-8"))
    assert envelope_digest(persisted) == envelope_digest(envelope)


def test_geak_pass_does_not_override_blocked(tmp_path, fake, adaptive) -> None:
    fake.answer("Blocked")
    manifest = {"changed_files": ["k.py"], "mean_case_speedup": 1.4, "correctness_passed": True}
    publication = _publication(tmp_path, manifest=manifest)

    assert _qualify(adaptive, tmp_path, publication=publication).status == "blocked"
    (sent,) = fake.requests
    assert sent["envelope"]["geak_harness"]["observations"]["correctness_passed"] is True
    assert sent["envelope"]["geak_harness"]["trust_level"] == "hypothesized"


@pytest.mark.parametrize("mode", ["reviewed", "adaptive"])
@pytest.mark.parametrize("rebind_request_id", [False, True])
def test_stale_attestation_cannot_qualify_a_changed_candidate(request, tmp_path, fake, mode, rebind_request_id) -> None:
    provider = request.getfixturevalue(mode)
    original = _qualify(provider, tmp_path)
    assert original.eligible
    fake.answer(
        ELIGIBLE,
        replay=str(Path(original.artifacts_dir) / "attestation.json"),
        rebind_request_id=rebind_request_id,
    )

    changed = _qualify(provider, tmp_path, patch=_PATCH.replace(b"X = 2", b"X = 3"))

    assert changed.status == "failed"
    rebound = {"reviewed": "candidate identity does not match", "adaptive": "different candidate envelope"}
    assert (rebound[mode] if rebind_request_id else "different request") in changed.reason


def test_execution_mode_is_recorded_and_held_to_the_contract(tmp_path, fake, reviewed) -> None:
    assert _qualify(reviewed, tmp_path).execution_mode == "real"

    fake.answer(ELIGIBLE, set={"execution_mode": "emulated"})
    result = _qualify(reviewed, tmp_path)

    assert result.status == "failed"
    assert "execution mode" in result.reason


def test_autospec_without_confidence_stays_inconclusive(tmp_path, fake, adaptive) -> None:
    fake.answer("Inconclusive", trust_class="partial_inferred")

    assert _qualify(adaptive, tmp_path).status == "inconclusive"


def test_kth_stopping_before_binding_is_inconclusive_not_a_provider_failure(tmp_path, fake, adaptive) -> None:
    fake.answer(
        "Inconclusive",
        set={"binding": {}, "subject_digest": "", "envelope_digest": "", "kth_revision": ""},
    )

    assert _qualify(adaptive, tmp_path).status == "inconclusive"


@pytest.mark.parametrize(
    ("mode", "scenario", "reason"),
    [
        ("reviewed", {"exit": 1, "write": False}, "exit 1"),
        ("reviewed", {"raw": "{not json"}, "missing or malformed"),
        ("reviewed", {"raw": "[]"}, "not a JSON object"),
        ("reviewed", {"write": False, "exit": 0}, "missing or malformed"),
        ("reviewed", {"drop": ["mandatory_oracle_coverage"]}, "missing"),
        ("adaptive", {"drop": ["binding"]}, "missing"),
        ("reviewed", {"set": {"schema_version": "3.0.0"}}, "unsupported KTH attestation schema"),
        ("adaptive", {"set": {"schema_version": "1.0.0"}}, "unsupported KTH attestation schema"),
        ("reviewed", {"set": {"verdict": "Pass"}}, "not one of the three"),
        ("reviewed", {"exit": 2}, "disagrees with exit code"),
        ("reviewed", {"set": {"request_id": "hyperloom-replayed"}}, "different request"),
        ("adaptive", {"set": {"request_id": "hyperloom-replayed"}}, "different request"),
        ("reviewed", {"set": {"kth_sha": "unknown"}}, "full harness revision"),
        ("adaptive", {"set": {"kth_revision": "unknown"}}, "full harness revision"),
        ("reviewed", {"set": {"subject_digest": "sha256:" + "0" * 64}}, "subject digest"),
        ("adaptive", {"set": {"subject_digest": "sha256:" + "0" * 64}}, "subject digest"),
        ("reviewed", {"set": {"candidate_identity.patch_sha256": "0" * 64}}, "candidate identity"),
        ("reviewed", {"set": {"execution_binding.patch_sha256": "0" * 64}}, "not bound to the submitted patch"),
        ("reviewed", {"set": {"qualification_plan.plan_id": "host/other"}}, "different plan"),
        ("adaptive", {"set": {"envelope_digest": "sha256:" + "0" * 64}}, "different candidate envelope"),
        ("adaptive", {"set": {"binding.patch_digest": "sha256:" + "0" * 64}}, "binding does not match"),
        ("adaptive", {"set": {"binding.base_commit": "e" * 40}}, "binding does not match"),
        ("adaptive", {"set": {"binding.plan_digest": "sha256:" + "0" * 64}}, "binding does not match"),
        ("reviewed", {"set": {"mandatory_oracle_coverage.complete": False}}, "mandatory execution"),
        ("reviewed", {"set": {"mandatory_oracle_coverage.executed_cases": []}}, "mandatory execution"),
        ("reviewed", {"set": {"findings": []}}, "without findings"),
        ("reviewed", {"allowed_paths": ["other.py"]}, "does not cover"),
        ("adaptive", {"trust_class": "partial_inferred"}, "fully verified specification"),
        ("adaptive", {"set": {"autospec_uncertainty": True}}, "fully verified specification"),
        ("adaptive", {"set": {"evidence_plan.uncovered": ["semantics.x"]}}, "mandatory execution"),
    ],
)
def test_anything_short_of_a_bound_complete_eligible_fails_closed(
    request, tmp_path, fake, mode, scenario, reason
) -> None:
    provider = request.getfixturevalue(mode)
    fake.answer(ELIGIBLE, **scenario)

    result = _qualify(provider, tmp_path)

    assert result.status == "failed"
    assert not result.eligible
    assert reason in result.reason


def test_source_bound_plan_needs_an_artifact_digest(tmp_path, fake, reviewed) -> None:
    fake.answer(ELIGIBLE, bind_source=True)
    assert "without an artifact digest" in _qualify(reviewed, tmp_path).reason

    fake.answer(ELIGIBLE, bind_source=True, set={"execution_binding.artifact_sha256": "a" * 64})
    assert _qualify(reviewed, tmp_path).eligible


def test_revision_is_held_to_the_configured_pin(tmp_path, fake) -> None:
    pinned = KthQualificationProvider(executable=str(fake.executable), expected_revision="5" * 40)
    assert "not the configured" in _qualify(pinned, tmp_path).reason
    matching = KthQualificationProvider(executable=str(fake.executable), expected_revision=REVISION)
    assert _qualify(matching, tmp_path).eligible


def test_timeout_fails_closed(tmp_path, fake) -> None:
    fake.answer(ELIGIBLE, sleep=5)
    provider = KthQualificationProvider(executable=str(fake.executable), timeout_s=0.5)

    result = _qualify(provider, tmp_path)

    assert result.status == "failed"
    assert "timed out" in result.reason


def test_missing_executable_fails_closed(tmp_path) -> None:
    provider = KthQualificationProvider(executable=str(tmp_path / "no-such-kth"))

    result = _qualify(provider, tmp_path)

    assert result.status == "failed"
    assert "could not run" in result.reason


@pytest.mark.parametrize(
    "steering",
    [
        {"kth_qualification": {"plan_id": "fixture/weak"}},
        {"verdict": ELIGIBLE},
        {"manifest": {"changed_files": ["k.py"], "thresholds": {"atol": 1.0}}},
        {"manifest": {"changed_files": ["k.py"], "command": "python bench.py"}},
        {"manifest": {"changed_files": ["k.py"], "reference": "ref.py"}},
        {"manifest": {"changed_files": ["k.py"], "detector_ids": ["REF"]}},
        {"manifest": {"changed_files": ["k.py"], "acceptance_policy": "lenient"}},
    ],
)
def test_publication_cannot_steer_qualification(tmp_path, fake, reviewed, steering) -> None:
    result = _qualify(reviewed, tmp_path, publication=_publication(tmp_path, **steering))

    assert result.status == "failed"
    assert "steer" in result.reason
    assert fake.requests == []


def test_every_attempt_is_a_fresh_request(tmp_path, fake, reviewed) -> None:
    first = _qualify(reviewed, tmp_path)
    fake.answer(ELIGIBLE, write=False, exit=0)

    second = _qualify(reviewed, tmp_path)

    assert first.request_id != second.request_id
    assert first.artifacts_dir != second.artifacts_dir
    assert second.status == "failed"


def test_artifacts_do_not_leak_secrets(tmp_path, fake, reviewed, monkeypatch) -> None:
    secret = "sk-ant-api03-" + "Z" * 40
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    fake.answer(ELIGIBLE, echo_env=["ANTHROPIC_API_KEY"], stderr=f"debug token {secret}\n")

    result = _qualify(reviewed, tmp_path)

    assert result.eligible
    assert "ANTHROPIC_API_KEY <unset>" in (Path(result.artifacts_dir) / "stdout.log").read_text(encoding="utf-8")
    for path in Path(result.artifacts_dir).iterdir():
        assert secret not in path.read_text(encoding="utf-8"), path.name


def test_disabled_gate_builds_no_provider(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_KTH_ENABLE", raising=False)
    assert KthQualificationProvider.from_env() is None
    monkeypatch.setenv("HYPERLOOM_KTH_ENABLE", "0")
    assert KthQualificationProvider.from_env() is None


def test_enabled_gate_reads_host_configuration(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_KTH_ENABLE", "1")
    monkeypatch.setenv("HYPERLOOM_KTH_QUALIFY_EXECUTABLE", "/opt/kth/bin/kth-qualify")
    monkeypatch.setenv("HYPERLOOM_KTH_TIMEOUT_S", "45")
    monkeypatch.setenv("HYPERLOOM_KTH_EXPECTED_SHA", "A" * 40)
    monkeypatch.setenv("HYPERLOOM_KTH_PLANS", '{"aiter/ops/k.py": "host/k-v1"}')

    provider = KthQualificationProvider.from_env()

    assert provider == KthQualificationProvider(
        executable="/opt/kth/bin/kth-qualify",
        timeout_s=45.0,
        expected_revision="a" * 40,
        reviewed_plans={"aiter/ops/k.py": "host/k-v1"},
    )


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("HYPERLOOM_KTH_ENABLE", "maybe", EnvValueError),
        ("HYPERLOOM_KTH_TIMEOUT_S", "soon", EnvValueError),
        ("HYPERLOOM_KTH_TIMEOUT_S", "0", KthConfigurationError),
        ("HYPERLOOM_KTH_EXPECTED_SHA", "main", KthConfigurationError),
        ("HYPERLOOM_KTH_PLANS", "k.py=host/k", KthConfigurationError),
        ("HYPERLOOM_KTH_PLANS", '["host/k"]', KthConfigurationError),
        ("HYPERLOOM_KTH_PLANS", '{"k.py": "../escape plan"}', KthConfigurationError),
    ],
)
def test_misconfigured_gate_refuses_to_start(monkeypatch, name, value, error) -> None:
    monkeypatch.setenv("HYPERLOOM_KTH_ENABLE", "1")
    monkeypatch.setenv(name, value)

    with pytest.raises(error):
        KthQualificationProvider.from_env()


def _installed_kth() -> str:
    return os.environ.get("HYPERLOOM_TEST_KTH_EXECUTABLE", "")


@pytest.mark.skipif(not _installed_kth(), reason="set HYPERLOOM_TEST_KTH_EXECUTABLE to an installed kth-qualify")
@pytest.mark.parametrize(
    ("plan_id", "status"),
    [
        ("fixture/cpu-attention-control-v1", "eligible"),
        ("fixture/cpu-partial-write-v1", "blocked"),
        ("", "inconclusive"),
    ],
)
def test_installed_kth_answers_through_the_contract(tmp_path, monkeypatch, plan_id, status) -> None:
    """Runs the real ``kth-qualify``; its fixture plans need ``KTH_ALLOW_FIXTURE_PLANS=1``."""
    monkeypatch.setenv("KTH_ALLOW_FIXTURE_PLANS", "1")
    repo = tmp_path / "repo"
    (repo / "kernels").mkdir(parents=True)
    git = shutil.which("git") or "git"
    subprocess.run([git, "init", "-q", str(repo)], check=True)
    (repo / "kernels" / "attention.py").write_text("X = 1\n", encoding="utf-8")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t"}
    env["GIT_COMMITTER_EMAIL"] = "t@t"
    subprocess.run([git, "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run([git, "-C", str(repo), "commit", "-qm", "base"], check=True, env=env)
    head = subprocess.run([git, "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    patch = b"diff --git a/kernels/attention.py b/kernels/attention.py\n--- a/kernels/attention.py\n"
    patch += b"+++ b/kernels/attention.py\n@@ -1 +1 @@\n-X = 1\n+X = 2\n"
    plans = {"kernels/attention.py": plan_id} if plan_id else {}
    provider = KthQualificationProvider(executable=_installed_kth(), timeout_s=600, reviewed_plans=plans)
    publication = _publication(tmp_path)
    publication = type(publication)(**{**publication.__dict__, "kernel_path": "kernels/attention.py"})

    result = provider.qualify(
        publication,
        base_commit=head.stdout.strip(),
        patch=patch,
        changed_paths=("kernels/attention.py",),
        artifacts_root=tmp_path / "kth",
    )

    assert result.status == status, result.reason
