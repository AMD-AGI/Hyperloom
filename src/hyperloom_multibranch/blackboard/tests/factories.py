# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builders of the objects blackboard tests write, with the defaults a typical test needs."""

from __future__ import annotations

from collections.abc import Mapping

from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.contracts.measurement import AttachmentRef
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict
from hyperloom_multibranch.contracts.blackboard.peers import (
    FailureClass,
    Intent,
    IntentLayer,
    Judgement,
    Purpose,
    SandboxFacts,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
    Verdict,
)
from hyperloom_multibranch.contracts.blackboard.records import SessionProfile

START = 1_800_000_000.0
TTL = 60.0


class FakeClock:
    """A clock that moves a millisecond on every reading, so entries keep the order they were written in."""

    def __init__(self, start: float = START) -> None:
        self.now = start

    def __call__(self) -> float:
        self.now += 0.001
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def profile(**overrides: object) -> SessionProfile:
    fields: dict[str, object] = {
        "session_id": "sess-1",
        "model": "deepseek-ai/DeepSeek-R1",
        "model_type": "deepseek_v3",
        "architectures": ("DeepseekV3ForCausalLM",),
        "hardware": "MI355X",
        "framework": "vllm",
        "framework_version": "0.11.0",
        "precision": "fp8",
        "tp": 8,
        "conc": 64,
        "isl": 1024,
        "osl": 1024,
        "kernel_optimizer": "forge",
        "roofline_ceiling": {"output_tps": 1500.0},
    }
    fields.update(overrides)
    return SessionProfile(**fields)


def intent(
    branch: str,
    parent: str,
    target: str = "moe gemm",
    mechanism: str = "tuned kernel",
    layer: IntentLayer = IntentLayer.KERNEL,
) -> Intent:
    return Intent(branch, parent, target, mechanism, layer)


def delta(server_args: str = "--max-num-seqs 256") -> RecipeDelta:
    return RecipeDelta(config=ConfigDelta(server_args=server_args))


def request(
    request_id: str,
    of: Intent,
    change: RecipeDelta | None = None,
    *,
    kind: TestKind = TestKind.TEST,
    branch: str = "",
    purpose: Purpose = Purpose.EXPLORE,
    attachments: tuple[AttachmentRef, ...] = (),
) -> TestRequest:
    return TestRequest(
        kind=kind,
        purpose=purpose,
        branch_id=branch or of.branch,
        parent_commit=of.parent_commit,
        delta=change or delta(),
        intent_fingerprint=of.fingerprint,
        attachments=attachments,
        request_id=request_id,
    )


_FAILURE = {Validity.CANDIDATE_FAILED: FailureClass.CANDIDATE, Validity.MEASUREMENT_FAILED: FailureClass.INFRA}


def result(
    request_id: str,
    *,
    gpu_seconds: float = 600.0,
    validity: Validity = Validity.SUCCEEDED,
    profile_location: str = "",
    failure_reason: str = "",
    node: str = "node-1",
) -> TestResult:
    return TestResult(
        request_id=request_id,
        validity=validity,
        failure_class=_FAILURE.get(validity),
        failure_reason=failure_reason or None,
        sandbox=SandboxFacts(sandbox_id=f"sandbox-{node}", node_name=node, gpu_ids=(0, 1), backend="local"),
        manifest_sha256="a" * 64,
        gpu_seconds=gpu_seconds,
        profile_location=profile_location,
    )


def judgement(
    gain_pct: float,
    request_ids: tuple[str, ...],
    *,
    recorded: Mapping[str, float] | None = None,
    verdict: Verdict = Verdict.IMPROVED,
    foundation_fix: bool = False,
    fixes_correctness: bool = False,
    threshold_pct: float = 1.0,
) -> Judgement:
    return Judgement(
        verdict=verdict,
        gain_pct=gain_pct,
        threshold_pct=threshold_pct,
        request_ids=request_ids,
        recorded_value=dict(recorded or {"output_tps_pct": gain_pct}),
        accuracy=AccuracyVerdict.PASSED,
        fixes_correctness=fixes_correctness,
        foundation_fix=foundation_fix,
    )
