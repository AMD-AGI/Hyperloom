# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builders of the objects notebook tests write, with the defaults a typical test needs."""

from __future__ import annotations

from collections.abc import Mapping

from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.contracts.notebook.peers import (
    FailureClass,
    Intent,
    IntentLayer,
    Judgement,
    SandboxFacts,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
    Verdict,
)
from hyperloom_multibranch.contracts.notebook.records import SessionProfile

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
) -> TestRequest:
    return TestRequest(
        request_id=request_id,
        kind=kind,
        branch=branch or of.branch,
        parent_commit=of.parent_commit,
        intent_fingerprint=of.fingerprint,
        delta=change or delta(),
    )


def result(
    request_id: str,
    *,
    gpu_seconds: float = 600.0,
    validity: Validity = Validity.SUCCESS,
    profile_location: str = "",
    failure_reason: str = "",
    node: str = "node-1",
) -> TestResult:
    return TestResult(
        request_id=request_id,
        validity=validity,
        sandbox=SandboxFacts(node=node, gpu_ids=(0, 1), backend="local"),
        manifest_sha256="m" * 64,
        failure=FailureClass.CANDIDATE if validity is Validity.CANDIDATE_FAILURE else None,
        failure_reason=failure_reason,
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
) -> Judgement:
    return Judgement(
        verdict=verdict,
        gain_pct=gain_pct,
        threshold_pct=1.0,
        request_ids=request_ids,
        recorded_value=dict(recorded or {"output_tps_pct": gain_pct}),
        accuracy_ok=True,
        fixes_correctness=fixes_correctness,
        foundation_fix=foundation_fix,
    )
