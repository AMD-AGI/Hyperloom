# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The projections are pure, so they are tested against a hand-built context."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.kernel.kernel_context import (
    ArtifactRef,
    EvidenceIndex,
    KernelContext,
    ServingFacts,
    WorkloadFacts,
)
from hyperloom.orchestrator.kernel.lane_inputs import (
    FusionAgent,
    FusionExecution,
    GemmExecution,
    GemmShapeSources,
    fusion_input,
    gemm_input,
)


def _context(**overrides) -> KernelContext:
    workload = WorkloadFacts(
        model_name="example/model",
        model_path="example/model",
        resolved_model_path="/models/example",
        precision="fp8",
        quant_type="blockscale",
        gpu_type="mi355x",
        tp=8,
        isl=1024,
        osl=256,
        conc=64,
        max_model_len=4096,
    )
    serving = ServingFacts(framework="sglang", framework_repo_root="/src/sglang")
    evidence = EvidenceIndex(decode_trace=ArtifactRef(path="/traces/decode.json", available=True))
    return KernelContext(
        session_dir=Path("/session"),
        workload=overrides.get("workload", workload),
        serving=overrides.get("serving", serving),
        evidence=overrides.get("evidence", evidence),
    )


def _gemm_execution(**overrides) -> GemmExecution:
    values = {
        "forge_framework": "sglang",
        "global_timeout": 3600,
        "per_tuner_timeout": 1800,
        "mp": 8,
        "tp": 8,
        "conc": 64,
        "gpu_type": "mi355x",
    }
    values.update(overrides)
    return GemmExecution(**values)


def _fusion_execution(**overrides) -> FusionExecution:
    values = {"framework": "sglang", "timeout": 7200}
    values.update(overrides)
    return FusionExecution(**values)


_AGENT = FusionAgent(backend="claude", model="claude-x", sandbox_mode="workspace-write", max_turns=100)


class TestGemmProjection:
    def test_the_tuner_is_handed_the_local_directory_not_the_repo_id(self):
        payload = gemm_input(
            _context(),
            workspace=Path("/ws"),
            shapes=GemmShapeSources(),
            execution=_gemm_execution(),
        )

        assert payload["model_path"] == "/models/example"

    def test_a_lane_ceiling_of_zero_is_omitted_rather_than_sent(self):
        """gemm-tune reads an absent flag as "run every routed tuner"."""
        assert "max_tuners" not in gemm_input(
            _context(),
            workspace=Path("/ws"),
            shapes=GemmShapeSources(),
            execution=_gemm_execution(max_tuners=0),
        )
        assert (
            gemm_input(
                _context(),
                workspace=Path("/ws"),
                shapes=GemmShapeSources(),
                execution=_gemm_execution(max_tuners=3),
            )["max_tuners"]
            == 3
        )

    def test_the_per_tuner_cap_stays_below_the_session_cap(self):
        payload = gemm_input(
            _context(),
            workspace=Path("/ws"),
            shapes=GemmShapeSources(),
            execution=_gemm_execution(),
        )

        assert payload["timeout"] < payload["global_timeout"]


class TestFusionProjection:
    def test_the_ab_benchmark_runs_at_the_workload_operating_point(self):
        """forge-fuse has always accepted these; nothing forwarded them.

        Left unset the A/B compared the fused and unfused kernels at the CLI's
        default sequence lengths and decode batch rather than the ones the
        session measured, so the verdict was about a workload nobody runs.
        """
        payload = fusion_input(
            _context(),
            workspace=Path("/ws"),
            agent=_AGENT,
            execution=_fusion_execution(),
        )

        assert payload["ab_isl"] == 1024
        assert payload["ab_osl"] == 256
        assert payload["decode_batch"] == 64

    def test_the_framework_checkout_is_named_rather_than_searched_for(self):
        payload = fusion_input(
            _context(),
            workspace=Path("/ws"),
            agent=_AGENT,
            execution=_fusion_execution(),
        )

        assert payload["framework_root"] == "/src/sglang"

    def test_an_unknown_operating_point_is_omitted_not_zeroed(self):
        """A zero would instruct forge-fuse; an absent flag leaves it deciding."""
        payload = fusion_input(
            _context(workload=WorkloadFacts(model_path="example/model")),
            workspace=Path("/ws"),
            agent=_AGENT,
            execution=_fusion_execution(),
        )

        for absent in ("tp", "ab_isl", "ab_osl", "decode_batch", "max_model_len", "block_size"):
            assert absent not in payload

    def test_a_repository_root_nobody_configured_is_omitted(self):
        payload = fusion_input(
            _context(serving=ServingFacts(framework="sglang")),
            workspace=Path("/ws"),
            agent=_AGENT,
            execution=_fusion_execution(),
        )

        assert "framework_root" not in payload

    def test_the_decode_trace_comes_from_the_shared_evidence_index(self):
        payload = fusion_input(
            _context(),
            workspace=Path("/ws"),
            agent=_AGENT,
            execution=_fusion_execution(),
        )

        assert payload["trace_path"] == "/traces/decode.json"

    def test_a_trace_that_vanished_is_not_forwarded_as_a_path(self):
        payload = fusion_input(
            _context(evidence=EvidenceIndex(decode_trace=ArtifactRef(path="/gone.json", available=False))),
            workspace=Path("/ws"),
            agent=_AGENT,
            execution=_fusion_execution(),
        )

        assert payload["trace_path"] == ""
