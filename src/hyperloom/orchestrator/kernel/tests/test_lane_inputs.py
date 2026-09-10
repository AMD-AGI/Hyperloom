# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What each lane is handed, read off the context in one place."""

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
    fusion_input,
)

_AGENT = FusionAgent(backend="claude", model="claude-x", sandbox_mode="workspace-write", max_turns=100)


def _context(tmp_path: Path, **workload) -> KernelContext:
    trace = tmp_path / "decode.trace.json.gz"
    trace.write_text("{}", encoding="utf-8")
    return KernelContext(
        session_dir=tmp_path,
        workload=WorkloadFacts(model_path="/models/m", **workload),
        serving=ServingFacts(framework="sglang"),
        evidence=EvidenceIndex(decode_trace=ArtifactRef.of(trace)),
    )


def _fusion(context: KernelContext, tmp_path: Path, **execution) -> dict:
    return fusion_input(
        context,
        workspace=tmp_path / "ws",
        agent=_AGENT,
        execution=FusionExecution(framework="sglang", timeout=7200, **execution),
    )


class TestTheOperatingPointReachesTheAb:
    """The A/B has to drive the server at the point the session measured."""

    def test_the_sequence_lengths_and_decode_batch_are_forwarded(self, tmp_path):
        context = _context(tmp_path, isl=1024, osl=256, conc=32, tp=8, max_model_len=8192)

        payload = _fusion(context, tmp_path)

        assert payload["ab_isl"] == 1024
        assert payload["ab_osl"] == 256
        assert payload["decode_batch"] == 32
        assert payload["tp"] == 8
        assert payload["max_model_len"] == 8192

    def test_a_fact_the_session_never_stated_is_omitted_not_zeroed(self, tmp_path):
        # forge-fuse reads an absent flag as "use your own default", which is a
        # different instruction from an explicit 0.
        payload = _fusion(_context(tmp_path), tmp_path)

        for key in ("ab_isl", "ab_osl", "decode_batch", "tp", "max_model_len", "block_size", "max_recipes"):
            assert key not in payload


class TestTheFrameworkRoot:
    def test_a_configured_checkout_is_named(self, tmp_path):
        context = _context(tmp_path)
        context = KernelContext(
            session_dir=context.session_dir,
            workload=context.workload,
            serving=ServingFacts(framework="sglang", framework_repo_root="/src/sglang"),
            evidence=context.evidence,
        )

        assert _fusion(context, tmp_path)["framework_root"] == "/src/sglang"

    def test_no_configured_checkout_leaves_forge_fuse_to_auto_detect(self, tmp_path):
        assert "framework_root" not in _fusion(_context(tmp_path), tmp_path)


class TestTheTraceAndTheAgent:
    def test_the_decode_trace_comes_from_the_evidence_index(self, tmp_path):
        context = _context(tmp_path)

        assert _fusion(context, tmp_path)["trace_path"] == context.evidence.decode_trace.path

    def test_a_trace_that_is_gone_is_not_offered_as_a_path(self, tmp_path):
        context = KernelContext(
            session_dir=tmp_path,
            workload=WorkloadFacts(model_path="/models/m"),
            evidence=EvidenceIndex(decode_trace=ArtifactRef.of(tmp_path / "never-written.json")),
        )

        assert _fusion(context, tmp_path)["trace_path"] == ""

    def test_the_agent_and_execution_knobs_ride_through_unchanged(self, tmp_path):
        payload = _fusion(_context(tmp_path), tmp_path, max_recipes=3, gpu="2", discover_mode="patterns")

        assert payload["agent_backend"] == "claude"
        assert payload["llm_model"] == "claude-x"
        assert payload["agent_sandbox_mode"] == "workspace-write"
        assert payload["max_turns"] == 100
        assert payload["max_recipes"] == 3
        assert payload["gpu"] == "2"
        assert payload["discover_mode"] == "patterns"
        assert payload["timeout"] == 7200
