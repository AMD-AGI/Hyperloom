# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the rewrite lane tells the opportunity analyst, and what it leaves out."""

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer.session.session_paths import (
    forge_cycle_dir,
    next_forge_attempt_dir,
)
from hyperloom.orchestrator.kernel.forge_handoff import write_forge_handoff
from hyperloom.orchestrator.kernel.kernel_context import (
    ArtifactRef,
    EvidenceIndex,
    KernelContext,
    ServingFacts,
    WorkloadFacts,
)


def _context(session_dir: Path, **overrides) -> KernelContext:
    parts = {
        "workload": WorkloadFacts(
            model_name="example/model",
            model_path="/models/example",
            model_class="decoder",
            precision="fp8",
            quant_type="blockscale",
            gpu_type="mi355x",
            tp=8,
            ep=2,
            isl=1024,
            osl=256,
            conc=64,
            max_model_len=4096,
        ),
        "serving": ServingFacts(
            framework="sglang",
            framework_version="0.5.0",
            server_args="--tp 8 --mem-fraction-static 0.9",
            extra_server_args="--enable-torch-compile",
            extra_envs={"PROFILE_ENV": "enabled", "SAFE_SETTING": "1"},
            unset_envs=("STALE_SETTING",),
        ),
        "evidence": EvidenceIndex(),
    }
    parts.update(overrides)
    return KernelContext(session_dir=session_dir, macro_cycle=3, **parts)


class TestWorkload:
    def test_the_accelerator_and_quantization_are_stated(self, tmp_path):
        # The task contract requires a normalized identity.gpu and forge-loop
        # derives --gpu-target from it, so the analyst must not have to guess.
        workload = write_forge_handoff(_context(tmp_path), tmp_path / "h") / "workload.md"

        text = workload.read_text(encoding="utf-8")
        assert "GPU:** `mi355x`" in text
        assert "Quantization:** `blockscale`" in text
        assert "Precision:** `fp8`" in text

    def test_a_fact_the_session_never_stated_reads_as_absent(self, tmp_path):
        context = _context(tmp_path, workload=WorkloadFacts(model_name="example/model"))

        text = (write_forge_handoff(context, tmp_path / "h") / "workload.md").read_text(encoding="utf-8")

        assert "GPU:** `not available`" in text
        assert "Tensor parallelism:** `not available`" in text


class TestServingContext:
    def test_the_launch_surface_is_reported(self, tmp_path):
        text = (write_forge_handoff(_context(tmp_path), tmp_path / "h") / "serving-context.md").read_text(
            encoding="utf-8"
        )

        assert "--tp 8 --mem-fraction-static 0.9" in text
        assert "--enable-torch-compile" in text
        assert "PROFILE_ENV=enabled" in text
        assert "SAFE_SETTING=1" in text
        assert "STALE_SETTING" in text

    def test_a_sealed_baseline_names_the_commit_a_rewrite_diffs_from(self, tmp_path):
        class _Sealed:
            commit = "abc123"

        context = _context(tmp_path, serving=ServingFacts(framework="sglang", repository_roots=("/src/discovered",)))

        text = (
            write_forge_handoff(context, tmp_path / "h", baselines={"/src/sglang": _Sealed()}) / "serving-context.md"
        ).read_text(encoding="utf-8")

        # The seal outranks discovery: it is what the campaign will diff from.
        assert "- `/src/sglang` @ `abc123`" in text
        assert "/src/discovered" not in text

    def test_without_a_seal_the_discovered_roots_are_still_named(self, tmp_path):
        context = _context(
            tmp_path,
            serving=ServingFacts(framework="sglang", repository_roots=("/src/aiter", "/src/sglang")),
        )

        text = (write_forge_handoff(context, tmp_path / "h") / "serving-context.md").read_text(encoding="utf-8")

        assert "- `/src/aiter` @ `unknown`" in text
        assert "- `/src/sglang` @ `unknown`" in text

    def test_no_repository_at_all_is_said_rather_than_left_blank(self, tmp_path):
        context = _context(tmp_path, serving=ServingFacts(framework="sglang"))

        text = (write_forge_handoff(context, tmp_path / "h") / "serving-context.md").read_text(encoding="utf-8")

        assert "## Source Repositories\n\n- not available" in text


class TestTraceEvidence:
    def test_the_shape_sources_the_analyst_is_told_to_cross_check_are_named(self, tmp_path):
        # Its prompt tells it to check TraceLens against the serving logs and to
        # derive shape cases; both were resolved for the GEMM lane alone before.
        server_log = tmp_path / "server.log"
        manifest = tmp_path / "trace_shape_manifest.json"
        for path in (server_log, manifest):
            path.write_text("{}", encoding="utf-8")
        context = _context(
            tmp_path,
            evidence=EvidenceIndex(
                server_log=ArtifactRef.of(server_log),
                shape_manifest=ArtifactRef.of(manifest),
            ),
        )

        text = (write_forge_handoff(context, tmp_path / "h") / "trace-evidence.md").read_text(encoding="utf-8")

        assert f"Serving log:** `{server_log.resolve()}` (available)" in text
        assert f"Trace shape manifest:** `{manifest.resolve()}` (available)" in text

    def test_a_path_that_is_gone_is_not_the_same_as_one_nobody_produced(self, tmp_path):
        missing = tmp_path / "missing" / "kernel_candidates.json"
        context = _context(tmp_path, evidence=EvidenceIndex(kernel_candidates=ArtifactRef.of(missing)))

        text = (write_forge_handoff(context, tmp_path / "h") / "trace-evidence.md").read_text(encoding="utf-8")

        assert f"`{missing.resolve()}` (missing)" in text
        assert "Profile raw trace:** not provided" in text

    def test_a_trace_health_warning_is_carried_through(self, tmp_path):
        context = _context(
            tmp_path,
            evidence=EvidenceIndex(
                trace_health_warnings=({"code": "partial_trace", "message": "one source was unavailable"},)
            ),
        )

        text = (write_forge_handoff(context, tmp_path / "h") / "trace-evidence.md").read_text(encoding="utf-8")

        assert "- **partial_trace:** one source was unavailable" in text

    def test_no_warnings_says_none(self, tmp_path):
        text = (write_forge_handoff(_context(tmp_path), tmp_path / "h") / "trace-evidence.md").read_text(
            encoding="utf-8"
        )

        assert "## Trace Health Warnings\n\n- none" in text


class TestTheAttemptDirectory:
    def test_the_handoff_rides_inside_the_attempt_that_consumes_it(self, tmp_path):
        attempt = next_forge_attempt_dir(tmp_path / "session", 3)

        written = write_forge_handoff(_context(tmp_path), attempt / "handoff")

        assert written == attempt / "handoff"
        assert (written / "workload.md").is_file()

    def test_each_kernel_entry_gets_its_own_attempt_directory(self, tmp_path):
        """A second KERNEL entry needs an output root the controller has not used."""
        session = tmp_path / "session"

        first = next_forge_attempt_dir(session, 3)
        first.mkdir(parents=True)
        second = next_forge_attempt_dir(session, 3)

        assert first == forge_cycle_dir(session, 3) / "attempt-0"
        assert second == forge_cycle_dir(session, 3) / "attempt-1"
        assert next_forge_attempt_dir(session, 4) == forge_cycle_dir(session, 4) / "attempt-0"

    def test_a_foreign_directory_does_not_disturb_attempt_numbering(self, tmp_path):
        session = tmp_path / "session"
        cycle = forge_cycle_dir(session, 0)
        (cycle / "attempt-0").mkdir(parents=True)
        (cycle / "handoff").mkdir()
        (cycle / "attempt-not-a-number").mkdir()

        assert next_forge_attempt_dir(session, 0) == cycle / "attempt-1"
