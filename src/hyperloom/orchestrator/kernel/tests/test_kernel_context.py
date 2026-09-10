# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one session view every KERNEL lane is built from."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.kernel import kernel_context as kc
from hyperloom.orchestrator.state.shared_state import SharedState


class TestPrecisionAndQuantResolution:
    """What the runtime is actually serving at, not what the session was told."""

    @staticmethod
    def _write_cfg(model_dir: Path, cfg: dict) -> str:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        return str(model_dir)

    def test_an_explicit_request_wins(self):
        state = SharedState(precision="bf16")
        assert kc.resolve_precision_and_quant(
            state,
            {"precision": "fp8", "quant_type": "blockscale"},
        ) == ("fp8", "blockscale")

    def test_the_running_server_args_outrank_the_session_field(self):
        state = SharedState(precision="bf16")
        state.current_best = {"extra_server_args": "--quantization fp4", "extra_envs": {}}

        assert kc.resolve_precision_and_quant(state, {}) == ("fp4", "fp4")

    def test_a_per_token_env_wins_over_the_checkpoint_format(self):
        state = SharedState(precision="bf16")
        state.current_best = {"extra_server_args": "--quantization fp8", "extra_envs": {}}
        state.reference_envs = {"SGLANG_USE_AITER_FP8_PER_TOKEN": "true"}

        assert kc.resolve_precision_and_quant(state, {}) == ("fp8", "per_token")

    def test_a_plain_checkpoint_under_dynamic_fp8_routes_per_token(self, tmp_path):
        model = self._write_cfg(tmp_path / "plain", {"hidden_size": 2048})
        state = SharedState(precision="bf16", model_path=model)
        state.current_best = {"extra_server_args": "--quantization fp8", "extra_envs": {}}
        assert kc.resolve_precision_and_quant(state, {}) == ("fp8", "per_token")

    def test_a_block_quantized_checkpoint_routes_blockscale(self, tmp_path):
        model = self._write_cfg(
            tmp_path / "block",
            {"hidden_size": 7168, "quantization_config": {"weight_block_size": [128, 128]}},
        )
        state = SharedState(precision="bf16", model_path=model)
        state.current_best = {"extra_server_args": "--quantization fp8", "extra_envs": {}}
        assert kc.resolve_precision_and_quant(state, {}) == ("fp8", "blockscale")

    def test_an_unreadable_config_keeps_auto(self):
        # No readable config: do not force a tuner; let forge sniff the log.
        state = SharedState(precision="bf16", model_path="/models/does-not-exist")
        state.current_best = {"extra_server_args": "--quantization fp8", "extra_envs": {}}
        assert kc.resolve_precision_and_quant(state, {}) == ("fp8", "auto")

    def test_no_session_precision_and_no_quantization_arg_is_bf16(self, monkeypatch):
        state = SharedState(precision="")
        state.current_best = {"extra_server_args": "", "extra_envs": {}}
        import hyperloom.orchestrator.kernel.roofline_ceiling as rc

        def _raise(*_a, **_k):
            raise RuntimeError("no runtime workload")

        monkeypatch.setattr(rc, "resolve_runtime_workload", _raise)
        assert kc.resolve_precision_and_quant(state, {}) == ("bf16", "auto")


class TestArtifactRef:
    def test_a_path_that_is_there_is_available(self, tmp_path):
        present = tmp_path / "trace.json"
        present.write_text("{}", encoding="utf-8")

        ref = kc.ArtifactRef.of(present)

        assert ref.path == str(present.resolve())
        assert ref.available is True
        assert ref.usable == ref.path

    def test_a_path_nobody_produced_is_not_the_same_as_one_that_vanished(self, tmp_path):
        # Both are unusable, and a lane that has to report why needs to tell them apart.
        absent = kc.ArtifactRef.of("")
        vanished = kc.ArtifactRef.of(tmp_path / "gone.json")

        assert (absent.path, absent.available) == ("", False)
        assert vanished.path and vanished.available is False
        assert absent.usable == vanished.usable == ""


class TestFactResolutionOrder:
    """Request, then the live profile, then the session -- and never the environment."""

    def test_the_request_outranks_the_session(self, tmp_path):
        state = SharedState(tp=1, conc=8, model_path=str(tmp_path))

        facts = kc.build_workload_facts(state, overrides={"tp": 8, "conc": 64})

        assert (facts.tp, facts.conc) == (8, 64)

    def test_an_unstated_fact_stays_absent_rather_than_defaulted(self, tmp_path):
        # The handoff has to be able to report "not available"; a lane that
        # needs a concrete value substitutes its own.
        state = SharedState(model_path=str(tmp_path))

        facts = kc.build_workload_facts(state)

        assert (facts.tp, facts.conc, facts.gpu_type) == (0, 0, "")

    def test_the_environment_is_not_a_source(self, tmp_path, monkeypatch):
        # SharedState already carries these; a second source is how the lanes
        # came to disagree in the first place.
        for name, value in (("TP", "8"), ("CONC", "64"), ("GPU_TYPE", "mi355x"), ("MODEL_PATH", "/from/env")):
            monkeypatch.setenv(name, value)
        state = SharedState(model_path=str(tmp_path))

        facts = kc.build_workload_facts(state)

        assert (facts.tp, facts.conc, facts.gpu_type) == (0, 0, "")
        assert facts.model_path == str(tmp_path)

    def test_the_gpu_type_is_normalized_lowercase(self, tmp_path):
        # The controller's task contract requires ``"gpu": "mi355x"``, never "MI355X".
        state = SharedState(gpu_type="MI355X", model_path=str(tmp_path))

        assert kc.build_workload_facts(state).gpu_type == "mi355x"


class TestEvidenceIndexing:
    def test_an_absent_artifact_is_indexed_as_unavailable(self, tmp_path):
        state = SharedState()

        evidence = kc.build_evidence_index(state, tmp_path, workload=kc.WorkloadFacts())

        assert evidence.profile_trace.available is False
        assert evidence.server_log.usable == ""
        assert evidence.trace_health_warnings == ()

    def test_the_source_resolution_sits_beside_the_candidates_file(self, tmp_path):
        candidates = tmp_path / "tracelens" / "kernel_candidates.json"
        candidates.parent.mkdir(parents=True)
        candidates.write_text("[]", encoding="utf-8")
        resolution = candidates.parent / "kernel_source_resolution.json"
        resolution.write_text("{}", encoding="utf-8")
        state = SharedState()
        state.last_trace_analyze = {"candidates_path": str(candidates)}

        evidence = kc.build_evidence_index(state, tmp_path, workload=kc.WorkloadFacts())

        assert evidence.kernel_candidates.available is True
        assert evidence.kernel_source_resolution.path == str(resolution.resolve())
        assert evidence.kernel_source_resolution.available is True


class TestPersistedContext:
    def test_the_context_is_written_beside_the_run_it_fed(self, tmp_path):
        context = kc.KernelContext(
            session_dir=tmp_path,
            macro_cycle=2,
            workload=kc.WorkloadFacts(gpu_type="mi355x", tp=8),
        )

        path = kc.write_kernel_context(context, tmp_path / "attempt")

        assert path.name == kc.CONTEXT_FILENAME
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["macro_cycle"] == 2
        assert written["workload"]["gpu_type"] == "mi355x"
        assert written["session_dir"] == str(tmp_path)
