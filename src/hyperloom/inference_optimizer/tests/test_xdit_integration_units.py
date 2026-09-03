# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the xDiT (scriptable diffusion) framework integration.

Covers the cross-cutting contracts for xDiT: the framework registry, the
server-args env resolver, the do-not-set blacklist + compatibility filter, the
scriptable quality gate, scriptable measurement validity, the per-framework
YAML resolvers, the explore cold-start grid, the roofline snapshot units and
their latency sidecar, the TraceLens arch spec, and scriptable trace health.
"""

from __future__ import annotations

import json

import pytest

from hyperloom.inference_optimizer import framework_registry as fr
from hyperloom.orchestrator.actions.executors import _accuracy_gate as ag
from hyperloom.orchestrator.actions.executors import benchmark_result as br
from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors import explore as ex


class TestFrameworkRegistry:
    def test_xdit_registered_and_scriptable(self):
        assert "xdit" in fr.names()
        assert fr.is_supported("XDiT") is True
        assert fr.is_scriptable("xdit") is True
        assert fr.FRAMEWORKS["xdit"].kind == fr.SCRIPTABLE

    def test_serving_frameworks_not_scriptable(self):
        for name in ("sglang", "vllm", "atom"):
            assert fr.is_scriptable(name) is False
            assert fr.FRAMEWORKS[name].kind == fr.SERVING

    def test_extra_args_env_and_unit(self):
        assert fr.extra_args_env("xdit") == "EXTRA_XDIT_ARGS"
        assert fr.extra_args_env("sglang") == "EXTRA_SGLANG_ARGS"
        assert fr.throughput_unit("xdit") == "img/s"
        assert fr.throughput_unit("vllm") == "tok/s"

    def test_registry_capability_fields(self):
        assert fr.FRAMEWORKS["xdit"].supports_server_reuse is False
        assert fr.FRAMEWORKS["sglang"].supports_server_reuse is True
        assert fr.FRAMEWORKS["xdit"].repo_url == "https://github.com/xdit-project/xDiT.git"

    def test_unknown_falls_back_to_default(self):
        assert fr.extra_args_env("rust-burn") == fr.extra_args_env(fr.DEFAULT_FRAMEWORK)
        assert fr.is_scriptable("rust-burn") is False


class TestFormatPrimaryMetric:
    def test_serving_shows_tok_s_per_gpu(self):
        assert fr.format_primary_metric("sglang", 123.4) == "123.4 tok/s/GPU"
        assert fr.format_primary_metric("vllm", 0.0) == "0.0 tok/s/GPU"

    def test_xdit_shows_e2el_mean_ms_not_tok_s(self):
        # xDiT throughput is img/s; display equivalent per-image latency ms.
        out = fr.format_primary_metric("xdit", 0.15528)
        assert out == "6440.0 ms"
        assert "tok/s" not in out

    def test_xdit_non_positive_is_na_ms(self):
        assert fr.format_primary_metric("xdit", 0.0) == "n/a ms"
        assert fr.format_primary_metric("xdit", None) == "n/a ms"

    def test_unknown_framework_defaults_to_serving_label(self):
        assert fr.format_primary_metric("rust-burn", 10.0) == "10.0 tok/s/GPU"

    def test_none_or_empty_framework_falls_back_to_serving(self):
        # A None/empty framework must not crash and defaults to serving unit.
        assert fr.primary_metric_unit(None) == "tok/s/GPU"
        assert fr.primary_metric_unit("") == "tok/s/GPU"
        assert fr.primary_metric_value(None, 12.0) == 12.0
        assert fr.format_primary_metric(None, 12.0) == "12.0 tok/s/GPU"
        assert fr.format_primary_metric("", 12.0) == "12.0 tok/s/GPU"
        assert fr.format_primary_metric(None, None) == "0.0 tok/s/GPU"


class TestServerArgsEnvName:
    def test_exact(self):
        assert gr.server_args_env_name("xdit") == "EXTRA_XDIT_ARGS"
        assert gr.server_args_env_name("atom") == "EXTRA_ATOM_ARGS"
        assert gr.server_args_env_name("vllm") == "EXTRA_VLLM_ARGS"
        assert gr.server_args_env_name(None) == "EXTRA_SGLANG_ARGS"

    def test_version_suffix_substring(self):
        assert gr.server_args_env_name("vllm@0.21") == "EXTRA_VLLM_ARGS"
        assert gr.server_args_env_name("xdit-rocm") == "EXTRA_XDIT_ARGS"


class TestXditBlacklist:
    def test_precision_lock_and_quantized_attn(self):
        assert gr.xdit_blacklist_reason({"XDIT_USE_FP4_GEMMS": "1"}) is not None
        assert gr.xdit_blacklist_reason({"XDIT_USE_FP8_GEMMS": "1"}) is not None
        assert gr.xdit_blacklist_reason({"XDIT_ATTENTION_BACKEND": "aiter_fp8"}) is not None
        assert gr.xdit_blacklist_reason({"RCCL_MSCCL_ENABLE": "1"}) is not None

    def test_combo_crash(self):
        reason = gr.xdit_blacklist_reason({"AMD_DIRECT_DISPATCH": "1", "AMDGCN_USE_BUFFER_OPS": "1"})
        assert reason is not None

    def test_safe_envs_pass(self):
        assert gr.xdit_blacklist_reason({"XDIT_ATTENTION_BACKEND": "aiter"}) is None
        assert gr.xdit_blacklist_reason({"AMDGCN_USE_BUFFER_OPS": "1"}) is None
        assert gr.xdit_blacklist_reason({"XDIT_USE_FP4_GEMMS": "0"}) is None
        assert gr.xdit_blacklist_reason({}) is None

    def test_compatibility_filter_drops_blacklisted(self):
        good = gr.GridVariant(name="xdit_ok", extra_server_args="", extra_envs={"XDIT_ATTENTION_BACKEND": "aiter"})
        bad = gr.GridVariant(name="xdit_fp4", extra_server_args="", extra_envs={"XDIT_USE_FP4_GEMMS": "1"})
        kept, dropped = gr.apply_compatibility_filter([good, bad], framework="xdit", model_path="")
        kept_names = {v.name for v in kept}
        dropped_names = {d["name"] for d in dropped}
        assert "xdit_ok" in kept_names
        assert "xdit_fp4" in dropped_names
        assert any(d["source"] == "xdit_blacklist" for d in dropped)


class TestQualityGate:
    def test_quality_gate_passed_explicit(self):
        assert ag.quality_gate_passed({"passed": True}) is True
        assert ag.quality_gate_passed({"passed": False}) is False
        # Serving (require=False): a missing/empty gate does not block.
        assert ag.quality_gate_passed(None) is True
        assert ag.quality_gate_passed({}) is True

    def test_quality_gate_passed_required_fails_closed(self):
        # Scriptable (require=True): missing/empty/ambiguous gate fails closed.
        assert ag.quality_gate_passed(None, require=True) is False
        assert ag.quality_gate_passed({}, require=True) is False
        # Non-empty gate with no passed and no usable thresholds fails when required.
        assert ag.quality_gate_passed({"note": "n/a"}, require=True) is False
        # An explicit pass / usable thresholds still pass when required.
        assert ag.quality_gate_passed({"passed": True}, require=True) is True
        assert ag.quality_gate_passed({"lpips": 0.01, "lpips_max": 0.05}, require=True) is True

    def test_quality_gate_passed_skipped_reference_established(self, monkeypatch):
        # The baseline establishing the reference (skipped) must pass even when
        # required and a reference is configured.
        monkeypatch.setenv("XDIT_QUALITY_REF", "/tmp/ref.png")
        gate = {"passed": True, "skipped": True, "reason": "reference_established"}
        assert ag.quality_gate_passed(gate, require=True) is True

    def test_quality_gate_passed_skipped_fails_closed_when_ref_configured(self, monkeypatch):
        # A variant that SKIPPED the gate while a reference was configured did
        # not actually compare -> fail closed (scriptable require=True).
        monkeypatch.setenv("XDIT_QUALITY_REF", "/tmp/ref.png")
        for reason in ("no_reference_or_image", "reference_missing", "image_libs_unavailable"):
            gate = {"passed": True, "skipped": True, "reason": reason}
            assert ag.quality_gate_passed(gate, require=True) is False, reason

    def test_quality_gate_passed_skipped_fails_closed_regardless_of_env(self, monkeypatch):
        # A comparison is always expected, so a non-established skip is
        # unverifiable and fails closed even when the process env carries no
        # XDIT_QUALITY_REF.
        monkeypatch.delenv("XDIT_QUALITY_REF", raising=False)
        gate = {"passed": True, "skipped": True, "reason": "no_reference_or_image"}
        assert ag.quality_gate_passed(gate, require=True) is False
        # Serving (require=False) never fails closed on a skip.
        assert ag.quality_gate_passed(gate, require=False) is True

    def test_quality_gate_passed_thresholds(self):
        ok = {"lpips": 0.01, "lpips_max": 0.05, "ssim": 0.99, "ssim_min": 0.95, "mse": 0.001, "mse_max": 0.002}
        bad = {"lpips": 0.10, "lpips_max": 0.05}
        assert ag.quality_gate_passed(ok) is True
        assert ag.quality_gate_passed(bad) is False

    def test_parse_quality_gate(self, tmp_path):
        report = tmp_path / "benchmark_report.json"
        report.write_text(json.dumps({"quality_gate": {"passed": True, "ssim": 0.97}}), encoding="utf-8")
        out = ag.parse_quality_gate(tmp_path)
        assert out["quality_gate"]["passed"] is True

    def test_parse_eval_results_maps_quality_to_accuracy(self, tmp_path):
        report = tmp_path / "benchmark_report.json"
        report.write_text(json.dumps({"quality_gate": {"passed": False}}), encoding="utf-8")
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] == 0.0
        assert out["task"] == "quality_gate"

    def test_parse_eval_results_quality_pass(self, tmp_path):
        report = tmp_path / "benchmark_report.json"
        report.write_text(json.dumps({"quality_gate": {"passed": True}}), encoding="utf-8")
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] == 1.0

    def test_parse_eval_results_scriptable_missing_gate_fails_closed(self, tmp_path):
        # No quality_gate: scriptable must fail closed (accuracy 0.0), not GSM8K.
        out = ag.parse_eval_results(tmp_path, framework="xdit")
        assert out["accuracy"] == 0.0
        assert out["task"] == "quality_gate"
        assert "error" in out

    def test_parse_eval_results_scriptable_report_without_gate_fails_closed(self, tmp_path):
        report = tmp_path / "benchmark_report.json"
        report.write_text(json.dumps({"output_throughput": 0.42}), encoding="utf-8")
        out = ag.parse_eval_results(tmp_path, framework="xdit")
        assert out["accuracy"] == 0.0
        assert out["task"] == "quality_gate"

    def test_parse_eval_results_scriptable_gate_without_passed_fails_closed(self, tmp_path):
        report = tmp_path / "benchmark_report.json"
        report.write_text(json.dumps({"quality_gate": {"ssim": 0.97}}), encoding="utf-8")
        out = ag.parse_eval_results(tmp_path, framework="xdit")
        assert out["accuracy"] == 0.0

    def test_parse_eval_results_serving_missing_gate_skips(self, tmp_path):
        # Serving with no gate/GSM8K: returns accuracy None so the caller skips.
        out = ag.parse_eval_results(tmp_path, framework="vllm")
        assert out.get("accuracy") is None


class TestScriptableMeasurement:
    def test_valid_without_completed_requests(self):
        m = {"workload_kind": "scriptable", "output_throughput": 0.29, "completed_requests": None}
        assert br.is_valid_measurement(m) is True

    def test_quality_fail_rejected(self):
        m = {
            "workload_kind": "scriptable",
            "output_throughput": 0.29,
            "quality_gate": {"passed": False},
        }
        assert br.is_valid_measurement(m) is False

    def test_quality_threshold_fail_rejected_without_passed_key(self):
        # A gate that fails on thresholds (no explicit passed=False) is rejected.
        m = {
            "workload_kind": "scriptable",
            "output_throughput": 0.29,
            "quality_gate": {"lpips": 0.40, "lpips_max": 0.05},
        }
        assert br.is_valid_measurement(m) is False

    def test_quality_missing_gate_still_valid(self):
        # A missing/empty gate stays non-blocking for selection (require=False);
        # required-gate enforcement happens upstream.
        m = {"workload_kind": "scriptable", "output_throughput": 0.29}
        assert br.is_valid_measurement(m) is True

    def test_serving_still_requires_completed(self):
        m = {"framework": "sglang", "output_throughput": 100.0, "completed_requests": None}
        assert br.is_valid_measurement(m) is False

    def test_extract_carries_quality_gate(self):
        report = {
            "success": True,
            "framework": "xdit",
            "workload_kind": "scriptable",
            "throughput_unit": "img/s",
            "quality_gate": {"passed": True},
            "throughput": {"output_throughput": 0.29, "completed_requests": 25},
        }
        m = br.extract_benchmark_measurement(report)
        assert m["workload_kind"] == "scriptable"
        assert m["quality_gate"] == {"passed": True}
        assert m["valid_measurement"] is True


class TestConfigResolvers:
    def test_baseline_config_xdit(self, monkeypatch):
        from hyperloom.orchestrator.actions.executors import _workload_envs as we

        monkeypatch.setenv("FRAMEWORK", "xdit")
        assert we.default_baseline_config().name == "baseline_xdit.yaml"

    def test_profile_config_xdit(self, monkeypatch):
        from hyperloom.orchestrator.actions.executors import profile as pf

        monkeypatch.setenv("FRAMEWORK", "xdit")
        assert pf._default_profile_config().name == "profile_xdit.yaml"


class TestExploreGrid:
    def test_xdit_grid_non_empty_and_safe(self):
        grid = ex._default_grid_for_framework("xdit", model_class="dit", conc=1)
        assert grid, "xdit cold-start grid must be non-empty"
        for v in grid:
            assert v.name.startswith("xdit_")
            assert gr.xdit_blacklist_reason(v.extra_envs) is None


class TestRegistryRepoUrlConsistency:
    """Guard against repo_url drift between framework_registry and repo_map."""

    def test_registry_urls_match_repo_map(self):
        try:
            from hyperloom.agents.framework.repo_map import _FRAMEWORK_TO_REPO_URL
        except ImportError:
            pytest.skip("hyperloom.agents.framework not installed")
        for name, spec in fr.FRAMEWORKS.items():
            if spec.repo_url is not None:
                assert spec.repo_url == _FRAMEWORK_TO_REPO_URL.get(name, ""), (
                    f"repo_url mismatch for {name}: "
                    f"registry={spec.repo_url!r} vs repo_map={_FRAMEWORK_TO_REPO_URL.get(name)!r}"
                )


class TestLifecycleScriptableSkip:
    """Verify server_lifecycle correctly skips scriptable frameworks."""

    def test_scriptable_ineligible(self, tmp_path):
        import yaml

        cfg = {
            "benchmark": {
                "framework": "xdit",
                "workload_kind": "scriptable",
                "benchmark_script": "xdit_mi355x.sh",
            }
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        from hyperloom.orchestrator.actions.executors._server_lifecycle import (
            resolve_lifecycle_params,
        )

        result = resolve_lifecycle_params(cfg_file)
        assert result["eligible"] is False
        assert "scriptable" in result["reason"].lower()

    def test_serving_eligible_with_builtin_script(self, tmp_path):
        import yaml

        cfg = {
            "benchmark": {
                "framework": "sglang",
                "benchmark_script": "sglang_mi355x.sh",
            }
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        from hyperloom.orchestrator.actions.executors._server_lifecycle import (
            resolve_lifecycle_params,
        )

        result = resolve_lifecycle_params(cfg_file)
        assert result["eligible"] is True


class TestRooflineSnapshotUnits:
    """The roofline snapshot table renders the achieved primary metric in the
    framework-correct unit (serving tok/s vs scriptable per-image ms)."""

    def test_fmt_tput_serving_tok_s(self):
        from hyperloom.orchestrator.kernel import roofline_snapshot as rs

        assert rs._fmt_tput(123.0, "vllm") == "123.0 tok/s"
        assert rs._fmt_tput(None, "vllm") == "—"

    def test_fmt_tput_scriptable_renders_latency_ms(self):
        from hyperloom.orchestrator.kernel import roofline_snapshot as rs

        out = rs._fmt_tput(0.15528, "xdit")
        assert out == "6440.0 ms"
        assert "tok/s" not in out

    def test_build_snapshot_carries_framework(self):
        from hyperloom.orchestrator.kernel import roofline_snapshot as rs

        snap = rs.build_roofline_snapshot(
            snapshot_id=1, ts="t", analysis_md_path="", achieved_tok_per_sec=0.155, framework="xdit"
        )
        assert snap["framework"] == "xdit"

    def test_metrics_table_scriptable_achieved_is_ms(self):
        from hyperloom.orchestrator.kernel import roofline_snapshot as rs

        snap = rs.build_roofline_snapshot(
            snapshot_id=1, ts="t", analysis_md_path="", achieved_tok_per_sec=0.15528, framework="xdit"
        )
        cmp = rs.build_roofline_comparison_from_history([snap])
        table = "\n".join(rs.format_roofline_metrics_table(cmp))
        assert "6440.0 ms" in table
        assert "decode memory-roofline ceiling" not in table

    def test_snapshot_carries_latency_siblings_and_within(self):
        """e2e_mean_ms / roofline_ideal_ms are stored at the tok/s level and
        drive a unit-agnostic within/gap when no decode ceiling applies."""
        from hyperloom.orchestrator.kernel import roofline_snapshot as rs

        snap = rs.build_roofline_snapshot(
            snapshot_id=1,
            ts="t",
            analysis_md_path="",
            achieved_tok_per_sec=0.15528,  # img/s
            framework="xdit",
            e2e_mean_ms=6440.0,
            roofline_ideal_ms=644.0,
        )
        assert snap["e2e_mean_ms"] == 6440.0
        assert snap["roofline_ideal_ms"] == 644.0
        # No tok/s ceiling -> within = ideal / measured.
        assert snap["within_roofline_pct"] == 10.0
        assert snap["gap_to_roofline_pct"] == 90.0
        assert snap["theoretical_peak_tok_per_sec"] is None

    def test_metrics_table_scriptable_shows_compute_ceiling(self):
        """The compact table surfaces the ms compute-roofline floor + within%."""
        from hyperloom.orchestrator.kernel import roofline_snapshot as rs

        snap = rs.build_roofline_snapshot(
            snapshot_id=1,
            ts="t",
            analysis_md_path="",
            achieved_tok_per_sec=0.15528,
            framework="xdit",
            e2e_mean_ms=6440.0,
            roofline_ideal_ms=644.0,
        )
        cmp = rs.build_roofline_comparison_from_history([snap])
        table = "\n".join(rs.format_roofline_metrics_table(cmp))
        assert "Compute-roofline ideal (per-image latency floor):" in table
        assert "644.0 ms" in table
        assert "10.0%" in table
        assert "decode memory-roofline ceiling" not in table

    def test_serving_snapshot_latency_siblings_are_none(self):
        """Serving snapshots keep tok/s within/gap and leave ms siblings unset."""
        from hyperloom.orchestrator.kernel import roofline_snapshot as rs

        snap = rs.build_roofline_snapshot(
            snapshot_id=1,
            ts="t",
            analysis_md_path="",
            theoretical_peak_tok_per_sec=200.0,
            achieved_tok_per_sec=150.0,
            framework="vllm",
        )
        assert snap["e2e_mean_ms"] is None
        assert snap["roofline_ideal_ms"] is None
        assert snap["within_roofline_pct"] == 75.0


class TestScriptableLatencyRooflineSidecar:
    """``_scriptable_latency_roofline`` must find the diffusion sidecar even when
    ``kernel_roofline_path`` is empty (diffusion trace_analyze emits only
    ``diffusion_roofline.json``, so the run-dir cannot be derived from it)."""

    def _make_state(self, tmp_path):
        from hyperloom.orchestrator.state.shared_state import SharedState

        return SharedState.load_or_init(tmp_path)

    def _write_sidecar(self, session_dir, *, sigma_ideal_us=611192.0, analytic_us=None):
        run_dir = session_dir / "kernel-agent" / "runs" / "20260705T193725Z" / "20260705T200313Z_tl-abc"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {"totals": {"sigma_ideal_roofline_us": sigma_ideal_us}}
        if analytic_us is not None:
            payload["analytic_dit_ceiling"] = {"ideal_compute_us": analytic_us}
        sidecar = run_dir / "diffusion_roofline.json"
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        return sidecar

    def test_empty_kernel_roofline_path_falls_back_to_session_dir(self, tmp_path):
        state = self._make_state(tmp_path)
        self._write_sidecar(tmp_path)
        # kernel_roofline_path empty (the diffusion case) must still resolve.
        e2e_ms, ideal_ms = state._scriptable_latency_roofline("xdit", 0.740741, "")
        assert e2e_ms == pytest.approx(1350.0, abs=1.0)
        assert ideal_ms == pytest.approx(611.192, abs=0.01)

    def test_analytic_ceiling_preferred_over_totals(self, tmp_path):
        state = self._make_state(tmp_path)
        self._write_sidecar(tmp_path, sigma_ideal_us=611192.0, analytic_us=644000.0)
        _e2e, ideal_ms = state._scriptable_latency_roofline("xdit", 0.740741, "")
        assert ideal_ms == pytest.approx(644.0, abs=0.01)

    def test_missing_sidecar_yields_zero_ideal(self, tmp_path):
        state = self._make_state(tmp_path)
        e2e_ms, ideal_ms = state._scriptable_latency_roofline("xdit", 0.740741, "")
        assert e2e_ms > 0
        assert ideal_ms == 0.0

    def test_serving_framework_is_noop(self, tmp_path):
        state = self._make_state(tmp_path)
        self._write_sidecar(tmp_path)
        assert state._scriptable_latency_roofline("vllm", 100.0, "") == (0.0, 0.0)

    def test_explicit_kernel_roofline_path_still_resolves(self, tmp_path):
        state = self._make_state(tmp_path)
        run_dir = tmp_path / "kernel-agent" / "runs" / "ts" / "ts_tl-xyz"
        (run_dir / "reports").mkdir(parents=True, exist_ok=True)
        (run_dir / "diffusion_roofline.json").write_text(
            json.dumps({"totals": {"sigma_ideal_roofline_us": 500000.0}}), encoding="utf-8"
        )
        krp = str(run_dir / "reports" / "kernel_roofline.json")
        _e2e, ideal_ms = state._scriptable_latency_roofline("xdit", 0.740741, krp)
        assert ideal_ms == pytest.approx(500.0, abs=0.01)


class TestHyperloomArchSpec:
    """TraceLens arch spec derived from hyperloom's HW_SPECS_ACHIEVABLE."""

    def _tab(self):
        import sys
        from pathlib import Path

        tool_dir = Path(__file__).resolve().parents[4] / "src" / "hyperloom" / "agents" / "kernel" / "tools"
        if str(tool_dir) not in sys.path:
            sys.path.insert(0, str(tool_dir))
        import tracelens_arch_benchmark as tab  # noqa: WPS433

        return tab

    def test_build_spec_mi355x(self):
        tab = self._tab()
        spec = tab.build_hyperloom_arch_spec("mi355x")
        assert spec is not None
        assert spec["mem_bw_gbps"] == pytest.approx(8000.0)
        maf = spec["max_achievable_tflops"]
        assert maf["matrix_bf16"] == pytest.approx(1686.0)
        assert maf["matrix_fp8"] == pytest.approx(3567.0)
        assert maf["matrix_fp4"] == pytest.approx(5663.0)
        assert all(v > 0 for v in maf.values())

    def test_build_spec_case_insensitive_and_named(self):
        tab = self._tab()
        spec = tab.build_hyperloom_arch_spec("MI300X")
        assert spec is not None
        assert "matrix_bf16" in spec["max_achievable_tflops"]

    def test_build_spec_unknown_platform_none(self):
        tab = self._tab()
        assert tab.build_hyperloom_arch_spec("nvidia-h100") is None

    def test_write_spec_roundtrip(self, tmp_path):
        tab = self._tab()
        out = tab.write_hyperloom_arch_spec(tmp_path, "mi355x", lambda _m: None)
        assert out is not None and out.is_file()

        data = json.loads(out.read_text())
        assert data["max_achievable_tflops"]["matrix_bf16"] == pytest.approx(1686.0)


class TestValidateTraceStructureScriptable:
    """For scriptable (xDiT) traces, the LLM/InferenceX structure checks are
    skipped; only the zero-ops (repeat=0 empty window) health signal applies."""

    def _write_trace(self, trace_dir, *, with_kernels: bool, with_annotations: bool = False) -> None:
        import gzip

        if with_kernels:
            # Healthy diffusion trace: cpu_op + kernel, no execute_*/user_annotation.
            events = [{"name": "cpu_op", "cat": "cpu_op"}, {"name": "some_gemm", "cat": "kernel"}]
        else:
            # Metadata-only (repeat=0 empty window) trace: no cpu_op / kernel.
            events = [{"name": "process_labels", "cat": "process_labels"}]
        if with_annotations:
            events.append({"name": "step[DECODE bs=8]", "cat": "user_annotation"})
        payload = {"traceEvents": events}
        p = trace_dir / "profile.trace.json.gz"
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps(payload))

    def test_scriptable_no_attribution_degradation(self, tmp_path):
        from hyperloom.orchestrator.actions.executors import profile as pf

        self._write_trace(tmp_path, with_kernels=True)
        health = pf._validate_trace_structure(tmp_path, "xdit")
        # No capture-dir warning, no execute_* degradation for scriptable.
        assert health["per_kernel_attribution_degraded"] is False
        assert health["zero_ops"] is False
        assert not any("capture_traces" in i for i in health["issues"])
        assert not any("[3]" in i for i in health["issues"])

    def test_scriptable_zero_ops_still_flagged(self, tmp_path):
        from hyperloom.orchestrator.actions.executors import profile as pf

        self._write_trace(tmp_path, with_kernels=False)
        health = pf._validate_trace_structure(tmp_path, "xdit")
        # The diffusion-relevant health signal is preserved.
        assert health["zero_ops"] is True

    def test_serving_still_flags_missing_annotations(self, tmp_path):
        from hyperloom.orchestrator.actions.executors import profile as pf

        self._write_trace(tmp_path, with_kernels=True)
        health = pf._validate_trace_structure(tmp_path, "vllm")
        # Same trace, serving framework: the LLM checks run and flag the missing
        # execute_*/user_annotation events.
        assert health["per_kernel_attribution_degraded"] is True

    def test_serving_sees_annotations_recorded_under_cat(self, tmp_path):
        """The annotation category lives in ``cat``; ``name`` holds the label.

        A marker keyed on ``"name": "user_annotation"`` therefore matches a label
        no producer writes: 53 of the 62 reference captures were reported as
        carrying no annotations while each held several hundred, and the zero-hot-
        kernel path then blamed CUDA-graph folding for an unrelated result.
        """
        from hyperloom.orchestrator.actions.executors import profile as pf

        self._write_trace(tmp_path, with_kernels=True, with_annotations=True)

        health = pf._validate_trace_structure(tmp_path, "sglang")

        assert health["per_kernel_attribution_degraded"] is False
        assert not any("[3]" in i for i in health["issues"])

    def test_empty_framework_falls_back_to_session_framework(self, monkeypatch, tmp_path):
        """An unset framework must not be treated as serving.

        Session 20260803T134328Z: the roofline-composite ctx carries no
        framework, so the scriptable profile leg was validated as serving and
        reported the two serving-only issues ([1] capture_traces/ missing and
        [3] no execute_*), each pointing at EXTRA_VLLM_ARGS / EXTRA_SGLANG_ARGS
        that a scriptable framework never sets.
        """
        from hyperloom.orchestrator.actions.executors import profile as pf

        self._write_trace(tmp_path, with_kernels=True)
        monkeypatch.setenv("FRAMEWORK", "custom")

        health = pf._validate_trace_structure(tmp_path, "")

        assert not any("capture_traces" in i for i in health["issues"])
        assert not any("[3]" in i for i in health["issues"])
        assert health["per_kernel_attribution_degraded"] is False


class TestQualityGateReportSelection:
    """Verify parse_quality_gate picks the most recently modified report."""

    def test_mtime_preferred_over_name_sort(self, tmp_path):
        import time

        sub_a = tmp_path / "aaa"
        sub_a.mkdir()
        sub_z = tmp_path / "zzz"
        sub_z.mkdir()

        report_a = sub_a / "benchmark_report.json"
        report_a.write_text(json.dumps({"quality_gate": {"passed": False}}), encoding="utf-8")
        time.sleep(0.05)
        report_z = sub_z / "benchmark_report.json"
        report_z.write_text(json.dumps({"quality_gate": {"passed": True}}), encoding="utf-8")

        out = ag.parse_quality_gate(tmp_path)
        assert out["quality_gate"]["passed"] is True
        assert "zzz" in out["source_file"]
