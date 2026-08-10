###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""``custom`` reports its own unit and claims no analytic diffusion ceiling.

Scriptable is not the same predicate as diffusion. ``custom`` runs an
entrypoint the operator supplies, so the denoiser config the analytic ceiling
reads is never available -- but the throughput unit its registry entry declares
still is. These lock both halves, and the standalone fallbacks that answer the
same two questions when the ``hyperloom`` package is not importable.
"""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import bypass_trace_analysis as bta  # noqa: E402
import tracelens_analysis as tl  # noqa: E402

from hyperloom.inference_optimizer import framework_registry as fr  # noqa: E402


#: A minimal two-kernel trace, enough for the bypass route to produce candidates.
#: Kept local rather than imported from the neighbouring test module, so this file
#: carries no cross-test import to break under parallel collection.
_BYPASS_TRACE_EVENTS = [
    {"cat": "cpu_op", "name": "aten::paged_attn", "args": {"External id": 100}},
    {"cat": "cpu_op", "name": "aten::mm", "args": {"External id": 200}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 5, "External id": 100}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 7, "External id": 200}},
    {"cat": "kernel", "ph": "X", "name": "paged_attention_v1", "ts": 1000, "dur": 300, "args": {"correlation": 5}},
    {"cat": "kernel", "ph": "X", "name": "Cijk_Alik_Bljk_HHS", "ts": 1300, "dur": 200, "args": {"correlation": 7}},
]


def _without_hyperloom(monkeypatch):
    """Make every ``hyperloom`` import fail, as in a standalone tool invocation."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("hyperloom"):
            raise ImportError("simulated standalone invocation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)


class TestDiffusionCeilingGate:
    def test_custom_claims_no_diffusion_ceiling(self):
        """Hyperloom never sees the operator's model, so it cannot bound it."""
        assert tl._has_diffusion_ceiling("custom") is False

    def test_the_shipped_scriptable_framework_keeps_its_own(self):
        assert tl._has_diffusion_ceiling("xdit") is True

    @pytest.mark.parametrize("framework", ["sglang", "vllm", "atom", "", None, "bogus"])
    def test_non_diffusion_frameworks_have_none(self, framework):
        assert tl._has_diffusion_ceiling(framework) is False

    def test_custom_is_still_scriptable(self):
        """The gate narrows the ceiling only; custom keeps the scriptable route
        (plain pytorch perf report, no decode steady-state splitter)."""
        assert tl._is_scriptable_framework("custom") is True

    def test_custom_stays_excluded_without_the_registry(self, monkeypatch):
        """The name check runs first, so the standalone path cannot re-admit it."""
        _without_hyperloom(monkeypatch)
        assert tl._has_diffusion_ceiling("custom") is False
        assert tl._has_diffusion_ceiling("xdit") is True


def _write_reports_for(tmp_path, framework):
    """Drive ``write_reports`` far enough to reach the diffusion-sidecar gate."""
    from argparse import Namespace

    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    tracelens_dir = run_dir / "tracelens"
    tracelens_dir.mkdir(parents=True, exist_ok=True)
    analysis_md = tracelens_dir / "analysis.md"
    analysis_md.write_text("# TraceLens upstream report\n", encoding="utf-8")
    args = Namespace(
        trace_input=str(trace),
        model_name="whatever-the-operator-shipped",
        framework=framework,
        target_platform="MI300X",
        analysis_mode="inference",
        runtime_env="local",
        dry_run=False,
    )
    return tl.write_reports(
        run_dir,
        trace_input_type="file",
        trace_files=[trace],
        candidates=[],
        args=args,
        existing_report_path=analysis_md,
    )


def _stub_trace_derived_report(monkeypatch):
    """Stand in for the TraceLens aggregation, which needs real perf CSVs.

    Returns only what the trace can produce on its own -- no denoiser config is
    involved -- so what the gate does to the analytic half is observable.
    """
    import diffusion_roofline as dr

    report = {"totals": {"sigma_ideal_roofline_us": 1234.0, "kernel_roofline_efficiency": 0.5}, "gpu_busy_ratio": 0.9}
    monkeypatch.setattr(dr, "build_report", lambda *args, **kwargs: dict(report))
    return report


class TestTheGateIsWiredIn:
    """The helper above is only worth anything if ``write_reports`` consults it."""

    def test_the_analytic_gate_asks_has_diffusion_ceiling(self, tmp_path, monkeypatch):
        """Pins the call site, not just the predicate: swapping the gate back to
        ``_is_scriptable_framework`` leaves this recorder untouched."""
        _stub_trace_derived_report(monkeypatch)
        asked: list[str | None] = []
        monkeypatch.setattr(tl, "_has_diffusion_ceiling", lambda fw: asked.append(fw) or False)
        _write_reports_for(tmp_path, "custom")
        assert asked == ["custom"]

    def test_custom_keeps_the_trace_derived_sidecar(self, tmp_path, monkeypatch):
        """The totals are aggregated from the perf CSVs alone, so withholding the
        analytic ceiling must not withhold them too -- `_scriptable_latency_roofline`
        degrades to `totals.sigma_ideal_roofline_us` when the ceiling is absent."""
        _stub_trace_derived_report(monkeypatch)
        artifacts = _write_reports_for(tmp_path, "custom")
        assert "diffusion_roofline" in artifacts
        emitted = json.loads(Path(artifacts["diffusion_roofline"]).read_text(encoding="utf-8"))
        assert emitted["totals"]["sigma_ideal_roofline_us"] == 1234.0
        assert "analytic_ceiling" not in emitted
        assert "analytic_ceiling_error" not in emitted

    def test_a_serving_framework_gets_no_sidecar_at_all(self, tmp_path, monkeypatch):
        _stub_trace_derived_report(monkeypatch)
        assert "diffusion_roofline" not in _write_reports_for(tmp_path, "sglang")


class TestThroughputUnit:
    @pytest.mark.parametrize("framework", sorted(fr.FRAMEWORKS))
    def test_the_unit_is_whatever_the_registry_declares(self, framework):
        """Asserted against the table itself, so a new entry cannot regress."""
        assert bta._throughput_unit(framework) == fr.throughput_unit(framework)

    def test_custom_is_not_mislabelled_as_tokens(self):
        assert bta._throughput_unit("custom") == "unit/s"

    @pytest.mark.parametrize("framework", ["", None, "bogus"])
    def test_unknown_frameworks_fall_back_to_tokens(self, framework):
        assert bta._throughput_unit(framework) == "tok/s"

    @pytest.mark.parametrize("framework", sorted(fr.FRAMEWORKS))
    def test_the_standalone_fallback_mirrors_the_registry(self, monkeypatch, framework):
        """Two routes answer this question; they must not disagree by environment."""
        expected = fr.throughput_unit(framework)
        _without_hyperloom(monkeypatch)
        assert bta._throughput_unit(framework) == expected


class TestTheTwoRoutesAgree:
    """bypass and TraceLens are two spellings of one feature (`request_handlers`
    picks between them), so a framework must not be scriptable on one and not the
    other -- the sidecar they each emit is the same artifact."""

    @pytest.mark.parametrize("framework", sorted(fr.FRAMEWORKS))
    def test_both_routes_read_scriptable_from_the_registry(self, framework):
        expected = fr.is_scriptable(framework)
        assert bta._is_scriptable_framework(framework) is expected
        assert tl._is_scriptable_framework(framework) is expected

    @pytest.mark.parametrize("framework", sorted(fr.FRAMEWORKS))
    def test_both_standalone_fallbacks_agree_with_the_registry(self, monkeypatch, framework):
        expected = fr.is_scriptable(framework)
        _without_hyperloom(monkeypatch)
        assert bta._is_scriptable_framework(framework) is expected
        assert tl._is_scriptable_framework(framework) is expected

    def _bypass_sidecar_for(self, tmp_path, capsys, framework):
        """Run the bypass route end to end and report whether the sidecar landed."""
        trace = tmp_path / "t.trace.json"
        trace.write_bytes(json.dumps({"traceEvents": _BYPASS_TRACE_EVENTS}).encode("utf-8"))
        argv = [
            "--trace-input",
            str(trace),
            "--session-id",
            f"utest-{framework}",
            "--workspace-path",
            str(tmp_path),
            "--framework",
            framework,
            "--target-platform",
            "MI300X",
            "--model-name",
            "utest",
            "--top-k",
            "8",
        ]
        assert bta.main(argv) == 0
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        assert lines, "no stdout produced"
        result = json.loads(lines[-1])
        return result["artifact_paths"].get("diffusion_roofline")

    def test_the_bypass_route_emits_the_sidecar_for_custom(self, tmp_path, capsys, monkeypatch):
        """Trace-derived on that route too, so a name check would wrongly skip it."""
        monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
        path = self._bypass_sidecar_for(tmp_path, capsys, "custom")
        assert path and Path(path).is_file()
        assert "totals" in json.loads(Path(path).read_text(encoding="utf-8"))

    def test_the_bypass_route_omits_the_sidecar_for_a_serving_framework(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("HYPERLOOM_BYPASS_STEADY_STATE", raising=False)
        assert self._bypass_sidecar_for(tmp_path, capsys, "sglang") is None
