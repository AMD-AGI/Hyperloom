###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""``custom`` reports its own unit and claims no analytic diffusion ceiling.

Scriptable is not the same predicate as diffusion. ``custom`` runs an
entrypoint the operator supplies, so the denoiser config the analytic ceiling
reads is never available. These lock the diffusion-ceiling gate and the
scriptable classification, and the standalone fallbacks that answer them when
the ``hyperloom`` package is not importable.
"""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import tracelens_analysis as tl  # noqa: E402

from hyperloom.inference_optimizer import framework_registry as fr  # noqa: E402


def _without_hyperloom(monkeypatch):
    """Make every ``hyperloom`` import fail, as in a standalone tool invocation."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("hyperloom"):
            raise ImportError("simulated standalone invocation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)


class TestDiffusionCeilingGate:
    @pytest.mark.parametrize("framework", sorted(fr.FRAMEWORKS))
    def test_the_gate_is_whatever_the_registry_declares(self, framework):
        """Asserted against the table itself, so a new entry is classified when it
        is added rather than when someone remembers this call site."""
        assert tl._has_diffusion_ceiling(framework) is fr.has_denoiser_config(framework)

    def test_custom_claims_no_diffusion_ceiling(self):
        """Hyperloom never sees the operator's model, so it cannot bound it."""
        assert tl._has_diffusion_ceiling("custom") is False

    def test_the_shipped_scriptable_framework_keeps_its_own(self):
        assert tl._has_diffusion_ceiling("xdit") is True

    @pytest.mark.parametrize("framework", ["", None, "bogus"])
    def test_unknown_frameworks_claim_none(self, framework):
        assert tl._has_diffusion_ceiling(framework) is False

    def test_custom_is_still_scriptable(self):
        """The gate narrows the ceiling only; custom keeps the scriptable route
        (plain pytorch perf report, no decode steady-state splitter)."""
        assert tl._is_scriptable_framework("custom") is True

    @pytest.mark.parametrize("framework", sorted(fr.FRAMEWORKS))
    def test_the_standalone_fallback_mirrors_the_registry(self, monkeypatch, framework):
        expected = fr.has_denoiser_config(framework)
        _without_hyperloom(monkeypatch)
        assert tl._has_diffusion_ceiling(framework) is expected


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


class TestScriptableReadsFromRegistry:
    @pytest.mark.parametrize("framework", sorted(fr.FRAMEWORKS))
    def test_scriptable_is_read_from_the_registry(self, framework):
        expected = fr.is_scriptable(framework)
        assert tl._is_scriptable_framework(framework) is expected

    @pytest.mark.parametrize("framework", sorted(fr.FRAMEWORKS))
    def test_the_standalone_fallback_agrees_with_the_registry(self, monkeypatch, framework):
        expected = fr.is_scriptable(framework)
        _without_hyperloom(monkeypatch)
        assert tl._is_scriptable_framework(framework) is expected
