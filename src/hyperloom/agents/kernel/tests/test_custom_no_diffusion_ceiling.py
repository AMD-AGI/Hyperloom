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
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import bypass_trace_analysis as bta  # noqa: E402
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
    def test_custom_claims_no_diffusion_ceiling(self):
        """Hyperloom never sees the operator's model, so it cannot bound it."""
        assert tl._has_diffusion_ceiling("custom") is False

    @pytest.mark.parametrize("framework", ["xdit", "hunyuan_image3"])
    def test_shipped_scriptable_frameworks_keep_theirs(self, framework):
        assert tl._has_diffusion_ceiling(framework) is True

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


class TestTheGateIsWiredIn:
    """The helper above is only worth anything if ``write_reports`` consults it."""

    def test_custom_gets_no_diffusion_roofline_artifact(self, tmp_path):
        artifacts = _write_reports_for(tmp_path, "custom")
        assert "diffusion_roofline" not in artifacts

    def test_the_sidecar_gate_asks_has_diffusion_ceiling(self, tmp_path, monkeypatch):
        """Pins the call site, not just the predicate: swapping the gate back to
        ``_is_scriptable_framework`` leaves this recorder untouched."""
        asked: list[str | None] = []
        monkeypatch.setattr(
            tl,
            "_has_diffusion_ceiling",
            lambda fw: asked.append(fw) or False,
        )
        _write_reports_for(tmp_path, "custom")
        assert asked == ["custom"]


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
