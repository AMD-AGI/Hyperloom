###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the shared diffusion per-denoise-step divisor helpers."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _denoise_steps as ds  # noqa: E402


def test_divisor_prefers_requested_over_inferred():
    # A declared --num-denoise-steps wins; Hyperloom cannot know what an
    # operator's prof.step() brackets.
    assert ds.resolve_perstep_divisor(requested_steps=9, inferred_steps=3) == 9
    assert ds.resolve_perstep_divisor(requested_steps=16, inferred_steps=0) == 16


def test_divisor_falls_back_to_inferred_when_nothing_requested():
    assert ds.resolve_perstep_divisor(requested_steps=0, inferred_steps=8) == 8
    assert ds.resolve_perstep_divisor(requested_steps=None, inferred_steps=8) == 8


def test_divisor_none_when_nothing_known():
    assert ds.resolve_perstep_divisor(0, 0) is None
    assert ds.resolve_perstep_divisor(None, None) is None


class TestCallSiteBinding:
    """Pin what each route BINDS to each parameter, not just the helper.

    Asserting the helper alone proves only that it is deterministic. These
    recorders fail if either call site's arguments are swapped, which is the
    regression that made the divisor route-dependent in the first place.
    """

    _TRACE = {
        "traceEvents": [
            {"cat": "cpu_op", "name": "aten::mm", "args": {"External id": 100}},
            {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 5, "External id": 100}},
            {"cat": "kernel", "ph": "X", "name": "k0", "ts": 1000, "dur": 200, "args": {"correlation": 5}},
            # Three ProfilerStep markers -> inferred == 3, distinct from requested.
            {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#0", "ts": 1000, "dur": 1},
            {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#1", "ts": 1001, "dur": 1},
            {"cat": "gpu_user_annotation", "ph": "X", "name": "ProfilerStep#2", "ts": 1002, "dur": 1},
        ]
    }

    def _trace(self, tmp_path: Path) -> Path:
        f = tmp_path / "trace.json"
        f.write_text(json.dumps(self._TRACE), encoding="utf-8")
        return f

    def test_bypass_binds_requested_and_inferred_correctly(self, tmp_path, monkeypatch, capsys):
        import bypass_trace_analysis as bta

        seen: dict = {}

        def _recorder(*, requested_steps=None, inferred_steps=None):
            seen.update(requested_steps=requested_steps, inferred_steps=inferred_steps)
            return ds.resolve_perstep_divisor(requested_steps=requested_steps, inferred_steps=inferred_steps)

        monkeypatch.setattr(bta, "resolve_perstep_divisor", _recorder)
        trace = self._trace(tmp_path)
        # xdit so the diffusion sidecar (and therefore the helper) is reached.
        rc = bta.main(
            [
                "--trace-input",
                str(trace),
                "--session-id",
                "binding",
                "--workspace-path",
                str(tmp_path / "ws"),
                "--framework",
                "xdit",
                "--target-platform",
                "MI300X",
                "--model-name",
                "m",
                "--top-k",
                "4",
                "--num-denoise-steps",
                "9",
            ]
        )
        capsys.readouterr()
        assert rc == 0
        assert seen, "the bypass route never reached resolve_perstep_divisor"
        assert seen["requested_steps"] == 9, "the declared count must bind to requested_steps"
        assert seen["inferred_steps"] == 3, "the trace-derived count must bind to inferred_steps"

    def test_tracelens_binds_requested_and_inferred_correctly(self, tmp_path, monkeypatch):
        import diffusion_roofline as dr
        import tracelens_analysis as tl

        seen: dict = {}

        def _recorder(*, requested_steps=None, inferred_steps=None):
            seen.update(requested_steps=requested_steps, inferred_steps=inferred_steps)
            return ds.resolve_perstep_divisor(requested_steps=requested_steps, inferred_steps=inferred_steps)

        # The call site imports from the module at call time, so patch the source.
        monkeypatch.setattr(ds, "resolve_perstep_divisor", _recorder)
        monkeypatch.setattr(dr, "build_report", lambda *a, **k: {"totals": {}})

        trace = self._trace(tmp_path)
        run_dir = tmp_path / "run"
        (run_dir / "tracelens").mkdir(parents=True, exist_ok=True)
        md = run_dir / "tracelens" / "analysis.md"
        md.write_text("# upstream\n", encoding="utf-8")
        args = argparse.Namespace(
            trace_input=str(trace),
            model_name="m",
            framework="xdit",
            target_platform="MI300X",
            analysis_mode="inference",
            runtime_env="local",
            dry_run=False,
            num_denoise_steps=9,
        )
        tl.write_reports(
            run_dir,
            trace_input_type="file",
            trace_files=[trace],
            candidates=[],
            args=args,
            existing_report_path=md,
        )
        assert seen, "the TraceLens route never reached resolve_perstep_divisor"
        assert seen["requested_steps"] == 9, "the declared count must bind to requested_steps"
        assert seen["inferred_steps"] == 3, "the trace-derived count must bind to inferred_steps"


def test_bypass_cli_default_honours_the_shared_env_var(monkeypatch):
    """Both CLIs must derive the default from the same place.

    The TraceLens CLI has always read ``HYPERLOOM_NUM_DENOISE_STEPS`` for this
    default while bypass hardcoded 0. That was harmless while the inferred count
    always won, but once the requested count takes precedence the env var would
    change the divisor on one route only.
    """
    import importlib

    import bypass_trace_analysis as bta

    monkeypatch.setenv("HYPERLOOM_NUM_DENOISE_STEPS", "9")
    importlib.reload(bta)
    argv = ["--trace-input", "x", "--session-id", "s", "--workspace-path", "w"]
    assert bta._build_arg_parser().parse_args(argv).num_denoise_steps == 9
    # An explicit flag still wins over the env.
    assert bta._build_arg_parser().parse_args(argv + ["--num-denoise-steps", "4"]).num_denoise_steps == 4

    monkeypatch.delenv("HYPERLOOM_NUM_DENOISE_STEPS", raising=False)
    importlib.reload(bta)
    assert bta._build_arg_parser().parse_args(argv).num_denoise_steps == 0


def _write_trace(path: Path, names: list[str], gz: bool):
    ev = {"traceEvents": [{"name": n, "ph": "X", "ts": i, "dur": 1} for i, n in enumerate(names)]}
    raw = json.dumps(ev).encode("utf-8")
    if gz:
        with gzip.open(path, "wb") as fh:
            fh.write(raw)
    else:
        path.write_bytes(raw)


def test_count_profiler_steps_gz_and_plain(tmp_path):
    names = ["ProfilerStep#1", "aten::mm", "ProfilerStep#2", "ProfilerStep#2", "ProfilerStep#3"]
    gzp = tmp_path / "t.pt.trace.json.gz"
    _write_trace(gzp, names, gz=True)
    assert ds.count_profiler_steps(str(gzp)) == 3  # distinct #1/#2/#3
    plain = tmp_path / "t.json"
    _write_trace(plain, names, gz=False)
    assert ds.count_profiler_steps(str(plain)) == 3


def test_count_profiler_steps_across_chunk_boundary(tmp_path, monkeypatch):
    # Tiny chunks force ProfilerStep#N markers to straddle boundaries; the
    # overlap carry must still match them without double-counting.
    monkeypatch.setattr(ds, "_CHUNK_BYTES", 8)
    names = ["ProfilerStep#1", "aten::mm", "ProfilerStep#2", "ProfilerStep#3", "ProfilerStep#2"]
    gzp = tmp_path / "t.pt.trace.json.gz"
    _write_trace(gzp, names, gz=True)
    assert ds.count_profiler_steps(str(gzp)) == 3


def test_count_profiler_steps_none_and_dir(tmp_path):
    no_steps = tmp_path / "n.json"
    _write_trace(no_steps, ["aten::mm", "aten::add"], gz=False)
    assert ds.count_profiler_steps(str(no_steps)) == 0
    # directory input picks the first trace file
    d = tmp_path / "run"
    d.mkdir()
    _write_trace(d / "x.pt.trace.json.gz", ["ProfilerStep#5", "ProfilerStep#6"], gz=True)
    assert ds.count_profiler_steps(str(d)) == 2
    assert ds.count_profiler_steps(str(tmp_path / "missing.json")) == 0
