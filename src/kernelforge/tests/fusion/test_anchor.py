# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolving the kernel an operator named, and reading what runs around it."""

from __future__ import annotations

import gzip
import json

import pytest

from kernelforge.fusion.anchor import (
    AnchorResolutionError,
    KernelAnchor,
    build_anchored_discovery_prompt,
    collapse_whitespace,
    describe_anchor,
    resolve_anchor,
)
from kernelforge.fusion.discover import resolve_proposed_source_file

ANCHOR = (
    "void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda"
    "(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >"
    "(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::"
    "{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)"
)


def _write_trace(path, events, gz=False):
    payload = {"traceEvents": events}
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _kernel(name, ts, dur=4.0, stream=7):
    return {"cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": {"device": 0, "stream": stream}}


def _gemm_cast_attention(path, repeats=9, variants=("SPK2", "SPK4", "SPK7")):
    """The real decode shape: a cast between a GEMM and a prefill-attention kernel.

    The GEMM and attention kernels differ only in template parameters between
    launches, which is exactly what name-level aggregation gets wrong.
    """
    events = []
    ts = 0.0
    for i in range(repeats):
        variant = variants[i % len(variants)]
        events += [
            _kernel(f"hgemm_bf16_32x64x128x4_{variant}_W1x4x1_BLDS1_TN_AS1_0", ts, 30.0),
            _kernel(ANCHOR, ts + 100, 4.0),
            _kernel(f"void sglang::flash_c{4 if i % 2 else 128}_prefill<{512}l, float>(Params)", ts + 200, 20.0),
        ]
        ts += 1000
    _write_trace(path, events)
    return path


class TestResolution:
    def test_exact_name_resolves_every_launch(self, tmp_path):
        trace = _gemm_cast_attention(tmp_path / "d.trace.json", repeats=9)

        report = resolve_anchor(trace, KernelAnchor(name=ANCHOR))

        assert report.occurrences == 9
        assert report.category == "cast"
        assert report.total_us == pytest.approx(36.0)
        assert report.avg_us == pytest.approx(4.0)

    def test_a_fragment_is_not_a_name(self, tmp_path):
        """Substring matching would silently pick a kernel the operator did not name."""
        trace = _gemm_cast_attention(tmp_path / "d.trace.json")

        with pytest.raises(AnchorResolutionError) as excinfo:
            resolve_anchor(trace, KernelAnchor(name="bfloat16tofloat32_copy"))

        message = str(excinfo.value)
        assert "no kernel" in message
        # The miss has to hand back something actionable, not just a refusal.
        assert "bfloat16tofloat32_copy_kernel_cuda" in message

    def test_a_name_wrapped_by_a_trace_viewer_still_matches(self, tmp_path):
        trace = _gemm_cast_attention(tmp_path / "d.trace.json")
        wrapped = ANCHOR.replace(", ", ",\n   ")

        report = resolve_anchor(trace, KernelAnchor(name=wrapped))

        assert report.occurrences == 9

    def test_gzipped_trace_resolves(self, tmp_path):
        trace = tmp_path / "d.trace.json.gz"
        _write_trace(trace, [_kernel("Cijk_gemm", 0, 30.0), _kernel(ANCHOR, 100, 4.0)], gz=True)

        assert resolve_anchor(trace, KernelAnchor(name=ANCHOR)).occurrences == 1

    def test_empty_trace_says_so(self, tmp_path):
        trace = tmp_path / "d.trace.json"
        _write_trace(trace, [{"cat": "cpu_op", "name": "aten::add", "ts": 0, "dur": 5}])

        with pytest.raises(AnchorResolutionError, match="no GPU kernel events"):
            resolve_anchor(trace, KernelAnchor(name=ANCHOR))


class TestNeighbourhoodAggregation:
    def test_template_variants_collapse_into_one_pattern(self, tmp_path):
        """Category aggregation, not name aggregation.

        The neighbours differ only in template parameters, so aggregating by name
        reports three unrelated patterns at ~33% each and buries a stable GEMM
        epilogue. Measured on a real DSv4 trace this was 33.3% by name against
        97.8% by category.
        """
        trace = _gemm_cast_attention(tmp_path / "d.trace.json", repeats=9)

        report = resolve_anchor(trace, KernelAnchor(name=ANCHOR))

        assert report.signature == "gemm -> cast -> attention"
        assert report.consistency == pytest.approx(1.0)
        assert len(report.patterns) == 1
        # The variants are still reported underneath, so nothing is hidden by the collapse.
        assert report.before is not None and len(report.before.names) == 3
        assert sum(count for _name, count in report.before.names) == 9

    def test_span_is_bounded_by_the_compute_kernels(self, tmp_path):
        trace = _gemm_cast_attention(tmp_path / "d.trace.json")

        span = resolve_anchor(trace, KernelAnchor(name=ANCHOR)).span

        assert [item["category"] for item in span] == ["gemm", "cast", "attention"]
        assert [item["name"] for item in span].count(ANCHOR) == 1
        assert [item for item in span if item["is_anchor"]][0]["name"] == ANCHOR

    def test_an_unstable_neighbourhood_is_flagged(self, tmp_path):
        trace = tmp_path / "d.trace.json"
        events = []
        for i in range(4):
            after = "Cijk_gemm_after" if i % 2 else "rms_norm_kernel"
            events += [
                _kernel("Cijk_gemm", i * 1000, 30.0),
                _kernel(ANCHOR, i * 1000 + 100, 4.0),
                _kernel(after, i * 1000 + 200, 10.0),
            ]
        _write_trace(trace, events)

        report = resolve_anchor(trace, KernelAnchor(name=ANCHOR))

        assert report.consistency == pytest.approx(0.5)
        assert any("unstable" in w for w in report.warnings)

    def test_neighbours_are_read_per_stream(self, tmp_path):
        """Execution order only holds within a stream, so a neighbour on another one is not adjacent."""
        trace = tmp_path / "d.trace.json"
        _write_trace(
            trace,
            [
                _kernel("Cijk_gemm", 0, 30.0, stream=1),
                _kernel(ANCHOR, 10, 4.0, stream=7),
                _kernel("rms_norm_kernel", 20, 10.0, stream=1),
            ],
        )

        report = resolve_anchor(trace, KernelAnchor(name=ANCHOR))

        assert report.before is None and report.after is None
        assert report.signature == "<none> -> cast -> <none>"


class TestPromptEvidence:
    def test_the_anchor_name_reaches_the_prompt_untruncated(self, tmp_path):
        """The hot-kernel table truncates to 90 chars; a mangled anchor must not be cut."""
        trace = _gemm_cast_attention(tmp_path / "d.trace.json")
        report = resolve_anchor(trace, KernelAnchor(name=ANCHOR))

        model = tmp_path / "deepseek_v4.py"
        model.write_text("class DeepseekV4Attention:\n    def forward(self): ...\n", encoding="utf-8")

        prompt = build_anchored_discovery_prompt(
            model_type="deepseek_v4",
            framework="sglang",
            source_files=[str(model)],
            report=report,
            shapes={"hidden_size": 2048},
        )

        assert len(ANCHOR) > 90
        assert ANCHOR in prompt
        assert "must be part of every proposal" in prompt
        # The scope constraint is what keeps a proposal wireable; it must survive into this prompt too.
        assert "SCOPE" in prompt
        assert "class DeepseekV4Attention" in prompt
        # Without repo scope the chain has to stay inside the one file embedded above.
        assert "One patch, one file." in prompt

    def test_the_description_names_both_sides(self, tmp_path):
        trace = _gemm_cast_attention(tmp_path / "d.trace.json")

        text = describe_anchor(resolve_anchor(trace, KernelAnchor(name=ANCHOR)))

        assert "gemm -> cast -> attention" in text
        assert "immediately before" in text and "immediately after" in text
        assert "hgemm_bf16" in text and "flash_c" in text


def test_collapse_whitespace_leaves_template_parameters_alone():
    """Only whitespace may be normalized: template parameters decide which kernel this is."""
    assert collapse_whitespace("a<4,\n  b>  (int)") == "a<4, b> (int)"
    assert collapse_whitespace("hgemm_SPK4") != collapse_whitespace("hgemm_SPK7")


class TestProposedSourceFileRouting:
    """Which in-scope file a proposal wires itself into.

    Discovery is asked to copy a path back verbatim. Accepting a shorter but
    unambiguous answer keeps a good proposal from being silently re-pointed at
    the model file, which is how a chain ends up wired where it does not live.
    """

    FILES = ["/fw/models/deepseek_v4.py", "/fw/kernels/gemm.py", "/fw/layers/compressor.py"]

    def test_an_exact_path_is_taken(self):
        assert (
            resolve_proposed_source_file("/fw/layers/compressor.py", self.FILES, self.FILES[0])
            == "/fw/layers/compressor.py"
        )

    def test_a_bare_filename_still_identifies_the_file(self):
        assert resolve_proposed_source_file("compressor.py", self.FILES, self.FILES[0]) == "/fw/layers/compressor.py"

    def test_a_trailing_path_fragment_resolves(self):
        assert resolve_proposed_source_file("layers/compressor.py", self.FILES, self.FILES[0]) == "/fw/layers/compressor.py"

    def test_an_unlisted_file_falls_back_to_the_primary(self):
        """An invented target is one the loop cannot track, keep, or revert."""
        assert resolve_proposed_source_file("/fw/other/indexer.py", self.FILES, self.FILES[0]) == self.FILES[0]

    def test_no_answer_falls_back_to_the_primary(self):
        assert resolve_proposed_source_file("", self.FILES, self.FILES[0]) == self.FILES[0]
        assert resolve_proposed_source_file(None, self.FILES, self.FILES[0]) == self.FILES[0]
