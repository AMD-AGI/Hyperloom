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


class TestPinnedTimestamp:
    """Timestamps are nanoseconds on the way in and on the way out; the trace records microseconds."""

    def test_a_referenced_launch_is_reported_without_narrowing_the_aggregate(self, tmp_path):
        trace = _gemm_cast_attention(tmp_path / "d.trace.json", repeats=9)

        report = resolve_anchor(trace, KernelAnchor(name=ANCHOR, ts_ns=3_100_000))

        assert report.pinned_ts_ns == 3_100_000
        assert report.pinned_is_dominant is True
        # Pinning is a reference, not a filter.
        assert report.occurrences == 9

    def test_the_reported_timestamp_round_trips_as_input(self, tmp_path):
        """What the report prints must be pasteable straight back into --fuse-kernel-ts."""
        trace = _gemm_cast_attention(tmp_path / "d.trace.json", repeats=9)

        first = resolve_anchor(trace, KernelAnchor(name=ANCHOR, ts_ns=5_100_000))
        again = resolve_anchor(trace, KernelAnchor(name=ANCHOR, ts_ns=first.pinned_ts_ns))

        assert again.pinned_ts_ns == first.pinned_ts_ns
        assert again.warnings == []

    def test_sub_microsecond_precision_survives(self, tmp_path):
        """Nanoseconds are the point: two launches inside the same microsecond stay distinct."""
        trace = tmp_path / "d.trace.json"
        _write_trace(
            trace,
            [
                _kernel("Cijk_gemm", 1000.0, 30.0),
                _kernel(ANCHOR, 1000.100, 4.0),
                _kernel("rms_norm_kernel", 1000.200, 10.0),
                _kernel("Cijk_gemm", 1000.300, 30.0),
                _kernel(ANCHOR, 1000.400, 4.0),
                _kernel("reduce_kernel", 1000.500, 10.0),
            ],
        )

        # span follows the launch you pinned; before/after describe the aggregate over all launches.
        first = resolve_anchor(trace, KernelAnchor(name=ANCHOR, ts_ns=1_000_100))
        second = resolve_anchor(trace, KernelAnchor(name=ANCHOR, ts_ns=1_000_400))

        assert first.pinned_ts_ns == 1_000_100 and second.pinned_ts_ns == 1_000_400
        # The span runs compute anchor to compute anchor, so the trailing gemm belongs to the first one.
        assert [item["category"] for item in first.span] == ["gemm", "cast", "rmsnorm", "gemm"]
        assert [item["category"] for item in second.span] == ["gemm", "cast", "reduce"]
        # Both matched a launch exactly: neither needed snapping to a neighbour a fraction of a microsecond away.
        assert not [w for w in first.warnings + second.warnings if "nearest" in w]

    def test_a_rounded_timestamp_snaps_to_the_nearest_launch_and_says_so(self, tmp_path):
        trace = _gemm_cast_attention(tmp_path / "d.trace.json", repeats=9)

        report = resolve_anchor(trace, KernelAnchor(name=ANCHOR, ts_ns=3_105_000))

        assert report.pinned_ts_ns == 3_100_000
        assert any("nearest" in w for w in report.warnings)

    def test_a_microsecond_value_is_named_as_the_wrong_unit(self, tmp_path):
        """The one mistake this unit invites, reported instead of left as a silent 1000x miss."""
        trace = _gemm_cast_attention(tmp_path / "d.trace.json", repeats=9)

        report = resolve_anchor(trace, KernelAnchor(name=ANCHOR, ts_ns=3100))

        assert any("looks like microseconds" in w for w in report.warnings)

    def test_an_unrepresentative_launch_is_called_out(self, tmp_path):
        trace = tmp_path / "d.trace.json"
        events = []
        for i in range(4):
            events += [
                _kernel("Cijk_gemm", i * 1000, 30.0),
                _kernel(ANCHOR, i * 1000 + 100, 4.0),
                _kernel("Cijk_gemm_after", i * 1000 + 200, 30.0),
            ]
        events += [
            _kernel("rms_norm_kernel", 9000, 10.0),
            _kernel(ANCHOR, 9100, 4.0),
            _kernel("rms_norm_kernel", 9200, 10.0),
        ]
        _write_trace(trace, events)

        report = resolve_anchor(trace, KernelAnchor(name=ANCHOR, ts_ns=9_100_000))

        assert report.pinned_is_dominant is False
        assert any("not in the dominant pattern" in w for w in report.warnings)


class TestPromptEvidence:
    def test_the_anchor_name_reaches_the_prompt_untruncated(self, tmp_path):
        """The hot-kernel table truncates to 90 chars; a mangled anchor must not be cut."""
        trace = _gemm_cast_attention(tmp_path / "d.trace.json")
        report = resolve_anchor(trace, KernelAnchor(name=ANCHOR))

        prompt = build_anchored_discovery_prompt(
            model_type="deepseek_v4",
            framework="sglang",
            source_text="class DeepseekV4Attention:\n    def forward(self): ...\n",
            report=report,
            shapes={"hidden_size": 2048},
        )

        assert len(ANCHOR) > 90
        assert ANCHOR in prompt
        assert "must be part of every proposal" in prompt
        # The scope constraint is what keeps a proposal wireable; it must survive into this prompt too.
        assert "SCOPE" in prompt
        assert "class DeepseekV4Attention" in prompt

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
