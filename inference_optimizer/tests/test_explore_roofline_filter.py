# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the roofline-categorized advisory annotator.

Plan P2_15 demoted the hard filter to an advisory annotator: variants are never
dropped; it only reports which ones the latest snapshot flags ``likely_saturated``.
"""

from __future__ import annotations

from dataclasses import dataclass

from inference_optimizer.orchestrator.action_executors._explore_roofline_filter import (
    categorize_variant,
    compute_saturation_advisory,
)


@dataclass
class _FakeVariant:
    name: str
    extra_server_args: str = ""
    extra_envs: dict | None = None


# categorize_variant
def test_categorize_host_overhead_flag():
    cats = categorize_variant("--num-continuous-decode-steps 4", {})
    assert cats == frozenset({"host_overhead"})


def test_categorize_torch_compile_is_compute():
    cats = categorize_variant(
        "--enable-torch-compile --torch-compile-max-bs 128", {},
    )
    assert cats == frozenset({"compute"})


def test_categorize_attention_backend_is_multi_direction():
    cats = categorize_variant("--attention-backend aiter", {})
    assert cats == frozenset({"compute", "memory"})


def test_categorize_combo_flags_unions():
    cats = categorize_variant(
        "--num-continuous-decode-steps 4 --enable-torch-compile", {},
    )
    assert cats == frozenset({"host_overhead", "compute"})


def test_categorize_unknown_flag_is_empty():
    cats = categorize_variant("--never-heard-of-this-flag", {})
    assert cats == frozenset()


def test_categorize_env_specific_then_prefix():
    cats = categorize_variant(
        "", {"SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "1"},
    )
    assert cats == frozenset({"host_overhead"})
    cats = categorize_variant("", {"AITER_USE_TILELANG_GEMM": "1"})
    assert cats == frozenset({"compute"})


def test_categorize_combines_args_and_envs():
    cats = categorize_variant(
        "--disable-radix-cache",
        {"SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "1"},
    )
    assert cats == frozenset({"memory", "host_overhead"})


# compute_saturation_advisory
def test_advisory_no_saturation_returns_empty():
    grid = [
        _FakeVariant(name="v1", extra_server_args="--num-continuous-decode-steps 4"),
        _FakeVariant(name="v2", extra_server_args="--enable-torch-compile"),
    ]
    snapshot = {"compute": 10.0, "memory": 20.0, "host_overhead": 5.0, "comm": 0.0}
    assert compute_saturation_advisory(grid, snapshot) == []


def test_advisory_flags_only_fully_saturated_variants():
    grid = [
        _FakeVariant(name="ncds4", extra_server_args="--num-continuous-decode-steps 4"),
        _FakeVariant(name="memflag", extra_server_args="--mem-fraction-static 0.85"),
        _FakeVariant(
            name="combo",
            extra_server_args="--num-continuous-decode-steps 4 --mem-fraction-static 0.9",
        ),
        _FakeVariant(name="unknown", extra_server_args="--brand-new-flag"),
    ]
    snapshot = {"compute": 10.0, "memory": 92.0, "host_overhead": 30.0, "comm": 0.0}
    advisory = compute_saturation_advisory(grid, snapshot)
    names = [entry["name"] for entry in advisory]
    assert names == ["memflag"]
    assert advisory[0]["categories"] == ["memory"]
    assert advisory[0]["saturated_directions"] == ["memory"]
    assert advisory[0]["reason"] == "likely_saturated"


def test_advisory_flips_with_saturated_axis():
    grid = [
        _FakeVariant(name="ncds4", extra_server_args="--num-continuous-decode-steps 4"),
        _FakeVariant(
            name="ncds16-cgmaxbs128",
            extra_server_args="--num-continuous-decode-steps 16 --cuda-graph-max-bs 128",
        ),
        _FakeVariant(
            name="combo-host-bubble-attack",
            extra_server_args=(
                "--num-continuous-decode-steps 4 "
                "--scheduler-recv-interval 4 "
                "--cuda-graph-max-bs 128"
            ),
        ),
        _FakeVariant(name="memflag", extra_server_args="--max-running-requests 256"),
    ]
    snapshot_mem_bound = {
        "compute": 25.0, "memory": 92.0, "host_overhead": 18.0, "comm": 0.0,
    }
    names = {n["name"] for n in compute_saturation_advisory(grid, snapshot_mem_bound)}
    assert names == {"memflag"}

    snapshot_host_bound = {
        "compute": 25.0, "memory": 30.0, "host_overhead": 95.0, "comm": 0.0,
    }
    names = {n["name"] for n in compute_saturation_advisory(grid, snapshot_host_bound)}
    assert names == {
        "ncds4", "ncds16-cgmaxbs128", "combo-host-bubble-attack",
    }


def test_advisory_threshold_override():
    grid = [
        _FakeVariant(
            name="ncds4", extra_server_args="--num-continuous-decode-steps 4",
        ),
    ]
    snapshot = {
        "host_overhead": 75.0, "memory": 5.0, "compute": 5.0, "comm": 0.0,
    }
    assert compute_saturation_advisory(grid, snapshot) == []
    advisory = compute_saturation_advisory(grid, snapshot, threshold_pct=70.0)
    assert [n["name"] for n in advisory] == ["ncds4"]


def test_advisory_empty_snapshot_returns_empty():
    grid = [_FakeVariant(name="any", extra_server_args="--enable-torch-compile")]
    assert compute_saturation_advisory(grid, None) == []
    assert compute_saturation_advisory(grid, {}) == []
