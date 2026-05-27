"""Unit tests for the opt-in roofline-categorized variant filter."""

from __future__ import annotations

from dataclasses import dataclass

from inference_optimizer.orchestrator.action_executors._explore_roofline_filter import (
    categorize_variant,
    filter_variants_by_roofline,
)


@dataclass
class _FakeVariant:
    name: str
    extra_sglang_args: str = ""
    extra_envs: dict | None = None


# ---------------------------------------------------------------------------
# categorize_variant
# ---------------------------------------------------------------------------
def test_categorize_host_overhead_flag():
    cats = categorize_variant("--num-continuous-decode-steps 4", {})
    assert cats == frozenset({"host_overhead"})


def test_categorize_torch_compile_is_compute():
    cats = categorize_variant(
        "--enable-torch-compile --torch-compile-max-bs 128", {},
    )
    assert cats == frozenset({"compute"})


def test_categorize_attention_backend_is_multi_direction():
    """``--attention-backend`` swaps both kernel and memory pattern."""
    cats = categorize_variant("--attention-backend aiter", {})
    assert cats == frozenset({"compute", "memory"})


def test_categorize_combo_flags_unions():
    """A variant that stacks flags from multiple categories takes their union."""
    cats = categorize_variant(
        "--num-continuous-decode-steps 4 --enable-torch-compile", {},
    )
    assert cats == frozenset({"host_overhead", "compute"})


def test_categorize_unknown_flag_is_empty():
    """Conservative: unknown flags → uncategorized, keeper-by-default."""
    cats = categorize_variant("--never-heard-of-this-flag", {})
    assert cats == frozenset()


def test_categorize_env_specific_then_prefix():
    """Specific env-name overrides bag with the same prefix."""
    # Specific name wins over the SGLANG_ prefix general rule.
    cats = categorize_variant(
        "", {"SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "1"},
    )
    assert cats == frozenset({"host_overhead"})

    # Generic AITER_* prefix → compute (no specific override).
    cats = categorize_variant("", {"AITER_USE_TILELANG_GEMM": "1"})
    assert cats == frozenset({"compute"})


def test_categorize_combines_args_and_envs():
    cats = categorize_variant(
        "--disable-radix-cache",
        {"SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "1"},
    )
    assert cats == frozenset({"memory", "host_overhead"})


# ---------------------------------------------------------------------------
# filter_variants_by_roofline
# ---------------------------------------------------------------------------
def test_filter_no_saturation_is_noop():
    """No direction over threshold → grid passes through untouched."""
    grid = [
        _FakeVariant(name="v1", extra_sglang_args="--num-continuous-decode-steps 4"),
        _FakeVariant(name="v2", extra_sglang_args="--enable-torch-compile"),
    ]
    snapshot = {"compute": 10.0, "memory": 20.0, "host_overhead": 5.0, "comm": 0.0}
    kept, dropped = filter_variants_by_roofline(grid, snapshot)
    assert [g.name for g in kept] == ["v1", "v2"]
    assert dropped == []


def test_filter_drops_only_when_all_target_dirs_saturated():
    """Memory-bound roofline → host_overhead variants pass; memory variants drop."""
    grid = [
        _FakeVariant(name="ncds4", extra_sglang_args="--num-continuous-decode-steps 4"),
        _FakeVariant(name="memflag", extra_sglang_args="--mem-fraction-static 0.85"),
        _FakeVariant(name="combo",
                     extra_sglang_args="--num-continuous-decode-steps 4 --mem-fraction-static 0.9"),
        _FakeVariant(name="unknown", extra_sglang_args="--brand-new-flag"),
    ]
    # Memory at 92 %, host_overhead at 30 % → memory is saturated, host is not.
    snapshot = {"compute": 10.0, "memory": 92.0, "host_overhead": 30.0, "comm": 0.0}
    kept, dropped = filter_variants_by_roofline(grid, snapshot)
    kept_names = [g.name for g in kept]
    # ncds4: targets host_overhead → not all saturated → keep
    # memflag: targets memory only → all saturated → DROP
    # combo: targets host_overhead + memory → host_overhead not saturated → keep
    # unknown: uncategorized → keep
    assert "ncds4" in kept_names
    assert "combo" in kept_names
    assert "unknown" in kept_names
    assert "memflag" not in kept_names
    assert len(dropped) == 1
    assert dropped[0]["name"] == "memflag"
    assert dropped[0]["categories"] == ["memory"]
    assert dropped[0]["saturated_directions"] == ["memory"]
    assert dropped[0]["reason"] == "all_target_directions_saturated"


def test_filter_drops_host_overhead_variant_on_qwen3_32b_shape():
    """Bandwidth-bound Qwen3-32B-style snapshot → host-overhead reducers
    correctly identified as wasted work."""
    grid = [
        _FakeVariant(name="ncds4", extra_sglang_args="--num-continuous-decode-steps 4"),
        _FakeVariant(name="ncds16-cgmaxbs128",
                     extra_sglang_args="--num-continuous-decode-steps 16 --cuda-graph-max-bs 128"),
        _FakeVariant(name="combo-host-bubble-attack",
                     extra_sglang_args=("--num-continuous-decode-steps 4 "
                                        "--scheduler-recv-interval 4 "
                                        "--cuda-graph-max-bs 128")),
        _FakeVariant(name="memflag", extra_sglang_args="--max-running-requests 256"),
    ]
    # Roofline says memory is at 92 % (bandwidth-bound), host_overhead is at
    # only 18 % (host is NOT the bottleneck). Memory variant correctly drops
    # (target direction is saturated); host-overhead reducers stay because
    # their target direction has plenty of headroom.
    snapshot = {
        "compute": 25.0, "memory": 92.0, "host_overhead": 18.0, "comm": 0.0,
    }
    kept, dropped = filter_variants_by_roofline(grid, snapshot)
    kept_names = {g.name for g in kept}
    dropped_names = {d["name"] for d in dropped}
    assert kept_names == {"ncds4", "ncds16-cgmaxbs128", "combo-host-bubble-attack"}
    assert dropped_names == {"memflag"}
    # Now flip: host is saturated, memory is not.
    snapshot2 = {
        "compute": 25.0, "memory": 30.0, "host_overhead": 95.0, "comm": 0.0,
    }
    kept2, dropped2 = filter_variants_by_roofline(grid, snapshot2)
    dropped_names2 = {d["name"] for d in dropped2}
    # ncds4 + ncds16-cgmaxbs128: pure host_overhead → dropped
    assert "ncds4" in dropped_names2
    assert "ncds16-cgmaxbs128" in dropped_names2
    # combo-host-bubble-attack: also pure host_overhead → dropped
    assert "combo-host-bubble-attack" in dropped_names2
    # memflag: pure memory → kept (memory not saturated)
    assert "memflag" in {g.name for g in kept2}


def test_filter_threshold_override():
    """Custom threshold lets operators tune sensitivity."""
    grid = [_FakeVariant(name="ncds4",
                         extra_sglang_args="--num-continuous-decode-steps 4")]
    snapshot = {"host_overhead": 75.0, "memory": 5.0,
                "compute": 5.0, "comm": 0.0}
    # Default threshold (80) → not saturated, kept.
    kept, dropped = filter_variants_by_roofline(grid, snapshot)
    assert [g.name for g in kept] == ["ncds4"]
    assert dropped == []
    # Tighter threshold (70) → saturated, dropped.
    kept2, dropped2 = filter_variants_by_roofline(
        grid, snapshot, threshold_pct=70.0,
    )
    assert kept2 == []
    assert len(dropped2) == 1


def test_filter_empty_snapshot_is_noop():
    grid = [_FakeVariant(name="any", extra_sglang_args="--enable-torch-compile")]
    kept, dropped = filter_variants_by_roofline(grid, None)
    assert [g.name for g in kept] == ["any"]
    assert dropped == []
    kept2, dropped2 = filter_variants_by_roofline(grid, {})
    assert [g.name for g in kept2] == ["any"]
    assert dropped2 == []
