# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the roofline_csv leaf module (schema/reader/writer/normalization).

Covers the foundation invariants the CSV interface depends on (plan §10):
round-trip, numeric coercion (blank->None), bound normalization, canonical-key
stability + parity vs roofline_match_key, fail-soft, ceiling upsert, and the
arch-peak resolver (dtype synonyms, int8-blank->None).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hyperloom.orchestrator.kernel.roofline_csv as rc


# --------------------------------------------------------------------------- #
# Schema round-trip + numeric coercion.
# --------------------------------------------------------------------------- #


def test_kernel_roundtrip_preserves_analytical_and_coerces(tmp_path: Path) -> None:
    rows = [
        {
            "name": "aten::mm_lm_head",
            "kernel_id": "k001",
            "kernel_category": "gemm",
            "precision": "fp16",
            "flops": 617779200,
            "bytes_moved": 78036787,
            "arithmetic_intensity": 7.92,
            "flops_per_byte": 7.92,
            "bound_type": "memory_bound",
            "peak_tflops": 1345.0,
            "hbm_bw_gbps": 7200.0,
            "roofline_source": "analytical",
            "tier": "native",
            # ideal_us/compute_us/read_us/write_us intentionally omitted (native blank).
        }
    ]
    path = tmp_path / "kernel_roofline.csv"
    rc.write_kernel_roofline(rows, path)
    out = rc.read_kernel_roofline(path)

    key = rc.canonical_key("aten::mm_lm_head")
    assert key in out
    row = out[key]
    # numeric coercion str->float
    assert row["flops"] == 617779200.0 and isinstance(row["flops"], float)
    assert row["arithmetic_intensity"] == 7.92
    # blank optional column -> None, NOT 0.0
    assert row["ideal_us"] is None
    assert row["compute_us"] is None
    # bound normalized memory_bound -> memory
    assert row["bound_type"] == "memory"
    # identity/string columns preserved
    assert row["kernel_id"] == "k001"
    assert row["precision"] == "fp16"


def test_blank_numeric_is_none_not_zero(tmp_path: Path) -> None:
    path = tmp_path / "k.csv"
    rc.write_kernel_roofline([{"name": "x", "flops": 10, "ideal_us": ""}], path)
    row = rc.read_kernel_roofline(path)[rc.canonical_key("x")]
    assert row["ideal_us"] is None
    assert row["flops"] == 10.0


# --------------------------------------------------------------------------- #
# bound normalization.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("compute_bound", "compute"),
        ("memory_bound", "memory"),
        ("compute", "compute"),
        ("memory", "memory"),
        ("COMPUTE_BOUND", "compute"),
        ("", None),
        (None, None),
    ],
)
def test_canon_bound(raw, expected) -> None:
    assert rc.canon_bound(raw) == expected


# --------------------------------------------------------------------------- #
# canonical_key: stability, shape-aware, and parity with roofline_match_key.
# --------------------------------------------------------------------------- #


def test_canonical_key_shape_aware_distinguishes_gemms() -> None:
    # Two distinct-shape GEMMs must NOT collapse when the shape-aware triple is given.
    k1 = rc.canonical_key("Cijk_foo", category="gemm", mnk="2048x2048x2048", precision="bf16")
    k2 = rc.canonical_key("Cijk_bar", category="gemm", mnk="4096x4096x4096", precision="bf16")
    assert k1 != k2
    # Same inputs -> same key (stable).
    assert k1 == rc.canonical_key("whatever", category="gemm", mnk="2048x2048x2048", precision="bf16")


def test_canonical_key_default_matches_roofline_match_key() -> None:
    # Join correctness: the default key must equal the authoritative roofline_match_key
    # for a representative set of names (else HL-native and external CSVs won't join).
    # tracelens_analysis uses bare-name sibling imports, so its tools dir must be on
    # sys.path to import it (the §4 import-resolution caveat, test-only shim).
    import sys

    tools_dir = Path(__file__).resolve().parents[3] / "agents" / "kernel" / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    pytest.importorskip("_task_group_contract")
    from hyperloom.agents.kernel.tools.tracelens_analysis import roofline_match_key

    names = [
        "Cijk_Ailk_Bljk_HB",
        "gemm_a16w16_asm_kernel",
        "flash_attn_fwd_kernel",
        "moe_ck2stages_gemm",
        "vectorized_layer_norm_kernel",
        "topk_softmax",
        "rope_rotary_embed",
        "ncclAllReduceKernel",
        "elementwise_memcpy",
        "softmax_warp",
        "skinny_gemm_a8w8",
        "some_unmatched_kernel_name",
    ]
    for n in names:
        assert rc.canonical_key(n) == roofline_match_key(n), n


# --------------------------------------------------------------------------- #
# Fail-soft.
# --------------------------------------------------------------------------- #


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert rc.read_kernel_roofline(tmp_path / "nope.csv") == {}
    assert rc.read_ceiling(tmp_path / "nope.csv") == {}
    assert rc.read_arch_peaks(tmp_path / "nope.csv") == {}


def test_malformed_row_skipped_others_kept(tmp_path: Path) -> None:
    path = tmp_path / "k.csv"
    # 'flops' with a non-numeric cell -> coerced to None (not a crash); row still read.
    path.write_text(
        "name,flops\n" "good,123\n" "bad,not_a_number\n",
        encoding="utf-8",
    )
    out = rc.read_kernel_roofline(path)
    assert out[rc.canonical_key("good")]["flops"] == 123.0
    assert out[rc.canonical_key("bad")]["flops"] is None


# --------------------------------------------------------------------------- #
# Ceiling upsert (multiple producers each write a subset of arms/ops).
# --------------------------------------------------------------------------- #


def test_ceiling_upsert_accumulates(tmp_path: Path) -> None:
    path = tmp_path / "roofline_ceiling.csv"
    cfg_a = {"fw": "vllm", "conc": 4}   # two distinct configs -> distinct identities
    cfg_b = {"fw": "vllm", "conc": 32}
    # Producer A writes config A's arm.
    rc.write_ceiling([{"row_type": "arm", **rc.ceiling_key_columns(cfg_a), "peak_tok_per_sec": 626.96, "bound_kind": "memory"}], path)
    # Producer B writes config B's arm — must NOT clobber config A.
    rc.write_ceiling(
        [{"row_type": "arm", **rc.ceiling_key_columns(cfg_b), "peak_tok_per_sec": 641.30, "bound_kind": "memory"}], path
    )
    # Producer C writes an L2 op row (scoped to its config's identity).
    rc.write_ceiling(
        [{"row_type": "op", **rc.ceiling_key_columns(cfg_b), "op_name": "moe_experts", "flops": 1.16e11, "bound_kind": "compute"}],
        path,
    )

    out = rc.read_ceiling(path)
    key_a, key_b = rc.ceiling_key(cfg_a), rc.ceiling_key(cfg_b)
    assert out[key_a]["peak_tok_per_sec"] == 626.96
    assert out[key_b]["peak_tok_per_sec"] == 641.30
    assert out[("op", key_b, "moe_experts")]["bound_kind"] == "compute"
    # config A bound_kind normalized
    assert out[key_a]["bound_kind"] == "memory"


def test_ceiling_upsert_same_arm_last_write_wins(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    cfg = {"fw": "vllm", "conc": 4}
    rc.write_ceiling([{"row_type": "arm", **rc.ceiling_key_columns(cfg), "peak_tok_per_sec": 1.0}], path)
    rc.write_ceiling([{"row_type": "arm", **rc.ceiling_key_columns(cfg), "peak_tok_per_sec": 2.0}], path)
    assert rc.read_ceiling(path)[rc.ceiling_key(cfg)]["peak_tok_per_sec"] == 2.0


# --------------------------------------------------------------------------- #
# Arch-peak resolver: dtype synonyms, int8 blank -> None.
# --------------------------------------------------------------------------- #


def test_arch_peak_resolver_synonyms_and_int8_blank(tmp_path: Path) -> None:
    rc.write_arch_peaks(
        [
            {
                "name": "mi350x",
                "mem_bw_gbps": 7200.0,
                "matrix_bf16_tflops": 1445.0,
                "matrix_fp8_tflops": 3028.0,
                "matrix_mx4_tflops": 4630.0,
                "matrix_int8_tflops": "",  # blank -> None (not producible by external)
            }
        ],
        tmp_path / "gpu_arch_peaks.csv",
    )
    r = rc.RooflineResolver(tmp_path)
    # dtype synonyms map to the right column
    assert r.arch_peak("mi350x", "bfloat16") == 1445.0
    assert r.arch_peak("mi350x", "float8_e4m3fn") == 3028.0
    assert r.arch_peak("mi350x", "mxfp4") == 4630.0
    # int8 blank -> None (caller decides bf16-fallback vs skip)
    assert r.arch_peak("mi350x", "int8") is None
    # unknown device / dtype -> None
    assert r.arch_peak("unknown", "bf16") is None
    assert r.arch_peak("mi350x", "weird") is None
    assert r.mem_bw("mi350x") == 7200.0


def test_resolver_kernel_and_ceiling_lookup(tmp_path: Path) -> None:
    rc.write_kernel_roofline(
        [{"name": "flash_attn_fwd", "flops": 5.0, "bound_type": "compute_bound"}],
        tmp_path / "kernel_roofline.csv",
    )
    cfg = {"fw": "vllm", "conc": 4}
    rc.write_ceiling(
        [{"row_type": "arm", **rc.ceiling_key_columns(cfg), "peak_tok_per_sec": 100.0}],
        tmp_path / "roofline_ceiling.csv",
    )
    r = rc.RooflineResolver(tmp_path)
    k = r.kernel("flash_attn_fwd")
    assert k is not None and k["flops"] == 5.0 and k["bound_type"] == "compute"
    assert r.ceiling(cfg)["peak_tok_per_sec"] == 100.0
    assert r.ceiling({"fw": "vllm", "conc": 999}) is None


def test_resolver_none_dir_is_inert() -> None:
    r = rc.RooflineResolver(None)
    assert r.kernel("x") is None
    assert r.ceiling({"fw": "vllm", "conc": 4}) is None
    assert r.arch_peak("mi350x", "bf16") is None


# --------------------------------------------------------------------------- #
# Projection helpers wired into the analytical producers (A / K).
# --------------------------------------------------------------------------- #


def test_arch_row_from_spec_maps_and_renames_fp4() -> None:
    spec = {
        "name": "MI355X",
        "mem_bw_gbps": 8000.0,
        "max_achievable_tflops": {
            "matrix_bf16": 1686.0,
            "matrix_fp4": 5663.0,  # TraceLens fp4 -> schema mx4
            "matrix_int8": 0,  # non-positive -> dropped (blank)
        },
    }
    row = rc.arch_row_from_spec(spec)
    assert row["name"] == "MI355X"
    assert row["mem_bw_gbps"] == 8000.0
    assert row["matrix_bf16_tflops"] == 1686.0
    assert row["matrix_mx4_tflops"] == 5663.0  # renamed from fp4
    assert "matrix_int8_tflops" not in row  # 0 dropped
    # Round-trips through the arch-peaks CSV columns.
    out = tmp_arch_roundtrip(row)
    assert out["MI355X"]["matrix_mx4_tflops"] == 5663.0


def tmp_arch_roundtrip(row: dict) -> dict:
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as d:
        p = _P(d) / "gpu_arch_peaks.csv"
        rc.write_arch_peaks([row], p)
        return rc.read_arch_peaks(p)


def test_arch_row_from_spec_nameless_is_none() -> None:
    assert rc.arch_row_from_spec({"mem_bw_gbps": 8000.0}) is None


def test_kernel_row_from_view_keeps_only_analytical() -> None:
    view = {
        "name": "triton_gemm",
        "kernel_id": "k1",
        "source_file": "m.py",
        "reusable_native_kernel": True,
        "kernel_category": "gemm",
        "bound_type": "compute_bound",
        "arithmetic_intensity": 42.0,
        "flops_per_byte": 42.0,
        "roofline_source": "analytical",
        # MEASURED / DERIVED — must NOT appear on the CSV row:
        "duration_us": 10.0,
        "gpu_pct": 5.0,
        "call_count": 3,
        "efficiency_percent": 88.0,
        "compute_utilization_pct": 70.0,
    }
    row = rc.kernel_row_from_view(view)
    assert row["name"] == "triton_gemm"
    assert row["bound_type"] == "compute"  # normalized
    assert row["arithmetic_intensity"] == 42.0
    assert row["kernel_category"] == "gemm"
    for measured_or_derived in ("duration_us", "gpu_pct", "call_count", "efficiency_percent", "compute_utilization_pct"):
        assert measured_or_derived not in row


def test_kernel_row_from_view_bottleneck_fallback() -> None:
    # bound_type absent -> falls back to bottleneck.
    row = rc.kernel_row_from_view({"name": "k", "bottleneck": "memory_bound"})
    assert row["bound_type"] == "memory"


def test_kernel_row_from_view_nameless_is_none() -> None:
    assert rc.kernel_row_from_view({"bound_type": "compute"}) is None
