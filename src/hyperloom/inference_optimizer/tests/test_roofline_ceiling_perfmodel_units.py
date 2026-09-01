# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bottom-up PerfModel roofline and the HF metadata it reads.

The op formulas mirror TraceLens PerfModel, so they are pinned against the
arithmetic in their own docstrings rather than against recorded outputs: a
recorded number cannot tell a corrected formula apart from a broken one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import roofline_ceiling as rc


# ---- op formulas ----


def test_gemm_flops_is_two_mnk():
    assert rc._gemm_flops(4, 8, 16) == 2.0 * 4 * 8 * 16


def test_gemm_bytes_separates_activation_from_weight_precision():
    """A quantized weight is read at its own width; activations stay bf16."""
    m, n, k = 4, 8, 16
    both_fp8 = rc._gemm_bytes(m, n, k, weight_bpe=1.0)
    assert both_fp8 == m * k * 1.0 + k * n * 1.0 + m * n * 1.0

    split = rc._gemm_bytes(m, n, k, weight_bpe=1.0, act_bpe=2.0)
    assert split == m * k * 2.0 + k * n * 1.0 + m * n * 2.0
    # Only the weight read stays narrow, so the split total is the larger one.
    assert split > both_fp8


def test_sdpa_flops_counts_both_matmuls():
    b, n_q, h_q, n_kv, h_kv, d = 2, 3, 8, 5, 2, 64
    expected = b * h_q * (2.0 * n_q * n_kv * d) * 2
    assert rc._sdpa_flops(b, n_q, h_q, n_kv, h_kv, d, d, causal=False) == expected


def test_causal_masking_halves_prefill_attention_only():
    """Halved when the two lengths match; decode (N_Q=1) is untouched."""
    args = (2, 7, 8, 7, 2, 64, 64)
    assert rc._sdpa_flops(*args, causal=True) == rc._sdpa_flops(*args, causal=False) / 2.0

    decode = (2, 1, 8, 7, 2, 64, 64)
    assert rc._sdpa_flops(*decode, causal=True) == rc._sdpa_flops(*decode, causal=False)


def test_sdpa_bytes_reads_kv_at_the_kv_head_count():
    """GQA is the point of the split: K/V are sized by H_KV, Q and out by H_Q."""
    b, n_q, h_q, n_kv, h_kv, d, bpe = 2, 3, 8, 5, 2, 64, 2.0
    expected = (b * n_q * h_q * d + b * n_kv * h_kv * d * 2 + b * n_q * h_q * d) * bpe
    assert rc._sdpa_bytes(b, n_q, h_q, n_kv, h_kv, d, d, False, bpe) == expected


def test_sdpa_bytes_ignores_causal():
    """Causal masking skips compute, not the KV read."""
    args = (2, 7, 8, 7, 2, 64, 64)
    assert rc._sdpa_bytes(*args, True, 2.0) == rc._sdpa_bytes(*args, False, 2.0)


def test_fused_moe_flops_counts_gate_up_down_and_aggregation():
    m, k, n, topk = 4, 16, 32, 2
    expected = 2.0 * m * k * n * topk * 2 + 2.0 * m * k * n * topk + m * k * (2 * topk - 1)
    assert rc._fused_moe_flops(m, k, n, topk) == expected


def test_fused_moe_active_experts_saturate_with_batch_size():
    """Coupon collector: one token touches topk experts, a large batch touches all."""
    k, n, num_experts, topk, bpe = 16, 32, 8, 2, 2.0

    def _expert_bytes(m):
        # Subtract the activation terms to leave the expert-weight reads.
        return rc._fused_moe_bytes(m, k, n, num_experts, topk, bpe) - 2 * m * k * bpe

    one_token = _expert_bytes(1)
    assert one_token == pytest.approx(topk * n * k * bpe * 3)

    all_experts = num_experts * n * k * bpe * 3
    assert _expert_bytes(4096) == pytest.approx(all_experts)
    assert one_token < _expert_bytes(8) < all_experts


def test_fused_moe_bytes_defaults_activations_to_the_weight_width():
    args = (4, 16, 32, 8, 2)
    assert rc._fused_moe_bytes(*args, 1.0) == rc._fused_moe_bytes(*args, 1.0, act_bpe=1.0)


# ---- compute_roofline_from_perfmodel ----


def _dense_meta(**over) -> rc.ModelMeta:
    """A small dense model whose every PerfModel input is populated."""
    base = dict(
        weight_bytes=16 * 1024**3,
        num_layers=4,
        num_kv_heads=2,
        head_dim=64,
        weight_dtype_bytes=2.0,
        hidden_size=512,
        intermediate_size=1024,
        vocab_size=32000,
        num_attention_heads=8,
    )
    base.update(over)
    return rc.ModelMeta(**base)


_UNSET = object()


def _perfmodel(meta=_UNSET, **over):
    kw = dict(
        meta=_dense_meta() if meta is _UNSET else meta,
        gpu_type="mi300x",
        concurrency=8,
        isl=128,
        osl=64,
    )
    kw.update(over)
    return rc.compute_roofline_from_perfmodel(**kw)


def test_perfmodel_breaks_a_dense_forward_into_its_operators():
    out = _perfmodel()
    assert out is not None
    names = [op.name for op in out.ops]
    assert names == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", "sdpa"]
    assert out.bound_kind in {"compute", "memory"}
    assert out.decode_tok_per_s > 0
    assert out.prefill_tok_per_s > 0


def test_perfmodel_op_shares_are_normalised_over_the_forward():
    out = _perfmodel()
    assert sum(op.pct_time for op in out.ops) == pytest.approx(1.0)
    assert all(op.time_s > 0 and op.flops > 0 and op.bytes_moved > 0 for op in out.ops)
    assert all(op.ai == pytest.approx(op.flops / op.bytes_moved) for op in out.ops)


def test_perfmodel_decode_sits_between_its_own_memory_and_compute_ceilings():
    """The roofline takes the slower side, so its rate is the lower of the two."""
    out = _perfmodel()
    assert out.decode_tok_per_s == pytest.approx(min(out.decode_mem_tok_per_s, out.decode_cmp_tok_per_s))
    slower = "memory" if out.decode_mem_tok_per_s <= out.decode_cmp_tok_per_s else "compute"
    assert out.bound_kind == slower


def test_perfmodel_routes_a_moe_model_through_the_fused_expert_op():
    """A MoE model replaces the three dense FFN GEMMs with one fused op."""
    out = _perfmodel(_dense_meta(num_experts=8, experts_per_tok=2, moe_intermediate_size=256))
    names = [op.name for op in out.ops]
    assert "moe_fused" in names
    assert not {"gate_proj", "up_proj", "down_proj"} & set(names)


def test_perfmodel_scales_the_hardware_by_the_gpu_count():
    one, four = _perfmodel(num_gpus=1), _perfmodel(num_gpus=4)
    assert four.hbm_bw_gbps == one.hbm_bw_gbps * 4
    assert four.peak_achievable_tflops == one.peak_achievable_tflops * 4
    assert four.decode_tok_per_s > one.decode_tok_per_s


def test_perfmodel_declines_what_it_cannot_model():
    assert _perfmodel(None) is None
    assert _perfmodel(_dense_meta(hidden_size=0)) is None
    assert _perfmodel(_dense_meta(num_attention_heads=0)) is None
    assert _perfmodel(_dense_meta(num_layers=0)) is None
    # Unknown GPU, and a GPU with no achievable TFLOPS at this precision.
    assert _perfmodel(gpu_type="h100") is None
    assert _perfmodel(precision_tag="int3") is None


def test_perfmodel_drops_the_lm_head_when_the_vocab_is_unknown():
    out = _perfmodel(_dense_meta(vocab_size=0))
    assert "lm_head" not in [op.name for op in out.ops]


# ---- load_model_meta ----


def _write_model(dir_path: Path, cfg: dict, *, weight_bytes: int = 4096) -> Path:
    """Write a minimal local HF model dir: config.json + one weight shard."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (dir_path / "model.safetensors").write_bytes(b"\0" * weight_bytes)
    return dir_path


_DENSE_CFG = {
    "num_hidden_layers": 4,
    "hidden_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 1024,
    "vocab_size": 32000,
    "torch_dtype": "bfloat16",
}


def test_load_model_meta_reads_the_dense_shape(tmp_path):
    meta = rc.load_model_meta(_write_model(tmp_path / "m", _DENSE_CFG, weight_bytes=8192))

    assert meta.weight_bytes == 8192
    assert (meta.num_layers, meta.num_kv_heads, meta.hidden_size) == (4, 2, 512)
    assert meta.head_dim == 512 // 8
    assert meta.weight_dtype_bytes == 2.0
    # Dense: no expert decomposition, so the whole weight set is active.
    assert (meta.num_experts, meta.expert_weight_bytes) == (0, 0)
    assert meta.active_weight_bytes == meta.weight_bytes


def test_load_model_meta_prefers_the_safetensors_index_over_the_shard_sizes(tmp_path):
    """The index records the byte-exact total; the shards on disk may be sparse."""
    d = _write_model(tmp_path / "m", _DENSE_CFG, weight_bytes=10)
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 123456}}), encoding="utf-8")
    assert rc.load_model_meta(d).weight_bytes == 123456


def test_load_model_meta_falls_back_when_the_index_is_unusable(tmp_path):
    d = _write_model(tmp_path / "m", _DENSE_CFG, weight_bytes=777)
    (d / "model.safetensors.index.json").write_text("{not json", encoding="utf-8")
    assert rc.load_model_meta(d).weight_bytes == 777


def test_load_model_meta_takes_the_head_dim_the_config_states(tmp_path):
    """An explicit head_dim wins over hidden_size / heads, which need not divide."""
    cfg = {**_DENSE_CFG, "head_dim": 128}
    assert rc.load_model_meta(_write_model(tmp_path / "m", cfg)).head_dim == 128


def test_load_model_meta_treats_a_missing_kv_head_count_as_multi_head(tmp_path):
    cfg = {k: v for k, v in _DENSE_CFG.items() if k != "num_key_value_heads"}
    assert rc.load_model_meta(_write_model(tmp_path / "m", cfg)).num_kv_heads == 8


def test_load_model_meta_sizes_weights_by_the_quantization_method(tmp_path):
    """quant_method outranks torch_dtype: the checkpoint is stored quantized."""
    cfg = {**_DENSE_CFG, "quantization_config": {"quant_method": "fp8"}}
    assert rc.load_model_meta(_write_model(tmp_path / "m", cfg)).weight_dtype_bytes == 1.0


def test_load_model_meta_decomposes_a_moe_checkpoint(tmp_path):
    """Only the routed experts a token activates count toward its weight IO."""
    cfg = {
        **_DENSE_CFG,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
    }
    meta = rc.load_model_meta(_write_model(tmp_path / "m", cfg, weight_bytes=64 * 1024**2))

    assert (meta.num_experts, meta.experts_per_tok) == (8, 2)
    assert 0 < meta.expert_weight_bytes < meta.weight_bytes
    # Non-expert weights, plus the 2-of-8 share of the expert weights.
    assert meta.active_weight_bytes == (
        meta.weight_bytes - meta.expert_weight_bytes + int(2 / 8 * meta.expert_weight_bytes)
    )
    assert meta.active_weight_bytes < meta.weight_bytes


def test_load_model_meta_sizes_routed_experts_at_their_own_precision(tmp_path):
    """fp4 experts under an fp8 model: the global dtype would over-count them."""
    cfg = {
        **_DENSE_CFG,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "quantization_config": {"quant_method": "fp8"},
        "expert_dtype": "fp4",
    }
    meta = rc.load_model_meta(_write_model(tmp_path / "m", cfg, weight_bytes=64 * 1024**2))

    assert meta.weight_dtype_bytes == 1.0
    assert meta.expert_weight_dtype_bytes == 0.5
    assert meta.expert_weight_bytes > 0


#: The Quark MXFP4 shape, verbatim from
#: ``Qwen3.8-2.4T-A95B-Quark-MXFP4/config.json``: ``quant_method`` names the
#: toolkit, and the weight precision sits on the nested global weight spec.
_QUARK_MXFP4_QUANT_CFG = {
    "quant_method": "quark",
    "quant_mode": "eager_mode",
    "global_quant_config": {
        "input_tensors": {"dtype": "fp4", "is_dynamic": True, "group_size": 32},
        "weight": {"dtype": "fp4", "is_dynamic": False, "qscheme": "per_group", "group_size": 32},
        "output_tensors": None,
    },
    "layer_quant_config": {},
}


def test_quant_config_weight_bytes_reads_the_nested_quark_weight_spec():
    """quant_method says "quark"; only the nested weight spec names the precision."""
    assert rc._resolve_quant_config_weight_bytes(_QUARK_MXFP4_QUANT_CFG) == 0.5


def test_quant_config_weight_bytes_still_takes_a_precision_named_method():
    """The flat HF form keeps working: the method itself is the precision."""
    assert rc._resolve_quant_config_weight_bytes({"quant_method": "fp8"}) == 1.0
    # ``awq``/``gptq`` name a 4-bit method the dtype table does not carry.
    assert rc._resolve_quant_config_weight_bytes({"quant_method": "awq"}) == 0.5


def test_quant_config_weight_bytes_reads_compressed_tensors_bit_widths():
    cfg = {"quant_method": "compressed-tensors", "config_groups": {"group_0": {"weights": {"num_bits": 4}}}}
    assert rc._resolve_quant_config_weight_bytes(cfg) == 0.5


def test_quant_config_weight_bytes_takes_the_precision_most_groups_agree_on():
    """Per-layer groups need not agree, and dict order must not decide.

    A Quark MoE checkpoint stores the routed experts at fp4 and the attention
    projections at fp8. Returning whichever group iterated first declared that
    precision for the whole model -- and the answer flipped with dict order.
    """
    cfg = {
        "quant_method": "quark",
        "layer_quant_config": {
            "*.self_attn.q_proj": {"weight": {"dtype": "fp8"}},
            "*.mlp.experts.*": {"weight": {"dtype": "fp4"}},
            "*.mlp.experts.down_proj": {"weight": {"dtype": "fp4"}},
        },
    }
    assert rc._resolve_quant_config_weight_bytes(cfg) == 0.5
    # Same groups, opposite majority -> opposite answer, from the counts alone.
    cfg["layer_quant_config"]["*.mlp.experts.*"] = {"weight": {"dtype": "fp8"}}
    cfg["layer_quant_config"]["*.mlp.experts.down_proj"] = {"weight": {"dtype": "fp8"}}
    assert rc._resolve_quant_config_weight_bytes(cfg) == 1.0


def test_quant_config_weight_bytes_breaks_a_tie_toward_the_wider_type():
    # Undercounting weight bytes raises the roofline and reports a real
    # regression as "already at ceiling", so a tie resolves upward.
    cfg = {
        "quant_method": "quark",
        "layer_quant_config": {
            "a": {"weight": {"dtype": "fp4"}},
            "b": {"weight": {"dtype": "fp8"}},
        },
    }
    assert rc._resolve_quant_config_weight_bytes(cfg) == 1.0


def test_a_whole_checkpoint_scope_still_outranks_the_per_group_ones():
    cfg = {
        "quant_method": "quark",
        "global_quant_config": {"weight": {"dtype": "fp4"}},
        "layer_quant_config": {"a": {"weight": {"dtype": "fp8"}}, "b": {"weight": {"dtype": "fp8"}}},
    }
    assert rc._resolve_quant_config_weight_bytes(cfg) == 0.5


@pytest.mark.parametrize("quant_cfg", [None, {}, "fp8", {"quant_method": "unknown-toolkit"}])
def test_quant_config_weight_bytes_is_silent_without_a_decisive_signal(quant_cfg):
    """No signal returns 0.0 so the caller falls back to the checkpoint dtype."""
    assert rc._resolve_quant_config_weight_bytes(quant_cfg) == 0.0


def test_quant_config_weight_bytes_knows_mxfp8():
    """MiniMax-M3-MXFP8 names the method ``mxfp8`` with no nested weight spec."""
    assert rc._resolve_quant_config_weight_bytes({"quant_method": "mxfp8"}) == 1.0


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("num_experts_per_tok", 4),  # the common HF spelling
        ("num_experts_per_token", 16),  # Kimi-K3
        ("top_k_experts", 8),  # Gemma-4
    ],
)
def test_experts_per_tok_reads_every_alias_in_use(key, expected):
    assert rc._derive_experts_per_tok({key: expected}) == expected


def test_experts_per_tok_prefers_the_canonical_spelling():
    assert rc._derive_experts_per_tok({"num_experts_per_tok": 2, "top_k_experts": 9}) == 2


def test_experts_per_tok_is_zero_when_no_alias_is_present():
    assert rc._derive_experts_per_tok({"num_experts": 8}) == 0


def test_moe_hidden_size_prefers_the_latent_expert_width():
    """Kimi-K3 runs its experts at ``routed_expert_hidden_size``, not ``hidden_size``."""
    assert rc._derive_moe_hidden_size({"hidden_size": 7168, "routed_expert_hidden_size": 3584}) == 3584


def test_moe_hidden_size_falls_back_to_the_residual_width():
    assert rc._derive_moe_hidden_size({"hidden_size": 7168}) == 7168


def test_moe_decomposition_reads_a_gemma_style_topk_alias():
    """Regression: ``top_k_experts`` read as 0 degraded a 128-expert model to dense."""
    cfg = {
        "num_experts": 128,
        "top_k_experts": 8,
        "hidden_size": 2816,
        "num_hidden_layers": 30,
        "moe_intermediate_size": 704,
    }
    _, total, experts, per_tok = rc._compute_expert_decomposition(cfg, weight_bytes=51_611_872_412, dtype_bytes=2.0)
    assert (experts, per_tok) == (128, 8)
    assert total == 30 * 128 * 3 * 2816 * 704 * 2


def test_moe_decomposition_sizes_latent_experts_at_their_own_width():
    """Sizing Kimi-K3's experts at hidden_size doubles them past the checkpoint."""
    cfg = {
        "num_experts": 896,
        "num_experts_per_token": 16,
        "hidden_size": 7168,
        "routed_expert_hidden_size": 3584,
        "num_hidden_layers": 93,
        "moe_intermediate_size": 3072,
    }
    weight_bytes = 1_560_860_324_864
    _, total, experts, _ = rc._compute_expert_decomposition(cfg, weight_bytes=weight_bytes, dtype_bytes=0.5)
    assert experts == 896
    assert total == int(93 * 896 * 3 * 3584 * 3072 * 0.5)
    # The residual width would overshoot the checkpoint and safe-degrade.
    wide = {k: v for k, v in cfg.items() if k != "routed_expert_hidden_size"}
    assert rc._compute_expert_decomposition(wide, weight_bytes=weight_bytes, dtype_bytes=0.5)[2] == 0


def test_perfmodel_sizes_the_moe_op_at_the_latent_expert_width(tmp_path):
    """A narrower expert width must shrink the MoE op, not just the guard math."""

    def _breakdown(moe_hidden):
        meta = rc.ModelMeta(
            weight_bytes=1_000_000,
            num_layers=4,
            num_kv_heads=2,
            head_dim=64,
            weight_dtype_bytes=0.5,
            num_experts=128,
            experts_per_tok=8,
            expert_weight_dtype_bytes=0.5,
            hidden_size=4096,
            moe_intermediate_size=1024,
            moe_hidden_size=moe_hidden,
            vocab_size=32000,
            num_attention_heads=8,
        )
        b = rc.compute_roofline_from_perfmodel(
            meta=meta, gpu_type="mi355x", concurrency=8, isl=1024, osl=128, num_gpus=1, precision_tag="mxfp4"
        )
        return next(o for o in b.ops if o.name == "moe_fused")

    assert _breakdown(2048).bytes_moved < _breakdown(0).bytes_moved
    # 0 means "unset", which must behave exactly like the residual width.
    assert _breakdown(0).bytes_moved == _breakdown(4096).bytes_moved


def test_load_model_meta_keeps_a_quark_moe_checkpoint_decomposed(tmp_path):
    """Regression: reading only ``quant_method`` sized MXFP4 experts at bf16.

    The 4x overcount pushed ``total_expert_bytes`` past the checkpoint size,
    tripping the sanity guard and degrading a 512-expert model to dense — which
    dropped the MoE op from the breakdown entirely.
    """
    cfg = {
        **_DENSE_CFG,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 2048,
        "quantization_config": _QUARK_MXFP4_QUANT_CFG,
    }
    # Sized as the real checkpoint is: fp4 experts fit, bf16 experts would not.
    expert_elems = 4 * 512 * 3 * 512 * 2048  # layers * experts * 3 * hidden * moe_inter
    weight_bytes = int(expert_elems * 0.5 * 1.15)
    meta = rc.load_model_meta(_write_model(tmp_path / "m", cfg, weight_bytes=weight_bytes))

    assert meta.weight_dtype_bytes == 0.5
    assert (meta.num_experts, meta.experts_per_tok) == (512, 10)
    assert meta.moe_intermediate_size == 2048
    assert meta.expert_weight_bytes == int(expert_elems * 0.5)
    # At bf16 the same config degrades to dense, which is the bug being pinned.
    assert rc._compute_expert_decomposition(cfg, weight_bytes=weight_bytes, dtype_bytes=2.0) == (
        weight_bytes,
        0,
        0,
        0,
    )


def test_perfmodel_attributes_the_moe_ffn_for_a_quark_checkpoint(tmp_path):
    """The MoE op must appear, and dominate: it is most of the weight IO."""
    cfg = {
        **_DENSE_CFG,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 2048,
        "quantization_config": _QUARK_MXFP4_QUANT_CFG,
    }
    expert_elems = 4 * 512 * 3 * 512 * 2048
    meta = rc.load_model_meta(_write_model(tmp_path / "m", cfg, weight_bytes=int(expert_elems * 0.5 * 1.15)))
    breakdown = rc.compute_roofline_from_perfmodel(
        meta=meta, gpu_type="mi355x", concurrency=64, isl=8192, osl=1024, num_gpus=8, precision_tag="mxfp4"
    )

    ops = {o.name: o.pct_time for o in breakdown.ops}
    assert "moe_fused" in ops
    assert ops["moe_fused"] == max(ops.values())
    # Dense gate/up/down must not double-count the FFN alongside the MoE op.
    assert not {"gate_proj", "up_proj", "down_proj"} & set(ops)


def test_moe_decomposition_degrades_when_the_experts_exceed_the_checkpoint():
    """An implausible decomposition is dropped rather than published."""
    cfg = {
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "hidden_size": 512,
        "num_hidden_layers": 4,
        "moe_intermediate_size": 256,
    }
    active, total, experts, per_tok = rc._compute_expert_decomposition(cfg, weight_bytes=1024, dtype_bytes=2.0)
    assert (active, total, experts, per_tok) == (1024, 0, 0, 0)


@pytest.mark.parametrize(
    "cfg",
    [
        {"num_experts": 0, "num_experts_per_tok": 2},
        {"num_experts": 8, "num_experts_per_tok": 0},
        {"num_experts": 8, "num_experts_per_tok": 2, "hidden_size": 0},
    ],
)
def test_moe_decomposition_needs_a_complete_config(cfg):
    assert rc._compute_expert_decomposition(cfg, weight_bytes=999, dtype_bytes=2.0) == (999, 0, 0, 0)


def test_load_model_meta_declines_an_unreadable_model(tmp_path):
    assert rc.load_model_meta("") is None
    # A dir with a config but no weights, and one with weights but no config.
    no_weights = tmp_path / "a"
    no_weights.mkdir()
    (no_weights / "config.json").write_text(json.dumps(_DENSE_CFG), encoding="utf-8")
    assert rc.load_model_meta(no_weights) is None

    no_config = tmp_path / "b"
    no_config.mkdir()
    (no_config / "model.safetensors").write_bytes(b"\0" * 16)
    assert rc.load_model_meta(no_config) is None


def test_load_model_meta_declines_a_config_that_is_not_a_mapping(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\0" * 16)
    assert rc.load_model_meta(d) is None


# ---- state-level entry points ----


def _state(tmp_path: Path, benchmark: dict, **attrs):
    """A run state whose baseline provenance points at a materialized yaml."""
    import yaml

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "baseline.yaml"
    cfg.write_text(yaml.safe_dump({"benchmark": benchmark}), encoding="utf-8")
    from types import SimpleNamespace

    base = dict(
        last_baseline={"extras": {"materialized_config": str(cfg)}},
        gpu_type="mi300x",
        model_path="",
        precision="",
        framework="",
        tp=0,
        conc=0,
        isl=0,
        osl=0,
        current_best={},
        optimization_stack=[],
    )
    base.update(attrs)
    return SimpleNamespace(**base)


def _serving_benchmark(model_dir: Path, **envs) -> dict:
    base = {"TP": "1", "CONC": "8", "ISL": "128", "OSL": "64"}
    base.update({k: str(v) for k, v in envs.items()})
    return {
        "model": str(model_dir),
        "framework": "sglang",
        "envs": base,
    }


def test_runtime_workload_prefers_the_benchmark_envs_over_state(tmp_path):
    """The yaml is the geometry of record; state attrs only fill its gaps."""
    st = _state(tmp_path, _serving_benchmark(tmp_path / "m"), tp=8, conc=99, isl=1, osl=2)
    rt = rc.resolve_runtime_workload(st)
    assert (rt.tp, rt.concurrency, rt.isl, rt.osl) == (1, 8, 128, 64)
    assert rt.gpu_type == "mi300x"
    assert rt.framework == "sglang"


def test_runtime_workload_falls_back_to_state_when_the_envs_are_silent(tmp_path):
    st = _state(tmp_path, {"model": "/m", "envs": {}}, tp=4, conc=16, isl=512, osl=32)
    rt = rc.resolve_runtime_workload(st)
    assert (rt.tp, rt.concurrency, rt.isl, rt.osl) == (4, 16, 512, 32)


def test_runtime_workload_defaults_concurrency_to_one(tmp_path):
    """Every other geometry field may be unknown; a batch of zero cannot be."""
    rt = rc.resolve_runtime_workload(_state(tmp_path, {"envs": {}}))
    assert rt.concurrency == 1
    assert rt.tp == 0


def test_runtime_workload_survives_an_unreadable_baseline_yaml(tmp_path):
    from types import SimpleNamespace

    st = SimpleNamespace(last_baseline={"extras": {"materialized_config": str(tmp_path / "gone.yaml")}})
    assert rc.resolve_runtime_workload(st).concurrency == 1


def test_breakdown_from_state_reports_the_decode_ceiling(tmp_path):
    model = _write_model(tmp_path / "m", _DENSE_CFG, weight_bytes=8 * 1024**2)
    bd = rc.compute_roofline_breakdown_from_state(_state(tmp_path, _serving_benchmark(model)))

    assert bd.peak_tok_per_sec > 0
    assert bd.bound_kind in {"compute", "memory"}
    # The roofline takes the slower side, so the peak is the lower projection.
    assert bd.peak_tok_per_sec == pytest.approx(min(bd.mem_tok_per_sec, bd.cmp_tok_per_sec))


def test_breakdown_from_state_is_empty_when_the_model_is_unreadable(tmp_path):
    st = _state(tmp_path, _serving_benchmark(tmp_path / "absent"))
    assert rc.compute_roofline_breakdown_from_state(st) == rc._EMPTY_BREAKDOWN


def test_breakdown_from_state_routes_a_diffusion_run_to_the_image_ceiling(tmp_path):
    """xDiT is measured in images/sec, so it never reaches the token ceiling."""
    bench = {"model": str(tmp_path / "m"), "framework": "xdit", "envs": {"TP": "1"}}
    bd = rc.compute_roofline_breakdown_from_state(_state(tmp_path, bench))
    # No transformer config on disk, so the diffusion arm has nothing to model.
    assert bd == rc._EMPTY_BREAKDOWN


def test_peak_from_state_is_the_breakdown_peak(tmp_path):
    model = _write_model(tmp_path / "m", _DENSE_CFG, weight_bytes=8 * 1024**2)
    st = _state(tmp_path, _serving_benchmark(model))
    assert rc.compute_peak_from_state(st) == pytest.approx(
        rc.compute_roofline_breakdown_from_state(st).peak_tok_per_sec
    )


def test_select_peak_and_bound_takes_the_lower_side():
    assert rc.select_peak_and_bound(100.0, 250.0) == (100.0, "memory")
    assert rc.select_peak_and_bound(400.0, 250.0) == (250.0, "compute")


@pytest.mark.parametrize("mem, cmp", [(0.0, 250.0), (100.0, 0.0), (0.0, 0.0)])
def test_select_peak_and_bound_ignores_a_projection_it_could_not_compute(mem, cmp):
    """A zero is 'unknown', not 'infinitely slow'; it must not win the min."""
    peak, _kind = rc.select_peak_and_bound(mem, cmp)
    assert peak == max(mem, cmp)


def test_resolve_runtime_dtype_priority_and_ignores_workload_precision(tmp_path):
    """Recognized --quantization wins; unrecognized quant falls through; precision tags do not."""
    meta_fp32 = _dense_meta(weight_dtype_bytes=4.0)
    meta_fp8 = _dense_meta(weight_dtype_bytes=1.0)

    (tmp_path / "quant").mkdir()
    (tmp_path / "prequant").mkdir()
    (tmp_path / "dtype").mkdir()
    (tmp_path / "quant_vs_prequant").mkdir()
    (tmp_path / "prequant_vs_dtype").mkdir()
    (tmp_path / "fallback").mkdir()
    (tmp_path / "meta_eq_2").mkdir()
    (tmp_path / "meta_eq_0").mkdir()
    (tmp_path / "act_floor").mkdir()

    quant_state = _state(
        tmp_path / "quant",
        _serving_benchmark(tmp_path / "m", EXTRA_SGLANG_ARGS="--quantization fp8 --dtype bfloat16"),
        precision="fp4",
    )
    quant = rc.resolve_runtime_dtype(quant_state, meta_fp32)
    assert quant.source == "server_args_quantization"
    assert quant.quantization == "fp8"
    assert quant.weight_dtype_bytes == 1.0
    assert quant.activation_dtype_bytes == 2.0
    assert quant.compute_precision_tag == "fp8"

    prequant_state = _state(
        tmp_path / "prequant",
        _serving_benchmark(tmp_path / "m", EXTRA_SGLANG_ARGS="--quantization not-a-method"),
        precision="fp8",
    )
    prequant = rc.resolve_runtime_dtype(prequant_state, meta_fp8)
    assert prequant.source == "quantization_config"
    assert prequant.weight_dtype_bytes == 1.0
    assert prequant.compute_precision_tag == "fp8"

    dtype_state = _state(
        tmp_path / "dtype",
        _serving_benchmark(tmp_path / "m", EXTRA_SGLANG_ARGS="--dtype float32"),
        precision="fp8",
    )
    dtype = rc.resolve_runtime_dtype(dtype_state, meta_fp32)
    assert dtype.source == "server_args_dtype"
    assert dtype.quantization == "none"
    assert dtype.weight_dtype_bytes == 4.0
    assert dtype.activation_dtype_bytes == 4.0
    assert dtype.compute_precision_tag == "fp32"

    # 1-vs-2: server_args_quantization must beat quantization_config when both present.
    # A pre-quantized meta (weight_dtype_bytes=0.5 fp4) + recognised --quantization fp8
    # → branch 1 must win even though branch 2 would also fire.
    quant_vs_prequant_state = _state(
        tmp_path / "quant_vs_prequant",
        _serving_benchmark(tmp_path / "m", EXTRA_SGLANG_ARGS="--quantization fp8"),
        precision="fp4",
    )
    meta_fp4 = _dense_meta(weight_dtype_bytes=0.5)
    quant_vs_prequant = rc.resolve_runtime_dtype(quant_vs_prequant_state, meta_fp4)
    assert quant_vs_prequant.source == "server_args_quantization"
    assert quant_vs_prequant.weight_dtype_bytes == 1.0
    assert quant_vs_prequant.compute_precision_tag == "fp8"

    # 2-vs-3: quantization_config must beat server_args_dtype when both present.
    # A pre-quantized fp8 meta + --dtype float32 → branch 2 must win over branch 3.
    prequant_vs_dtype_state = _state(
        tmp_path / "prequant_vs_dtype",
        _serving_benchmark(tmp_path / "m", EXTRA_SGLANG_ARGS="--dtype float32"),
        precision="fp4",
    )
    prequant_vs_dtype = rc.resolve_runtime_dtype(prequant_vs_dtype_state, meta_fp8)
    assert prequant_vs_dtype.source == "quantization_config"
    assert prequant_vs_dtype.weight_dtype_bytes == 1.0
    assert prequant_vs_dtype.compute_precision_tag == "fp8"

    fallback_state = _state(tmp_path / "fallback", _serving_benchmark(tmp_path / "m"), precision="fp8")
    fallback = rc.resolve_runtime_dtype(fallback_state, meta_fp32)
    assert fallback.source == "config_torch_dtype"
    assert fallback.quantization == "none"
    assert fallback.weight_dtype_bytes == 2.0
    assert fallback.activation_dtype_bytes == 2.0
    assert fallback.compute_precision_tag == "bf16"

    # Upper edge of `0 < meta_w_bytes < 2.0`: 2.0 must fall through, not take
    # quantization_config (which a `<= 2.0` widening would incorrectly do).
    meta_eq_2 = rc.resolve_runtime_dtype(
        _state(tmp_path / "meta_eq_2", _serving_benchmark(tmp_path / "m")),
        _dense_meta(weight_dtype_bytes=2.0),
    )
    assert meta_eq_2.source == "config_torch_dtype"
    assert meta_eq_2.weight_dtype_bytes == 2.0

    # Lower edge: unknown (0.0) must not take quantization_config either.
    meta_eq_0 = rc.resolve_runtime_dtype(
        _state(tmp_path / "meta_eq_0", _serving_benchmark(tmp_path / "m")),
        _dense_meta(weight_dtype_bytes=0.0),
    )
    assert meta_eq_0.source == "config_torch_dtype"
    assert meta_eq_0.weight_dtype_bytes == 2.0

    # bf16 activation floor: --dtype fp8 is 1B, but activations stay >= 2.0.
    act_floor = rc.resolve_runtime_dtype(
        _state(
            tmp_path / "act_floor",
            _serving_benchmark(tmp_path / "m", EXTRA_SGLANG_ARGS="--dtype fp8"),
        ),
        meta_fp32,
    )
    assert act_floor.source == "server_args_dtype"
    assert act_floor.weight_dtype_bytes == 1.0
    assert act_floor.activation_dtype_bytes == 2.0


def test_compute_compute_bound_ceiling_fallback_and_degrade_to_zero(monkeypatch):
    # Patch vendor to a *different* positive value (500.0) so swapping the
    # operands of `achievable or vendor` would change the result.  With vendor==0
    # both orderings yield 100.0 and the precedence isn't pinned.
    monkeypatch.setattr(rc, "_resolve_achievable_tflops", lambda _gpu, _tag: 100.0)
    monkeypatch.setattr(rc, "_resolve_peak_tflops", lambda _gpu, _tag: 500.0)

    active = 1_000_000_000
    weight = 9_000_000_000
    # achievable (100.0) must win over vendor (500.0).
    expected = (100.0 * 1e12 * 2) / (2.0 * active / 2.0)
    got = rc.compute_compute_bound_ceiling_tok_per_sec(
        gpu_type="mi300x",
        num_gpus=2,
        precision_tag="bf16",
        active_weight_bytes=active,
        weight_bytes=weight,
        weight_dtype_bytes=2.0,
    )
    assert got == pytest.approx(expected)

    # active→total-weight fallback (active_weight_bytes=0 falls back to weight_bytes).
    fallback = rc.compute_compute_bound_ceiling_tok_per_sec(
        gpu_type="mi300x",
        num_gpus=2,
        precision_tag="bf16",
        active_weight_bytes=0,
        weight_bytes=weight,
        weight_dtype_bytes=2.0,
    )
    assert fallback == pytest.approx((100.0 * 1e12 * 2) / (2.0 * weight / 2.0))
    assert fallback > 0.0

    # Vendor-peak fallback: achievable absent (0.0), vendor-peak covers it.
    monkeypatch.setattr(rc, "_resolve_achievable_tflops", lambda _gpu, _tag: 0.0)
    monkeypatch.setattr(rc, "_resolve_peak_tflops", lambda _gpu, _tag: 200.0)
    vendor_fallback = rc.compute_compute_bound_ceiling_tok_per_sec(
        gpu_type="mi300x",
        num_gpus=1,
        precision_tag="bf16",
        active_weight_bytes=active,
        weight_bytes=weight,
        weight_dtype_bytes=2.0,
    )
    assert vendor_fallback == pytest.approx((200.0 * 1e12) / (2.0 * active / 2.0))
    assert vendor_fallback > 0.0

    monkeypatch.setattr(rc, "_resolve_achievable_tflops", lambda _gpu, _tag: 0.0)
    monkeypatch.setattr(rc, "_resolve_peak_tflops", lambda _gpu, _tag: 0.0)
    assert (
        rc.compute_compute_bound_ceiling_tok_per_sec(
            gpu_type="unknown-gpu",
            num_gpus=1,
            precision_tag="bf16",
            active_weight_bytes=active,
            weight_bytes=weight,
            weight_dtype_bytes=2.0,
        )
        == 0.0
    )
    monkeypatch.setattr(rc, "_resolve_achievable_tflops", lambda _gpu, _tag: 100.0)
    assert (
        rc.compute_compute_bound_ceiling_tok_per_sec(
            gpu_type="mi300x",
            num_gpus=1,
            precision_tag="bf16",
            active_weight_bytes=0,
            weight_bytes=0,
            weight_dtype_bytes=2.0,
        )
        == 0.0
    )
    assert (
        rc.compute_compute_bound_ceiling_tok_per_sec(
            gpu_type="mi300x",
            num_gpus=1,
            precision_tag="bf16",
            active_weight_bytes=active,
            weight_bytes=weight,
            weight_dtype_bytes=0.0,
        )
        == 0.0
    )
