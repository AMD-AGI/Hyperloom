"""A MoE checkpoint is not MoE in every layer.

``867f119cb`` restored the MoE FFN to the PerfModel breakdown, but charged it
to all ``num_hidden_layers`` and kept skipping the dense FFN entirely. Real
checkpoints run a dense prefix: GLM-5.3-Flash and GLM-5.3 both set
``first_k_dense_replace: 3``, so 3 of their 45 / 78 layers have gate/up/down
and no experts. Counting those layers as MoE overstates the largest term in
the breakdown and drops a real one.

The config numbers below are read off ``/shared_nfs/models/GLM-5.3*/config.json``
on the fleet, not invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import roofline_ceiling as rc


def _split(cfg: dict, layers: int, experts: int) -> tuple[int, int]:
    return rc._derive_moe_layer_counts(cfg, layers, experts)


class TestTheLayerSplit:
    def test_a_dense_model_is_all_dense_ffn(self):
        assert _split({}, 32, 0) == (0, 32)

    def test_a_moe_model_with_no_dense_prefix_is_all_moe(self):
        assert _split({}, 32, 128) == (32, 0)

    def test_glm_5_3_flash(self):
        # 45 layers, first_k_dense_replace=3.
        assert _split({"first_k_dense_replace": 3}, 45, 288) == (42, 3)

    def test_glm_5_3(self):
        # 78 layers, first_k_dense_replace=3, moe_layer_freq=1.
        assert _split({"first_k_dense_replace": 3, "moe_layer_freq": 1}, 78, 256) == (75, 3)

    def test_a_per_layer_freq_list_is_counted_directly(self):
        cfg = {"moe_layer_freq": [0, 0, 1, 1, 1, 0]}
        assert _split(cfg, 6, 64) == (3, 3)

    def test_an_integer_stride_applies_after_the_dense_prefix(self):
        # Layers 0-1 dense, then every 2nd layer (2, 4, 6) is MoE.
        assert _split({"first_k_dense_replace": 2, "moe_layer_freq": 2}, 8, 64) == (3, 5)

    def test_qwen_spells_the_stride_decoder_sparse_step(self):
        assert _split({"decoder_sparse_step": 2}, 8, 64) == (4, 4)

    def test_mlp_only_layers_stay_dense(self):
        assert _split({"mlp_only_layers": [0, 1]}, 8, 64) == (6, 2)

    @pytest.mark.parametrize(
        "cfg,layers",
        [
            ({"first_k_dense_replace": 3}, 45),
            ({"first_k_dense_replace": 3, "moe_layer_freq": 1}, 78),
            ({"moe_layer_freq": [1, 0, 1, 0]}, 4),
            ({"first_k_dense_replace": 2, "decoder_sparse_step": 3}, 11),
        ],
    )
    def test_the_two_counts_always_sum_to_the_stack(self, cfg, layers):
        moe, dense = _split(cfg, layers, 64)
        assert moe + dense == layers

    def test_an_oversized_prefix_cannot_produce_negative_counts(self):
        assert _split({"first_k_dense_replace": 999}, 8, 64) == (0, 8)

    def test_a_boolean_freq_is_not_read_as_a_stride(self):
        # ``True`` is an int in Python; treating it as stride 1 would be right
        # by accident, but treating ``False`` as 0 would divide by zero.
        assert _split({"moe_layer_freq": False}, 8, 64) == (8, 0)


def _flash_meta(**over) -> rc.ModelMeta:
    """GLM-5.3-Flash's real decoder shape."""
    base = dict(
        weight_bytes=64 * 1024**3,
        num_layers=45,
        num_kv_heads=8,
        head_dim=128,
        weight_dtype_bytes=2.0,
        num_experts=288,
        experts_per_tok=8,
        hidden_size=4096,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        vocab_size=154880,
        num_attention_heads=32,
        moe_layers=42,
        dense_ffn_layers=3,
    )
    base.update(over)
    return rc.ModelMeta(**base)


def _breakdown(meta):
    return rc.compute_roofline_from_perfmodel(
        meta=meta, gpu_type="mi355x", concurrency=8, isl=1024, osl=128, num_gpus=1
    )


class TestThePerfModelUsesTheSplit:
    def test_a_moe_model_with_a_dense_prefix_gets_both_ffns(self):
        names = [op.name for op in _breakdown(_flash_meta()).ops]

        assert "moe_fused" in names
        # The three layers that are genuinely dense were silently dropped.
        assert {"gate_proj", "up_proj", "down_proj"} <= set(names)

    def test_the_moe_op_is_charged_to_moe_layers_only(self):
        forty_two = next(o for o in _breakdown(_flash_meta()).ops if o.name == "moe_fused")
        all_forty_five = next(
            o for o in _breakdown(_flash_meta(moe_layers=45, dense_ffn_layers=0)).ops
            if o.name == "moe_fused"
        )

        assert forty_two.bytes_moved == pytest.approx(all_forty_five.bytes_moved * 42 / 45)
        assert forty_two.flops == pytest.approx(all_forty_five.flops * 42 / 45)

    def test_the_dense_ffn_is_charged_to_the_prefix_only(self):
        gate = next(o for o in _breakdown(_flash_meta()).ops if o.name == "gate_proj")
        q = next(o for o in _breakdown(_flash_meta()).ops if o.name == "q_proj")

        # q_proj repeats over all 45 layers; gate_proj over the 3 dense ones.
        # Same M, so the ratio is purely the repeat count.
        per_layer_gate = gate.flops / 3
        per_layer_q = q.flops / 45
        assert per_layer_gate == pytest.approx(per_layer_q * (12288 / 4096) * 1.0, rel=0.01)

    def test_attention_still_repeats_over_every_layer(self):
        ops = {o.name: o for o in _breakdown(_flash_meta()).ops}
        # Every layer has attention regardless of its FFN kind; this must not
        # have been dragged along by the FFN split.
        assert ops["sdpa"].flops == pytest.approx(
            next(o for o in _breakdown(_flash_meta(moe_layers=45, dense_ffn_layers=0)).ops if o.name == "sdpa").flops
        )

    def test_an_undecided_split_keeps_the_previous_behaviour(self):
        """A hand-built ModelMeta with neither count set must not change."""
        legacy = _flash_meta(moe_layers=0, dense_ffn_layers=0)
        ops = {o.name: o for o in _breakdown(legacy).ops}

        assert "gate_proj" not in ops  # as before: MoE model, no dense FFN
        all_layers = next(
            o for o in _breakdown(_flash_meta(moe_layers=45, dense_ffn_layers=0)).ops
            if o.name == "moe_fused"
        )
        assert ops["moe_fused"].flops == pytest.approx(all_layers.flops)

    def test_a_dense_model_is_unaffected(self):
        dense = rc.ModelMeta(
            weight_bytes=16 * 1024**3,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            weight_dtype_bytes=2.0,
            hidden_size=4096,
            intermediate_size=11008,
            vocab_size=32000,
            num_attention_heads=32,
            dense_ffn_layers=32,
        )
        undecided = rc.ModelMeta(**{**dense.__dict__, "dense_ffn_layers": 0})

        a = {o.name: o.flops for o in _breakdown(dense).ops}
        b = {o.name: o.flops for o in _breakdown(undecided).ops}
        assert a == b
        assert "moe_fused" not in a


def _write_config(root: Path, cfg: dict, total_size: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}}), encoding="utf-8"
    )
    return root


class TestLoadModelMetaFillsTheSplit:
    def test_the_dense_prefix_reaches_the_meta(self, tmp_path):
        root = _write_config(
            tmp_path / "glm",
            {
                "num_hidden_layers": 45,
                "first_k_dense_replace": 3,
                "n_routed_experts": 288,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 2048,
                "intermediate_size": 12288,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 154880,
                "torch_dtype": "bfloat16",
            },
            total_size=700 * 1024**3,
        )

        meta = rc.load_model_meta(root)

        assert meta is not None
        assert (meta.moe_layers, meta.dense_ffn_layers) == (42, 3)

    def test_expert_bytes_are_charged_to_moe_layers_only(self, tmp_path):
        cfg = {
            "num_hidden_layers": 45,
            "n_routed_experts": 288,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "intermediate_size": 12288,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 154880,
            "torch_dtype": "bfloat16",
        }
        total = 700 * 1024**3
        no_prefix = rc.load_model_meta(_write_config(tmp_path / "a", cfg, total))
        prefix = rc.load_model_meta(_write_config(tmp_path / "b", {**cfg, "first_k_dense_replace": 3}, total))

        assert no_prefix is not None and prefix is not None
        assert prefix.expert_weight_bytes == pytest.approx(
            no_prefix.expert_weight_bytes * 42 / 45, rel=1e-6
        )
        # Per-token active bytes go *up*, and that is the point. Bytes moved out
        # of the expert pool are dense-FFN weights, and a dense layer's weights
        # are read for every token, whereas only 8 of 288 experts are. Charging
        # them as experts under-counted the decode weight traffic.
        assert prefix.active_weight_bytes > no_prefix.active_weight_bytes

    def test_a_dense_checkpoint_reports_every_layer_as_dense_ffn(self, tmp_path):
        root = _write_config(
            tmp_path / "llama",
            {
                "num_hidden_layers": 32,
                "intermediate_size": 11008,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "torch_dtype": "bfloat16",
            },
            total_size=14 * 1024**3,
        )

        meta = rc.load_model_meta(root)

        assert meta is not None
        assert (meta.moe_layers, meta.dense_ffn_layers) == (0, 32)
