# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the predicted-gain calibration, the min-gain gate, source-hint
confirmation, already-fused detection, and the newly added patterns."""

from __future__ import annotations

import json

from kernelforge.fusion import calibration as cal
from kernelforge.fusion.diagnose import diagnose_from_shares, load_op_bytes_from_kineto_trace
from kernelforge.fusion.discover import parse_discovered_recipes
from kernelforge.fusion.locate import build_recipes, covered_by_vllm_compile_pass
from kernelforge.fusion.patterns import PATTERNS, match_patterns
from kernelforge.fusion.vllm_passes import PassState


def _candidate_diag(shares, busy=0.21, **kw):
    return diagnose_from_shares(shares, busy_fraction_of_wall=busy, **kw)


class TestCalibration:
    def test_prior_discounts_share(self):
        # cgnone share overstates cg-ON gain -> prior predicts a small fraction.
        g = cal.predict_cuda_graph_on_gain(0.35, decode_batch=16)
        assert 0 < g < 0.35
        assert abs(g - 0.35 * cal.DEFAULT_SHARE_TO_GAIN_DISCOUNT) < 1e-9

    def test_batch_factor_shrinks_gain(self):
        g16 = cal.predict_cuda_graph_on_gain(0.35, decode_batch=16)
        g64 = cal.predict_cuda_graph_on_gain(0.35, decode_batch=64)
        assert g64 < g16

    def test_measured_points_override_prior(self):
        pts = [(0.20, 0.02), (0.40, 0.06)]
        g = cal.predict_cuda_graph_on_gain(0.30, decode_batch=16, calibration=pts)
        assert abs(g - 0.04) < 1e-6  # linear interp midpoint

    def test_calibration_file_loading(self, tmp_path):
        p = tmp_path / "cal.json"
        p.write_text(json.dumps([{"share": 0.2, "gain": 0.02}, {"share": 0.4, "gain": 0.06}]), encoding="utf-8")
        pts = cal.load_calibration_points(str(p))
        assert pts == [(0.2, 0.02), (0.4, 0.06)]


class TestPredictedGainGate:
    def test_low_predicted_gain_is_annotated_not_vetoed(self):
        # Calibration finding: the share-derived predicted gain is unreliable
        # (under-predicts low-share/high-gain MoE), so it is annotated + surfaced in
        # the reason but does NOT veto a dispatch-bound candidate. The downstream
        # validate/loop measures the real speedup and is the true 3% filter.
        shares = {"gemm": 0.5, "add": 0.18, "rmsnorm": 0.12}  # lb=0.30
        ok = _candidate_diag(shares)
        assert ok.is_candidate
        still = _candidate_diag(shares, min_predicted_gain=0.10)
        assert still.is_candidate  # no longer vetoed by a high predicted-gain bar
        assert "predicted cg-ON gain" in still.reason

    def test_predicted_gain_populated(self):
        d = _candidate_diag({"gemm": 0.5, "add": 0.18, "rmsnorm": 0.12})
        assert d.predicted_e2e_gain > 0
        assert d.to_dict()["predicted_e2e_gain"] == round(d.predicted_e2e_gain, 4)


class TestNewPatterns:
    def test_granite_scaled_residual_triggers(self):
        d = _candidate_diag({"gemm": 0.4, "add": 0.14, "mul": 0.09, "rmsnorm": 0.10})
        ids = [p.id for p, _ in match_patterns(d, "sglang")]
        assert "scaled_residual_add_rmsnorm" in ids

    def test_falcon_h1_scale_combine_triggers(self):
        d = _candidate_diag({"gemm": 0.5, "mul": 0.16, "add": 0.12})
        ids = [p.id for p, _ in match_patterns(d, "sglang")]
        assert "hybrid_scale_combine" in ids

    def test_qk_norm_rope_threshold_raised(self):
        # rmsnorm+rope = 0.05+0.03 = 0.08 < new 0.12 threshold -> does not trigger.
        d = _candidate_diag({"gemm": 0.5, "add": 0.20, "rmsnorm": 0.05, "rope": 0.03})
        ids = [p.id for p, _ in match_patterns(d, "sglang")]
        assert "qk_norm_rope" not in ids

    def test_all_patterns_are_rocm_native(self):
        # Every decode fusion must be authored ROCm-native (review P0-3).
        assert all(p.rocm_native for p in PATTERNS)


def _fake_sglang(tmp_path, model_type: str, body: str):
    """Create a fake sglang tree with one model file; return (model_dir, root)."""
    mdir = tmp_path / "fw" / "python" / "sglang" / "srt" / "models"
    mdir.mkdir(parents=True)
    (mdir / f"{model_type}.py").write_text(body, encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": model_type, "hidden_size": 2048, "num_attention_heads": 16}),
        encoding="utf-8",
    )
    return str(model), str(tmp_path / "fw")


class TestSourceFilteringInLocate:
    def test_eager_source_confirms_residual_pattern(self, tmp_path):
        body = "hidden_states = hidden_states + residual\nx = RMSNorm(2048)\n"
        model, root = _fake_sglang(tmp_path, "eagerlm", body)
        d = _candidate_diag({"gemm": 0.5, "add": 0.18, "rmsnorm": 0.12})
        recipes = build_recipes(d, model_path=model, framework="sglang", framework_root=root)
        residual = next((r for r in recipes if r.pattern_id == "residual_add_rmsnorm"), None)
        assert residual is not None and residual.source_confirmed is True
        assert residual.already_satisfied is False

    def test_already_fused_source_drops_pattern(self, tmp_path):
        # Source already threads residual through the norm -> no-op recipe -> dropped.
        body = "y, residual = self.input_layernorm(hidden_states, residual)\n"
        model, root = _fake_sglang(tmp_path, "fusedlm", body)
        d = _candidate_diag({"gemm": 0.5, "add": 0.18, "rmsnorm": 0.12})
        recipes = build_recipes(d, model_path=model, framework="sglang", framework_root=root)
        assert all(r.pattern_id != "residual_add_rmsnorm" for r in recipes)

    def test_wrong_model_source_drops_pattern(self, tmp_path):
        # Source has none of the residual hints -> pattern not confirmed -> dropped.
        body = "def forward(self, x):\n    return self.mlp(x)\n"
        model, root = _fake_sglang(tmp_path, "otherlm", body)
        d = _candidate_diag({"gemm": 0.5, "add": 0.18, "rmsnorm": 0.12})
        recipes = build_recipes(d, model_path=model, framework="sglang", framework_root=root)
        assert all(r.pattern_id != "residual_add_rmsnorm" for r in recipes)

    def test_include_unconfirmed_keeps_annotated(self, tmp_path):
        body = "y, residual = self.input_layernorm(hidden_states, residual)\n"
        model, root = _fake_sglang(tmp_path, "fusedlm2", body)
        d = _candidate_diag({"gemm": 0.5, "add": 0.18, "rmsnorm": 0.12})
        recipes = build_recipes(d, model_path=model, framework="sglang", framework_root=root, include_unconfirmed=True)
        residual = next((r for r in recipes if r.pattern_id == "residual_add_rmsnorm"), None)
        assert residual is not None and residual.already_satisfied is True


# ───────────────────────── Deliverable 1: memory channel ─────────────────────
class TestMemoryChannelCalibration:
    def test_mem_share_grounds_gain_not_the_flat_discount(self):
        # With a measured memory share the prediction is grounded in bytes saved
        # (mem_share * MEM_SAVED_FRACTION), NOT the flat 0.13 launch-share discount.
        g = cal.predict_cuda_graph_on_gain(0.30, decode_batch=16, mem_share=0.20)
        assert abs(g - 0.20 * cal.DEFAULT_MEM_SAVED_FRACTION) < 1e-9
        legacy = cal.predict_cuda_graph_on_gain(0.30, decode_batch=16)  # discount route
        assert g != legacy

    def test_mem_none_keeps_legacy_discount(self):
        # Default-safe: no memory signal -> unchanged 0.13-discount behavior.
        g = cal.predict_cuda_graph_on_gain(0.30, decode_batch=16, mem_share=None)
        assert abs(g - 0.30 * cal.DEFAULT_SHARE_TO_GAIN_DISCOUNT) < 1e-9

    def test_mem_gain_capped_by_measured_share(self):
        # Cannot save more than the chain's own measured memory traffic.
        g = cal.predict_cuda_graph_on_gain(0.9, decode_batch=16, mem_share=0.05, mem_saved_fraction=5.0)
        assert g <= 0.05


class TestBytesExtraction:
    def _trace(self, tmp_path, events):
        p = tmp_path / "d.trace.json"
        p.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        return p

    def test_bytes_share_from_op_shapes(self, tmp_path):
        p = self._trace(
            tmp_path,
            [
                # add: 2 inputs of [16,2048] float(4B) = 2*16*2048*4 = 262144
                {
                    "cat": "cpu_op",
                    "name": "aten::add",
                    "args": {"Input Dims": [[16, 2048], [16, 2048]], "Input type": ["float", "float"]},
                },
                # rms_norm: 1 input [16,2048] bf16(2B) = 65536
                {
                    "cat": "cpu_op",
                    "name": "aten::rms_norm",
                    "args": {"Input Dims": [[16, 2048]], "Input type": ["c10::BFloat16"]},
                },
                {"cat": "kernel", "name": "Cijk_gemm", "dur": 100},  # kernels carry no shapes
            ],
        )
        bs = load_op_bytes_from_kineto_trace(p)
        assert set(bs) == {"add", "rmsnorm"}
        assert abs(bs["add"] - 262144 / (262144 + 65536)) < 1e-6
        assert abs(sum(bs.values()) - 1.0) < 1e-9

    def test_no_shapes_returns_empty(self, tmp_path):
        # Graph-on / shapeless traces -> {} -> callers fall back to the discount.
        p = self._trace(tmp_path, [{"cat": "kernel", "name": "elementwise", "dur": 10}])
        assert load_op_bytes_from_kineto_trace(p) == {}


class TestMemShareWiredIntoRecipes:
    def test_recipe_mem_share_and_predicted_gain(self, tmp_path):
        body = "hidden_states = hidden_states + residual\nx = RMSNorm(2048)\n"
        model, root = _fake_sglang(tmp_path, "memlm", body)
        d = _candidate_diag(
            {"gemm": 0.5, "add": 0.18, "rmsnorm": 0.12},
            category_bytes_share={"gemm": 0.6, "add": 0.10, "rmsnorm": 0.08},
        )
        recipes = build_recipes(d, model_path=model, framework="sglang", framework_root=root)
        r = next(r for r in recipes if r.pattern_id == "residual_add_rmsnorm")
        assert abs(r.mem_share - 0.18) < 1e-9  # add + rmsnorm bytes share
        expected = cal.predict_cuda_graph_on_gain(r.trigger_share, mem_share=0.18)
        assert abs(r.predicted_gain - expected) < 1e-9

    def test_no_bytes_leaves_mem_share_zero(self, tmp_path):
        body = "hidden_states = hidden_states + residual\nx = RMSNorm(2048)\n"
        model, root = _fake_sglang(tmp_path, "memlm2", body)
        d = _candidate_diag({"gemm": 0.5, "add": 0.18, "rmsnorm": 0.12})  # no bytes
        recipes = build_recipes(d, model_path=model, framework="sglang", framework_root=root)
        r = next(r for r in recipes if r.pattern_id == "residual_add_rmsnorm")
        assert r.mem_share == 0.0  # default-safe


# ─────────────────────── Deliverable 2: compile-pass gate ────────────────────
def _fake_vllm(tmp_path, model_type: str, body: str):
    mdir = tmp_path / "fw" / "vllm" / "model_executor" / "models"
    mdir.mkdir(parents=True)
    (mdir / f"{model_type}.py").write_text(body, encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": model_type, "hidden_size": 2048, "num_attention_heads": 16}),
        encoding="utf-8",
    )
    return str(model), str(tmp_path / "fw")


class TestCompilePassHelper:
    def test_qk_norm_rope_covered_on_vllm_only(self):
        kw = dict(matched_categories=["rmsnorm", "rope"], text="fuse q_norm/k_norm rmsnorm with rope rotary_emb")
        assert covered_by_vllm_compile_pass(framework="vllm", **kw) == "qk_norm_rope"
        assert covered_by_vllm_compile_pass(framework="vllm-aiter", **kw) == "qk_norm_rope"
        assert covered_by_vllm_compile_pass(framework="sglang", **kw) == ""  # not vLLM

    def test_plain_norm_fusion_not_covered(self):
        # residual add + rmsnorm (no quant, no rope) is NOT a vLLM compile pass.
        assert (
            covered_by_vllm_compile_pass(
                framework="vllm",
                matched_categories=["add", "rmsnorm"],
                text="fold residual add into the following rmsnorm",
            )
            == ""
        )

    def test_rope_kvcache_matches_by_keywords(self):
        assert (
            covered_by_vllm_compile_pass(framework="vllm", matched_categories=[], text="fuse rope with kv_cache write")
            == "fuse_rope_kvcache"
        )


def _pass_enabled(flag: str) -> PassState:
    """Probe stub: the matched compile pass IS switched on in the target install.

    Injected so these tests describe the gate rather than whatever vLLM happens to
    be importable; a pass that exists but is OFF is covered in
    ``test_compile_pass_enable.py``.
    """
    return PassState(flag=flag, present=True, enabled=True, config_file="/fw/vllm/config/compilation.py")


class TestCompilePassGateInRoutes:
    def test_pattern_route_drops_qk_on_vllm_keeps_on_sglang(self, tmp_path):
        body = "def _normalize_qk(self):\n    q_norm = 1\n    k_norm = 1\n    return self.rotary_emb(q_norm)\n"
        shares = {"gemm": 0.5, "rmsnorm": 0.12, "rope": 0.06, "add": 0.02}
        (tmp_path / "v").mkdir()
        (tmp_path / "s").mkdir()
        # vLLM: qk_norm_rope is an ENABLED compile pass -> dropped as already-satisfied.
        model_v, root_v = _fake_vllm(tmp_path / "v", "qklm", body)
        dv = _candidate_diag(shares)
        rv = build_recipes(dv, model_path=model_v, framework="vllm", framework_root=root_v, pass_probe=_pass_enabled)
        assert all(r.pattern_id != "qk_norm_rope" for r in rv)
        assert all(r.candidate_kind != "compile_pass" for r in rv)
        # sglang: no compile passes -> the pattern survives (control).
        model_s, root_s = _fake_sglang(tmp_path / "s", "qklm", body)
        ds = _candidate_diag(shares)
        rs = build_recipes(ds, model_path=model_s, framework="sglang", framework_root=root_s)
        assert any(r.pattern_id == "qk_norm_rope" for r in rs)

    def test_discovery_route_drops_compile_covered_proposal(self):
        payload = json.dumps(
            [
                {
                    "name": "rope_kv",
                    "env_flag": "FUSED_ROPE_KV",
                    "op_chain": "rotary_emb + kv_cache write",
                    "fusion_math": "apply rope then write kv_cache",
                    "priority": 0.9,
                },
                {
                    "name": "keep_me",
                    "env_flag": "FUSED_X",
                    "op_chain": "scale add combine",
                    "priority": 0.5,
                },
            ]
        )
        # vLLM route drops the rope+kvcache proposal (fuse_rope_kvcache is enabled).
        rv = parse_discovered_recipes(
            payload, model_type="m", framework="vllm", source_file="/x.py", shapes={}, pass_probe=_pass_enabled
        )
        assert [r.pattern_id for r in rv] == ["llm:keep_me"]
        # sglang route keeps both (no compile passes there).
        rs = parse_discovered_recipes(payload, model_type="m", framework="sglang", source_file="/x.py", shapes={})
        assert {r.pattern_id for r in rs} == {"llm:rope_kv", "llm:keep_me"}
