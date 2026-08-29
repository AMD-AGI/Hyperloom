# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Demand has to come from somewhere, and nothing upstream produces it.

The evidence parser, the demand schema and ``--demand`` all existed, but no
caller ever passed one: Hyperloom forwards the serving log and never a demand
file, so shapes kept coming from config.json -- measured at 0.4% coverage of the
keys the runtime actually looks up. The serving log it does forward is the same
log the parser reads, so the demand is derived from it here.

Lines below are transcribed from a real vLLM run on MI355X.
"""

from __future__ import annotations

import json

from kernelforge.gemm_tune import cli

_MISS = (
    "(EngineCore pid=52995) [aiter] shape is M:{m}, N:6144, K:4096 "
    "dtype='torch.bfloat16' otype='torch.bfloat16' bias=False, scaleAB=False, "
    "bpreshuffle=False, not found tuned config in "
    "/tmp/aiter_configs/bf16_tuned_gemm.csv, will use default config!"
)
_HIT = (
    "(EngineCore pid=52995) [aiter] shape is M:8192, N:4096, K:4096 "
    "dtype='torch.bfloat16' otype='torch.bfloat16' bias=False, scaleAB=False, "
    "bpreshuffle=False found padded_M: 8192, N:4096, K:4096 is tuned on "
    "cu_num = 256 in /tmp/aiter_configs/bf16_tuned_gemm.csv, libtype is asm, "
    "kernel name is knl"
)


# Verbatim from a production sglang MiniMax-M3-MXFP4 run (TP8, gfx950).
_MOE_MISS = (
    "[aiter] [fused_moe] no tuned FlyDSL config for "
    "('gfx950', 256, {tok}, 6144, 384, 128, 4, <ActivationType.Swiglu: 2>, "
    "'torch.bfloat16', 'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2', "
    "'QuantType.per_1x32', True, False), using heuristic FlyDSL fallback "
    "(kn1='flydsl_moe1_afp4_wfp4_bf16', kn2='flydsl_moe2_afp4_wfp4_bf16')"
)


def _log(tmp_path, lines):
    p = tmp_path / "server.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


class TestDerivingDemand:
    def test_misses_become_a_demand_file(self, tmp_path):
        src = _log(tmp_path, [_MISS.format(m=512), _MISS.format(m=512), _MISS.format(m=1024), _HIT])
        out = tmp_path / "out"
        out.mkdir()

        path = cli._demand_from_serving_log(src, out)

        assert path == str(out / "demand.json")
        report = json.loads((out / "demand.json").read_text(encoding="utf-8"))
        (entry,) = report["demands"]
        assert entry["tuner"] == "sglang_dense_bf16"
        assert entry["miss_count"] == 3
        # Two distinct keys, most-requested first.
        assert [k["M"] for k in entry["keys"]] == ["512", "1024"]
        assert entry["keys"][0]["requests"] == 2

    def test_a_log_with_no_misses_leaves_the_shape_source_alone(self, tmp_path):
        # An all-hit run has nothing to tune; falling back to config-derived
        # shapes is right, and inventing an empty demand would not be.
        src = _log(tmp_path, [_HIT, _HIT])
        out = tmp_path / "out"
        out.mkdir()

        assert cli._demand_from_serving_log(src, out) == ""
        assert not (out / "demand.json").exists()

    def test_a_moe_only_log_still_produces_a_demand_file(self, tmp_path):
        # The MoE dispatch key does not live in report["demands"], so gating on
        # dense misses threw it away: a pure-MoE model, or one whose dense
        # tables all hit, got no demand file at all and fmoe_ck then skipped
        # itself for "no runtime-observed MoE dispatch key" -- with the key
        # sitting in the log it had just been handed.
        from kernelforge.gemm_tune.evidence import moe_dispatch_keys

        src = _log(tmp_path, [_MOE_MISS.format(tok=16), _HIT])
        out = tmp_path / "out"
        out.mkdir()

        path = cli._demand_from_serving_log(src, out)

        assert path == str(out / "demand.json")
        report = json.loads((out / "demand.json").read_text(encoding="utf-8"))
        assert not report["demands"]  # no dense miss anywhere in this log
        (key,) = moe_dispatch_keys(report)
        assert key["tokens"] == [16]
        assert key["inter_dim"] == "384"

    def test_an_unrelated_log_is_not_demand(self, tmp_path):
        src = _log(tmp_path, ["INFO server started", "INFO ready"])
        out = tmp_path / "out"
        out.mkdir()

        assert cli._demand_from_serving_log(src, out) == ""

    def test_an_unreadable_log_never_fails_the_run(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()

        assert cli._demand_from_serving_log(str(tmp_path / "nope.log"), out) == ""

    def test_an_unwritable_output_dir_never_fails_the_run(self, tmp_path, monkeypatch):
        src = _log(tmp_path, [_MISS.format(m=512)])

        def _boom(*_a, **_k):
            raise OSError("read-only filesystem")

        monkeypatch.setattr("kernelforge.gemm_tune.evidence.write_demand", _boom)
        assert cli._demand_from_serving_log(src, tmp_path) == ""

    def test_the_derived_file_is_what_load_demand_expects(self, tmp_path):
        # It is handed on as if an operator had passed --demand, so it has to
        # round-trip through the same reader.
        from kernelforge.gemm_tune.evidence import demand_for_tuner, demand_shapes, load_demand

        src = _log(tmp_path, [_MISS.format(m=512), _MISS.format(m=1024)])
        out = tmp_path / "out"
        out.mkdir()

        report = load_demand(cli._demand_from_serving_log(src, out))
        shapes = demand_shapes(demand_for_tuner(report, "sglang_dense_bf16"))
        assert [(s["M"], s["N"], s["K"]) for s in shapes] == [
            (512, 6144, 4096),
            (1024, 6144, 4096),
        ]
