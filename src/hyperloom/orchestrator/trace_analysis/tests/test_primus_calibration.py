# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A training profile calibrates Primus' projection: each kernel is put on the term that priced it."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace_analysis import primus_calibration as pc

MAIN, AUTOGRAD = 1, 2


class TraceBuilder:
    """A torch profiler trace: kernels launched under nested host frames."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.ts = 0
        self.corr = 0

    def launch(self, kernel: str, dur_us: float, stack: list[tuple[str, str, dict]], tid: int = MAIN) -> None:
        """``stack`` is outermost first: ``(cat, name, args)``; module frames are ``nn.Module: X``."""
        self.ts += 1000
        self.corr += 1
        for depth, (cat, name, args) in enumerate(stack):
            self.events.append(
                {
                    "ph": "X",
                    "cat": cat,
                    "name": name,
                    "tid": tid,
                    "ts": self.ts + depth,
                    "dur": 500 - 2 * depth,
                    "args": args,
                }
            )
        at = self.ts + len(stack) + 10
        self.events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "hipLaunchKernel",
                "tid": tid,
                "ts": at,
                "dur": 5,
                "args": {"correlation": self.corr},
            }
        )
        self.events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": kernel,
                "tid": 7,
                "ts": at + 20,
                "dur": dur_us,
                "args": {"correlation": self.corr},
            }
        )

    def write(self, path: Path, step_us: float) -> str:
        self.events.append(
            {"ph": "X", "cat": "gpu_user_annotation", "name": "ProfilerStep#10", "tid": 7, "ts": 0, "dur": step_us}
        )
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump({"traceEvents": self.events}, fh)
        return str(path)


def _mod(name: str) -> tuple[str, str, dict]:
    return ("python_function", f"nn.Module: {name}", {})


def _py(name: str) -> tuple[str, str, dict]:
    return ("python_function", name, {})


def write_training_trace(path: Path) -> str:
    """One MoE layer and the output layer, a backward expert GEMM, the DP reduce and a pipeline send/recv."""
    t = TraceBuilder()
    layer = [_mod("GPTModel"), _mod("TransformerLayer_0")]
    t.launch("Cijk_Alik_Bljk_qkv", 6000, layer + [_mod("SelfAttention"), _mod("linear_qkv")])
    experts = layer + [_mod("MoELayer_0"), _mod("Experts"), ("cpu_op", "aten::_grouped_mm", {"Sequence number": 7})]
    t.launch("kernel_grouped_gemm_fwd", 1000, experts)
    t.launch(
        "kernel_grouped_gemm_bwd",
        4000,
        [("cpu_op", "autograd::engine::evaluate_function: GroupedMmBackward0", {"Sequence number": 7})],
        tid=AUTOGRAD,
    )
    t.launch("Cijk_Alik_Bljk_lm_head", 5000, [_mod("GPTModel"), _mod("output_layer")])
    t.launch("triton_cross_entropy_fwd", 2500, [_mod("GPTModel"), _py("fused_cross_entropy.py(20): forward")])
    t.launch("ncclDevKernel_Generic_1", 3000, [("user_annotation", "nccl:reduce_scatter_tensor_coalesced", {})])
    t.launch("ncclDevKernel_Generic_1", 100000, [_py("p2p_communication.py(505): send_forward_recv_backward")])
    return t.write(path, step_us=130000)


def _row(
    op: str,
    phase: str,
    kind: str,
    base_ms: float,
    *,
    scope: str = "moe",
    stage: int = 0,
    calls: int = 1,
    dims: dict | None = None,
) -> dict:
    return {
        "stage": stage, "scope": scope, "op": op, "phase": phase, "kind": kind, "dims": dims or {},
        "est_ms": base_ms, "base_ms": base_ms, "calls": calls, "est_total_ms": base_ms * calls, "src": "origami",
    }  # fmt: skip


def training_report(pp: int = 1) -> dict:
    """The uncalibrated report of the run :func:`write_training_trace` profiles."""
    stage_layers = [[0]] if pp == 1 else [[0], [1]]
    return {
        "schema": pc.REPORT_SCHEMA,
        "version": 1,
        "deployment": {
            "tp": 1,
            "pp": pp,
            "ep": 8,
            "cp": 1,
            "dp": 8 // pp,
            "gpus": 8,
            "microbatches": 1,
            "stage_layers": stage_layers,
            "layer_scopes": ["moe"] * pp,
        },
        "iteration": {"iteration_time_ms": 20.0},
        "kernels": [
            *(
                _row(
                    f"attn_{x}",
                    "fwd",
                    "gemm",
                    1.0,
                    dims={"m": 4096, "n": 512, "k": 2880, "dtype": "bf16", "layout": "NT"},
                )
                for x in "qkv"
            ),
            _row("moe_expert_gate", "fwd", "grouped_gemm", 2.0),
            _row("moe_expert_gate", "dgrad", "grouped_gemm", 1.0),
            _row("moe_expert_gate", "wgrad", "grouped_gemm", 1.0),
            _row("lm_head", "fwd", "gemm", 5.0, scope="output"),
            _row("dp_grad_reduce", "step", "comm", 1.0, scope="iteration"),
        ],
    }


@pytest.fixture
def trace(tmp_path):
    return write_training_trace(tmp_path / "exp-rank[0].pt.trace.json.gz")


def test_each_kernel_lands_on_the_term_that_priced_it(trace):
    measured = pc.classify_trace(trace, pc.layers_for_stage(training_report(), 0))
    assert measured.ms == pytest.approx(
        {
            ("moe", "attn_qkv", "fwd"): 6.0,
            ("moe", "moe_expert", "fwd"): 1.0,
            ("moe", "moe_expert", "bwd"): 4.0,
            ("output", "lm_head", "fwd"): 5.0,
            ("output", "cross_entropy", "fwd"): 2.5,
            ("iteration", "dp_grad_reduce", "step"): 3.0,
            ("iteration", pc.PP_P2P, pc.WAIT): 100.0,
        }
    )
    assert measured.step_span_ms == pytest.approx(130.0)


def test_calibration_scales_priced_classes_and_adds_what_the_model_misses(trace):
    cal = pc.build_calibration(training_report(), [trace])
    factors = {(f["op"], f["phase"]): f["factor"] for f in cal["factors"]}
    assert factors[("attn_q", "fwd")] == factors[("attn_v", "fwd")] == pytest.approx(2.0)
    assert factors[("moe_expert_gate", "fwd")] == pytest.approx(0.5)
    assert factors[("moe_expert_gate", "dgrad")] == factors[("moe_expert_gate", "wgrad")] == pytest.approx(2.0)
    assert factors[("lm_head", "fwd")] == pytest.approx(1.0)
    assert factors[("dp_grad_reduce", "step")] == pytest.approx(3.0)
    assert [(e["op"], e["scope"], e["phase"], e["share"]) for e in cal["extras"]] == [
        ("cross_entropy", "output", "fwd", 0.5)
    ]
    assert cal["measured"]["0"]["pp_p2p_ms"] == pytest.approx(100.0)


def test_pipeline_waits_stay_out_of_the_calibration_and_the_mix(trace):
    cal = pc.build_calibration(training_report(), [trace])
    assert all(e["op"] != pc.PP_P2P for e in cal["extras"]) and all(f["op_class"] != pc.PP_P2P for f in cal["factors"])
    mix = pc.mix_distance(training_report(), trace)
    assert mix["pp_p2p_ms"] == pytest.approx(100.0)
    assert mix["measured_ms"] == pytest.approx(21.5)
    assert all(not r["class"].startswith(pc.PP_P2P) for r in mix["classes"])


def test_a_report_calibrated_to_its_trace_has_the_trace_mix(trace):
    report = training_report()
    cal = pc.build_calibration(report, [trace])
    factor = {(f["op"], f["phase"]): f["factor"] for f in cal["factors"]}
    for row in report["kernels"]:
        row["est_ms"] = row["base_ms"] * factor[(row["op"], row["phase"])]
    report["kernels"].append(_row("cross_entropy", "fwd", "engine", 2.5, scope="output") | {"src": "calibration"})
    assert pc.mix_distance(report, trace)["total_variation"] == pytest.approx(0.0, abs=1e-6)
    assert pc.mix_distance(training_report(), trace)["total_variation"] > 0.2


def test_ranks_map_to_stages_with_pipeline_outermost():
    report = training_report(pp=2)
    assert [pc.stage_of_rank(report, r) for r in (0, 3, 4, 7)] == [0, 0, 1, 1]
    assert pc._rank_of("primus-megatron-exp[gpt_oss]-rank[5].123.pt.trace.json.gz") == 5


def test_report_ops_share_megatron_kernel_classes():
    assert [pc.op_class(o) for o in ("attn_k", "moe_expert_down", "moe_shared_up", "mlp_gate", "moe_unpermute", "attn_o")] == [
        "attn_qkv", "moe_expert", "moe_shared", "mlp", "moe_permute", "attn_o",
    ]  # fmt: skip
    assert [pc.direction(p) for p in ("fwd", "dgrad", "wgrad", "bwd", "step")] == ["fwd", "bwd", "bwd", "bwd", "step"]


def test_global_collectives_cost_what_the_last_stage_to_arrive_spends(tmp_path):
    traces = []
    for rank, waited_us in ((0, 2000), (4, 50000)):
        t = TraceBuilder()
        t.launch("Cijk_Alik_Bljk_qkv", 6000, [_mod("GPTModel"), _mod("TransformerLayer_0"), _mod("linear_qkv")])
        t.launch("ncclDevKernel_Generic_1", waited_us, [("user_annotation", "nccl:all_reduce", {})])
        traces.append(t.write(tmp_path / f"exp-rank[{rank}].pt.trace.json.gz", step_us=60000))
    measured, _model, per_stage = pc._pool(training_report(pp=2), traces)
    key = ("iteration", "comm_other", "step")
    assert per_stage[1].ms[key] == pytest.approx(50.0)
    assert measured[key] == pytest.approx(2 * 2.0)


def test_steady_state_time_scales_out_the_profiler(trace):
    plain = pc.build_calibration(training_report(), [trace])
    steady = pc.build_calibration(training_report(), [trace], steady_ms=65.0)
    assert steady["profiler_scale"] == pytest.approx(0.5)
    assert {f["op"]: f["factor"] for f in steady["factors"]}["lm_head"] == pytest.approx(
        0.5 * {f["op"]: f["factor"] for f in plain["factors"]}["lm_head"]
    )
