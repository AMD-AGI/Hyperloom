# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optional gfx950 regression using AITER's installed INT4/BF16 MoE stage1."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
flyc = pytest.importorskip("flydsl.compiler")

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires an AMD GPU and ROCm PyTorch", allow_module_level=True)
if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
    pytest.skip("requires gfx950 for the AITER W4A16 regression", allow_module_level=True)

pytest.importorskip("aiter")

from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.moe_kernels import _s1_args_std, compile_flydsl_moe_stage1
from aiter.ops.shuffle import pack_int8_to_packed_int4, shuffle_scale_for_int4, shuffle_weight

from kernelforge.assembly.flydsl import with_assembly


def test_aiter_w4a16_roundtrip_rejects_wrong_silu_and_rebinds_arguments(tmp_path, monkeypatch):
    toolchain = Path("/opt/rocm/llvm/bin")
    if not all((toolchain / name).is_file() for name in ("llvm-mc", "ld.lld")):
        pytest.skip("requires ROCm LLVM tools under /opt/rocm/llvm/bin")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path / "dumps"))
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("FLYDSL_COMPILE_LLVM_DIR", raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    model_dim, inter_dim, experts, topk = 1024, 256, 4, 2
    tile_m, tile_n, tile_k = 32, 256, 256
    stream = torch.cuda.Stream()

    def make_case(tokens, seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        random_options = {"device": "cuda", "generator": generator}
        activations = (torch.randn(tokens, model_dim, **random_options) / model_dim**0.5).bfloat16()
        quantized = torch.randint(-8, 8, (experts, 2 * inter_dim, model_dim), dtype=torch.int8, **random_options)
        # Powers of two make dequantization exact, independently of BF16 rounding policy.
        scales = (2.0 ** torch.randint(-5, -2, (experts, model_dim // 32, 2 * inter_dim), **random_options)).bfloat16()
        packed = pack_int8_to_packed_int4(shuffle_weight(quantized, (16, 16)))
        packed = packed.view(experts, 2 * inter_dim, model_dim // 2)
        shuffled_scales = shuffle_scale_for_int4(scales).flatten()
        routes = torch.rand(tokens, experts, **random_options).topk(topk, dim=-1).indices.int()
        weights = torch.rand(tokens, topk, **random_options).softmax(-1)
        ids, sorted_weights, expert_ids, valid_ids, _ = moe_sorting(
            routes, weights, experts, model_dim, torch.bfloat16, block_size=tile_m
        )
        expected = torch.empty(tokens, topk, inter_dim, device="cuda", dtype=torch.bfloat16)
        for expert in range(experts):
            token_ids, slots = torch.where(routes == expert)
            dequantized = quantized[expert].float() * scales[expert].T.repeat_interleave(32, dim=-1).float()
            gate_up = activations[token_ids].float() @ dequantized.T
            gate, up = gate_up.chunk(2, dim=-1)
            expected[token_ids, slots] = (
                torch.nn.functional.silu(gate) * up * weights[token_ids, slots, None]
            ).bfloat16()
        out = torch.full_like(expected, float("nan"))
        empty = torch.empty(0, device="cuda")
        tensors = (out, activations, packed, empty, shuffled_scales, ids, expert_ids, sorted_weights, valid_ids)
        blocks = min(min(tokens * topk * tile_m, ids.numel()) // tile_m, expert_ids.numel())
        args = _s1_args_std(*tensors, tokens, inter_dim, model_dim, blocks, stream)
        return tensors, args, expected

    def check(function, case, exact=None):
        tensors, args, expected = case
        out = tensors[0]
        out.fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        function(*args)
        stream.synchronize()
        torch.testing.assert_close(out, expected, rtol=0.02, atol=0.003)
        if exact is not None:
            torch.testing.assert_close(out, exact, rtol=0, atol=0)
        return out.clone()

    case = make_case(37, 671)
    launcher = compile_flydsl_moe_stage1(
        model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k, True, "bf16", "int4", "bf16"
    )
    stream.wait_stream(torch.cuda.current_stream())
    reference = flyc.compile(launcher, *case[1])
    stream.synchronize()
    baseline = check(reference, case)
    sources = list((tmp_path / "dumps").rglob("*_final_isa.s"))
    assert len(sources) == 1
    assembly = sources[0].read_text()
    target = re.search(r'\.amdgcn_target\s+"amdgcn-amd-amdhsa--([^\"]+)"', assembly)
    assert target is not None
    options = {"gpu_target": target[1], "toolchain_dir": toolchain}
    source = tmp_path / "candidate.s"
    source.write_text(assembly)
    candidate = with_assembly(reference, source, **options)
    check(candidate, case, exact=baseline)

    # Reverse the sign in exp(-x) without changing any addresses or launch metadata.
    wrong_silu, count = re.subn(r"0xbfb8aa3b", "0x3fb8aa3b", assembly)
    assert count > 0
    source.write_text(wrong_silu)
    negative = with_assembly(reference, source, **options)
    with pytest.raises(AssertionError):
        check(negative, case)
    check(reference, case, exact=baseline)
    check(candidate, case, exact=baseline)
    source.write_text(assembly)
    restored = with_assembly(reference, source, **options)
    check(restored, case, exact=baseline)

    for tokens, seed in ((1, 91), (65, 92), (257, 93)):
        changed = make_case(tokens, seed)
        expected = check(reference, changed)
        check(candidate, changed, exact=expected)
        check(restored, changed, exact=expected)

    for _ in range(3):
        restored(*case[1])
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        restored(*case[1])
    case[0][0].fill_(float("nan"))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(case[0][0], baseline, rtol=0, atol=0)
