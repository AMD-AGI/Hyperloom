# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise standalone assembly with the Qwen3 example's real ABI and stream."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires ROCm PyTorch", allow_module_level=True)
if not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx950"):
    pytest.skip("Qwen3 assembly example specializes gfx950", allow_module_level=True)


def _reference(qkv, positions, cache, qw, kw):
    q, k, _ = qkv.split([4096, 1024, 1024], -1)
    outputs = []
    for tensor, weight in [(q, qw), (k, kw)]:
        tensor = tensor.reshape(qkv.shape[0], -1, 128).double()
        normalized = (tensor * torch.rsqrt(tensor.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        normalized = (normalized.double() * weight.double()).bfloat16()
        left, right = normalized.chunk(2, -1)
        cos, sin = cache[positions].chunk(2, -1)
        cos, sin = cos[:, None, :].double(), sin[:, None, :].double()
        lc, rs = (left.double() * cos).bfloat16(), (right.double() * sin).bfloat16()
        rc, ls = (right.double() * cos).bfloat16(), (left.double() * sin).bfloat16()
        outputs.append(
            torch.cat(((lc.double() - rs.double()).bfloat16(), (rc.double() + ls.double()).bfloat16()), -1).flatten(1)
        )
    return outputs


def test_qwen3_abi_graph_rebinding_and_wrong_candidate_rejection(tmp_path, monkeypatch):
    example = Path(__file__).resolve().parents[3] / "examples/qwen3-qk-assembly"
    monkeypatch.syspath_prepend(str(example))
    from forge_qwen3_assembly.kernel import QkNormRope

    source = example / "forge_qwen3_assembly/qk_norm_rope.s"
    original = source.read_bytes()
    torch.manual_seed(8128)
    kernel = QkNormRope(tmp_path / "good")
    for tokens, scale in [(1, 1.0), (7, 1e-4), (33, 100.0), (65, 0.0), (129, 1.0)]:
        qkv = torch.randn(tokens, 6144, device="cuda", dtype=torch.bfloat16) * scale
        positions = torch.randint(0, 2048, (tokens,), device="cuda", dtype=torch.int64)
        angles = torch.randn(2048, 64, device="cuda")
        cache = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
        qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn_like(qw)
        args = (qkv, positions, cache, qw, kw, 32, 8, 1e-6)
        preserved = [t.clone() for t in args[:5]]
        actual = kernel(*args)
        expected = _reference(*args[:5])
        torch.cuda.synchronize()
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=1e-2, atol=1e-2)
        for tensor, snapshot in zip(args[:5], preserved):
            assert torch.equal(tensor, snapshot)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = kernel(*args)
    stream.synchronize()
    qkv.mul_(0.5)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    for value, reference in zip(captured, _reference(*args[:5])):
        torch.testing.assert_close(value, reference, rtol=1e-2, atol=1e-2)

    changed = tmp_path / "wrong.s"
    text = original.decode()
    entry = "forge_qk_norm_rope_h128:\n"
    assert text.count(entry) == 1
    changed.write_text(text.replace(entry, entry + "s_endpgm\n"))
    wrong = QkNormRope(tmp_path / "wrong", changed)
    actual = wrong(*args)
    torch.cuda.synchronize()
    with pytest.raises(AssertionError):
        for value, reference in zip(actual, _reference(*args[:5])):
            torch.testing.assert_close(value, reference, rtol=1e-2, atol=1e-2)
    wrong.kernel.close()
    for value, reference in zip(kernel(*args), _reference(*args[:5])):
        torch.testing.assert_close(value, reference, rtol=1e-2, atol=1e-2)
    torch.cuda.synchronize()
    del graph
    kernel.kernel.close()
    assert source.read_bytes() == original


def test_qwen3_normalization_rounding_boundary(tmp_path, monkeypatch):
    example = Path(__file__).resolve().parents[3] / "examples/qwen3-qk-assembly"
    monkeypatch.syspath_prepend(str(example))
    from forge_qwen3_assembly.kernel import QkNormRope

    torch.manual_seed(910)
    qkv = torch.randn(2048, 6144, device="cuda", dtype=torch.bfloat16) * 0.5
    positions = torch.randint(0, 2048, (2048,), device="cuda", dtype=torch.int64)
    angles = torch.randn(2048, 64, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
    qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    kw = torch.randn_like(qw)
    kernel = QkNormRope(tmp_path)
    actual = kernel(qkv, positions, cache, qw, kw, 32, 8, 1e-6)
    expected = _reference(qkv, positions, cache, qw, kw)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=1e-2, atol=1e-2)
    torch.cuda.synchronize()
    kernel.kernel.close()
