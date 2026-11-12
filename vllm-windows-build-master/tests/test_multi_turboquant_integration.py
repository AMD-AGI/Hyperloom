"""Exercise all six Multi-TurboQuant cache paths without loading a model."""

from __future__ import annotations

import torch
from packaging.version import Version

import vllm
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

from vllm.v1.attention.ops.multi_turboquant_kv import (
    tq_decode_active_blocks,
)

METHODS = (
    "isoquant3",
    "isoquant4",
    "planarquant3",
    "planarquant4",
    "turboquant25",
    "turboquant35",
)


def main() -> int:
    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")
    torch.manual_seed(16)
    block_size = 16
    head_size = 128
    num_heads = 3

    key = torch.randn((3, num_heads, head_size), device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    slots = torch.tensor([block_size, block_size + 1, -1], device=device, dtype=torch.int64)
    block_table = torch.tensor([[1]], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([2], device=device, dtype=torch.int32)

    for method in METHODS:
        impl = TritonAttentionImpl(
            num_heads=num_heads, num_kv_heads=num_heads, head_size=head_size,
            scale=head_size ** -0.5, alibi_slopes=None, sliding_window=None,
            kv_cache_dtype=method,
        )
        if Version(vllm.__version__) >= Version("0.29.0"):
            cache = torch.zeros(
                (2, num_heads, block_size, 2 * head_size),
                device=device, dtype=torch.uint8,
            )
            key_cache, value_cache = cache.transpose(1, 2).split(head_size, dim=-1)
        else:
            cache = torch.zeros(
                (2, 2, block_size, num_heads, head_size),
                device=device, dtype=torch.uint8,
            )
            key_cache, value_cache = cache.unbind(1)
        impl.do_kv_cache_update(None, key, value, cache, slots)
        decoded_key, decoded_value, remapped = tq_decode_active_blocks(
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            method,
            torch.float16,
            block_size,
        )
        assert decoded_key.shape == (1, block_size, num_heads, head_size)
        assert decoded_value.shape == decoded_key.shape
        assert torch.isfinite(decoded_key).all()
        assert torch.isfinite(decoded_value).all()
        assert remapped.tolist() == [[0]]
        assert not cache[0].any(), "writing block 1 must not corrupt block 0"
        errors = []
        for decoded, original in ((decoded_key, key), (decoded_value, value)):
            expected = original[:2].float()
            relative_l2 = (decoded[0, :2].float() - expected).norm() / expected.norm()
            assert relative_l2 < 0.4, (method, relative_l2.item())
            errors.append(round(relative_l2.item(), 4))
        print(f"{method}: backend write/decode passed, relative L2 {errors}")

    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
