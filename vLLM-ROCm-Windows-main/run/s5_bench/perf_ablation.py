# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""A ablation (reliable, bypasses the broken profiler): under cudagraph, NO-OP a component (skip its
compute, return a right-shape constant -> no weight read) and measure the decode tok/s ceiling. The
tok/s delta vs baseline = that component's REAL cost. ABLATE in {none, dense, moe, attn}.
- dense: no-op UnquantizedLinearMethod.apply for M=1 (all bf16 dense linears incl. dense MLP + attn proj)
- moe : no-op CompressedTensorsWNA16MoEMethod.apply for M=1 (all W4 experts)
- attn: no-op the attention impl for M=1 decode
Constant 1e-4 (not 0) to avoid rms_norm NaN. Output is garbage; only the STEP TIME matters."""
import os, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("VLLM_WIN_MOE_DECODE", "1")
os.environ.setdefault("VLLM_WIN_ROCM_C", "1")
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")

import torch
from vllm import LLM, SamplingParams

ABLATE = os.environ.get("ABLATE", "none")
GIB = 2 ** 30


def _patch():
    if ABLATE == "dense":
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        _o = UnquantizedLinearMethod.apply
        def apply(self, layer, x, bias=None):
            if x.shape[:-1].numel() == 1:
                return x.new_full((*x.shape[:-1], layer.weight.shape[0]), 1e-4)
            return _o(self, layer, x, bias)
        UnquantizedLinearMethod.apply = apply
        print("ABLATE dense: no-op UnquantizedLinearMethod.apply @ M=1")
    elif ABLATE == "moe":
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
            CompressedTensorsWNA16MoEMethod)
        _o = CompressedTensorsWNA16MoEMethod.apply
        def apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
            if x.shape[0] == 1:
                return x.new_full((x.shape[0], x.shape[-1]), 1e-4)
            return _o(self, layer, x, topk_weights, topk_ids, shared_experts_input)
        CompressedTensorsWNA16MoEMethod.apply = apply
        print("ABLATE moe: no-op CompressedTensorsWNA16MoEMethod.apply @ M=1")
    elif ABLATE == "attn":
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        _o = TritonAttentionImpl.forward
        def forward(self, layer, q, k, v, kv_cache, attn_metadata, output=None, **kw):
            if output is not None and q.shape[0] == output.shape[0] and q.shape[0] <= 8:
                output.fill_(1e-4)
                return output
            return _o(self, layer, q, k, v, kv_cache, attn_metadata, output=output, **kw)
        TritonAttentionImpl.forward = forward
        print("ABLATE attn: no-op TritonAttentionImpl.forward @ decode (KV write + attn compute)")
    elif ABLATE == "attncompute":
        # no-op ONLY unified_attention (the QK/softmax/PV Triton kernel = S5's exact target), keeping
        # rope + KV write. attn_compute fraction = this delta; the rest of attn(full) is rope+kvwrite.
        import vllm.v1.attention.ops.triton_unified_attention as _uam
        import vllm.v1.attention.backends.triton_attn as _tam
        def _noop_ua(q, k, v, out, *a, **kw):
            out.fill_(1e-4)
            return None
        _uam.unified_attention = _noop_ua
        if hasattr(_tam, "unified_attention"):
            _tam.unified_attention = _noop_ua
        print("ABLATE attncompute: no-op unified_attention (keep rope+KV write)")
    elif ABLATE == "kvwrite":
        # no-op ONLY the KV-cache write (reshape_and_cache), keep the attention compute -> isolates
        # the KV-write cost; attn_compute fraction = attn(full) - kvwrite.
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        _m = [n for n in dir(TritonAttentionImpl) if "kv_cache_update" in n or "kv_cache" in n.lower()]
        print("kvwrite candidate methods:", _m)
        if hasattr(TritonAttentionImpl, "do_kv_cache_update"):
            def _noop(self, *a, **k):
                return None
            TritonAttentionImpl.do_kv_cache_update = _noop
            print("ABLATE kvwrite: no-op TritonAttentionImpl.do_kv_cache_update @ decode")


_patch()
free0, total = torch.cuda.mem_get_info()
print(f"== ABLATE={ABLATE} | free {free0/GIB:.1f} GiB")
t0 = time.perf_counter()
llm = LLM(model="lcu0312/gemma-4-26B-A4B-it-AWQ-4bit", dtype="bfloat16", attention_backend="TRITON_ATTN",
    tensor_parallel_size=1, gpu_memory_utilization=0.96, max_model_len=2048, kv_cache_dtype="auto",
    trust_remote_code=True, enforce_eager=False,
    compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
    hf_overrides={"architectures": ["Gemma4ForCausalLM"]})
print(f"engine init: {time.perf_counter()-t0:.1f}s")

prompt = "Write a detailed essay about the history and future of GPU computing. " * 20
llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))
torch.cuda.synchronize()
best = 0.0
for _ in range(3):
    t0 = time.perf_counter()
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tps = len(out[0].outputs[0].token_ids) / dt
    best = max(best, tps)
print(f"ABLATE={ABLATE}  decode tok/s (best of 3): {best:.1f}")
print("ABLATION_OK")
