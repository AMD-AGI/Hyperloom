# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write the synthetic trace fixtures of this directory: ``python _make_traces.py [directory]``.

Nothing here was measured. Every trace is made of a made-up decoder (a ``DecoderModel`` of a few ``DecoderLayer``
modules, each a ``DecoderAttention`` and a ``DecoderMLP``) whose kernels carry the names of the public kernel libraries a
serving framework launches (aiter, CK, hipBLASLt, PyTorch, a FlyDSL GEMM), with round durations chosen so that each
trace has the structure its tests check, not any measured value:

* ``decode_*``: one decode step served as a CUDA-graph replay (FlyDSL GEMMs). ``decode_capture_bs64_launches`` is the
  graph capture of batch size 64, its launches inside the ``nn.Module`` frames that made them. The window
  (``decode_window_launch_events``, and the same GPU events alone in ``decode_window_gpu_events``) records the replay's
  kernels in the capture's slots but, for its last two layers, each under the name of the slot two places on, and six
  eager kernels queued behind the replay are stamped as starting inside it and ending at the window's end.
  ``decode_rank_trace_gpu_events`` is the window and a later eager step where those six run undisturbed.
* ``lost_replay_window_gpu_events``: a sibling decode step whose replay recorded no kernels, only the eager ones around
  the gap where it ran.
* ``root_window_gpu_events`` and ``root_extend_window_launch_events``: eager prefill steps (CK blockscale GEMMs); the
  first with one GEMM that starts under a microsecond before its predecessor ended, the second with every launch call
  and module frame.
* ``eager_window_gpu_events`` and ``eager_tp2_window_gpu_events``: eager-mode decode steps (hipBLASLt GEMMs, many short
  kernels and launch gaps), the second at TP2 with its all-reduces.
* ``nodewise_*`` and ``final_probe_*``: decode steps replayed node by node (CK blockscale GEMMs), every replay with all
  of its kernels in the capture's order, and the capture they replay; ``final_probe_prefill2_*`` two eager prefill steps.
  The two decode windows differ in how long paged attention takes. The probe's step times are chosen so that a wave of
  64 requests of 1024 prompt and 1024 output tokens spends exactly a tenth of its time in prefill.

The tests assert the shares and times these durations give; change a duration and regenerate, and those expectations
change with it.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

#: The GPU events' ``pid`` (the device) and stream, and the host thread's pid and tid.
DEVICE, STREAM, HOST = 0, 3, 4242
#: Where every trace's clock starts, in microseconds.
T0 = 1_000_000.0

# --- kernel names: the public libraries' own -----------------------------------------------------------------------------

EMBED = "void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)"
NORM = "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb1ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_dbiiiiiiibb"
NORM_LAST = "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi24ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_dbiiiiiiibb"
NORM_HEAD = "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_dbiiiiiiibb"
TOKEN_QUANT = "_ZN5aiter37dynamic_per_token_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPT_PKfiPKii"
TOKEN_QUANT0 = "_ZN5aiter37dynamic_per_token_scaled_quant_kernelIDF16bDB8_Li0EEEvPT0_PfPT_PKfiPKii"
GROUP_QUANT = (
    "_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb0ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_"
    "PfPKT_PKfliilPKii"
)
FLYDSL_GEMM = "kernel_gemm_0"
CK_GEMM = (
    "_ZN2ck59kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffleINS_58GridwiseGemmMultiD_blockscale_xdl_cshuffle_"
    "v3_b_preshuffleILi256ELi16ELi128EEELb1ELi1EEEvNT_8ArgumentE"
)
CK_GEMM_WIDE = (
    "_ZN2ck59kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffleINS_58GridwiseGemmMultiD_blockscale_xdl_cshuffle_"
    "v3_b_preshuffleILi256ELi128ELi128EEELb1ELi1EEEvNT_8ArgumentE"
)
QK_NORM_ROPE = "_inplace_qk_norm_rope_neox_kernel"
RESHAPE_CACHE = "reshape_and_cache_flash"
STORE_KV = "void sglang::store_kvcache_kernel<1024l, 1024l, 128u, false, long>(sglang::StoreKVCacheParams)"
ROTARY = (
    "void rotary_embedding_kernel<c10::BFloat16, float, true>(long const*, c10::BFloat16*, c10::BFloat16*, float const*, "
    "int, long, long, long, int, int, int)"
)
PAGED_FP8 = (
    "void paged_attention_ll4mi_QKV_mfma16_kernel<0, __hip_bfloat16, unsigned char, (vllm::Fp8KVCacheDataType)1, 1, 128, "
    "256, false, 4, 1, ck_tile::ComposedAttention<0u, false> >(__hip_bfloat16 const*, unsigned char const*, unsigned char "
    "const*, float, int const*, int const*, int const*, float const*, int, int, int, int, float*, float*, __hip_bfloat16*, "
    "float, float, float const*, float const*, float const*)"
)
PAGED_BF16 = (
    "void paged_attention_ll4mi_QKV_mfma16_kernel<0, __hip_bfloat16, __hip_bfloat16, (vllm::Fp8KVCacheDataType)0, 1, 128, "
    "256, false, 4, 1, ck_tile::ComposedAttention<0u, false> >(__hip_bfloat16 const*, __hip_bfloat16 const*, "
    "__hip_bfloat16 const*, float, int const*, int const*, int const*, float const*, int, int, int, int, float*, float*, "
    "__hip_bfloat16*, float, float, float const*, float const*, float const*)"
)
PAGED_REDUCE = (
    "void paged_attention_ll4mi_reduce_kernel<__hip_bfloat16, __hip_bfloat16, 128, 128, 256, 1, false>(__hip_bfloat16*, "
    "float const*, float const*, __hip_bfloat16 const*, int const*, int const*, int, int, float const*)"
)
PREFILL_ATTENTION = (
    "_ZN7ck_tile6kentryILi2ENS_38FmhaBatchPrefillWithPagedKVCacheKernelINS_40BlockFmhaBatchPrefillPipelineQRKSVSAsyncILi"
    "128EEEEEEvT0_"
)
ACT = "_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16TnPFT_RKS3_EXadL_Z4siluIS2_ES3_S5_EEEEvPS3_PS4_i"
TENSILE = "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT{tile}_MI16x16x1_SN_LDSB1_AFC0_AFEM1_ASEM1_CLR1_ISA950_K1"
TO_FLOAT = (
    "void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::"
    "TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_"
    "kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)"
)
FILL = (
    "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, std::array<char*, 1ul> >(int, "
    "at::native::FillFunctor<long>, std::array<char*, 1ul>)"
)
COPY = (
    "void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_"
    "kernel_cuda(at::TensorIteratorBase&)::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda("
    "at::TensorIteratorBase&)::{lambda(long)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::"
    "native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::"
    "direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(long)#1} const&)::{lambda(int)#1})"
)
MUL = (
    "void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor"
    "<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::"
    "BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int, bool)#1}>("
    "int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::"
    "MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::"
    "binary_internal::MulFunctor<float> > const&)::{lambda(int, bool)#1})"
)


def _index(kind: str, size: int) -> str:
    impl = f"at::native::{kind}_kernel_impl<at::native::OpaqueType<{size}> >"
    iterator = "at::TensorIterator&" if kind == "index_put" else "at::TensorIteratorBase&"
    lam = f"{impl}({iterator}, c10::ArrayRef<long>, c10::ArrayRef<long>)::{{lambda(char*, char const*, long)#1}}"
    return (
        f"void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<{lam}>(at::TensorIteratorBase&, "
        f"c10::ArrayRef<long>, c10::ArrayRef<long>, {lam} const&)::{{lambda(int)#1}}>(long, {{lambda(int)#1}})"
    )


def _vectorized(functor: str, arrays: int = 2) -> str:
    array = f"std::array<char*, {arrays}ul>"
    return f"void at::native::vectorized_elementwise_kernel<4, {functor}, {array} >(int, {functor}, {array})"


def _reduce(ops: str, out: str) -> str:
    op = f"at::native::ReduceOp<float, {ops}, unsigned int, {out}, 4, 4>"
    return f"void at::native::reduce_kernel<512, 1, {op} >({op})"


INDEX_PUT = _index("index_put", 8)
INDEX = _index("index", 8)
ADD_LONG = _vectorized("at::native::CUDAFunctorOnSelf_add<long>")
ADD_INT = _vectorized("at::native::CUDAFunctorOnSelf_add<int>")
ADD_SELF = _vectorized("at::native::CUDAFunctorOnSelf_add<float>")
ADD = _vectorized("at::native::CUDAFunctor_add<float>", 3)
POW = _vectorized(
    "at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::"
    "{lambda(float)#1}"
)
RSQRT = _vectorized("at::native::rsqrt_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}")
TO_BF16 = _vectorized("at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}")
MEAN = _reduce("at::native::MeanOps<float, float, float, float>", "float")
ARGMAX = _reduce("at::native::ArgMaxOps<float>", "long")
GREEDY = "void aiter::greedy_sample_kernel<float, 1024, 16>(float const*, int*, int, int)"
SMALL_COPY = "_small_copy_kernel"
KV_INDICES = "create_flashinfer_kv_indices_triton"
WRITE_REQ = "write_req_to_token_pool_triton"
VOCAB_EMBED = "_vocab_parallel_embedding_kernel"
CLAMP = "void sglang::clamp_position_kernel<long>(long*, long const*, unsigned long)"
SCAN = (
    "void rocprim::detail::trampoline_kernel<rocprim::detail::wrapped_scan_config<rocprim::default_config, long> >("
    "rocprim::detail::scan_args<long>)"
)
CROSS_REDUCE = (
    "_ZN5aiter26cross_device_reduce_1stageIDF16bLi2ELb0EEEvPNS_8RankDataES2_NS_11RankSignalsEPNS_6SignalEPT_ii"
)
AR_LL = "_ZN5aiter10ar_ll_gfx9IDF16bLi2ELi256EEEvPKPT_S2_PKS1_miPjm"
DTOD, HTOD = "Memcpy DtoD (Device -> Device)", "Memcpy HtoD (Host -> Device)"
#: The kernels a Triton JIT compiled, launched through the module API.
_MODULE_LAUNCHED = frozenset({SMALL_COPY, KV_INDICES, WRITE_REQ, VOCAB_EMBED})

Path_ = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Kernel:
    """One kernel (or copy) a step runs: its name, how long it takes, and the module instances around its launch, outermost
    first, as ``(class, instance key)``."""

    name: str
    us: float
    path: Path_ = ()

    @property
    def category(self) -> str:
        return "gpu_memcpy" if self.name.startswith("Memcpy") else "kernel"


# --- the made-up decoder's modules ----------------------------------------------------------------------------------------

MODEL: Path_ = (("DecoderModel", "model"),)


def _layer(n: int) -> Path_:
    return (*MODEL, ("DecoderLayer", f"layers.{n}"))


def _attention(n: int, *inner: tuple[str, str]) -> Path_:
    return (*_layer(n), ("DecoderAttention", f"layers.{n}.self_attn"), *inner)


def _mlp(n: int, *inner: tuple[str, str]) -> Path_:
    return (*_layer(n), ("DecoderMLP", f"layers.{n}.mlp"), *inner)


def _norm(n: int, which: str) -> Path_:
    return (*_layer(n), ("RMSNorm", f"layers.{n}.{which}"))


def _qkv(n: int) -> Path_:
    return _attention(n, ("QKVParallelLinear", f"layers.{n}.qkv_proj"))


def _o(n: int) -> Path_:
    return _attention(n, ("RowParallelLinear", f"layers.{n}.o_proj"))


def _radix(n: int) -> Path_:
    return _attention(n, ("RadixAttention", f"layers.{n}.attn"))


def _gate_up(n: int) -> Path_:
    return _mlp(n, ("MergedColumnParallelLinear", f"layers.{n}.gate_up_proj"))


def _down(n: int) -> Path_:
    return _mlp(n, ("RowParallelLinear", f"layers.{n}.down_proj"))


HEAD: Path_ = (*MODEL, ("VocabParallelEmbedding", "embed_tokens"))
FINAL_NORM: Path_ = (*MODEL, ("RMSNorm", "norm"))
LOGITS: Path_ = (("LogitsProcessor", "logits_processor"),)
SAMPLER: Path_ = (("Sampler", "sampler"),)


@dataclass(frozen=True)
class Gemms:
    """The four GEMMs of a layer, in microseconds."""

    qkv: float
    o: float
    gate_up: float
    down: float


def flydsl_decode_layer(n: int, gemms: Gemms) -> list[Kernel]:
    """A decode layer of FlyDSL GEMMs behind per-token fp8 quantisation, fp8 KV cache."""
    return [
        Kernel(NORM, 2.0, _norm(n, "input_layernorm")),
        Kernel(TOKEN_QUANT, 1.0, _qkv(n)),
        Kernel(FLYDSL_GEMM, gemms.qkv, _qkv(n)),
        Kernel(QK_NORM_ROPE, 3.0, _attention(n)),
        Kernel(RESHAPE_CACHE, 2.0, _radix(n)),
        Kernel(PAGED_FP8, 20.0, _radix(n)),
        Kernel(PAGED_REDUCE, 4.0, _radix(n)),
        Kernel(TOKEN_QUANT, 1.0, _o(n)),
        Kernel(FLYDSL_GEMM, gemms.o, _o(n)),
        Kernel(NORM, 2.0, _norm(n, "post_attention_layernorm")),
        Kernel(TOKEN_QUANT, 1.0, _gate_up(n)),
        Kernel(FLYDSL_GEMM, gemms.gate_up, _gate_up(n)),
        Kernel(ACT, 4.0, _mlp(n, ("SiluAndMul", f"layers.{n}.act_fn"))),
        Kernel(TOKEN_QUANT0, 1.0, _down(n)),
        Kernel(FLYDSL_GEMM, gemms.down, _down(n)),
    ]


def ck_layer(n: int, gemms: Gemms, attention: Kernel, small: float) -> list[Kernel]:
    """A layer of CK blockscale GEMMs behind per-group fp8 quantisation, with q/k norms; ``small`` is what each of its
    short kernels takes."""
    copy = Kernel(COPY, small, _attention(n))
    head_norm = Kernel(NORM_HEAD, small, _attention(n, ("RMSNorm", f"layers.{n}.q_norm")))
    return [
        Kernel(NORM, 1.5 * small, _norm(n, "input_layernorm")),
        Kernel(GROUP_QUANT, small, _qkv(n)),
        Kernel(CK_GEMM, gemms.qkv, _qkv(n)),
        copy,
        head_norm,
        copy,
        replace(head_norm, path=_attention(n, ("RMSNorm", f"layers.{n}.k_norm"))),
        Kernel(ROTARY, 1.5 * small, _attention(n, ("RotaryEmbedding", f"layers.{n}.rotary_emb"))),
        Kernel(STORE_KV, small, _radix(n)),
        replace(attention, path=_radix(n)),
        *([Kernel(PAGED_REDUCE, 2.5 * small, _radix(n))] if attention.name == PAGED_BF16 else []),
        Kernel(GROUP_QUANT, small, _o(n)),
        Kernel(CK_GEMM, gemms.o, _o(n)),
        Kernel(NORM, 1.5 * small, _norm(n, "post_attention_layernorm")),
        Kernel(GROUP_QUANT, small, _gate_up(n)),
        Kernel(CK_GEMM_WIDE, gemms.gate_up, _gate_up(n)),
        Kernel(ACT, 2.5 * small, _mlp(n, ("SiluAndMul", f"layers.{n}.act_fn"))),
        Kernel(GROUP_QUANT, small, _down(n)),
        Kernel(CK_GEMM_WIDE, gemms.down, _down(n)),
    ]


def model_step(layers: list[list[Kernel]], embed: float, norm: float, logits: float, to_float: float) -> list[Kernel]:
    """The kernels of one forward: the embedding, the layers, the final norm and the logits."""
    return [
        Kernel(EMBED, embed, HEAD),
        *(kernel for layer in layers for kernel in layer),
        Kernel(NORM_LAST, norm, FINAL_NORM),
        Kernel(TENSILE.format(tile="128x64x128"), logits, LOGITS),
        Kernel(TO_FLOAT, to_float, LOGITS),
    ]


#: The FlyDSL decode step's GEMMs, and how much each layer's take of them: every layer's copy of a GEMM within a few
#: percent of the others.
FLYDSL_GEMMS = Gemms(qkv=12.0, o=10.0, gate_up=40.0, down=24.0)
LAYER_FACTORS = (1.00, 1.04, 0.96, 1.02)


def flydsl_decode_step() -> list[Kernel]:
    layers = []
    for n, factor in enumerate(LAYER_FACTORS):
        g = FLYDSL_GEMMS
        layers.append(flydsl_decode_layer(n, Gemms(g.qkv * factor, g.o * factor, g.gate_up * factor, g.down * factor)))
    return model_step(layers, embed=3.0, norm=2.0, logits=30.0, to_float=2.0)


#: The CK decode step's GEMMs (the node-by-node profiles), every layer alike.
CK_DECODE_GEMMS = Gemms(qkv=16.0, o=14.0, gate_up=30.0, down=40.0)


def ck_decode_step(paged_us: float, layers: int = 4) -> list[Kernel]:
    attention = Kernel(PAGED_BF16, paged_us)
    return model_step(
        [ck_layer(n, CK_DECODE_GEMMS, attention, small=2.0) for n in range(layers)],
        embed=4.0,
        norm=3.0,
        logits=40.0,
        to_float=3.0,
    )


# --- writing events -----------------------------------------------------------------------------------------------------


def _round(value: float) -> float:
    return round(value, 3)


@dataclass
class Trace:
    """The events of one trace file, with the correlation ids it hands out."""

    events: list[dict[str, Any]] = field(default_factory=list)
    next_correlation: int = 5001

    def correlation(self) -> int:
        self.next_correlation += 1
        return self.next_correlation - 1

    def gpu(self, kernel: Kernel, ts: float, dur: float, correlation: int) -> None:
        args: dict[str, Any] = {"device": DEVICE, "stream": STREAM, "correlation": correlation}
        if kernel.category == "kernel":
            args |= {"kind": "Dispatch Kernel", "grid": [64, 1, 1], "block": [256, 1, 1]}
        else:
            args |= {"kind": kernel.name.split()[1], "bytes": 4096}
        self._add("X", kernel.category, kernel.name, DEVICE, STREAM, ts, dur, args)

    def call(self, name: str, ts: float, dur: float, correlation: int | None = None, kernel: str = "") -> None:
        args: dict[str, Any] = {}
        if correlation is not None:
            args |= {"cid": correlation % 1000, "correlation": correlation}
        if kernel:
            args |= {"kernel": kernel, "grid": [64, 1, 1], "block": [256, 1, 1], "shared memory": 0}
        self._add("X", "cuda_runtime", name, HOST, HOST, ts, dur, args)

    def launch(self, kernel: Kernel, ts: float, dur: float, correlation: int) -> None:
        if kernel.category == "gpu_memcpy":
            self.call("hipMemcpyAsync", ts, dur, correlation)
        else:
            call = "hipModuleLaunchKernel" if kernel.name in _MODULE_LAUNCHED else "hipLaunchKernel"
            self.call(call, ts, dur, correlation, kernel.name)

    def frame(self, module: str, ts: float, dur: float) -> None:
        self._add("X", "python_function", f"nn.Module: {module}", HOST, HOST, ts, dur, {})

    def annotation(self, name: str, ts: float, dur: float) -> None:
        self._add("X", "user_annotation", name, HOST, HOST, ts, dur, {})

    def _add(self, ph: str, cat: str, name: str, pid: int, tid: int, ts: float, dur: float, args: dict) -> None:
        event = {"ph": ph, "cat": cat, "name": name, "pid": pid, "tid": tid, "ts": _round(ts), "dur": _round(dur)}
        self.events.append(event | {"args": args})

    def gpu_only(self) -> Trace:
        return Trace([e for e in self.events if e["cat"] in ("kernel", "gpu_memcpy", "gpu_memset")])

    def payload(self) -> dict[str, Any]:
        return {"traceEvents": sorted(self.events, key=lambda e: (e["ts"], e["cat"] != "python_function", -e["dur"]))}


def module_frames(trace: Trace, calls: Sequence[tuple[Path_, float, float]]) -> None:
    """The ``nn.Module`` frames around the launch calls ``calls`` (``(path, start, end)``, in call order), each module
    instance numbered per class in the order it first runs, as PyTorch's profiler names them."""
    numbers: dict[str, dict[str, int]] = defaultdict(dict)
    stack: list[tuple[str, str, float]] = []

    def close(depth: int, at: float) -> None:
        """End the frames deeper than ``depth``, the innermost at ``at`` and each one around it a little later."""
        while len(stack) > depth:
            cls, key, start = stack.pop()
            trace.frame(f"{cls}_{numbers[cls][key]}", start, at - start)
            at += 0.05

    last_end = 0.0
    for path, start, end in calls:
        common = 0
        while common < min(len(stack), len(path)) and stack[common][:2] == path[common]:
            common += 1
        close(common, last_end + 0.5)
        for depth in range(common, len(path)):
            cls, key = path[depth]
            numbers[cls].setdefault(key, len(numbers[cls]))
            stack.append((cls, key, start - 1.0 + 0.1 * depth))
        last_end = end
    close(0, last_end + 0.5)


def eager_step(trace: Trace, kernels: Sequence[Kernel], host: float, gpu: float, gaps: Sequence[float]) -> float:
    """``kernels`` launched one by one from ``host`` on, each in its modules, and run from ``gpu`` on with ``gaps[i]``
    before the i-th; the GPU time the last one ends."""
    calls = []
    for kernel, gap in zip(kernels, gaps, strict=True):
        correlation = trace.correlation()
        trace.launch(kernel, host, 4.0, correlation)
        calls.append((kernel.path, host, host + 4.0))
        gpu = max(gpu + gap, host + 6.0)
        trace.gpu(kernel, gpu, kernel.us, correlation)
        gpu += kernel.us
        host += 8.0
    module_frames(trace, calls)
    return gpu


def gpu_run(trace: Trace, kernels: Sequence[Kernel], start: float, gap: float, correlation: int) -> list[float]:
    """``kernels`` one after another from ``start``, ``gap`` apart, under one correlation id; their start times."""
    starts = []
    for kernel in kernels:
        starts.append(start)
        trace.gpu(kernel, start, kernel.us, correlation)
        start += kernel.us + gap
    return starts


def capture(kernels: Sequence[Kernel]) -> Trace:
    """The graph capture of ``kernels``: two fills that set it up, then every launch between the capture's begin and end
    calls, inside the module frames that made it."""
    trace = Trace(next_correlation=101)
    host = T0
    for _ in range(2):
        trace.call("hipLaunchKernel", host, 5.0, trace.correlation(), FILL)
        host += 10.0
    trace.call("hipStreamBeginCapture", host, 3.0)
    host += 10.0
    calls = []
    for kernel in kernels:
        trace.launch(kernel, host, 4.0, trace.correlation())
        calls.append((kernel.path, host, host + 4.0))
        host += 8.0
    trace.call("hipStreamEndCapture", host, 3.0)
    module_frames(trace, calls)
    return trace


#: The decode step's eager tail: sampling and the bookkeeping of the next step.
DECODE_TAIL = (
    Kernel(DTOD, 2.0),
    Kernel(SMALL_COPY, 2.0),
    Kernel(GREEDY, 6.0, SAMPLER),
    Kernel(INDEX, 3.0),
    Kernel(INDEX_PUT, 3.0),
    Kernel(COPY, 2.0),
    Kernel(ADD_LONG, 2.0),
    Kernel(KV_INDICES, 3.0),
    Kernel(SCAN, 4.0),
    Kernel(ADD_INT, 2.0),
    Kernel(INDEX_PUT, 3.0),
    Kernel(COPY, 2.0),
    Kernel(DTOD, 2.0),
)
#: The positions in ``DECODE_TAIL`` of the six the profiler stamps from inside the replay to the window's end, and the
#: replay slots they are stamped as starting in.
MISREPORTED = {2: 8, 4: 18, 5: 28, 6: 38, 9: 48, 10: 58}
#: The replay slots whose names the window records two slots on: the last two layers of the FlyDSL step.
SHIFTED = (1 + 15 * 2, 1 + 15 * 4)


def _tail(trace: Trace, tail: Sequence[Kernel], host: float, gpu: float, gap: float) -> tuple[list[int], list[float]]:
    """The eager tail's launch calls (the sampler's inside its frame) from ``host`` on, its GPU events from ``gpu`` on;
    their correlation ids and GPU starts."""
    correlations, starts = [], []
    for kernel in tail:
        correlation = trace.correlation()
        trace.launch(kernel, host, 4.0, correlation)
        if kernel.path == SAMPLER:
            trace.frame("Sampler_0", host - 2.0, 8.0)
        correlations.append(correlation)
        starts.append(gpu)
        gpu += kernel.us + gap
        host += 12.0
    return correlations, starts


def decode_window() -> tuple[Trace, Trace]:
    """The FlyDSL decode step as its window records it, and the rank trace: the window, then an eager step where the
    tail's kernels run undisturbed."""
    step = flydsl_decode_step()
    trace = Trace()
    host, gpu = T0, T0 + 400.0
    replay = trace.correlation()
    trace.annotation("step[DECODE bs=64]", host, 300.0)
    trace.call("hipGraphLaunch", host + 10.0, 25.0, replay)
    a, b = SHIFTED
    recorded = [*step[:a], *(replace(k, name=step[a + (i - a + 2) % (b - a)].name) for i, k in enumerate(step[a:b], a))]
    recorded += step[b:]
    starts = gpu_run(trace, recorded, gpu, 1.0, replay)
    end = starts[-1] + step[-1].us
    correlations, tail_starts = _tail(trace, DECODE_TAIL, host + 60.0, end + 12.0, 6.0)
    window_end = tail_starts[-1] + DECODE_TAIL[-1].us
    for i, (kernel, correlation, start) in enumerate(zip(DECODE_TAIL, correlations, tail_starts, strict=True)):
        if i in MISREPORTED:
            start = starts[MISREPORTED[i]] + 0.25
            trace.gpu(kernel, start, window_end - start, correlation)
        else:
            trace.gpu(kernel, start, kernel.us, correlation)
    rank = trace.gpu_only()
    clean_correlations, clean_starts = _tail(Trace(next_correlation=9001), DECODE_TAIL, 0.0, window_end + 2000.0, 6.0)
    for kernel, correlation, start in zip(DECODE_TAIL, clean_correlations, clean_starts, strict=True):
        rank.gpu(kernel, start, kernel.us, correlation)
    return trace, rank


def lost_replay_window() -> Trace:
    """A decode step whose one replay recorded no kernels: the eager ones before the 5 ms it ran, and those after."""
    trace = Trace()
    before = (Kernel(DTOD, 2.0), Kernel(SMALL_COPY, 2.0), Kernel(KV_INDICES, 3.0))
    after = (Kernel(GREEDY, 6.0), Kernel(INDEX, 3.0), Kernel(INDEX_PUT, 3.0), Kernel(ADD_LONG, 2.0))
    after += (Kernel(ADD_INT, 2.0), Kernel(COPY, 2.0), Kernel(DTOD, 2.0))
    at = T0
    for kernel in before:
        trace.gpu(kernel, at, kernel.us, trace.correlation())
        at += kernel.us + 5.0
    at = T0 + 5000.0 - sum(k.us + 5.0 for k in after) + 5.0
    for kernel in after:
        trace.gpu(kernel, at, kernel.us, trace.correlation())
        at += kernel.us + 5.0
    return trace


#: The eager prefill step's GEMMs (the probe's and the root's slice), and the root window's.
PREFILL_GEMMS = Gemms(qkv=640.0, o=480.0, gate_up=2880.0, down=1440.0)
ROOT_GEMMS = Gemms(qkv=400.0, o=300.0, gate_up=1200.0, down=700.0)


def prefill_model(gemms: Gemms, layers: int, attention: float, small: float, logits: float) -> list[Kernel]:
    """The forward of an eager prefill step: ``layers`` CK layers, ``small`` what each short kernel takes."""
    return model_step(
        [ck_layer(n, gemms, Kernel(PREFILL_ATTENTION, attention), small) for n in range(layers)],
        embed=2.0 * small,
        norm=2.0 * small,
        logits=logits,
        to_float=small,
    )


def prefill_step(layers: int) -> list[Kernel]:
    """One eager prefill step of ``layers`` layers and its bookkeeping, as launched: 25.6 ms at four layers."""
    bookkeeping = [Kernel(HTOD, 20.0), Kernel(WRITE_REQ, 20.0), Kernel(KV_INDICES, 20.0)]
    model = prefill_model(PREFILL_GEMMS, layers, attention=520.0, small=20.0, logits=300.0)
    return [*bookkeeping, *model, Kernel(GREEDY, 20.0, SAMPLER)]


def prefill_window(steps: int, layers: int, tokens: int) -> Trace:
    """``steps`` eager prefill steps of ``tokens`` prompt tokens each, every kernel with its launch call and module
    frames."""
    trace = Trace()
    host, gpu = T0, T0 + 50.0
    for _ in range(steps):
        kernels = prefill_step(layers)
        trace.annotation(f"step[EXTEND bs=16 toks={tokens}]", host, 8.0 * len(kernels) + 10.0)
        gpu = eager_step(trace, kernels, host + 5.0, gpu, [1.0] * len(kernels))
        host = gpu + 100.0
        gpu = host + 150.0
    return trace


def root_window() -> Trace:
    """The root's eager prefill step, back to back, its GEMMs 80% of its kernel time; the gate_up GEMM of its last layer
    starts 0.4 us before its predecessor ends."""
    kernels = prefill_model(ROOT_GEMMS, 2, attention=250.0, small=10.0, logits=440.0)
    kernels = [k for k in kernels if k.name != TO_FLOAT]
    early = max(i for i, k in enumerate(kernels) if k.name == CK_GEMM_WIDE and k.us == ROOT_GEMMS.gate_up)
    trace = Trace()
    at = T0
    for i, kernel in enumerate(kernels):
        start = at - 0.4 if i == early else at
        trace.gpu(kernel, start, kernel.us, trace.correlation())
        at = start + kernel.us
    return trace


def _norm_eager(n: int, at: str) -> list[Kernel]:
    path = _norm(n, at)
    return [Kernel(POW, 6.0, path), Kernel(MEAN, 5.0, path), Kernel(ADD, 4.0, path), Kernel(RSQRT, 4.0, path)]


def eager_window() -> Trace:
    """An eager-mode decode step at TP1: the norm as its elementwise ops, hipBLASLt GEMMs, launch gaps of 10 us."""
    kernels = [Kernel(VOCAB_EMBED, 3.0)]
    tiles = {
        "qkv": ("64x16x256", 6.0),
        "o": ("32x16x256", 5.0),
        "gate_up": ("128x64x128", 14.0),
        "down": ("64x16x512", 9.0),
    }
    for n in range(2):
        tensile = {name: Kernel(TENSILE.format(tile=tile), us) for name, (tile, us) in tiles.items()}
        kernels += [*_norm_eager(n, "input_layernorm"), Kernel(MUL, 5.0), Kernel(TO_BF16, 4.0), tensile["qkv"]]
        kernels += [Kernel(ROTARY, 3.0), Kernel(STORE_KV, 2.0), Kernel(PAGED_BF16, 8.0), Kernel(PAGED_REDUCE, 3.0)]
        kernels += [tensile["o"], Kernel(ADD_SELF, 5.0), *_norm_eager(n, "post_attention_layernorm"), Kernel(MUL, 5.0)]
        kernels += [Kernel(TO_BF16, 4.0), tensile["gate_up"], Kernel(ACT, 4.0), tensile["down"], Kernel(ADD_SELF, 5.0)]
    kernels += [*_norm_eager(2, "norm"), Kernel(TO_BF16, 4.0), Kernel(TENSILE.format(tile="256x16x64"), 10.0)]
    kernels += [Kernel(ARGMAX, 5.0), Kernel(DTOD, 2.0)]
    early = next(i for i, k in enumerate(kernels) if k.name == TENSILE.format(tile="128x64x128"))
    return _spaced(kernels, 10.0, early)


def eager_tp2_window() -> Trace:
    """An eager-mode decode step at TP2: fused norms, hipBLASLt GEMMs, the attention's one-shot all-reduce and the MLP's
    two-stage one, launch gaps of 18 us."""
    kernels = [Kernel(VOCAB_EMBED, 3.0)]
    tiles = {
        "qkv": ("96x64x128", 6.0),
        "o": ("16x16x512", 4.0),
        "gate_up": ("160x64x128", 12.0),
        "down": ("64x16x512", 8.0),
    }
    for _ in range(8):
        tensile = {name: Kernel(TENSILE.format(tile=tile), us) for name, (tile, us) in tiles.items()}
        kernels += [Kernel(NORM, 3.0), tensile["qkv"], Kernel(ROTARY, 3.0), Kernel(STORE_KV, 2.0)]
        kernels += [Kernel(PAGED_BF16, 8.0), Kernel(PAGED_REDUCE, 3.0), tensile["o"], Kernel(AR_LL, 6.0)]
        kernels += [
            Kernel(NORM, 3.0),
            tensile["gate_up"],
            Kernel(ACT, 4.0),
            tensile["down"],
            Kernel(CROSS_REDUCE, 25.0),
        ]
    kernels += [Kernel(NORM_LAST, 3.0), Kernel(TENSILE.format(tile="128x128x256"), 10.0), Kernel(ARGMAX, 5.0)]
    kernels += [Kernel(DTOD, 2.0)]
    early = next(i for i, k in enumerate(kernels) if k.name == CROSS_REDUCE)
    return _spaced(kernels, 18.0, early)


def _spaced(kernels: Sequence[Kernel], gap: float, early: int) -> Trace:
    """``kernels`` ``gap`` apart, the ``early``-th starting 0.6 us before its predecessor ends."""
    trace = Trace()
    at = T0
    for i, kernel in enumerate(kernels):
        start = at - 0.6 if i == early else at + (gap if i else 0.0)
        trace.gpu(kernel, start, kernel.us, trace.correlation())
        at = start + kernel.us
    return trace


#: The decode tail of a node-by-node step: sampling and bookkeeping, 32 us in all.
NODEWISE_TAIL = (
    Kernel(DTOD, 2.0),
    Kernel(SMALL_COPY, 2.0),
    Kernel(GREEDY, 6.0, SAMPLER),
    Kernel(ADD_LONG, 2.0),
    Kernel(ADD_INT, 2.0),
    Kernel(KV_INDICES, 3.0),
    Kernel(CLAMP, 2.0),
    Kernel(INDEX_PUT, 3.0),
    Kernel(COPY, 2.0),
    Kernel(SCAN, 4.0),
    Kernel(INDEX, 2.0),
    Kernel(DTOD, 2.0),
)


def nodewise_window(paged_us: float, steps: int = 4) -> Trace:
    """``steps`` decode steps replayed node by node: every replay records its kernels in the capture's order."""
    trace = Trace()
    host, gpu = T0, T0 + 400.0
    kernels = ck_decode_step(paged_us)
    for _ in range(steps):
        replay = trace.correlation()
        trace.annotation("step[DECODE bs=64]", host, 300.0)
        trace.call("hipGraphLaunch", host + 10.0, 25.0, replay)
        starts = gpu_run(trace, kernels, gpu, 0.2, replay)
        end = starts[-1] + kernels[-1].us
        correlations, tail_starts = _tail(trace, NODEWISE_TAIL, host + 60.0, end + 5.0, 2.0)
        for kernel, correlation, start in zip(NODEWISE_TAIL, correlations, tail_starts, strict=True):
            trace.gpu(kernel, start, kernel.us, correlation)
        gpu = tail_starts[-1] + NODEWISE_TAIL[-1].us + 10.0
        host = gpu - 400.0
    return trace


#: Paged attention's time per layer in the two node-by-node profiles: the probe's step is 900 us, 30% of it attention.
NODEWISE_PAGED_US, PROBE_PAGED_US = 40.0, 67.5
DECODE_STEP = {
    "batch_size": 64,
    "num_requests": 64,
    "context_requests": 0,
    "context_sum": 0,
    "generation_requests": 64,
    "generation_sum": 64,
}
PREFILL_STEP = {
    "batch_size": 16384,
    "num_requests": 16,
    "context_requests": 16,
    "context_sum": 16384,
    "generation_requests": 0,
    "generation_sum": 0,
}
PROBE_STAMP = "1700000000.0"


def _manifest(*sizes: int) -> list[dict[str, Any]]:
    return [
        {"file": f"DecodeCudaGraphRunner_bs_{size}_rank0.json.gz", "batch_size": size, "mode": "FULL"} for size in sizes
    ]


def fixtures() -> dict[str, Any]:
    """Every fixture of this directory by file name: a trace's payload, or a JSON document."""
    window, rank = decode_window()
    decode_capture = capture(flydsl_decode_step())
    nodewise_capture = capture(ck_decode_step(NODEWISE_PAGED_US))
    probe_details = [
        {
            "output_path": f"decode_only_steady_state_prefill_0_prefilldecode_0_decode_32_bs64_conc64_{PROBE_STAMP}"
            "-TP-0.trace.json.gz",
            "steps": [DECODE_STEP] * 4,
        },
        {
            "output_path": f"prefilldecode_steady_state_prefill_4_prefilldecode_0_decode_0_bs16384_conc16_{PROBE_STAMP}"
            "-TP-0.trace.json.gz",
            "steps": [PREFILL_STEP] * 2,
        },
    ]
    traces = {
        "decode_capture_bs64_launches": decode_capture,
        "decode_window_launch_events": window,
        "decode_window_gpu_events": window.gpu_only(),
        "decode_rank_trace_gpu_events": rank,
        "lost_replay_window_gpu_events": lost_replay_window(),
        "root_window_gpu_events": root_window(),
        "root_extend_window_launch_events": prefill_window(1, 2, 8192),
        "eager_window_gpu_events": eager_window(),
        "eager_tp2_window_gpu_events": eager_tp2_window(),
        "nodewise_capture_bs64_launches": nodewise_capture,
        "nodewise_decode4_window_launch_events": nodewise_window(NODEWISE_PAGED_US),
        "final_probe_decode4_window_launch_events": nodewise_window(PROBE_PAGED_US),
        "final_probe_prefill2_window_launch_events": prefill_window(2, 4, 16384),
    }
    documents: dict[str, Any] = {f"{name}.json.gz": trace.payload() for name, trace in traces.items()}
    documents["decode_capture_execution_details.json"] = _manifest(56, 64, 72)
    documents["nodewise_capture_execution_details.json"] = _manifest(64)
    documents["final_probe_execution_details.json"] = probe_details
    return documents


def encode(name: str, document: Any) -> bytes:
    """The bytes of fixture ``name``: gzip without a file name or time for a trace, indented JSON otherwise."""
    if name.endswith(".gz"):
        return gzip.compress(json.dumps(document, separators=(",", ":")).encode("utf-8"), compresslevel=9, mtime=0)
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def write_all(directory: Path) -> list[Path]:
    """Write every fixture into ``directory``; the paths written."""
    written = []
    for name, document in fixtures().items():
        path = directory / name
        path.write_bytes(encode(name, document))
        written.append(path)
    return written


def _main(argv: Iterable[str]) -> int:
    args = list(argv)
    directory = Path(args[0]) if args else Path(__file__).resolve().parent
    for path in write_all(directory):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
