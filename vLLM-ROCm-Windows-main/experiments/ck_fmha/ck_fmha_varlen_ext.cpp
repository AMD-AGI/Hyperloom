// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
// torch-callable CK ck_tile FMHA-fwd VARLEN (group mode) for gfx1100 -- the flash_attn_varlen ABI vLLM
// prefill needs. ck_fmha_varlen(q,k,v,o, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale,
// causal) on thd-packed tensors: q,o=[Tq,Hq,D]; k,v=[Tk,Hk,D]; cu_seqlens=int32[B+1] on device. fp16, d128.
// GQA via Hk<Hq. Wraps the CK group-mode nlogits d128-fp16 causal + non-causal instances. Built by
// build_ck_varlen.py. Phase-2 vLLM-prefill integration artifact.
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "fmha_fwd.hpp"

using mask_c  = ck_tile::SimplifiedGenericAttentionMask<true>;   // causal (mask_top_left)
using mask_nc = ck_tile::SimplifiedGenericAttentionMask<false>;  // no mask

// group mode = kIsGroupMode(param 3) = true; nlogits = kHasLogitsSoftCap(param 12) = false
template <typename DT, typename MASK>
using vtrait = fmha_fwd_traits_<128, DT, true, 128, 64, 32, 128, 32, 128, true,
                             ck_tile::BlockFmhaPipelineEnum::QRKSVS_HPAD, false, MASK,
                             ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, false,
                             ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE, true, true, true, true,
                             false, false, false>;
using trait_c     = vtrait<FmhaFwdFp16, mask_c>;
using trait_nc    = vtrait<FmhaFwdFp16, mask_nc>;
using trait_c_bf  = vtrait<FmhaFwdBf16, mask_c>;
using trait_nc_bf = vtrait<FmhaFwdBf16, mask_nc>;
template <> float fmha_fwd_<trait_c,     ck_tile::gfx11_t>(const ck_tile::stream_config&, fmha_fwd_args);
template <> float fmha_fwd_<trait_nc,    ck_tile::gfx11_t>(const ck_tile::stream_config&, fmha_fwd_args);
template <> float fmha_fwd_<trait_c_bf,  ck_tile::gfx11_t>(const ck_tile::stream_config&, fmha_fwd_args);
template <> float fmha_fwd_<trait_nc_bf, ck_tile::gfx11_t>(const ck_tile::stream_config&, fmha_fwd_args);

void ck_fmha_varlen(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
                    torch::Tensor cu_seqlens_q, torch::Tensor cu_seqlens_k,
                    int64_t max_seqlen_q, int64_t max_seqlen_k, double scale, bool causal) {
    // thd-packed varlen (group mode). i_perm=false layout: token stride = nhead*hdim, head stride = hdim.
    const int Hq = q.size(1), D = q.size(2), Hk = k.size(1);
    const int B = cu_seqlens_q.size(0) - 1;
    fmha_fwd_args a{};
    a.q_ptr = q.data_ptr(); a.k_ptr = k.data_ptr(); a.v_ptr = v.data_ptr(); a.o_ptr = o.data_ptr();
    a.seqstart_q_ptr = cu_seqlens_q.data_ptr();     // int32[B+1] device
    a.seqstart_k_ptr = cu_seqlens_k.data_ptr();
    a.seqlen_k_ptr = nullptr;
    a.seqlen_q = (int)max_seqlen_q;                 // unused in group mode
    a.seqlen_k = (int)max_seqlen_k;                 // unused in group mode
    a.batch = B; a.max_seqlen_q = (int)max_seqlen_q;
    a.hdim_q = D; a.hdim_v = D; a.nhead_q = Hq; a.nhead_k = Hk;
    a.scale_s = (float)scale;
    a.stride_q = Hq * D; a.stride_k = Hk * D; a.stride_v = Hk * D; a.stride_o = Hq * D;  // token stride
    a.nhead_stride_q = D; a.nhead_stride_k = D; a.nhead_stride_v = D; a.nhead_stride_o = D;
    a.batch_stride_q = 0; a.batch_stride_k = 0; a.batch_stride_v = 0; a.batch_stride_o = 0;  // group: unused
    a.min_seqlen_q = 0;
    if (causal) { a.window_size_left = -1; a.window_size_right = 0; a.mask_type = 1; }
    else        { a.window_size_left = -1; a.window_size_right = -1; a.mask_type = 0; }
    ck_tile::stream_config s{(hipStream_t)c10::hip::getCurrentHIPStream(), false};
    const bool bf16 = (q.scalar_type() == at::kBFloat16);
    if (bf16) { if (causal) fmha_fwd_<trait_c_bf,  ck_tile::gfx11_t>(s, a);
                else        fmha_fwd_<trait_nc_bf, ck_tile::gfx11_t>(s, a); }
    else      { if (causal) fmha_fwd_<trait_c,     ck_tile::gfx11_t>(s, a);
                else        fmha_fwd_<trait_nc,    ck_tile::gfx11_t>(s, a); }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("ck_fmha_varlen", &ck_fmha_varlen, "CK FMHA varlen fwd (gfx1100)"); }
