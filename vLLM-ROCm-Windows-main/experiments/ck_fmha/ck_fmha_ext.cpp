// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
// torch-callable CK ck_tile FMHA-fwd for gfx1100: ck_fmha_fwd(q,k,v,o, scale, causal) on [B,S,H,D]
// contiguous tensors (GQA via Hk<Hq), fp16. Wraps the compiled causal + non-causal d128 instances.
// Built by build_ck_ext.py. This is the phase-2 integration artifact (validate vs SDPA, then wire to vLLM).
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "fmha_fwd.hpp"

using mask_c  = ck_tile::SimplifiedGenericAttentionMask<true>;   // causal (mask_top_left)
using mask_nc = ck_tile::SimplifiedGenericAttentionMask<false>;  // no mask

using trait_c = fmha_fwd_traits_<128, FmhaFwdFp16, false, 128, 64, 32, 128, 32, 128, true,
                             ck_tile::BlockFmhaPipelineEnum::QRKSVS_HPAD, false, mask_c,
                             ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, false,
                             ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE, true, true, true, true,
                             false, false, false>;
using trait_nc = fmha_fwd_traits_<128, FmhaFwdFp16, false, 128, 64, 32, 128, 32, 128, true,
                             ck_tile::BlockFmhaPipelineEnum::QRKSVS_HPAD, false, mask_nc,
                             ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, false,
                             ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE, true, true, true, true,
                             false, false, false>;
template <>
float fmha_fwd_<trait_c, ck_tile::gfx11_t>(const ck_tile::stream_config&, fmha_fwd_args);
template <>
float fmha_fwd_<trait_nc, ck_tile::gfx11_t>(const ck_tile::stream_config&, fmha_fwd_args);

void ck_fmha_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
                 double scale, bool causal) {
    // q,o: [B,Hq,S,D]; k,v: [B,Hk,S,D]; contiguous; fp16  (CK iperm=1 default layout = bhsd, == SDPA)
    const int B = q.size(0), Hq = q.size(1), S = q.size(2), D = q.size(3);
    const int Hk = k.size(1);
    fmha_fwd_args a{};
    a.q_ptr = q.data_ptr(); a.k_ptr = k.data_ptr(); a.v_ptr = v.data_ptr(); a.o_ptr = o.data_ptr();
    a.seqlen_q = S; a.seqlen_k = S; a.batch = B; a.max_seqlen_q = S;
    a.hdim_q = D; a.hdim_v = D; a.nhead_q = Hq; a.nhead_k = Hk;
    a.scale_s = (float)scale;
    a.stride_q = D; a.stride_k = D; a.stride_v = D; a.stride_o = D;                          // seqlen stride
    a.nhead_stride_q = S * D; a.nhead_stride_k = S * D; a.nhead_stride_v = S * D; a.nhead_stride_o = S * D;
    a.batch_stride_q = (ck_tile::index_t)Hq * S * D; a.batch_stride_k = (ck_tile::index_t)Hk * S * D;
    a.batch_stride_v = (ck_tile::index_t)Hk * S * D; a.batch_stride_o = (ck_tile::index_t)Hq * S * D;
    if (causal) { a.window_size_left = -1; a.window_size_right = 0; a.mask_type = 1; }
    else        { a.window_size_left = -1; a.window_size_right = -1; a.mask_type = 0; }
    ck_tile::stream_config s{(hipStream_t)c10::hip::getCurrentHIPStream(), false};
    if (causal) fmha_fwd_<trait_c, ck_tile::gfx11_t>(s, a);
    else        fmha_fwd_<trait_nc, ck_tile::gfx11_t>(s, a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("ck_fmha_fwd", &ck_fmha_fwd, "CK FMHA fwd (gfx1100)"); }
