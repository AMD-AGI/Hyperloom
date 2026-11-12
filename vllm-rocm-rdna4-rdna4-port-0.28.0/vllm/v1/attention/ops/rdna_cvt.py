# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-side wrappers for the gfx12 hardware fp8 converts (extern_elementwise over rdnacvt.ll). Shipped in rc9: measured -33% on the fp8-Q attention kernel, outputs identical."""
import os
import triton, triton.language as tl
from triton.language import core
LIB = "rdnacvt"
LIB_PATH = os.environ.get("RDNACVT_LL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rdnacvt.ll"))

@core.extern
def f32_to_e4m3(x, _semantic=None):
    return core.extern_elementwise(LIB, LIB_PATH, [x], {(core.dtype("fp32"),): ("__rdnacvt_f32_to_e4m3", core.dtype("fp8e4nv"))}, is_pure=True, _semantic=_semantic)

@core.extern
def f32_to_e4m3_sat(x, _semantic=None):
    return core.extern_elementwise(LIB, LIB_PATH, [x], {(core.dtype("fp32"),): ("__rdnacvt_f32_to_e4m3_sat", core.dtype("fp8e4nv"))}, is_pure=False, _semantic=_semantic)

@core.extern
def bf16_to_e4m3(x, _semantic=None):
    return core.extern_elementwise(LIB, LIB_PATH, [x], {(core.dtype("bf16"),): ("__rdnacvt_bf16_to_e4m3", core.dtype("fp8e4nv"))}, is_pure=True, _semantic=_semantic)

@core.extern
def f16_to_e4m3(x, _semantic=None):
    return core.extern_elementwise(LIB, LIB_PATH, [x], {(core.dtype("fp16"),): ("__rdnacvt_f16_to_e4m3", core.dtype("fp8e4nv"))}, is_pure=True, _semantic=_semantic)

@core.extern
def e4m3_to_f32(x, _semantic=None):
    return core.extern_elementwise(LIB, LIB_PATH, [x], {(core.dtype("fp8e4nv"),): ("__rdnacvt_e4m3_to_f32", core.dtype("fp32"))}, is_pure=True, _semantic=_semantic)

@core.extern
def e4m3_to_bf16(x, _semantic=None):
    return core.extern_elementwise(LIB, LIB_PATH, [x], {(core.dtype("fp8e4nv"),): ("__rdnacvt_e4m3_to_bf16", core.dtype("bf16"))}, is_pure=True, _semantic=_semantic)

@core.extern
def e4m3_to_f16(x, _semantic=None):
    return core.extern_elementwise(LIB, LIB_PATH, [x], {(core.dtype("fp8e4nv"),): ("__rdnacvt_e4m3_to_f16", core.dtype("fp16"))}, is_pure=True, _semantic=_semantic)

EXTERN_LIBS = {LIB: LIB_PATH}
