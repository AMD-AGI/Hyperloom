// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl forge_qk_norm_rope_h128
.type forge_qk_norm_rope_h128,@function
.p2align 8
forge_qk_norm_rope_h128:
s_load_dwordx8 s[4:11], s[0:1], 0
s_load_dwordx8 s[12:19], s[0:1], 32
s_load_dwordx2 s[20:21], s[0:1], 64
s_waitcnt lgkmcnt(0)
s_mov_b32 s22, s3
s_mov_b32 s23, s19
s_cmp_lt_u32 s3, s19
s_cbranch_scc1 .Lq
s_sub_u32 s22, s3, s19
s_mov_b32 s23, s20
s_mov_b64 s[10:11], s[12:13]
s_mov_b64 s[14:15], s[16:17]
.Lq:
s_mul_i32 s24, s2, s21
s_lshl_b32 s25, s3, 8
s_add_u32 s24, s24, s25
s_add_u32 s4, s4, s24
s_addc_u32 s5, s5, 0
s_mul_i32 s24, s2, s23
s_add_u32 s24, s24, s22
s_lshl_b32 s24, s24, 8
s_add_u32 s14, s14, s24
s_addc_u32 s15, s15, 0
s_lshl_b32 s24, s2, 3
s_add_u32 s6, s6, s24
s_addc_u32 s7, s7, 0
s_load_dword s24, s[6:7], 0
v_lshlrev_b32 v1, 1, v0
global_load_ushort v2, v1, s[4:5]
global_load_ushort v3, v1, s[4:5] offset:128
global_load_ushort v4, v1, s[10:11]
global_load_ushort v5, v1, s[10:11] offset:128
s_waitcnt lgkmcnt(0)
s_lshl_b32 s24, s24, 8
s_add_u32 s8, s8, s24
s_addc_u32 s9, s9, 0
global_load_ushort v6, v1, s[8:9]
global_load_ushort v7, v1, s[8:9] offset:128
s_waitcnt vmcnt(0)
v_lshlrev_b32 v2, 16, v2
v_lshlrev_b32 v3, 16, v3
v_lshlrev_b32 v4, 16, v4
v_lshlrev_b32 v5, 16, v5
v_lshlrev_b32 v6, 16, v6
v_lshlrev_b32 v7, 16, v7
v_mul_f32 v8, v2, v2
v_mul_f32 v9, v3, v3
v_add_f32 v8, v8, v9
s_nop 4
v_add_f32 v8, v8, v8 quad_perm:[1,0,3,2]
s_nop 4
v_add_f32 v8, v8, v8 quad_perm:[2,3,0,1]
s_nop 4
v_add_f32 v8, v8, v8 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:0
s_nop 4
v_add_f32 v8, v8, v8 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:0
s_nop 4
v_add_f32 v8, v8, v8 row_bcast:15 row_mask:0xa bank_mask:0xf
s_nop 4
v_add_f32 v8, v8, v8 row_bcast:31 row_mask:0xc bank_mask:0xf
s_nop 4
v_readlane_b32 s24, v8, 63
s_nop 4
v_mov_b32 v8, s24
v_mul_f32 v8, 0x3c000000, v8
v_add_f32 v8, s18, v8
v_rsq_f32 v8, v8
s_nop 4
v_mul_f32 v2, v2, v8
v_mul_f32 v3, v3, v8
v_cvt_pk_bf16_f32 v9, v2, v3
v_lshlrev_b32 v2, 16, v9
v_and_b32 v3, 0xffff0000, v9
v_mul_f32 v2, v2, v4
v_mul_f32 v3, v3, v5
v_cvt_pk_bf16_f32 v9, v2, v3
v_lshlrev_b32 v2, 16, v9
v_and_b32 v3, 0xffff0000, v9
v_mul_f32 v10, v2, v6
v_mul_f32 v11, v3, v7
v_mul_f32 v12, v3, v6
v_mul_f32 v13, v2, v7
v_cvt_pk_bf16_f32 v9, v10, v11
v_lshlrev_b32 v10, 16, v9
v_and_b32 v11, 0xffff0000, v9
v_cvt_pk_bf16_f32 v9, v12, v13
v_lshlrev_b32 v12, 16, v9
v_and_b32 v13, 0xffff0000, v9
v_sub_f32 v10, v10, v11
v_add_f32 v12, v12, v13
v_cvt_pk_bf16_f32 v9, v10, v12
global_store_short v1, v9, s[14:15]
v_lshrrev_b32 v9, 16, v9
global_store_short v1, v9, s[14:15] offset:128
s_waitcnt vmcnt(0)
s_endpgm
.size forge_qk_norm_rope_h128, .-forge_qk_norm_rope_h128
.section .rodata
.p2align 6
.amdhsa_kernel forge_qk_norm_rope_h128
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_next_free_vgpr 14
.amdhsa_next_free_sgpr 26
.amdhsa_accum_offset 16
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_system_sgpr_workgroup_id_x 1
.amdhsa_system_sgpr_workgroup_id_y 1
.amdhsa_system_vgpr_workitem_id 0
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: forge_qk_norm_rope_h128
    .symbol: forge_qk_norm_rope_h128.kd
    .kernarg_segment_size: 72
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 26
    .vgpr_count: 14
    .max_flat_workgroup_size: 64
    .args:
      - { .offset: 0, .size: 8, .value_kind: global_buffer, .name: qkv }
      - { .offset: 8, .size: 8, .value_kind: global_buffer, .name: positions }
      - { .offset: 16, .size: 8, .value_kind: global_buffer, .name: cache }
      - { .offset: 24, .size: 8, .value_kind: global_buffer, .name: qw }
      - { .offset: 32, .size: 8, .value_kind: global_buffer, .name: kw }
      - { .offset: 40, .size: 8, .value_kind: global_buffer, .name: qout }
      - { .offset: 48, .size: 8, .value_kind: global_buffer, .name: kout }
      - { .offset: 56, .size: 4, .value_kind: by_value, .name: epsilon }
      - { .offset: 60, .size: 4, .value_kind: by_value, .name: qheads }
      - { .offset: 64, .size: 4, .value_kind: by_value, .name: kheads }
      - { .offset: 68, .size: 4, .value_kind: by_value, .name: stride }
...
.end_amdgpu_metadata
