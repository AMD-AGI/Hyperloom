; RDNA4 (gfx12) hardware fp8 conversions as an extern_elementwise library for Triton (no compiler rebuild).
; Linked at kernel compile time via extern_libs={"rdnacvt": "<path>/rdnacvt.ll"}; Triton's AMD backend links any lib
; whose name is contained in an undefined function name (need_extern_lib), so every symbol here carries "rdnacvt".
; The MODE.FP16_OVFL/FP8 clamp bit (23) is set by Triton whenever a kernel has a f32->fp8 cast (SetFP8ClampingAttr);
; kernels that ONLY use these externs will not have it set -> overflow gives NaN (0x7f) instead of +-448.
; __rdnacvt_f32_to_e4m3_sat additionally sets the bit itself (s_setreg) so it saturates like Triton's software path.
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.cvt.pk.fp8.f32(float, float, i32, i1)
declare float @llvm.amdgcn.cvt.f32.fp8(i32, i32)
declare <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32, i1)
declare void @llvm.amdgcn.s.setreg(i32, i32)

define i8 @__rdnacvt_f32_to_e4m3(float %x) #0 {
  %p = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %x, float %x, i32 poison, i1 false)
  %b = trunc i32 %p to i8
  ret i8 %b
}

define i8 @__rdnacvt_f32_to_e4m3_sat(float %x) #0 {
  ; hwreg(HW_REG_MODE, 23, 1) = 1473 ; value 1 -> saturate finite overflow to +-448 (inf still -> NaN)
  call void @llvm.amdgcn.s.setreg(i32 1473, i32 1)
  %p = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %x, float %x, i32 poison, i1 false)
  %b = trunc i32 %p to i8
  ret i8 %b
}

define i8 @__rdnacvt_bf16_to_e4m3(bfloat %x) #0 {
  %f = fpext bfloat %x to float
  %p = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %f, float %f, i32 poison, i1 false)
  %b = trunc i32 %p to i8
  ret i8 %b
}

define i8 @__rdnacvt_f16_to_e4m3(half %x) #0 {
  %f = fpext half %x to float
  %p = call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %f, float %f, i32 poison, i1 false)
  %b = trunc i32 %p to i8
  ret i8 %b
}

define float @__rdnacvt_e4m3_to_f32(i8 %x) #0 {
  %z = zext i8 %x to i32
  %f = call float @llvm.amdgcn.cvt.f32.fp8(i32 %z, i32 0)
  ret float %f
}

define bfloat @__rdnacvt_e4m3_to_bf16(i8 %x) #0 {
  %z = zext i8 %x to i32
  %f = call float @llvm.amdgcn.cvt.f32.fp8(i32 %z, i32 0)
  %b = fptrunc float %f to bfloat
  ret bfloat %b
}

define half @__rdnacvt_e4m3_to_f16(i8 %x) #0 {
  %z = zext i8 %x to i32
  %f = call float @llvm.amdgcn.cvt.f32.fp8(i32 %z, i32 0)
  %h = fptrunc float %f to half
  ret half %h
}

attributes #0 = { alwaysinline nounwind readnone willreturn "no-builtins" }
