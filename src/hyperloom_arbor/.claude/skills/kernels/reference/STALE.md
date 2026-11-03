# Lines in this copy that disagree with the hardware skill

Checked 2026-10-20 by a grep of every copied file for `304`, `32 banks`, `32-bank` and `5.3 TB/s`,
then for `96 KB`, `TF32`, FNUZ and `f8f6f4` near gfx942. Where a line here and the hardware skill
(`.claude/skills/hardware/SKILL.md`) disagree, the hardware skill is right. KernelForge's files stay
as copied; apply the correction when you read them.

## Wrong for gfx950 (MI350X/MI355X)

| File:line | The claim | The right value |
| --- | --- | --- |
| `local_knowledge/framework/aiter/operators/fused_moe_grouped_gemm/tuning.md:25` | align-and-sort `BLOCK_M` decides "how tightly per-expert tiles pack the 304 CUs"; the page covers gfx942, gfx950 and gfx1250 | 304 CUs is MI300X; MI350X/MI355X have 256 (8 XCDs × 32), and AITER's `cu_num` is 256 there |
| same file, `:65` | `ksplit` "to reach ≥1024 tiles across 304 CUs" | 256 CUs on gfx950 |
| same file, `:68` | "per-expert tiles total should give ≈ k·304 blocks" | ≈ k·256 on gfx950 |
| `knowledge_base/flydsl/gemm_optimization.md:326` | worked FP8 GEMM example, no arch named: "158 GB/s (well below HBM peak 5.3 TB/s)" | 5.3 TB/s is MI300X's spec; gfx950: 8.0 TB/s spec, 6.0–6.1 TB/s measured cold on MI350X (hardware skill, measured table) |
| `knowledge_base/flydsl/gemm_optimization.md:322` | same example: "LDS = … 16 KB (well under 64 KB)" | 64 KiB is gfx942's LDS; gfx950 has 160 KiB per CU |

No copied page says gfx950 has 32 LDS banks. The page that did, `knowledge_base/shared/gpu_arch_gfx950.md`
(304 CUs, 32 banks, "similar to MI300X's 5.3 TB/s"), was not copied.

## Other lines that disagree with the hardware skill

| File:line | The claim | What the hardware skill says |
| --- | --- | --- |
| `knowledge_base/ck/ck_playbook.md:3096`, `:3139` | "96 KB LDS per CU (gfx940)"; "96 KB LDS retained — same as gfx90a". The same file says 64 KB at `:347` | MI300X (gfx942): 64 KiB LDS per CU, 32 banks (`hardware/reference/cdna3_mi300/memory_hierarchy.md:30`) |
| `knowledge_base/ck/ck_playbook.md:3147` | "gfx950 inherits gfx942's scaled MFMA — same `_f8f6f4` family" | gfx942 has no FP4, FP6 or block-scaled MFMA; the scaled `f8f6f4` MFMA is new in gfx950 (`hardware/reference/cdna3_mi300/matrix_core.md:41`) |
| `knowledge_base/aiter/backend_dispatch.md:158` | gfx942 row: FP8 "OCP + FNUZ", MXFP4 "Yes" | gfx942's FP8 is FNUZ in hardware, and it has no FP4 matrix instructions; AITER's MXFP4 W4A4 is rejected on gfx942, where `--moe-backend auto` picks W4A16 (`serving-playbook/image-manifest.md` A4, #1011) |
| `knowledge_base/ck/ck_playbook.md:2031` | CK's CMake gate "TF32 = gfx942+" | reads as including gfx950, which has no TF32 (hardware SKILL.md, quick facts); a build-gate statement, so check CK's `CMakeLists.txt` before relying on it |

## Hits that are correct (checked, no action)

- `knowledge_base/ck/ck_playbook.md:1693`: "32 banks on gfx942 / 64 banks on gfx950".
- `knowledge_base/ck/ck_playbook.md:1707`, `:1813`, `:1905`: "32-bank" layouts of gfx9-era and gfx942 code.
- `knowledge_base/ck/ck_playbook.md:3135`, `:3203`: "a flat 304-CU grid", about MI300X.
- `knowledge_base/ck/ck_playbook.md:3156`, `:3209`: CK's swizzles assume 32 banks while gfx950 has 64 (a finding, not an error).
- `knowledge_base/flydsl/gemm_optimization.md:97`, `knowledge_base/flydsl/lds_optimization.md:121`: "gfx942 (32 banks)", with gfx950's 64 on the next line.
- `local_knowledge/framework/aiter/overall/tuning_db.md:41`: `cu_num` 256 → gfx950, 80/304 → gfx942.
- `knowledge_base/flydsl/dsl_api.md:79`: "304" is a source line number.
