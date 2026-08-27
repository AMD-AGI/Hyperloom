# Serving patches

This directory holds versioned source patches that KernelForge owns as
serving-side optimization enablers. Each patch encodes kernel/backend knowledge
that a serving engine cannot derive on its own; a separate Hyperloom applier
discovers and applies the right patch for the installed serving-engine version.

KernelForge is the owner of these assets. Hyperloom is only the consumer.

## What the current patch does

`fp8_blockscale_ck_routing.patch` adds **M-aware CK routing** for fp8
block-scale GEMM on AMD MI300X (gfx942).

Upstream sglang hardcodes the block-FP8 GEMM to the Triton path on this target.
The patch instead routes the GEMM by the M dimension:

- small (decode) M  -> CK `gemm_a8w8_blockscale` (multiple-x faster)
- large (prefill) M -> keep Triton (avoids a large-M regression)

CK and Triton produce numerically identical output, so this is a pure
performance routing change, not a correctness change.

### Env gate

The routing is controlled by `SGLANG_FP8_BLOCKSCALE_CK_MAX_M`:

- `0` (default) = **OFF**, zero behavior change vs. upstream.
- `> 0`         = route GEMMs with `M <= SGLANG_FP8_BLOCKSCALE_CK_MAX_M` to CK.

Because the default is OFF, applying the patch is safe and fully
backward-compatible: behavior only changes when the env var is explicitly set.

## Directory layout convention

```
serving_patches/
  sglang/
    SUPPORTED_VERSIONS.txt              # manifest: one supported version per line
    sglang_<major>_<minor>_<patch>/     # e.g. sglang_0_5_12 for sglang 0.5.12
      fp8_blockscale_ck_routing.patch
```

The version subdirectory name is the engine version with dots replaced by
underscores and an `sglang_` prefix (matching Hyperloom's
`_versioned_patches_subdir_name("0.5.12") -> "sglang_0_5_12"` convention).

`SUPPORTED_VERSIONS.txt` lists the versions for which a verified patch exists.
It supports `#` comments and blank lines.

## Applying

The patches are standard `git format` diffs with `a/python/... b/python/...`
paths.

- Editable sglang source tree: `git apply -p1 <patch>`
- Wheel / site-packages install: `git apply -p3 <patch>`

The Hyperloom applier selects the correct strip level automatically. Do not
apply patches here by hand as part of KernelForge workflows.

## TODO

- Upstream the M-aware routing to sglang so this patch can eventually be
  retired.
