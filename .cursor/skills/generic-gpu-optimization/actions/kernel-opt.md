# Action: Kernel Optimization via GEAK MCP

## Eligibility (mirrors training-optimization, framework-agnostic)
A kernel from `$RESULT_DIR/kernel_candidates.json` is eligible if:
- `gpu_pct >= 3.0`
- `source_path` is inside `$REPO_ROOT` (rewriting vendor sources is forbidden)
- `kernel_lang` ∈ `{hip, triton}`

## Procedure

### Step 1: Pick the highest-scoring candidate
```python
import json
candidates = json.load(open(f"{RESULT_DIR}/kernel_candidates.json"))["candidates"]
candidates = [c for c in candidates if c["gpu_pct"] >= 3.0
                                    and c["source_path"]
                                    and c["kernel_lang"] in ("hip", "triton")]
candidates.sort(key=lambda c: c["gpu_pct"], reverse=True)
target = candidates[0]
```

### Step 2: Extract a self-contained kernel file
The submission to GEAK should be a single file containing:
1. The kernel source (function + any device-side helpers it transitively calls)
2. A comment block at the top with:
   - Kernel name + current `gpu_pct`
   - Sample input shapes/dtypes from the profile
   - Hardware: `${GPU} (${GPU_ARCH})`
   - Project context (one paragraph)

For HIP, gather transitive `__device__` helpers via `rg`:
```bash
EXTRACT_OUT="$RESULT_DIR/geak-input-${ATTEMPT_ID}.${EXT}"
python3 "$SKILL_ROOT/scripts/extract_kernel.py" \
    --source "$target.source_path" \
    --kernel "$target.name" \
    --shapes "$target.sample_shapes" \
    --gpu "$GPU" \
    --out "$EXTRACT_OUT"
```

### Step 3: Submit to GEAK MCP
Use the `geak-agent` MCP server (configured in `.cursor/mcp.json`):

```
1. geak_set_model_config(model="claude-opus-4-6", mode="kernel-rewrite")
2. geak_create_task(
       input_type="file",
       input_path=EXTRACT_OUT,
       hardware=GPU_ARCH,
       prompt="Optimize this kernel for AMD ${GPU}. Preserve numerical
               correctness. Target a 1.5-3x speedup. Return a drop-in
               replacement."
   )
3. geak_submit_task(task_id=...)
4. POLL geak_get_task(task_id=...) every 30s, max 30 min
5. geak_get_outputs(task_id=...) → download_url
6. geak_download_file(url=..., local=$RESULT_DIR/geak-output-${ATTEMPT_ID}.${EXT})
```

(See `.cursor/skills/training-optimization/GEAK-KERNEL-OPTIMIZATION.md` for the
full reference of the MCP tool sequence — it is identical here.)

### Step 4: Local validation
Before patching the repo:

```bash
# 1. Compile check (HIP)
hipcc -c "$RESULT_DIR/geak-output-${ATTEMPT_ID}.hip" \
      --offload-arch=$GPU_ARCH -o /tmp/geak_check.o
[ $? -eq 0 ] || { echo "GEAK output won't compile — discard"; exit 1; }

# 2. Microbench (optional but recommended)
python3 "$SKILL_ROOT/scripts/kernel_microbench.py" \
    --orig "$target.source_path" \
    --new  "$RESULT_DIR/geak-output-${ATTEMPT_ID}.${EXT}" \
    --shapes "$target.sample_shapes"
```

### Step 5: Apply as a patch
```bash
cp "$target.source_path" "$target.source_path.geak.bak"
# Splice the new kernel body into the original file
python3 "$SKILL_ROOT/scripts/splice_kernel.py" \
    --target "$target.source_path" \
    --kernel "$target.name" \
    --replacement "$RESULT_DIR/geak-output-${ATTEMPT_ID}.${EXT}"

# Save patch
git -C "$REPO_ROOT" diff -- "$target.source_path" > \
    "$RESULT_DIR/patches/$(printf '%02d' $ATTEMPT_ID)-${target.name}.patch"
```

### Step 6: Trigger build.md → correctness.md → baseline.md
Same protocol as any other action. KEEP only if:
- Build succeeds
- Correctness PASS
- Metric improves > bench_noise_pct

### Step 7: Revert protocol
```bash
mv "$target.source_path.geak.bak" "$target.source_path"
git -C "$REPO_ROOT" checkout -- "$target.source_path"
```

## Outputs
- `$RESULT_DIR/geak-input-N.{hip,py}`, `geak-output-N.{hip,py}`
- `$RESULT_DIR/patches/NN-<kernel>.patch` (if kept)
- Entry in `results.tsv`

## Notes
- Submit ONE kernel per task. Multi-kernel submissions degrade output quality.
- For Inductor-generated Triton (torch.compile path), see
  `.cursor/skills/inference-optimization/kernel-opt/geak.md` for the cache-patching
  trick — it applies verbatim to any pytorch-script project.
- Never blindly accept GEAK output that fails the microbench, even if
  end-to-end metrics improve — it usually means the workload doesn't actually
  exercise the kernel.
