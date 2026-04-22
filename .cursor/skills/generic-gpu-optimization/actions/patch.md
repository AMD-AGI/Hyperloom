# Action: Apply / Revert a Generic Code Patch

## When to use
For any code change that isn't a compile flag, env var, or kernel rewrite. Most
common case: profile suggests a small structural fix (e.g. cache a value, change
launch-config heuristic, replace a `for`-loop with a fused pass).

## Procedure

### Step 1: Save baseline of every file you'll touch
```bash
for f in "${FILES_TO_PATCH[@]}"; do
    cp "$f" "$f.${ATTEMPT_ID}.bak"
done
```

### Step 2: Apply the change
Make the edit. Keep the change SCOPED — one logical change per attempt. If
you're tempted to "also fix" a nearby issue, push it as a separate action.

### Step 3: Save the patch
```bash
git -C "$REPO_ROOT" diff -- "${FILES_TO_PATCH[@]}" > \
    "$RESULT_DIR/patches/$(printf '%02d' $ATTEMPT_ID)-${PATCH_NAME}.patch"
```

### Step 4: Trigger build → correctness → baseline (standard pipeline)

### Step 5: Keep or revert
If KEEP: leave the change in place. The patch file in `$RESULT_DIR/patches/` is
the durable artifact — it can be re-applied to a clean checkout via:
```bash
git -C $REPO_ROOT apply $RESULT_DIR/patches/*.patch
```

If REVERT:
```bash
for f in "${FILES_TO_PATCH[@]}"; do
    mv "$f.${ATTEMPT_ID}.bak" "$f"
done
git -C "$REPO_ROOT" checkout -- "${FILES_TO_PATCH[@]}"
```

## Outputs
- `$RESULT_DIR/patches/NN-<name>.patch` (always saved, even on revert, for the report)
