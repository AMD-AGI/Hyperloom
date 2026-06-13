# Quantization Prelude

Read this only when the user explicitly asks to quantize before optimizing.

Pass `--quantize "<scheme prompt>"`. This runs the quantization-agent once as an
AMD Quark PTQ prelude, then rewrites `--model` to the exported quantized model
so the optimization loop runs on it.

```bash
inference_optimizer optimize \
  --model "$MODEL_PATH" \
  --framework vllm \
  --max-hours 2 \
  --quantize "fp8 global scheme, fp8 kv_cache, exclude lm_head; accept up to 5% relative eval gap"
```

Rules:

- The `--quantize` text is the request only: scheme, kv-cache, excluded layers,
  acceptable eval gap. Do not repeat the model path or export dir.
- The adapter folds `--model` and
  `<workspace_root>/quantization/<model>/quantized` into the prompt
  automatically.
- `--quantize-scheme` is the structured alternative: `none`, `fp8`,
  `ptpc_fp8`, `mxfp4`, `mxfp4_fp8`. Free-text `--quantize` wins when both are
  given.
- `mxfp4*` schemes are MI355X-only.
- Keep `--precision` consistent with quantization, for example
  `--quantize-scheme fp8` with `--precision fp8`.
- Quantization is one-shot and skipped on `--resume`.
- Failed or unusable quantization hard-stops with `SystemExit(3)`; it never
  silently optimizes the source model.
- A scheme/GPU mismatch via `--quantize-scheme` is skipped, emits
  `QUANTIZATION_SKIPPED:`, and sets `$HYPERLOOM_QUANTIZATION_SKIPPED`.
- `$QUARK_ROOT` must point at a Quark checkout with
  `.claude/skills/quark-torch-*`, and installed `amd-quark` must match it.

Report the `Quantization prelude: model -> <dir>` stdout line; that is the model
path used for the run.
