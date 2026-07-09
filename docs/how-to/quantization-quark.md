# Quantization with AMD Quark

The optional quantization prelude drives [AMD Quark](https://quark.docs.amd.com/)
before the optimization loop starts, then rewrites `--model` to the exported
quantized model so the rest of Hyperloom optimizes that artifact.

Quantization is disabled unless you explicitly enable the deterministic master
switch:

```bash
export HYPERLOOM_QUANTIZE_ENABLED=1
```

Then choose one request style:

```bash
# Free-text power-user path. Failure hard-stops the run with exit code 3.
inference_optimizer optimize \
  --model /models/source \
  --quantize "fp8 global scheme, fp8 kv_cache, exclude lm_head"

# Structured path. `none` or omit means no quantization.
inference_optimizer optimize \
  --model /models/source \
  --quantize-scheme fp8
```

Structured schemes are `fp8`, `ptpc_fp8`, `mxfp4`, and `mxfp4_fp8`.
`mxfp4` / `mxfp4_fp8` are MI355X-only; when a structured scheme is unsupported
on the selected GPU, Hyperloom prints `QUANTIZATION_SKIPPED`, sets
`HYPERLOOM_QUANTIZATION_SKIPPED`, and continues on the unquantized model.

## Quark Checkout

`quantization_agent` requires an AMD Quark checkout at runtime. Hyperloom does
not bundle Quark or implement quantization itself; it invokes Quark's published
skills (`quark-torch-ptq` -> `quark-torch-result-validator` ->
`quark-torch-llm-eval`) end to end.

Quark is published on PyPI (`pip install amd-quark`), but the current external
release does not ship the `.claude/skills/quark-torch-*` skill entry points that
Hyperloom drives. Until those skills are public, use an internal AMD Quark
repository checkout.

Hyperloom resolves the Quark root in this order:

1. `--quark-root`
2. `QUARK_ROOT`
3. The Core42-only default `/wekafs/hyperloom/Quark`

Outside Core42, set `QUARK_ROOT` explicitly. The path must contain
`.claude/skills/quark-torch-ptq/SKILL.md` plus the validator and eval skills. If
the resolved checkout is missing after quantization is enabled, the run fails
fast instead of silently optimizing the unquantized source model.

```text
HYPERLOOM_QUANTIZE_ENABLED=1
QUARK_ROOT=/workspace/Quark
```
