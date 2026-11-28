---
myst:
    html_meta:
        "description": "Quantization prelude variables for Hyperloom: primary switch for the AMD Quark PTQ quantization prelude and the Quark checkout path used by the quantization agent."
        "keywords": "Hyperloom, environment variables, quantization, HYPERLOOM_QUANTIZE_ENABLED, QUARK_ROOT, AMD Quark, PTQ, --quantize, --quantize-scheme"
---
# Hyperloom Quantization prelude

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_QUANTIZE_ENABLED` | Unset | Primary switch (`1` to enable) for the AMD Quark PTQ quantization prelude driven by `--quantize` / `--quantize-scheme`. |
| `QUARK_ROOT` | Unset | AMD Quark checkout used by the quantization agent. Set this explicitly when quantization is enabled. |
