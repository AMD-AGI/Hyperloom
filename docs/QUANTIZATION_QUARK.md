# Quantization (optional): AMD Quark dependency

The optional quantization prelude (`inference_optimizer optimize --quantize ...`, backed by the `quantization_agent` sub-agent) drives [AMD Quark](https://quark.docs.amd.com/) to produce a quantized model before the optimization loop runs. You only need Quark if you use this path; the rest of Hyperloom works without it.

- **Dependency.** `quantization_agent` requires an AMD Quark checkout **at runtime**. It does not bundle Quark or implement quantization itself — it invokes Quark's published skills (`quark-torch-ptq` → `quark-torch-result-validator` → `quark-torch-llm-eval`) end-to-end.
- **Obtaining Quark.** Quark is published on PyPI (`pip install amd-quark`). However, the current external release does **not** ship the `.claude/skills/quark-torch-*` skill-invocation entry points that the agent drives, so **today you must use the internal Quark repository** checkout. Switch to the public package once it bundles those skills.
- **Pointing at a local checkout.** The agent resolves the Quark root in this order:
  1. the explicit `quark_root=` argument (Python API / `--quark-root` CLI flag),
  2. the `QUARK_ROOT` environment variable,
  3. the built-in default (Core42-only): `/wekafs/hyperloom/Quark`.

  Outside Core42 you must set `QUARK_ROOT` (or pass `--quark-root`). The resolved path must contain `.claude/skills/quark-torch-ptq/SKILL.md` (plus the validator / eval skills under the same tree). If none of the above resolves to an existing directory, the run fails fast with `quark_root_missing` rather than silently optimizing the un-quantized model. Set it in your `.env` when your checkout lives elsewhere:

  ```env
  # Only needed for the --quantize prelude; path to your internal amd-quark checkout.
  QUARK_ROOT=/workspace/Quark
  ```
