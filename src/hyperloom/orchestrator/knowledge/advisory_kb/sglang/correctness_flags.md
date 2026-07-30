# SGLang — Correctness-critical flags (SGLang runs only)

The plugin's correctness-critical set is thin for SGLang (most such flags are ATOM/vLLM). Shared
ROCm/hardware correctness facts live in `generic/` (hardware.md, correctness_flags.md). Add
SGLang-specific correctness flags here as they are discovered from runs or the model file
(`python/sglang/srt/models/<model_family>.py`).

## Verify EP == TP where required for SGLang
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#parallelism (doc, not run-validated)
- impact: correctness
- domain_tags: moe

SGLang's expert-parallel path can require EP == TP (stricter than the generic EP ≤ TP). Confirm the
constraint for the specific SGLang build before sweeping `--ep-size`.
