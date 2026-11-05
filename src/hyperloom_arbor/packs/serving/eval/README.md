# Component evaluators

Empty on purpose: writing the first evaluators for a model is early work (pack.yaml §7). Each
`eval_<component>.py`:

- measures one component on the model's real shapes (the dossier's part 1), at the row counts the
  operating point produces in prefill and in decode;
- checks its own output against the path it replaces before its timing counts;
- graph-captures anything under about 20 µs;
- takes `--json <path>` and writes one record: component, shapes, variants, per-variant timings
  (every sample, gated on the worst), correctness, device.

Adding an evaluator is always in scope; none is pinned.
