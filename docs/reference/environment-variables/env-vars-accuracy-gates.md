---
myst:
    html_meta:
        "description": "Accuracy gate variables for Hyperloom: eval generation bounds, per-generation token ceiling, truncation probe settings, and the fixed 5-point tolerance for candidate KEEP decisions."
        "keywords": "Hyperloom, environment variables, accuracy gates, eval, HYPERLOOM_EVAL_LIMIT, HYPERLOOM_EVAL_MAX_TOKENS, HYPERLOOM_EVAL_PROBE, lm-eval"
---
# Hyperloom Accuracy gates

A candidate that clears the throughput bar must also hold accuracy before it is
kept. Grading runs only *after* the throughput bar is cleared, and reads the
score back from the run's own eval output, so a gate never costs an extra eval
and a regressing candidate never spends a verdict on itself.

In every lane a measured drop beyond the tolerance is a `REVERT`. A missing
verdict while a positive baseline accuracy is on record drops to
`NEEDS_REVIEW` — eval should have worked and didn't. No baseline accuracy at
all degrades to a throughput-only `KEEP` rather than blocking every candidate,
so eval-less environments still make progress. The accuracy eval always runs:
a baseline whose eval fails stops the run instead of falling back to a
throughput-only measurement.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_QUALITY_REF`<br>`HYPERLOOM_QUALITY_REF_WRITE` | Derived under the session dir | The scriptable quality gate's reference artifact: `_WRITE` establishes it on the baseline, the other compares against it on every later candidate. What the artifact holds is the workload's own business — xDiT stores an image, an operator-supplied `custom` workload stores whatever its script compares. Also emitted as `XDIT_QUALITY_REF` / `XDIT_QUALITY_REF_WRITE` for bench scripts written before the rename; either name is read, both are written. |
| `INFERENCE_OPTIMIZER`<br>`_REQUIRE_KERNEL`<br>`_ACCURACY` | On | Gates the `KEEP` for a kernel patch integrated by the kernel lane. Set to `0` / `false` / `no` / `off` to fall back to a throughput-only `KEEP`. Disable only when the eval lane is known-broken: this gate is what stops a faster-but-wrong kernel from being kept. |
| `HYPERLOOM_EVAL_LIMIT` | Unset (full task set) | Caps the number of eval problems (`lm_eval --limit`). Useful for smoke runs; see the noise caveat below before using it on a run whose `KEEP` decisions matter. |

The tolerance is deliberately not an env knob: `ACCURACY_THRESHOLD` in
`src/hyperloom/common/gain_math.py` is a fixed
`0.05`, that is, a candidate must stay within 5 percentage points of the recorded
baseline accuracy.

Note that the score is measured once per candidate, not averaged over repeats.
On a full GSM8K run (1319 problems) the 5-point tolerance sits several standard
errors away from the baseline, so single-run noise does not trip it. Capping the
eval with a small `HYPERLOOM_EVAL_LIMIT` shrinks that margin sharply and can make
the gate noise-sensitive — prefer the full task set whenever a gate decision
depends on the result.

## Eval generation bounds

InferenceX runs lm-eval with `max_tokens=min(16384, ctx-4096)`, so a sample that
does not converge spends that entire budget, and 1319 of them can consume the
whole baseline timeout. Every generation is therefore given at most
`HYPERLOOM_EVAL_MAX_TOKENS` tokens, and never more than the served context
leaves after 4096 tokens of prompt room (half the context when it is no larger);
a launch whose context is unknown is refused with `eval_context_too_short`. Generation also stops on every
terminator the model's `generation_config.json` and `tokenizer_config.json`
declare — lm-eval's concurrent request path sends none, so a model like Qwen3,
which declares `eos_token_id` `[151645, 151643]`, would otherwise run with no
end-of-turn stop condition at all. The task's own stop strings keep their place
ahead of the model's, since its answer extraction depends on them, and the first
four are sent; stop token ids have no such limit.

A truncation probe counts the generations cut at the token limit. Once their
share reaches the probe's ratio after its minimum samples, every remaining
generation is answered empty instead of being sent, so a model that does not
terminate scores near 0 instead of running the eval out. The round's
`eval/hyperloom_eval_probe.json` records whether it tripped, how many samples it
saw and the share cut; check it before concluding a score is low for any other
reason.

All of these are part of the eval's definition, which keeps them equal across the
baseline and candidate arms. That symmetry is the whole point: the gate
compares a *difference* of two scores, so a bound that reaches only one arm biases
the verdict instead of merely limiting it. Prefer leaving these alone; if you do
change one, change it for the whole session rather than a single round. They are
read from the benchmark point's environment (`--extra-env`); a number that does not
parse refuses the round.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_EVAL_MAX_TOKENS` | `4096` | Per-generation token ceiling, a positive integer. |
| `HYPERLOOM_EVAL_PROBE` | On | Set to `0` / `false` / `no` / `off` to let every generation run to its ceiling. |
| `HYPERLOOM_EVAL_PROBE_MIN_SAMPLES` | `128` | Generations the probe counts before it may trip. |
| `HYPERLOOM_EVAL_PROBE_LENGTH_RATIO` | `0.75` | Share of generations cut at the ceiling that trips the probe. |
