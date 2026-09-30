<!-- when: proposing or accepting an explore/sweep variant that touches a speculative-decoding knob (draft length / gamma / block-size / num-draft-tokens / method, or a synthetic-acceptance value) -->
<!-- phase: FRAMEWORK_AGENT, SWEEP -->
# Speculative decoding — validate against golden Acceptance Length (AL)

## What golden AL is

InferenceX maintains a committed reference of **golden Acceptance Length
(AL)** curves at `golden_al_distribution/` in the InferenceX repo. Each curve
maps `(model, thinking mode, draft length / num_speculative_tokens)` to a mean
AL measured with SPEED-Bench's `coding` category — the standard InferenceX (and
increasingly vLLM/SGLang/TensorRT-LLM upstream) uses so speculative-decoding
submissions for the same model are comparable. Methodology, the AL formula, and
per-engine wiring (`SGLANG_SIMULATE_ACC_LEN`, vLLM's
`synthetic_acceptance_length`, `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`,
ATOM's `--spec-decode-acceptance-length`) are documented in that directory's own
`README.md`.

## AL is not a tunable knob — the golden YAML is authoritative

During the explore/sweep stage the agent optimizes throughput/latency knobs, but
**acceptance length (AL) is not one of them.** AL is a fixed property of the
`(model, method, draft length, thinking mode)` cell as committed in the
InferenceX `golden_al_distribution/` YAML — **read it from there and treat it as
ground truth**, never set, raise, or "tune" it to a value that makes a variant
look better.

Concretely:

- Do **not** invent, round, or adjust an AL / `SGLANG_SIMULATE_ACC_LEN` /
  `synthetic_acceptance_length` value to taste. If synthetic acceptance is used,
  its value **must** be the golden AL looked up for the exact gamma/draft length
  of the variant.
- Do **not** treat a higher AL as a knob to push for more throughput; AL is
  determined by the golden curve, and forcing it above golden invalidates the
  result.
- If the variant changes a draft-length / gamma / method knob, the AL reference
  **changes with it** and must be re-looked-up from the golden YAML — it is
  never carried over or hand-set.

## Do NOT guess the golden-AL filename from the model name

Filenames vary by both model *and* speculative method, and a model can have
multiple files for different draft/verify variants (e.g. `dsv4_mtp.yaml`,
`dsv4-pro-0813-dspark.yaml`, `kimik2.5_eagle3.yaml`,
`minimaxm3_eagle3_gqa.yaml`). Hyphens vs. underscores and method suffixes are
inconsistent — never construct `{model-slug}-dspark.yaml` and assume it exists.
Instead:

1. Read `golden_al_distribution/README.md`'s **"Current golden curves"** table
   (or list the directory) and match on **model name AND speculative method**
   (DSpark / MTP / EAGLE3 / EAGLE3-GQA / block-verify variant) — the method the
   current recipe actually uses, not just the model name.
2. If no row matches the model+method, there is **no golden AL reference** for
   this configuration. Do not fabricate one — treat any speculative-decoding
   change as unvalidated (call it out as such) rather than assuming it passes.
3. The table is a living document (new models land via `speedbench-al.yml`) —
   treat any cached copy as possibly stale; re-fetch when unfamiliar.

## When this applies (framework-agnostic)

Apply to any framework whose recipe touches the speculative-decoding knobs:

- **SGLang DSpark**: `--speculative-dspark-block-size` (gamma) — this is the
  golden-table key. `--speculative-num-draft-tokens` is the verify window
  (gamma + 1) and is **not** the lookup key for DSpark curves.
- **SGLang MTP/EAGLE**: `--speculative-num-steps` (the draft length, same as
  vLLM's `num_speculative_tokens`). With `--speculative-eagle-topk 1`,
  `--speculative-num-draft-tokens` is `num_steps + 1` and is **not** the
  lookup key.
- **vLLM**: `num_speculative_tokens` inside `--speculative-config` JSON, and any
  switch of `method` (eagle3 / mtp / dspark) — golden AL is keyed on method as
  well as model, so a method change invalidates the previously matched row.
- **TensorRT-LLM / ATOM**: their respective draft-length equivalents.

On an **agentic recipe** (the session's `AGENTX_SERVER_SCRIPT` sits under
`agentic/`) the recipe already simulates acceptance for its own draft, so the
draft and its acceptance are pinned: a variant that changes the draft length,
block size, top-k, method or draft model, sets any synthetic-acceptance value,
or removes `--speculative-config` is refused when the variant is built. The
lookup above then only confirms the recipe's own cell; spend variants on other
knobs. `--speculative-config` keys that do not describe the draft (e.g.
`attention_backend`) still merge into the recipe's config.

Before proposing or marking `KEEP` on any such variant, cross-check against the
model's golden AL curve (if one exists per the lookup above):

1. Note the draft length the config uses. **Method matters for which knob is the
   key:** DSpark → gamma / block-size (not `--speculative-num-draft-tokens`);
   MTP / EAGLE / EAGLE3 → vLLM `num_speculative_tokens` / SGLang
   `--speculative-num-steps` (not `--speculative-num-draft-tokens`).
2. Look up golden AL for that exact `(model, method, draft length, thinking
   mode)` cell (e.g. `dsv4-pro-0813-dspark.yaml` key `6` → `3.77` for gamma 6,
   not draft_tokens 7).
