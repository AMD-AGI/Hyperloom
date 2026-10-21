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
  of the launch it is set on.
- Do **not** treat a higher AL as a knob to push for more throughput; AL is
  determined by the golden curve, and forcing it above golden invalidates the
  result.
- Whether a variant may change a draft-length / gamma / method knob, or turn
  synthetic acceptance on, depends on who owns the launch (see *Native Magpie
  AgentX runs* below) and, on a Hyperloom-managed launch, on the base launch (see
  *Pinned when the base simulates acceptance* below). Where a variant may use
  synthetic acceptance — a Hyperloom-managed base with no speculative decoding —
  its AL is looked up for the variant's own draft length; it is never carried
  over or hand-set.

## Native Magpie AgentX runs: Magpie owns acceptance

On a native Magpie AgentX run the recipe and Magpie own the launch protocol,
including acceptance and the recipe-pinned speculative-decoding inputs. Do not
add, change or unset synthetic acceptance (a `SGLANG_SIMULATE_ACC_*` env or a
synthetic-acceptance value), and do not change a speculative-decoding input the
recipe pins (the golden-acceptance inputs `_validate_golden_launch` compares when
the resolved recipe locks them). Magpie rejects, before it is benchmarked, a candidate that violates
its protected lists or the recipe's golden-acceptance lock, and every other change
bundled into it goes unmeasured; a change Magpie lets through is still not
allowed by this policy.
The exact rules are Magpie's: read `PROTECTED_ARGS`, `PROTECTED_ENV`,
`PROTECTED_ENV_PREFIXES` and `_validate_golden_launch` in its
`Magpie/modes/benchmark/agentx_launch.py` (the session prompt gives the installed
path) rather than relying on a copy here. The golden-AL lookup below then only
confirms the recipe's own cell; spend variants on other knobs.

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

**Pinned when the base simulates acceptance.** This and the next two paragraphs
apply to a Hyperloom-managed launch (any run that is not native Magpie AgentX,
such as the MLPerf agentic backend). When the session's base launch
**simulates acceptance** (`SGLANG_SIMULATE_ACC_LEN`
is set, `--speculative-config` carries `rejection_sample_method: synthetic` or a
`synthetic_acceptance_length`, or an acceptance-length flag such as ATOM's
`--spec-decode-acceptance-length` is passed), the draft and its simulated
acceptance are pinned: a candidate that changes the draft length, top-k, block
size, method or draft model, or adds, changes or unsets any
`SGLANG_SIMULATE_ACC_*` env or synthetic-acceptance value, or drops
`--speculative-config` (a replacing argument string that omits it included), is
refused before it is benchmarked with `error_class: speculative_decoding_pinned`.
The lookup above then only confirms the base's own cell; spend variants on other
knobs. A `--speculative-config` replaces the whole object, so a variant that
changes a key that does not describe the draft (e.g. `attention_backend`) must
restate the base's pinned keys unchanged, or set that key alone with vLLM's
per-key form `--speculative-config.<key> <value>`, which merges over the object.

A base that **verifies a real draft** (GLM's
`SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`, Kimi's
`rejection_sample_method: block`) measures its acceptance, so a variant may change
the draft but may not switch to simulated acceptance (adding
`SGLANG_SIMULATE_ACC_LEN` or a synthetic `--speculative-config`): a declared AL is
not comparable with a measured one, and the variant is refused the same way.

A base that runs **no speculative decoding** has no acceptance to compare
against: a variant may enable it with a real draft, or with synthetic acceptance
whose AL is the golden value for its own draft length per the rules above.

Before proposing or marking `KEEP` on any such variant, cross-check against the
model's golden AL curve (if one exists per the lookup above):

1. Note the draft length the config uses. **Method matters for which knob is the
   key:** DSpark → gamma / block-size (not `--speculative-num-draft-tokens`);
   MTP / EAGLE / EAGLE3 → vLLM `num_speculative_tokens` / SGLang
   `--speculative-num-steps` (not `--speculative-num-draft-tokens`).
2. Look up golden AL for that exact `(model, method, draft length, thinking
   mode)` cell (e.g. `dsv4-pro-0813-dspark.yaml` key `6` → `3.77` for gamma 6,
   not draft_tokens 7).
