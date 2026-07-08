# Draft KB

Use this action when Conductor asks Critic to extract KB entries from a completed
action result, final optimization report, or run summary.

## Expected Input

The packet may contain:

- Final optimization report.
- Baseline and final benchmark results.
- Patch list and keep/revert decisions.
- Accuracy gate result.
- Sweep result.
- Robustness findings or RCA summaries.
- Existing KB snippets or conflict notes.
- Model family, model name, GPU, framework, and environment metadata.

Treat absent fields as unknown. Do not infer validation that the report does not
state.

## Candidate Selection

Create a KB draft only for lessons that are reusable across future runs:

- A controlled optimization that produced a validated gain.
- A pitfall that caused a revert, crash, or misleading benchmark.
- A benchmark methodology lesson.
- A framework, kernel, communication, or architecture constraint.
- A recovery pattern confirmed by Robustness or RCA.
- A target/framework comparison backed by measured evidence.

Reject candidates that are:

- Speculative ideas without a completed test.
- One-off implementation details with no reusable lesson.
- Results without comparable benchmark context.
- Micro-benchmark-only wins with no active-path or E2E evidence.
- Duplicates of an existing KB entry unless the new entry supersedes it.

## Categories

Use only these categories:

- `backend_exploration`
- `kernel_optimization`
- `call_stack_optimization`
- `server_params`
- `pitfall`
- `benchmark_methodology`
- `architecture_constraint`
- `target_comparison`
- `framework_comparison`
- `lesson`
- `crash_recovery`
- `dream_consolidation`

## Entry Fields

Every draft must include:

- `model_family`: partition key for the central KB.
- `model`: model name.
- `category`: one of the allowed categories.
- `action`: concise description of what was tried.
- `lesson`: reusable conclusion.

Include these when available:

- `gpu`
- `framework`
- `tags`
- `result`
- `confidence`
- `source`
- `context`
- `supersedes`
- `validated_date`
- `strategy_tested`

## Confidence

- Start at `0.9` for controlled A/B evidence with passing accuracy.
- Reduce to `0.75` if benchmark is controlled but the report omits minor
  environment details.
- Reduce to `0.6` or below if the result is useful but only partially validated.
- Reject instead of lowering confidence when benchmark or correctness evidence is
  missing for a claimed performance win.

## Result Field

For optimization entries, prefer:

```json
{
  "status": "KEEP",
  "gain_pct": 4.2,
  "baseline_tput_per_gpu": 1200.0,
  "final_tput_per_gpu": 1250.4,
  "accuracy": "pass"
}
```

For pitfalls or recovery entries, use the fields that best preserve the lesson:

```json
{
  "status": "REVERT",
  "failure_type": "accuracy_failed",
  "recovery": "restore previous dispatch path and clear compiled cache"
}
```

## Output

Return a JSON object matching `kb_draft_schema` in
[references/verdict_schema.md](../references/verdict_schema.md).

## Ownership Boundary

Critic is the only KB read/write/synthesis entrypoint. Other agents should
consume KB hints injected by Conductor, not read or write KB directly.
