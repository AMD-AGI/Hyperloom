---
name: hyperloom-llm-report
description: Render the per-LLM-call report for a finished Hyperloom session — every API call's ISL/OSL, tokens (input/thinking/output), wall-clock, USD cost and tool calls, arranged as a phase → task → subtask → sub-sub-task tree, including the GEAK subprocess spend. Use when asked where a run's time or money went, to compare two models' runs, or to audit the trace ledgers' coverage.
---

# Hyperloom per-LLM-call report

Turns a session's trace ledgers into one report answering: *for every LLM API
call this run made, what did it cost in tokens, seconds and dollars, and which
task did it belong to?*

## Inputs

Everything comes from `<session_dir>/reports/trace/`:

| File | Cardinality | Written by |
| --- | --- | --- |
| `llm_calls.jsonl` | one row per **agentic turn** | in-process backends |
| `llm_calls_detail.jsonl` | one row per **API call**, joined on `call_id` | in-process backends |
| `ext/<component>-<pid>.jsonl` | turn rows from an out-of-process child | that child |
| `ext/<component>-<pid>.detail.jsonl` | its per-API-call rows | that child |
| `ext/geak-*.jsonl` | GEAK's spend | the harvester, from GEAK's Claude Code transcripts |

## Steps

1. **Locate the session directory.** It is the directory containing
   `reports/trace/`. If the user gave a model name rather than a path, look
   under `$USER_DATA_PATH` and the Hyperloom session roots for the newest
   session for that model. Confirm the path before rendering — a report for the
   wrong run is worse than no report.

2. **Render.** From the Hyperloom checkout:

   ```bash
   PYTHONPATH=src python3 -m hyperloom.inference_optimizer.tools.dump_llm_call_report \
       --session-dir <SESSION_DIR>
   ```

   Writes `llm_call_report.md` and `llm_call_report.json` to
   `$USER_DATA_PATH/reports/<session-id>/` (or `<session_dir>/reports/` when
   `USER_DATA_PATH` is unset). `--output-dir` overrides; `--max-depth N` trims
   the tree without changing any total.

3. **Read the Coverage section before quoting any number**, and carry its
   caveats into whatever you tell the user:

   - **Unpriced calls.** Excluded from every USD figure. If any exist, the cost
     total is a floor, not the bill — say so rather than presenting it as the
     bill.
   - **Cost sources.** `provider` is the vendor's own charge; `derived` is the
     shipped rate card applied to token counts. A total spanning both is an
     estimate.
   - **Apportioned timing.** `latency_ms` is always measured. The
     thinking/output split is measured only where the backend saw partial
     stream events; otherwise it is the measured span divided in proportion to
     the token counts. Report those two columns as estimates.
   - **Turns without detail rows.** Counted once, from the turn row, so their
     "per-call" line is really the turn's aggregate.
   - **No `geak` subtree.** If the run invoked the kernel agent and no `geak`
     node appears, GEAK's spend — historically the majority of a session's
     bill — is missing, and the session total is badly understated. Check that
     `reports/trace/ext/geak-*.jsonl` exists and that the harvester ran.

4. **Answer the question that was asked.** The markdown tree is the artifact;
   the JSON (`tree` → `totals` per node) is what to compute from when comparing
   two runs. When comparing models, compare `usd_total` and `ms_total` per
   phase, not session totals alone — a cheaper session that spent it all in one
   phase is the finding.

## Reading the numbers

- **ISL** = `input_tokens + cache_read + cache_creation`; **OSL** =
  `output_tokens + reasoning_output_tokens`.
- **Thinking sits beside output, not inside it.** `output_tokens` excludes
  thinking; OSL is output + thinking. Likewise `cost_usd` = input + output +
  thinking + cache, so thinking is a part of the total but not a part of
  `cost_output_usd`.
- **Share-of-parent** is by cost where the parent has any, and by call count
  otherwise.

## Do not

- Do not sum `llm_calls.jsonl` and `llm_calls_detail.jsonl` together — the
  detail rows are the expansion of the turn rows, and the tool already avoids
  counting a turn twice.
- Do not append to `llm_calls.jsonl` from any tooling; that append is not
  atomic across processes. Out-of-process producers write their own `ext/` shard.
