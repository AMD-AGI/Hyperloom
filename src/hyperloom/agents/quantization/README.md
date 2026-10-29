# quantization-agent (`hyperloom.agents.quantization`)

Hyperloom sub-agent that drives AMD Quark's PTQ workflow from a single
natural-language prompt.

## Scope

**What it is.** A thin Claude-SDK driver. It loads `SKILL.md` as the runtime
contract and lets the LLM invoke Quark's published skills end-to-end
(`quark-torch-ptq` → `quark-torch-result-validator` → `quark-torch-llm-eval`),
classifies each attempt's workspace state into a 30-row outcome matrix, and
exposes one diagnose-fix-retry loop on top.

**What it is not.** It does not implement quantization algorithms (Quark
does), does not talk to GPUs directly, and does not bundle any model files.
All ML work runs inside Quark's skills.

## Requirements

- `$QUARK_ROOT` pointing to an `amd-quark` checkout that contains
  `.claude/skills/quark-torch-ptq/SKILL.md` (and the validator / eval skills under
  the same tree).
- Python deps (`claude-agent-sdk`, `PyYAML`) come from Hyperloom's top-level
  `pyproject.toml` — no separate install script.
- A working Claude SDK authentication (e.g. `ANTHROPIC_API_KEY` in env, or
  any auth method the SDK accepts).

## Usage — prompt is the only input

All quantization configuration travels through the user prompt. There is no
structured `quant_config` dict, no per-knob CLI flag for algorithm /
exclude_layers / calibration / scheme. Whatever Quark's intake + plan skills
can infer from the prompt is what gets used.

### Example prompt

```text
Quantize Qwen/Qwen3-8B with mxfp4 as the global scheme; override
self_attn modules to fp8, and use fp8 for the kv_cache. Export the result
to quantized/ in your workspace.
Use gsm8k as the benchmark and accept up to a 5% relative eval gap.
No human interaction at any point.
```

### CLI

```bash
python -m hyperloom quantize \
    --prompt "$PROMPT" \                       # natural-language request
    --workspace /scratch/qwen3-8b \            # model workspace; each run writes under runs/<run_id>/
    --quark-root /path/to/Quark \              # or $QUARK_ROOT
    --interactive off \                        # auto | on | off
    --acceptable-eval-gap 0.05 \               # max relative quality gap
    --max-requantize-attempts 1                # Python-level retry cap
```

The equivalent console script is `hyperloom quantize`.

Exit codes: `0` success/partial · `1` failed · `2` argparse error. An
operator-rejected checkpoint is not a distinct code — it surfaces as `partial`
(exit 0) or `failed` (exit 1) with the reason in `assessment.notes`
(`eval_gap_exceeded_rejected` / `operator_declined_retry`).

The CLI prints a JSON summary (`status` + `quantized_model_dir` + `run_dir` +
`assessment`) on stdout.

### Python (async)

```python
import asyncio
from hyperloom.agents.quantization import QuantSkillRunOptions, quantize_via_prompt

async def main():
    result = await quantize_via_prompt(
        PROMPT,
        workspace="/scratch/qwen3-8b",
        options=QuantSkillRunOptions(
            quark_root="/path/to/Quark",
            interactive=False,
            acceptable_eval_gap=0.05,
            max_requantize_attempts=1,
        ),
    )
    print(result.status)                       # success | partial | failed
    print(result.quantized_model_dir)          # Path | None
    print(result.run_dir)                      # Path | None
    print(result.assessment.final)             # OutcomeId | None

asyncio.run(main())
```

## Return shape

`QuantSkillRunResult` (frozen dataclass):

| field                 | type                | meaning                                                                   |
| --------------------- | ------------------- | ------------------------------------------------------------------------- |
| `status`              | `str`               | `"success"` / `"partial"` / `"failed"`                                    |
| `quantized_model_dir` | `Path` or `None`    | absolute path to the exported model (HF format); `None` on failure        |
| `assessment`          | `Assessment`        | structured per-attempt verdict                                            |
| `run_dir`             | `Path` or `None`    | this run's directory under `<workspace>/runs/`; `None` if the run never started |

`Assessment` (frozen dataclass):

| field        | type                          | meaning                                                            |
| ------------ | ----------------------------- | ------------------------------------------------------------------ |
| `final`      | `OutcomeId` or `None`         | primary verdict (`None` = clean success)                           |
| `attempts`   | `tuple[OutcomeId \| None, …]` | per-attempt outcomes in chronological order                        |
| `recovered`  | `bool`                        | `True` iff `len(attempts) > 1` and final ∈ success                 |
| `eval_gap`   | `float` or `None`             | `relative_gap` from the final attempt's `eval_report.json`         |
| `notes`      | `tuple[str, …]`               | retry-loop decision notes                                          |

The full 30-outcome enumeration lives in `driver/outcomes.py` (`OutcomeId`,
`AUTO_RECOVER`, `AUTO_FAIL`, `ASK`, `SUCCESS_TAGS`).

## Workspace artifacts

Every `quantize_via_prompt` call is one run, and each run gets its own
directory, `<workspace>/runs/<utc-timestamp>-<random>/` (`run_dir` in the
result). Every attempt of the run uses it as the agent's workspace, so all the
files below, the exported model included, belong to that run: a rerun of the
same model writes a new run directory and never reads or overwrites an earlier
run's files. Files written directly under `<workspace>` by runs before this
layout are left in place and not read. Callers can read these directly:

- `session_context.json` — handshake payload passed to the SDK at session start.
- `run_manifest.yaml` — Quark's workflow manifest (inputs, outputs, exec phases).
- `model_analysis.json`, `quant_plan.json` — intake + plan outputs.
- `validation_report.md` — validator results (4 steps: auxiliary / md5 /
  config / fuzzy, parsed from the report text).
- `source_eval.md`, `quantized_eval.md` — raw `quark-torch-llm-eval` Markdown.
- `eval_report.json` — synthesized eval summary (`source_score`,
  `quantized_score`, `relative_gap`, `within_threshold`).
- `eval_gap_threshold.txt` — the acceptable gap the prompt stated (single
  float); it applies to this run only.
- `last_phase.txt` — current Quark phase ID (used for classification).
- `fix_hypothesis_attempt_N.md` — diagnosis + concrete fix (precondition for
  retry N+1). Retries are counted per run: every `quantize_via_prompt` call
  gets its own `max_requantize_attempts`.
- `blocked.md` — present when the SDK aborted; may carry `outcome_id: <id>`
  to short-circuit classification.
- `<marker>.attempt_N` — the Nth archived set of markers in this run: the
  `blocked.md`, `last_phase.txt`, `eval_report.json` and `eval_skipped.txt` an
  earlier attempt left, renamed before the next attempt starts so that attempt
  is judged on its own files. An attempt that stops without writing its own
  `blocked.md`, `eval_report.json` or `eval_skipped.txt` is judged only on the
  error it raised (`unclassified_failure` when it raised none, or when the
  error maps to an auto-recover outcome), never on a model and reports an
  earlier attempt left, and never as a clean success, so it fails the run.

## Environment knobs

| var                                  | default | effect                                                                     |
| ------------------------------------ | ------- | -------------------------------------------------------------------------- |
| `QUARK_ROOT`                         | —       | Path to amd-quark checkout. Required unless passed as `options=QuantSkillRunOptions(quark_root=...)`. |
| `HYPERLOOM_QUANT_STRICT_VALIDATION`  | `1`     | When `0`, MUST-validate SKIPPED demotes to `partial` instead of `failed`.  |

Claude SDK auth (`ANTHROPIC_API_KEY` or equivalent) is handled by
`claude-agent-sdk` and is not read directly by this package.

## Tests

```bash
pytest src/hyperloom/agents/quantization/tests/
```

All tests run offline — no network, no GPU, no Claude SDK calls (a fake-SDK
fixture is injected). The classifier suite covers each of the 30 outcome
IDs; the retry-loop suite covers counter persistence, hypothesis-gate,
operator promotion, and budget exhaustion.

## Public API

`from hyperloom.agents.quantization import …`

| symbol                | purpose                                             |
| --------------------- | --------------------------------------------------- |
| `quantize_via_prompt` | async entry; runs the full diagnose-fix-retry loop. |
