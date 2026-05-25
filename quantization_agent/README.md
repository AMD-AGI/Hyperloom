# quantization_agent

Hyperloom sub-agent that drives AMD Quark's PTQ workflow from a single
natural-language prompt.

## Scope

**What it is.** A thin Claude-SDK driver. It loads `SKILL.md` as the runtime
contract and lets the LLM invoke Quark's published skills end-to-end
(`quark-ptq` → `quark-quantization-result-validator` → `quark-llm-eval`),
classifies each attempt's workspace state into a 30-row outcome matrix, and
exposes one diagnose-fix-retry loop on top.

**What it is not.** It does not implement quantization algorithms (Quark
does), does not talk to GPUs directly, and does not bundle any model files.
All ML work runs inside Quark's skills.

## Requirements

- `$QUARK_ROOT` pointing to an `amd-quark` checkout that contains
  `.claude/skills/quark-ptq/SKILL.md` (and the validator / eval skills under
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
把 /group/.../Qwen/Qwen3-8B 量化为 mxfp4：其中 self_attn 模块用 fp8，
kv_cache 用 fp8，导出到 <workspace>/quantized。导出后做
quark-quantization-result-validator 校验，然后做 quark-llm-eval (gsm8k)。
可接受 5% 的 eval gap。全程不需要人工交互。
```

### CLI

```bash
python -m quantization_agent.cli \
    --prompt "$PROMPT" \                       # natural-language request
    --workspace /scratch/run-1/wks \           # per-run scratch dir
    --quark-root /path/to/Quark \              # or $QUARK_ROOT
    --interactive off \                        # auto | on | off
    --acceptable-eval-gap 0.05 \               # max relative quality gap
    --max-requantize-attempts 1                # Python-level retry cap
```

Exit codes: `0` success/partial · `1` failed · `2` argparse error ·
`3` operator-rejected checkpoint.

The CLI prints a JSON summary (`status` + `quantized_model_dir` +
`assessment`) on stdout.

### Python (async)

```python
import asyncio
from quantization_agent import quantize_via_prompt

async def main():
    result = await quantize_via_prompt(
        PROMPT,
        workspace="/scratch/run-1/wks",
        quark_root="/path/to/Quark",
        interactive=False,
        acceptable_eval_gap=0.05,
        max_requantize_attempts=1,
    )
    print(result.status)                       # success | partial | failed
    print(result.quantized_model_dir)          # Path | None
    print(result.assessment.final)             # OutcomeId | None

asyncio.run(main())
```

A `quantize_via_prompt_sync` wrapper is also exported for non-async callers.

## Return shape

`QuantSkillRunResult` (frozen dataclass):

| field                 | type                | meaning                                                                   |
| --------------------- | ------------------- | ------------------------------------------------------------------------- |
| `status`              | `str`               | `"success"` / `"partial"` / `"failed"`                                    |
| `quantized_model_dir` | `Path` or `None`    | absolute path to the exported model (HF format); `None` on failure        |
| `assessment`          | `Assessment`        | structured per-attempt verdict                                            |

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

The agent writes a small, stable set of files under `--workspace`. Callers
can read these directly:

- `session_context.json` — handshake payload passed to the SDK at session start.
- `run_manifest.yaml` — Quark's workflow manifest (inputs, outputs, exec phases).
- `model_analysis.json`, `quant_plan.json` — intake + plan outputs.
- `validation_report.md` + `val_<step>.json` — validator results (4 steps).
- `source_eval.md`, `quantized_eval.md` — raw `quark-llm-eval` Markdown.
- `eval_report.json` — synthesized eval summary (`source_score`,
  `quantized_score`, `relative_gap`, `within_threshold`).
- `eval_gap_threshold.txt` — resolved acceptable gap (single float).
- `last_phase.txt` — current Quark phase ID (used for classification).
- `requantize_attempts.txt` — persistent integer retry counter.
- `fix_hypothesis_attempt_N.md` — diagnosis + concrete fix (precondition for
  retry N+1).
- `blocked.md` — present when the SDK aborted; may carry `outcome_id: <id>`
  to short-circuit classification.

## Environment knobs

| var                                  | default | effect                                                                     |
| ------------------------------------ | ------- | -------------------------------------------------------------------------- |
| `QUARK_ROOT`                         | —       | Path to amd-quark checkout. Required unless passed as `quark_root=` kwarg. |
| `HYPERLOOM_QUANT_STRICT_VALIDATION`  | `1`     | When `0`, MUST-validate SKIPPED demotes to `partial` instead of `failed`.  |

Claude SDK auth (`ANTHROPIC_API_KEY` or equivalent) is handled by
`claude-agent-sdk` and is not read directly by this package.

## Tests

```bash
pytest quantization_agent/tests/
```

All tests run offline — no network, no GPU, no Claude SDK calls (a fake-SDK
fixture is injected). The classifier suite covers each of the 30 outcome
IDs; the retry-loop suite covers counter persistence, hypothesis-gate,
operator promotion, and budget exhaustion.

## Public API

`from quantization_agent import …`

| symbol                       | purpose                                                                 |
| ---------------------------- | ----------------------------------------------------------------------- |
| `quantize_via_prompt`        | async entry; runs the full diagnose-fix-retry loop.                     |
| `quantize_via_prompt_sync`   | `asyncio.run` wrapper for non-async callers.                            |
| `QuantSkillRunResult`        | return dataclass (`status` / `quantized_model_dir` / `assessment`).     |
| `Assessment`                 | per-attempt verdict dataclass.                                          |
| `OutcomeId`                  | StrEnum of all 30 outcome IDs.                                          |
| `AUTO_RECOVER`               | frozenset of outcomes SKILL.md is expected to self-heal in-session.     |
| `AUTO_FAIL`                  | frozenset of outcomes that always end the run as `failed`.              |
| `ASK`                        | frozenset of outcomes that surface to the operator.                     |
| `ASK_RETRYABLE`              | subset of `ASK` that increments the retry counter when a fix exists.    |
| `SUCCESS_TAGS`               | outcomes treated as success when ending a multi-attempt trail.          |
| `UNCLASSIFIED_FAILURE`       | catch-all outcome (#30).                                                |
