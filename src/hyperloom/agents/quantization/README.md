# quantization-agent (`hyperloom.agents.quantization`)

Hyperloom sub-agent that drives AMD Quark's PTQ workflow from a single
natural-language prompt.

## Scope

**What it is.** A thin agent-runtime driver with selectable Claude, Codex, or
Hermes transport. Every provider receives the same `SKILL.md` runtime contract
and invokes Quark's published skills end-to-end
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
- One selected agent runtime and its authentication: Claude Agent SDK,
  native Codex CLI/OAuth, or Hermes Agent/profile. Provider fallback is not
  performed by this package.
- Hermes terminal/file execution must run inside a verifiable outer container;
  set `HYPERLOOM_HERMES_EXTERNAL_SANDBOX=1` only inside that boundary. The
  transport fails closed without it.

## Usage — prompt is the only input

All quantization configuration travels through the user prompt. There is no
structured `quant_config` dict, no per-knob CLI flag for algorithm /
exclude_layers / calibration / scheme. Whatever Quark's intake + plan skills
can infer from the prompt is what gets used.

### Example prompt

```text
Quantize Qwen/Qwen3-8B with mxfp4 as the global scheme; override
self_attn modules to fp8, and use fp8 for the kv_cache. Export the result
to <workspace>/quantized.
Use gsm8k as the benchmark and accept up to a 5% relative eval gap.
No human interaction at any point.
```

### CLI

```bash
python -m hyperloom.agents.quantization.cli \
    --provider hermes \                         # claude | codex | hermes
    --prompt "$PROMPT" \                       # natural-language request
    --workspace /scratch/run-1/wks \           # per-run scratch dir
    --quark-root /path/to/Quark \              # or $QUARK_ROOT
    --interactive off \                        # auto | on | off
    --acceptable-eval-gap 0.05 \               # max relative quality gap
    --max-requantize-attempts 1                # Python-level retry cap
```

The equivalent console script is `quantization-agent`.

Exit codes: `0` success/partial · `1` failed · `2` argparse error. An
operator-rejected checkpoint is not a distinct code — it surfaces as `partial`
(exit 0) or `failed` (exit 1) with the reason in `assessment.notes`
(`eval_gap_exceeded_rejected` / `operator_declined_retry`).

The CLI prints a JSON summary (`status` + `quantized_model_dir` +
`assessment`) on stdout.

### Python (async)

```python
import asyncio
from hyperloom.agents.quantization import quantize_via_prompt

async def main():
    result = await quantize_via_prompt(
        PROMPT,
        workspace="/scratch/run-1/wks",
        quark_root="/path/to/Quark",
        interactive=False,
        acceptable_eval_gap=0.05,
        max_requantize_attempts=1,
        provider="hermes",
    )
    print(result.status)                       # success | partial | failed
    print(result.quantized_model_dir)          # Path | None
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

- `session_context.json` — handshake payload used by the selected agent runtime.
- `run_manifest.yaml` — Quark's workflow manifest (inputs, outputs, exec phases).
- `model_analysis.json`, `quant_plan.json` — intake + plan outputs.
- `validation_report.md` — validator results (4 steps: auxiliary / md5 /
  config / fuzzy, parsed from the report text).
- `source_eval.md`, `quantized_eval.md` — raw `quark-torch-llm-eval` Markdown.
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
| `HYPERLOOM_CODEX_HOME`               | —       | Dedicated absolute, non-symlink native OAuth home for the Codex CLI. When set, gateway credential variables are stripped from the child. |

Authentication belongs to the selected runtime: Claude Agent SDK credentials,
Codex OAuth/config, or the selected Hermes profile. Credentials are not passed
through the prompt or written to workspace artifacts.

## Tests

```bash
pytest src/hyperloom/agents/quantization/tests/
```

All tests run offline — no network, no GPU, no real agent-runtime calls (fake
SDK/subprocess fixtures are injected). The classifier suite covers each of the 30 outcome
IDs; the retry-loop suite covers counter persistence, hypothesis-gate,
operator promotion, and budget exhaustion.

## Public API

`from hyperloom.agents.quantization import …`

| symbol                | purpose                                             |
| --------------------- | --------------------------------------------------- |
| `quantize_via_prompt` | async entry; runs the full diagnose-fix-retry loop. |
