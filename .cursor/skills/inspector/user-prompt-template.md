# User Prompt Template — Inspector

The inspector is read-only and same-agent. It cannot enforce anything by
itself. The user prompt below is the **single point** that turns the
inspector into an operational check by binding the main agent to:

1. Run a self-check after every phase of the target skill.
2. Read the on-disk `audit_report.json` written by the inspector.
3. Quietly remediate any `block` / `fatal` finding before advancing.

The audit machinery is intentionally **implicit**: the user reading this
prompt sees a one-paragraph "Audit conventions" block, not a multi-page
binding contract. The chat during the run shows only one
`[Inspection] ... -> <path>` line per audit. The full report, the run-wide
audit history, and the next-window sentinel all live under
`$RESULT_DIR/.audit/`.

---

## Blank Template (copy-paste base)

```
Run the <TARGET_SKILL_NAME> skill at <TARGET_SKILL_DIR>.

Goal: <one-line task description, e.g. "optimize Qwen3-14B inference on 8xMI355X">

Run inputs:
  MODEL_NAME=<...>
  MODEL=<absolute model path>
  TP=<...>
  CONC=<...>
  ISL=<...>
  OSL=<...>
  FRAMEWORK=<sglang|vllm>
  WORK_DIR=<...>
  RESULT_DIR=<...>
  TRACE_DIR=<...>
  <any other env vars the target skill expects>

Audit conventions

After completing each phase listed in the target skill's Orchestrator
Loop, run the self-check defined in .cursor/skills/inspector/SKILL.md
(steps S1–S5) with TARGET_SKILL_DIR=<TARGET_SKILL_DIR> and
PHASE_NAME=<the phase that just ended>. The inspector writes its full
report to $RESULT_DIR/.audit/<PHASE>_<ts>.json and prints a single
[Inspection] ... -> <path> line into the chat. Treat that line as your
only audit-related chat output:

  - PASS / WARN: continue to the next phase. Do not narrate the audit.
  - BLOCK: open the on-disk report, execute every violation's
           remediation field as the next natural step in the run, then
           re-run the self-check for the same phase. Cap at 3 remediation
           cycles before treating the phase as FATAL.
  - FATAL: roll back the phase, mark the run invalid, jump to REPORT,
           and emit one business-language line explaining the stop
           reason (e.g. "Stopping run: GSM8K accuracy regressed below
           the 0.65 floor.").

Do not echo the inspector's verdict in additional prose, do not paste the
audit_report.json into chat, and do not skip the self-check even if you
believe everything is fine. The on-disk report at
$RESULT_DIR/.audit/ is the only authoritative audit artifact.

Begin by reading <TARGET_SKILL_DIR>/SKILL.md, then proceed.
```

---

## Filled Example: inference-optimization for Qwen3-14B

```
Run the inference-optimization skill at .cursor/skills/inference-optimization.

Goal: optimize Qwen3-14B output throughput on 8xMI355X using SGLang.

Run inputs:
  MODEL_NAME=Qwen3-14B
  MODEL=/shared_nfs/models/Qwen3-14B
  TP=8
  CONC=16
  ISL=1024
  OSL=256
  FRAMEWORK=sglang
  WORK_DIR=/workspace/inference-optimization/Qwen3-14B
  RESULT_DIR=/shared_nfs/inference-optimization/results/qwen3-14b-2026-04-21
  TRACE_DIR=/shared_nfs/inference-optimization/traces/qwen3-14b-2026-04-21
  KERNEL_OPT_BACKENDS=geak,codex
  KERNEL_OPT_IMAGE=ci-images.amd.com/sglang:2026-04-15
  GEAK_LOCAL=true
  PORT=8888

Audit conventions

After each of the following phases of the inference-optimization
Orchestrator Loop, run the self-check at .cursor/skills/inspector/SKILL.md
with TARGET_SKILL_DIR=.cursor/skills/inference-optimization, the matching
PHASE_NAME, and the action files actually executed:

  1. SETUP                 PHASE_ACTION_FILES=[actions/setup.md]
  2. CLASSIFY              [actions/classify.md]
  3. TARGET_ANALYSIS       [actions/target_analysis.md]   (skip if TARGET_DIR unset)
  4. KB_WARMUP             [actions/kb_warmup.md]
  5. BASELINE              [actions/baseline.md]          (mandatory GSM8K eval)
  6. PROFILE               [actions/profile.md]
  7. BUILD_ACTION_STACK    [actions/build_action_stack.md]
  8. DFS_LOOP_<N>          per-iteration; for kernel-opt iterations include BOTH
                           [actions/kernel-opt.md, actions/integrate.md] (IR-3)
  9. SWEEP                 [actions/sweep.md]
 10. REPORT                [actions/report.md]            (terminal phase)

The inspector writes audit_report.json to
$RESULT_DIR/.audit/<PHASE>_<ts>.json and prints one [Inspection] line
per phase. Read the on-disk report only when verdict is BLOCK or FATAL,
and execute violations[*].remediation as natural next steps.

Pass MARKER_SENTENCE="Run the inference-optimization skill at
.cursor/skills/inference-optimization." to find_transcript.py to
disambiguate parallel sessions.

Begin by reading .cursor/skills/inference-optimization/SKILL.md, then
proceed.
```

---

## Filled Example: training-optimization (any model)

```
Run the training-optimization skill at .cursor/skills/training-optimization.

Goal: minimize ms/iter for Llama-3-8B continued pretraining on 8xMI355X.

Run inputs:
  MODEL_NAME=Llama-3-8B
  MODEL=/shared_nfs/models/Llama-3-8B
  GLOBAL_BATCH_SIZE=512
  TP=2
  PP=2
  DP=2
  WORK_DIR=/workspace/training-optimization/Llama-3-8B
  RESULT_DIR=/shared_nfs/training-optimization/results/llama3-8b-2026-04-21

Audit conventions

After each phase listed in
.cursor/skills/training-optimization/SKILL.md's Orchestrator Loop, run
the self-check at .cursor/skills/inspector/SKILL.md with
TARGET_SKILL_DIR=.cursor/skills/training-optimization. Include in
PHASE_ACTION_FILES every action .md actually executed in that phase plus
any action that an Iron Rule cross-references (e.g. integrate.md after
kernel-opt.md). Read the on-disk audit_report.json from
$RESULT_DIR/.audit/ only when verdict is BLOCK or FATAL; execute
violations[*].remediation as natural next steps; do not narrate the
audit in chat beyond the inspector's own [Inspection] line.
```

The fully-generic invocation is the same regardless of target skill — only
`TARGET_SKILL_DIR`, the phase list, and the `RUN_ENV` keys differ. The
inspector reads the target skill's action `.md` files at runtime; no
per-skill configuration is needed.

---

## Quick-Start Cheat Sheet (for users in a hurry)

If you just want to add the inspector to an existing user prompt with
minimal edits, append this snippet to the end of your prompt:

```
After every phase, run the self-check at .cursor/skills/inspector/SKILL.md
with TARGET_SKILL_DIR=<the skill path>, PHASE_NAME=<symbolic name>, and
PHASE_ACTION_FILES=<list of every action .md actually executed in this
phase, including any IR-cross-referenced file such as integrate.md after
kernel-opt.md>. The inspector writes audit_report.json to
$RESULT_DIR/.audit/<PHASE>_<ts>.json and prints one [Inspection] line.
If verdict is BLOCK, read the report and run every violation's
remediation field as your natural next step, then re-run the self-check.
If verdict is FATAL, roll back and stop with one business-language
sentence. Do not skip the self-check even if you think everything is
fine.
```

---

## What if I am writing my own user prompt and refuse to use this template?

The minimum binding rules the prompt MUST contain to make the inspector
operational are:

1. The self-check runs after every phase.
2. The agent MUST execute every `block` violation's `remediation` field
   (read from the on-disk `audit_report.json`) before advancing.
3. The agent MUST re-run the self-check after remediation, capped at 3
   cycles, before treating as FATAL.
4. FATAL triggers rollback and jump-to-REPORT with one business-language
   stop sentence.
5. The agent does not echo verdict prose into the chat beyond the
   inspector's own `[Inspection] ...` line. Audit history is recovered
   from `$RESULT_DIR/.audit/` not from the chat.

Without all five, the inspector is suggestive at best.
