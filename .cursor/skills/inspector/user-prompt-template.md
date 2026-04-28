# User Prompt Template — Inspector Binding Contract

The inspector is read-only and same-agent. It cannot enforce anything by
itself. The user prompt below is the **single point** that turns the inspector
into an enforceable check by binding the main agent to:

1. Invoke the inspector after every phase of the target skill ends.
2. Parse the inspector's `audit_report.json`.
3. Remediate every `block` / `fatal` violation before advancing.

Copy the template, fill in the slots in `<...>`, and use it as your initial
prompt to the agent. The "BINDING CONTRACT" section is verbatim — do not edit
its rules; only edit the slot values above and below it.

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

================================================================
INSPECTOR BINDING CONTRACT (do not modify or omit any rule)
================================================================

You will execute the target skill, but you will be audited after every phase
by the inspector skill at .cursor/skills/inspector/. The contract below is
binding for the entire conversation. Failing to follow it invalidates the
run.

Rule 1 — After-each-phase invocation
  After completing each numbered phase listed in the target skill's
  Orchestrator Loop (or, if no Orchestrator Loop is defined, each top-level
  action that completes), you MUST stop and invoke the inspector before
  reading any further action `.md` from the target skill.

  To invoke the inspector, do this in a single agent turn:
    (a) Read .cursor/skills/inspector/SKILL.md (if not already in context).
    (b) Execute the inspector's S1-S5 procedure verbatim with these inputs:
          TARGET_SKILL_DIR=<TARGET_SKILL_DIR>
          PHASE_NAME=<the phase that just ended, uppercase symbol>
          PHASE_INDEX=<integer index in the phase list, 1-based>
          PHASE_ACTION_FILES=<JSON array of one or more action files for this
                              phase, e.g. [actions/baseline.md] or, for a
                              DFS_LOOP iteration that ran kernel-opt,
                              [actions/kernel-opt.md, actions/integrate.md] —
                              IR-3 of the inference-optimization skill makes
                              integration mandatory after kernel-opt, so you
                              MUST include both files when auditing such a
                              loop. The legacy singular PHASE_ACTION_FILE is
                              still accepted but discouraged.>
          RUN_ENV=<JSON object with the env var values currently exported>
          MARKER_SENTENCE=<the first sentence of this user prompt, used by
                           find_transcript.py to disambiguate parallel sessions>
          MIN_TRANSCRIPT_LINES=<optional, default 5; raise only if the workflow
                               legitimately produces extremely short transcripts>

Rule 2 — Verdict parsing
  The inspector emits one fenced ```json block bracketed by
  === INSPECTOR_BEGIN ... === / === INSPECTOR_END ... === markers. Parse the
  JSON's `verdict` field. Branch as follows:

    PASS  -> Echo "inspector PASS for phase=<X>; advancing to phase=<X+1>".
             Continue with the next phase of the target skill.

    WARN  -> Echo "inspector WARN for phase=<X>; <one-line summary>; continuing".
             Continue with the next phase. If a `warn` violation has a
             `remediation` field that is cheap to apply (estimated <5 min)
             you SHOULD do so opportunistically before advancing, but you
             are not required to.

    BLOCK -> Mandatory remediation loop:
             For each violation in audit_report.violations whose severity is
             "block", in array order:
               (i)   Echo the violation `id` and `expected` field.
               (ii)  Execute the violation's `remediation` field verbatim.
                     If the remediation requires an unresolved env var, STOP,
                     ask the user, and resume after the user replies.
               (iii) After all block remediations have been attempted,
                     re-invoke the inspector for the SAME phase (Rule 1).
               (iv)  Loop until verdict is PASS or WARN, capped at 3
                     remediation attempts per phase. If attempt 3 still
                     returns BLOCK, treat the phase as FATAL (next branch).

    FATAL -> Hard stop:
             1. Echo "inspector FATAL for phase=<X>; rolling back".
             2. Perform rollback for this phase: revert any kernel patches,
                kill any servers started, rename `$RESULT_DIR/<phase>` to
                `$RESULT_DIR/<phase>.invalid_<ts>`.
             3. Set state.run_invalid = true in your working state.
             4. Skip all remaining DFS / loop work in the target skill.
             5. Jump directly to the target skill's REPORT phase. Include
                the fatal `audit_report.json` verbatim in the report and
                annotate the run as inspector-invalidated.
             6. Do NOT re-invoke inspector after FATAL.

Rule 3 — No skipping
  You MUST invoke the inspector after every phase even if you believe
  everything is fine. The inspector is the only mechanism that records
  evidence of completion. Skipping = treating the phase as never run.

Rule 4 — No arguing
  The inspector's findings are authoritative within its scope. Do not write
  prose like "I disagree with the inspector's finding" or "this BLOCK is a
  false positive". If you believe extraction is wrong, escalate by adding
  a brief note in the report at REPORT phase ("inspector flagged X; manual
  judgment: Y"). Do not skip the remediation loop.

Rule 5 — Self-check before advancing
  Before reading the next phase's action `.md`, you MUST be able to point
  to a recent `=== INSPECTOR_END phase=<current_phase> verdict=<PASS|WARN> ===`
  line in this conversation. If you cannot, return to Rule 1.

Rule 6 — Inspector self-failure
  If inspector itself emits an `inspector_self_failure` violation (e.g.
  transcript not found, target action `.md` missing), treat as BLOCK and
  remediate by fixing the inspector inputs (most often: missing env var or
  wrong TARGET_SKILL_DIR). Do NOT proceed to the next phase by ignoring a
  self-failure — that defeats the entire audit.

================================================================
END BINDING CONTRACT
================================================================

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

================================================================
INSPECTOR BINDING CONTRACT (do not modify or omit any rule)
================================================================

You will execute the target skill, but you will be audited after every phase
by the inspector skill at .cursor/skills/inspector/. The contract below is
binding for the entire conversation. Failing to follow it invalidates the
run.

Rule 1 — After-each-phase invocation
  After completing each of the following phases of the inference-optimization
  Orchestrator Loop, stop and invoke the inspector:
     1. SETUP        (after actions/setup.md completes)
     2. CLASSIFY     (after actions/classify.md completes)
     3. TARGET_ANALYSIS  (only if TARGET_DIR was provided; skip otherwise)
     4. KB_WARMUP    (after the kb_query.py warm-up call)
     5. BASELINE     (after actions/baseline.md completes, INCLUDING the
                      mandatory GSM8K eval)
     6. PROFILE      (after actions/profile.md completes)
     7. BUILD_ACTION_STACK (after the heuristic scoring populates the stack)
     8. DFS_LOOP_<N> (after each DFS LOOP iteration N=1,2,3,...; emit a
                      separate inspector audit per iteration)
     9. SWEEP        (after actions/sweep.md completes)
    10. REPORT       (after actions/report.md completes; this is the final
                      audit, then end the run)

  To invoke the inspector, in a single agent turn:
    (a) Read .cursor/skills/inspector/SKILL.md (if not already in context).
    (b) Execute the inspector's S1-S5 procedure with:
          TARGET_SKILL_DIR=.cursor/skills/inference-optimization
          PHASE_NAME=<one of: SETUP|CLASSIFY|TARGET_ANALYSIS|KB_WARMUP|
                              BASELINE|PROFILE|BUILD_ACTION_STACK|
                              DFS_LOOP_1|DFS_LOOP_2|...|SWEEP|REPORT>
          PHASE_INDEX=<1..10 or DFS sub-iteration>
          PHASE_ACTION_FILES=[actions/<phase_lower>.md]
                            (e.g. [actions/baseline.md];
                             for DFS_LOOP_<N> use the action file actually
                             executed in that iteration:
                               backends.md  -> [actions/backends.md]
                               params.md    -> [actions/params.md]
                               kernel-opt   -> [actions/kernel-opt.md,
                                                 actions/integrate.md]
                                 ^^ IR-3 binding: kernel-opt loops MUST also
                                    audit integrate.md, otherwise the
                                    "skipped integration" failure mode (see
                                    2026-04-21 Qwen3-30B-A3B run) goes
                                    undetected. The inspector's
                                    parse_action_outputs.py supports
                                    --action repeated; use it.)
          RUN_ENV={"MODEL_NAME":"Qwen3-14B",
                   "MODEL":"/shared_nfs/models/Qwen3-14B",
                   "TP":"8","CONC":"16","ISL":"1024","OSL":"256",
                   "FRAMEWORK":"sglang",
                   "WORK_DIR":"/workspace/inference-optimization/Qwen3-14B",
                   "RESULT_DIR":"/shared_nfs/inference-optimization/results/qwen3-14b-2026-04-21",
                   "TRACE_DIR":"/shared_nfs/inference-optimization/traces/qwen3-14b-2026-04-21",
                   "PORT":"8888","SKILL_ROOT":".cursor/skills/inference-optimization"}
          MARKER_SENTENCE="Run the inference-optimization skill at .cursor/skills/inference-optimization."

Rule 2-6: same as the blank template (verdict parsing, no skipping, no arguing,
self-check before advancing, inspector self-failure handling).

================================================================
END BINDING CONTRACT
================================================================

Begin by reading .cursor/skills/inference-optimization/SKILL.md, then proceed.
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

================================================================
INSPECTOR BINDING CONTRACT  (rules 1-6 verbatim from blank template above)
================================================================

Rule 1 — After-each-phase invocation
  After completing each phase listed in
  .cursor/skills/training-optimization/SKILL.md's Orchestrator Loop, invoke
  the inspector with TARGET_SKILL_DIR=.cursor/skills/training-optimization
  and the appropriate PHASE_NAME / PHASE_ACTION_FILES (a list — include
  every action `.md` actually executed in that phase, plus any action that
  an Iron Rule cross-references, e.g. integrate.md after kernel-opt.md).

(rules 2-6 unchanged)
```

The fully-generic invocation is the same regardless of target skill — only
`TARGET_SKILL_DIR`, the phase list, and the `RUN_ENV` keys differ. The
inspector reads the target skill's action `.md` files at runtime; no per-skill
configuration is needed.

---

## Quick-Start Cheat Sheet (for users in a hurry)

If you just want to add inspector to an existing user prompt with minimal
edits, append this snippet to the end of your prompt:

```
After every phase of the skill above, invoke the inspector at
.cursor/skills/inspector/SKILL.md with TARGET_SKILL_DIR=<the skill path>,
PHASE_NAME=<symbolic name>, PHASE_ACTION_FILES=<list of every action .md
actually executed in this phase, including any IR-cross-referenced file
such as integrate.md after kernel-opt.md>, RUN_ENV=<JSON of current env
vars>, then parse the audit_report.json. If verdict is BLOCK, run every
violation's `remediation` field before continuing and re-invoke the
inspector. If verdict is FATAL, jump to the target skill's REPORT phase.
Do not skip the inspector even if you think everything is fine.
```

This abbreviated form is less explicit than the full contract, so the
"agent-forgets-to-invoke" risk is higher. Use the full template for runs
longer than ~30 minutes.

---

## What if I am writing my own user prompt and refuse to use this template?

The minimum binding rules the prompt MUST contain to make the inspector
operational are:

1. Inspector is invoked after every phase (Rule 1 above).
2. The agent MUST execute every `block` violation's `remediation` field
   before advancing (Rule 2 BLOCK branch).
3. The agent MUST re-invoke inspector after remediation (Rule 2 BLOCK branch
   step iv).
4. FATAL triggers rollback and jump-to-REPORT (Rule 2 FATAL branch).
5. No-skip + no-arguing imperatives (Rules 3, 4).

Without all five, the inspector is suggestive at best.
