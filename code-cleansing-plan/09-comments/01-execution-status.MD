# Step 09 Execution Status

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Design Intent Understood

`09-compress-comments-and-docs.MD` asks to compress process-history comments, fix stale headers, and keep prompts/SKILL/README aligned with current behavior. Execution order step 1 is "fix obviously stale file headers".

## Comment Style Rules (adopted)

Keep a comment only when it explains something the code cannot show:

- external protocol / data-format constraints,
- runtime invariants,
- non-obvious failure modes,
- why a seemingly redundant guard exists,
- compat deletion conditions.

Delete or compress:

- PR / Gap / Nxx process narrative,
- line-by-line restatements of the code,
- "why it once broke" stories that no longer explain the current design,
- descriptions of removed modules,
- block comments duplicating the function docstring.

Preferred form for retained compat notes:

```
Legacy X for archived sessions. Remove after old-session resume is dropped.
```

## Implemented In This Step

### Fixed stale `agent_role.py` module header

The header listed `robustness` as an active 4th role in the table while also claiming the "robustness role ... Removed in the legacy release" just below — a direct contradiction flagged during exploration.

Rewrite:

- The roster is exactly the four active roles (orchestration, kernel, critic, robustness).
- `sage` merged into Critic; `triage` renamed to `robustness`; parliament `OBJECTION`/`VOTE` removed.
- No separate framework role: framework PR runs as the Coordinator-owned FRAMEWORK_PR phase (consistent with Step 07).

This is a module docstring only; no behavior change and no test asserts it.

### Compressed specialist/prompt/SKILL stale wording

Updated:

- `inference_optimizer/SKILL.md`
- `orchestrator/system_prompts/prompt_builder.py`
- `orchestrator/specialist_runner.py`

Changes:

- Replaced stale `runs/{...,backends,params,...}` session-layout example
  with `explore` / `sweep`.
- Compressed the retired-module section to the current invariant:
  do not recreate retired `backends` / `params` / `validate_stack` /
  scoring modules.
- Updated the user report checklist from "params accepted/rejected" to
  "explore accepted/rejected".
- Removed PR-A / Arbor process narrative from specialist emit-hint and
  tool-whitelist comments while keeping the worktree and KB-write
  invariants.

## Verification

- Lint: no errors.
- Tests: `test_agent_roles_and_policy`, `test_role_realignment` pass.

## Stale-Doc Candidates For Future Passes

These are known drift points; defer to avoid touching prompt-asserted text in this batch:

- System prompts / SKILL long historical narrative (touch carefully; covered by prompt-visibility tests — change structure, not asserted contract lines).
- Coordinator / SharedState / policy in-line PR/Gap/Nxx narrative: large surface; compress incrementally.

## Risk Control Applied

- Comment-only edit; no behavior change in this batch.
- Did not touch prompt-asserted text (prompt visibility / action catalogue contracts stay intact).
- Kept the change single-purpose.
