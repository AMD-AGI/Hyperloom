# Atom full-support live verification — acceptance report

**Verdict:** **DEFERRED — live run pending**

**Reason:** The implementing sandbox cannot execute the 12-hour live
session described in [`7.2_launch.md`](7.2_launch.md). Blockers:

1. **GPU count.** The sandbox exposes 4 ROCm GPUs; the launch plan
   targets 8×MI355X. `--tp 4` would still work numerically on a
   4-GPU box, but IR-1 (every visible GPU unoccupied) and the
   plan's TP-isolation policy assume the planned 8-GPU layout.
2. **LLM gateway credentials.** `SAFE_API_KEY` and
   `OPENAI_BASE_URL` are unset in the sandbox shell. Without these
   the Coordinator's LLM backends fail on first call. The launch
   plan (7.2) requires both to be exported.
3. **`install.sh` not sourced in this shell.** `KERNEL_AGENT_ENV` is
   unset and `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` is empty.
   Per IR-2, the launch must follow a fresh
   `bash inference_optimizer/scripts/install.sh` in the same shell.

The verdict will flip to GREEN / RED after the live operator runs
the launch on the actual 8×MI355X box. The criteria tables below
are scaffolded for that follow-up; the must-have / nice-to-have
columns stay `_TBD_` until evidence arrives.

---

## Must-have criteria

| ID | Criterion | Pass? | Evidence |
|---|---|---|---|
| M1 | Session completes without unrecovered crashes | _TBD_ | `exit_reason=`_<value>_; pid exited `<code>` |
| M2 | All atom-specific paths exercised (baseline + EXPLORE + KERNEL + FRAMEWORK_PR + roofline) | _TBD_ | per-path evidence — phase log + per-action `runs/<kind>/<task>/manifest.json` |
| M3 | No `framework_atom_action_unsupported` denials (Phase 2/3 lifts hold) | _TBD_ | `grep -c "framework_atom_action_unsupported" <session_log>` |
| M4 | No legacy `extra_sglang_args` literals in writer paths during live traffic | _TBD_ | `grep -c "extra_sglang_args" <session_log>` (deprecation warnings indicate the compat path was hit by a writer that Phase 4 missed) |
| M5 | Final report produced under `<session_dir>/reports/` | _TBD_ | `ls <session_dir>/reports/` |

### Pre-launch must-have evidence (sandbox)

Items below were verifiable WITHOUT a live launch and are flagged
PASS now so the live operator can focus on the remaining live-only
gates above:

| ID | Criterion | Pass? | Evidence |
|---|---|---|---|
| M0-a | `inference_optimizer.cli optimize --help` renders end-to-end | PASS (after fix-up #1) | argparse `%` escape; see `post_session_log.md` fix-up #1 |
| M0-b | `--framework` accepts `atom` | PASS | `--framework {sglang,vllm,atom}` |
| M0-c | atom auto-tighten preserves kernel/framework/roofline; only `--nodes>=2` fails fast | PASS | `_apply_atom_auto_tighten` dry-run shows `auto_disabled=[]` |
| M0-d | atom + aiter importable in the active venv | PASS | `python -c "import atom, aiter"` → `ok` |
| M0-e | Magpie `atom_mi355x.sh` carries `--torch-profiler-dir` bridge | PASS | line 85 of the script |
| M0-f | Magpie `benchmark_images.yaml` maps atom → `rocm/atom:latest` for both gfx942 and gfx950 | PASS | Phase 5.1 |

---

## Nice-to-have criteria

| ID | Criterion | Status | Notes |
|---|---|---|---|
| N1 | `cumulative_gain_validated ≥ 10%` | _TBD_ | live observation only |
| N2 | ≥ 1 KERNEL_OPT KEEP on an atom file under `/app/ATOM/atom/` or `/app/aiter-test/aiter/` | _TBD_ | proves Phase 2's source-path allowlist works |
| N3 | ≥ 1 FRAMEWORK_PR candidate reached Critic verdict | _TBD_ | proves Phase 3's `ROCm/ATOM` repo-map entry resolves PRs |
| N4 | ≥ 1 robustness signal fired and was handled | _TBD_ | robustness-agent CLI tick |
| N5 | `session_steward_specialist` ran ≥ 1 time (IR-7 plateau) | _TBD_ | search session log for `session_steward_specialist` dispatch |
| N6 | atom default-grid (Phase 6.2) emitted ≥ 5 variants and ≥ 1 KEPT | _TBD_ | tabulate per `post_session_log.md` Phase 6 section |
| N7 | atom specialist hints (Phase 6.1) cited in proposals' `source_evidence` | _TBD_ | qualitative: did specialists open `atom/entrypoints/` / `atom/model_engine/` paths? |

---

## Follow-up issues

To be filed by the live operator. Template:

```text
1. <title> — <one-line summary> (#issue-link)
   * Component: <kernel-agent / framework-agent / explore / robustness / ...>
   * Severity: <blocker / major / minor>
   * Surfaced in: <session_id>
```

Sandbox-side follow-ups raised during preflight (already addressed
in the Phase 7 commit per user requirement #5, no GH issue needed):

* ~~`--help` ValueError on `--enable-roofline`~~ — fixed in this
  commit (see `post_session_log.md` fix-up #1).
* ~~Stale `--framework` help text claiming atom auto-disables
  kernel/framework~~ — fixed in this commit (see fix-up #2).

---

## Plan-level status

| Phase | Status | Notes |
|---|---|---|
| 1 (profile_atom) | code-verified ✓; live-verify _TBD_ | `profile_atom.yaml` present; Magpie `--torch-profiler-dir` bridge present |
| 2 (kernel-agent on atom) | code-verified ✓; live-verify _TBD_ | source-roots probe, PolicyGate allowlist, server-flag pre-flight probe all in place |
| 3 (framework-agent on atom) | code-verified ✓; live-verify _TBD_ | `ROCm/ATOM` repo-map entry present |
| 4 (rename `extra_sglang_args` → `extra_server_args`) | code-verified ✓; live deprecation-warning sweep _TBD_ | compat helper + static guard committed |
| 5 (Magpie image registry + multi-node-only-list comments) | code-verified ✓ (Magpie repo); live image-pull _TBD_ | `rocm/atom:latest` for both arches |
| 6 (UX polish) | code-verified ✓; live grid-coverage _TBD_ | specialist hints + atom default grid committed; `--mark-trace` deferred per 6.3 outcome |
| 7 (live verify) | **DEFERRED — sandbox cannot launch live** | this report |

**Atom full-support PR series:** code commits C1–C6 land; **C7
(live-verify) is the only commit waiting on a real GPU session**.
Once the live operator completes the run and replaces the TBD
placeholders in `post_session_log.md`, the verdict here flips to
GREEN or RED. The PR series can merge with the live-verify
acceptance report attached as either:

* a separate PR (this commit + an empty placeholder follow-up
  commit that replaces TBDs post-run), OR
* a held-back final commit on the same branch (the operator
  re-amends this file in place before merge).

The current author's recommendation is the first option: land the
code commits now; replace TBDs in a follow-up commit after the live
run. This decouples the code-review timeline from the live-test
scheduling.
