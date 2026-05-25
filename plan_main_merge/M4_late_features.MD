# M4 — Late main features (post-m3-done merge plan)

> Branch: `feature/zhenggong/lossen_explore` @ `fadeef2` (post m3-done +
> cross-project alignment fixes). Target: `origin/main` (~30 commits past
> the original `MAIN_FEATURES_DROPPED.md` §7 audit window).
>
> Strategy: **ADOPT ALL** — every Category B feature bundle main shipped
> after M3 is v0.8-compatible accumulation; merge with the `MP` cherry-pick
> form (port the diff while reconciling against our v0.8 surface).
>
> 54 unmerged paths in dry-run: **30** are covered by the original
> `docs/integration/MAIN_FEATURES_DROPPED.md` §§1/5/6/7 with explicit
> resolutions. The other **24** are documented here in §§1–4.

---

## §1 Category A — v0.6 sub-pattern roll-up (3 DU files → `git rm`)

These three files snuck past the original §1 itemization because they
either renamed late (pmc_roofline → roofline composite) or sat in
v0.6-only test scaffolding that the original audit didn't enumerate
file-by-file. All three retire under exactly the same rule the original
§1 enforces — KB_design §3.4 / §3.9 v0.6 component removal.

| File | Resolution | Origin | Rationale |
|---|---|---|---|
| `inference_optimizer/actions/pmc_roofline.md` | `git rm` | main 3ade42a (rename `select_kernels` → `trace_analyze` [N1]) | `pmc_roofline` action retired by the `roofline` composite (M3 §6); the md file was deleted on this branch already |
| `inference_optimizer/tests/test_p2_3_param_executors.py` | `git rm` | main fc31475 (magpie env fix) | Tests v0.6 `params.py` executor — `params.py` itself is `git rm`'d by §1 |
| `inference_optimizer/tests/test_phase2_mission_and_validate_stack.py` | `git rm` | main c116013 (merge into roofline-v2) | Tests v0.6 `validate_stack` — `validate_stack.py` retired & inlined into `explore` per §1 / §4.4 |

Verification gate: after `git rm`, the surviving test surface (the
v0.8 equivalents) is `test_explore_executor.py` + the N-series tests +
`test_required_step_gates.py` — already on this branch.

---

## §2 Category B — main late feature work (12 files, all ADOPT MP)

main shipped four feature bundles after `m3-staging` was cut. All four
are **v0.8-compatible accumulation**: none of them depend on scoreboard,
v0.6 `backends_attempts` counters, or any retired surface. Each gets
ADOPT MP — port the diff hunk-by-hunk while reconciling against our
v0.8 names.

### §2.1 B-1 breakdown v1.1 (4 files)

Main PRs: `bbb143d` + `cd43641` + `a709990` + `9d62e8e` + `0e9e78a` /
`0e31054`.

Adds four new top-level sections to `session_breakdown.json`:

1. `roofline_v2` — copies the roofline_snapshot output (already
   structurally present on this branch as `roofline_snapshot.py`).
2. `data_provenance` — per-collector `source_path` + sha lineage so
   the breakdown reader can audit every numerical claim.
3. `session_image_timing` — install / startup / warm-start wall-clock
   breakdown.
4. `kdp_backend_speedup` — Kernel Decision Path: which `kernel_opt`
   ladder rungs beat baseline by how much, grouped by backend.

| File | Action |
|---|---|
| [inference_optimizer/breakdown/collectors.py](inference_optimizer/breakdown/collectors.py) | Merge main's new collectors (roofline / provenance / kdp) on top of ours; keep our v0.8 keys |
| [inference_optimizer/breakdown/exporter.py](inference_optimizer/breakdown/exporter.py) | Adopt main's 7-hunk diff; reconcile JSON schema keys against our `session_breakdown.json` writer |
| [inference_optimizer/breakdown/schema.py](inference_optimizer/breakdown/schema.py) | Add the four new section schemas |
| [inference_optimizer/tests/test_breakdown_smoke.py](inference_optimizer/tests/test_breakdown_smoke.py) | Adopt main's expanded assertions on the four new sections |

**Risk**: medium — main's writer assumes some collectors that we may
not have on this branch (e.g. a `roofline_v2` collector that pulls
from `roofline_snapshot.py`). If we hit a missing-collector reference,
write a thin shim that returns `None` for sections we don't yet
populate, rather than blocking the section. The smoke test must remain
green either way.

### §2.2 B-2 profile post-trace validator (1 file + tests)

Main commit `7e3c8e9` — full Deval `check_torch_trace.py` parity. Runs
after profile.py captures the trace, validates structural invariants
(kernel ops well-formed, no truncated chunks, expected event types
present) and surfaces the result as a non-fatal warning that can be
escalated to N36's chunk-quality gate.

| File | Action |
|---|---|
| [inference_optimizer/orchestrator/action_executors/profile.py](inference_optimizer/orchestrator/action_executors/profile.py) (7 hunks, ~319 diff lines) | ADOPT MP — add validator block, **route its warnings through N36's existing chunk-quality gate** rather than introducing a parallel gate |
| `inference_optimizer/tests/test_profile_and_kernel_handlers.py` profile half | Adopt main's validator test cases |

**Risk**: medium — N36's chunk-quality gate already exists on this
branch. Merging needs to make sure we don't run two gates with
contradicting verdicts. Resolution: validator emits structured warnings;
N36 gate consumes them.

### §2.3 B-3 kernel-opt batch bundle (1 file + tests)

A long tail of kernel-opt improvements main shipped:

| Commit | Effect |
|---|---|
| `e9415ad` | PR-I: align `_batch_kernel_candidates` `min_gpu_pct` default with SharedState gate |
| `1252b36` | Default batch fanout 3 → 8 + auto-inject `candidates_path` |
| `7bba999` | Filter rejected / exhausted / in-flight kernels from batch + group fallback |
| `90b187a` | Retire kernel after 1 failed ladder + `report-gate` helper |
| `75c2b73` | Isolate batch sub-task exceptions from `asyncio.gather` |
| `0270b67` | Deny `report` while untried hot reusable kernels remain |
| `c900791` | Yield `hot_kernel_unfinished` gate to N19c (no death-spiral) |
| `2e06fc8` | Ladder best-selection: prefer KEEP over higher micro |
| `81f4e06` / `1010c06` | Trust GEAK correctness by default; opt-out via `env=0` |
| `ce36e70` | GEAK 90min wrapper timeout + multi-KEEP integrate queue |
| `1760ce0` (robustness) | Silence `no_levers_found` while `kernel_opt` is in flight |

| File | Action |
|---|---|
| [inference_optimizer/orchestrator/kernel_request_handlers.py](inference_optimizer/orchestrator/kernel_request_handlers.py) (4 hunks, ~154 diff lines) | ADOPT MP all 11 commits |
| `inference_optimizer/tests/test_profile_and_kernel_handlers.py` kernel half | Adopt main's batch / retire / KEEP-over-micro tests |

**Decision flag**: `81f4e06` "trust GEAK correctness by default". Our F3
correctness gate currently requires explicit verification. main's commit
default-trusts GEAK's `status=complete`. Per plan-mode pre-read this is
already mitigated by `1010c06` ("opt-in trust GEAK correctness when
status=complete") — so the production default is conditional on
`status=complete`, not unconditional trust. **Adopt as-is**.

### §2.4 B-4 robustness no_levers_found defer (4 files)

Main commits `97318ee` + `468504f` + `1760ce0`.

Defers the `no_levers_found` alert until EXPLORE phase actually
started (the alert was firing too early during PRELUDE), and bumps the
multi-node elapsed floor from 45 → 60 min (kernel_opt + multi-node
warmup is longer than the original threshold assumed).

| File | Action |
|---|---|
| [robustness-agent/src/robustness_agent/role/prompt_inputs.py](robustness-agent/src/robustness_agent/role/prompt_inputs.py) | ADOPT MP — add the explore-phase gate |
| [robustness-agent/src/robustness_agent/signals/progress.py](robustness-agent/src/robustness_agent/signals/progress.py) | ADOPT MP — add the `kernel_opt in flight` suppression |
| [robustness-agent/src/robustness_agent/runtime/cli.py](robustness-agent/src/robustness_agent/runtime/cli.py) | ADOPT MP — multi-node elapsed floor 45 → 60 min |
| [robustness-agent/tests/test_signals_progress.py](robustness-agent/tests/test_signals_progress.py) | ADOPT MP — adopt main's expanded assertions |

**Risk**: low — additive logic; the v0.8 signals on this branch already
have an explore-phase awareness, this just tightens it.

---

## §3 Category C — 8aeaa11 hotfix overlap (2 UU files, logic merge)

Our `8aeaa11` (cumulative_gain hotfix) and main's `fc31475` (magpie env
pin) both modified two of the same functions for orthogonal reasons.
The merge is mechanical: keep both diffs.

| File | Our 8aeaa11 change | main fc31475 change | Resolution |
|---|---|---|---|
| [inference_optimizer/orchestrator/action_executors/_grid_runner.py](inference_optimizer/orchestrator/action_executors/_grid_runner.py) | Add `soft_deadline_sec=` to `_run_magpie` + plumb to `run_with_session_kill` | Add `MAGPIE_INFERENCEX_PATH=$INFERENCEX_PATH` env pin in same function | Keep both — env pin runs before subprocess launch; soft_deadline_sec is a kwarg to the run helper |
| [inference_optimizer/orchestrator/action_executors/baseline.py](inference_optimizer/orchestrator/action_executors/baseline.py) | Record `subprocess_runtime_sec` on success path | Add magpie env pin to baseline subprocess too | Keep both — env pin happens at command construction; runtime_sec is captured post-run |

---

## §4 Category D — housekeeping (8 files)

Mostly drift-resolution. Default policy "ours", deviation noted per file.

| File | Status | Resolution |
|---|---|---|
| `.github/workflows/manual-poke-issues.yml` | AA | Take main `0b55fc5` (skip issues with `stale` label) + our independent additions |
| `framework-agent/scripts/install.sh` | AA | Take main `26d2c64` (portability fix) + `73773fa` (`/opt/venv/bin/python` preference) + our F2 additions |
| `inference_optimizer/SKILL.md` | 1 hunk | Ours (v0.8 canonical SKILL on this branch — main `62f87fc` is a roofline-v2 merge that pre-dates m3-done) |
| `inference_optimizer/actions/_meta/report.yaml` | 1 hunk | Ours (just-committed description shortening at `fadeef2`; main `62f87fc` description has the same intent but pre-dates the catalogue-size hardening) |
| `inference_optimizer/tests/conftest.py` | 1 hunk | Case-by-case fixture drift — most likely "ours" if main only adds robustness/multi-node fixtures we already have |
| `inference_optimizer/tests/test_coordinator_audit_wiring.py` | 1 hunk | Follows `coordinator.py` §5 resolution — case-by-case |
| `inference_optimizer/tests/test_coordinator_runtime.py` | 1 hunk | Follows `coordinator.py` + `shared_state.py` §5 resolution — case-by-case |
| `inference_optimizer/tests/test_resume.py` | 1 hunk | Follows `shared_state.py` + `paths.py` §5 resolution — case-by-case |

---

## Acceptance (must hold after merge)

1. `bash docs/integration/check_preserved.sh` returns `ALL PRESERVED OK`
   (the v0.6 retirement guard from M3 must not regress).
2. `pytest inference_optimizer/tests/ -q -k 'not slow and not integration and not hang'`
   stays ≥ 1815 passed (the m4-staging baseline).
3. `pytest robustness-agent/tests` stays ≥ 651 passed.
4. The four `breakdown/` files import cleanly and `test_breakdown_smoke.py`
   passes (B-1 acceptance).
5. `test_profile_and_kernel_handlers.py` passes end-to-end (B-2 + B-3
   acceptance). Expect ~9 hunks of test changes; the unified test file
   must remain a single coherent surface.
6. No conflict markers (`<<<<<<<`) in any tracked file.

---

## Roll-back

If any B-bucket file becomes non-trivial mid-merge, `git checkout
--ours $FILE && echo "$FILE deferred to post-m4 PR per M4 §2.X" >>
docs/integration/M4_deferred.md` and continue. Final commit may note
deferred items so reviewers see them explicitly.
