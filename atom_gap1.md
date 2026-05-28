# atom_gap1.md — design-vs-code gap report

Audit date: 2026-05-28
Branch: `feature/zhenggong/atom` (`bb8a6b3` HEAD)
Reference docs: [`atom_full_support.md`](atom_full_support.md),
[`atom_plan/00_overview.md`](atom_plan/00_overview.md),
plus every phase's `00_README.md` + sub-section `.md`.

---

## TL;DR

Phases 1, 5, 6.1, 6.3 are clean PASSes. Phases 2 and 3 implement
every must-have but deviate from the plan's prescribed
intermediate state in two places (the `framework_atom_action_unsupported`
rule was deleted outright instead of shrunk to `{"framework_pr"}`,
and `KEYWORDS["atom"]` was merged into the shared technical-term
list rather than getting its own per-framework bag). Phase 4 ships
the compat helper, static guard, and SharedState load-time
migration but never finishes the **reader-site sweep** the design
called for (only `baseline.py` goes through `read_extra_server_args`).
Phase 6.2 ships `_atom_default_grid()` + `_default_grid_for_framework()`
**with full tests** but the helpers have **zero production callers**
— the EXPLORE executor still hard-fails on an empty grid and the
Coordinator never injects atom seed variants into cold-start
rounds. Two doc strings (`SKILL.md`, `cli.py --framework` blurb)
are stale relative to the post-Phase-2/3 state.

The `Out of scope` block in `00_overview.md` is correctly held
across the board (multi-node TP still blocks, no Docker build
pipeline added, `EXTRA_*_ARGS` env names retained).

---

## Gap inventory (high → low severity)

### G1 (high) — Phase 6.2 atom default grid is dead code

**Design:** `_default_grid_for_framework('atom', ...)` returns the
curated seed grid; EXPLORE cold-start uses it as a fallback when
the orchestration LLM / specialist round produces no variants. The
Coordinator or `ExploreExecutor` wires the dispatcher.

**Code state:**
* `_atom_default_grid()` and `_default_grid_for_framework()` exist
  in `inference_optimizer/orchestrator/action_executors/explore.py`
  (lines 209–321) with full test coverage in
  `inference_optimizer/tests/test_explore_executor.py`
  (lines 779–923).
* **No production caller.** `grep` for `_default_grid_for_framework(`
  shows only `test_explore_executor.py`. The helpers are not
  exported from `action_executors/__init__.py` (only
  `ExploreExecutor` / `explore_executor`).
* `ExploreExecutor.__call__` still hard-fails on an empty grid
  (`explore.py:485-497` returns
  `error_class="empty_grid"`).

**Effect:** atom EXPLORE cold-start still depends on the
orchestration LLM inventing variants from prompt prose, exactly
the failure mode Phase 6.2 was designed to remove. The
acceptance-gate language "produces at least 5 variants for a
representative atom session" is satisfied at the helper level but
the gain never reaches a live session.

**Recommended fix:**
* Add a Coordinator-side seed in `_warm_specialist_params` (or
  the explore-task warm path), OR
* In `ExploreExecutor.__call__`, when `params.grid` is empty
  AND `bench.framework == "atom"`, fall through to
  `_default_grid_for_framework("atom", model_class=..., conc=...)`
  instead of the `empty_grid` failure.
* Add a regression test that an atom-framework explore task with
  no `grid` produces a non-empty default grid (covers the
  `Wire 6.2 step 2` test the design asked for).

**Effort:** 30 min — 1 hour. Single Coordinator or executor edit
+ one integration test.

---

### G2 (high-medium) — Phase 4 reader-site migration is incomplete

**Design (`4.2_reader_site_migration.md`):** every reader that
historically called `.get("extra_sglang_args")` must now go
through `read_extra_server_args(payload)` (or the canonical key
directly, but **with** the helper applied where legacy values may
still arrive).

**Code state:**
* Compat helper present at
  `inference_optimizer/compat/payload_aliases.py` (PASS).
* **Only `inference_optimizer/orchestrator/action_executors/baseline.py`**
  (lines 46, 365, 452) imports and uses
  `read_extra_server_args`.
* Other readers (`coordinator.py:261, 4687, 7055`,
  `explore.py:157`, `kernel_request_handlers.py:1649`,
  `shared_state.py:1873, 1174`) read `.get("extra_server_args")`
  directly with no compat fallback.
* SharedState rewrites legacy keys on load
  (`_migrate_legacy_extra_sglang_args_keys` at
  `shared_state.py:936-950, 158-185`) — this catches
  `state.json` resume paths but NOT in-flight task / intent
  payloads that bypass SharedState.

**Effect:** functionally OK as long as all writers emit
`extra_server_args` (Phase 4.3 PASS). Risk is sub-agent envelopes
or external callers that still emit `extra_sglang_args` would be
silently dropped at non-baseline readers instead of emitting the
designed deprecation warning.

**Recommended fix:**
* Either run the reader sweep the plan asked for (route every
  `.get("extra_server_args")` through `read_extra_server_args`),
  OR
* Update the plan to declare load-time migration sufficient and
  document the trade-off in `atom_full_support.md`.

**Effort:** 1-2 hours for the full sweep + tests; 15 min for
documentation-only resolution.

---

### G3 (medium) — Phase 4.5 sub-agent shim coverage is uneven

**Design:** every sub-agent that reads payload fields gets a
local shim mirroring `read_extra_server_args` so subprocess-side
deprecation warnings fire consistently. Round-trip
`test_envelope_canonical_read_write` per sub-agent.

**Code state:**
* `kernel-agent/tools/_payload_aliases.py` — exists, used by
  `kernel_optimization.py:974-981` (PASS).
* `robustness-agent/src/robustness_agent/_payload_aliases.py` —
  shim exists, but `repeated_payload.py:73,79` hashes
  `params.extra_server_args` only and **doesn't use the shim**.
* `critic-agent/` and `framework-agent/` — no shim files (likely
  N/A: neither package reads payload extra-args fields; spot-check
  confirmed).
* No `test_envelope_canonical_read_write` test file exists.

**Effect:** robustness signal hashing on a legacy-keyed envelope
would silently produce a different hash from the canonical name.
Critic + framework-agent appear genuinely unaffected.

**Recommended fix:**
* Wire `robustness-agent`'s shim into `repeated_payload.py`'s
  `_extract_extra_args` (or equivalent).
* Add the round-trip envelope test prescribed by 4.5.

**Effort:** 30-45 min.

---

### G4 (medium) — Phase 6.2 stale SKILL.md text

**File:** `inference_optimizer/SKILL.md:844-847`.

**Text:**
> no separate `profile_atom.yaml` — the baseline YAML carries
> `profiler.torch_profiler.enabled` overridable via `PROFILE=1` env

**Reality (Phase 1 shipped):** `inference_optimizer/scripts/configs/profile_atom.yaml`
exists; `_default_profile_config()` returns it when
`FRAMEWORK=atom`.

**Same file lines 846-847:**
> Which framework-specific seed grid the `explore` action falls
> back to when no `params.grid` is supplied

**Reality:** explore does NOT fall back to a seed grid — it
returns `error_class="empty_grid"` (see G1). This sentence
either needs deleting OR needs G1 fixed first.

**Recommended fix:** update both lines after G1 is resolved.
Until then the SKILL operator-facing prose is misleading.

**Effort:** 5 min once G1 is decided.

---

### G5 (medium) — Phase 3.3 fa-runtime atom branches are narrower than the design

**Design (`3.3_fa_runtime_atom_branches.md`):**
* per-framework `KEYWORDS["atom"]` bag in
  `framework-agent/src/framework_agent/keywords.py`,
* `framework_optimization/atom/` KB partition with
  `path_for_framework("atom")` resolving to it,
* atom parametrisation in `test_sources_dispatch.py`,
* a `prompts.py` per the design touch-point.

**Code state:**
* Atom terms are merged into the shared `_TECHNICAL_TERMS` /
  global whitelist (`keywords.py:27-36`), not a per-framework
  bag.
* `kb.py` has atom in its domain keyword list (`kb.py:37`) but
  no `framework_optimization/atom/` partition directory exists.
* `framework-agent/tests/test_sources_dispatch.py` still defaults
  `framework: "sglang"` at line 18; no atom parametrisation.
* `framework-agent/src/framework_agent/` has no `prompts.py`
  (the file the design touch-point listed).
* `test_keywords_includes_atom` and
  `test_kb_atom_partition_path_resolves` from the design
  inventory are missing.

**Effect:** `fa phase-discover --framework atom` works end-to-end
(`test_main_candidates_accepts_atom_framework` in
`framework-agent/tests/test_cli.py:112-139` confirms), but the
keyword-bag mechanism the plan envisioned for per-framework
scoring is collapsed into the shared list. Whether that's a real
quality drop is empirical — it would surface (or not) in Phase 7.

**Recommended fix:** decide whether the per-framework keyword
bag is worth the refactor. If not, update
`3.3_fa_runtime_atom_branches.md` to note the simpler shared-list
approach and remove the missing-test acceptance items.

**Effort:** 1-2 hours if you want the per-framework bag; 10 min
to update the plan if you don't.

---

### G6 (medium-low) — Phase 2/3 G2 guard scope drift

**Design (`00_overview.md` test surface):**
> `framework_atom_action_unsupported` rule's frozenset is empty
> (or the rule is removed entirely) by end of Phase 3.

**Code state:** rule fully removed (PASS for behaviour), BUT
`policy.py:1380-1394` still has a historical-context comment
mentioning the rule name verbatim, and the existing static guards
only forbid `rule="framework_atom_action_unsupported"` /
`_ATOM_UNSUPPORTED_ACTIONS:` definitions — not bare comments.

**Effect:** zero. The behaviour is correct; the comment is intentional
provenance for future archaeologists.

**Recommended fix:** either tighten the guard (forbid the literal
string everywhere except an explicit `# historical:` allowlisted
line) OR document this comment as deliberate and close.

**Effort:** 5 min — documentation-only.

---

### G7 (low) — Phase 5.3 pr.md checklist is stale

**File:** `/hyperloom/atom_support/Magpie/pr.md:163-164`.

**Stale text:**
> ImageSelector returns "vLLM-inherited image" for `atom/gfx942`
> and `atom/gfx950`

**Reality (Phase 5.1 shipped):** both arches resolve to
`rocm/atom:latest`. The body text earlier in the same file
(`pr.md:109-124`) reflects the post-5.1 state correctly; the
checklist row was not updated alongside it.

**Recommended fix:** flip the two checklist lines to
`rocm/atom:latest`.

**Effort:** 2 min (single Magpie commit, no behaviour impact).

---

### G8 (low) — Phase 2/3 test name + missing-test drift

**Design test inventory (in 2.6, 3.5):**

| Prescribed test | Status |
|---|---|
| `test_policy_atom_action_gate.py` | renamed → `test_policy_atom_invariants.py` (intentional) |
| `test_atom_unsupported_actions_set_contains_only_framework_pr` | replaced by removal guards (G6) |
| `test_kernel_request_handlers_and_tracelens_analysis_atom_paths_in_sync` | PARTIAL — file-read presence check, not subset-equality assertion |
| `test_kb_atom_partition_path_resolves` | MISSING (covered by G5) |
| `test_keywords_includes_atom` | MISSING (existence covered by `test_extract_atom_*`) |
| `test_runtime_cli_accepts_atom_framework` | PARTIAL (`test_main_candidates_accepts_atom_framework` covers the actual end-to-end) |
| `test_sources_dispatch` atom parametrisation | MISSING (covered by G5) |
| kernel-agent-side pytest for atom `_REUSABLE_SOURCE_ROOTS` | MISSING — design allowed if imports exist; not added |

**Effect:** all behavioural assertions covered; the names just
don't match. No functional gap.

**Recommended fix:** update the plan docs OR rename the tests
for traceability. Plan-update is cheaper.

**Effort:** 15 min.

---

### G9 (low) — CLI `--framework atom` blurb auto-tighten cross-reference

**File:** `inference_optimizer/cli.py` (after Phase 7 fix-up #2,
the help string says the auto-tighten "only enforces ``--nodes
1``"). Help text now correct.

**Remaining slightly-stale prose:** the long comment block in
`_apply_atom_auto_tighten` (`cli.py:394-432`) is accurate
post-Phase-3, but the **comment** that frames the function as
auto-tighten ("After Phase 3 of atom_plan/ … function's only
remaining responsibility is the multi-node fail-fast guard")
keeps the historical "tighten" name. Renaming
`_apply_atom_auto_tighten` → `_assert_atom_single_node` would
match the post-Phase-3 reality.

**Recommended fix:** optional rename; would require touching
the call site at `cli.py:3462` and the two test files
(`test_cli_atom_auto_tighten.py`,
`test_kernel_request_handlers_units.py:167`). Defer unless
someone is doing the live-verify follow-up.

**Effort:** 20 min for the rename; 0 min to leave as-is.

---

## Out-of-scope items — all correctly deferred

| Non-goal (`00_overview.md` §252–275) | Confirmed deferred |
|---|---|
| atom multi-node TP wiring | `cli.py:433-440` — `sys.exit(2)` on `--nodes >= 2` |
| atom Docker image build pipeline | No Dockerfile/build in Hyperloom |
| `--backend vllm` bench client refactor | Not touched |
| Rename `EXTRA_SGLANG_ARGS` / `EXTRA_VLLM_ARGS` / `EXTRA_ATOM_ARGS` env names | Old names retained throughout |
| atom-specific GEAK / OOB / Cursor recalibration | Deferred to live-test signal |
| TraceLens `atom_*` patch set | 6.3 investigated; deferred per outcome |

---

## Phase 7 (live verify) — DEFERRED, not a gap

Documented in `atom_plan/phase7_live_verification/acceptance_report.md`.
Sandbox cannot run the 12-hour session (4 vs 8 GPUs, no LLM
gateway credentials, install.sh not sourced in this shell). Not a
code gap; pending live operator.

---

## Suggested commit cadence for closing the gaps

| Commit | Gap | Subject |
|---|---|---|
| F1 | G1 | `feat(atom): wire _default_grid_for_framework into ExploreExecutor cold-start` |
| F2 | G2 | `refactor(payload-aliases): route remaining readers through read_extra_server_args` |
| F3 | G3 | `fix(robustness-agent): apply payload-alias shim in repeated_payload` |
| F4 | G4 | `docs(skill): refresh atom profile / explore seed grid wording post-phase-6` |
| F5 | G5 | `feat(framework-agent): per-framework KEYWORDS["atom"] bag (optional)` |
| F6 | G7 | `chore(magpie): pr.md image-selector checklist reflects rocm/atom:latest` (Magpie repo) |
| F7 | G6/G8/G9 | `docs(atom): close out plan-vs-code naming drift` (atom_plan/) |

F1–F3 are functional; F4–F7 are docs/test-naming hygiene. F1 is
the only one that affects live behaviour — everything else is
guard / clarity work.

---

## What is NOT a gap

* All five static regression guards from `00_overview.md` lines
  240–248 hold (writers don't emit legacy keys outside the
  allowlist; no `atom_no_profiler` short-circuit;
  `framework_atom_action_unsupported` is gone; atom source paths
  in both root lists; `_FRAMEWORK_TO_REPO_URL` dicts byte-for-byte
  identical).
* `--framework atom` accepts via CLI (Phase 7 fix-up #1 + #2).
* atom + aiter importable in the active venv.
* Magpie `atom_mi355x.sh` carries the `--torch-profiler-dir`
  bridge.
* `_apply_atom_auto_tighten` only enforces `--nodes 1`.
* All sub-section tests pass locally (Phase 6 sweep: 21 + 14 +
  56 + 81 cases green; pre-existing `test_run_grid_salvages_*`
  and `cachetools`-import failures are environment, not code).

---

## How to use this report

1. Land F1 in a single PR before Phase 7 live-verify runs —
   that's the only material functional difference between the
   design and the code right now.
2. Decide G2/G3/G5 on a cost/benefit basis. Functional impact is
   low; cleanup value depends on whether sub-agent envelopes are
   expected to carry legacy keys in the wild.
3. The remaining items (G4/G6/G7/G8/G9) are doc/naming hygiene
   and can land as a single sweep commit before the atom
   full-support PR series closes.

If only one gap could be closed before live verification, close
**G1**. Everything else is correctness-preserving cleanup.
