# Atom full-support — TL;DR

> Detailed implementation plan lives under [`atom_plan/`](atom_plan/00_overview.md).
> This file is the executive summary; for any decision, design, or
> touch-point detail, follow the link.

---

## What this is

Hyperloom historically auto-disabled kernel-agent, framework-agent, and
the analysis lane (profile / roofline / TraceLens) under
`--framework atom`. The previous round of patches unlocked the analysis
lane after Magpie's `atom_mi*x.sh` learned to bridge `PROFILE=1` to
atom's `--torch-profiler-dir`. This plan finishes the job: kernel-agent
and framework-agent also light up on atom.

After the plan lands, `--framework atom` runs every phase
`--framework sglang` and `--framework vllm` do, except multi-node
(atom upstream has no multi-node TP wiring).

## Decision summary

| Decision | Choice |
|---|---|
| Multi-node atom | Keep `--nodes>=2` fail-fast guard |
| Kernel-agent on atom | Full enablement (atom source + aiter) |
| Framework-agent on atom | Full enablement (scout `https://github.com/ROCm/ATOM`) |
| Field rename | `extra_sglang_args` → `extra_server_args` (one-release alias) |
| Live verification | Qwen3-32B FP8 TP4 MI355X, `--max-hours 12` |
| Magpie commit | Modify + commit locally; upstream PR by author |

## Phase map (one folder per phase under `atom_plan/`)

| Phase | What | Effort |
|---|---|---|
| 1 | Fix the IR-8 regression (missing `profile_atom.yaml`) | 1–2 h |
| 2 | Open kernel-agent for atom | 4–6 h |
| 3 | Open framework-agent for atom | 3–4 h |
| 4 | Rename `extra_sglang_args` → `extra_server_args` (~46 files) | 6–10 h |
| 5 | Magpie image registry + comment updates | 1–2 h |
| 6 | UX polish (specialist hints, atom seed grid, `--mark-trace`) | 3–5 h |
| 7 | Live verification on real GPU session | ~14 h wall |

## Commit cadence

Nine English commits total (C1–C9), one per phase except Phase 2 and 3
which each split a code commit from a test commit, and Phase 4 which
stays atomic. Phase 5's commit lives in the Magpie repo. Full table
in [`atom_plan/00_overview.md`](atom_plan/00_overview.md#commit-policy-for-the-implementer-not-for-the-plan-author).

## Live verification

**Live-verified:** _DEFERRED — sandbox-side preflight passed
2026-05-28; the 12-hour Qwen-Qwen3-32B FP8 TP=4 MI355X session
described in [`atom_plan/phase7_live_verification/7.2_launch.md`](atom_plan/phase7_live_verification/7.2_launch.md)
is pending a real 8×MI355X box with LLM-gateway credentials. See
[`atom_plan/phase7_live_verification/acceptance_report.md`](atom_plan/phase7_live_verification/acceptance_report.md)
for the must-have / nice-to-have criteria table and
[`atom_plan/phase7_live_verification/post_session_log.md`](atom_plan/phase7_live_verification/post_session_log.md)
for the structured log skeleton (and the two sandbox-side fix-ups
that landed in the Phase 7 commit: `--enable-roofline` argparse `%`
escape and the stale `--framework` help string)._

## Out of scope

* atom multi-node TP wiring (atom upstream lacks it)
* atom-specific Docker image build pipeline
* Bench client refactor away from `--backend vllm`
* Renaming the per-framework env names `EXTRA_*_ARGS`
* atom-specific GEAK / OOB / Cursor backend recalibration
* TraceLens `atom_*` patch set authoring

## Read next

* [`atom_plan/00_overview.md`](atom_plan/00_overview.md) — full plan
  index, risk register, cross-cutting test surface
* [`atom_plan/phase1_fix_ir8_regression/00_README.md`](atom_plan/phase1_fix_ir8_regression/00_README.md)
  — smallest, most concrete phase; good starting point
