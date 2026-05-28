# Atom full-support live verification — session log

**Session date:** _TBD — pending live-run on 8×MI355X box_
**Session ID:** _TBD — `Qwen-Qwen3-32B/<UTC_ts>`_
**Session dir:** _TBD — `/workspace/hyperloom-KB/Qwen-Qwen3-32B/<UTC_ts>/`_
**Launch log:** _TBD — `/workspace/hyperloom-KB/Qwen-Qwen3-32B-launch-<ts>.log`_
**Final state:** _TBD — `<session_dir>/state.json`_
**Status:** **DEFERRED** — the implementing sandbox has 4 visible
ROCm GPUs (not the planned 8), no LLM gateway credentials
(`SAFE_API_KEY` / `OPENAI_BASE_URL` unset), and `install.sh` has not
been run in this shell (`KERNEL_AGENT_ENV` unset). The 12-hour live
test described in 7.2 is therefore not executable from here; the
sub-section's deliverable is filed as a skeleton + pre-run preflight
evidence for the operator who will run the actual session.

---

## Pre-run preflight evidence (sandbox, 2026-05-28)

The portion of [`7.1_preflight.md`](7.1_preflight.md) that does NOT
require live GPU launch was exercised in the implementing sandbox.
Live-only steps (H1–H3, S1–S2, launch in 7.2) are marked DEFERRED.

### Hardware (H1–H3) — DEFERRED

`rocm-smi` reports 4 GPUs in this sandbox (planned target is 8); no
real launch was attempted. The 4-GPU readout DOES confirm:

* arch family is correct (`gfx950`-family idle GPUs visible),
* `rocm-smi --showmemuse` reports VRAM% = 0 on every device,
* `rocm-smi --showpids` shows two `UNKNOWN` PIDs with 0 VRAM (zombie
  KFD records per 7.1.H2 — would be ignored at launch),

so the H1–H3 instrumentation works as designed; only the 4-vs-8 GPU
count and absence of `--tp 4` co-tenant policy prevent an actual
launch from this box.

### Install state (I1–I5)

| Step | Status | Evidence |
|---|---|---|
| I1 `KERNEL_AGENT_ENV` sourced | DEFERRED | `KERNEL_AGENT_ENV` is unset in the sandbox; `install.sh` was not re-run in this shell. Operator must source `${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}` before launch. |
| I2 `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` includes atom | DEFERRED | Env var unset because I1 was not run. Phase 2.1 ships the source-root probe; verify post-install. |
| I3 atom + aiter importable | **PASS** | `python -c "import atom, aiter; print('ok')"` → `ok atom=/app/ATOM/atom/__init__.py aiter=/app/aiter-test/aiter/__init__.py`. Phase 2's enablement points the operator at these paths. |
| I4 Magpie atom_mi355x.sh carries `--torch-profiler-dir` | **PASS** | `grep -n "PROFILER_ARGS+=(--torch-profiler-dir" /hyperloom/atom_support/Magpie/Magpie/scripts/benchmark/atom_mi355x.sh` matches at line 85. Phase 1's IR-8 patch is in place. |
| I5 `benchmark_images.yaml` atom = `rocm/atom:latest` | **PASS** | Phase 5.1 confirmed: both `gfx942` and `gfx950` resolve to `rocm/atom:latest`. |

### Session-root preparation (S1–S2) — DEFERRED

`/workspace/hyperloom-KB` is writable on the sandbox box, but no
previous Qwen-Qwen3-32B run exists to clean up — the previous-run
cleanup branch (S2) is not exercised. Operator should follow S2's
archived-rename pattern when a prior run is found.

### Smoke-level sanity (P1–P2)

| Step | Status | Evidence |
|---|---|---|
| P1 `--framework` choices include `atom` | **PASS** (after fix) | `python -m inference_optimizer.cli optimize --help` now reports `--framework {sglang,vllm,atom}` after a sandbox fix-up (see "Fix-up #1" below). |
| P2 atom auto-tighten preserves kernel/framework/roofline | **PASS** | Dry-run of `cli._apply_atom_auto_tighten(args=Namespace(no_kernel=False, no_framework=False, enable_roofline=True, nodes=1))`: stdout reports `framework=atom: no auto-disable applied (kernel-agent + framework-agent + profile / roofline / TraceLens all wired for atom); --nodes>=2 guard active`. `auto_disabled=[]`. Both Phase 2.4 and Phase 3.2 lifts hold. |

---

## Timeline (chronological) — _TBD on live run_

```text
T+0:00 — Launch
T+0:05 — PRELUDE entered
T+0:20 — Baseline succeeded
T+0:30 — Roofline / profile completed
T+0:35 — EXPLORE entered (first specialist round)
...
T+H:MM — exit_reason / final state
```

Replace placeholder timeline above with chronological observations
captured by the operator running the live 12-hour session.

---

## Deviations + fix-ups

### Fix-up #1 — argparse `%` format crash in `--enable-roofline` help text

* **When:** 2026-05-28, during sandbox preflight P1
* **Symptom:** `python -m inference_optimizer.cli optimize --help`
  crashed with `ValueError: unsupported format character 'w'
  (0x77) at index 93` raised by argparse's `_expand_help`. Stack
  pointed at `_format_action -> _expand_help -> "% params"`.
* **Root cause:** `cli.py:4548` — the `--enable-roofline` help
  string contained the literal substring `+10% watermark`. argparse
  funnels every help string through `% params` formatting at render
  time; a bare `%` is interpreted as a format specifier (`%w` is
  unsupported, hence the cryptic error).
* **Fix:** Escape the literal `%` as `%%` in the help text
  (`+10%% watermark`). One-line change in `inference_optimizer/cli.py`.
* **Outcome:** `--help` now renders end-to-end; preflight P1 passes.
  Per user requirement #5 the fix landed in-place alongside this
  log entry rather than being deferred to a follow-up issue.

### Fix-up #2 — Stale `--framework` help string mentioning Phase-2/3 auto-disables

* **When:** 2026-05-28, while reading the now-renderable help text
* **Symptom:** `--framework` help text claims atom has "no profiler /
  framework-source-patcher integration" and that "B3 auto-tightens
  incompatible phases off when atom is selected". Both clauses are
  stale: Phase 1 wired Magpie's `--torch-profiler-dir` bridge, Phase
  2 opened kernel-agent (`/app/ATOM/atom/` in PolicyGate allowlist),
  Phase 3 opened framework-agent (`ROCm/ATOM` repo-map entry), and
  `_apply_atom_auto_tighten` no longer flips any phase off — it only
  fails fast on `--nodes>=2`.
* **Root cause:** Stale CLI help string not updated by Phases 1–3.
* **Fix:** Rewrote the help paragraph to reflect the current state:
  atom is single-node-only (`--nodes>=2` fails fast); profile /
  roofline, kernel-agent, and framework-agent are all enabled. One
  edit in `inference_optimizer/cli.py`.
* **Outcome:** `--help` text now matches `_apply_atom_auto_tighten`'s
  actual behaviour.

### Fix-up template (for the live run)

```text
### Deviation #N — <one-line summary>
- **When:** T+H:MM
- **Symptom:** <log excerpt>
- **Root cause:** <analysis>
- **Fix:** <what was done; or "reported to user, no fix possible at runtime">
- **Outcome:** <session continued / had to restart / etc.>
```

---

## Per-phase observations — _TBD on live run_

### Phase 1 (profile_atom)

_TBD_ — operator verifies `profile_atom.yaml` is selected for
profile / roofline tasks (look for `framework: atom` rendering in
the materialised YAML inside `runs/profile/<task>/`).

### Phase 2 (kernel-agent on atom)

_TBD_ — operator verifies kernel-agent dispatches a kernel-opt task
on at least one EXPLORE-surfaced gap, with `source_file` resolving
under `/app/ATOM/atom/` or `/app/aiter-test/aiter/` (Phase 2.1's
source-root allowlist).

### Phase 3 (framework-agent on atom)

_TBD_ — operator verifies the FRAMEWORK_PR phase fires (look for
`phase=FRAMEWORK_PR` log lines + at least one `fa phase-discover`
subprocess invocation against `https://github.com/ROCm/ATOM.git`).

### Phase 4 (extra_server_args rename)

_TBD_ — operator greps the live session log for any
`extra_sglang_args` deprecation warnings (the compat helper emits a
warning when the legacy key is read). Expectation: zero (Phase 4
moved every writer to the new name; legacy alias path should not
fire on a fresh atom session).

### Phase 5 (Magpie)

_TBD_ — operator confirms the `rocm/atom:latest` image was selected
for the baseline run (look for the image tag in the Magpie launch
log inside `runs/baseline/<task>/`).

### Phase 6 (UX polish)

_TBD_ — operator tabulates atom default-grid variant outcomes:

```text
atom_level_2:           <KEEP/REVERT/KEEP_UNSTABLE> — gain X%
atom_level_3:           <KEEP/REVERT/KEEP_UNSTABLE> — gain X%
atom_prefix_cache:      <KEEP/REVERT/KEEP_UNSTABLE> — gain X%
atom_kv_fp8:            <KEEP/REVERT/KEEP_UNSTABLE> — gain X%
atom_ep:                <KEEP/REVERT/KEEP_UNSTABLE> — gain X%
atom_dp_attn:           <KEEP/REVERT/KEEP_UNSTABLE> — gain X%
atom_mtp_3:             <KEEP/REVERT/KEEP_UNSTABLE> — gain X%
atom_mtp_1:             <KEEP/REVERT/KEEP_UNSTABLE> — gain X%
atom_cudagraph_bracket: <KEEP/REVERT/KEEP_UNSTABLE> — gain X%
```

Also note whether the atom-flavoured specialist hints (Phase 6.1)
correlate with higher-quality specialist proposals — coarse signal:
specialist proposals citing `atom/` source paths in `source_evidence`
rather than empty / sglang-flavoured citations.

---

## Final numbers — _TBD on live run_

```text
baseline_tput:                 <N>
current_best.tput:             <N>
cumulative_gain_validated:     <X%>
optimization_stack length:     <N>
exit_reason:                   <one of the SKILL-defined reasons>
total wall time:               <H:MM>
```

---

## How to compile this log from the live session

Per 7.6, the live operator should:

1. Read `<session_log>` (raw stdout/stderr from the optimisation
   process).
2. Read `<session_dir>/state.json` (final SharedState — phase, budget,
   optimization_stack, explore_search ledger).
3. Read `<session_dir>/breakdown/*` (session-breakdown collector
   outputs).
4. Read `<session_dir>/runs/baseline/<task>/` and the same per
   action-type (explore, kernel_opt, framework_pr, roofline, etc.)
   for per-task evidence.

Cross-check against the placeholders above and replace each TBD with
the actual evidence.
