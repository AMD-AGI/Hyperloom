# `power_management` action playbook

## Purpose

Sweep host-level GPU power knobs via `rocm-smi` on top of the plateaued
kernel config, rebench each variant, and KEEP the variant that
maximises **throughput** (power is a free variable — no perf/W).

**Throughput-only, fan + cap maxed at all times.** The power cap and
fan are pinned to MAX during BOTH the kernel climb AND the settle
sweep, so the kernel phase and the sweep's `auto_baseline` row are the
*same hardware state* and power is never a confound for the
KEEP/REVERT/plateau signal. Power tuning is the *single* knob class
tuned once, at the KERNEL plateau — NOT during the greedy kernel climb
(a varying mid-climb power state would confound the irreversible
hill-climb, and only the final combo ships anyway).

**GFX is tuned via `--setperfdeterminism` only.** Index pinning
(`--setsclk`) is removed for GFX: the DPM table is unreliable and coarse in general, so only
determinism gives usable resolution. The top engine clock comes from
`--showsclkrange` (the clean DVFS range), not the scrambled index
table.

The Coordinator holds a fixed **incumbent** state (`--setperflevel
auto` + ceiling power cap + `--setfan 100%`) through the climb and runs
one **roofline-routed settle sweep** at the plateau. The sweep's
`auto_baseline` row reproduces exactly that incumbent state, and the
winner is the highest-median-throughput variant that clears the noise
floor over `auto_baseline` (else auto is kept). See "The settle sweep"
below.

## `rocm-smi` command reference

Every shell command this action issues comes from the table below.
The executor wraps each setter in `--autorespond yes` (rocm-smi
otherwise prompts on destructive changes), prefixes `sudo ` when
`geteuid() != 0`, and appends `-d <id>` once per entry in
`devices` (empty `devices` = all GPUs). The probes are always
`--json` so we parse a stable structure rather than the
human-readable tabular output.

### Probes (read-only)

| Command | Purpose | Consumed by |
|---|---|---|
| `rocm-smi --showmaxpower --json` | Per-GPU manufacturer ceiling in watts | Incumbent climb cap + settle cap-max on every row; `dry_run` skips it |
| `rocm-smi --showperflevel --json` | Current perflevel per GPU | Lazy revalidation drift check |
| `rocm-smi --showsclkrange` | Clean DVFS engine-clock range (e.g. `500Mhz - 2400Mhz`) | **Primary** top-sclk source for the determinism ladder (100/95/90% of it) |
| `rocm-smi --showclkfrq --json` | DPM table; per-clock `sclk[N]` / `mclk[N]` levels | Top-sclk **fallback**; the GFX-high sclk pin on memory rows; the selectable `mclk[N]` levels for the memory-axis capability gate |
| `rocm-smi --showmclkrange` | Memory-clock range (informational) | Memory-axis capability probe (range readout) |
| `rocm-smi -a --json` | Full state snapshot | `rocm_smi_state_before/after` per-variant audit field |

The top engine clock is taken from `--showsclkrange` (the clean
range), NOT the `--showclkfrq` index table, because the DPM table on
this silicon is coarse and scrambled (`0:500`, `1:158`, `2:2400` MHz —
index order ≠ frequency order). The `--showclkfrq` max is kept only as
a backstop for hosts whose `--showsclkrange` is unavailable.

The upstream `rocm-smi` Python CLI does NOT expose a per-GPU
*hardware-minimum* powercap reading: `--showmaxpower` only reports
the ceiling, and `--showpower` reports the live workload draw rather
than the cap setpoint. The default-grid synthesis therefore uses the
operator/default soft floor (`power_cap_floor_w`, default 150 W) for
the lower bound of its span; the floor-lift logic in
`_resolve_variants` is dormant under the current CLI but stays live
so a future CLI revision (or an `amd-smi` re-target) that does
expose the hardware minimum would pick it up automatically.

### Setters (mutate GPU state — require sudo or root)

| Command | Effect | Notes |
|---|---|---|
| `rocm-smi --setpoweroverdrive WATTS` | Hard cap on board power | Upstream rocm-smi flag (the `--setpowercap` alias some docs reference is not exposed by the binary at `/opt/rocm/bin/rocm-smi`). Value must be at or below the probed manufacturer ceiling per GPU |
| `rocm-smi --setperflevel LEVEL` | DVFS mode | `LEVEL` ∈ `{auto, high, manual, profile_standard, profile_peak, profile_compute}` — `low` / `profile_min_*` are rejected by `_build_variant_from_payload` |
| `rocm-smi --setsclk N` | Pin SCLK to DPM index `N` | **Requires** `--setperflevel manual` first. **Not used as a GFX-tuning lever** (the index table is coarse/scrambled); retained only to pin GFX to the **highest-frequency** index on memory-axis rows so GFX stays high while mclk steps. The pin index is chosen by frequency (`_gfx_high_sclk_idx`), NOT by the largest index, precisely because the scrambled table can put a low clock at the top index |
| `rocm-smi --setmclk N` | Pin MCLK to DPM index `N` | Same `manual`-perflevel gate; the memory-axis lever |
| `rocm-smi --setpcie N` | Pin PCIe DPM to index `N` | Same `manual`-perflevel gate |
| `rocm-smi --setperfdeterminism MHZ` | Lock SCLK to a deterministic frequency | Takes a frequency (MHz), not a DPM index — the **sole** GFX-tuning lever. The MHz targets come from `--showsclkrange` (the clean top sclk) × the ladder pcts |
| `rocm-smi --setfan PCT%` | Fan duty cycle | Range 0–100; rejected outside that band |

### Resets (always issued between variants and on shutdown)

The executor runs these unconditionally in `_reset_defaults()` —
order matters because `--resetperfdeterminism` must precede
`--resetclocks` (perfdeterminism implicitly locks sclk):

```bash
rocm-smi --resetperfdeterminism --autorespond yes
rocm-smi --resetclocks            --autorespond yes
rocm-smi --resetpoweroverdrive    --autorespond yes
rocm-smi --resetfans              --autorespond yes
```

### Flag-presence detection

The `rocm-smi` Python CLI version is independent of both the ROCm
release and the `rocm_smi_lib` C library version, so support is
detected by **flag presence** rather than by parsing a version
string. The runtime probe (`_probe_powercap_range`) invokes
`rocm-smi --showmaxpower --json` and treats a non-zero exit code
or a JSON without a `Max ... Power` field as "this host needs the
LLM to supply an explicit `params.grid`". Hosts whose binary lacks
`--showmaxpower` / `--setpoweroverdrive` fail cleanly on the rung-2
probe (and on the executor itself, with `error_class="empty_grid"`
or `error_class="rocm_smi_set_failed"`) — bring those hosts forward
or migrate them to `amd-smi` (a separate integration target) rather
than expecting compatibility with further legacy command syntax.
See `scripts/probe_power_management_capability.py` for an
end-to-end flag-presence audit of the same command surface.

## Why `rocm-smi`, not a server flag

Server-flag knobs (`explore`, and workload shape via `sweep`) live
inside SGLang / vLLM and travel through `extra_server_args` /
`extra_envs`, so a winning variant is captured verbatim on
`current_best.extra_server_args` and reproduced by every subsequent
server launch. Power state is different — it is a property of the *host
GPU* set by an external root-privileged tool that the server inherits.
After a `KEEP` the executor deliberately **leaves the winning power
state applied** at the end of the action, but does NOT write the
setting onto `current_best.extra_server_args`. Re-applying the chosen
state across a server restart or a long pause is the operator's
responsibility (and the final report names the chosen variant).

## When the Coordinator surfaces it

**Catalogue:** `pipeline_phase: explore` groups the action with other
shallow search arms in the prompt catalogue. That label is *not* the
Coordinator runtime phase.

**Runtime phase:** `power_management` runs during **`PHASE_KERNEL_AGENT`**
(see `phase_state.PHASE_ALLOWED_ACTIONS`). The Coordinator drives power
in exactly two deterministic steps (the LLM may also delegate an
explicit-grid PM round during KERNEL):

* **Fixed incumbent climb.** At KERNEL entry (`_on_enter_kernel` →
  `_apply_kernel_climb_max_state` → `apply_max_climb_state`) the
  Coordinator applies the incumbent stack once (single-node):
  `--setperflevel auto` + the `--showmaxpower` ceiling cap
  (`--setpoweroverdrive`) + `--setfan 100%`, and records it on
  `SharedState.host_state_applied`. This state is **held unchanged for
  the entire climb** — nothing re-tunes it mid-climb. The cap + fan
  stay maxed for the whole run (climb AND sweep) so power never
  confounds the KEEP/REVERT/plateau signal; GFX is left on the `auto`
  governor here because the settle sweep tunes GFX only via
  `--setperfdeterminism`, and its `auto_baseline` row must reproduce
  exactly this state.
* **Settle sweep.** Once at the KERNEL plateau (`exit_normal_kernel`
  returns `plateau_kernel`), before KERNEL → SWEEP
  (`_maybe_hold_kernel_for_power_sweep`). This is the single power tune
  of the run: the roofline-routed grid on the plateaued combo (see "The
  settle sweep" below), iterating on the run's tuned state via the
  `power_management_search` ledger. The Coordinator passes the roofline
  `bound_kind` (read at settle time) to route the grid (prune the
  det_95/det_90 rungs when compute-bound; skip the memory axis when
  memory-bound). Because the cap + fan stay maxed at all times, the
  `auto_baseline` row IS the kernel-climb state, so its N-rep median is
  both the gain reference AND the attribution baseline — no separate
  vendor-default measurement is taken. That median is reported as
  `kernel_baseline_tput`.
  * **A fresh winner** clears the keep gate vs the **reference** (the
    co-timed `auto_baseline` row's N-rep median, which re-measures the
    incumbent climb state; falls back to the climb `base_tput` if those
    reps all failed) → `final_state="applied_best"`; the winner is
    applied and the headline measurement is lifted.
  * **No fresh winner** → `final_state="kept_incumbent"`; the auto
    incumbent (perflevel auto + cap-max + fan-max) is re-applied and
    preserved (it holds through the remaining sweep / close phases and a
    resume re-applies it).

The settle sweep uses the higher PM keep cutoff
`KERNEL_PM_KEEP_THRESHOLD_PCT` (2.0 %) rather than the 1.0 % single-node
noise floor, because power changes carry thermal / reproducibility cost.

**Settle-hold timeout.** `_maybe_hold_kernel_for_power_sweep` stamps
`SharedState.power_settle_hold_started_ts` on the first defer and latches
`power_settle_sweep_done` (proceeding to SWEEP) if the hold exceeds the
deadline — so a lost / crashed settle task can't wedge the plateau. A
failed settle task also latches the gate via `_handle_unpromotable_result`.

### One tuned power state per run

There is exactly one tuned power state per run — the auto
incumbent or the settle winner — recorded on
`SharedState.host_state_applied`, with the power-only attribution on the
flat `SharedState.power_attribution` dict. There is no per-combo power
map and no mid-climb inheritance: the combo can change many times during
the climb, but power stays at the fixed incumbent state until the
settle sweep.

**Run-start / resume reconciliation** (`_ensure_run_start_power_reset`):
* **Fresh run** — mandatory reset to vendor defaults *and* clear the
  `host_state_applied` record (nothing is applied yet).
* **Resume into KERNEL** — reset, then re-apply the recorded
  `host_state_applied` (the fixed incumbent climb state — auto +
  cap-max + fan-max — or a settle winner). KERNEL is the only phase a
  resume can keep iterating in, so it's the only one whose tuned state
  is re-applied to the live GPU.
* **Resume into any other phase (EXPLORE / SWEEP / CLOSE) or multi-node**
  — reset the hardware to vendor defaults but **preserve** the
  `host_state_applied` record; it drives the report + recipe and must not
  be wiped just because the live GPU was reset.

**`--no-kernel` runs do no power management at all** — they route
EXPLORE → SWEEP and never enter KERNEL, so neither the incumbent climb
state nor the settle sweep ever fires.

**Applicability:** `applicable_when: baseline_tput > 0` and
`not is_multi_node`. On `>=2`-node RayJob clusters the catalogue
predicate hides the action; the executor still hard-refuses with
`error_class='multi_node_unsupported'` on stale resume / direct invoke.
See **Multi-node behaviour** below.

**Not a hard gate:** `_required_next_step()` does not mandate PM. The
Orchestration LLM must delegate it during SWEEP (or an operator must
enqueue it). Long sessions can finish without any PM round if the LLM
never proposes it and SWEEP budget expires first.

**Promotion boundary:** PM never writes power knobs onto
`current_best.extra_server_args` (host power state is not a server
flag). The executor's `best_variant` still updates
**`host_state_applied`** + `power_management_search`. On a fresh settle
winner (`final_state="applied_best"`) the Coordinator treats it as a
*measurement refresh*: it lifts `current_best.tput` +
`cumulative_gain_validated` (tput only — args untouched) to the combined
throughput and records the power-only delta on the flat
`SharedState.power_attribution`. On `kept_incumbent` the incumbent
auto tput is already the headline, so only the attribution is
recorded (no further lift). It is deliberately **not** appended to
`gain_per_stack_entry`, which stays index-aligned with
`optimization_stack`; power never lands in `optimization_stack`.

**Attribution (`power_attribution`):** the recorded delta is
`max(0, (combined − kernel_tput) / kernel_tput)`. `kernel_tput` is the
settle sweep's `kernel_baseline_tput` — the **median** kernel-only
throughput measured at true vendor defaults on the plateaued combo — so
`power_delta_pct` is the **full** power contribution (the held
MAX-state gain + any settle gain). The delta is clamped to ≥ 0 and the
entry carries `low_confidence: true` when the median baseline meets or
beats the combined tput (noise inverted the sign), so the recipe never
ships a negative power gain. `n_reps` records how many baseline reps
produced a usable number. This is the number the recipe's
`power_state.power_gain_pct` carries for the shipped config.

This attribution is **single-point**: it reflects power's contribution at
the one representative workload shape PM benches at (the baseline
config's ISL/OSL/CONC). The SWEEP phase later characterizes the fixed
(kernel + power) config across the full ISL/OSL/CONC grid but does **not**
re-tune or re-attribute power per operating point, so the recipe's
`power_gain_pct` is "full power contribution at the representative point",
not a per-shape figure. Audit rows land in
`SharedState.power_management_attempts` (same schema as
`explore_attempts` / `sweep_attempts`) for FAILURE RECOVERY and the
cross-call ledger (see below).

## Inputs (task.params)

All optional; see `actions/_meta/power_management.yaml` for the full
`params_schema`. The most useful overrides:

```yaml
grid:                    # list[dict] — LLM-injected power variants
  - {name: cap_250w, power_cap_w: 250}
  - {name: perf_high, perflevel: high}
  - {name: perf_det_1900, perf_deterministic_mhz: 1900}
  - {name: cap_300w_det_1850,
     power_cap_w: 300, perf_deterministic_mhz: 1850}
power_cap_floor_w: 150   # soft floor; lifted to rocm-smi hardware min when one is reported (current CLI doesn't expose it, so this stays the effective floor)
power_cap_ceiling_w: 0   # reject caps above this (0 → use --showmaxpower per-GPU ceiling)
revalidate_winners:      # 'lazy' (default) | 'always' | 'never' — see "Cross-call ledger"
force_retest: false      # bypass cross-call dedup (re-bench fingerprints from prior rounds)
dry_run: false           # skip rocm-smi + rebench; return resolved grid only
config_path:             # base Magpie YAML (defaults to baseline asset)
output_dir:              # workspace root (default: <SD>/runs/power_management/<task_id>/)
base_extra_args:         # current best EXTRA_SGLANG_ARGS to layer onto each variant
base_tput:               # current best throughput; the fallback gain reference. On the settle path the freshly-benched `auto_baseline` N-rep median is used as the reference instead (see `reference_tput`).
keep_threshold_pct:      # winner threshold (shared name with `explore` / `framework_agent`); default 1.0 single-node (Magpie noise floor). Multi-node default 2.0 is unreachable here — the action refuses `is_multi_node` sessions.
variant_timeout_sec:     # per-variant Magpie timeout (default 2400)
benchmark_script:        # sanitized *.sh override
result_dir:              # sanitized $RESULT_DIR override
```

The Coordinator additionally injects two SharedState surfaces into
every `power_management` task — operators / tests calling the executor
directly can pass these explicitly:

```yaml
power_management_search:  # SharedState.power_management_search ledger
                          # (tested fingerprints + accepted winners)
host_state_applied:       # SharedState.host_state_applied snapshot of
                          # the GPU state currently in force (or None);
                          # the settle sweep's auto incumbent
bound_kind:               # 'memory' | 'compute' | 'unknown' — roofline
                          # bottleneck that routes the settle grid
```

Grid-source: the roofline-routed settle sweep (or an explicit
`params.grid`, benched as-is). The settle sweep has no per-combo
seeding — power is tuned once on the plateaued combo.

## The settle sweep (roofline-routed grid)

When `params.grid` is omitted (the Coordinator-internal settle path),
the executor probes `rocm-smi --showmaxpower --json` (per-board
ceiling), `rocm-smi --showsclkrange` (the clean top sclk MHz),
`rocm-smi --showclkfrq --json` (top-sclk fallback + the selectable
`mclk[N]` levels), and `rocm-smi --showmclkrange` (memory range), then
builds **one roofline-routed grid** (`_build_settle_grid`). All rows
bench through the same per-variant `run_grid` machinery; the
highest-median-throughput row that clears the noise floor over
`auto_baseline` wins.

> An explicit `params.grid` is **not** routed — the operator owns the
> shape and it's benched as-is, with no `auto_baseline` injected.

Every challenger row AND every `auto_baseline` rep carries cap-max +
fan-max. Because `--resetpoweroverdrive` / `--resetfans` clear those
between variants, they are re-asserted on every row after each
inter-variant reset (reset order stays `resetperfdeterminism` →
`resetclocks` → `resetpoweroverdrive` → `resetfans`).

### Grid rows

| Row | Applied state | Gate |
|---|---|---|
| `auto_baseline` (incumbent, **N reps**) | `--setperflevel auto` + cap-max + fan-max | always |
| `high` (reference) | `--setperflevel high` + cap-max + fan-max | always |
| `det_100` (anchor) | `--setperfdeterminism <top>` + cap-max + fan-max | always |
| `det_95`, `det_90`, `det_85` | `--setperfdeterminism <pct·top>` + cap-max + fan-max | NOT compute-bound |
| `mclk_*` (mem stepped) | `--setperflevel manual` + `--setsclk <top>` + `--setmclk <N>` + cap-max + fan-max | NOT memory-bound AND ≥2 mclk levels |

* **GFX determinism ladder** = `{1.00, 0.95, 0.90, 0.85} × top_sclk`
  (configurable via `_SETTLE_DETERMINISM_PCTS`). `top_sclk` comes from
  `--showsclkrange` (clean range), falling back to the `--showclkfrq`
  max. `det_100` always anchors; `det_95` / `det_90` / `det_85` are
  pruned when `bound_kind == "compute"` (lowering GFX can only hurt a
  compute-bound workload, so only the anchor is kept as a
  determinism-vs-auto reference).
* **Determinism rows carry no perflevel / sclk_idx** —
  `--setperfdeterminism` sets `perflevel=DETERMINISM`, a single global
  mode that cannot coexist with a manual clock pin (see
  `_is_contradictory_combo`), and `--setmclk` is a no-op outside
  `perflevel=manual`. So determinism rows carry only the MHz value +
  cap + fan.
* **Memory axis is capability-gated.** The `mclk_*` rows step the
  memory clock (with `perflevel=manual` and `--setsclk` pinned to the
  highest-frequency GFX index so GFX stays high) only when the workload is **not**
  memory-bound AND `--showclkfrq` exposes **≥2 distinct selectable
  mclk levels**. With `<2` levels the rows are omitted and the reason
  is logged + surfaced in `grid_degraded`. On this MI355X (a single
  2000 MHz mclk level) the memory axis is skipped-with-reason.
* **HW-gated determinism, no detection needed.** Determinism is
  GFX-only and PMFW-dependent (gated on MI300, defeatured on MI350). A
  row whose `--setperfdeterminism` apply fails is dropped via the
  per-variant failure path.
* Ladder rows already in the cross-call `tested` ledger are dropped
  (unless `force_retest=true`). The `auto_baseline` row is **exempt**
  from cross-call dedup — it must re-measure the live incumbent every
  call to be a valid reference.

### Grid self-check (`grid_degraded`)

After the grid is built, the executor runs a self-check: if a ladder
*was expected* (per the roofline + capability gates) but produced **0**
rows, it emits a `log.warning` and attaches a structured
`grid_degraded` field to the result rather than silently collapsing to
an auto-only "sweep". This catches a regression in clock-table parsing
(e.g. `--showsclkrange` output drift) **before** it wastes a multi-hour
run measuring nothing but the incumbent against itself.

### Why a determinism ladder, not index pinning

For a pure max-throughput goal, the only way to beat "auto + max power"
with standard knobs is to discover that the auto governor
*self-throttles* (a power / thermal limit, or a clock that sags under
load) and a slightly-lower-but-**stable** locked frequency sustains
higher throughput. `--setperfdeterminism` locks GFX to a precise MHz
target, giving a fine, evenly-spaced ladder (100 / 95 / 90 / 85 % of top).
The DPM **index** table (`--setsclk`) is rejected for GFX because it is
coarse and scrambled on this silicon (`0:500`, `1:158`, `2:2400` MHz):
`top-1` is a ~90 % frequency cliff, not a ~5 % refinement, so it can't
probe the "slightly lower but stable" regime the search is looking for.
Index pinning is retained only for the **memory** axis (`--setmclk`),
where it remains the only lever and the levels are usable.

Regular knobs only. Beyond-max levers (power-overdrive above the
ceiling, a power-cap-down ladder, a `--setsrange` clock floor) are a
future iteration (see "Future iteration" below). PCIe is not tuned
(negligible lever for single-GPU inference).

#### Result payload — settle-sweep fields

| Field | Meaning |
|---|---|
| `bound_kind` | `"memory"` / `"compute"` / `"unknown"` — roofline bottleneck that routed the grid |
| `grid_degraded` | `null` when healthy, else `{expected, reason, ...}` — an expected ladder produced 0 rows |
| `reference_source` | `"auto_baseline"` (N-rep median) / `"base_tput"` (fallback when reps failed) |

### Future iteration — power-gating axis (deferred)

A `det_100 + power-cap-down` ladder is a future GFX lever and a direct
test of the thermal / power-throttle hypothesis. It is deliberately
**not** implemented in v1 for two reasons:

* A cap set as a percent of the 1400 W ceiling is **inert** for
  memory-bound workloads that never approach that draw — a meaningful
  cap must be relative to *measured* load power (a two-pass step).
* It should be **evidence-gated** by per-variant telemetry
  (`--showmetrics` / `--showtemp`: hotspot vs Tjmax, socket power vs
  cap, `current_gfxclk` sag). When added, gate it behind telemetry
  showing a genuine thermal / power limit at fan-100, and pair it with
  `det_100` (a ceiling that floats), **never** `perflevel high` (a hard
  pin that fights the cap). This requires adding per-variant temp /
  power / clock sampling to support both attribution and the gate.

### Cross-call ledger + winner re-validation

`power_management` is structurally the same shape as `explore`: a
per-call DFS over a knob grid that remembers what was tried, what won,
and what to deepen on the next round. Server flags use
`SharedState.explore_search`; host power uses
`SharedState.power_management_search` plus
`SharedState.host_state_applied` (the applied rocm-smi snapshot).

**What the executor reads on entry:**

* `power_management_search.tested` — content-fingerprint set of every
  variant benched across all prior rounds. The executor drops any
  variant whose fingerprint matches before the grid runs, so an LLM
  rename of an already-tested variant collapses to the same row.
  `force_retest=true` bypasses the dedup. (The `auto_baseline` row is
  exempt — it always re-measures the live incumbent.)
* `power_management_search.accepted` — prior winners. Subject to the
  `revalidate_winners` mode, these are rehydrated as the first
  grid rows so each call confirms the prior winner still holds
  before exploring fresh variants.
* `host_state_applied.measured_state` — the rocm-smi readback at the
  time the last winner was applied. The `lazy` revalidation mode
  re-probes `--showperflevel` on entry and treats any divergence as
  drift (operator tweak, reboot, thermal event) that warrants
  escalating to a full re-validation. The cap-setpoint side of drift
  detection is dormant on the current CLI (no probe exposes the
  setpoint); perflevel drift alone has been sufficient in practice
  because every supported variant either pins or releases perflevel
  alongside the cap.

**Variant fingerprint** is a 16-char hash over `(power_cap_w,
perflevel, sclk_idx, mclk_idx, pcie_idx, perf_deterministic_mhz,
fan_pct, devices)` — explicitly excludes `name` and `note` so display
renames don't bypass dedup. Same width as `explore_search` fingerprints
so prompt formatters apply uniformly.

**`revalidate_winners` modes:**

| Mode | Re-validates prior winners when... |
|---|---|
| `lazy` (default) | The probe detects host drift (cache stale) |
| `always` | Every call, regardless of cache state |
| `never` | Never — trusts the cache unconditionally |

**Re-validation rows:** when the mode + cache state warrant it
(`always`, or `lazy` + detected drift), prior accepted winners are
rehydrated and benched BEFORE the fresh roofline-routed rows, so a
quick "yes, the prior winner still holds" re-confirmation arrives
before the slower fresh sweep. Each is re-checked against the current
effective cap bounds first (a probe-lifted floor can invalidate a
historical value).

**What the executor writes on exit (Coordinator persists):**

* `power_management_search_update` — the merged ledger with the new
  round's tested fingerprints + rejected entries. Accepted is left
  for the Coordinator's promote path
  (`record_power_management_accepted`) to maintain.
* `host_state_applied` — either a fresh snapshot (when a winner was
  re-applied) or `None` (when defaults were restored). The snapshot
  contains the exact rocm-smi commands needed to recreate the state,
  the device subset, the probed bounds at apply time, and a
  measured-state readback for the next call's drift check.
* One audit row per call lands in `power_management_attempts` with
  the standard schema; the prompt's FAILURE RECOVERY block consumes
  it like any other audit action.

### Floor derivation (`power_cap_floor_w`)

`power_cap_floor_w` is a **soft** lower bound. After the probe runs,
the executor lifts the effective floor to
`max(power_cap_floor_w, <hardware minimum from rocm-smi>)` — i.e. the
GPU's hardware-reported minimum always wins when it is stricter. This
catches the otherwise silent failure mode where an LLM proposes (say)
`power_cap_w=175` on a GPU whose minimum is 200 W: instead of waiting
for `--setpoweroverdrive` to error out at apply time, the variant is
rejected up front with `reason="below floor"` and the lifted value
appears in the result payload's `power_floor_w` field.

The current upstream `rocm-smi` Python CLI does NOT expose a
hardware-minimum reading (`--showmaxpower` is a ceiling-only probe).
`_probe_powercap_range` therefore returns `min_w=0` as a sentinel,
the lift-to-hardware path is a no-op, and the effective floor stays
at the operator/default soft value. The lift logic is kept live so a
future CLI revision (or an `amd-smi` re-target) that DOES report the
minimum will pick it up automatically. The probe step is skipped
under `dry_run=true`, so dry runs only see the operator / default
floor regardless.

## Multi-node behaviour

**The executor refuses to run on a `>=2`-node RayJob cluster.**
`rocm-smi` is a node-local tool — every `--setpoweroverdrive` /
`--setperflevel` / `--setsclk` invocation mutates the GPUs of the
host the executor runs on (the orchestrator head node). On a
multi-node session the peer workers driving the same Magpie request
set would remain at their boot-time defaults, producing a measurement
where the cluster's GPUs are running in a heterogeneous power state
against a single-node baseline. The resulting `gain_pct` would
describe a configuration that cannot be reproduced past the action
boundary (only the head node carries the cap), and the cross-call
ledger would accumulate fingerprints whose `power_settings` block
silently lies about what the GPUs were actually set to.

The refusal lives in two layers:

* `actions/_meta/power_management.yaml` declares
  `applicable_when: not is_multi_node`, which keeps the action out of
  the Orchestration LLM's catalogue when the run is multi-node — the
  LLM never sees `power_management` as a proposable arm.
* `PowerManagementExecutor.__call__` re-checks `is_multi_node()` at
  entry and returns
  `{status: "failed", error_class: "multi_node_unsupported", ...}`
  before any rocm-smi call. This is the backstop for stale
  `state.json` resumes / direct operator invocations / ledger replay
  paths that bypass the catalogue. The runtime guard is skipped under
  `dry_run=true` so unit tests + the probe script can still exercise
  the variant-resolution code paths under simulated multi-node config.

A future relaxation that wanted to enable multi-node power-management
would need to (a) fan out the rocm-smi setters to every worker node
(probably via a Ray task driven from the executor), (b) reconcile per-
node `[min,max]` cap ranges into a single effective grid, and (c)
adopt the multi-node Magpie noise floor (2.0 % instead of 1.0 %) for
the gain gate. Until those land, single-node-only is the honest
contract.

## Elevation (sudo / root)

The setter and reset calls (`--setpoweroverdrive`,
`--resetpoweroverdrive`, etc.) require root. The executor auto-detects via `os.geteuid()`:

* `geteuid() == 0` — the normal case inside the `rocm/sglang` and
  `rocm/vllm` Docker images Hyperloom ships in. The rendered shell
  commands drop the `sudo` prefix entirely; those images typically do
  not install `sudo` at all.
* `geteuid() != 0` — bare-metal ops or non-root containers. The
  rendered commands carry `sudo` and the host MUST have a `NOPASSWD`
  sudoers entry for `rocm-smi`, e.g.
  `myuser ALL=(ALL) NOPASSWD: /opt/rocm/bin/rocm-smi`.

The probe (`scripts/probe_power_management_capability.py`) mirrors the
same detection logic — running it under each deployment mode confirms
the elevation path works end-to-end without touching the GPU.

## Outputs

Returns a dict shaped like `explore` / `sweep` so the existing audit
machinery picks it up:

```yaml
status:           "succeeded" | "failed" | "no_winners"
base_tput:        <float, tok/s/GPU>
reference_tput:   <float — gain reference: co-timed `auto_baseline` N-rep median, else base_tput>
reference_source: "auto_baseline" | "base_tput"
grid_size:        <int>
all_results:      [VariantResult.to_dict() + {power_settings:{...}}, ...]
winners:          [<variants clearing the keep gate vs reference_tput>]
best_variant:     <highest output_throughput variant or None>
best_gain_pct:    <best_variant.gain_pct or 0.0>
power_floor_w:    <int — applied floor>
power_ceiling_w:  <int — applied ceiling (probed or operator-supplied)>
final_state:      "applied_best" | "kept_incumbent" | "reset_to_default" | "reset_after_failure"
kernel_baseline_tput: <float | null — `auto_baseline` N-rep median (incumbent kernel-climb state)>
kernel_baseline_reps: <int — reps that produced a usable baseline number>
workspace:        <str — output_root, all per-variant rebench dirs sit here>
# Roofline-routed settle-sweep surfaces
bound_kind:       "memory" | "compute" | "unknown"   # roofline bottleneck that routed the grid
grid_degraded:    null | {expected, reason, ...}     # expected ladder produced 0 rows
# Cross-call deepening surfaces
revalidate_mode:  "lazy" | "always" | "never"
cache_stale:      <bool — True when probe diverged from host_state_applied.measured_state>
dropped_variants: [{name, fingerprint, reason}, ...]   # cross-call dedup output
# Coordinator-consumed (persisted into SharedState by _promote_to_shared_state)
power_management_search_update: {schema_version, tested, rejected, accepted, name_index, cursor, last_round}
host_state_applied: {variant_name, power_settings, smi_commands, device_ids,
                     probed_range_w, top_sclk_mhz, measured_state, gain_pct,
                     ts, session_dir, task_id} | None
```

`host_state_applied` is the canonical "what is the GPU set to right
now?" record. The final report renders its `smi_commands` block beside
`current_best.extra_server_args` so a fresh operator can replicate the
end-of-run machine state by re-running both. On `applied_best` it
carries the settle winner; on `kept_incumbent` the Coordinator preserves
the auto-incumbent snapshot. When `final_state` is
`reset_to_default` or `reset_after_failure`, the Coordinator clears
`SharedState.host_state_applied` to `None` so a stale cache never
misrepresents the GPU.

## Failure handling

The executor wraps every `rocm-smi` mutation in a `try / finally` so
that even on a Magpie timeout, GPU crash, or `KeyboardInterrupt` the
defaults get restored (`--resetpoweroverdrive --resetclocks
--resetperfdeterminism --resetfans`). If `rocm-smi` is not on PATH
the action returns
`{status: "failed", error_class: "rocm_smi_unavailable", ...}`; if
`--showmaxpower` fails (parse error, missing flag) AND the operator
did not supply an explicit `params.grid`, the action returns
`{status: "failed", error_class: "empty_grid", ...}`. Repeated failures show up in `power_management_attempts` and the
prompt FAILURE RECOVERY block; v0.8 has no bandit `eff_score` /
`scoring.apply_failure` path for shallow actions.

The accuracy gate does NOT fire on `power_management` (none of the
power knobs touch precision-affecting flags).
