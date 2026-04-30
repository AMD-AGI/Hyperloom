# Iron Rules — Soft (Plan A: kernel agent's view)

Plan A demoted IR-1 / IR-2 / IR-6 / IR-7 from BLOCK to WARN. Kernel
agent **must still respect them** in spirit — they protect gain
estimates and prevent kernel source corruption — but a single
isolated violation no longer aborts the run. Helper scripts emit
WARNING lines on stderr; you should mirror them in
`response{result.warnings[]}` so the executor sees them.

The three BLOCK rules (IR-3 / IR-4 / IR-5) **cannot** be softened —
they are in `../SKILL.md` "Hard rules" and apply unchanged.

---

## IR-1 (WARN) — Submit kernel candidates IN PARALLEL via GEAK MCP

**Why**: GEAK rounds are 10-30 minutes each; serial submission burns
hours of budget.

**How**: in `run_optimization`, fan out one Bash invocation per
(candidate, backend) pair using `&` background + `wait`. Or open
multiple Bash tool calls in a single Claude turn — the SDK runs them
concurrently.

**If violated**: helper script logs WARN, you continue but record
`warnings.append("submitted N candidates sequentially")` in your
RESPONSE result.

---

## IR-2 (WARN) — Never modify kernel source before GEAK runs

**Why**: GEAK includes its own kernel adaptation logic; pre-edited
sources confuse it (decorator stripping, stride changes, etc.).

**How**: between `select_kernels` and `run_optimization`, do **NOT**
use the `Edit` tool on any kernel source path. Read-only via `Read`
is fine.

**If violated**: helper detects mtime change since selection; logs
WARN; you can still proceed but record `warnings.append("kernel
<name> modified after selection")`.

---

## IR-6 (WARN) — patch_inductor.py argv discipline

**Why**: `--target-file` is mandatory for the patcher to find the
right cache file. Missing `--best-config` when changing
`block_size`/`num_warps` produces numerically broken output (garbled
tokens at inference time).

**How**: `apply_patch.sh` always passes `--target-file`; it passes
`--best-config` when the input patch metadata declares tuning of
those keys.

**If violated**: `patch_inductor.py` logs a WARN to stderr and writes
a manifest with `target_file: null`. Apply still proceeds. Mirror the
warning in your RESPONSE.

**Strict mode**: setting `INFERENCE_OPTIMIZER_IR6_STRICT=1` in the env
restores the legacy hard-block behavior — patch_inductor.py rc=2 + no
manifest. Use only for CI.

---

## IR-7 (WARN) — Never modify GEAK MCP config

**Why**: GEAK is shared infrastructure; runtime mutation breaks other
users + your future runs.

**How**: only `geak_get_*` and `geak_create_task` / `geak_submit_task`
tool calls are allowed. **Exception**: `kernel_opt` action's first
turn calls `geak_set_model_config` ONCE to inject observability
tracing headers; this is the ONLY allowed write.

**If violated**: `run_geak.sh` rejects the invocation if the env
indicates a forbidden write attempt. RESPONSE with
`status="failed", result.reason="ir7_violation_geak_config_mutation"`.

---

## Quick reference

| IR | Severity | What it protects |
|---|---|---|
| IR-1 | WARN | Throughput of kernel-opt loop (parallel candidates) |
| IR-2 | WARN | GEAK rewrite quality (no pre-edit) |
| IR-3 | **BLOCK** | Gain validation (must integrate after kernel-opt) |
| IR-4 | **BLOCK** | Process safety (kill + GPU mem check before launch) |
| IR-5 | **BLOCK** | Conductor self-protection (no `pkill -f sglang`) |
| IR-6 | WARN | Patch correctness (--target-file + --best-config) |
| IR-7 | WARN | GEAK shared infra integrity |

The full IR text + predicates lives in
`src/inference_optimizer/orchestrator/iron_rules.py`.
