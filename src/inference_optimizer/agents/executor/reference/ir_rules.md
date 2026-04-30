# Iron Rules — Executor's View (Plan A)

These are the rules you (the executor) need to remember. The full set
(IR-1..IR-7) lives in `src/inference_optimizer/orchestrator/iron_rules.py`,
but Plan A divides them by ownership:

| Rule | Owner | Severity | You need to know? |
|---|---|---|---|
| IR-1 (parallel candidates) | **kernel agent** | WARN (Plan A) | No — kernel agent enforces |
| IR-2 (no source mod pre-GEAK) | **kernel agent** | WARN (Plan A) | No — see below |
| IR-3 (integrate after kernel-opt) | **kernel agent** | **BLOCK** | Indirect — via apply_patch |
| IR-4 (kill_server + GPU check) | shared | **BLOCK** | YES |
| IR-5 (no `pkill -f sglang`) | shared | **BLOCK** | YES |
| IR-6 (patch_inductor flags) | **kernel agent** | WARN (Plan A) | No — kernel agent enforces |
| IR-7 (no GEAK config mutation) | **kernel agent** | WARN (Plan A) | No — kernel agent enforces |

The four kernel-flavoured rules (IR-1/2/6/7) live in the kernel agent's
`reference/ir_soft_rules.md`. You cannot violate them because you don't
delegate kernel work directly anymore — `request{target=kernel}` hands
the responsibility to the kernel agent.

---

## IR-3 — Integration after kernel-opt (BLOCK)

When the kernel agent returns `response{kind=optimization_done}` with
candidates, you **MUST** follow up with `request{kind=apply_patch}`
before declaring the kernel-opt round complete. Skipping apply_patch
means the gain is unverified — the kernel agent's own SKILL warns about
this, but you are also responsible for tracking the round to closure.

If you accidentally drop the apply_patch step (e.g. you got distracted
by another event), the kernel agent will eventually emit
`alert{severity=medium, summary="optimization_done not followed by
apply_patch"}` — react by issuing the missing request.

---

## IR-4 — Always kill_server + check_gpu_memory before launch (BLOCK)

Before any server launch (in `delegate(baseline)`,
`delegate(bench_runner)`, etc.), the underlying ActionExecutor:

1. Kills any existing server (`pgrep -f sglang.launch_server` + `kill <pid>`)
2. Waits `SERVER_KILL_WAIT_S=10s`
3. Verifies GPU memory released (`rocm-smi` / `nvidia-smi`)

You don't normally invoke server lifecycle directly; the bundled
ActionExecutors handle it. If you DO open a Bash tool in quick mode
(rare), follow the same pattern — PolicyGate's allowlist requires it.

---

## IR-5 — Forbidden: `pkill -f sglang` (BLOCK)

Allowed:

```bash
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
kill $(pgrep -f 'python.*-m vllm.entrypoints')     2>/dev/null
```

Forbidden (would kill Ray workers / the conductor itself in claw mode):

```bash
pkill -f sglang   # NEVER
pkill -f vllm     # NEVER
```

PolicyGate's `QUICK_BASH_DENYLIST` rejects `pkill -f sglang|vllm` from
your `Bash` calls in quick mode. Other modes don't gate this in the
allowlist sense (you typically don't have raw Bash there), but the
discipline is the same.

---

## How violations surface

- **Pre-emptive (PolicyGate)** — `delegate(kernel_opt)` /
  `delegate(integrate)` / `update_state` to disallowed core fields /
  `pkill -f sglang` in quick → return `policy_denied` observation in
  your next inbox tick.
- **Action-time** — kernel agent's own scripts emit WARN lines on
  stderr for IR-1/2/6/7 violations and forward them in
  `response{result.warnings[]}`. Read them but don't panic.
- **Run-end (auditor, planned)** — silent IR-3/4/5 BLOCK violations
  that somehow got past the gates are caught at report time and flagged.
