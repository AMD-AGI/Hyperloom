# Failure Codebook

Lookup table for "I see error X in my inbox / observation event — what
should I do?" Use this BEFORE retrying anything.

## How to read this file

Each entry is keyed by the **observable signal** in your inbox (event
topic + kind, or observation kind). The "Recovery" column is the
recommended next intent.

| Signal | What it means | Recovery |
|---|---|---|
| `observation{kind=policy_denied, rule=role}` | Your role can't emit this intent type | Read SKILL.md "Allowed intents" — pick one in the list |
| `observation{kind=policy_denied, rule=mode}` | Action not allowed in current execution mode | Read `action_catalogue.md` to find one allowed in your mode |
| `observation{kind=policy_denied, rule=state_field}` | You tried to set a CORE field via `update_state` | Drop those fields; only `current_action` / `current_tput` / `crash_count` etc. allowed |
| `observation{kind=policy_denied, rule=payload}` | Required payload field missing | Re-read SKILL.md / actions/X.md for the exact JSON shape |
| `observation{kind=policy_denied, rule=bash}` | Bash command on quick allowlist denylist OR not on allowlist | Use a `delegate(...)` instead — actions wrap shell calls safely |
| `event{kind=delegate_dedup_to_terminal}` | Same `(action_name, params)` already terminal | Read `actions/retry_after_dedup.md` |
| `event{kind=delegate_failed}` | Sub-agent / ActionExecutor returned error | Check `payload.evidence.error`; pivot per the table below |
| `event{kind=lease_acquire_failed, lane=X}` | Lane X held by another in-flight task | Wait 1-2 ticks, OR pick an action that doesn't need lane X |
| `event{kind=lease_expired, lane=X}` | A previous task held lane X past its TTL → forcibly reaped | Read `state.json` for current run state; usually safe to retry the action with a fresh task_id |
| `decision{kind=state_updated}` with NO change to expected field | Action ran but didn't produce expected metric | Treat as failed; pivot like `delegate_failed` |
| `alert{severity=high, summary~="OOM"}` | Watchdog detected OOM cluster | DON'T immediately retry; pick a smaller config (lower CONC, smaller ISL) or `report` |
| `alert{severity=high, summary~="accuracy_drop"}` | Accuracy gate hit > 1% drop after a KEEP | The conductor already REVERTED; you don't need to act, but consider lowering future `predicted_gain_pct` for that action class |

## `delegate_failed` evidence patterns

Look at `payload.evidence.error` (or `payload.evidence.reason`):

| Evidence | Meaning | Recovery |
|---|---|---|
| `lane_contention` | Couldn't acquire required lane | Wait or pick different action (see table above) |
| `executor_crash` | Python ActionExecutor raised | Likely an env/setup bug; emit `alert(severity=high)` and switch to a different family |
| `backend_error` | LLM backend (Claude/Codex) call failed | Transient; safe to retry the SAME delegate (idempotency hash matches → dispatcher will dedup, but that's fine — pivot to a different action_name) |
| `unknown_action` | `action_name` not in registry | Typo; use names from the live "Available actions" table |
| `side_effect_evidence_insufficient_for_safe_replay` | Crash during a side-effect action; can't safely retry | Mark as terminal; emit `alert` for human review; move on to a different action |
| `BaselineExecutor: missing required env vars [MODEL, TP]` | Run not configured | Emit `alert(severity=critical)`; this is a setup failure, not your fault |

## Bash command rejected — what to use instead

| Rejected command | Use this delegate instead |
|---|---|
| `pkill -f sglang` | `delegate(server_lifecycle_restart, ...)` (or just rely on `BaselineExecutor` to handle restart) |
| `python -m sglang.launch_server ...` | `delegate(baseline, ...)` — it owns server lifecycle |
| `patch_inductor.py ...` | `delegate(integrate, ...)` — it handles patching per IR-6 |
| `git commit -m ...` | Never; agent doesn't write to git |
| `pip install ...` | Out of scope for executor; emit `alert(severity=high)` for human |
| `make` / `cmake` / `ninja` | `delegate(framework_rebuild, ...)` (marathon only) |

## Repeated `policy_denied` — last resort

If 3+ consecutive intents return `policy_denied`, you are deeply
mis-aligned with PolicyGate. Stop emitting new intents and emit ONE
diagnostic message:

```json
{
  "intents": [
    {
      "intent_type": "alert",
      "payload": {
        "severity": "medium",
        "summary": "Executor receiving repeated policy_denied; pausing for human review.",
        "detail": "Last 3 denials: <denial 1 rule+reason>; <denial 2>; <denial 3>"
      }
    }
  ]
}
```

Then emit `heartbeat` only on subsequent ticks until either:

- A `STOP_AGENT_executor` sentinel appears, OR
- A clear corrective signal arrives (e.g. mode change is impossible
  mid-run; what's likelier is that you finally read SKILL.md correctly).
