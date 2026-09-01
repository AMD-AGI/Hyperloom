# Recovery & Failure Handling

## Recovery

If the CLI exits with `Claude SDK exit code 1` or
`Primus.00009 token not present`, the gateway rejected the request. Check that
`OPENAI_BASE_URL` / `OPENAI_API_KEY` are set in `.env` (or the calling shell) and
that the gateway is reachable:

```bash
curl -sS -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_BASE_URL/models" | head
```

If `_preflight()` itself fails, run install in `--check-only` mode to see which
piece is missing, then re-run full install:

```bash
bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh" --check-only
bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"
```

If install repeatedly fails while building GEAK / `mini-swe-agent` with missing
files such as `src/minisweagent/...`, the workspace-shared GEAK mirror may be
half-created (`.git` exists but `src/` is incomplete) or the filesystem may be
showing stale metadata. Do not manually clone GEAK, delete only `build/`, or
edit the checkout in place. Stop any other installer using the same dependency
root, remove the entire
`${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/GEAK`
directory, then rerun the full install so `install.sh` owns the fresh clone.
Multiple concurrent installs sharing one dependency root also share this
checkout; avoid running them at the same time.

In sandboxes where `/workspace/hyperloom` is unwritable, override the
**workspace root** with `USER_DATA_PATH` (not the per-session subdir):

```bash
export USER_DATA_PATH="/shared/hyperloom-sessions"   # workspace root
mkdir -p "$USER_DATA_PATH"
```

The CLI calls `make_session_dir(model_name=…)` once at startup; that creates
`$USER_DATA_PATH/<model_basename>/<UTC_ts>/` and pins
`$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`.

## Failure Handling

Auth / SDK drift (`Claude SDK exit code 1`, `Primus.00009 token not present`,
`ANTHROPIC_AUTH_TOKEN not set`, `BackendError: claude-agent-sdk not installed`,
`Fatal error in message reader`) is owned by `_preflight()`; see Recovery above
for the supervisor + install rerun loop. Manual SDK fallback if frozen pip
blocks `_ensure_python_sdks()`:
`python -m pip install 'claude-agent-sdk>=0.2.110' 'openai>=1.50' 'httpx>=0.27'`.
Transient SDK errors retry/resume up to the Coordinator emergency threshold.

### Model-gate errors (preflight #10)

Custom orchestration models are enabled by default and are validated against the
configured gateway catalog. Set `INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0`
only when you intentionally want the strict AMD Claude allowlist
(`claude-opus-5` / `claude-opus-4-8` / `claude-opus-4-7` / `claude-opus-4-6`).

| Symptom | Fix |
|---|---|
| `--claude-model=... is not allowed` | You likely set `INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=0`; unset it or set it to `1`, then ensure the model appears in the gateway `/models` catalog. |
| `gateway catalog unreachable after retries` (4 probes at 0/1/3/5s) | Reproduce: `curl -k -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_BASE_URL/models" \| jq '.data[].id'`. Gateway answers → proxy/SSL is wrong; gateway down → fix gateway. Fail-fast is intentional vs. 401 mid-baseline. |

### Critic-agent runtime errors

Inspect `$SESSION_DIR/critic-workdir/<latest>/{request,judge_bundle,review,emit}.json`.
Bypass with `--critic-mock` for offline / smoke runs. See
[critic.md](critic.md).

| Symptom | Fix |
|---|---|
| `--critic-agent selected but critic-agent runtime not found` | `export CRITIC_AGENT_ROOT=/path/to/src/hyperloom/agents/critic`, or check the `src/hyperloom/agents/critic/` install. |
| `hyperloom.agents.critic.runtime.cli prepare-review/commit-review exited rc=2` | Schema/validation bug (per `src/hyperloom/agents/critic/README.md` §Exit codes). Inspect workdir payload; retry with `--critic-mock` while fixing. |
| `hyperloom.agents.critic.runtime.cli ... timed out after 30s` | Critic runtime is stuck. If `CRITIC_KB_CLIENT_MODE=live`, drop to the default `inmemory` mode. A default local-only run should not block on remote I/O. |
| All verdicts `('needs_review','critic_unavailable')` + `kb_skipped=missing_critical_context` | Static context load failed. Check `manifest.json` has non-empty `model_name`/`framework`; grep `logs/cli.log` for `critic_agent_backend static_context`. |

### Run-time signals

- `No accelerator` (Magpie): subprocess `PATH` must lead with
  `$(dirname "$PYTHON")` (or set `MAGPIE_PYTHON`); use `ROCR_VISIBLE_DEVICES`,
  not `HIP_VISIBLE_DEVICES`.
- Repeated `trace_analyze` with unchanged trace/config: bug — reuse
  `last_trace_analyze`.
- `correctness_passed=false`: do not integrate; the kernel-agent report must
  contain explicit correctness evidence.
- `stop_reason=global_converged`: the cyclic phase machine exhausted leverage
  across macro-cycles for the current budget (R7: consecutive no-gain cycles);
  stop and report, only resume if the user changes workload / search space /
  model / strategy.
  (Leverage exhaustion *within* a single phase is now the non-terminal
  phase-exit reason `optimize_no_more_leverage` / `kernel_no_more_leverage`,
  which switches lever rather than ending the run.)
- `stop_reason=policy_loop`: a legacy stop_reason kept in the vocabulary for
  resuming old sessions; nothing in the runtime sets it. Repeated `policy_denied`
  for the same (action, rule) pair is advisory only — there is no auto-prune at
  streak ≥5 and no `policy_loop` stop at streak ≥10. Inspect
  `SharedState.policy_denial_history` via the `why_denied` tool or the
  `=== Recent policy denials ===` block, then change something substantive (a new
  `params.grid` variant, a different `benchmark_script`, or a sibling action
  family). Do not hand-edit `state.json`.
- `stop_reason=time_exhausted`: resume same session (`--resume-from`); do not
  start fresh.
- `dead_holder_reaped` (lease reclaimed because the holder pid vanished): the
  optimizer process died without unwinding, which on a sandboxed runtime usually
  means the pod itself went away mid-task. Read
  `runtime/pod_history.jsonl` — one append-only line per optimizer-lock
  acquisition (`acquired_at` / `hostname` / `pid`), so more than one line means
  the session was rebuilt and each entry marks a takeover. Prefer it over
  `manifest.json` (pins only the *first* owner) and over `runtime/optimizer.lock`
  (holds only the *current* one); a multi-rebuild session reads as single-pod
  from those two alone. Correlate the takeover timestamps with the task that
  failed — a roofline reclaimed here typically leaves a partial `torch_trace/`
  and no `analysis.md`.
