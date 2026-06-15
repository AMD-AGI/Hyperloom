# Recovery & Failure Handling

## Recovery

If the CLI exits with `Claude SDK exit code 1` or
`Primus.00009 token not present`, the gateway rejected the request. Check that
`OPENAI_BASE_URL` / `SAFE_API_KEY` are set in `.env` (or the calling shell) and
that the gateway is reachable:

```bash
curl -sS -H "Authorization: Bearer $SAFE_API_KEY" "$OPENAI_BASE_URL/models" | head
```

If `_preflight()` itself fails, run install in `--check-only` mode to see which
piece is missing, then re-run full install:

```bash
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh" --check-only
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
```

If install repeatedly fails while building GEAK / `mini-swe-agent` with missing
files such as `src/minisweagent/...`, the workspace-shared GEAK mirror may be
half-created (`.git` exists but `src/` is incomplete) or the filesystem may be
showing stale metadata. Do not manually clone GEAK, delete only `build/`, or
edit the checkout in place. Stop any other installer using the same dependency
root, remove the entire
`${HYPERLOOM_OPEN_SOURCE_ROOT:-${TMPDIR:-/tmp}/hyperloom/open-source-repos}/GEAK`
directory, then rerun the full install so `install.sh` owns the fresh clone.
Multiple concurrent installs sharing one dependency root also share this
checkout; avoid running them at the same time.

In sandboxes where `/workspace/hyperloom` is unwritable, override the
**workspace root** with `USER_DATA_PATH` (not the per-session subdir):

```bash
export USER_DATA_PATH="/wekafs/xiaofei/sessions"   # workspace root
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
`python -m pip install 'claude-agent-sdk>=0.1.65' 'openai>=1.50' 'httpx>=0.27'`.
Transient SDK errors retry/resume up to the Coordinator emergency threshold.

### Model-gate errors (preflight #10)

Allowlist: `claude-opus-4-7` (preferred) → `claude-opus-4-6` (fallback). The
gate is intentional — opus-4-5 / haiku silently degraded prior runs.

| Symptom | Fix |
|---|---|
| `--claude-model=... is not allowed` | Drop `--claude-model` / `$CLAUDE_MODEL`. Update `_CLAUDE_ALLOWED_MODELS` in `cli.py` only when a successor is blessed. |
| `gateway catalog unreachable after retries` (4 probes at 0/1/3/5s) | Reproduce: `curl -k -H "Authorization: Bearer $SAFE_API_KEY" "$OPENAI_BASE_URL/models" \| jq '.data[].id'`. Gateway answers → proxy/SSL is wrong; gateway down → fix gateway. Fail-fast is intentional vs. 401 mid-baseline. |

### Critic-agent runtime errors

Inspect `$SESSION_DIR/critic-workdir/<latest>/{request,judge_bundle,review,emit}.json`.
Bypass with `--critic-mock` for offline / smoke runs. See
[critic.md](critic.md).

| Symptom | Fix |
|---|---|
| `--critic-agent selected but critic-agent runtime not found` | `export CRITIC_AGENT_ROOT=/path/to/critic-agent`, or `git -C "$REPO_ROOT" submodule update --init critic-agent`. |
| `runtime.cli prepare-review/commit-review exited rc=2` | Schema/validation bug (per `critic-agent/AGENTS.md` §Exit codes). Inspect workdir payload; retry with `--critic-mock` while fixing. |
| `runtime.cli ... timed out after 30s` | KB stuck. If `CRITIC_KB_CLIENT_MODE=live`, drop to `inmemory`. Reproducing in `inmemory` is a bug — that path must not block on I/O. |
| All verdicts `('needs_review','critic_unavailable')` + `kb_skipped=missing_critical_context` | Static context load failed. Check `manifest.json` has non-empty `model_name`/`framework`; grep `logs/cli.log` for `critic_agent_backend static_context`. |

### Run-time signals

- `No accelerator` (Magpie): subprocess `PATH` must lead with
  `$(dirname "$PYTHON")` (or set `MAGPIE_PYTHON`); use `ROCR_VISIBLE_DEVICES`,
  not `HIP_VISIBLE_DEVICES`.
- Repeated `trace_analyze` with unchanged trace/config: bug — reuse
  `last_trace_analyze`.
- `correctness_passed=false`: do not integrate; the kernel-agent report must
  contain explicit correctness evidence.
- `stop_reason=no_more_leverage`: stop and report; only resume if the user
  changes workload / search space / model / strategy.
- `stop_reason=policy_loop`: Coordinator hit ≥10 consecutive `policy_denied`
  events for the same action/rule pair; all top actions may be locked or pruned.
  Inspect `SharedState.policy_denial_history` and the per-tick `Policy denials`
  block. To recover: manually edit `state.json` to remove the action from
  `pruned_families`, clear `policy_denial_streak` / `stop_reason`, and re-propose
  with fresh `params.grid` content (omit stale `idempotency_key`).
- `stop_reason=time_exhausted`: resume same session (`--resume`); do not start
  fresh.
