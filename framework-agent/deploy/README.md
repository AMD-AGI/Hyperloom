# framework-agent - Primus-Claw plugin deployment

This directory contains the artifacts that register framework-agent as a
**Primus-Claw plugin** (mirrors the existing `Hyperloom` plugin shape,
id=4 in production). The plugin runs on the same `sglang:v0.5.9-rocm700-mi30x`
container image and the same `/wekafs/HyperloomV2/` mount as Hyperloom, so
no new image build / Helm chart / Dockerfile is required for MVP.

## Contents

| File | Purpose |
|---|---|
| `claw-tool-prompt.json` | Prompt-tool payload for `POST /v1/tools/prompt`. Captures the `templates.default` shell flow + `targetPanel` + `selectedConfigKeys`. |
| `claw-plugin.json` | Plugin payload for `POST /v1/plugins/upsert`. References the prompt tool by id (filled in by the upload script after stage 1). |
| `upload-plugin.sh` | Two-stage uploader: creates / updates the prompt tool first, then upserts the plugin row. Auto-picks `/api` ingress prefix for external `https://` bases. |

## Prerequisites

1. **wekafs layout** — `framework-agent` source must be reachable at
   `/wekafs/HyperloomV2/framework-agent/` inside the plugin container.
   The plugin's prompt template runs `bash scripts/install.sh` to do a
   `pip install -e .` from there.

2. **Primus Cortex reachability** — the plugin container must be able to
   reach `http://primus-cortex-pr-api.primus-cortex.svc.cluster.local`
   from inside the `primus-claw{,-dev}` namespace. This is the same
   reachability the `Hyperloom` plugin already relies on.

3. **Claw API auth** — `upload-plugin.sh` reads a Bearer token from
   `CLAW_API_TOKEN` (env var) and the endpoint from `CLAW_API_BASE`
   (e.g. `http://primus-claw-api.primus-claw-dev.svc.cluster.local`).

## Upload

```bash
# Full API root including the higress path prefix
export CLAW_API_BASE="https://core42.primus-safe.amd.com/claw-api"
# OR in-cluster:
# export CLAW_API_BASE="http://primus-claw-api.primus-claw.svc.cluster.local"
# OR via kubectl port-forward:
#   kubectl port-forward -n primus-claw svc/primus-claw-api 19080:80 &
#   export CLAW_API_BASE="http://127.0.0.1:19080"

export CLAW_API_TOKEN="ak-<your-token>"

# Optional: dry-run first to inspect the resolved URLs + payloads
DRY_RUN=1 bash framework-agent/deploy/upload-plugin.sh

# Real upload
bash framework-agent/deploy/upload-plugin.sh
```

The script:

1. `POST <base>/v1/tools/prompt` with `claw-tool-prompt.json` → captures the
   returned `tool_id`.
2. Patches `claw-plugin.json` so `tools[0].id` is the real id.
3. `POST <base>/v1/plugins/upsert` with the patched plugin body.

Both endpoints are **idempotent in practice**: `POST /v1/tools/prompt`
rejects with a unique-violation when name+version collide (bump the
`version` in `claw-tool-prompt.json` to re-create), and
`POST /v1/plugins/upsert` is the dedicated idempotent path for plugins.

Final stdout looks like:

```
HTTP 201
  tool_id = 86
HTTP 200
OK: plugin_id=5  action=updated  tool_id=86
```

## What the prompt template does at runtime

1. Mounts wekafs (handled by the Plugin runtime, like Hyperloom).
2. `bash /wekafs/HyperloomV2/framework-agent/scripts/install.sh` -
   `pip install -e` into the container venv.
3. Reads UI panel inputs (`$framework`, `$repoUrl`, `$gapDescription`,
   `$baselineThroughput`, ...) and writes an `ExploreRequest` JSON.
4. Runs `fa explore --request <file> --out <plan_summary.json>` (plan).
5. If `$execute=true`, re-runs with `--execute` (GPU build/bench).
6. Surfaces `winner_ref` + `kb_contribution` to the agent loop.

## UI panel inputs (`selectedConfigKeys`)

| Key | Type | Example | Notes |
|---|---|---|---|
| `framework` | string | `sglang` | one of sglang/vllm/triton |
| `repoUrl` | string | `https://github.com/sgl-project/sglang.git` | GitHub-style URL |
| `gapDescription` | string | `improve sglang fp8 MoE on MI300X` | drives keyword extraction |
| `baselineThroughput` | float | `1.0e8` | tok/s; must be > 0 |
| `baselineAccuracy` | float | `0.95` | optional |
| `maxCandidates` | int | `3` | per-repo limit |
| `kbDomain` | string | `framework` | one of 8 KB domains, empty disables KB write |
| `execute` | bool | `false` | gate the build/bench step |

## Phase 2 (after MVP passes)

| Item | Status |
|---|---|
| Add `type=mcp` tool for an in-cluster MCP LLM endpoint | TBD |
| Implement `ClawHttpBackend` in `framework_agent.kb` so contributions land in claw-memory-service `/api/kb/*` instead of local FS | TBD - see `claw-dev/docs-zh/framework-agent-primus-claw-dev-test.md` §4 |
| Increase `resource.gpu` to 8 when running full `sglang launch_server` benchmark suites | TBD |
| Add CI build + image push pipeline (independent image instead of reusing sglang upstream) | low priority |

## Reference

- Production Hyperloom plugin (id=4) - same shape, replace `inference_optimizer` with `framework-agent`.
- `claw-dev/docs-zh/framework-agent-primus-claw-dev-test.md` - full design & test plan.
- `framework-agent/SKILL.md` - skill contract loaded by the prompt template.
