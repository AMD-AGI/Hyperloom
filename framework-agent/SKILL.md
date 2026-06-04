# Framework Agent — Sibling Skill

> **Purpose**: vllm/sglang source-layer optimisation companion for
> `inference_optimizer`. Two protocols co-exist in this package:
>
> 1. **`fa candidates` / `fa explore`** — legacy PR exploration tool
>    (PR discovery via Primus Cortex + GitHub Search, applied as
>    bandit-arm imports). Used by `inference_optimizer`'s
>    `framework_pr` arm.
> 2. **`fa agent` (PR-D+)** — new sibling-skill protocol for the 5th
>    Framework role in `inference_optimizer`. Two-stage subprocess
>    bridge (`prepare-task` / `commit-result`) used by
>    `FrameworkAgentBackend`. PR-D ships the skeleton; PR-E adds AST
>    scanner; PR-F wires the IO handler.
>
> Both share the same Python package (`framework_agent`) and
> `fa` / `framework-agent` console entry points.

## Co-existing layouts

```
framework-agent/
├── src/framework_agent/        # python package
│   ├── runtime/cli.py          # legacy: fa candidates / fa explore / fa kb
│   ├── explorer.py             # legacy: explore loop (libcst-free)
│   ├── isolation.py            # legacy: per-candidate worktree+venv
│   ├── decision.py             # legacy: 3-gate winner decision
│   ├── logging_setup.py        # shared: structured logging
│   ├── sources/                # legacy: primus_cortex + github
│   └── agent/                  # NEW (PR-D+): sibling-skill cli + envelope + scanner
├── kb/
│   └── framework_optimization/ # NEW (PR-I): KB partition seeds + lessons
├── scripts/install.sh          # NEW (PR-D): idempotent installer
├── pyproject.toml              # libcst, patch-ng, jsonschema deps land in PR-D
└── tests/                      # legacy + agent/ tests
```

## Lifecycle by PR

| PR | What this skill grows |
|---|---|
| PR-A1/A2/B (IO) | Coordinator side only — handlers + mock backend land in `inference_optimizer`. This skill is unchanged. |
| **PR-D** | `agent/cli.py` (`prepare-task` / `commit-result`), `agent/envelope.py` (jsonschema), `scripts/install.sh`. |
| **PR-E** | `agent/source_resolver.py` + `agent/ast_scanner.py` (libcst) + `agent/grep_scanner.py` (fallback) + `agent/flag_discovery.py`. |
| **PR-F** | `FrameworkAgentBackend` (IO side) wires to `fa agent` via subprocess; `discovered_flags` start flowing into `SharedState`. |
| **PR-G** | `agent/patch_proposer.py` + `agent/kb_priors.py` — real LLM-loop diff generation. |
| **PR-H** | Coordinator-side `framework_integrate_handler` real impl (lives in `inference_optimizer`, not this skill). |
| **PR-I** | `kb/framework_optimization/` seeds + robustness recover wiring. |

## Operation Protocol (PR-D+ vision)

`fa agent` exposes two subcommands matching design §9.1:

```bash
# Stage A: LLM bundle preparation. inputs come from the IO Coordinator
# (target_framework, session_dir, kb_partition, ast_scan_enabled,
# ast_frameworks). Outputs a JSON bundle the LLM (Claude / Codex) consumes.
fa agent prepare-task \
  --task /path/to/task.json \
  --output-bundle /path/to/bundle.json

# Stage B: envelope validation + persistence. LLM-generated envelope is
# validated against the §4.6 jsonschema, persisted to
# runs/framework/<task_id>/envelope.json, and echoed to stdout for the
# Coordinator to consume.
fa agent commit-result \
  --envelope /path/to/envelope.json \
  --task-id <task_id>
```

The four `RESPONSE` envelopes (`OptimizeSuccess` / `OptimizeFailure` /
`IntegrateSuccess` / `IntegrateFailure`) live in
`src/framework_agent/agent/envelope.py` (PR-D).

## Smoke (PR-D+)

```bash
fa agent --help                # rc=0, lists subcommands
fa agent prepare-task --help   # rc=0, lists --task / --output-bundle
fa agent commit-result --help  # rc=0
```

## FRAMEWORK_PR phase subcommand

The Coordinator-side **FRAMEWORK_PR phase** (between PRELUDE and
EXPLORE in `inference_optimizer`) drives the `phase-discover`
subcommand. It reads `--request <json>` and writes `--out <json|->`
(envelope style mirrors `critic-agent/runtime/cli.py`):

```bash
# Discover a batch of PR candidates for the current run's gaps.
#    request: {model, framework, gpu_type, gaps[], repo_url?, max_search_candidates, batch_id}
#    output:  {batch_id, framework, repo_url, candidates: [...]}
fa phase-discover --request req.json --out -
```

Distinction from `fa explore` / `fa candidates`: those run a
standalone investigation loop (no IO coordinator); `fa phase-*` are
thin shims driven by IO's per-candidate pump in the FRAMEWORK_PR
phase. Don't use them outside that context.

## KB partition (PR-I)

Read priors before generating a patch; write lessons only after a KEEP
verdict from `framework_integrate`. Path:
`kb/framework_optimization/`. Seeds shipped in PR-I cover 8 known
patterns (vllm chunked prefill, sglang radix tree, PagedAttention
block size, sampler / KV quant / FP8 boundaries, block_manager OOM
pitfall, scheduler.py token loss pitfall).
