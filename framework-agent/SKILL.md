---
name: framework-agent
description: |
  Explores serving framework PRs and refs for Hyperloom inference runs.
  Discovers candidate PRs from Primus Cortex (internal) + GitHub (public),
  runs them in isolated git worktrees + venvs, and contributes findings
  back to a 4-file KB. Operates standalone - does NOT integrate with
  inference_optimizer's 5-role mesh. Use when testing vLLM, SGLang, ROCm
  fork, or upstream performance PRs before handing a winner to operator.
globs:
  - "**/framework*"
  - "**/vllm*"
  - "**/sglang*"
  - "**/scheduler*"
  - "**/engine_args*"
---

# Framework Agent

Framework Agent is a **standalone** PR/ref exploration agent for serving
frameworks. It does NOT bring up broken model/backend triplets, does NOT
patch vendor files, does NOT optimize kernels. It does NOT mutate the
active environment; promotion is manual-only.

## Architecture choice

This skill is **NOT** part of the inference_optimizer 5-role mesh. It is
a sibling skill that can be invoked by:

1. Operators (via `fa explore --request request.json`)
2. CI pipelines (same CLI)
3. LLM specialists in Arbor/TBO/Hyperloom (via `from framework_agent.runtime.tools_api import ...`)

Compared to inference_optimizer's kernel-agent / critic-agent /
robustness-agent, framework-agent does NOT emit Coordinator intents,
does NOT participate in REQUEST/RESPONSE protocol, does NOT have a
PolicyGate-enforced role. It's a pure tool, not a protocol agent.

## Setup

This skill is two commands:

    export REPO_ROOT="$(pwd)"
    bash $WORKSPACE_PATH/framework-agent/scripts/install.sh
    . $WORKSPACE_PATH/framework-agent/runtime/env.sh

`install.sh` is idempotent and prepares:

- `pip install -e .` of this package
- optional `ast` extra can install libcst for future AST scanning; base install does not need it
- detect /sgl-workspace/{vllm,sglang} availability (WARN-only)
- inherit auth aliases (GITHUB_TOKEN, PRIMUS_CORTEX_PR_API)
- write framework-agent.env.sh with FRAMEWORK_AGENT_ROOT, default KB dir

## CLI subcommands

| Subcommand | Purpose | Reads | Writes |
|---|---|---|---|
| `fa explore` | Full pipeline: enumerate -> enrich -> filter -> isolate -> execute -> decide -> kb | request.json | summary.json + work_dir/ |
| `fa candidates` | Enumerate + filter only (no execute) | request.json | candidates.json |
| `fa schema` | Print request schema summary | - | stdout |
| `fa kb {list,show,search,synthesize}` | KB operations | - | various |

Use `--execute` only when the request contains trusted command templates
and the node is ready for GPU work.

## Allowed tools (per-stage)

| Stage | Tools |
|---|---|
| keyword extraction | Read (local), pure-Python |
| enumeration | urllib (Primus Cortex + GitHub API) |
| enrichment | urllib (Primus Cortex pr_get + pr_files) |
| filter | pure-Python |
| audit dump | urllib + file write |
| isolation | git, python -m venv |
| execute | Bash (template render + subprocess) |
| decide | pure-Python |
| kb | file append |

## Output

A single `summary.json` per `explore` run (schema in references/).

`work_dir/` layout (one dir per candidate, with per-stage logs + audit
material + KB partition writes).

## Operation Protocol (for LLM specialists)

If invoked via `tools_api.py`:

    from framework_agent.runtime.tools_api import (
        find_relevant_prs_smart,
        fetch_pr_audit_material,
        evaluate_candidate_outcome,
    )

LLM may freely call these - they wrap the same internal modules as the
CLI. No two-phase prepare/commit dance is required (unlike critic-agent).

## Safety Rules

- Never edit `inference_optimizer/` from this agent.
- Never install candidate packages into the main environment.
- Each candidate gets its own git worktree and virtualenv.
- Candidate promotion is manual-only.
- A winner must pass throughput + accuracy gates before `winner=true`.
- Primus Cortex transport / parse error is a hard fail, NOT a silent fallback.

## Failure handling

| Symptom | Recovery |
|---|---|
| FRAMEWORK_AGENT_ROOT not set | install.sh re-run, source env.sh |
| Primus Cortex unreachable when configured | rc=2 hard-fail |
| GitHub rate-limit (best-effort path) | return empty, log WARN, continue |
| Single candidate git fetch fails | per-candidate skip, continue rest |
| disk_preflight: < 20 GiB free | rc=2 hard-fail |
| AST scanner dependency missing | base install does not include AST; install `.[ast]` to enable |

## KB partitions

This skill writes to `${FRAMEWORK_AGENT_KB_DIR}/<domain>/empirical_kb.md`.
Default domain mapping: see references/kb_4file_schema.md (delivered in a
follow-up PR).

Ingest happens at end of every `fa explore` run, including failures
(category=pitfall when revert).
