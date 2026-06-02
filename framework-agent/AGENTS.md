# Hosting the Framework Agent skill

The Framework Agent does not need a dedicated long-running service. The
intended delivery is one of:

## Mode 1 - Standalone CLI on a GPU node

Used by operators and CI pipelines. No host server needed:

    fa explore --request request.json --out summary.json --execute

## Mode 2 - Library mode inside an Arbor/TBO specialist

When Arbor's orchestrator dispatches a framework-related specialist, the
specialist prompt should include:

    """
    You have access to the framework-agent toolkit:

    from framework_agent.runtime.tools_api import (
        find_relevant_prs_smart,
        fetch_pr_audit_material,
        evaluate_candidate_outcome,
    )

    Use `find_relevant_prs_smart(gap_description, repos)` to discover
    candidate PRs from both Primus Cortex (internal) and GitHub (public).
    """

The specialist runtime (Claude/Codex subprocess) imports the module and
calls the helpers as regular Python functions. No serialization
boundary, no IPC.

## Mode 3 - Codex/Claude A2A chat server (future, not in current scope)

For uniformity with critic-agent style, future versions may host the
framework-agent skill inside an A2A chat server that exposes two
bash commands: `fa explore --request` and `fa kb synthesize`. Not
implemented in current scope.

## Required environment

| Variable | Purpose | Default |
|---|---|---|
| `WORKSPACE_PATH` | Root for the skill files | `/workspace` |
| `FRAMEWORK_AGENT_ROOT` | Root of this skill | `${WORKSPACE_PATH}/framework-agent` |
| `FRAMEWORK_AGENT_KB_DIR` | KB write target | `${FRAMEWORK_AGENT_ROOT}/kb` |
| `PRIMUS_CORTEX_PR_API` | Primus Cortex base URL fallback | unset (CLI flag wins) |
| `GITHUB_TOKEN` / `GH_TOKEN` | GitHub auth | unset (anonymous fallback) |
| `FRAMEWORK_AGENT_LOG_LEVEL` | logging level | INFO |
| `FRAMEWORK_AGENT_DISK_MIN_GB` | disk_preflight threshold | 20 |

## What framework-agent does NOT need

- `SAFE_API_KEY` / `OPENAI_BASE_URL` are not read by this package.
- No NATS / PostgreSQL / Ray / TraceLens / GEAK / OOB dependency.
- No coupling with kernel-agent / critic-agent / robustness-agent.

LLM credentials, if used by callers running in Library mode, are owned
by the outer runtime (Arbor / TBO / etc.), not by this package.
