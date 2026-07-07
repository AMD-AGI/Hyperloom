# Hosting the Framework Agent skill

The Framework Agent does not need a dedicated long-running service. The
intended delivery is one of:

## Mode 1 - Standalone CLI on a GPU node

Used by operators and CI pipelines. No host server needed:

    fa explore --request request.json --out summary.json --execute

## Mode 2 - Library mode inside a Hyperloom specialist

When the Hyperloom Orchestrator (the system published as Arbor,
arXiv:2606.12563) dispatches a framework-related specialist, the
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

- No NATS / PostgreSQL / Ray / TraceLens / GEAK / OOB dependency.
- No coupling with kernel-agent / critic-agent / robustness-agent.

LLM credentials, if used by callers running in Library mode, are owned
by the outer runtime (Hyperloom / TBO / etc.), not by this package.

## Two objectives: perf optimisation vs enablement

The package serves two distinct objectives, gated differently:

- **Perf** — discover / audit / apply an existing PR and KEEP it
  only if throughput improves. Patch authoring for perf is owned by the
  Hyperloom `specialist → integrate_patch` path; this package does the
  discovery + static audit + git-apply/bench execution.
- **Enablement** (opt-in) — make a currently **non-runnable**
  `(model, backend)` combo *run at all*. This path DOES author bridging
  patches (via the `enablement_specialist` domain / `SpecialistRunner`
  worktree authoring) and is gated on **runnability** (server boots + minimal
  correctness), not throughput. Pure, GPU-free building blocks live in this
  package:
  - `framework_agent.enablement` — failure-signature classifier
    (`classify_failure`) + `EnablementRequest` + the `runnable_decision` gate.
  - `framework_agent.enablement_discovery` — bridging-repo selection
    (framework + ROCm/HIP/aiter via `repo_map.bridge_repo_urls`) and
    enablement-intent ranking.
  - `framework_agent.enablement_authoring` — the authoring `EnablementMandate`
    (allowed source roots + task description + patch invariants) handed to the
    specialist runner.

  Editing ROCm/HIP source (`/opt/rocm`) is a first-class, **default-on** part
  of the enablement path: `orchestrator/framework_paths.resolve_rocm_hip_source_roots`
  is always merged into the IO-side allowlist (alongside the always-allowed
  `aiter`). Authored patches must still pass `git apply --check` grounding and
  the forbidden-numeric-claim guard.

### Optional LLM use (`fa phase-audit --request ... use_llm=true`)

`fa phase-audit` runs a deterministic **static** local-source judge by
default (no network, no LLM). Its optional refine layer is **opt-in**
(`request.use_llm=true`): a single chat-completion that reads `SAFE_API_KEY`
+ `OPENAI_BASE_URL` (or request overrides). It is best-effort and
evidence-gated — any missing credential / failure / invalid output keeps the
static verdict, and it can never upgrade to an `already_*` status the static
layer didn't already back with concrete evidence. This is a single bounded
call for *judging*; it is **not** an agentic authoring backend.
