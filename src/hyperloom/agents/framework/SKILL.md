# Framework Agent — Sibling Skill

> **Purpose**: vllm/sglang source-layer optimisation companion for
> `inference_optimizer`. It discovers framework / ref candidates
> (via Primus Cortex + GitHub Search), optionally builds/benchmarks
> them in isolated worktrees, and serves the Coordinator's
> FRAMEWORK_AGENT phase. Exposed through the `fa` / `framework-agent`
> console entry points.

## Layout

This skill lives in the `hyperloom` src-layout distribution:

```
src/hyperloom/agents/framework/     # hyperloom.agents.framework
├── runtime/cli.py                 # fa schema / candidates / explore / kb
├── runtime/tools_api.py           # library API behind the CLI verbs
├── explorer.py                    # explore loop (libcst-free)
├── isolation.py                   # per-candidate worktree + venv
├── decision.py                    # 3-gate winner decision
├── kb.py                          # knowledge-base operations
├── repo_map.py                    # framework -> canonical repo URL
├── keywords.py / models.py        # request models + keyword helpers
├── logging_setup.py               # shared: structured logging
├── sources/                       # primus_cortex + github discovery
└── tests/
```

The `fa` CLI ships with the distribution, so there is no separate installer
for this skill: `pip install -e '.[test]'` from the repo root provides it.

The KB a session reads and writes is `INFERENCE_OPTIMIZER_FA_KB_PATH` when
set, else `<workspace>/framework-kb` (`USER_DATA_PATH` or the pod-local
default). It is deliberately not `<workspace>/kb`, the legacy recipe root:
every directory under this root is reported as a framework domain. The
orchestrator's writeback resolves the same root, so both halves move together.
The runtime partition written under it is `framework_optimization/<framework>/`.
Read-only seed data shipped in the wheel lives separately under the package.

## Subcommands

```bash
fa schema                 # print the request schema summary (debug)
fa candidates --request req.json [--out -]   # enumerate PR/ref candidates, no build/bench
fa explore   --request req.json [--execute] [--out -]   # plan (default) or build/bench loop
fa kb list|show|search|contribute|synthesize ...   # knowledge-base ops
```

`fa candidates` / `fa explore` run a standalone investigation loop (no
IO coordinator). This agent has no Coordinator entry point: candidate
discovery is a specialist, and its verdicts arrive in the deliverable rather
than from a per-candidate audit call.

## Candidate refs feed the targeted build

A discovered candidate reference now drives the enablement targeted build
(compiled-component acquisition), not just the git-apply/bench path:

- A candidate is resolved to a checkoutable ref — a PR reference becomes that
  PR's head ref — so support that only exists in an unreleased PR/branch is
  reachable, not just released-tag autoselect.
- The source PR URL is recorded as build provenance and surfaces in the session
  breakdown's `build_attempts[].installed_versions` as `source_pr_url`.

## Enablement ladder (methodology)

When a candidate is for enablement (making a `(model, backend)` combo that is
non-runnable, or boots but fails its accuracy eval, run correctly — not perf),
the repair follows a tiered ladder — diagnose the
missing capability layer once, then climb only as far as needed: Rung 0
diagnose, 1 serve-flag/config wire-up, 2 in-tree source patch, 3 attempt-scoped
runtime, 4 source localization, 5 off-loop compiled build. A supported-but-un-wired
model needs only the cheap top rungs; a genuinely-new architecture climbs higher.
The canonical rendered text is `build_enablement_ladder_book` in
`hyperloom.agents.framework.enablement_ops` (injected into the enablement
authoring specialist's prompt); see also `docs/conceptual/optimization-loop.md`
("Enablement escalation ladder").

## KB partition (`fa kb`)

Read priors before generating a patch; write lessons only after a KEEP
verdict. Operations all live under the `framework_optimization` domain:

```bash
fa kb list                                  # list available KB domains
fa kb show --domain framework_optimization  # show files in a domain
fa kb search --query "chunked prefill"      # case-insensitive content search
fa kb contribute --domain framework_optimization --body-file finding.md
fa kb synthesize --domain framework_optimization --findings findings.json [--with-llm]
```

`contribute` appends to `${KB}/<domain>/empirical_kb.md`, creating the domain
dir and file if missing. The `framework_optimization` partition is created on first contribution.
