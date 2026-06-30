# Framework Agent — Sibling Skill

> **Purpose**: vllm/sglang source-layer optimisation companion for
> `inference_optimizer`. It discovers framework / ref candidates
> (via Primus Cortex + GitHub Search), optionally builds/benchmarks
> them in isolated worktrees, and serves the Coordinator's
> FRAMEWORK_AGENT phase. Exposed through the `fa` / `framework-agent`
> console entry points.

## Layout

```
framework-agent/
├── src/framework_agent/        # python package
│   ├── runtime/cli.py          # fa schema / candidates / explore / phase-discover / kb
│   ├── runtime/tools_api.py    # library API behind the CLI verbs
│   ├── explorer.py             # explore loop (libcst-free)
│   ├── isolation.py            # per-candidate worktree + venv
│   ├── decision.py             # 3-gate winner decision
│   ├── kb.py                   # knowledge-base operations
│   ├── repo_map.py             # framework -> canonical repo URL
│   ├── keywords.py / models.py # request models + keyword helpers
│   ├── logging_setup.py        # shared: structured logging
│   └── sources/                # primus_cortex + github discovery
├── kb/
│   └── framework_optimization/ # KB partition (empirical_kb.md / lessons, written at runtime)
├── scripts/install.sh          # idempotent installer
├── scripts/env.sh              # env loader
├── pyproject.toml
└── tests/
```

## Subcommands

```bash
fa schema                 # print the request schema summary (debug)
fa candidates --request req.json [--out -]   # enumerate PR/ref candidates, no build/bench
fa explore   --request req.json [--execute] [--out -]   # plan (default) or build/bench loop
fa phase-discover --request req.json [--out -]   # Coordinator FRAMEWORK_AGENT phase entry point
fa phase-audit --request req.json [--out -]   # static local-source judging of a discovered candidate
fa kb list|show|search|contribute|synthesize ...   # knowledge-base ops
```

`fa candidates` / `fa explore` run a standalone investigation loop (no
IO coordinator). `fa phase-discover` is the thin shim driven by the
Coordinator's per-candidate pump in the FRAMEWORK_AGENT phase (between
PRELUDE and EXPLORE in `inference_optimizer`); it is the **only**
subcommand `inference_optimizer/orchestrator/framework_agent_client.py`
invokes. Don't use `phase-discover` outside that context.

## FRAMEWORK_AGENT phase (`fa phase-discover`)

Reads `--request <json>` and writes `--out <json|->` (envelope style
mirrors `critic-agent/runtime/cli.py`):

```bash
# Discover a batch of PR candidates for the current run's gaps.
#    request: {model, framework, gpu_type, gaps[], repo_url?, max_search_candidates, batch_id}
#    output:  {batch_id, framework, repo_url, candidates: [...]}
fa phase-discover --request req.json --out -
```

## FRAMEWORK semantic audit (`fa phase-audit`)

Given a discovered candidate + the live framework source roots, decide whether
the PR's change is already present locally so the Coordinator can skip
already-merged PRs and seed the authoring specialist with evidence:

```bash
# request: {candidate, framework, framework_source_roots[],
#           diff_text|patches_path|primus_cortex_url, work_dir?, use_llm?, model?}
# output:  {candidate_id, semantic_status, applicability, confidence,
#           evidence[], risks[], recommended_next_step, layer, metrics}
fa phase-audit --request req.json --out -
```

- **static layer (default, hermetic)**: parse the diff, resolve each touched
  file under `framework_source_roots`, measure added-line / symbol presence +
  context-anchor presence.
  - `semantic_status` ∈ `already_equivalent` / `already_superset` /
    `partially_present` / `not_present` / `unknown`.
  - `applicability` ∈ `direct_apply` / `needs_rewrite` / `not_applicable` /
    `needs_human_review`; `recommended_next_step` ∈ `skip` /
    `direct_framework` / `author_via_specialist`.
  - `already_*` is evidence-gated (downgraded to `unknown` without a concrete
    symbol/line hit).
- **llm layer (opt-in, `use_llm=true`)**: single chat-completion refine; needs
  `SAFE_API_KEY` + `OPENAI_BASE_URL`; best-effort; never authors patches.

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

`contribute` appends to `${KB}/<domain>/empirical_kb.md`; the partition
is created on first write (no seed files are shipped).
