# Arbor, vendored

`src/hyperloom_arbor/` is a copy of Arbor, AMD's autonomous inference-optimization harness: the agent loop that
multi-branch (MB) runs as a component when a session is given `--autonomous-agent` (`docs/hosts.md` here, and
`docs/conceptual/multibranch-arbor.md` in this repository).

| | |
| --- | --- |
| Source | `AMD-AGI/Arbor` (internal) |
| Tag | `v0.9.1` |
| Commit | `123cd15` |
| Files | 310 |
| Licence | Apache-2.0, [`LICENSE`](LICENSE) (`REUSE.toml` at the repository root says so per file) |

## What it holds

The engine and what its agents read:

- `arbor/`, the harness: the loop, the gates, component mode, the dossier and stage machinery;
- `.claude/skills/`, the skills its agents use (the dossier, the profiler, kernels, hardware, the serving playbook and
  the rest). The profiler skill is this copy's own: it profiles with TraceLens (below);
- `packs/serving` (the template MB names by default), `packs/agentx`, `packs/_shared`, `packs/parallel-dossier` and
  the `evolve-asm` packs;
- `docs/`, the operator and host documentation; `evals/`, the session tracer the skills' hooks call; `scripts/pack/`;
  `configs/serving-template.json`.

## What was left out or changed

- **Left out:** Arbor's tests (they run upstream at the tag), its campaign results, its workload-specific packs,
  configs and dossiers, and its cluster launchers. Code or documents that name one of them keep the name; the file is
  not here.
- **Replaced by placeholders:** the names of internal clusters, hosts, paths, accounts, people and repositories, and of
  workloads that are not public: `Cluster A`, `Cluster B`, `<node>`, `<login node>`, `<partition>`, `<account>`,
  `/data/...`, `/shared/...`, `<fork>`, `<colleague>`, `Model M`, `Model W` and the like. Identifiers that carried such
  a name were renamed the same way everywhere in this copy (for example `CAMPAIGN_CONFIG` for a launcher variable), so
  the copy is consistent with itself.
- **Profiling on TraceLens:** upstream, Arbor's profiler skill traces with a tracer of its own, which is not released
  here. In this copy the skill (`.claude/skills/profiler/SKILL.md`) profiles with TraceLens instead, as this repository
  does:
  - the host's profile first (`component.profile`: MB's PRELUDE TraceLens profile of the root);
  - then, for what that does not cover, a PyTorch-profiler trace of the served program, read with
    `TraceLens_generate_perf_report_pytorch_inference`.

  Every mention of the other tracer, in the agents' program (`arbor/pack/render.py`), the `serving` pack, the dossier
  skill and the docs, says TraceLens instead.

## What it carries from elsewhere

**KernelForge pages,** in `.claude/skills/kernels/reference/` and `.claude/skills/hardware/reference/` (MIT, as
`src/kernelforge`), among them one FlyDSL reference kernel (Apache-2.0, FlyDSL Project Contributors).

## How it is used

Arbor runs from its own root: `python -m arbor run --config <config>` with this directory as the working directory
(`README.md`, `docs/getting-started.md`). It is not a Python package of this repository, so the wheel does not ship it
(`pyproject.toml` excludes `hyperloom_arbor` from the packages). MB's `--autonomous-agent` still takes a git source and
clones it; starting an Arbor from this tree is a change to MB of its own.

## How it is updated

Never edit it here: a change is made in Arbor and the copy is taken again, whole, from a new tag, with the same
omissions, placeholders and TraceLens profiler skill, by the Arbor maintainers' vendoring tool. The tool refuses a copy that still names any of
the replaced terms. Linting, formatting and code metrics skip this tree (`pyproject.toml`: `[tool.ruff]`,
`[tool.hyperloom.code_metrics]`, `[tool.bandit]`), as for the other vendored code; secret scanning and the licence check
do not.
