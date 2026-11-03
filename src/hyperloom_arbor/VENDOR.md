# Arbor, vendored

`src/hyperloom_arbor/` is a copy of Arbor, AMD's autonomous inference-optimization harness: the agent loop that
multi-branch (MB) runs as a component when a session is given `--autonomous-agent` (`docs/hosts.md` here, and
`docs/conceptual/multibranch-arbor.md` in this repository).

| | |
| --- | --- |
| Source | `AMD-AGI/Arbor` (internal) |
| Tag | `v0.9.1` |
| Commit | `123cd15` |
| Files | 303 |
| Licence | Apache-2.0, [`LICENSE`](LICENSE) (`REUSE.toml` at the repository root says so per file) |

## What it holds

The engine and what its agents read:

- `arbor/`, the harness: the loop, the gates, component mode, the dossier and stage machinery;
- `.claude/skills/`, the skills its agents use (the dossier, the profiler, kernels, hardware, the serving playbook's
  rules and knob catalog, and the rest). The profiler skill is this copy's own: it profiles with TraceLens (below);
- `packs/serving` (the template MB names by default), `packs/agentx`, `packs/_shared`, `packs/parallel-dossier` and
  the `evolve-asm` packs;
- `docs/`, the operator and host documentation; `evals/`, the session tracer the skills' hooks call; `scripts/pack/`;
  `configs/serving-template.json`.

## What was left out or changed

- **Left out:** Arbor's tests (they run upstream at the tag), its campaign results, its workload-specific packs,
  configs and dossiers, and its cluster launchers. Code or documents that name one of them keep the name; the file is
  not here.
  - **The serving playbook keeps its rules and its knob catalog only.**
    - **Left out:** the InferenceX study's evidence corpus, and the five reference pages generated from it (big
      directions, regime tables, AMD knobs, numerics knobs, the image manifest). They carry other vendors' recipes
      and figures, and quantified results.
    - **Its `SKILL.md`** is this copy's own: the playbook's rules, and what the catalog holds.
    - **`knobs.json`,** which the harness reads (`arbor/knobs.py`), stays, without its entries for another vendor's
      GPUs. Its entries still cite a page section and a study record id (`#655`); neither is carried in this copy.
    - **What the agent is told** points to what the copy carries: the dossier's template has no section built from
      the pages, and the dossier skill, the agents' program, the `serving` pack, the always-on MI355X knowledge and
      the hardware skill name `knobs.json` and the playbook's rules instead. Code comments and the kernels skill's
      citations keep a page's name, under the convention above.
- **No competitors' figures:** a figure that sets another vendor's GPU or library against a number is taken out:
  - in the `evolve-asm` pack's MoE context;
  - in the dossier skill;
  - in a KernelForge page on HipKittens.

  AMD's own figures stay. The vendoring tool refuses a copy that still has one.
- **Replaced by placeholders:** the names of internal clusters, hosts, paths, accounts, people, personal GitHub
  accounts and repositories, and of workloads or chips that are not public.
  - **The placeholders:** `Cluster A`, `Cluster B`, `<node>`, `<login node>`, `<partition>`, `<account>`, `/data/...`,
    `/shared/...`, `<fork>`, `<colleague>`, `Model M`, `Model W` and the like.
  - **Renamed identifiers:** an identifier that carried such a name was renamed the same way everywhere in this copy
    (for example `CAMPAIGN_CONFIG` for a launcher variable), so the copy is consistent with itself.
  - **GitHub owners:** a GitHub owner is named only when it is an organisation; the vendoring tool refuses a copy that
    names any other.
- **Profiling on TraceLens:** upstream, Arbor's profiler skill traces with a tracer of its own, which is not released
  here. In this copy the skill (`.claude/skills/profiler/SKILL.md`) profiles with TraceLens instead, as this repository
  does:
  - the host's profile first (`component.profile`: MB's PRELUDE TraceLens profile of the root);
  - then, for what that does not cover, a PyTorch-profiler trace of the served program, read with
    `TraceLens_generate_perf_report_pytorch_inference`.

  Every mention of the other tracer, in the agents' program (`arbor/pack/render.py`), the `serving` pack, the dossier
  skill and the docs, says TraceLens instead.
- **KernelForge's examples are this repository's:** the four examples under `.claude/skills/kernels/reference/examples/`
  are taken from `src/kernelforge/examples`, which runs `kernelforge forge-loop --kernel-backend ...`. Arbor's own
  copies predate that rename. KernelForge's knowledge pages here use the same names.
- **One permission:** the round-end A/B's output directory is created for the group only (`0o770`) instead of
  world-writable (`arbor/rounds.py`). The round-end A/B never runs under a host.

## What it carries from elsewhere

**KernelForge pages,** in `.claude/skills/kernels/reference/` and `.claude/skills/hardware/reference/`. They are MIT, as
in `src/kernelforge`, except two kernels, each annotated as `src/kernelforge`'s copy of it is:
- a FlyDSL reference kernel (Apache-2.0, FlyDSL Project Contributors);
- an SGLang Triton kernel (Apache-2.0, SGLang Team and contributors).

## How it is used

Arbor runs from its own root: `python -m arbor run --config <config>` with this directory as the working directory
(`README.md`, `docs/getting-started.md`). It is not a Python package of this repository, so the wheel does not ship it
(`pyproject.toml` excludes `hyperloom_arbor` from the packages). MB starts it with `--autonomous-agent arbor`, which
names this tree.

Its `install.sh` pins TraceLens for a standalone Arbor. Under MB, Arbor uses the TraceLens this repository's own
install pins.

## How it is updated

Never edit it here: a change is made in Arbor and the copy is taken again, whole, from a new tag, by the Arbor
maintainers' vendoring tool, with the same omissions, placeholders, profiler skill and examples. The tool refuses a
copy that still names any of the replaced terms.

Linting, formatting and code metrics skip this tree, as for the other vendored code (`pyproject.toml`: `[tool.ruff]`,
`[tool.hyperloom.code_metrics]`, `[tool.bandit]`; `.pre-commit-config.yaml`'s formatting and lint hooks). Secret
scanning (`detect-private-key`, `gitleaks`) and the licence check (`reuse`) do not.
