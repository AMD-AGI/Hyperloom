# Arbor, vendored

`src/hyperloom_arbor/` is a copy of Arbor, AMD's autonomous inference-optimization harness: the agent loop that
multi-branch (MB) runs as a component when a session is given `--autonomous-agent` (`docs/hosts.md` here, and
`docs/conceptual/multibranch-arbor.md` in this repository).

| | |
| --- | --- |
| Source | `AMD-AGI/Arbor` (internal) |
| Tag | `v0.9.8` |
| Commit | `f212239` |
| Files | 373 |
| Licence | Apache-2.0, [`LICENSE`](LICENSE) (`REUSE.toml` at the repository root says so per file) |

## What it holds

The engine and what its agents read:

- `arbor/`, the harness: the loop, the gates, component mode, the dossier and stage machinery;
- `.claude/skills/`, the skills its agents use (the flat dossier as `dossier-v1`, the profiler, kernels, hardware,
  the serving playbook's rules and knob catalog, and the rest). The profiler skill is this copy's own: it profiles with
  TraceLens (below);
- the skill families a pack loads (`lifecycle.load`, `docs/skill-integration.md`), each in `.claude/skills/<family>/`
  with its `family.json`: `dossier` (the dossier split into five skills), `benchmark` (the benchmark studied once)
  and `pack` (drafting a pack). Up to `v0.9.2` they were under `plugins/`;
- `packs/serving` (the template MB names by default), `packs/agentx`, `packs/_shared`, `packs/parallel-dossier` and
  the `evolve-asm` packs. From `v0.9.8`, `packs/serving-qwen`: the serving template for a model asked through its chat
  template, its correctness check running GSM8K through `/v1/chat/completions`;
- `docs/`, the operator and host documentation; `evals/`, the session tracer the skills' hooks call; `scripts/pack/`;
  `configs/serving-template.json`;
- from `v0.9.5`, `docs/cli.md` (every command's parameters) and `docs/schema/arbor-cli-payload.schema.json` (the JSON
  Schema of a command's `--payload`), both generated from the commands' parsers by `python3 -m arbor schema --write`.
  The copy's parsers give the same two files, so the scrub changes no parameter's help text.
- from `v0.9.8`, `docs/interface.md`: the host contract (`arbor.host/2`), generated from the code by
  `python3 -m arbor interface --write`, with its records' JSON Schemas in `docs/schema/host/`; and
  `docs/quickstart-checkpoints.md` (pause, continue, move and fork a campaign).

## What was left out or changed

- **Left out:** Arbor's tests (they run upstream at the tag), its campaign results, its workload-specific packs,
  configs and dossiers, and its cluster launchers. Code or documents that name one of them keep the name; the file is
  not here.
  - **Arbor's own term guard** (`arbor/oss_guard.py`, which only its tests run) is left out with the tests.
  - **Skills for developing Arbor** (`.claude/skills/arbor-dev`, the `develop` mode) are left out: a campaign never
    loads them, and this copy is not edited here. The two skill docs say so where they would describe them.
  - **Knowledge search** (`arbor/knowledge/` and its `knowledge` skill family) is left out: its registry names a source
    of another vendor's kernels, and its evaluation set names internal workloads. The code that would use it imports it
    only when a pack loads `knowledge:search`, which no pack here does.
  - **Architecture research** (the `architecture` family, `architecture:performance`, and its guide) is left out: it
    studies how other vendors' frameworks implement a model, on their GPUs. A pack loads it only by naming it, which
    no pack here does.
  - **The dossier's part 9** is left out. It holds a reference platform's traces, and the reference platform is
    another vendor's GPU. The `dossier:reference` skill goes, as do the part's row in `arbor/dossier/parts.py` and every
    step that reads it. The part was optional, so a dossier is whole without it.
  - **A hipBLASLt tuning skill,** built on one internal study, is left out.
  - **The KB exporter's field table** (`docs/kb/arbor-v1.md`, from `v0.9.8`) is left out: its worked example is a
    record of an internal campaign. The exporter's declarations and code keep their rules without pointing to it.
  - **Records of internal campaigns** are left out: the barren/learn design notes, the config index, a campaign
    archive skill, two notes about one model's profile and MoE, and two one-shot campaign scripts under
    `scripts/pack/`.
  - **The serving playbook keeps its rules and its knob catalog only.**
    - **Left out:** the InferenceX study's evidence corpus, and the five reference pages generated from it (big
      directions, regime tables, AMD knobs, numerics knobs, the image manifest). They carry other vendors' recipes
      and figures, and quantified results.
    - **Its `SKILL.md`** is this copy's own: the playbook's rules, and what the catalog holds.
    - **`knobs.json`,** which the harness reads (`arbor/knobs.py`), stays, without its entries for another vendor's
      GPUs. Its entries still cite the page section they came from, which is not carried in this copy.
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
  - **Renamed identifiers:** an identifier that carried such a name was renamed the same way everywhere in this copy,
    so the copy is consistent with itself.
  - **GitHub owners:** a GitHub owner is named only when it is an organisation; the vendoring tool refuses a copy that
    names any other.
- **No internal tools:** an internal scheduler's job-id variable is not read.
- **Our packs and campaign ids are not named:**
  - **The pack-drafting skill** lists only the parents this copy carries.
  - **The collector's project examples** are model names.
- **Gone from Arbor's own tree since `v0.9.4`,** so no longer changed here:
  - **The internal profiler:** the AgentX pack profiles with rocprofv3, and `arbor/profiling/vllm_driver.py` reads
    `ARBOR_PROFILE_*`.
  - **Site values in the engine:** the parallel mode has no launcher defaults (a run config names `cluster.lane_script`
    and `cluster.ab_script`, which take `CAMPAIGN_CONFIG` and `CAMPAIGN_ARBOR`), and the collectors have no share root
    (a site sets `ARBOR_SHARE_ROOT`). Both are a site's settings (below).
  - **The permission bypass** in the project settings (below).
- **Defects by their class:** two AMD software defects that Arbor's text tells as found in its runs are described by
  their kind only.
- **The project settings have no permission bypass** (`.claude/settings.json`; Arbor's own since `v0.9.4`). MB starts
  its agents with its own permission mode. The `evolve-asm` orchestrators still pass one to the agent SDK, since they run their agents
  unattended; MB does not run them.
- **No internal references:** Arbor's code and docs cite the board topic, comment or ticket where a decision was made.
  This copy keeps the decision and drops the citation.
  - **Also dropped:** internal plans, branches, sessions and decision records, roadmap ids, and study record ids.
  - **The private repository:** this copy carries no access or clone instructions for it. The README, getting started
    and the skills guide point to this copy instead.
- **No results or stories of internal campaigns:**
  - **Arbor's own tree** has cleared most of them from its code, skills and docs since `v0.9.3`, and the `benchmark`
    family's `contract.md` is Arbor's own text.
  - **The rest** are cut to the rule they taught: worked examples drawn from our campaigns, our runs' figures, and
    our own runs told as stories.
  - **The vendoring tool** still refuses a copy that has one.
- **`CONTRIBUTING.md`** is this copy's own: it says where a change to Arbor is made.
- **Profiling on TraceLens:** upstream, Arbor's profiler skill traces with a tracer of its own, which is not released
  here. In this copy the skill (`.claude/skills/profiler/SKILL.md`) profiles with TraceLens instead, as this repository
  does:
  - the host's profile first (`component.profile`: MB's PRELUDE TraceLens profile of the root);
  - then, for what that does not cover, a PyTorch-profiler trace of the served program, read with
    `TraceLens_generate_perf_report_pytorch_inference`.

  Every mention of the other tracer, in the agents' program (`arbor/pack/render.py`), the `serving` and `serving-qwen`
  packs, the dossier skill and the docs, says TraceLens instead. From `v0.9.3` to `v0.9.7`, the `serving` pack's
  profiling rule and its dossier steps still described the other tracer, and told the agent not to use the PyTorch
  profiler (fixed in `v0.9.7` by #2640); the vendoring tool now refuses that wording.
- **KernelForge's examples are this repository's:** the four examples under `.claude/skills/kernels/reference/examples/`
  are taken from `src/kernelforge/examples`, which runs `kernelforge forge-loop --kernel-backend ...`. Arbor's own
  copies predate that rename. KernelForge's knowledge pages here use the same names.
- **One permission:** the round-end A/B's output directory is created for its owner only (`0o700`) instead of
  world-writable (`arbor/rounds.py`). The round-end A/B never runs under a host.

## What it carries from elsewhere

**KernelForge pages,** in `.claude/skills/kernels/reference/` and `.claude/skills/hardware/reference/`. They are MIT, as
in `src/kernelforge`, except two kernels, each annotated as `src/kernelforge`'s copy of it is:
- a FlyDSL reference kernel (Apache-2.0, FlyDSL Project Contributors);
- an SGLang Triton kernel (Apache-2.0, SGLang Team and contributors).

From `v0.9.3`, every fact in the hardware cards cites a public AMD source (an ISA guide, a white paper, a data sheet,
the ROCm documentation), and what had none was cut.

**This repository's weight-precision check,** in `arbor/measure/weight_precision.py`: a copy of
`src/hyperloom_multibranch/toolbox/weight_precision.py`, unedited but for its two imports, which point at
`arbor/measure/_hyperloom_shim.py` (`REMOVED`, a two-field `LaunchConfig`, and the server-argument tokenizer copied from
`src/hyperloom/inference_optimizer/grid_server_args.py`). Arbor refuses a weight-precision change before its server
starts, as the toolbox's acceptance does, with the same reading of the flags. MIT, as this repository.

## How it is used

Arbor runs from its own root: `python -m arbor run --config <config>` with this directory as the working directory
(`README.md`, `docs/getting-started.md`). It is not a Python package of this repository, so the wheel does not ship it
(`pyproject.toml` excludes `hyperloom_arbor` from the packages). MB starts it with `--autonomous-agent arbor`, which
names this tree.

Its `install.sh` pins TraceLens for a standalone Arbor. Under MB, Arbor uses the TraceLens this repository's own
install pins.

**A site's settings** (from `v0.9.4`): when it is imported, Arbor reads `~/.arbor/settings.env`, or the file
`ARBOR_SETTINGS` names, and sets each `ARBOR_*` value there that the environment does not set already; other keys are
ignored. A site's own collector projects go in the file `ARBOR_PROJECTS` names. A machine without the file runs as
before. Under MB, no user's file is read:
- **Every Arbor command MB runs** has `ARBOR_SETTINGS=/dev/null` (`specialist/arbor_adapter.py`): the sessions, through
  `ARBOR_PYTHON`, and the directions commands (`generate`, `split`), through `_directions`. So a local, unconfined run
  does not pick up the running user's values (a roll, a review gate, an agent backend). An empty value would not turn
  the file off.
- **Hosted,** each branch also has its own home, and its confined environment carries no `ARBOR_*` value.

## How it is updated

Never edit it here: a change is made in Arbor and the copy is taken again, whole, from a new tag, by the Arbor
maintainers' vendoring tool, with the same omissions, placeholders, profiler skill and examples. The tool refuses a
copy that still names any of the replaced terms, or cites an internal board or ticket.

This copy also carries a scrub on top of the tag. The vendoring tool applies it as a patch over its own output and
refuses a tag the patch no longer fits, so a re-vendor keeps it.
- **Up to `v0.9.2`, the patch removed:** internal board, job and plan references, operator decision logs, measured
  performance figures and the history of internal runs.
- **From `v0.9.3`, Arbor's own tree carries that cleanup.** The patch carries what this repository's public-repository
  rule needs beyond it: part 9, the internal scheduler, our packs' names, the two defects, and the remaining stories
  and figures of internal runs.
- **From `v0.9.4`,** the internal profiler, the site values and the permission bypass are gone from Arbor's own tree
  too, and the patch no longer carries them.
- **From `v0.9.7` on `main`,** the patch also carries the scrub this repository's review made on the copy: stories
  and figures of internal runs, an internal config's name, and the dossier's references to the part it leaves out.

The tool also refuses a copy that still has any of these:
- a job id, a dated operator log, an internal plan or a measured step time;
- in Arbor's own text: a date (bar timestamp examples), a run, round, generation or iteration id, or a run's counts,
  times or figures;
- from `v0.9.3`: the internal profiler or scheduler, part 9, the permission bypass, the defect's kernel, the study
  skill, or our packs' and campaigns' names.

Linting, formatting and code metrics skip this tree, as for the other vendored code (`pyproject.toml`: `[tool.ruff]`,
`[tool.hyperloom.code_metrics]`, `[tool.bandit]`; `.pre-commit-config.yaml`'s formatting and lint hooks). Secret
scanning (`detect-private-key`, `gitleaks`) and the licence check (`reuse`) do not.
