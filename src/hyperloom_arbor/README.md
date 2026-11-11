# Arbor

Arbor optimizes LLM inference serving on AMD Instinct GPUs with agents. A campaign is a loop: an
agent changes the serving tree (vLLM or SGLang: flags, environment, Python, kernels), the harness
launches the server, benchmarks it and checks correctness, and a gate keeps the change or reverts
it. Every kept change is recorded as a patch that applies to a fresh tree, with what the agents
learned along the way.

## Ways to run it

- **Standalone:** Arbor owns a GPU node, its container and its clock. Launchers provision the node
  and start the driver; on a cluster with a short job limit, a long campaign rolls from job to job.
- **Inside a host (component mode):** a host such as Hyperloom's multibranch owns the cards, the
  deadline and the merge; Arbor works on the worktree and the GPU lease it is given, and hands back
  a snapshot of every KEEP. See [docs/hosts.md](docs/hosts.md).
- **Parallel mode:** rounds of lanes on separate nodes, merged with a same-node A/B per lane. Built
  and tested on fakes; it has not run on GPUs.

## Quick start (no GPU)

```bash
python3 -m pip install pyyaml claude-agent-sdk
python3 -m arbor run --config configs/serving-template.json --dry-run
```

The dry run drives the real loop on synthetic numbers in a temporary repository: no GPU, container
or agent. A first campaign on a node: [docs/getting-started.md](docs/getting-started.md).

## Documentation

| Document | For |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | install, a dry run, a first campaign |
| [docs/quickstart-master-worker.md](docs/quickstart-master-worker.md) | the environment in full (packages, credentials, the container, the workspace, the session mount, the GPUs), the config, then the master, a worker, and several workers |
| [docs/quickstart-hyperloom.md](docs/quickstart-hyperloom.md) | Arbor inside Hyperloom's multi-branch mode: the container, the venv and the environment MB needs, which Arbor runs, the flags of `run`, the map's review, two-stage runs, and what goes wrong |
| [docs/quickstart-skills.md](docs/quickstart-skills.md) | a new skill from idea to campaign: where it takes part, its layout and header, checking it, loading it in a pack, seeing it run |
| [docs/quickstart-kernel.md](docs/quickstart-kernel.md) | kernel work: a rewrite's plan, checked, then run as one campaign's kernel-plan lane, stage by stage to a booked gain; what the harness checks at each stage, and what goes wrong |
| [docs/how-it-works.md](docs/how-it-works.md) | the loop, the agents, intents and the gate, planning, the records |
| [docs/packs.md](docs/packs.md) | the pack format, validators, skills, the packs here, writing one |
| [docs/dossier-and-profiler.md](docs/dossier-and-profiler.md) | using the dossier and profiler skills; whether a dossier's directions can go into a pack |
| [docs/model-architecture-performance.md](docs/model-architecture-performance.md) | optional architecture and cache-capacity research between dossier discovery and profiling |
| [docs/skill-integration.md](docs/skill-integration.md) | how skills fit into a campaign: the chapters, the dossier split into five skills, where a new skill joins, two skills that write the same thing, and the `arbor-dev` skills for developing Arbor |
| [docs/skills.md](docs/skills.md) | contributing skills: runtime and development types, families under `.claude/skills/`, the lifecycle's chapters and events, the run, trigger and use modes a skill declares, checking a skill, loading it in a pack, conflicts |
| [docs/master-worker.md](docs/master-worker.md) | a session's master and its workers: which side runs each chapter and pass, the work order and the worker report, where the split stands, and `arbor version` for callers |
| [docs/reference.md](docs/reference.md) | the command line, every config key, environment variables, the session layout |
| [docs/operations.md](docs/operations.md) | launchers, rolling chunks, parallel mode, dashboards |
| [docs/hosts.md](docs/hosts.md) | running Arbor inside a host (component mode) |
| [docs/who-runs-what.md](docs/who-runs-what.md) | component mode under Hyperloom's MB: which side runs each step, and the files between them |
| [configs/README.md](configs/README.md), [evals/README.md](evals/README.md) | the configs; trajectory traces |
| [CONTRIBUTING.md](CONTRIBUTING.md) | branches, tests, issues |

Design records: [docs/component-mode.md](docs/component-mode.md),
[docs/barren-and-learn-design.md](docs/barren-and-learn-design.md),
[docs/hyperloom-upgrade-plan.md](docs/hyperloom-upgrade-plan.md).

## Repository layout

| Path | What |
|---|---|
| `arbor/` | the harness: the driver, the loop and its agents' prompts, the gate, the pack loader, component and parallel modes, collectors and the dashboard |
| `packs/` | what a campaign optimizes and how it is measured |
| `configs/` | where a campaign runs |
| `scripts/` | launchers and tools: parallel mode, pack rendering, the component test |
| `.claude/skills/` | skills the agents may use (dossier, profiler, kernels, hardware, ...) |
| `docs/` | this documentation, design records, and model dossiers (`docs/dossier/`, read by the dossier skill) |
| `evals/` | trajectory traces of campaigns |
| `results/` | records of past campaigns |
| `bench.py`, `install.sh`, `config.json`, `config.json.template` | tools of the retired Local Mode, still in the tree; not used by the harness |

## Development

Development happens in this repository. Work lands on `main`, one change per commit, and `main`
takes fast-forwards only ([CONTRIBUTING.md](CONTRIBUTING.md)).

```bash
python3 -m pytest -q arbor/tests evals/tests
```

[CONTRIBUTING.md](CONTRIBUTING.md) has the branches, the tests and their known failures, and where
to raise issues.
