# inference_optimizer

The **inference_optimizer** package is the canonical entry point for
Hyperloom's autonomous LLM inference optimization on AMD GPUs. It
houses the Coordinator (a Python state machine) that drives the
four-agent architecture — Orchestration, Kernel, Critic, and
Robustness — through baseline measurement, profiling, parameter
search, kernel optimization, and validated promotion.

This is the package referenced by `inference_optimizer/SKILL.md` and
installed by `pip install hyperloom-inference_optimizer`.

## Where to read next

* **[SKILL.md](SKILL.md)** — the agent-facing instructions: full
  optimization protocol, prompt templates, failure handling, and
  knowledge-base usage. Cursor and Claw load this on demand.
* **[../README.md](../README.md)** — repository-level overview,
  hosted and local quickstart, results table, and migration notes.
* **[../docs/HOW_THE_OPTIMIZATION_LOOP_WORKS.md](../docs/HOW_THE_OPTIMIZATION_LOOP_WORKS.md)**
  — DFS scoring, dynamic branching, and KB-driven priors with a
  worked example.
* **[../docs/ENV_AND_AUTH.md](../docs/ENV_AND_AUTH.md)** — credential
  and environment configuration.
* **[../docs/CONFIGURATION_REFERENCE.md](../docs/CONFIGURATION_REFERENCE.md)**
  — exhaustive list of every environment variable read by the runtime.
* **[../docs/INTEGRATION_SESSION_BREAKDOWN.md](../docs/INTEGRATION_SESSION_BREAKDOWN.md)**
  — the `session_breakdown.json` contract for downstream consumers.

## Quick CLI

The package installs the `inference_optimizer` console script:

```bash
inference_optimizer optimize \
    --model /path/to/model \
    --framework sglang \
    --gpu-type mi300x \
    --model-class moe_mla \
    --isl 1024 --osl 1024 \
    --max-hours 2.0
```

Resume an interrupted session:

```bash
inference_optimizer optimize --resume
```

See `inference_optimizer optimize --help` for the full flag set and
[SKILL.md](SKILL.md) for the prompt-driven launch workflow used inside
Cursor and Claw.

## Layout

```
inference_optimizer/
├── SKILL.md                   # Agent instructions (Cursor / Claw entry point)
├── cli.py                     # `inference_optimizer optimize` entry point
├── manifest.py                # Session manifest writer
├── paths.py                   # USER_DATA_PATH-rooted path helpers
├── orchestrator/              # Coordinator + agent roles + action executors
│   ├── action_executors/      # baseline / roofline / explore / sweep / report …
│   ├── backends/              # Critic / Robustness subprocess adapters
│   └── system_prompts/        # Orchestration prompt construction
├── baseline_comparison/       # InferenceX reference fetching & target analysis
├── breakdown/                 # session_breakdown.json producer (downstream contract)
├── actions/                   # Per-action markdown specs + scheduling metadata
├── scripts/                   # install.sh, dump scripts, baseline/profile configs
└── tests/                     # Unit + regression tests
```

## Package metadata

* **License:** Apache-2.0 (see top-level `LICENSE`).
* **Python:** 3.10+.
* **Distribution:** PyPI as `hyperloom-inference_optimizer`. Source of
  truth: `pyproject.toml` at the repo root.
