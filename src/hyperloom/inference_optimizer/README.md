# inference_optimizer

The **inference_optimizer** package is the canonical entry point for
Hyperloom's autonomous LLM inference optimization on AMD GPUs. It
houses the Coordinator (a Python state machine) that drives the
four-agent architecture — Orchestration, Kernel, Critic, and
Robustness — through baseline measurement, profiling, parameter
search, kernel optimization, and validated promotion.

This is the package referenced by `src/hyperloom/inference_optimizer/SKILL.md`;
it is installed from the `hyperloom-inference_optimizer` wheel published on
GitHub Releases (see `examples/README.md`), not from PyPI.

## Where to read next

* **[SKILL.md](SKILL.md)** — the agent-facing instructions: full
  optimization protocol, prompt templates, failure handling, and
  knowledge-base usage. Cursor and Claw load this on demand.
* **[../../../README.md](../../../README.md)** — repository-level overview,
  hosted and local quickstart, results table, and migration notes.
* **[../../../docs/conceptual/optimization-loop.md](../../../docs/conceptual/optimization-loop.md)**
  — the conversational orchestration loop, phase chain, and KB-driven
  priors with a worked example.
* **[../../../docs/reference/authentication.md](../../../docs/reference/authentication.md)** — credential
  and environment configuration.
* **[../../../docs/reference/environment-variables.md](../../../docs/reference/environment-variables.md)**
  — exhaustive list of every environment variable read by the runtime.
* **[../../../docs/reference/session-breakdown.md](../../../docs/reference/session-breakdown.md)**
  — the `session_breakdown.json` contract for downstream consumers.

## Quick CLI

Use the module entry point; it works both for normal installs and
`pip install --target` layouts where console scripts are not on `PATH`:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize \
    --model /path/to/model \
    --framework sglang \
    --gpu-type mi300x \
    --model-class moe_mla \
    --isl 1024 --osl 1024 \
    --max-hours 2.0
```

Resume an interrupted session:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --resume
```

See `python -m hyperloom.inference_optimizer.cli optimize --help` for the full flag set and
[SKILL.md](SKILL.md) for the prompt-driven launch workflow used inside
Cursor and Claw.

## Layout

```
src/hyperloom/inference_optimizer/
├── SKILL.md                    # Agent instructions (Cursor / Claw entry point)
├── references/                 # SKILL reference chapters (benchmark/cache/critic/…)
├── cli/                        # `python -m hyperloom.inference_optimizer.cli optimize` entry point
│   ├── __init__.py             # main()/_build_parser()/_preflight()/_run_optimize()
│   ├── backends/bootstrap/executors/kb/model_gate/model_config_utils.py
│   └── credentials/multi_node/quantization/recover.py
├── session/                    # Session paths, manifest writer, single-optimizer lock
│   ├── manifest.py             # Session manifest writer
│   ├── paths.py                # USER_DATA_PATH-rooted path helpers
│   ├── session_paths.py        # Per-session artifact path helpers
│   └── lock.py                 # Single-optimizer session lock
├── baseline_comparison/        # InferenceX reference fetching & target analysis
├── breakdown/                  # session_breakdown.json producer (downstream contract)
├── actions/                    # Per-action markdown specs + scheduling metadata
├── tools/                      # Operator CLIs (dump_session_breakdown/event_counts/…)
├── experiments/                # A/B and roofline-audit scripts
├── assets/                     # install.sh + bare-metal/profile configs
├── data/                       # Framework/recipe reference data (framework/, recipes/)
└── tests/                      # Unit + regression tests
```

The Coordinator + agent roles + action executors live in the sibling
`hyperloom.orchestrator` package (`src/hyperloom/orchestrator/`), not under
`inference_optimizer/`.

## Package metadata

* **License:** MIT (see top-level `LICENSE`).
* **Python:** 3.10+.
* **Distribution:** built from `pyproject.toml` (at the repo root) as the
  `hyperloom-inference_optimizer` wheel and attached to GitHub Releases; it is
  not published to PyPI. Install the release wheel directly (see
  `examples/README.md` for the `gh release download` + `pip install` flow).
