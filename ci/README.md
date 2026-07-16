# Hyperloom CI Helpers

This directory contains the public, portable pieces of the Hyperloom CI
tooling: configuration parsing, matrix generation, artifact normalization,
report generation, and unit tests.

The live optimization submission path integrates with a private SaFE/Claw
deployment. Public forks should treat that path as an adapter that requires
their own service URL, workspace, storage volume, images, and credentials. The
default public workflow should run the pure Python tests only.

## Internal-Only Adapters

The following helpers are retained in this public tree as integration adapters,
but they are not self-contained CI for public forks:

- `orchestrator.py` creates and monitors private SaFE/Claw optimization sessions.
- `optimize_submit.py` and `optimize_submit_lib/safe_client.py` submit to a
  compatible SaFE API and require caller-provided credentials and workspaces.
- `publish_results.py` and `publish_artifacts.py` publish to an external results
  service configured by the caller.

The GPU/SaFE GitHub Actions workflows are similarly internal by default. Jobs
that require AMD self-hosted runners are guarded with
`github.repository == 'AMD-AGI/Hyperloom'` so forks skip them instead of waiting
forever for unavailable runner labels. Portable quality gates remain on
`ubuntu-latest`: lint, coverage/tests, docs, CodeQL, and secret scanning.

## Quick Start

```bash
pip install -r ci/requirements.txt
pytest ci
```

Generate a dry-run prompt without contacting SaFE/Claw:

```bash
SAFE_BASE_URL=https://safe.example.invalid \
SAFE_OPTIMIZE_WORKSPACE=my-workspace \
SAFE_OPTIMIZE_VOLUME=/mnt/shared \
HARBOR_PREFIX=registry.example.invalid/my-images \
python ci/orchestrator.py --dry-run --output-dir ./ci-output
```

Run a direct submit only after configuring a real compatible backend:

```bash
SAFE_BASE_URL=https://safe.example.invalid \
SAFE_API_KEY=... \
SAFE_OPTIMIZE_REGISTER_WORKSPACE=my-register-workspace \
SAFE_OPTIMIZE_SUBMIT_WORKSPACE=my-submit-workspace \
SAFE_OPTIMIZE_VOLUME=/mnt/shared \
python ci/optimize_submit.py --model Qwen/Qwen3-8B
```

## Layout

```
ci/
├── optimize_submit.py       # SaFE/Claw submit facade; requires private backend config
├── optimize_submit_lib/     # Submit, detect, artifact, and report helpers
├── orchestrator.py          # Prompt and report orchestration
├── inferenceX_parser.py     # InferenceX config parsing + public benchmark fetching
├── report_generator.py      # Markdown, JSON, and GitHub summary generation
├── ci-config.yaml           # Example model/runtime configuration
├── inferenceX_models.yaml   # InferenceX API model name mapping
├── requirements.txt         # Python dependencies
└── test_*.py                # Portable unit tests
```

## Configuration

Important environment variables:

| Variable | Required for live submit | Description |
|---|---:|---|
| `SAFE_BASE_URL` / `SAFE_API_URL` | Yes | Base URL of a compatible SaFE API deployment. |
| `SAFE_API_KEY` / `CLAW_API_KEY` | Yes | Bearer token for the SaFE/Claw API. |
| `SAFE_OPTIMIZE_REGISTER_WORKSPACE` | Yes | Workspace used to register or download models. |
| `SAFE_OPTIMIZE_SUBMIT_WORKSPACE` | Yes | Workspace used to run optimization tasks. |
| `SAFE_OPTIMIZE_VOLUME` | Yes | Shared storage volume mounted by the backend. |
| `HARBOR_PREFIX` | No | Optional container registry prefix for configured images. |
| `HYPERLOOM_RESULTS_SERVICE_URL` | No | Optional results service URL; publishing is skipped when unset. |

## Public CI Scope

Open-source CI should focus on deterministic checks that do not need internal
infrastructure:

```bash
pytest ci
python ci/generate_hf_matrix.py --help
python ci/generate_matrix.py --help
```

The repository does not include a self-contained GPU performance reproduction
harness. Performance runs require the caller to provide their own serving
framework images, GPU environment, model storage, and API credentials.
