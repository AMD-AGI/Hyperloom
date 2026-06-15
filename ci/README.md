# Hyperloom Inference Optimization CI/CD

Automated inference optimization pipeline: fetch configuration and benchmark
data from InferenceX, launch a Claw Agent to run the Hyperloom skill, and
generate optimization reports.

## File Layout

```
ci/
├── orchestrator.py          # Main orchestrator entry point
├── claw_client.py           # Claw API wrapper (session/message/SSE/files)
├── inferenceX_parser.py     # InferenceX config parsing + API data fetching
├── report_generator.py      # Report generation (markdown + JSON + GitHub Summary)
├── ci-config.yaml           # Model list + runtime configuration
├── prompt_template.md       # Prompt template sent to the Claw Agent
├── inferenceX_models.yaml   # InferenceX API model name mapping
├── AB_TEST.md               # A/B test usage guide (for GEAK/TraceLens and related teams)
├── test_claw_flow.py        # End-to-end Claw API test script
├── requirements.txt         # Python dependencies
└── README.md
```

> For A/B testing (comparing the optimization effect of two tool combinations),
> see **[AB_TEST.md](AB_TEST.md)**.

## Quick Start

```bash
pip install -r requirements.txt

# Dry run: generate prompts only, do not execute
HARBOR_PREFIX=harbor.core42.example-internal-host.invalid/proxy \
KERNEL_OPT_WORKSPACE=core42-sandbox \
  python orchestrator.py --dry-run

# Actual execution (single model)
HARBOR_PREFIX=harbor.core42.example-internal-host.invalid/proxy \
KERNEL_OPT_WORKSPACE=core42-sandbox \
CLAW_API_KEY=ak-xxx \
  python orchestrator.py --models qwen3.5-bf16-mi355x-sglang --output-dir ./results

# Run all models
python orchestrator.py --trigger manual --output-dir ./results
```

## Environment Variables

| Variable | Required | Description |
|------|------|------|
| `HARBOR_PREFIX` | Yes | Image registry prefix, for example `harbor.core42.example-internal-host.invalid/proxy` |
| `KERNEL_OPT_WORKSPACE` | Yes | Workspace used for kernel optimization (shared by GEAK + OOB), for example `core42-sandbox` |
| `CLAW_API_KEY` | Yes | SaFE API key (with `ak-` prefix) |
| `WEBHOOK_URL` | No | Notification webhook (compatible with Slack / Teams Incoming Webhook) |

## CLI Arguments

```
python orchestrator.py [OPTIONS]

--config PATH        Path to ci-config.yaml (defaults to the local ci/ directory)
--models KEYS        Comma-separated model keys; run only a subset
--trigger TYPE       Trigger type: manual / scheduled / inferenceX
--dry-run            Print prompts only; do not execute
--output-dir DIR     Output directory for reports (default: ci-output/)
```

## Adding a Model

1. Check `inferenceX_models.yaml` to confirm the API model name
2. Check InferenceX `amd-master.yaml` to confirm the key
3. Make sure the model is already downloaded to `/hyperloom/models/`
4. Add a new entry under `models` in `ci-config.yaml`:

```yaml
- inferenceX_key: dsr1-fp8-mi355x-sglang       # Key in amd-master.yaml
  inferenceX_api_name: DeepSeek-R1-0528        # InferenceX API model name
  model_path_override: /hyperloom/models/xxx   # Local model path
  optimization_depth: full                     # full / param-only / baseline-only
  kernel_opt_backends: geak, claude            # Kernel optimization backends
  target_gpu: b200                             # Competitor GPU used for comparison
```

## GitHub Actions

### Configure Secrets

Settings → Secrets and variables → Actions → New repository secret:

| Secret | Value |
|--------|---|
| `HARBOR_PREFIX` | `harbor.core42.example-internal-host.invalid/proxy` |
| `KERNEL_OPT_WORKSPACE` | `core42-sandbox` |
| `CLAW_API_KEY` | `ak-xxx` |
| `WEBHOOK_URL` | Teams/Slack Incoming Webhook URL (optional) |

### Trigger Methods

- **Scheduled**: every Monday at 02:00 UTC
- **Manual**: Actions → Run workflow → optionally enter a subset of models
- **InferenceX update**: checked by the scheduled workflow against new commits on `main`

### Execution Model

Each model runs in an independent job (matrix strategy). One failed model does
not affect the others.

### Outputs

- **GitHub Summary**: comparison table shown on each model job page
- **Artifact `report-{model_key}`**: per-model `optimization_report.md` + summary
- Reports are retained for 90 days

## Webhook Notifications

Slack and Teams Incoming Webhooks are supported. One notification is sent after
each model finishes:

```
Hyperloom CI [Qwen3.5-397B-A17B]: completed | Gain: +3.0% | Trigger: manual
```

Teams setup: Teams Channel → Connectors → Incoming Webhook → copy the URL →
store it in the `WEBHOOK_URL` secret.
