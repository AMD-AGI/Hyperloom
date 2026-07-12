---
myst:
    html_meta:
        "description": "Guide to running Hyperloom optimization jobs on a Slurm cluster: container runtimes (pyxis/enroot or docker), GPU passthrough without gres, LLM gateway access through a jump host, and the ready-to-use submission scripts."
        "keywords": "Hyperloom, Slurm, sbatch, srun, pyxis, enroot, docker, AMD GPU, MI355X, MI300X, ROCm, LLM gateway, DeepSeek, inference optimization"
---
# Run Hyperloom on Slurm

This topic describes how to submit and run Hyperloom inference-optimization jobs
on a **Slurm** cluster. It is the batch-scheduler counterpart to the
[self-hosting and operations guide](../operations.md), which covers the
Kubernetes layout for the same workload.

```{note}
This topic is for site-reliability and platform engineers running Hyperloom on
their own AMD GPU cluster through Slurm. For the hosted
[Primus-Claw experience](https://crusoe.primus-safe.amd.com/hyperloom/) AMD owns
operations and this document does not apply.
```

A single Hyperloom optimization is a long-running Python process (the
Coordinator) that drives an inference server (sglang / vllm) and Ray-scheduled
kernel workers on the GPUs of one node. On Slurm this maps to a one-node,
one-task job that launches a ROCm container and runs the `inference_optimizer`
CLI inside it. The scripts in this directory wrap that flow and encode the
adaptations required on real clusters.

For application-level configuration see
[Environment variables](../environment-variables.md); for credential setup see
[Authentication and credentials](../authentication.md); for recurring symptoms
see [Troubleshooting](../troubleshooting.md).

---

## Files in this directory

| File | Purpose |
|---|---|
| {download}`submit.sh <submit.sh>` | Generic submitter: validates the model key, wires the config dir, applies cluster overrides, and submits one job per model. |
| {download}`submit-vultr.sh <submit-vultr.sh>` | Preset wrapper (MI355X / docker / `/mnt/vast`) around `submit.sh`. Copy and adapt for your own cluster. |
| {download}`run_hyperloom.sbatch <run_hyperloom.sbatch>` | The job body: starts the container, sets the gateway host alias, mounts the CA bundle, installs the optimizer, and launches the backend. |
| {download}`models.tsv <models.tsv>` | Model table (key / repo / framework / image / TP / precision / ISL / OSL / concurrency / model-class / max-hours / target-gain). |
| {download}`proxy.env <proxy.env>` | Gateway connectivity profile (host alias, endpoint + key, CA bundle, model-name overrides). **Fill in before use.** |

---

## Cluster prerequisites

The scripts assume the following. Where your cluster differs, see the adaptation
notes called out below.

| Aspect | Assumption | Consequence |
|---|---|---|
| Scheduler | Slurm (`sbatch` / `srun` / `sinfo` / `squeue` available) | — |
| Container runtime | pyxis/enroot **or** docker | Auto-detected; override with `HL_CONTAINER_RUNTIME`. |
| GPU | AMD Instinct (for example MI355X `gfx950` or MI300X), 8 per node | Image and `--gpu-type` must match the hardware variant. |
| Slurm GPU gres | GPUs may or may not be registered as gres | If **not** registered, you cannot use `--gpus`; see the *Clusters without GPU gres* section below. |
| Shared filesystem | A cross-node mount (WekaFS, VAST/NFS, ...) | Holds source, artifacts, and the CA bundle. |
| LLM gateway | Reachable directly or through a jump host | May need a host alias plus the AMD internal CA. |
| Container registry | Public or private | A private registry unreachable from compute nodes requires a locally cached image. |

---

## One-time setup

### 1. Configure the LLM gateway key

Hyperloom is an LLM-driven agent: the orchestrator and kernel agent call the
gateway throughout the run, so a working key and a reachable endpoint are hard
requirements (the installer's preflight enforces them).

Edit `proxy.env` and replace the three `ak-REPLACE_ME` placeholders with your
SaFE API key (issued from the Primus-SaFE LLM Gateway page), and set
`HL_HOST_ALIAS_IP` to your gateway jump-host IP if the domain does not resolve
on compute nodes. Verify the key before submitting:

```bash
set -a; . ./proxy.env; set +a
curl -sS --resolve global.primus-safe.amd.com:443:<GATEWAY_JUMP_HOST_IP> \
  "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

HTTP 200 plus a model list means the key works. **Note the model names in the
list** — they are needed in the *LLM model-name constraints* section below.

### 2. Build a combined CA bundle

If the gateway uses an internal CA, the job
still needs to reach **huggingface.co** and **github.com** over public TLS. A
CA bundle containing only the internal certificate makes public TLS fail with
`CERTIFICATE_VERIFY_FAILED`. Merge the system roots with the internal CA into a
single bundle:

```bash
cat /etc/ssl/certs/ca-certificates.crt internal-ca.pem > ca-combined.pem
```

Point `HL_CA_BUNDLE_HOST` in `proxy.env` at `ca-combined.pem`. Leave it empty if
the container image already trusts every CA you need.

### 3. Confirm the container image is reachable

If `models.tsv` references a private registry (a `<REGISTRY>/...` prefix) that
compute nodes cannot resolve, docker fails with `no such host`. Either
`docker login` the reachable registry, or replace the image with a
node-local cached image name (drop the registry prefix). List cached images:

```bash
ssh <node> 'docker images | grep -iE "sglang|vllm"'
```

Match the image variant to the hardware (for example a `mi35x` build for MI355X,
a `mi30x` build for MI300X).

### 4. Stage the Hyperloom source

Copy a **complete** source checkout (src-layout, `src/hyperloom/...`) to the
shared mount and point `submit.sh --source-dir` at it to skip the runtime
`git clone`. It must be a complete snapshot: an old flat layout or a snapshot
missing `src/hyperloom/inference_optimizer/cli/` causes `ModuleNotFoundError`
(see the *Troubleshooting* section below). If `--source-dir` is omitted, the job
clones `main` from GitHub at runtime, which requires GitHub egress on the node.

---

## Submit a job

```bash
# Generic (pyxis/enroot or docker auto-detected; requests TP GPUs by default)
./submit.sh dsv4pro_sglang

# Preset for an MI355X / docker / /mnt/vast cluster
./submit-vultr.sh dsv4pro_sglang

# Print the sbatch command without submitting
./submit.sh --dry-run dsv4pro_sglang

# claude-code backend (the image must ship the claude CLI)
./submit.sh -b claude dsv4pro_sglang
```

Model keys shipped in `models.tsv`: `dsv4pro_sglang`, `dsv4pro_vllm`,
`dsv4flash_vllm`, `dsv4flash_sglang`.

Useful `submit.sh` options (run `./submit.sh --help` for the full list):

| Option | Meaning |
|---|---|
| `-b, --backend <python\|claude>` | Launch backend (default `python`). |
| `-p, --partition <name>` | Slurm partition. |
| `-g, --gpus <n>` | GPUs to request (default: the model's TP; use `0` on clusters without GPU gres). |
| `-c, --constraint <feat>` | Slurm `--constraint` (for example `gfx950`). |
| `--gpu-type <TYPE>` | Override the optimizer `--gpu-type` (lowercase, for example `mi355x`). |
| `--shared-mount <path>` | Shared FS bind-mounted into the container. |
| `--source-dir <path>` | Existing Hyperloom checkout to use instead of cloning. |
| `--data-root <path>` | Artifact root (default `<shared-mount>/hyperloom-slurm`). |
| `--dry-run` | Print the sbatch command, do not submit. |

### Clusters without GPU gres

If compute nodes do **not** register GPUs as Slurm gres (`scontrol show node`
reports `Gres=(null)`), any `--gpus N>0` request stays `PENDING (Resources)`
forever. The convention is then "one job takes a whole node; the container
grabs the GPUs directly":

- pass `-g 0` so the job requests only a placeholder CPU and lands on the node;
- `submit.sh` emits `--gpus` only when `-g` is non-zero;
- `run_hyperloom.sbatch` carries no `#SBATCH --gpus` directive.

The container still receives every GPU through `--device=/dev/kfd
--device=/dev/dri`, so TP=8 works. If your GPUs **are** registered as gres
(`Gres=gpu:...:8`), use the standard `--gpus 8` and skip `-g 0`.

---

## LLM model-name constraints

The job calls the gateway for orchestration and for the kernel agent (GEAK).
Model names must exist in your key's catalog, and the orchestration model is
hard-allowlisted:

| Use | Environment variable | Allowed values | Notes |
|---|---|---|---|
| Orchestration | `CLAUDE_MODEL` / `CURSOR_DEFAULT_MODEL` / `LLM_MODEL` | `claude-opus-4-7` (preferred) or `claude-opus-4-6` | Enforced by the optimizer's model gate; other names are rejected. |
| Kernel agent (GEAK) | `GEAK_MODEL_NAME` | for example `claude-opus-4-8` | Not subject to the orchestration gate. |
| Codex / OOB | `CODEX_MODEL` | for example `gpt-5.4` | Use a gpt/codex-family model. |

- Do **not** use suffixed variants (for example `claude-opus-4-7-thinking-xhigh`);
  the gateway returns `Invalid model name` (which can surface misleadingly as
  `401 missing subscription key`).
- To use an orchestration model outside the allowlist, set
  `INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1` (this switches to catalog
  validation). Not recommended.
- These variables are set in `proxy.env` and are on the docker `-e` allowlist in
  `run_hyperloom.sbatch`. If you add a new model variable, add it to that
  allowlist too, or it will not reach the container.

---

## Artifacts and monitoring

```bash
squeue -p <partition>                                 # queue
tail -f hyperloom-hl-<key>-<backend>-<jobid>.out      # job output (submit dir)
SID=$(ls -t <data-root>/<key> | head -1)
cat <data-root>/<key>/$SID/state.json
```

The artifact directory `<data-root>/<model_key>/<CLAW_SESSION_ID>/` contains:

- `state.json` — live status (`baseline_tput`, `current_best`, `cumulative_gain`);
- `manifest.json` — session manifest;
- `ci_metrics.json` — baseline/optimized throughput plus `gain_pct`;
- `optimizer_runs/` — `launch_<sid>.json` and logs;
- `runtime/` — `kernel-agent.env.sh` and other files produced by the installer.

A healthy run logs, in order: container start → install complete → framework
root discovery (`sglang=ok aiter=ok`) → model gate passed → Ray head
(`--num-gpus=8`) → model load → baseline throughput → optimization iterations.
GPU memory and utilization ramp up only after the model-load step.

---

## Tunable environment variables

| Variable | Default | Meaning |
|---|---|---|
| `HL_SHM_SIZE` | `64g` | docker `--shm-size`; raise it for high concurrency. |
| `HL_CONTAINER_RUNTIME` | `auto` | Force `docker` or `pyxis`. |
| `HL_GPU_TYPE_OVERRIDE` | — | Override `--gpu-type` (lowercase) when hardware differs from the table row. |
| `HL_SHARED_MOUNT` | `/wekafs` | Shared FS bind-mounted into the container. |
| `HL_DATA_ROOT` | `<shared-mount>/hyperloom-slurm` | Artifact root. |
| `HL_CA_BUNDLE_HOST` | — | Combined CA bundle path (see *Build a combined CA bundle* under One-time setup). |
| `INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL` | unset | Set `1` to relax the orchestration model allowlist (not recommended). |

---

## Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| Job stuck `PD (Resources)` while the node is `idle` | Node has no GPU gres; `--gpus` can never be satisfied | Submit with `-g 0`; confirm no `#SBATCH --gpus` (see the *Clusters without GPU gres* section). |
| `docker: ... no such host` when pulling the image | Private registry unreachable from the node | Use a node-cached local image name, or `docker login` a reachable registry. |
| `CERTIFICATE_VERIFY_FAILED: unable to get local issuer` (huggingface/github) | CA bundle has only the internal cert | Use a combined CA bundle. |
| `ModuleNotFoundError: No module named '...cli'` | Incomplete or old-layout source snapshot | Stage a complete src-layout checkout (`src/hyperloom/...`). |
| `--gpu-type: invalid choice: 'MI355X'` | The CLI is case-sensitive | Use lowercase (for example `mi355x`; `submit-vultr.sh` already does). |
| `--claude-model=... is not allowed` | Orchestration model not in the allowlist | Use `claude-opus-4-7` for orchestration. |
| `Invalid model name` / `401 missing subscription key` | Model name not in the key catalog (often a suffixed variant) | Use a name from `curl $OPENAI_BASE_URL/models`. |
| Server fails to start / OOM in shm | `/dev/shm` too small | Raise `HL_SHM_SIZE` (default `64g`). |
| DNS failure / connection timeout to the gateway | Host alias not applied | The docker path uses `--add-host`; confirm the node can reach the jump host on `:443`. |

Suggested triage order: **GPU gres → image reachability → combined CA → complete
source → gpu-type casing → orchestration-model allowlist.**

---

## Known limitations

- A source snapshot without `.git` (a plain rsync copy) leaves `code_revision`
  empty in `manifest.json`. Sync with `.git` if you need provenance.
- The docker path exposes all node GPUs to the container, so TP=8 occupies a
  whole node. Running multiple jobs per node requires isolating devices with
  `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` (the scripts do not do this).
- The installer clones Magpie / InferenceX / GEAK / TraceLens from GitHub, so
  nodes need GitHub egress. A fully offline setup must bake these into the image
  and pre-stage model weights on the shared mount.
- If the gateway's upstream lacks subscription credentials, real completions may
  return `missing subscription key` even though listing models and auth succeed;
  that is a gateway-side issue.
