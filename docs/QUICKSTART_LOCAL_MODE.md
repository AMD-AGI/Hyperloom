# Quickstart — Local Mode (Cursor)

Local Mode runs Hyperloom inside a Docker container on an AMD GPU machine. You attach Cursor to that container and launch the optimization loop from Cursor Chat.

There are two ways to get a GPU environment:

- **[Your own GPU machine](#quickstart--your-own-gpu-machine)** — the primary, fully self-serve path.
- **[Primus-SaFE platform](#optional-quickstart--primus-safe-platform)** — an optional path for AMD-internal users on the Primus-SaFE Authoring platform.

---

## Quickstart — Your own GPU machine

**Prerequisites:**

- An AMD GPU machine supporting **MI300X** or **MI355X**.
- Access to an OpenAI-compatible (LiteLLM-style) LLM gateway: set **both** `SAFE_API_KEY` (your gateway API key) and `OPENAI_BASE_URL` (your gateway’s `/v1` endpoint). The Primus-SaFE LiteLLM gateway is one option (get your key via the [LLM Gateway](https://global.primus-safe.amd.com/litellm-gateway)), but any compatible gateway works.

### 1. Start the container

Pick the ROCm image matching your GPU (browse all tags at **[hub.docker.com/r/primussafe/sglang/tags](https://hub.docker.com/r/primussafe/sglang/tags)**):

- SGLang MI300X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix`
- SGLang MI355X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix`
- vLLM MI300X: `docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`
- vLLM MI355X: `docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`

Start a long-running container that can access the GPU:

```bash
docker run -d \
  --name hyperloom-local \
  --shm-size 64g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  docker.io/primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix \
  tail -f /dev/null
```

> **Notes:** You need a model available inside the container — download one after attaching (e.g. `huggingface-cli download ...`), or reuse a host model by adding `-v /path/to/models:/models`. The `-profilerfix` images patch rocprofiler so it captures kernels launched under HipGraphLaunch ([SGLang issue #352](https://github.com/sgl-project/sglang/issues/352)).

### 2. Connect Cursor to the container

With Cursor already connected to your remote GPU host and the container running, attach to it using the **Dev Containers** extension. Open the command palette (`Ctrl+Shift+P`) and run **Dev Containers: Attach to Running Container...**:

> If you do not have the Dev Containers extension yet, install it from the [Dev Containers extension page](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) (make sure it is installed in the remote environment).

<img src="figs/cursor_dev_containers_attach.png" alt="Cursor command palette: Dev Containers: Attach to Running Container" width="700" />

Select the container you started (`hyperloom-local`):

<img src="figs/cursor_select_container.png" alt="Selecting the hyperloom-local container to attach to" width="700" />

Cursor opens a new window attached to the running container. Open a workspace folder inside the container to continue.

### 3. Clone Hyperloom and bootstrap Local Mode

In the container, make sure GitHub authentication and AMD-AGI repository access are available — `local_setup.sh` reuses that access to clone dependency repositories. Then clone and bootstrap in one go:

```bash
git clone https://github.com/AMD-AGI/Hyperloom.git && cd Hyperloom
export SAFE_API_KEY=ak-your-safe-apikey            # <-- paste your gateway key
export OPENAI_BASE_URL=https://global.primus-safe.amd.com/api/v1/llm-proxy/v1  # <-- set to your gateway base URL (Primus-SaFE example)
export USER_DATA_PATH=/workspace/hyperloom && mkdir -p "$USER_DATA_PATH"
bash src/hyperloom/inference_optimizer/assets/local_setup.sh
```
> **Tip:** Instead of exporting these variables each session, you can persist credentials in a `.env` file. See [Appendix — Environment configuration (.env)](#appendix-environment-configuration-env) for the basic recipe, or [Auth & Environment Guide](ENV_AND_AUTH.md) for the full reference.

- `SAFE_API_KEY` — your key from the [LLM Gateway](https://global.primus-safe.amd.com/litellm-gateway). Exporting it in the shell is enough; to persist it instead, use the [`.env` appendix](#appendix-environment-configuration-env).
- `USER_DATA_PATH` — Hyperloom's runtime directory for dependency code, logs, state, and results (not the source directory). It **must be an absolute path** and can point at any location with enough space.
- Not using the Primus-SaFE gateway? See **ENV_AND_AUTH.md §2.3** (Non-AMD / self-hosted gateway) for self-hosted gateways and model overrides: [ENV_AND_AUTH.md](ENV_AND_AUTH.md).

When it finishes, `local_setup.sh` prints the workspace path to open in Cursor, a prompt template to paste into Cursor Chat (with a `Model:` field to fill in), and the env file to source before launch.

---

## Optional Quickstart — Primus-SaFE platform

AMD-internal users can run Local Mode on the **Primus-SaFE Authoring** platform instead of their own machine:

1. Create an Authoring Pod on Primus-SaFE Authoring and select an SGLang or vLLM image. On this platform, use the Harbor mirror prefix `harbor.<datacenter_name>.primus-safe.amd.com/proxy/primussafe/sglang:<tag>` (the internal mirror of the Docker Hub images above).
2. When the Pod is ready, connect to it with Cursor Remote SSH (follow the connection instructions shown in the Primus-SaFE Authoring UI).
3. Inside the Pod, follow [Step 3](#3-clone-hyperloom-and-bootstrap-local-mode) above to clone Hyperloom and run the bootstrap.

---

## Launch Inference Optimization

After setup, open the Hyperloom workspace printed by `local_setup.sh` in Cursor, then paste the generated prompt template into Cursor Chat. Fill in the model path and adjust the other workload parameters before sending.

`local_setup.sh` prints something like this when it finishes:

````text
Open this folder in Cursor as the workspace:
  /path/to/Hyperloom

Paste this into Cursor Chat and fill in your workload:

@src/hyperloom/inference_optimizer/SKILL.md

Optimize inference for this workload:
- Model: /path/to/your/model   # <-- replace with the path to your downloaded model
- Framework: sglang
- GPU: MI300X
- TP: 8
- CONC: 64
- ISL: 1024
- OSL: 1024
- Goal: improve throughput by at least 10%
- Budget: 24 hours

Before launch, run exactly:
```bash
source '/path/to/hyperloom-run/runtime/local-setup.env.sh'
export USER_DATA_PATH='/path/to/hyperloom-run'
```
````

The fields you commonly edit:

| Field | Maps to | Description | Default |
|---|---|---|---|
| `Model` | `--model` | Path to your model. | required |
| `Framework` | `--framework` | `sglang` or `vllm` (do not mix within one session). | `sglang` |
| `GPU` | `--gpu-type` | e.g. `MI300X`, `MI325X`, `MI355X`. | auto-detect |
| `Goal` | `--target-gain` | Optional stop condition, such as a target throughput gain. | unset |
| `Budget` | `--max-hours` | Maximum optimization time. | `2.0` hours |

For the full list of workload fields, CLI flags, and defaults, see [Step 2 — Launch in `src/hyperloom/inference_optimizer/SKILL.md`](../src/hyperloom/inference_optimizer/SKILL.md). For first-launch errors, see that skill's §"Failure Handling".

<details>
<summary><strong>Resume an existing session</strong></summary>

```text
@src/hyperloom/inference_optimizer/SKILL.md

Resume the existing Hyperloom optimization session.

Requirements:
1. Launch `inference_optimizer optimize --resume`; do not start a new session.
2. Do not pass `--model`; read the model and workload from the saved manifest/state.
3. Before launching, verify `manifest.json` and `state.json` exist.
4. Report the log path, PID, initial health check result, current phase, cumulative gain, and best config.
5. Monitor the process every 300s until the optimization is complete or failed.
```

</details>

For the optional AMD Quark quantization prelude, see [Quantization (AMD Quark)](QUANTIZATION_QUARK.md).

## Appendix — Environment configuration (.env)

<details>
<summary><strong>Basic <code>.env</code> recipe and advanced options</strong></summary>

To persist gateway credentials between shells instead of exporting them each session:

```bash
cp .env.template .env
```

Edit `.env`:

```env
SAFE_API_KEY=ak-your-safe-apikey
OPENAI_BASE_URL=https://global.primus-safe.amd.com/api/v1/llm-proxy/v1
```

Shell `export` always wins over `.env` — see [ENV_AND_AUTH.md §1](ENV_AND_AUTH.md#1-credential-precedence).

For setups beyond the single-gateway default above, see [Auth & Environment Guide](ENV_AND_AUTH.md):

- **Split Anthropic + OpenAI entrypoints** — [ENV_AND_AUTH.md §2.2](ENV_AND_AUTH.md#22-split-entrypoints-native-anthropic-openai)
- **Non-AMD / self-hosted gateway + custom models** — **ENV_AND_AUTH.md §2.3**: [ENV_AND_AUTH.md](ENV_AND_AUTH.md)
- **Optional `CURSOR_API_KEY` / `CURSOR_DEFAULT_MODEL`** (Cursor kernel-opt backend) — [ENV_AND_AUTH.md §3.1](ENV_AND_AUTH.md#31-cursor_api_key-cursor-sdk-kernel-opt-backend)
- **Optional `TRACELENS_INTERNAL_ROOT`** (internal TraceLens extension) — [ENV_AND_AUTH.md §4.2](ENV_AND_AUTH.md#42-dependency-checkouts-auto-provisioned)

</details>

## Troubleshooting

**TLS / certificate errors against the AMD LLM Gateway (AMD network only).** If you are using the Primus-SaFE gateway and HTTPS requests to it fail with a certificate verification error inside the container, install the AMD certificate bundle:

```bash
curl -fsSL https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs/setup.sh | bash
```

This is only needed for the AMD-hosted gateway; it does not apply when you point `OPENAI_BASE_URL` at your own gateway.
