# Quickstart — Using a Docker container

These instructions allow you to run Hyperloom inside a Docker container on an AMD GPU machine. You attach Cursor to that container and launch the optimization loop from Cursor Chat.

**Prerequisites:**

- A supported AMD GPU: **MI300X / MI308X / MI325X / MI355X** (MI308X and MI325X run with the MI300X runner scripts).
- Access to an OpenAI-compatible (LiteLLM-style) LLM gateway: set **both** `OPENAI_BASE_URL` (your gateway’s `/v1` endpoint) and `ANTHROPIC_BASE_URL`.

### 1. Start the container

Pick the ROCm image matching your GPU (browse SGLang tags at **[hub.docker.com/r/primussafe/sglang/tags](https://hub.docker.com/r/primussafe/sglang/tags)** and vLLM tags at [hub.docker.com/r/primussafe/vllm-openai-rocm/tags](https://hub.docker.com/r/primussafe/vllm-openai-rocm/tags)):

- SGLang MI300X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix`
- SGLang MI355X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix`
- vLLM MI300X: `docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`
- vLLM MI355X: `docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`

Start a long-running container that can access the GPU. The SGLang images have no default entrypoint, so `tail -f /dev/null` runs as-is:

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

The vLLM images ship with an `ENTRYPOINT` of `vllm serve`, so a trailing `tail -f /dev/null` would be parsed as `vllm serve tail -f /dev/null` and the container would exit immediately. Override the entrypoint to keep it idle:

```bash
docker run -d \
  --name hyperloom-local \
  --shm-size 64g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --entrypoint tail \
  docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix \
  -f /dev/null
```

> **Notes:** You need a model available inside the container — download one after attaching (e.g. `huggingface-cli download ...`), or reuse a host model by adding `-v /path/to/models:/models`. The `-profilerfix` images patch rocprofiler so it captures kernels launched under HipGraphLaunch ([SGLang issue #352](https://github.com/sgl-project/sglang/issues/352)).

### 2. Connect Cursor to the container

With Cursor already connected to your remote GPU host and the container running, attach to it using the **Dev Containers** extension. Open the command palette (`Ctrl+Shift+P`) and run **Dev Containers: Attach to Running Container...**:

> If you do not have the Dev Containers extension yet, install it from the [Dev Containers extension page](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) (make sure it is installed in the remote environment).

<img src="../images/cursor_dev_containers_attach.png" alt="Cursor command palette: Dev Containers: Attach to Running Container" width="700" />

Select the container you started (`hyperloom-local`):

<img src="../images/cursor_select_container.png" alt="Selecting the hyperloom-local container to attach to" width="700" />

Cursor opens a new window attached to the running container. Open a workspace folder inside the container to continue.

### 3. Install Hyperloom

#### 3.1 Install with an agent (Recommended)

In the container, make sure GitHub authentication is available, then clone
Hyperloom:

```bash
git clone https://github.com/AMD-AGI/Hyperloom.git && cd Hyperloom
```

Start a coding agent from the repo root (i.e. run `claude` here or open this folder in Cursor) and install Hyperloom using the following prompt:

> Follow the tutorial `examples/hyperloom-local-demo.md` and run the hyperloom demo.

#### 3.2 Install from source

##### 3.2.1 Clone Hyperloom and configure credentials

In the container, make sure GitHub authentication is available, then clone
Hyperloom:

```bash
git clone https://github.com/AMD-AGI/Hyperloom.git && cd Hyperloom
cp .env.template .env
```

Configure the LLM gateway layouts:

**Split Anthropic + OpenAI entrypoints.** Use this when Claude and GPT models
live on different upstream providers or gateways:

```bash
export ANTHROPIC_BASE_URL=https://api.anthropic.com
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
```

**Model IDs.** For split entrypoints or a self-hosted gateway, pin model IDs
that your gateway serves:

```bash
export CLAUDE_MODEL=your-orchestration-model
export CODEX_MODEL=your-kernel-model
```

Also opt out of the AMD-only model gate so
preflight validates against your gateway's `/models` catalog:

```bash
export INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1
```

Shell exports are enough for one session. To persist them, put the same values
in `.env`; shell exports still win over `.env`. For model overrides, Cursor
keys, and endpoint overrides, see
[Authentication and credentials](../reference/authentication.md).

##### 3.2.2. Install runtime dependencies

Run `install.sh` after credentials are available:

```bash
export USER_DATA_PATH=/workspace/hyperloom && mkdir -p "$USER_DATA_PATH"
bash src/hyperloom/inference_optimizer/assets/install.sh
source "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
```

- `USER_DATA_PATH` — Hyperloom's runtime directory for dependency code, logs, state, and results (not the source directory). Use an absolute path pointing at any location with enough space.

When it finishes, source `kernel-agent.env.sh` before launching.

If you explicitly run the forge kernel backend, prepare KernelForge before
`install.sh`:

```bash
export KERNEL_OPT_BACKEND_ORDER=forge
bash src/hyperloom/inference_optimizer/assets/local_setup.sh --no-next-steps
source "$USER_DATA_PATH/runtime/local-setup.env.sh"
bash src/hyperloom/inference_optimizer/assets/install.sh
source "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
```

Only the forge backend requires KernelForge access. The standard LLM/runtime
setup still happens through `install.sh`.

---

## Next Step

After setup, open the Hyperloom checkout in Cursor. Then follow
[Run a Hyperloom optimization](../how-to/optimize.md) to launch, monitor, or
resume a run.

For the optional AMD Quark quantization prelude, see [Quantization with AMD Quark](../how-to/quantization-quark.md).

## Appendix — Environment configuration (.env)

<details>
<summary><strong>Basic <code>.env</code> recipe and advanced options</strong></summary>

To persist gateway credentials between shells instead of exporting them each session:

```bash
cp .env.template .env
```

Edit `.env`:

```text
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
```

Shell `export` always wins over `.env` — see [Credential precedence](../reference/authentication.md#credential-precedence).

For more information, see [Authentication and credentials](../reference/authentication.md):

- **Split Anthropic + OpenAI entrypoints** — [Split entrypoints](../reference/authentication.md#split-entrypoints-native-anthropic-openai)
- **Self-hosted gateway + custom models** — [Non-AMD / self-hosted gateway](../reference/authentication.md#non-amd-self-hosted-gateway)
- **Optional `TRACELENS_INTERNAL_ROOT`** — [Dependency checkout variables](../reference/authentication.md#dependency-checkout-variables)

</details>

## Troubleshooting

For TLS or certificate verification failures against the LLM gateway, see
[TLS or certificate errors against the LLM gateway](../reference/troubleshooting.md#tls-or-certificate-errors-against-the-llm-gateway).
