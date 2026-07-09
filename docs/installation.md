# Installation

Hyperloom can be used two ways: through the hosted **Hyperloom UI** (no local
setup), or in **Local Mode** on a remote AMD GPU environment driven from Cursor.

```{note}
The canonical, always-current installation walkthrough lives in the repository
[`README.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/README.md). This
page summarizes the entry points and links to the detailed guides.
```

## Prerequisites

Bind your [LLM Gateway](https://llm.amd.com/) key to
[Hyperloom](https://crusoe.primus-safe.amd.com/hyperloom/) to obtain your API
key. This key provides access to TraceLens, GEAK, and OOB services for both the
UI and the local workflow.

## Option 1 — Hosted UI (PrimusClaw)

The fastest path, with no local GPU setup:

1. Go to [crusoe.primus-safe.amd.com/hyperloom](https://crusoe.primus-safe.amd.com/hyperloom/).
2. Select **Claw Agent** or **Get Started** to enter PrimusClaw.
3. Choose the **Hyperloom**, **TraceLens-only**, or **GEAK-only** tab for your task.

## Option 2 — Local Mode (Cursor)

Local Mode runs Hyperloom in a remote AMD GPU environment, then connects Cursor
to it. Complete the three steps in order:

1. **Prepare the GPU environment** — an MI300X/MI308X/MI325X/MI355X machine running an
   SGLang or vLLM ROCm image (see [Compatibility Matrix](compatibility.md)).
2. **Connect Cursor** to that environment via Remote SSH / Dev Containers.
3. **Clone and bootstrap the local dependency checkout paths**:

   ```bash
   git clone https://github.com/AMD-AGI/Hyperloom.git
   cd Hyperloom
   cp .env.template .env   # then edit credentials
   export USER_DATA_PATH=/path/to/hyperloom-run
   bash src/hyperloom/inference_optimizer/assets/local_setup.sh
   ```

4. **Before launching an optimization**, source the generated local setup,
   install runtime dependencies, and source the kernel-agent environment in the
   same shell that starts `inference_optimizer optimize`:

   ```bash
   source "$USER_DATA_PATH/runtime/local-setup.env.sh"
   bash src/hyperloom/inference_optimizer/assets/install.sh
   source "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
   ```

`local_setup.sh` clones or updates dependency checkouts and writes
`local-setup.env.sh`; `install.sh` performs runtime dependency installation,
including Magpie, TraceLens, GEAK, Ray, and CLI auth/config files.

For the full step-by-step walkthrough (Docker example, Cursor attach, env
reference), see the [README](https://github.com/AMD-AGI/Hyperloom/blob/main/README.md)
and the [Auth & Environment Guide](ENV_AND_AUTH.md).

## Optional — quantization (AMD Quark)

The optional `--quantize` prelude requires an [AMD Quark](https://quark.docs.amd.com/)
checkout at runtime. Set `QUARK_ROOT` to point at it. See the README's
quantization section for details.

## Related guides

- [Environment & authentication](ENV_AND_AUTH.md)
- [Configuration reference](reference/environment-variables.md)
- [Operations & self-host runbook](OPERATIONS.md)
- [Troubleshooting](reference/troubleshooting.md)
