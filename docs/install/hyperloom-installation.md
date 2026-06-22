---
myst:
    html_meta:
        "description": "Install Hyperloom through the hosted UI or in Local Mode on an AMD GPU environment. Covers prerequisites, Cursor setup, and the optional AMD Quark quantization prelude."
        "keywords": "Hyperloom, installation, AMD GPU, Local Mode, Cursor, PrimusClaw, ROCm, LLM inference, GEAK, TraceLens, Magpie, SGLang, vLLM, AMD Quark, quantization"
---
# Install Hyperloom

Hyperloom can be used two ways: through the hosted Hyperloom UI (no local
setup), or in Local Mode on a remote AMD GPU environment driven from Cursor.

```{note}
The canonical, always-current installation walkthrough lives in the repository
[`README.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/README.md). This
topic summarizes the entry points and links to the detailed guides.
```

## Prerequisites

Bind your [LLM Gateway](https://llm.amd.com/) key to
[Hyperloom](https://core42.primus-safe.amd.com/hyperloom/) to obtain your API
key. This key provides access to TraceLens, GEAK, and OOB services for both the
UI and the local workflow.

## Option 1 — Hosted UI (PrimusClaw)

The fastest path, with no local GPU setup:

1. Go to [core42.primus-safe.amd.com/hyperloom](https://core42.primus-safe.amd.com/hyperloom/).
2. Select **Claw Agent** or **Get Started** to enter PrimusClaw.
3. Select the **Hyperloom**, **TraceLens-only**, or **GEAK-only** tab for your task.

## Option 2 — Local Mode (Cursor)

Local Mode runs Hyperloom in a remote AMD GPU environment, then connects Cursor
to it. Complete these steps:

1. Prepare the GPU environment — Use a MI300X/MI308X/MI325X/MI355X machine running an
   SGLang or vLLM ROCm image (see [Hyperloom compatibility matrix](../compatibility.md)).
2. Connect Cursor to that environment using Remote SSH or Dev Containers.
3. Clone and bootstrap the local dependency checkout paths:

   ```bash
   git clone https://github.com/AMD-AGI/Hyperloom.git
   cd Hyperloom
   cp .env.template .env   # then edit credentials
   export USER_DATA_PATH=/path/to/hyperloom-run
   bash inference_optimizer/scripts/local_setup.sh
   ```

4. Before launching an optimization, source the generated local setup,
   install the runtime dependencies, and source the kernel-agent environment in the
   same shell that starts `inference_optimizer optimize`:

   ```bash
   source "$USER_DATA_PATH/runtime/local-setup.env.sh"
   bash inference_optimizer/scripts/install.sh
   source "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
   ```

`local_setup.sh` clones or updates dependency checkouts and writes
`local-setup.env.sh`; `install.sh` performs runtime dependency installation,
including Magpie, TraceLens, GEAK, Ray, and CLI auth/config files.

For the full step-by-step walkthrough (Docker example, Cursor attach, env
reference), see the [README](https://github.com/AMD-AGI/Hyperloom/blob/main/README.md)
and the [Hyperloom authentication and credentials](../reference/authentication.md).

## Optional — quantization (AMD Quark)

The optional `--quantize` prelude requires an [AMD Quark](https://quark.docs.amd.com/)
checkout at runtime. Set `QUARK_ROOT` to point at it. See the Hyperloom README's
[quantization section](https://github.com/AMD-AGI/Hyperloom/blob/main/README.md#quantization-optional-amd-quark-dependency) for more information.

## Related guides

Use these guides for more detailed configuration and operational information:

- [Hyperloom authentication and credentials](../reference/authentication.md)
- [Environment variables](../reference/environment-variables.md)
- [Hyperloom self-hosting and operations guide](../reference/operations.md)
- [Troubleshooting Hyperloom](../troubleshooting.md)
