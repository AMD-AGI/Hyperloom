---
myst:
    html_meta:
        "description": "What to configure to run a hosted multi-branch Hyperloom session on a new cluster: every storage location, whether it is required, what it points at and which side reads it."
        "keywords": "Hyperloom, hosted, multi-branch, shared storage, runner cache, RSI archive, JIT cache, sandbox template, cluster"
---
# Running hosted multi-branch on a new cluster: what to configure

A hosted multi-branch session runs in two places:

- the **app host**: the container that runs `hyperloom-multibranch hosted run` (the session, its agents and the
  toolbox), and
- the **measurement sandboxes**: the sandboxes the grant creates from the `measurement` template, where the runner
  serves and measures each test.

Hyperloom assumes no storage layout. Any cluster with *some* shared storage that both sides can mount works: every
location below comes from a variable or from the campaign spec, and a required one that is missing ends the run
before anything is created, with a final `exit.json` (`"final": true`) whose `error` names the variable.

## App host

These are set in the app container's environment.

| Variable | Required | What it points at |
| --- | --- | --- |
| `USER_DATA_PATH` | yes | Absolute, writable directory. Session directories are created below `<model basename>/` here; put it on persistent storage so a restarted run finds its session again. |
| `HYPERLOOM_RUNNER_CACHE` | yes | Absolute, writable directory on shared storage that the measurement sandboxes also mount. The session builds the runner cache entry of the commit it runs here (`<cache>/<commit>/`) before the first test. This is the path *on the app host*; the sandboxes may mount the same storage elsewhere (see the template's `HYPERLOOM_RUNNER_CACHE` below). |
| `HYPERLOOM_RUNNER_WHEEL_TARGET` | yes | The measurement image's interpreter and platform, for example `cp310-manylinux_2_28_x86_64`, or `cp314-manylinux_2_28_x86_64` for the Python 3.14 `rocm/vllm` images. The runner cache's wheels are built for it, not for the app container, from the lock of that Python: `requirements/runner-cp314.lock` for `cp314`, `requirements/runner.lock` for the others. An entry holds the wheels of one Python, so deployments whose images run different Pythons need different `HYPERLOOM_RUNNER_CACHE` directories; a sandbox whose interpreter is not the entry's fails its start with `runner_start_failed`. |
| `HYPERLOOM_SOURCE_REPO` | yes | Where the run's Hyperloom source is fetched from: `OWNER/NAME` on GitHub, any git URL, or a local path (for example a mirror on shared storage when the app host has no GitHub egress). |
| `HYPERLOOM_SOURCE_SHA` | yes | The full 40-hex commit the run executes. |
| `HYPERLOOM_RSI_ARCHIVE` | yes | Writable directory where sessions archive their search trees for later sessions to learn from. Point every session of a deployment at the same directory to share what they learn; a private directory keeps a session's tree out of the shared archive. There is no default. |
| `INFERENCEX_PATH` | yes, unless `install.sh` left a checkout | A writable InferenceX checkout at the commit Hyperloom pins. Without one the session refuses to start, naming `INFERENCEX_PATH`. |

The model is named by the campaign spec (`model.path`), not by a variable. The app host reads the model's tokenizer
files at that path.

## Measurement sandbox template

These are set in the `measurement` template's environment (or its mounts). They are read inside the sandbox, so they
are independent of the app host's paths: when the template mounts the shared storage at another path, set the
template's values to the paths inside the sandbox.

| Variable | Required | What it points at |
| --- | --- | --- |
| `HYPERLOOM_RUNNER_CACHE` | no | Where the template mounts the runner cache (read-only is enough). Default `/runner-cache`. It must hold the same directory the app host's `HYPERLOOM_RUNNER_CACHE` names: the sandbox runs `<cache>/<HYPERLOOM_SOURCE_SHA>/bootstrap.sh`, installs from the wheels there and never reaches the network for them. A sandbox that does not find the entry fails its start with `runner_start_failed`. |
| `HYPERLOOM_SOURCE_SHA`, `HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256` | set by the session | The session passes both on every create; the template must list them in its accepted create environment. |
| `HYPERLOOM_FRAMEWORK_SNAPSHOT`, `HYPERLOOM_FRAMEWORK_SESSION_SHA256` | set by the session | When the app host has the image's aiter tables, the session snapshots them below `<session_dir>/framework` and passes both on every create; the template must list them in its accepted create environment, and the sandbox must see the session directory at the same path. Without the snapshot, a test that adds tuned rows to a default table is refused as `spec_rejected`. |
| `INFERENCEX_REPO` | no | Where the runner fetches InferenceX (default: upstream on GitHub). A sandbox without egress needs a mirror it can read, for example a bare repository on the shared storage. |
| `MAGPIE_REPO` | no | Where the runner fetches Magpie (default: upstream on GitHub). Same as `INFERENCEX_REPO`. |
| `HYPERLOOM_JIT_CACHE` | no | A read-only directory of compiled kernels shared between tests. Unset: every test builds its kernels cold, which is slower but measures the same. A test that patches aiter's `csrc` or `hsa` is refused without it, because it needs its own JIT directory. |

The sandbox serves the model at the path the campaign spec names, so the template must mount the model storage at
that same path.

## Dev sandbox template

Only a session that lends dev sandboxes (`multibranch.dev_leases`, `--dev-leases`) or runs autonomous agents
(`multibranch.autonomous_agents`, `--autonomous-agent`) uses the `dev` template; it then also needs the grant's `dev` key
(`HYPERLOOM_SANDBOX_KEY_DEV`), and refuses to start without it, and a `dev` template of the session's TP in
`HYPERLOOM_SANDBOX_TEMPLATES`. The session runs only processes in a `dev` sandbox: their
files stay in the session directory, so the template must mount the storage `USER_DATA_PATH` is on at the same path, read
and write, and the model at the path the campaign spec names. The sandbox's commands run as root and are given the app
host's LLM gateway variables (`ANTHROPIC_*`, `OPENAI_*`) and its CA bundle variables (`SSL_CERT_FILE` and the like), whose
paths must exist in the sandbox too. A lease lives as long as its holder needs it, bounded by the grant's
`max_runtime_sec`.

## Example: clusters with an NFS share

On clusters where both the app host and the sandboxes mount one NFS share at `/shared_nfs`, NFS is the fast path and
all values below point into it. The template mounts the share at the same path, so its `HYPERLOOM_RUNNER_CACHE` is
the app host's value.

```shell
# App host
USER_DATA_PATH=/shared_nfs/hyperloom/runs/<run-id>/user-data
HYPERLOOM_RUNNER_CACHE=/shared_nfs/hyperloom/runner-cache
HYPERLOOM_RUNNER_WHEEL_TARGET=cp310-manylinux_2_28_x86_64
HYPERLOOM_SOURCE_REPO=https://github.com/AMD-AGI/Hyperloom.git
HYPERLOOM_RSI_ARCHIVE=/shared_nfs/hyperloom/rsi-archive/
# campaign spec: "model": {"path": "/shared_nfs/models/<model>"}

# Measurement template
HYPERLOOM_RUNNER_CACHE=/shared_nfs/hyperloom/runner-cache
INFERENCEX_REPO=/shared_nfs/hyperloom/mirrors/InferenceX.git
MAGPIE_REPO=/shared_nfs/hyperloom/mirrors/Magpie.git
```

For a measurement image on CPython 3.14 (the current `rocm/vllm` tags), set
`HYPERLOOM_RUNNER_WHEEL_TARGET=cp314-manylinux_2_28_x86_64` and give it a cache of its own, for example
`HYPERLOOM_RUNNER_CACHE=/shared_nfs/hyperloom/runner-cache-cp314` on the app host and in the template. To build an
entry by hand, for example to check an image before a session, run from a checkout with egress (pip 22.3 or newer):

```shell
PYTHONPATH=src python -m hyperloom.measure.runner.cache_build --sha <commit> --repo <checkout> \
    --out /shared_nfs/hyperloom/runner-cache-cp314 --wheel-target cp314-manylinux_2_28_x86_64
```

It prints the entry's `manifest_sha256`, the value a template pins with `HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256`.

On a cluster whose sandboxes mount the shared volume at, say, `/data`, keep the app host's values and set the
template's `HYPERLOOM_RUNNER_CACHE=/data/hyperloom/runner-cache` (and the mirrors below `/data`).
