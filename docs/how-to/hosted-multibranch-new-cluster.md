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

Some lines below describe what the hosted multi-branch entry, `hyperloom-multibranch hosted run`, does: the checks that
make an app-host variable required (all of them except `USER_DATA_PATH` and the source pair, which the host bootstrap
checks) and every statement about what the session does (it builds the runner cache entry, passes the create variables
and takes the framework snapshot). That entry lands with the multi-branch CLI. The sandbox side, the cache layout and the
commands at the end of this page do not depend on it; until it is in place, build the cache entry by hand (see the
example at the end).

## App host

These are set in the app container's environment.

| Variable | Required | What it points at |
| --- | --- | --- |
| `USER_DATA_PATH` | yes | Absolute, writable directory. Session directories are created below `<model basename>/` here; put it on persistent storage so a restarted run finds its session again. |
| `HYPERLOOM_RUNNER_CACHE` | yes, for the hosted multi-branch entry | Absolute, writable directory on shared storage that the measurement sandboxes also mount. The hosted multi-branch entry builds the runner cache entry of the commit it runs here (`<cache>/<commit>/`) before the first test. This is the path *on the app host*; the sandboxes may mount the same storage elsewhere (see the template's `HYPERLOOM_RUNNER_CACHE` below). |
| `HYPERLOOM_RUNNER_WHEEL_TARGET` | yes, for the hosted multi-branch entry | The measurement image's interpreter and platform, for example `cp310-manylinux_2_28_x86_64`, or `cp314-manylinux_2_28_x86_64` for the Python 3.14 `rocm/vllm` images. The hosted multi-branch entry reads it and builds the runner cache's wheels for it, not for the app container, from the lock of that Python: `requirements/runner-cp314.lock` for `cp314`, `requirements/runner.lock` for the others. An entry holds the wheels of one Python, so deployments whose images run different Pythons need different `HYPERLOOM_RUNNER_CACHE` directories; a sandbox whose interpreter is not the entry's fails its start with `runner_start_failed`. A build by hand passes the tag as `--wheel-target` (see the example at the end). |
| `HYPERLOOM_SOURCE_REPO` | yes | Where the run's Hyperloom source is fetched from: `OWNER/NAME` on GitHub, any git URL, or a local path (for example a mirror on shared storage when the app host has no GitHub egress). |
| `HYPERLOOM_SOURCE_SHA` | yes | The full 40-hex commit the run executes. |
| `HYPERLOOM_RSI_ARCHIVE` | yes, for the hosted multi-branch entry | Writable directory where sessions archive their search trees for later sessions to learn from. Point every session of a deployment at the same directory to share what they learn; a private directory keeps a session's tree out of the shared archive. There is no default. |
| `INFERENCE_SUITE_TOOL_CACHE` | no | The tool cache `install.sh` staged the pinned InferenceX client in. Default `inference-suite-tools` under the dependency cache. Without a staging there that verifies, the session refuses to start, naming `python -m inference_suite stage --tool-cache <dir>`. |

The model is named by the campaign spec (`model.path`), not by a variable. The app host reads the model's tokenizer
files at that path.

## Measurement sandbox template

These are set in the `measurement` template's environment (or its mounts). They are read inside the sandbox, so they
are independent of the app host's paths: when the template mounts the shared storage at another path, set the
template's values to the paths inside the sandbox.

| Variable | Required | What it points at |
| --- | --- | --- |
| `HYPERLOOM_RUNNER_CACHE` | no | Where the template mounts the runner cache. Default `/runner-cache`. For a v2 sheet the runner uses the inference_suite staging `cache_build` published in `inference-suite/<staging key>/` (see [The inference_suite staging](#the-inference_suite-staging)) offline and builds none, so the whole cache can be read-only. It must hold the same directory the app host's `HYPERLOOM_RUNNER_CACHE` names. How the sandbox starts from it is described below the table. |
| `HYPERLOOM_SOURCE_SHA`, `HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256` | set by the hosted multi-branch entry | The hosted multi-branch entry passes both on every create; the template must list them in its accepted create environment. |
| `HYPERLOOM_FRAMEWORK_SNAPSHOT`, `HYPERLOOM_FRAMEWORK_SESSION_SHA256` | set by the hosted multi-branch entry | When the app host has the image's aiter tables, the hosted multi-branch entry snapshots them below `<session_dir>/framework` and passes both on every create; the template must list them in its accepted create environment, and the sandbox must see the session directory at the same path. Without the snapshot, a test that adds tuned rows to a default table is refused as `spec_rejected`. |
| `INFERENCEX_REPO` | no | Where the runner fetches InferenceX (default: upstream on GitHub). A sandbox without egress needs a mirror it can read, for example a bare repository on the shared storage. |
| `HYPERLOOM_JIT_CACHE` | no | A read-only directory of compiled kernels shared between tests. Unset: every test builds its kernels cold, which is slower but measures the same. A test that patches aiter's `csrc` or `hsa` is refused without it, because it needs its own JIT directory. |

The sandbox serves the model at the path the campaign spec names, so the template must mount the model storage at
that same path.

### Starting the runner from the cache

Start the runner with the step that `hyperloom.measure.runner.bootstrap.runner_template_step(...)` returns. It needs
`HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256` in the sandbox's environment, checks the entry's `manifest` against it and
`bootstrap.sh` against that manifest, runs the checked copy, and returns at once so the sandbox turns ready. A plain
`sh <cache>/<HYPERLOOM_SOURCE_SHA>/bootstrap.sh` step is safe only when no account but the builder can write the cache:
the script is what checks the manifest, so it cannot check itself.

The script reads the entry's `source.tar` once and unpacks it into the sandbox's scratch directory only when those bytes
are the archive the manifest records. It copies the wheels there, checking each against the manifest as it copies,
installs only those copies (`pip --no-index`), then runs the runner from the unpacked source; the scratch directory
needs room for the unpacked source and the wheels. Nothing the runner or pip runs comes from the cache unchecked. A
profile test later copies the staged ruppell tree (`<cache>/tools/ruppell-<commit>/`) and builds the copy only when it
has the pinned digest. The prompt sets a test reads (`<cache>/evalsets/<key>/`) are data, and they are read from the
cache again after their check. The start-up makes no network access; the runner still fetches InferenceX
(`INFERENCEX_REPO` above).

With the checked step, a sandbox that does not find the entry, finds one that does not match the pin, or has no pin
fails its start with `runner_start_failed`. An entry built before entries had `source.tar` is refused where it is built:
the build fails with `CacheConflict`.

## Dev sandbox template

Only a session that lends dev sandboxes (`multibranch.dev_leases`, `--dev-leases`) or runs autonomous agents
(`multibranch.autonomous_agents`, `--autonomous-agent`) uses the `dev` template; it then also needs the grant's `dev` key
(`HYPERLOOM_SANDBOX_KEY_DEV`) and its limits (`HYPERLOOM_GRANT_LIMITS`), and refuses to start without them, and a `dev`
template of the session's TP in `HYPERLOOM_SANDBOX_TEMPLATES`. The session runs only processes in a `dev` sandbox: their
files stay in the session directory, so the template must mount the storage `USER_DATA_PATH` is on at the same path, read
and write, and the model at the path the campaign spec names. The sandbox's commands run as root and are given the app
host's LLM gateway variables (`ANTHROPIC_*`, `OPENAI_*`) and its CA bundle variables (`SSL_CERT_FILE` and the like), whose
paths must exist in the sandbox too. The gateway variables include its credentials, which any command in the sandbox can
read, so give the app host a gateway key that agents' commands may see. A lease lives as long as its holder needs it,
bounded by the grant's `max_runtime_sec`.

## Example: clusters with an NFS share

On clusters where both the app host and the sandboxes mount one NFS share at `/mnt/shared`, NFS is the fast path and
all values below point into it. The template mounts the share at the same path, so its `HYPERLOOM_RUNNER_CACHE` is
the app host's value.

```shell
# App host
USER_DATA_PATH=/mnt/shared/hyperloom/runs/<run-id>/user-data
HYPERLOOM_RUNNER_CACHE=/mnt/shared/hyperloom/runner-cache
HYPERLOOM_RUNNER_WHEEL_TARGET=cp310-manylinux_2_28_x86_64
HYPERLOOM_SOURCE_REPO=https://github.com/AMD-AGI/Hyperloom.git
HYPERLOOM_RSI_ARCHIVE=/mnt/shared/hyperloom/rsi-archive/
# campaign spec: "model": {"path": "/mnt/shared/models/<model>"}

# Measurement template
HYPERLOOM_RUNNER_CACHE=/mnt/shared/hyperloom/runner-cache
INFERENCEX_REPO=/mnt/shared/hyperloom/mirrors/InferenceX.git
```

For a measurement image on CPython 3.14 (the current `rocm/vllm` tags), set
`HYPERLOOM_RUNNER_WHEEL_TARGET=cp314-manylinux_2_28_x86_64` and give it a cache of its own, for example
`HYPERLOOM_RUNNER_CACHE=/mnt/shared/hyperloom/runner-cache-cp314` on the app host and in the template. To build an
entry by hand, for example to check an image before a session, run from a checkout with egress (pip 22.3 or newer):

```shell
PYTHONPATH=src python -m hyperloom.measure.runner.cache_build --sha <commit> --repo <checkout> \
    --out /mnt/shared/hyperloom/runner-cache-cp314 --wheel-target cp314-manylinux_2_28_x86_64
```

Give `--wheel-target` explicitly. The command does not read `HYPERLOOM_RUNNER_WHEEL_TARGET` (only `--out` defaults from
the environment, to `HYPERLOOM_RUNNER_CACHE`), and without it the wheels are built for the builder's own interpreter. Run
it with `umask 022` so the sandbox's account can read the entry, on a file system that supports `flock` (NFSv4 does).

It prints the entry's `manifest_sha256`, the value a template pins with `HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256`,
and the directory of the inference_suite staging it published.
A finished entry is never modified: a build of the same commit into the same directory from a checkout whose
`bootstrap.sh` differs (it embeds that checkout's `bootstrap.py` and `cache_layout.py`) is refused with `CacheConflict`,
naming both `bootstrap.sh` hashes, and so is one for another wheel target, because a campaign may have pinned that
entry's manifest. Build by hand from another checkout into a `HYPERLOOM_RUNNER_CACHE` directory of its own.

### The inference_suite staging

After the entry, `cache_build` publishes the commit's inference_suite staging, the AgentX client and InferenceX subset
a v2 test's rounds run, with the commit's own `inference_suite` (`python -m inference_suite publish`). It downloads the
pinned python-build-standalone CPython 3.12 build and the wheels of `src/inference_suite/agentx-client.lock`, each
checked against its pinned sha256, fetches InferenceX at its pin, installs the client with `--no-deps` from those
files alone, and patches its aiperf to export `error_osl` (a client the patch does not apply to fails the build). The builder needs egress to GitHub and PyPI; the sandbox needs none.

```text
<HYPERLOOM_RUNNER_CACHE>/inference-suite/                  2775, group of the cache (setgid)
    <staging key>/                                         published: never written again
        staging_manifest.json
        inferencex/<InferenceX pin>/infx/...               the InferenceX subset
        inferencex/<InferenceX pin>/agentx-python/         the CPython build with the client installed
    .staging-<staging key>-<uuid>/                         a build in progress; no reader looks inside
```

The staging key is the sha256 of the InferenceX pin, the interpreter build, the client lock and the client's patch, so a
change to any of them publishes a new directory beside the old one. Builders share the directory without a lock: each builds in its own
`.staging-*` directory, syncs it and renames it to the key. A builder that loses the rename removes its own directory
and uses the winner once the winner's normalised manifest (each package's name, version and wheel sha256, the
interpreter and the InferenceX pin) is the one it built; one that differs is refused, naming the packages. A
`.staging-*` directory left by a builder that died can be removed once no build is running.

Every account that runs `cache_build` must be in the cache's group, and the cache root must be setgid in it
(`chgrp <group> <cache> && chmod 2775 <cache>`); `inference-suite/` is then created `2775` in that group, so builders
of different accounts can each rename into it. Run the build with `umask 022` so the sandbox's account can read the
staging. A test whose application staging differs from the runner's published one fails with `preflight_mismatch`,
naming each package that differs.

A new cluster also needs two more things staged into the runner cache, each with its own command, run where the cache
is written (the prompt set on a host that has the model's tokenizer files). Without the prompt set, a test that asks for
the short-prompt consistency check cannot load it; without the ruppell tree, a profile test runs without ruppell:

```shell
PYTHONPATH=src python -m hyperloom.measure.short_prompts.build --model /mnt/shared/models/<model> \
    --cache /mnt/shared/hyperloom/runner-cache
PYTHONPATH=src python -m hyperloom.measure.ruppell_stage --archive <tar.gz path or URL> \
    --cache /mnt/shared/hyperloom/runner-cache
```

`--commit` of the second command defaults to the commit a profile test pins, and `--archive` is `git archive`'s tarball
of that commit, a file or a URL. Run both with `umask 022` too: they create the entry's directories and manifest with
the process's umask, and the sandbox's account must be able to read them.

On a cluster whose sandboxes mount the shared volume at, say, `/data`, keep the app host's values and set the
template's `HYPERLOOM_RUNNER_CACHE=/data/hyperloom/runner-cache` (and the mirrors below `/data`).
