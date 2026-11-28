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
| `USER_DATA_PATH` | yes | Absolute, writable directory. Session directories are created below `<model basename>/` here; put it on persistent storage so a restarted run finds its session again. When the legs of a run may be different host accounts of one group, make it, and the `HYPERLOOM_STATE_DIR` the host gives the run, a directory of that group with the setgid bit (`chmod 2775`): what a leg makes below them then takes the group, and the next leg's account can write there; the entry warns, naming each, when one is not setgid, and refuses one this account cannot write. Below a directory that other accounts can write (that group's, `/tmp`), what a leg writes there never goes through a link, since any account of the group could have put one there: give these two paths, and everything below them, without links in such a directory. |
| `HYPERLOOM_USER_DATA_PATH` | no | A shared data root that replaces `USER_DATA_PATH` for the run, for a host that reserves `USER_DATA_PATH` to itself and points it at the submitting user's directory. The host bootstrap exports it as `USER_DATA_PATH` before its checks, so everything said about `USER_DATA_PATH` here applies to it: sessions go to `<root>/<model basename>/MB-<time>-<id>`, every account of the run must be able to write it, make it `chmod 2775` for the deployment's group, and the dev template must mount its storage at the same path. A restarted run finds its session through `session.json` in the state directory, so changing it moves only new sessions. |
| `HYPERLOOM_RUNNER_CACHE` | yes, for the hosted multi-branch entry | Absolute, writable directory on shared storage that the measurement sandboxes also mount. The hosted multi-branch entry builds the runner cache entry of the commit it runs here (`<cache>/<commit>/`) before the first test. When the legs of a run may be different host accounts of one group, make it a directory of that group with the setgid bit (`chmod 2775`): every builder then takes the `.<commit>.flock` lock (mode `0664`). An entry is built in a new temporary directory of the builder's own (`$TMPDIR`, which needs room for one entry and its build environment), never in this directory, and copied in writable by its builder alone; a `.<commit>.tmp-*` directory that a builder killed while copying leaves is removed only by a later builder of the same account, and others build beside it. This is the path *on the app host*; the sandboxes may mount the same storage elsewhere (see the template's `HYPERLOOM_RUNNER_CACHE` below). |
| `HYPERLOOM_RUNNER_WHEEL_TARGET` | yes, for the hosted multi-branch entry | The measurement image's interpreter and platform, for example `cp310-manylinux_2_28_x86_64`, or `cp314-manylinux_2_28_x86_64` for the Python 3.14 `rocm/vllm` images. The hosted multi-branch entry reads it and builds the runner cache's wheels for it, not for the app container, from the lock of that Python: `requirements/runner-cp314.lock` for `cp314`, `requirements/runner.lock` for the others. An entry holds the wheels of one Python, so deployments whose images run different Pythons need different `HYPERLOOM_RUNNER_CACHE` directories; a sandbox whose interpreter is not the entry's fails its start with `runner_start_failed`. A build by hand passes the tag as `--wheel-target` (see the example at the end). |
| `HYPERLOOM_RUNNER_GPU_CARD` | yes, when the tests run in the grant's sandboxes | Path, on the app host, to a recording of `amd-smi static --json` taken on a node the measurement sandboxes run on: run `amd-smi static --json > runner-gpu-card.json` there once, inside a sandbox or on the node itself, and put the file anywhere the app host reads, for example beside the runner cache on shared storage. The app container holds none of the runners' cards, so the session takes its card from this file: the base launch is chosen for the card's arch, the roofline and TraceLens' vendor peaks come from its compute units, clock and bandwidth, and every measurement sheet carries it. Every card it describes must be one kind (arch, compute units, memory, top engine clock). When the variable is unset, the file is missing or unreadable, or it describes no card or more than one kind, the session does not start and the error names the variable. A runner whose own cards are another kind than the file's fails every test at preflight (`preflight_mismatch`, `GPU 0 is <its card>, not <the file's card>`), so record the file again when the grant moves to other cards. A run measured on the app container's own cards (below) reads those and ignores this variable. |
| `HYPERLOOM_SOURCE_REPO` | yes | Where the run's Hyperloom source is fetched from: `OWNER/NAME` on GitHub, any git URL, or a local path (for example a mirror on shared storage when the app host has no GitHub egress). |
| `HYPERLOOM_SOURCE_SHA` | yes | The full 40-hex commit the run executes. |
| `HYPERLOOM_RSI_ARCHIVE` | yes, for the hosted multi-branch entry | Writable directory where sessions archive their search trees for later sessions to learn from. Point every session of a deployment at the same directory to share what they learn; a private directory keeps a session's tree out of the shared archive. There is no default. When the sessions may be different host accounts of one group, make it a directory of that group with the setgid bit (`chmod 2775`). |
| `HYPERLOOM_JIT_STORE` | no | Absolute directory of the shared compiled-code cache, on the app host's own disk (for example a `0700` directory below its temporary directory). Where tests run in the grant's measurement sandboxes, no sandbox reads or mounts it: before a test's server starts, its runner asks the app for the cache keys it needs (`/out/jit-want.json`), and the app proves the stored snapshots and writes one tar per kind into the sandbox (`/in/jit-cache/`), in parts of at most 30 MiB (the sandbox file API refuses a single write over 32 MB), each kind on its own so that one that fails does not stop the others, where the runner checks every file again before it is used. After a test's signed result verified, the app publishes into the store what that test compiled. Unset, no store is made or pruned for those sessions. **With `placement.measurement: local`** (see "One node" below) the session uses the store as `hyperloom-multibranch run` does: its runners, in the app container, read it, the app publishes into it what each accepted test compiled, and it is checked and pruned as a session starts. Unset, the store is `hyperloom-jit-store-<uid>` below the app container's temporary directory, with the same fixed bounds. **Do not put the store on storage any sandbox can write** (such as the shared tree the `dev` and `measurement` templates mount): code a candidate runs there, as root, could put snapshots into it that later tests load, and no check on the app can tell them apart. The cache lives as long as the app's disk does; the first tests of a session on a new app host build cold. As a session starts it prunes keys no test has used for 14 days, or the least recently used ones while what the live keys list comes to more than 200 GiB; a pruned key's files are removed two days later, so a running test never loses what it is reading. The bound is on the live keys only: pruned keys and blobs no key lists wait out their two days uncounted, so the directory can hold more than 200 GiB meanwhile. A prune takes at most 60 s of a session's start: it first removes pruned keys past their two days, and where it stopped is kept in the store (`prune-cursor.json`), so later starts go on from there and cover the whole store. The size bound needs every key read within one start; a store whose keys take longer than 60 s to read is never bounded by size, and each such start logs a warning saying so. These bounds (14 days unused, 200 GiB of live keys, two days before removal, 60 s per start) are fixed, not set by the deployment: put the store, or with `placement.measurement: local` and the variable unset the temporary directory, on a disk with room for more than 200 GiB. It must be a directory of the account the app runs as that only it can write (made `0700` when missing), with no directory above it that another account can change unless it is sticky. A store the app refuses, or cannot make or check, is not used: the session logs one warning naming the store and why, and the run goes on without it. Where tests run in the grant's sandboxes that session neither serves nor publishes, and every test builds cold (a candidate that patches aiter's `csrc` or `hsa` is refused by its runner); with `placement.measurement: local` its runners are told no cache, nothing is published, every test builds cold, and candidates that patch aiter's `csrc` or `hsa` are refused for the whole session. |
| `INFERENCE_SUITE_TOOL_CACHE` | no | The tool cache `install.sh` staged the pinned InferenceX client in. Default `inference-suite-tools` under the dependency cache. Without a staging there that verifies, the session refuses to start, naming `python -m inference_suite stage --tool-cache <dir>`. |
| `HYPERLOOM_HOST_LAYER` | yes, unless `HYPERLOOM_SPECIALIST_USER` names an account the app can run agents as | The hosting layer's directory. A hosted run launches an agent under bubblewrap only with the layer's `bin/bwrap`, never one on `PATH` (the framework image may carry its own). Where the container's root mount is unbindable (apptainer), bubblewrap runs only under the layer's `bin/unshare`, never one on `PATH`; the layer image does not build it, so a deployment there adds a static util-linux `unshare` whose `--help` lists `--user`, `--map-current-user`, `--mount` and `--propagation` (2.37.2 does) to the layer directory. |

The model is named by the campaign spec (`model.path`), not by a variable. The app host reads the model's tokenizer
files at that path.

A hosted run starts nothing unconfined in the app container, beside its credentials: its specialists run as the account
`HYPERLOOM_SPECIALIST_USER` names where it can be used, else under the layer's bubblewrap (`HYPERLOOM_HOST_LAYER`).
Where neither can isolate them, the run ends before anything starts, naming both reasons (exit code 1); a bubblewrap
that gives no answer within 10 s leaves the host to start the run again. Where the root mount is unbindable, a root app
runs every launch under `unshare --mount`, and an app that is not root under `unshare --user --map-current-user --mount
--propagation private` (the layer's `bin/unshare`, above). Under that user-namespace tier no launch can join another
sandbox's namespaces: the specialists are confined, but a run measured on this container's cards is refused (see
below), and a run measured in the grant's sandboxes starts.

Each leg of a run may be another host account of one group. The entry checks `HYPERLOOM_STATE_DIR` and
`USER_DATA_PATH` before it hands over (see the table above); before a session starts,
`hyperloom-multibranch hosted run` then makes `HYPERLOOM_RUNNER_CACHE` mode `2775` when it is missing, creates and
removes a file in it and in `HYPERLOOM_RSI_ARCHIVE`, and ends the run with one message naming every one it cannot
write, with its errno, owner and mode. One that is not setgid is a warning: what a leg makes there takes that leg's
group.

## Measurement sandbox template

These are set in the `measurement` template's environment (or its mounts). They are read inside the sandbox, so they
are independent of the app host's paths: when the template mounts the shared storage at another path, set the
template's values to the paths inside the sandbox.

| Variable | Required | What it points at |
| --- | --- | --- |
| `HYPERLOOM_RUNNER_CACHE` | no | Where the template mounts the runner cache. Default `/runner-cache`. For a v2 sheet the runner uses the inference_suite staging `cache_build` published in `inference-suite/<staging key>/` (see [The inference_suite staging](#the-inference_suite-staging)) offline and builds none, so the whole cache can be read-only. A read-only cache keeps no InferenceX commit in `tools/InferenceX.git`: every test then logs a warning and fetches InferenceX from `INFERENCEX_REPO` (or upstream) again, so leave `tools/` writable where that fetch is slow or can fail. It must hold the same directory the app host's `HYPERLOOM_RUNNER_CACHE` names. How the sandbox starts from it is described below the table. |
| `HYPERLOOM_SOURCE_SHA`, `HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256` | set by the hosted multi-branch entry | The hosted multi-branch entry passes both on every create; the template must list them in its accepted create environment. |
| `HYPERLOOM_FRAMEWORK_SNAPSHOT`, `HYPERLOOM_FRAMEWORK_SESSION_SHA256` | set by the hosted multi-branch entry | When the app host has the image's aiter tables, the hosted multi-branch entry snapshots them below `<session_dir>/framework` and passes both on every create; the template must list them in its accepted create environment, and the sandbox must see the session directory at the same path. Without the snapshot, a test that adds tuned rows to a default table is refused as `spec_rejected`. |
| `HYPERLOOM_BASE_BINS` | set by the hosted multi-branch entry | Only for a session whose framework's server is built from its pinned checkout (`--framework llamacpp`): the entry lays that base build below `<session_dir>` and passes the directories of its executables on every create, and the runner puts them first on its `PATH`, so a test that does not rebuild the framework runs that build. The template must list it in its accepted create environment, and the sandbox must see the session directory at the same path. |
| `INFERENCEX_REPO` | no | Where the runner fetches InferenceX (default: upstream on GitHub). A sandbox without egress needs a mirror it can read, for example a bare repository on the shared storage. A path or `file://` URL may name a mirror another account owns: the runner fetches from its resolved path, and that fetch alone runs with a global git config of its own whose `safe.directory` names that path and its `.git`, so no system-wide `safe.directory` is needed. |
| `HYPERLOOM_JIT_CACHE` | no | Leave it unset in the `measurement` template: its runners are served the compiled-code cache by the app (see `HYPERLOOM_JIT_STORE`). A runner that has it set reads that directory directly and does not ask the app, which is only for a runner on the store's own machine (the local toolbox sets it for its own runners, as in a run with `placement.measurement: local`). |
| `HYPERLOOM_WEIGHT_PREFETCH` | no | `1` reads the model's checkpoint into the page cache while each server boots, for a model on a network file system. Unset, it is off. With it on, the round that boots the server leaves `weight_prefetch.json` in its run_dir. See [Checkpoint prefetch](../reference/environment-variables.md#checkpoint-prefetch). |

The sandbox serves the model at the path the campaign spec names, so the template must mount the model storage at
that same path.

### One sandbox per test

Every test runs in a measurement sandbox of its own, closed when the test ends, and the session runs as many tests at
once as the grant fits at the session's TP: `min(max_gpus // tp, max_sandboxes)`. On a cluster that queues sandboxes,
each test's sandbox therefore queues for a node. `multibranch.make_before_break` in a campaign spec is accepted and
ignored, with a warning in the log.

### AgentX window and boot budget

The `multibranch` block of the campaign spec carries the options of `run` that set how long a test runs:

```json
"workload": {"conc": 16, "precision": "fp8"},
"multibranch": {
  "workload": "agentx",
  "agentx_duration_sec": 3600,
  "agentx_objective": "p50_interactivity",
  "boot_budget_sec": 5400
}
```

- `agentx_duration_sec` (`--agentx-duration-sec`), a positive whole number of seconds: the trace window each test
  replays; the harness's default (3600 s) when not given.
- `agentx_objective` (`--agentx-objective`), `p50_interactivity` (the default) or `total_token_throughput`: the axis
  the session is graded on.
- `boot_budget_sec` (`--boot-budget-sec`), a positive whole number of seconds, at most 21600: how long, at least, the
  server of a test may take to be ready; see [Boot budget](run-multibranch.md#options-of-run).

The two AgentX fields are taken with `multibranch.workload: agentx` only; with another workload, or a value of the
wrong type, the run ends for good (exit code 2), naming the field. A `boot_budget_sec` only raises the limits the
session sets, and the server's own start timeout follows it: a round sets that timeout to the budget plus 3600 s when
that is above its usual 10800 s.

### Starting the runner from the cache

Start the runner with the step that `hyperloom.measure.runner.bootstrap.runner_template_step(...)` returns. It needs
`HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256` in the sandbox's environment, checks the entry's `manifest` against it and
`bootstrap.sh` against that manifest, runs the checked copy, and returns at once so the sandbox turns ready. A plain
`sh <cache>/<HYPERLOOM_SOURCE_SHA>/bootstrap.sh` step is safe only when no account but the builder can write the cache:
the script is what checks the manifest, so it cannot check itself.

The step makes `/in`, `/out` and `/scratch` (or the directories the template names); the runner reads the in
directory and removes the compiled-code cache tars served there once unpacked (an in directory it cannot write keeps
them), and it must be able to write the out and scratch directories. On an image whose root is read-only the
platform must bind those writable into the sandbox. A scratch or in directory the step cannot make, or a scratch
directory it cannot write, ends the start with a `runner_start_failed` status that names it; when the out directory
itself cannot be made or written, no status can be left there, so the step prints the reason on stderr and exits 1. The
runner's boot log goes to `/runner-boot.log` (or the template's `boot_log`) when the step can write it, else to
`runner-boot.log` in the out directory; when neither can be written the start ends with that reason. A template embeds
the step when it is built, so an existing template keeps the old step, and its boot log at `/runner-boot.log`, until it
is rebuilt.

The script reads the entry's `source.tar` once and unpacks it into the sandbox's scratch directory only when those bytes
are the archive the manifest records. It copies the wheels there, checking each against the manifest as it copies,
installs only those copies (`pip --no-index`), then runs the runner from the unpacked source; the scratch directory
needs room for the unpacked source and the wheels. Nothing the runner or pip runs comes from the cache unchecked.
The prompt sets a test reads (`<cache>/evalsets/<key>/`) are data, and they are read from the
cache again after their check. The start-up makes no network access; the runner still fetches InferenceX
(`INFERENCEX_REPO` above).

With the checked step, a sandbox that does not find the entry, finds one that does not match the pin, or has no pin
fails its start with `runner_start_failed`. An entry built before entries had `source.tar` is refused where it is built:
the build fails with `CacheConflict`.

## The session's base launch

A hosted multi-branch session resolves its base launch as `hyperloom-multibranch run` does: from InferenceX
(`multibranch.inferencex_commit` pins the commit), then the framework's cookbook, then the default table. When none of
them gives a launch that starts on the cluster's hardware, give the launch yourself as `multibranch.reference_script`,
the path of a shell launch script on the app host (for example on the storage `USER_DATA_PATH` is on), which the
session freezes as its base as `--reference-script` does. It wins over every recipe source. Its literal `export
NAME=value` lines are the server's environment, and its server command line carries the flags; the command's TP and EP
must be the spec's `hardware.tp` and `hardware.ep`, and an export or a flag set from a shell variable is refused.

```shell
# /mnt/shared/hyperloom/launch/dsv4-flash-mi355x.sh
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_USE_AITER=1
python3 -m sglang.launch_server --model-path /mnt/shared/models/DeepSeek-V4-Flash --tp 8 --ep-size 8 \
    --trust-remote-code --mem-fraction-static 0.8
```

```json
"framework": {"name": "sglang", "image": "<image>"},
"hardware": {"gpu_type": "mi355x", "tp": 8, "ep": 8},
"multibranch": {"workload": "inferencex", "reference_script": "/mnt/shared/hyperloom/launch/dsv4-flash-mi355x.sh"}
```

A spec does not take `launch.server_args` or `launch.extra_env`, which only the retired flow `single` read: a spec
that sets either ends the run (exit code 2), naming `multibranch.reference_script`, instead of running without them. A
script that cannot serve as the base ends the run the same way, naming why.

## Dev sandbox template

Only a session that lends dev sandboxes (`multibranch.dev_leases`, `--dev-leases`) or runs autonomous agents
(`multibranch.autonomous_agents`, `--autonomous-agent`) uses the `dev` template; a session with neither reads no `dev`
template and starts whether or not the host gives it a `dev` key. A session that lends needs the grant's `dev` key
(`HYPERLOOM_SANDBOX_KEY_DEV`), its limits (`HYPERLOOM_GRANT_LIMITS`), a `dev` template of the session's TP in
`HYPERLOOM_SANDBOX_TEMPLATES`, and a grant that fits one dev sandbox beside one test (`max_gpus` at least twice the TP,
`max_sandboxes` at least 2); without any of them the run ends before it starts (exit code 2), naming what is missing,
instead of being restarted. The session runs only processes in a `dev` sandbox: their
files stay in the session directory, so the template must mount the storage `USER_DATA_PATH` is on at the same path, read
and write, and the model at the path the campaign spec names. The sandbox's commands run as root and are given the app
host's LLM gateway variables (`ANTHROPIC_*`, `OPENAI_*`) and its CA bundle variables (`SSL_CERT_FILE` and the like), whose
paths must exist in the sandbox too. The gateway variables include its credentials, which any command in the sandbox can
read, so give the app host a gateway key that agents' commands may see. A lease lives as long as its holder needs it,
bounded by the grant's `max_runtime_sec`.

## One node: measuring on the app container's own cards

A spec with `placement.measurement: local` and `placement.profile: hosted-gpu` measures on the cards of the app
container itself, as `hyperloom-multibranch run` does on a machine's: tests, dev sandboxes and autonomous agents all
run in the app container, so the run needs no measurement or `dev` template, no sandbox key, no
`HYPERLOOM_SOURCE_REPO`/`HYPERLOOM_SOURCE_SHA` for the runner cache, and no grant. `placement.measurement: local` with
`hosted-cpu` is refused. What changes on the app host:

- Give the container the node's cards. As many tests run at once as its cards hold at the spec's TP, at most
  `placement.max_parallel_tests`.
- Every test runs the app container's own framework, so `HYPERLOOM_ACTIVE_IMAGE_REF` and
  `HYPERLOOM_ACTIVE_IMAGE_DIGEST` name the image the app container runs, and the spec's `framework.image` must name it by
  that digest (`<repository>@sha256:<digest>`); any other image is refused (exit code 2), and missing pins end the run
  (exit code 1).
- `HYPERLOOM_GRANT_LIMITS` is optional. Without it the spec's `budget.max_hours` bounds the session; with a
  `max_runtime_sec`, the earlier of the two does. Its other totals are not read.
- `HYPERLOOM_RUNNER_CACHE` is still required: the session stages its prompt sets and the inference_suite staging
  every test runs there.
- `HYPERLOOM_JIT_STORE` is used as `hyperloom-multibranch run` uses it: the tests in this container read the store,
  the app publishes into it what each accepted test compiled, and it is checked and pruned as a session starts. Unset,
  the store is `hyperloom-jit-store-<uid>` below the container's temporary directory, pruned with the same fixed bounds
  (200 GiB of live keys), so that directory needs the room. A store the app refuses leaves the run without it, and
  candidates that patch aiter's `csrc` or `hsa` are then refused for the whole session (see the table above).
- The agent and dev sandboxes and the tests run in this container, beside the run's credentials, so the run starts only
  where they run isolated, as the specialists must: under the layer's bubblewrap (`HYPERLOOM_HOST_LAYER`). Where it
  cannot sandbox them, the run ends before anything starts, naming them and why (exit code 1). Their commands join a
  holder's namespaces, which bubblewrap cannot do under the `unshare --user` tier (an app that is not root on a node
  whose root mount is unbindable): there the run is refused at start even though the specialists are confined
  (`a launch cannot join another sandbox's namespaces here: ...`); measure such a deployment in the grant's sandboxes.
- Without a visible-device mask, the cards are as many as `rocm-smi` shows, and a container that holds fewer render
  nodes (`/dev/dri/renderD*`) than that is refused, naming both counts: set `ROCR_VISIBLE_DEVICES` to the cards it was
  given.
- The session's card is read on this container (`amd-smi`, else `rocm-smi`, else torch), not from
  `HYPERLOOM_RUNNER_GPU_CARD`: the cards it uses must be one kind, or the run does not start.

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
HYPERLOOM_RUNNER_GPU_CARD=/mnt/shared/hyperloom/runner-gpu-card.json  # amd-smi static --json on a sandbox's node
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

A new cluster also needs the prompt set staged into the runner cache, with its own command, run where the cache is
written, on a host that has the model's tokenizer files. Without it, a test that asks for the short-prompt consistency
check cannot load it:

```shell
PYTHONPATH=src python -m hyperloom.measure.short_prompts.build --model /mnt/shared/models/<model> \
    --cache /mnt/shared/hyperloom/runner-cache
```

Run it with `umask 022` too: it creates the entry's directories and manifest with the process's umask, and the
sandbox's account must be able to read them.

On a cluster whose sandboxes mount the shared volume at, say, `/data`, keep the app host's values and set the
template's `HYPERLOOM_RUNNER_CACHE=/data/hyperloom/runner-cache` (and the mirrors below `/data`).
