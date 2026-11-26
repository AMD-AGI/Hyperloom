---
myst:
    html_meta:
        "description": "Run a Hyperloom session on a framework whose server is built from a checkout you name, with llama.cpp on a Radeon PRO W7900D as the example: prerequisites, the checkout, the build setup step, the reference launch script, model and draft files, CPU placement and what the session produces."
        "keywords": "Hyperloom, llama.cpp, llama-server, GGUF, Radeon PRO W7900D, gfx1100, HIP, FRAMEWORK_REPO_PATH, build_setup, reference script, multi-branch"
---
# Run a session on a framework built from its checkout

Most frameworks are served by the package the image installs. A framework registered with a native component is
served by a program the session builds from a checkout you name: every server of the session, the baseline's
included, is that build, and a candidate that changes the source is compiled before it is measured. llama.cpp
(`--framework llamacpp`) is the one such framework, and this page uses it on a Radeon PRO W7900D (`gfx1100`), one of the boards the
{doc}`compatibility matrix </compatibility>` lists for it. On a Radeon AI PRO R9700 pass `--gpu-type r9700`, and
`--gfx gfx1201` to `build_setup`; a multi-branch session has run there, a single-branch one has not.

## Prerequisites

- Hyperloom installed in the image the session runs in (see [Install](../install/install.md)), with
  `HYPERLOOM_ACTIVE_IMAGE_REF` and `HYPERLOOM_ACTIVE_IMAGE_DIGEST` naming that image: a build is keyed by the image
  it was compiled in.
- CMake, Ninja and the ROCm clang (`llvm/bin/clang++` under the ROCm root, `/opt/rocm` unless `ROCM_HOME`,
  `CUDA_HOME`, `ROCM_PATH` or `HIP_PATH` names another) in that image, and 10 GB of free disk for a build. The
  build is static, with HIP on (`-DGGML_HIP=ON`) for the card's architecture, and fetches nothing.
- A clone of llama.cpp at the commit to optimize, with no change to its tracked files. The session builds its `HEAD`.
- A model directory that holds one `.gguf` file, the one the server loads, beside the model's Hugging Face tokenizer
  files, which the load client reads.

## Name the checkout

`FRAMEWORK_REPO_PATH` names the clone; `hyperloom optimize --framework-path DIR` sets it. A session that finds no
checkout, a checkout of another project, or one whose tracked files differ from its `HEAD` is refused when it
starts.

## Write the reference launch script

No recipe source knows llama.cpp, so the base launch is a script you pass as `--reference-script`: the
`llama-server` line with every flag fixed in its text.

```bash
#!/usr/bin/env bash
llama-server -ngl 99 -fa on -np 1 -c 32768 --jinja
```

The runner sets the model file, host, port and alias every round: the script may leave them out, and the ones it
names are dropped. A flag `llama-server --help` of the built server does not list is refused before the server
starts. The rules every reference script follows are in [Choose the base launch](optimize.md#choose-the-base-launch).

A file the launch names with `--spec-draft-model` (`-md`, `--model-draft`) by an absolute path, such as
`-md /models/drafts/my-model-draft.gguf`, is an auxiliary model. The specialists are shown its directory and look
there for other builds of it, so put the draft files the session may choose from in that directory. Of a home
directory, or a directory that holds the session directory, only the named file is shown.

## Run a single-branch session

```bash
hyperloom optimize --model /models/my-model-gguf --framework llamacpp --gpu-type w7900d \
  --framework-path /src/llama.cpp --reference-script ref.sh \
  --tp 1 --conc 1 --isl 1024 --osl 512 --precision q4_k_m --max-hours 8
```

`llama-server` has no tensor-parallel size, so the session runs at `--tp 1`. The session builds the checkout when it
starts and puts that build first on its `PATH`; a tree that does not build refuses the session.

## Run a multi-branch session

A multi-branch session builds through a build store. Set it up once per host, and again whenever the checkout's
`HEAD` moves, as root (the compile runs as user 65534), then source the file it writes before every session:

```bash
export FRAMEWORK_REPO_PATH=/src/llama.cpp
python -m hyperloom_multibranch.toolbox.build_setup /data/hyperloom/builds \
  --image-root /opt/venv/lib/python3.12/site-packages --gfx gfx1100 --framework-repo "$FRAMEWORK_REPO_PATH"
. /data/hyperloom/builds/build.env

hyperloom-multibranch run --model /models/my-model-gguf --framework llamacpp --gpu-type w7900d \
  --workload inferencex --tp 1 --conc 1 --isl 1024 --osl 512 --precision q4_k_m \
  --reference-script ref.sh --max-hours 8
```

`--image-root` is the image's `site-packages`, and `--gfx` the architecture of the board `--gpu-type` names: the builds
are compiled for it, and the session does not check it against the board. `build_setup` adds the checkout's `HEAD` to
the store's sources as a detached worktree of the clone, reports what the session's check at start would refuse, and
exits 1 if there is anything. Keep `FRAMEWORK_REPO_PATH` exported for the session, which reads it again; without the
sourced `build.env` the session is refused, naming `HYPERLOOM_BUILD_STORE`. The rest of the environment and the options
of `run` are in [Running a multi-branch session](run-multibranch.md).

Each specialist works in a worktree of the checkout. Before its first test that changes server arguments, a
specialist records a verdict on every option the base launch leaves unset; such a test is refused until its branch
has left none. The specialists can search llama.cpp's pull requests and port one; give the session a GitHub token
for that, as [Upstream pull-request search](run-multibranch.md#upstream-pull-request-search) describes.

## Keep builds off the measured server's CPUs

On a host that compiles while it measures, name the two CPU sets apart: `HYPERLOOM_SERVER_CPUS` pins every server a
round starts and `HYPERLOOM_BUILD_CPUS` every compile. Both are described in
[CPU placement on a shared host](../reference/environment-variables.md#cpu-placement-on-a-shared-host).

```bash
export HYPERLOOM_SERVER_CPUS=32-63 HYPERLOOM_BUILD_CPUS=0-31
```

## Send real prompts

`DATASET_PATH` replaces the random prompts of a fixed workload with the lines of a JSONL file; see
[Workload configuration](../reference/environment-variables.md#workload-configuration).

## What the session produces

A single-branch session keeps its builds under `<session dir>/native`: `base.json` records the commit it builds
over and the build site, `store` holds the builds, `checks` and `tunes` those of a tree check and of a knob sweep,
and `current` links to the executables of the build it accepted last. `<session dir>/server_options.json` lists the options the built server takes, read against the base launch.
A candidate that changes the checkout is compiled before its benchmark, and one that does not compile is reverted.
The report and the other artifacts are those of any session: see
[Output and artifacts](optimize.md#output-and-artifacts).

A multi-branch session writes `MB-report.md` and `MB-report.json` under `MB-reports` in its session directory, as
for any framework. A test that changes llama.cpp's source waits for its build holding no card, and a compile error
comes back as the candidate's failure.
