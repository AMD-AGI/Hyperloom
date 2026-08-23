# Container portability test

Proves the port works on a stack that is **not ours**: a clean ROCm image,
**stock upstream llama.cpp** built from source inside the container, and this
fork installed from git.

This matters because the port was first validated against a private llama.cpp
fork and a pinned ROCm userspace. That proves the code runs *here*. It says
nothing about anyone else's machine, which is the only question that counts for
someone trying to use their own card.

## Build

```bash
docker build -t hyperloom-rdna:test \
    --build-arg AMDGPU_TARGETS=gfx1201 \
    -f examples/rdna/container/Containerfile /tmp/empty-ctx
```

`AMDGPU_TARGETS` must match your silicon — the HIP build is arch-specific, and
a wrong value produces binaries that cannot run on your GPU. `mkdir /tmp/empty-ctx`
first: nothing is copied from the working tree, everything is cloned from git,
so the build context is deliberately empty.

Pin the inputs when you want a reproducible run:
`--build-arg LLAMA_CPP_REF=<tag>`, `--build-arg FORK_REF=<branch>`,
`--build-arg ROCM_IMAGE=rocm/dev-ubuntu-24.04:<tag>`.

## Run

```bash
MODEL=/path/to/model.gguf examples/rdna/container/run.sh
```

`run.sh` resolves the model path, mounts its directory read-only, passes the
devices, forwards the knobs (`PP`, `TG`, `REPS`, `CORRECTNESS_OPS`, `PPL_FILE`,
`PPL_MAX`, `HIP_VISIBLE_DEVICES`), and fails immediately with a clear message if
`/dev/kfd` is missing rather than 90 seconds into a run. Arguments pass through
to the probe. It works with `DOCKER=podman` too.

By hand, if you prefer:

```bash
docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined \
    -v /path/to/models:/models:ro \
    -e MODEL=/models/your-model.gguf \
    hyperloom-rdna:test
```

**No `--group-add` is needed.** The container runs as root, which reaches
`/dev/kfd` and `/dev/dri` through `CAP_DAC_OVERRIDE` — verified: `rocminfo`
sees both GPUs and `test-backend-ops` passes without it. It is required *only*
when dropping to a non-root user, and then it must be the host's **numeric**
GID, because `--group-add render` resolves the group name inside the image,
which has no `render` group. `run.sh` handles that via `RUN_AS_USER=1000:1000`.

Defaults to the full probe (`--bench`), so it runs the correctness gate and the
benchmark and prints a paste-ready report. Add `-e CORRECTNESS_OPS=ALL` for a
thorough gate, or `-e PPL_FILE=/models/corpus.txt -e PPL_MAX=<float>` for a
model-level gate too.

**On a multi-GPU host, pick your compute GPU deliberately** (`-e HIP_VISIBLE_DEVICES=0`,
the default here). If one card is driving a display, do not benchmark on it —
a GPU hang there takes the desktop with it.

## Use a STOCK quant format

The container builds **upstream** llama.cpp, which only knows upstream types.
A GGUF in a fork-specific quant (`Q2_0`, `IU4`, `F8E4M3`, and friends) will not
load, and the failure looks like a broken port rather than an unknown type.

Use `Q4_K_M`, `Q8_0`, `Q4_0` or another upstream format here. That is the point
of the exercise: if the runner only works against a private fork's kernels,
that is a fork feature, not RDNA support.

## What this test already caught

Both of these were found by *running* it, and both would have looked like
"RDNA support is broken" to someone trying the port for the first time:

- **`libhipblas.so.3: cannot open shared object file`.** The ROCm dev image
  ships its libraries in `/opt/rocm/lib` but registers that path with neither
  `ld.so.conf.d` nor `LD_LIBRARY_PATH` — `ldconfig -p` knows no hipblas at all.
  Compilation still succeeds, because cmake passes `-L` explicitly; every
  ROCm-linked binary then dies at **load** time. The Containerfile now writes
  the `ld.so.conf.d` entry, runs `ldconfig`, and **verifies** the result in the
  same layer, so a regression fails the build rather than the run. Verified
  after the fix: 4 hipblas entries in `ldconfig -p`, and **0** unresolved
  libraries in both `llama-bench` and `test-backend-ops`.
- **`unable to find group render`.** `--group-add` resolves the group name
  inside the image, which has no `render` group. The first fix was to document
  the numeric-GID workaround; the actual fix was to check whether the flag was
  needed at all. It is not — the container runs as root and reaches the devices
  via `CAP_DAC_OVERRIDE`, confirmed by `rocminfo` and a passing
  `test-backend-ops` with no group flags. It is now absent from the documented
  command, and `run.sh` adds it only under `RUN_AS_USER`.

Note also that the probe behaved correctly during the broken run: it reported
`PARTIAL` with the linker error in the correctness row, rather than a false
`CONFIRMED`. The gate refusing to pass is the gate working.

## Reading the result

The probe's exit code is the verdict: `0` CONFIRMED, `1` PARTIAL, `2` UNSUPPORTED.
`CONFIRMED` requires **both** the correctness gate (`test-backend-ops` vs the
CPU reference) and a measurement Hyperloom's own validator accepts — see
[`AGENTS.md`](../../../AGENTS.md).

A container result is worth more than a host result, because it rules out
everything your host happens to have lying around: your ROCm, your compiler,
your fork, your environment variables. If it passes here, the port is the
reason.
