# Pre-release E2E — setup (docker + vLLM)

You are running the Hyperloom pre-release E2E test non-interactively. Complete the
setup step for a **docker + vLLM** leg, then stop. Do not run the demo yet.

> **IMPORTANT — you own the container.** You are on the privileged host pod. This is a
> `docker` leg, so **you** must start the backend container yourself by following the
> `hyperloom-setup` skill and the demo skill's **docker mode**: run `docker run` to start
> a long-lived single-GPU container, then run setup **inside** it with `docker exec`.
> Docker is already available on this host (a pod-local `dockerd` is running).

## Environment (already prepared)

A `.env` file exists in the current workspace (`REPO_ROOT`) with these values already
set: `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `USER_DATA_PATH`,
`HYPERLOOM_RUN_MODE=docker`, `FRAMEWORK=vllm`, `MODEL_PATH`, `TARGET_GAIN`,
`DEMO_HOURS`, `HYPERLOOM_CONTAINER_NAME`, `HYPERLOOM_SHM_SIZE`, plus the CI
isolation values `E2E_RENDERD`, `E2E_KFD_GID`, `E2E_DRI_GID`, `E2E_LEG_CPUS`,
`E2E_LEG_MEM`, `E2E_NFS_MOUNT`. The wheel is already installed via
`pip install --target .` so a `hyperloom/` package directory is present.

`HYPERLOOM_IMAGE` is **not** in `.env` — you choose it (see "Image selection" below).

## Fixed decisions

Follow the `hyperloom-setup` skill in **docker** mode with these fixed decisions — do
**not** ask interactive questions; use the values already in `.env`:

- **Run mode:** docker. Start the container yourself (see the hard constraints below for
  the exact `docker run` flags), then `docker exec` the setup inside it. Do **not** set
  `HYPERLOOM_DOCKER_TARGET_HOST` (run on the current host).
- **Framework:** vLLM — provided by the container image; run setup with
  `--install-framework none --yes` inside the container (do **not** `--install-framework
  vllm`).
- **LLM provider / model / `USER_DATA_PATH`:** use the values already in `.env`; do
  not change them.

## Image selection

`HYPERLOOM_IMAGE` is not preset. You **MUST** use the **exact, verbatim** `docker.io/...`
tag string that is written on the demo skill's `vllm` row in its **"Suggested Docker
images"** section — nothing else. This is a hard release-gate constraint, not a
suggestion:

- **Do NOT freelance the tag.** Do not bump the version, do not pick a "newer" or
  "latest" build, do not substitute a different tag from your memory, from Docker Hub, or
  from anywhere other than the skill file. The pinned tag is the one the release is gated
  on; a different tag is a **failure**, even if it also pulls successfully.
- Read the tag by extracting it **from the skill file itself** rather than typing it out,
  e.g. (the demo skill's path is in `HYPERLOOM_SKILL_PATH` in `.env`; otherwise it is the
  `SKILL.md` of the demo skill you are running):

  ```bash
  HYPERLOOM_IMAGE="$(grep -E '^- `vllm`' "$HYPERLOOM_SKILL_PATH" | grep -oE 'docker\.io/[^`]+' | head -1)"
  echo "using image: $HYPERLOOM_IMAGE"
  ```

  This is a `vllm` leg, so use the single arch-independent `vllm` row — no GPU detection
  is needed.
- If that command yields an empty string, or the resulting image cannot be pulled, **stop
  and report the failure** — do **not** substitute any other tag to work around it.

Export the extracted value as `HYPERLOOM_IMAGE` for the `docker run` below.

## Hard constraints (automated release gate)

Your `docker run` **MUST** use exactly the flags below. These **replace** the demo
skill's default `--device /dev/dri` (all GPUs) and `--group-add video` (a group *name*
the pod has no entry for) with single-card isolation and numeric GIDs. Every other
aspect of the skill's docker flow (long-lived `--entrypoint tail … -f /dev/null`,
mounting `$REPO_ROOT:$REPO_ROOT`, `docker exec` setup, running optimize inside) is
unchanged. Read the values from `.env`:

- **Container name:** `--name "$HYPERLOOM_CONTAINER_NAME"` (already unique per leg;
  another leg may share this host's dockerd, so do not rename it to a fixed value).
- **Image:** the `HYPERLOOM_IMAGE` you selected above (the skill's `vllm` tag).
- **Single-GPU isolation (REPLACES `--device /dev/dri`):**
  `--device /dev/kfd --device /dev/dri/renderD${E2E_RENDERD}`
- **Numeric group-add (REPLACES `--group-add video`):**
  `--group-add ${E2E_KFD_GID} --group-add ${E2E_DRI_GID}`
- **Resource caps:** `--cpus ${E2E_LEG_CPUS} --memory ${E2E_LEG_MEM} --shm-size ${HYPERLOOM_SHM_SIZE}`
- **Security:** `--security-opt seccomp=unconfined`
- **Device pin:** `-e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0`
- **Mounts:** `-v "$REPO_ROOT:$REPO_ROOT" -v "${E2E_NFS_MOUNT}:${E2E_NFS_MOUNT}"` —
  mounting **all** of `${E2E_NFS_MOUNT}` at the same absolute path is **required** so
  that `MODEL_PATH` (a symlink into another subtree under `${E2E_NFS_MOUNT}`) resolves
  inside the container. Do **not** mount only the model's parent directory.

Also:

- Do **not** add any other `--device`, and do **not** use `--group-add video`. Do **not**
  use `rocm-smi`/`rocminfo` to **choose which GPU** the leg runs on: the single bound
  `renderD` node + `HIP/ROCR_VISIBLE_DEVICES=0` are what pin this leg to its one card. Do
  not add `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` values other than `0`. (This vLLM
  image is architecture-independent, so no GPU-arch detection is needed.)
- Do **not** print, echo, or copy secret values (API keys) into output or logs. The key
  lives only in the pod-local `.env` (it reaches the container via `-v $REPO_ROOT:$REPO_ROOT`);
  do **not** write it anywhere else, and never onto NFS outside that `.env`.
- Do **not** modify `USER_DATA_PATH`.

## Termination

When setup completes successfully, stop and report only `setup complete: docker/vllm`.
Leave the container **running** so the demo turn can `docker exec` into it. If setup
hard-fails, report the failure and stop.
