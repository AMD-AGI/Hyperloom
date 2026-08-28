# Pre-release E2E — setup (docker + SGLang)

You are running the Hyperloom pre-release E2E test non-interactively. Complete the
setup step for a **docker + SGLang** leg, then stop. Do not run the demo yet.

> **IMPORTANT — you own the container.** You are on the privileged host pod. This is a
> `docker` leg, so **you** must start the backend container yourself by following the
> `hyperloom-setup` skill and the demo skill's **docker mode**: run `docker run` to start
> a long-lived single-GPU container, then run setup **inside** it with `docker exec`.
> Docker is already available on this host (a pod-local `dockerd` is running).

## Environment (already prepared)

A `.env` file exists in the current workspace (`REPO_ROOT`) with these values already
set: `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `USER_DATA_PATH`,
`HYPERLOOM_RUN_MODE=docker`, `FRAMEWORK=sglang`, `MODEL_PATH`, `TARGET_GAIN`,
`DEMO_HOURS`, and the container/isolation values the demo skill reads:
`HYPERLOOM_IMAGE`, `HYPERLOOM_CONTAINER_NAME`, `HYPERLOOM_SHM_SIZE`, plus the CI
isolation values `E2E_RENDERD`, `E2E_KFD_GID`, `E2E_DRI_GID`, `E2E_LEG_CPUS`,
`E2E_LEG_MEM`, `E2E_NFS_MOUNT`. The wheel is already installed via
`pip install --target .` so a `hyperloom/` package directory is present.

## Fixed decisions

Follow the `hyperloom-setup` skill in **docker** mode with these fixed decisions — do
**not** ask interactive questions; use the values already in `.env`:

- **Run mode:** docker. Start the container yourself (see the hard constraints below for
  the exact `docker run` flags), then `docker exec` the setup inside it. Do **not** set
  `HYPERLOOM_DOCKER_TARGET_HOST` (run on the current host).
- **Framework:** SGLang — provided by the container image; run setup with
  `--install-framework none --yes` inside the container (do **not** `--install-framework
  sglang`).
- **LLM provider / model / `USER_DATA_PATH`:** use the values already in `.env`; do
  not change them.

## Hard constraints (automated release gate)

Your `docker run` **MUST** use exactly the flags below. These **replace** the demo
skill's default `--device /dev/dri` (all GPUs) and `--group-add video` (a group *name*
the pod has no entry for) with single-card isolation and numeric GIDs. Every other
aspect of the skill's docker flow (long-lived `--entrypoint tail … -f /dev/null`,
mounting `$REPO_ROOT:$REPO_ROOT`, `docker exec` setup, running optimize inside) is
unchanged. Read the values from `.env`:

- **Container name:** `--name "$HYPERLOOM_CONTAINER_NAME"` (already unique per leg;
  another leg may share this host's dockerd, so do not rename it to a fixed value).
- **Image:** `"$HYPERLOOM_IMAGE"`.
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

- Do **not** add any other `--device`, do **not** use `--group-add video`, and do **not**
  choose GPUs via `rocm-smi`. The single bound `renderD` node + `HIP/ROCR_VISIBLE_DEVICES=0`
  are what pin this leg to its one card.
- Do **not** print, echo, or copy secret values (API keys) into output or logs. The key
  lives only in the pod-local `.env` (it reaches the container via `-v $REPO_ROOT:$REPO_ROOT`);
  do **not** write it anywhere else, and never onto NFS outside that `.env`.
- Do **not** modify `USER_DATA_PATH`.

## Termination

When setup completes successfully, stop and report only `setup complete: docker/sglang`.
Leave the container **running** so the demo turn can `docker exec` into it. If setup
hard-fails, report the failure and stop.
