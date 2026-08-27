#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Nested docker runner for the privileged 8-GPU pre-release host. This is the ONLY
# docker entry point for docker legs: it force-binds ONE GPU and caps CPU/memory to a
# 1/4 host share so the 4 docker legs are mutually comparable and comparable to the
# baremetal legs (design §8, point E). Prompts are forbidden from running docker
# directly or choosing GPUs via rocm-smi.
#
# Usage: docker-run-hyperloom.sh <gpu_index 0-7> <leg_id>
#
# Inputs (env, inherited from the host bootstrap):
#   NFS_ROOT CI_VERSION MODEL_3H MODEL_12H TARGET_GAIN
#   CLAUDE_MODEL CLAUDE_CLI_VERSION ANTHROPIC_API_KEY_B64 [ANTHROPIC_BASE_URL]
#   HYPERLOOM_IMAGE   backend container image (overrides per-backend default)
#   LEG_CPUS LEG_MEM LEG_SHM   per-container quota (default 32 / 128g / 64g)
set -euo pipefail

GPU_INDEX="${1:?usage: docker-run-hyperloom.sh <gpu_index> <leg_id>}"
LEG_ID="${2:?usage: docker-run-hyperloom.sh <gpu_index> <leg_id>}"

: "${NFS_ROOT:?}"; : "${CI_VERSION:?}"; : "${ANTHROPIC_API_KEY_B64:?}"
: "${CLAUDE_MODEL:?}"; : "${CLAUDE_CLI_VERSION:?}"
TARGET_GAIN="${TARGET_GAIN:-100}"

# 1/4 of a 128-core / 512Gi host, matching the baremetal leg envelope.
LEG_CPUS="${LEG_CPUS:-32}"
LEG_MEM="${LEG_MEM:-128g}"
LEG_SHM="${LEG_SHM:-64g}"

# Resolve per-leg backend / model / hours from the leg id.
case "$LEG_ID" in
  *-vllm-*)   BACKEND=vllm ;;
  *-sglang-*) BACKEND=sglang ;;
  *) echo "cannot infer backend from leg '$LEG_ID'" >&2; exit 2 ;;
esac
case "$LEG_ID" in
  *-3h)  HOURS=3;  MODEL_PATH="${MODEL_3H:?MODEL_3H required}" ;;
  *-12h) HOURS=12; MODEL_PATH="${MODEL_12H:?MODEL_12H required}" ;;
  *) echo "cannot infer duration from leg '$LEG_ID'" >&2; exit 2 ;;
esac

# Default backend images (overridable via HYPERLOOM_IMAGE); mirrors the demo skill's
# suggested ROCm images.
if [ -n "${HYPERLOOM_IMAGE:-}" ]; then
  IMAGE="$HYPERLOOM_IMAGE"
elif [ "$BACKEND" = vllm ]; then
  IMAGE="${HYPERLOOM_IMAGE_VLLM:-vllm/vllm-openai-rocm:v0.27.1}"
else
  IMAGE="${HYPERLOOM_IMAGE_SGLANG:-lmsysorg/sglang-rocm:v0.5.17-rocm724-mi35x-srt}"
fi

ROOT="${NFS_ROOT%/}/runs/${CI_VERSION}/${LEG_ID}"
mkdir -p "$ROOT"
BOOTSTRAP="${NFS_ROOT%/}/bootstrap/${CI_VERSION}/bootstrap-pre-release.sh"
NAME="hyperloom-${LEG_ID}"

# The model dir (e.g. /shared_nfs/models/<name>) lives OUTSIDE $ROOT/$NFS_ROOT, so the
# demo skill's rule applies: "If ... a pre-downloaded model directory is outside the
# workspace, add matching -v host_path:host_path mounts" (examples/*/SKILL.md). Without
# this the container's HYPERLOOM_MODEL_PATH resolves to a path that does not exist and
# optimize can never boot the server -> the leg hangs waiting for a final.json that
# never comes. Mount the model's PARENT dir (minimal exposure). Skip the extra -v when
# the model already lives under a dir we mount ($ROOT or $NFS_ROOT) to avoid a duplicate
# -v that docker rejects.
MODEL_DIR="$(dirname -- "$MODEL_PATH")"
MODEL_MOUNT=()
case "$MODEL_DIR/" in
  "${ROOT%/}/"*|"${NFS_ROOT%/}/"*) : ;;  # already covered by an existing mount
  *) MODEL_MOUNT=(-v "$MODEL_DIR:$MODEL_DIR") ;;
esac

# GPU index -> renderD node. VERIFIED on a real privileged MI355X x8 pod (2026-08-27):
# the 8 physical GPUs map to renderD128,136,144,...,184 -- i.e. stride 8, NOT +1. The
# rocm-smi GPU order matches this render-node order (GPU i == 0002/0003:00:0X.0 ==
# renderD(128+8*i)). See project memory. `cardN` numbering is NOT guaranteed to align
# with the GPU order, so we isolate via /dev/kfd + the single renderD node only.
RD=$((128 + GPU_INDEX * 8))

# Device group ownership: the pod's /etc/group has NO `video`/`render` NAMES, so
# `--group-add video` FAILS ("no matching entries in group file"). Resolve the numeric
# GIDs of the device nodes and pass those instead (verified working).
KFD_GID="$(stat -c %g /dev/kfd 2>/dev/null || echo 0)"
DRI_GID="$(stat -c %g /dev/dri/renderD${RD} 2>/dev/null || stat -c %g /dev/dri 2>/dev/null || echo 0)"

echo "[docker-run] leg=$LEG_ID gpu=$GPU_INDEX renderD$RD (kfd_gid=$KFD_GID dri_gid=$DRI_GID) image=$IMAGE cpus=$LEG_CPUS mem=$LEG_MEM"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# GPU isolation: expose /dev/kfd (shared) + exactly ONE renderD node, so the container
# sees a single device. HIP_VISIBLE_DEVICES=0 pins the app to that one card. CPU/mem
# hard-capped to the 1/4 share. seccomp=unconfined matches how the ROCm images expect
# to run (verified: rocm-smi enumerates the single bound card correctly).
exec docker run --rm --name "$NAME" \
  --device "/dev/kfd" \
  --device "/dev/dri/renderD${RD}" \
  --group-add "$KFD_GID" \
  --group-add "$DRI_GID" \
  --security-opt seccomp=unconfined \
  --cpus "$LEG_CPUS" --memory "$LEG_MEM" --shm-size "$LEG_SHM" \
  -e HIP_VISIBLE_DEVICES=0 \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e CI_VERSION="$CI_VERSION" \
  -e NFS_ROOT="$NFS_ROOT" \
  -e LEG_ID="$LEG_ID" \
  -e HYPERLOOM_RUN_MODE=docker \
  -e HYPERLOOM_BACKEND="$BACKEND" \
  -e HYPERLOOM_MODEL_PATH="$MODEL_PATH" \
  -e DEMO_HOURS="$HOURS" \
  -e TARGET_GAIN="$TARGET_GAIN" \
  -e CLAUDE_MODEL="$CLAUDE_MODEL" \
  -e CLAUDE_CLI_VERSION="$CLAUDE_CLI_VERSION" \
  -e ANTHROPIC_API_KEY_B64="$ANTHROPIC_API_KEY_B64" \
  ${ANTHROPIC_BASE_URL:+-e ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"} \
  -v "$ROOT:$ROOT" \
  -v "$NFS_ROOT:$NFS_ROOT" \
  "${MODEL_MOUNT[@]}" \
  --entrypoint bash \
  "$IMAGE" "$BOOTSTRAP"
