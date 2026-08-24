#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The-Monk
# SPDX-License-Identifier: MIT
#
# Run the RDNA portability test container correctly, so nobody has to rediscover
# the device-passthrough incantation.
#
#   ./run.sh                                  # uses $MODEL from the environment
#   MODEL=/path/model.gguf ./run.sh
#   MODEL=... ./run.sh --bench                # args pass through to the probe
#
# Knobs (all optional): IMAGE, DOCKER (docker|podman), HIP_VISIBLE_DEVICES,
#   PP, TG, REPS, CORRECTNESS_OPS, PPL_FILE, PPL_MAX, RUN_AS_USER
set -euo pipefail

IMAGE="${IMAGE:-hyperloom-rdna:test}"
DOCKER="${DOCKER:-docker}"
command -v "$DOCKER" >/dev/null 2>&1 || { echo "ERROR: $DOCKER not found. Set DOCKER=podman?" >&2; exit 3; }

# Fail here, not 90 seconds into a run, if the GPU was never exposed to us.
[ -e /dev/kfd ] || { echo "ERROR: /dev/kfd missing — no ROCm-capable GPU visible on this host." >&2; exit 3; }
[ -d /dev/dri ] || { echo "ERROR: /dev/dri missing." >&2; exit 3; }

: "${MODEL:?set MODEL=/path/to/model.gguf (a STOCK quant: Q4_K_M, Q8_0, Q4_0 ...)}"
[ -f "$MODEL" ] || { echo "ERROR: MODEL not found: $MODEL" >&2; exit 3; }
MODEL_ABS="$(readlink -f "$MODEL")"
MODEL_DIR="$(dirname "$MODEL_ABS")"

ARGS=(run --rm
      --device /dev/kfd --device /dev/dri
      --security-opt seccomp=unconfined
      -v "$MODEL_DIR:/models:ro"
      -e "MODEL=/models/$(basename "$MODEL_ABS")")

# Device access differs by engine, and getting it wrong looks like a broken GPU
# rather than a permissions problem:
#   docker (rootful)  root in-container reaches the devices via CAP_DAC_OVERRIDE;
#                     no --group-add needed.
#   podman ROOTLESS   in-container uid 0 maps to your host uid, so /dev/kfd is
#                     nobody:nogroup and CAP_DAC_OVERRIDE does NOT apply. Host
#                     supplementary groups are dropped unless --group-add
#                     keep-groups is passed. Without it rocminfo reports ZERO
#                     GPUs and test-backend-ops quietly runs CPU-only, which
#                     "passes" while proving nothing.
if [ "$(basename "$DOCKER")" = "podman" ] \
   && [ "$($DOCKER info --format '{{.Host.Security.Rootless}}' 2>/dev/null)" = "true" ]; then
    ARGS+=(--group-add keep-groups)
fi

# --group-add render is wrong on both engines: the NAME resolves inside the
# image, which has no render group. A non-root user needs the host NUMERIC gid.
if [ -n "${RUN_AS_USER:-}" ]; then
    ARGS+=(--user "$RUN_AS_USER")
    for g in render video; do
        gid="$(getent group "$g" | cut -d: -f3)"
        [ -n "$gid" ] && ARGS+=(--group-add "$gid")
    done
fi

# A perplexity corpus lives on the host too; mount it or the gate silently
# cannot run.
if [ -n "${PPL_FILE:-}" ]; then
    [ -f "$PPL_FILE" ] || { echo "ERROR: PPL_FILE not found: $PPL_FILE" >&2; exit 3; }
    PPL_ABS="$(readlink -f "$PPL_FILE")"
    ARGS+=(-v "$(dirname "$PPL_ABS"):/ppl:ro" -e "PPL_FILE=/ppl/$(basename "$PPL_ABS")")
    [ -n "${PPL_MAX:-}" ] && ARGS+=(-e "PPL_MAX=$PPL_MAX")
fi

for v in HIP_VISIBLE_DEVICES PP TG REPS CORRECTNESS_OPS; do
    [ -n "${!v:-}" ] && ARGS+=(-e "$v=${!v}")
done

echo ">> $DOCKER ${ARGS[*]} $IMAGE $*"
exec "$DOCKER" "${ARGS[@]}" "$IMAGE" "$@"
