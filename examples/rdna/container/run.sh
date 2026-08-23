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

# The container runs as root by default, and root reaches /dev/kfd and /dev/dri
# via CAP_DAC_OVERRIDE -- no --group-add needed. It is needed ONLY when dropping
# to a non-root user, and then it must be the HOST's NUMERIC gid: --group-add
# render resolves the name inside the image, which has no render group, and
# fails with "unable to find group render".
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
