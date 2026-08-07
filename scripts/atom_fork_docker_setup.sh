#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Bring up a Docker container for running Hyperloom's `--framework atom`
# path against a local ATOM (and optionally aiter) fork checkout, instead of
# whatever ATOM commit happens to be baked into the base image.
#
# Background: `atom` is a registered Hyperloom serving framework
# (hyperloom/inference_optimizer/framework_registry.py) but Hyperloom does
# not build/install it the way it does sglang/vllm -- `--install-framework`
# only auto-installs those two (install_baremetal.sh's
# resolve_installed_framework() has no atom probe), so ATOM must already be
# importable in the target environment. This script provisions that
# environment inside a container by copying host fork checkouts in and
# editable-reinstalling, following the recipe validated in
# ~/git/obsidian-vault/30 - Projects/Instinct Agentic Node/
# "DeepSeek-V4-Flash-0731 ATOM Bug Fixes (gfx942).md" and
# ~/git/instinct-agent-bench/docs/model-profiles/deepseek-v4-flash-0731-atom-native.md.
#
# Sets ATOM_USE_TRITON_MOE=1 explicitly (ATOM's own default on gfx94x
# anyway) -- do not set it to 0, gfx942 has no native aiter FP4 MoE kernel
# and it crashes immediately.
#
# TENSILE_SOLUTION_SELECTION_METHOD is deliberately NOT set by default. The
# companion note "... ATOM-Native Hang -- hipBLASLt GlobalSplitU Livelock
# (gfx942).md" found that sustained concurrent load against
# DeepSeek-V4-Flash-0731 specifically could livelock in a Tensile/hipBLASLt
# GEMM kernel (not an ATOM/aiter/NCCL bug), and that
# TENSILE_SOLUTION_SELECTION_METHOD=2 (Stream-K kernel selection instead of
# the default GlobalSplitU-heavy path) works around it. That workaround is
# scoped to the GEMM shapes this model hits -- Stream-K is not universally
# faster, so forcing it for every workload could just as easily regress
# performance elsewhere. Set TENSILE_SOLUTION_SELECTION_METHOD yourself (see
# env overrides below) only if you hit the same livelock symptom, rather
# than assuming any MI300X ATOM workload needs it.
#
# Usage:
#   scripts/atom_fork_docker_setup.sh
#
# Env overrides (defaults shown):
#   HYPERLOOM_IMAGE=rocm/atom:latest
#   HYPERLOOM_CONTAINER_NAME=atom-bench
#   ATOM_FORK_PATH=$HOME/git/ATOM
#   AITER_FORK_PATH=$HOME/git/aiter
#   HF_HOME_HOST=/data/hf_home
#   HYPERLOOM_SHM_SIZE=64g
#   TENSILE_SOLUTION_SELECTION_METHOD= (unset by default; workload-specific, see above)
#
# After this script completes, the container is ready for the Hyperloom
# `--framework atom` optimize path (see docs/how-to or the
# hyperloom-custom-advanced skill) -- export FRAMEWORK=atom explicitly in the
# exec shell before launching; Hyperloom's own framework auto-detection does
# not know about atom.

set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_script_dir}/.." && pwd)"

IMAGE="${HYPERLOOM_IMAGE:-rocm/atom:latest}"
CONTAINER_NAME="${HYPERLOOM_CONTAINER_NAME:-atom-bench}"
ATOM_FORK_PATH="${ATOM_FORK_PATH:-$HOME/git/ATOM}"
AITER_FORK_PATH="${AITER_FORK_PATH:-$HOME/git/aiter}"
HF_HOME_HOST="${HF_HOME_HOST:-/data/hf_home}"
SHM_SIZE="${HYPERLOOM_SHM_SIZE:-64g}"

log() { echo "[atom-fork-setup] $*"; }
die() { echo "[atom-fork-setup ERROR] $*" >&2; exit 1; }

[ -d "${ATOM_FORK_PATH}/.git" ] || die "ATOM_FORK_PATH '${ATOM_FORK_PATH}' is not a git checkout"
[ -d "${AITER_FORK_PATH}/.git" ] || die "AITER_FORK_PATH '${AITER_FORK_PATH}' is not a git checkout"
[ -d "${HF_HOME_HOST}" ] || die "HF_HOME_HOST '${HF_HOME_HOST}' does not exist on this host"

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  die "container '${CONTAINER_NAME}' already exists; docker rm -f ${CONTAINER_NAME} first (or set HYPERLOOM_CONTAINER_NAME)"
fi

DOCKER_RUN_ARGS=(
  -d
  --name "${CONTAINER_NAME}"
  --shm-size "${SHM_SIZE}"
  --entrypoint tail
  --device /dev/kfd
  --device /dev/dri
  --group-add video
  --network host
  --ipc host
  --security-opt seccomp=unconfined
  --cap-add=SYS_PTRACE
  -e HF_HOME=/data/hf_home
  -e AITER_LOG_LEVEL=WARNING
  -e ATOM_USE_TRITON_MOE=1
  -v "${REPO_ROOT}:${REPO_ROOT}"
  -v "${HF_HOME_HOST}:/data/hf_home"
)
if [ -n "${TENSILE_SOLUTION_SELECTION_METHOD:-}" ]; then
  log "TENSILE_SOLUTION_SELECTION_METHOD=${TENSILE_SOLUTION_SELECTION_METHOD} set explicitly; baking into container env"
  DOCKER_RUN_ARGS+=(-e "TENSILE_SOLUTION_SELECTION_METHOD=${TENSILE_SOLUTION_SELECTION_METHOD}")
fi

log "starting container '${CONTAINER_NAME}' from ${IMAGE}"
docker run "${DOCKER_RUN_ARGS[@]}" "${IMAGE}" -f /dev/null

log "copying ${ATOM_FORK_PATH} -> ${CONTAINER_NAME}:/opt/ATOM-src"
docker cp "${ATOM_FORK_PATH}" "${CONTAINER_NAME}:/opt/ATOM-src"
log "copying ${AITER_FORK_PATH} -> ${CONTAINER_NAME}:/opt/aiter-src"
docker cp "${AITER_FORK_PATH}" "${CONTAINER_NAME}:/opt/aiter-src"

log "reinstalling atom + aiter from copied fork sources"
docker exec "${CONTAINER_NAME}" bash -lc '
set -euo pipefail
git config --global --add safe.directory /opt/ATOM-src
git config --global --add safe.directory /opt/aiter-src

# Known quirk (validated recipe): the base image pre-installs a flydsl wheel
# declaring itself Version 0.1.9.dev599 in its dist-info METADATA. The
# legacy setup_requires/fetch_build_eggs path in aiter setup.py wants a
# newer flydsl and, left alone, either loops or (observed here) corrupts
# metadata.distributions() resolution when it tries to fetch/install one
# live. Force the METADATA to already claim the newer version so that
# legacy build-eggs check is satisfied without touching the package itself.
sed -i "s/^Version: 0.1.9.dev599/Version: 0.3.0/" \
  /opt/venv/lib/python3.12/site-packages/flydsl-0.1.9.dev599.dist-info/METADATA 2>/dev/null || true

pip uninstall -y atom amd-aiter || true

cd /opt/aiter-src
git submodule sync
git submodule update --init --recursive
rm -rf aiter/jit/build aiter/jit/*.so
AITER_USE_SYSTEM_TRITON=1 python setup.py develop

pip install --no-deps -e /opt/ATOM-src
pip install msgspec setproctitle uvloop msgpack "transformers==5.12.1" pytest
'

log "verifying installed sources resolve to the copied forks"
docker exec "${CONTAINER_NAME}" bash -lc '
echo "--- atom ---"
git -C /opt/ATOM-src log -1 --oneline
python3 -c "import atom, os; print(\"atom module:\", os.path.dirname(atom.__file__))"
echo "--- aiter ---"
git -C /opt/aiter-src log -1 --oneline
python3 -c "import aiter, os; print(\"aiter module:\", os.path.dirname(aiter.__file__))"
'

log "running Hyperloom setup backend inside the container (--install-framework none)"
# --frameworks atom makes Phase 1's import-probe recognize atom (it defaults
# to sglang,vllm only). --skip-base-check is still required on top of that:
# base_preflight()'s check_torch_triton_alignment gate compares the
# installed triton against torch's *SGLang-oriented* pinned requirement,
# unconditionally, regardless of which framework is actually being used --
# a false positive here, since aiter/atom were just built successfully
# against system triton via AITER_USE_SYSTEM_TRITON=1 above. Every other
# real check (ROCm, GPU arch, torch ROCm build, hipcc, atom import, aiter
# import) already passed before this gate fires.
docker exec -w "${REPO_ROOT}" "${CONTAINER_NAME}" bash -lc \
  "REPO_ROOT=\"\$(pwd -P)\"; PYTHONPATH=\"\${REPO_ROOT}\" python3 -m hyperloom.inference_optimizer.setup -- --install-framework none --frameworks atom --skip-base-check --yes"

log "done. Container '${CONTAINER_NAME}' is ready."
log "FRAMEWORK is NOT set in .env (Hyperloom's auto-detect doesn't probe for atom) -- export FRAMEWORK=atom explicitly before launching the optimizer."
