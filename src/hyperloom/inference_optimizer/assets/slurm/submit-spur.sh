#!/usr/bin/env bash
# Submitter for run_hyperloom-spur.sbatch on a SPUR cluster (AMD/Crusoe's
# Slurm-CLI-compatible scheduler). SPUR variant of submit.sh.
#
# Why a separate script: spur's `sbatch` diverges from stock Slurm in two ways
# that break submit.sh:
#   1. it does NOT pass positional args to the batch script, so the model key /
#      backend are exported as HL_MODEL_KEY / HL_BACKEND instead of `$1`/`$2`;
#   2. its `--export` does NOT honor inline KEY=VAL assignments -- it only
#      inherits the submit-time environment via ALL, so overrides are exported
#      into this shell first, then forwarded with `--export ALL`.
# Everything else mirrors submit.sh.
#
# Usage:
#   ./submit-spur.sh [options] <model_key> [more_model_keys...]
#   ./submit-spur.sh --all [options]
#
# Options (all optional; sensible defaults):
#   -b, --backend <python|claude|codex>
#                                   launch backend (default: python). python
#                                   runs the optimizer directly; claude/codex are
#                                   CLI carriers that load SKILL.md and drive it.
#   -p, --partition <name>          slurm partition
#   -A, --account <name>            slurm account
#   -g, --gpus <n>                  GPUs to request (default: model TP; use 0 on
#                                   clusters whose compute nodes have no GPU gres)
#   -t, --time <hh:mm:ss>           wall clock (default: sbatch default)
#   -c, --constraint <feat>         slurm --constraint (e.g. gfx950 for MI355X)
#       --gpu-type <TYPE>           override the optimizer --gpu-type for all jobs
#                                   (lowercase, e.g. mi355x)
#       --shared-mount <path>       shared FS bind-mounted into the container
#                                   (default: /wekafs; on spur use the cluster mount)
#       --source-dir <path>         existing Hyperloom checkout to use instead of
#                                   cloning (e.g. <shared-mount>/<user>/Hyperloom)
#       --data-root <path>          artifact root (default: <shared-mount>/hyperloom-slurm)
#       --model-base <path>         local model store; the model dir is resolved
#                                   as <model-base>/<basename repo_id> when it
#                                   exists, else the tsv repo_id (HF id) is used.
#                                   remember to bind-mount it via HL_EXTRA_MOUNTS.
#       --controller <addr>         spur controller (default: $SPUR_CONTROLLER_ADDR)
#       --dry-run                   print the sbatch command, do not submit
#   -h, --help                      show this help
#
# Examples:
#   export SPUR_CONTROLLER_ADDR=http://<spur-controller-host>:6817
#   ./submit-spur.sh -p <partition> -g 8 --shared-mount <shared-mount> \
#     --source-dir <shared-mount>/<user>/Hyperloom dsv4pro_sglang

set -euo pipefail

CONF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_TSV="$CONF_DIR/models.tsv"
SBATCH_FILE="$CONF_DIR/run_hyperloom-spur.sbatch"

BACKEND="python"
PARTITION=""
ACCOUNT=""
GPUS=""
TIME=""
CONSTRAINT=""
DATA_ROOT=""
SHARED_MOUNT=""
SOURCE_DIR=""
GPU_TYPE_OVERRIDE=""
MODEL_BASE=""
CONTROLLER=""
DRY_RUN=0
ALL=0
KEYS=()

die() { echo "error: $*" >&2; exit 1; }

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    -b|--backend)    BACKEND="${2:?}"; shift 2 ;;
    -p|--partition)  PARTITION="${2:?}"; shift 2 ;;
    -A|--account)    ACCOUNT="${2:?}"; shift 2 ;;
    -g|--gpus)       GPUS="${2:?}"; shift 2 ;;
    -t|--time)       TIME="${2:?}"; shift 2 ;;
    -c|--constraint) CONSTRAINT="${2:?}"; shift 2 ;;
    --gpu-type)      GPU_TYPE_OVERRIDE="${2:?}"; shift 2 ;;
    --shared-mount)  SHARED_MOUNT="${2:?}"; shift 2 ;;
    --source-dir)    SOURCE_DIR="${2:?}"; shift 2 ;;
    --data-root)     DATA_ROOT="${2:?}"; shift 2 ;;
    --model-base)    MODEL_BASE="${2:?}"; shift 2 ;;
    --controller)    CONTROLLER="${2:?}"; shift 2 ;;
    --all)           ALL=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage ;;
    -*)              die "unknown option: $1" ;;
    *)               KEYS+=("$1"); shift ;;
  esac
done

[ -f "$MODELS_TSV" ]   || die "missing $MODELS_TSV"
[ -f "$SBATCH_FILE" ]  || die "missing $SBATCH_FILE"
[ "$DRY_RUN" -eq 1 ] || command -v sbatch >/dev/null 2>&1 || die "sbatch not found on PATH"
case "$BACKEND" in python|claude|codex) ;; *) die "backend must be python, claude or codex" ;; esac

# spur controller: --controller wins, else inherit SPUR_CONTROLLER_ADDR.
[ -n "$CONTROLLER" ] && export SPUR_CONTROLLER_ADDR="$CONTROLLER"

# Collect keys.
if [ "$ALL" -eq 1 ]; then
  mapfile -t KEYS < <(awk -F'\t' '!/^#/ && NF>1 {print $1}' "$MODELS_TSV")
fi
[ "${#KEYS[@]}" -gt 0 ] || die "no model key given (pass keys or --all)"

# Field lookup helper: value of column $2 for the row whose key is $1.
lookup() { awk -F'\t' -v k="$1" -v c="$2" '$1==k{print $c; exit}' "$MODELS_TSV"; }

for KEY in "${KEYS[@]}"; do
  ROW="$(awk -F'\t' -v k="$KEY" '$1==k{print;exit}' "$MODELS_TSV")"
  [ -n "$ROW" ] || die "unknown model key: $KEY"

  TP="$(lookup "$KEY" 6)"
  GPU_TYPE="$(lookup "$KEY" 4)"
  REQ_GPUS="${GPUS:-$TP}"

  # Environment forwarded into the job. spur `--export` only inherits the
  # submit-time environment (ALL); inline KEY=VAL is ignored. And the model key
  # / backend cannot be positional args, so they ride along as env too.
  export HL_CONF_DIR="$CONF_DIR"
  export HL_MODEL_KEY="$KEY"
  export HL_BACKEND="$BACKEND"
  [ -n "$DATA_ROOT" ]         && export HL_DATA_ROOT="$DATA_ROOT"
  [ -n "$SHARED_MOUNT" ]      && export HL_SHARED_MOUNT="$SHARED_MOUNT"
  [ -n "$SOURCE_DIR" ]        && export HYPERLOOM_SOURCE_DIR="$SOURCE_DIR"
  [ -n "$GPU_TYPE_OVERRIDE" ] && export HL_GPU_TYPE_OVERRIDE="$GPU_TYPE_OVERRIDE"
  [ -n "$MODEL_BASE" ]        && export HL_MODEL_BASE="$MODEL_BASE"

  CMD=(sbatch
    --job-name "hl-${KEY}-${BACKEND}"
    --export ALL)
  [ "$REQ_GPUS" != "0" ] && CMD+=(--gpus "$REQ_GPUS")
  [ -n "$PARTITION" ]  && CMD+=(--partition "$PARTITION")
  [ -n "$ACCOUNT" ]    && CMD+=(--account "$ACCOUNT")
  [ -n "$TIME" ]       && CMD+=(--time "$TIME")
  [ -n "$CONSTRAINT" ] && CMD+=(--constraint "$CONSTRAINT")
  # NOTE: no positional args after the script -- spur ignores them.
  CMD+=("$SBATCH_FILE")

  echo "[submit-spur] key=$KEY gpu_type=$GPU_TYPE gpus=$REQ_GPUS backend=$BACKEND controller=${SPUR_CONTROLLER_ADDR:-<default>}"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  DRY-RUN: '; printf '%q ' "${CMD[@]}"; echo
  else
    "${CMD[@]}"
  fi
done
