#!/usr/bin/env bash
#
# One-time GEAK install on a Dynamo GPU pod (idempotent), driven over SSH by
# `inference_optimizer.multi_node install-geak`.
#
# The GEAK source tree is already cloned by the sandbox's install.sh onto the
# shared $USER_DATA_PATH mount (which the Dynamo pod also mounts), so we do NOT
# bake GEAK into the image — we just pip-install that shared checkout into the
# pod's framework venv so the `geak` CLI lands on PATH. The pod's on-disk ROCm
# torch is pinned first so GEAK's transitive deps can't swap it for PyPI CUDA
# torch.
#
# Emits a single JSON line on stdout. Exit 0 on installed/skipped.
#
set -uo pipefail

GEAK_SRC="${1:-}"
if [ -z "$GEAK_SRC" ]; then
  echo '{"status":"failed","reason":"geak source dir argument required"}'; exit 1
fi

if command -v geak >/dev/null 2>&1; then
  echo '{"status":"skipped","reason":"geak already on PATH"}'; exit 0
fi
if [ ! -d "$GEAK_SRC" ]; then
  echo "{\"status\":\"failed\",\"reason\":\"geak src missing on shared FS: ${GEAK_SRC}\"}"; exit 1
fi

PIP="/opt/venv/bin/pip"
command -v "$PIP" >/dev/null 2>&1 || PIP="python3 -m pip"

CONSTRAINT=""
TV="$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || true)"
if [ -n "$TV" ]; then
  echo "torch==${TV}" > /tmp/mn_geak_constraint.txt
  CONSTRAINT="--constraint /tmp/mn_geak_constraint.txt"
fi

if ! $PIP install -q --no-cache-dir --break-system-packages $CONSTRAINT "$GEAK_SRC" \
     > /tmp/mn_geak_install.log 2>&1; then
  TAIL="$(tail -c 800 /tmp/mn_geak_install.log 2>/dev/null | tr '\n' ' ' | sed 's/"/\\"/g')"
  echo "{\"status\":\"failed\",\"reason\":\"pip install failed\",\"log_tail\":\"${TAIL}\"}"
  exit 1
fi

if command -v geak >/dev/null 2>&1; then
  echo '{"status":"installed"}'
else
  echo '{"status":"failed","reason":"geak still missing after pip install"}'
  exit 1
fi
