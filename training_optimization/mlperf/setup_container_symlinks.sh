#!/usr/bin/env bash
# 在已运行的容器内执行（无需重启）：将 /workspace/code、/data、/model、/results
# 软链接到本仓库与 config 中的 DATADIR、MODELDIR、LOGDIR，与 run_with_docker.sh 的 bind mount 一致。
#
# 用法：
#   source config_MI355X_1x8x1_fp8.sh   # 或你的 config
#   bash setup_container_symlinks.sh
# 或一步：
#   bash setup_container_symlinks.sh /path/to/config_MI355X_1x8x1_fp8.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${1:-}" ]]; then
  # shellcheck source=/dev/null
  set -a && source "$1" && set +a
fi

: "${CODE_ROOT:=${SCRIPT_DIR}}"
: "${DATADIR:?请先 source config（需 DATADIR）或: bash $0 /path/to/config.sh}"
: "${MODELDIR:?需 MODELDIR}"
: "${LOGDIR:?需 LOGDIR}"

for p in "$CODE_ROOT" "$DATADIR" "$MODELDIR" "$LOGDIR"; do
  if [[ ! -e "$p" ]]; then
    echo "错误: 源路径不存在: $p" >&2
    exit 1
  fi
done

_ensure_symlink() {
  local link="$1" target="$2"
  if [[ -L "$link" ]]; then
    rm -f "$link"
  elif [[ -e "$link" ]]; then
    mv "$link" "${link}.bak.$(date +%s)"
    echo "已备份原路径: ${link}.bak.*"
  fi
  mkdir -p "$(dirname "$link")" 2>/dev/null || true
  ln -sfn "$target" "$link"
  echo "ln -sfn $target -> $link"
}

_ensure_symlink "/workspace/code" "$CODE_ROOT"
_ensure_symlink "/data" "$DATADIR"
_ensure_symlink "/model" "$MODELDIR"
_ensure_symlink "/results" "$LOGDIR"

echo "完成。可检查: ls -la /workspace/code/src/train.py /data /model /results"
