#!/usr/bin/env bash
set -euo pipefail

# GSM8K accuracy evaluation for the PRISM inference optimization accuracy gate.
#
# Runs EleutherAI lm-evaluation-harness against a running model server via its
# OpenAI-compatible chat endpoint. Uses InferenceX's lm-eval pinning and patches.
#
# Required env:
#   MODEL           — model name/path
#   PORT            — server port (default 8888)
#
# Optional env:
#   EVAL_TASK       — gsm8k | gpqa_diamond (default gsm8k)
#   NUM_FEWSHOT     — few-shot examples (default 5 for gsm8k, 0 for gpqa)
#   RESULTS_DIR     — output directory
#   INFERENCEX_PATH — InferenceX checkout path
#   CONC            — concurrent eval requests (default 32)
#   MAX_GEN_TOKENS  — max generation tokens (default: auto from server)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL="${MODEL:?MODEL must be set}"
PORT="${PORT:-8888}"
EVAL_TASK="${EVAL_TASK:-gsm8k}"
CONC="${CONC:-32}"
# Prefer the already-mirrored INFERENCEX_PATH set by setup.md.
# Fallback to a common NFS location if unset.
_INFERENCEX_DEFAULT="/shared_nfs/nehaprakriya/PRISM/inference_optimization/InferenceX"
INFERENCEX_PATH="${INFERENCEX_PATH:-$_INFERENCEX_DEFAULT}"

# Mirror to /tmp when the path is read-only (same logic as setup.md)
if [ -d "$INFERENCEX_PATH" ] && ! touch "$INFERENCEX_PATH/.rw_probe" 2>/dev/null; then
    _TMP_IX="/tmp/InferenceX"
    if [ ! -d "$_TMP_IX" ]; then
        echo "eval_accuracy: mirroring $INFERENCEX_PATH → $_TMP_IX (read-only source)"
        cp -a "$INFERENCEX_PATH" "$_TMP_IX"
    fi
    INFERENCEX_PATH="$_TMP_IX"
else
    rm -f "$INFERENCEX_PATH/.rw_probe" 2>/dev/null
fi
RESULTS_DIR="${RESULTS_DIR:-${RESULT_DIR:-.}/eval_${EVAL_TASK}}"
NUM_FEWSHOT="${NUM_FEWSHOT:-}"
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)            EVAL_TASK="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --model)           MODEL="$2"; shift 2 ;;
        --results-dir)     RESULTS_DIR="$2"; shift 2 ;;
        --conc)            CONC="$2"; shift 2 ;;
        --num-fewshot)     NUM_FEWSHOT="$2"; shift 2 ;;
        --max-gen-tokens)  MAX_GEN_TOKENS="$2"; shift 2 ;;
        *)                 echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

BENCHMARK_LIB="$INFERENCEX_PATH/benchmarks/benchmark_lib.sh"
if [ ! -f "$BENCHMARK_LIB" ]; then
    echo "ERROR: benchmark_lib.sh not found at $BENCHMARK_LIB"
    echo "Set INFERENCEX_PATH to your InferenceX checkout."
    exit 1
fi

if [ -z "$NUM_FEWSHOT" ]; then
    case "$EVAL_TASK" in
        gsm8k)         NUM_FEWSHOT=5 ;;
        gpqa_diamond)  NUM_FEWSHOT=0 ;;
        *)             NUM_FEWSHOT=0 ;;
    esac
fi

echo "============================================"
echo " PRISM Accuracy Gate — $EVAL_TASK"
echo " Model:      $MODEL"
echo " Few-shot:   $NUM_FEWSHOT"
echo " Port:       $PORT"
echo " Results:    $RESULTS_DIR"
echo "============================================"

if ! curl -s --max-time 5 "http://0.0.0.0:$PORT/health" > /dev/null 2>&1; then
    echo "ERROR: Server not healthy on port $PORT"
    exit 1
fi

if [ -z "$MAX_GEN_TOKENS" ]; then
    SERVER_MAX=$(python3 -c "
import json, urllib.request
try:
    r = urllib.request.urlopen('http://0.0.0.0:${PORT}/v1/models', timeout=5)
    data = json.loads(r.read())
    mml = data.get('data', [{}])[0].get('max_model_len', 4096)
    print(min(mml - 512, 4096))
except Exception:
    print(2048)
" 2>/dev/null || echo "2048")
    MAX_GEN_TOKENS="$SERVER_MAX"
fi
echo "Max generation tokens: $MAX_GEN_TOKENS"

export MODEL MODEL_NAME="${MODEL}" PORT

source "$BENCHMARK_LIB"
_install_lm_eval_deps
_patch_lm_eval

OPENAI_CHAT_BASE="http://0.0.0.0:${PORT}/v1/chat/completions"
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

TASK_YAML="$INFERENCEX_PATH/utils/evals/${EVAL_TASK}.yaml"
if [ ! -f "$TASK_YAML" ]; then
    echo "ERROR: Task YAML not found at $TASK_YAML"
    exit 1
fi

pushd "$INFERENCEX_PATH" > /dev/null

python3 -m lm_eval --model local-chat-completions --apply_chat_template \
    --tasks "utils/evals/${EVAL_TASK}.yaml" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --output_path "${RESULTS_DIR}" \
    --log_samples \
    --model_args "model=${MODEL_NAME},base_url=${OPENAI_CHAT_BASE},api_key=${OPENAI_API_KEY},eos_string=</s>,max_retries=5,num_concurrent=${CONC},timeout=600,tokenized_requests=False,max_length=${MAX_GEN_TOKENS}" \
    --gen_kwargs "max_tokens=${MAX_GEN_TOKENS},temperature=0,top_p=1"
EVAL_EXIT=$?

popd > /dev/null

if [ $EVAL_EXIT -ne 0 ]; then
    echo "ERROR: lm-eval exited with code $EVAL_EXIT"
    exit $EVAL_EXIT
fi

echo ""
echo "============================================"
echo " Extracting scores"
echo "============================================"

RESULTS_DIR="$RESULTS_DIR" EVAL_TASK="$EVAL_TASK" python3 << 'PYEOF'
import json, glob, os, sys

results_dir = os.environ["RESULTS_DIR"]
task = os.environ["EVAL_TASK"]

result_files = sorted(glob.glob(os.path.join(results_dir, "**", "results*.json"), recursive=True))
if not result_files:
    print(f"WARNING: No results*.json found in {results_dir}")
    sys.exit(0)

latest = result_files[-1]
with open(latest) as f:
    data = json.load(f)

results = data.get("results", {})
summary = {}
for task_name, metrics in results.items():
    row = {"task": task_name}
    for key in ["exact_match,strict-match", "exact_match,flexible-extract",
                "exact_match,extract_abcd", "acc,none", "exact_match,none"]:
        if key in metrics:
            row[key] = metrics[key]
    summary[task_name] = row

print(f"\nResults: {latest}")
print(f"{'Task':<30s}  {'Metric':<35s}  {'Score':>8s}")
print("-" * 78)
for task_name, row in summary.items():
    for key, val in row.items():
        if key == "task":
            continue
        pct = f"{val*100:.1f}%" if isinstance(val, float) else str(val)
        print(f"{task_name:<30s}  {key:<35s}  {pct:>8s}")

summary_path = os.path.join(results_dir, f"eval_summary_{task}.json")
with open(summary_path, "w") as f:
    json.dump({"source_file": latest, "scores": summary,
               "model": os.environ.get("MODEL", ""), "task": task}, f, indent=2)
print(f"\nSummary: {summary_path}")
PYEOF
