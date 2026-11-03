#!/usr/bin/env bash
# The serving template's correctness check: a smoke request, then GSM8K 5-shot on a fixed subset
# through lm_eval against the running server. From the balanced pack's correctness.sh.
#
# Env: PORT, MODEL_PATH, MODEL_NAME, CORRECTNESS_OUTPUT (from Arbor).
# From the config's server_env, optional:
#   GSM8K_LIMIT   questions (default 400)
#   GSM8K_MIN     the pass floor. SET IT for the model: its reference score on the full set less
#                 about three standard errors of the subset (on 400 questions one is ~1.1 points).
#                 Unset, a sanity floor of 0.30 applies: it fails a broken model, nothing finer.
# Always exits 0; the verdict is the JSON it writes.
set -uo pipefail

: "${PORT:?PORT env var not set}"
: "${MODEL_PATH:?MODEL_PATH env var not set}"
: "${CORRECTNESS_OUTPUT:?CORRECTNESS_OUTPUT env var not set}"
LIMIT=${GSM8K_LIMIT:-400}
if [ -n "${GSM8K_MIN:-}" ]; then MIN=$GSM8K_MIN; FLOOR=model; else MIN=0.30; FLOOR=sanity; fi
OUT_DIR=$(dirname "$CORRECTNESS_OUTPUT")/gsm8k
mkdir -p "$OUT_DIR"
[ "$FLOOR" = sanity ] && echo "[correctness] WARNING: GSM8K_MIN is not set: using the sanity floor 0.30, not the model's"

write() {   # passed score error
    python3 -c 'import json,sys; json.dump({"passed": sys.argv[1]=="1", "score": float(sys.argv[2]), "error": sys.argv[3] or None, "gsm8k_limit": int(sys.argv[4]), "gsm8k_min": float(sys.argv[5]), "floor": sys.argv[6]}, open(sys.argv[7], "w"))' \
        "$1" "$2" "$3" "$LIMIT" "$MIN" "$FLOOR" "$CORRECTNESS_OUTPUT"
}
WIRE=$(curl -sf "http://127.0.0.1:$PORT/v1/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null) || WIRE=""
if [ -z "$WIRE" ]; then
    write 0 0 "server unreachable: /v1/models did not answer on port $PORT"; exit 0
fi

# ---- smoke --------------------------------------------------------------------
code=$(curl -s -o "$OUT_DIR/smoke.json" -w "%{http_code}" "http://127.0.0.1:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$WIRE\", \"prompt\": \"Question: What is 2+2?\\nAnswer:\", \"max_tokens\": 64, \"temperature\": 0}") || code=000
if [ "$code" != 200 ]; then
    write 0 0 "smoke request: HTTP $code $(head -c 200 "$OUT_DIR/smoke.json" 2>/dev/null)"; exit 0
fi
echo "[correctness] smoke: $(python3 -c 'import json,sys; print(repr(json.load(open(sys.argv[1]))["choices"][0]["text"][:80]))' "$OUT_DIR/smoke.json" 2>/dev/null)"

# ---- GSM8K subset ----------------------------------------------------------------
# lm_eval's own gsm8k task names its dataset "gsm8k", an id huggingface_hub 1.x no longer resolves
# (HfUriError): the task runs from a copy of it that names the same set openai/gsm8k.
TASKS=gsm8k
TASK=$(python3 -c 'import os, lm_eval; print(os.path.join(os.path.dirname(lm_eval.__file__), "tasks", "gsm8k", "gsm8k.yaml"))' 2>/dev/null)
if [ -f "$TASK" ]; then
    sed 's|^dataset_path: gsm8k$|dataset_path: openai/gsm8k|' "$TASK" > "$OUT_DIR/gsm8k.yaml" && TASKS=$OUT_DIR/gsm8k.yaml
fi
echo "[correctness] GSM8K 5-shot, first $LIMIT questions, pass at >= $MIN"
HF_HUB_OFFLINE=0 lm_eval --model local-completions \
    --model_args "model=$WIRE,base_url=http://127.0.0.1:$PORT/v1/completions,tokenizer=$MODEL_PATH,num_concurrent=32,max_retries=3,tokenized_requests=False" \
    --tasks "$TASKS" --num_fewshot 5 --limit "$LIMIT" --batch_size 32 --gen_kwargs "temperature=0" --seed 1234 \
    --output_path "$OUT_DIR" > "$OUT_DIR/lm_eval.log" 2>&1
rc=$?
score=$(python3 - "$OUT_DIR" <<'PYEOF'
import glob, json, os, sys
files = sorted(glob.glob(os.path.join(sys.argv[1], "**", "results_*.json"), recursive=True), key=os.path.getmtime)
if not files:
    sys.exit(1)
r = json.load(open(files[-1]))["results"]["gsm8k"]
print(r.get("exact_match,flexible-extract", r.get("exact_match,strict-match")))
PYEOF
) || score=""
if [ -z "$score" ] || [ "$score" = None ]; then
    # A GSM8K set that cannot be loaded (no egress, an offline cache without it) says nothing
    # about the candidate: "correctness inconclusive" makes the gate book NOT_MEASURED, not REVERT.
    hub=$(grep -m1 -E "Couldn't reach '[^']+' on the Hub|OfflineModeIsEnabled|DatasetNotFoundError|LocalEntryNotFoundError|HfUriError" "$OUT_DIR/lm_eval.log" | cut -c1-200)
    [ -n "$hub" ] && { write 0 0 "correctness inconclusive: the GSM8K dataset could not be loaded: $hub"; exit 0; }
    write 0 0 "lm_eval produced no GSM8K score (exit $rc): $(tail -3 "$OUT_DIR/lm_eval.log" | tr '\n' ' ' | cut -c1-300)"; exit 0
fi
passed=$(python3 -c "import sys; print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)" "$score" "$MIN")
write "$passed" "$score" "$([ "$passed" = 1 ] || echo "GSM8K $score below $MIN on $LIMIT questions")"
echo "[correctness] GSM8K exact_match (flexible) = $score on $LIMIT questions: $([ "$passed" = 1 ] && echo PASS || echo FAIL)"
exit 0
