#!/usr/bin/env bash
# The serving-qwen pack's correctness check: the serving template's, through the model's chat template. A smoke
# question on /v1/chat/completions, then GSM8K 5-shot on a fixed subset through lm_eval's chat backend
# (local-chat-completions, --apply_chat_template, the five shots as turns).
#
# Why a pack of its own: an instruct or thinking model answers a raw completion prompt with an end of sequence. Such
# a model scores near zero on the serving template's raw 5-shot GSM8K (an end of sequence first on most prompts), and
# normally through its chat template. Everything else in this pack is the serving template's.
#
# Env: PORT, MODEL_PATH, MODEL_NAME, CORRECTNESS_OUTPUT (from Arbor).
# From the config's server_env, optional:
#   GSM8K_LIMIT        questions (default 400)
#   GSM8K_MIN          the pass floor. SET IT for the model: its reference score on the full set through the chat
#                      template, less about three standard errors of the subset (on 400 questions one is ~1.1 points).
#                      Unset, a sanity floor of 0.30 applies: it fails a broken model, nothing finer.
#   GSM8K_MAX_TOKENS   the answer's token budget (default 2048): a thinking model reasons before it answers.
# Always exits 0; the verdict is the JSON it writes.
set -uo pipefail

: "${PORT:?PORT env var not set}"
: "${MODEL_PATH:?MODEL_PATH env var not set}"
: "${CORRECTNESS_OUTPUT:?CORRECTNESS_OUTPUT env var not set}"
LIMIT=${GSM8K_LIMIT:-400}
MAX_TOKENS=${GSM8K_MAX_TOKENS:-2048}
if [ -n "${GSM8K_MIN:-}" ]; then MIN=$GSM8K_MIN; FLOOR=model; else MIN=0.30; FLOOR=sanity; fi
OUT_DIR=$(dirname "$CORRECTNESS_OUTPUT")/gsm8k
mkdir -p "$OUT_DIR"
[ "$FLOOR" = sanity ] && echo "[correctness] WARNING: GSM8K_MIN is not set: using the sanity floor 0.30, not the model's"

write() {   # passed score error
    python3 -c 'import json,sys; json.dump({"passed": sys.argv[1]=="1", "score": float(sys.argv[2]), "error": sys.argv[3] or None, "gsm8k_limit": int(sys.argv[4]), "gsm8k_min": float(sys.argv[5]), "floor": sys.argv[6], "template": "chat"}, open(sys.argv[7], "w"))' \
        "$1" "$2" "$3" "$LIMIT" "$MIN" "$FLOOR" "$CORRECTNESS_OUTPUT"
}
WIRE=$(curl -sf "http://127.0.0.1:$PORT/v1/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null) || WIRE=""
if [ -z "$WIRE" ]; then
    write 0 0 "server unreachable: /v1/models did not answer on port $PORT"; exit 0
fi

# ---- smoke, through the chat template ----------------------------------------------
code=$(curl -s -o "$OUT_DIR/smoke.json" -w "%{http_code}" "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$WIRE\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer with just the number.\"}], \"max_tokens\": $MAX_TOKENS, \"temperature\": 0}") || code=000
if [ "$code" != 200 ]; then
    write 0 0 "smoke request: HTTP $code $(head -c 200 "$OUT_DIR/smoke.json" 2>/dev/null)"; exit 0
fi
echo "[correctness] smoke: $(python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["choices"][0]["message"]; print(repr(((m.get("content") or "") or (m.get("reasoning_content") or "")[-80:])[:80]))' "$OUT_DIR/smoke.json" 2>/dev/null)"

# ---- GSM8K subset, through the chat template -----------------------------------------
# lm_eval's own gsm8k task names its dataset "gsm8k", an id huggingface_hub 1.x no longer resolves
# (HfUriError): the task runs from a copy of it that names the same set openai/gsm8k.
TASKS=gsm8k
TASK=$(python3 -c 'import os, lm_eval; print(os.path.join(os.path.dirname(lm_eval.__file__), "tasks", "gsm8k", "gsm8k.yaml"))' 2>/dev/null)
if [ -f "$TASK" ]; then
    sed 's|^dataset_path: gsm8k$|dataset_path: openai/gsm8k|' "$TASK" > "$OUT_DIR/gsm8k.yaml" && TASKS=$OUT_DIR/gsm8k.yaml
fi
echo "[correctness] GSM8K 5-shot through the chat template, first $LIMIT questions, pass at >= $MIN"
HF_HUB_OFFLINE=0 lm_eval --model local-chat-completions \
    --model_args "model=$WIRE,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=32,max_retries=3" \
    --tasks "$TASKS" --num_fewshot 5 --apply_chat_template --fewshot_as_multiturn --limit "$LIMIT" \
    --batch_size 1 --gen_kwargs "temperature=0,max_gen_toks=$MAX_TOKENS" --seed 1234 \
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
echo "[correctness] GSM8K exact_match (flexible), chat template = $score on $LIMIT questions: $([ "$passed" = 1 ] && echo PASS || echo FAIL)"
exit 0
