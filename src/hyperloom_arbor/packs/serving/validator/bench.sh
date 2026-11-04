#!/usr/bin/env bash
# The serving template's bench: `vllm bench serve` on random tokens at the campaign's operating
# point, with one discarded warmup, written as Arbor's bench_output.json. From the balanced
# pack's bench_8k1k.sh, with the operating point taken from the environment.
#
# Arbor injects: PORT, MODEL_PATH, MODEL_NAME, CONCURRENCY, NUM_PROMPTS, BENCH_OUTPUT, plus the
# config's server_env. From server_env (or the environment), optional:
#   ISL, OSL        input and output length (default 1024 / 1024)
#   NUM_GPUS        GPUs the server uses; output_throughput is per GPU (default TP, else 1)
#   BENCH_SEED      the measured replay's seed (default 1)
# The server is already running. The vllm bench client works against SGLang's OpenAI endpoint too.
set -euo pipefail

: "${PORT:?Arbor must inject PORT}"
: "${MODEL_PATH:?Arbor must inject MODEL_PATH}"
: "${CONCURRENCY:?Arbor must inject CONCURRENCY}"
: "${NUM_PROMPTS:?Arbor must inject NUM_PROMPTS}"
: "${BENCH_OUTPUT:?Arbor must inject BENCH_OUTPUT}"
ISL=${ISL:-1024}
OSL=${OSL:-1024}
SEED=${BENCH_SEED:-1}
NUM_GPUS=${NUM_GPUS:-${TP:-1}}
OUT_DIR=$(dirname "$BENCH_OUTPUT")
mkdir -p "$OUT_DIR"

# Post to the name the server registered, not an assumption (the v2 pack's correctness.sh lost a
# correct iteration to a 404 on exactly this).
WIRE=$(curl -sf "http://127.0.0.1:$PORT/v1/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null) \
    || WIRE=${MODEL_NAME:-model}

B=(vllm bench serve --backend openai --host 127.0.0.1 --port "$PORT" --endpoint /v1/completions
   --model "$WIRE" --tokenizer "$MODEL_PATH" --dataset-name random --random-range-ratio 0.0
   --request-rate inf --ignore-eos --temperature 0)

WARM_OSL=$(( OSL < 256 ? OSL : 256 ))
echo "[bench] warmup: 16 prompts, $ISL/$WARM_OSL, C8, seed 7 (discarded)"
"${B[@]}" --random-input-len "$ISL" --random-output-len "$WARM_OSL" --num-prompts 16 --max-concurrency 8 \
    --seed 7 > "$OUT_DIR/bench_warmup.log" 2>&1 || echo "[bench] warmup exited non-zero (see bench_warmup.log)"

echo "[bench] measured: $NUM_PROMPTS prompts, $ISL/$OSL, C$CONCURRENCY, seed $SEED, model $WIRE"
"${B[@]}" --random-input-len "$ISL" --random-output-len "$OSL" --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" --seed "$SEED" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT_DIR" --result-filename bench_raw.json \
    2>&1 | tee "$OUT_DIR/bench_measured.log" | tail -25

python3 - "$OUT_DIR/bench_raw.json" "$BENCH_OUTPUT" "$NUM_GPUS" "$NUM_PROMPTS" "$CONCURRENCY" "$SEED" "$ISL" "$OSL" <<'PYEOF'
import json, sys
raw_p, out_p = sys.argv[1], sys.argv[2]
ngpu, nprompts, conc, seed, isl, osl = (int(x) for x in sys.argv[3:9])
r = json.load(open(raw_p))

def ms(*keys):
    for k in keys:
        v = r.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None

completed = int(r.get("completed") or 0)
tput = float(r.get("output_throughput") or 0.0)
p50_tpot, p90_tpot = ms("median_tpot_ms", "p50_tpot_ms"), ms("p90_tpot_ms")
p50_ttft, p90_ttft = ms("median_ttft_ms", "p50_ttft_ms"), ms("p90_ttft_ms")
ok = completed == nprompts and tput > 0
result = {
    "output_throughput": tput / ngpu if ok else 0.0,     # tok/s per GPU: the gated metric
    "output_throughput_total": tput,
    "total_throughput_per_gpu": float(r.get("total_token_throughput") or 0.0) / ngpu,
    "p50_interactivity": 1000.0 / p50_tpot if p50_tpot else None,
    "p90_interactivity": 1000.0 / p90_tpot if p90_tpot else None,
    "median_ttft_s": p50_ttft / 1000 if p50_ttft is not None else None,
    "p90_ttft_s": p90_ttft / 1000 if p90_ttft is not None else None,
    "median_tpot_ms": p50_tpot, "p90_tpot_ms": p90_tpot,
    "duration_s": r.get("duration"),
    "conc": conc, "num_prompts": nprompts, "num_requests_successful": completed,
    "num_gpus": ngpu, "seed": seed, "isl": isl, "osl": osl,
    "raw_path": raw_p,
    "status": "ok" if ok else f"incomplete: {completed}/{nprompts} requests completed",
}
json.dump(result, open(out_p, "w"), indent=2)
fmt = lambda v, s=".2f": format(v, s) if isinstance(v, (int, float)) else "n/a"
print(f"[bench] output_throughput={fmt(result['output_throughput'], '.1f')} tok/s/GPU "
      f"({fmt(tput, '.1f')} total)  p90_interactivity={fmt(result['p90_interactivity'])} tok/s/user  "
      f"p50 TTFT={fmt(result['median_ttft_s'], '.3f')} s  requests={completed}/{nprompts}  {result['status']}")
sys.exit(0 if ok else 1)
PYEOF
