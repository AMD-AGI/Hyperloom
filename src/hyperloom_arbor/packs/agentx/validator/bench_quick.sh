#!/usr/bin/env bash
# Quick bench — real throughput measurement in ~60 seconds.
# Uses SGLang/vLLM's built-in benchmark_serving instead of aiperf trace replay.
# For testing the harness and fast iteration. Not for InferenceX submission.

set -euo pipefail

: "${PORT:?Arbor must inject PORT}"
: "${MODEL_NAME:?Arbor must inject MODEL_NAME}"
: "${CONCURRENCY:?Arbor must inject CONCURRENCY}"
: "${BENCH_OUTPUT:?Arbor must inject BENCH_OUTPUT}"

NUM_PROMPTS="${NUM_PROMPTS:-100}"
ISL="${ISL:-1024}"
OSL="${OSL:-256}"
TP="${TP:-4}"

echo "[bench_quick] PORT=$PORT MODEL=$MODEL_NAME CONC=$CONCURRENCY NUM_PROMPTS=$NUM_PROMPTS ISL=$ISL OSL=$OSL"

# Use vllm bench serve (works for both vLLM and SGLang OpenAI-compatible endpoints)
RESULT_DIR="$(dirname "$BENCH_OUTPUT")"
mkdir -p "$RESULT_DIR"
RAW_OUTPUT="$RESULT_DIR/bench_raw.json"

python3 -c "
import json, time, sys, urllib.request, urllib.error, concurrent.futures, threading

PORT = $PORT
MODEL = '$MODEL_NAME'
CONC = $CONCURRENCY
NUM = $NUM_PROMPTS
ISL = $ISL
OSL = $OSL

url = f'http://localhost:{PORT}/v1/chat/completions'
prompt_text = 'Write a detailed essay about the history of computing. ' * (ISL // 20)

completed = 0
total_input_tokens = 0
total_output_tokens = 0
errors = 0
lock = threading.Lock()

def send_request(i):
    global completed, total_input_tokens, total_output_tokens, errors
    try:
        data = json.dumps({
            'model': MODEL,
            'messages': [{'role': 'user', 'content': prompt_text}],
            'max_tokens': OSL,
            'temperature': 0.7,
        }).encode()
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read())
        usage = result.get('usage', {})
        with lock:
            completed += 1
            total_input_tokens += usage.get('prompt_tokens', 0)
            total_output_tokens += usage.get('completion_tokens', 0)
    except Exception as e:
        with lock:
            errors += 1

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as pool:
    list(pool.map(send_request, range(NUM)))
elapsed = time.time() - start

output_tps = total_output_tokens / elapsed if elapsed > 0 else 0
total_tps = (total_input_tokens + total_output_tokens) / elapsed if elapsed > 0 else 0
output_tps_per_gpu = output_tps / ($TP)
total_tps_per_gpu = total_tps / ($TP)

result = {
    'output_throughput': round(output_tps_per_gpu, 1),
    'total_throughput_per_gpu': round(total_tps_per_gpu, 1),
    'conc': CONC,
    'tp': $TP,
    'num_requests_successful': completed,
    'num_requests_total': NUM,
    'elapsed_seconds': round(elapsed, 1),
    'total_output_tokens': total_output_tokens,
    'total_input_tokens': total_input_tokens,
    'errors': errors,
    'status': 'ok' if completed > 0 else 'failed',
}

with open('$BENCH_OUTPUT', 'w') as f:
    json.dump(result, f, indent=2)

print(f'[bench_quick] {completed}/{NUM} requests in {elapsed:.1f}s')
print(f'[bench_quick] output_throughput={output_tps_per_gpu:.1f} tok/s/GPU  total={total_tps_per_gpu:.1f} tok/s/GPU')
print(f'[bench_quick] errors={errors}')
"
