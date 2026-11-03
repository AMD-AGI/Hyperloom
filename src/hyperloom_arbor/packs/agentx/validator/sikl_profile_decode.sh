#!/usr/bin/env bash
# SIKL decode profile — model-aware dispatcher.
# Detects model from MODEL_PATH/MODEL_NAME and routes to the right profiler.
# Falls back to torch.profiler-based profiling when no SIKL model dir exists.
#
# Outputs: $SIKL_OUTPUT (JSON) with per-op GPU time breakdown.
# Injected by Arbor: CONTAINER, MODEL_PATH, TP, SIKL_OUTPUT

set -euo pipefail

: "${MODEL_PATH:?must set MODEL_PATH}"
: "${TP:=4}"
: "${SIKL_OUTPUT:=/tmp/sikl_decode_result.json}"
: "${CONC:=8}"
: "${ISL:=4096}"
: "${REPEAT:=5}"

echo "[sikl] Model: $MODEL_PATH  TP=$TP  CONC=$CONC  ISL=$ISL"
echo "[sikl] Output: $SIKL_OUTPUT"

MODEL_NAME_LOWER=$(echo "${MODEL_PATH}${MODEL_NAME:-}" | tr '[:upper:]' '[:lower:]')

# ── Route to model-specific SIKL dir if available ──────────────────────────
if echo "$MODEL_NAME_LOWER" | grep -q "glm"; then
    SIKL_DIR="${SIKL_DIR:-/it-share-4/yanyuan-glm52-cache/SIKL/sikl/benchmarks/models/glm52}"
    if [ -d "$SIKL_DIR" ]; then
        cd "$SIKL_DIR"
        python3 profile_decode.py \
          --model "$MODEL_PATH" --tp-size "$TP" \
          --enable-dp-attention --dp-size "$TP" \
          --dsa-topk-backend aiter \
          --ISL "$ISL" --CONC "$CONC" --repeat "$REPEAT" \
          2>&1 | tee /tmp/sikl_decode.log
        _parse_sikl_log "$SIKL_OUTPUT"
        exit 0
    fi
fi

# ── Fallback: torch.profiler-based op breakdown ────────────────────────────
# Works for any vLLM model. Starts a minimal server, sends decode requests,
# uses vLLM /start_profile + /stop_profile to capture GPU kernel breakdown.
echo "[sikl] No model-specific SIKL dir — using vLLM profiler fallback"

python3 - "$MODEL_PATH" "$TP" "$CONC" "$ISL" "$REPEAT" "$SIKL_OUTPUT" << 'PYEOF'
import json, os, re, subprocess, sys, tempfile, time, threading
import urllib.request, urllib.error

model_path, tp, conc, isl, repeat, out_path = sys.argv[1:]
tp = int(tp); conc = int(conc); isl = int(isl); repeat = int(repeat)
port = int(os.environ.get("PORT", "8899"))  # use different port to not clash

ROCR = os.environ.get("ROCR_VISIBLE_DEVICES", "0,1,2,3")

# ── 1. Start vLLM server ────────────────────────────────────────────────────
cmd = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", model_path,
    "--tensor-parallel-size", str(tp),
    "--port", str(port),
    "--max-model-len", "32768",
    "--max-num-seqs", str(conc),
    "--moe-backend", "aiter",
    "--gpu-memory-utilization", "0.85",
    "--tokenizer-mode", "deepseek_v41",
    "--disable-uvicorn-access-log",
]
env = os.environ.copy()
env["ROCR_VISIBLE_DEVICES"] = ROCR
env["VLLM_ROCM_USE_AITER"] = "1"
env["VLLM_ROCM_USE_AITER_MOE"] = "1"
env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
env["PYTHONUNBUFFERED"] = "1"

print("[sikl] Starting vLLM server for profiling...")
srv = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# Wait for server ready
base = f"http://localhost:{port}"
deadline = time.time() + 600
ready = False
while time.time() < deadline:
    try:
        urllib.request.urlopen(f"{base}/health", timeout=2)
        ready = True
        break
    except Exception:
        if srv.poll() is not None:
            print("[sikl] Server died during startup")
            break
        time.sleep(5)

if not ready:
    srv.terminate()
    json.dump({"status": "server_failed", "ops": [], "top_bottleneck": None}, open(out_path, "w"))
    sys.exit(1)

print("[sikl] Server ready. Starting profile capture...")

# ── 2. Start profiling ───────────────────────────────────────────────────────
try:
    urllib.request.urlopen(urllib.request.Request(
        f"{base}/start_profile", method="POST",
        headers={"Content-Type": "application/json"}, data=b'{}'
    ), timeout=10)
except Exception as e:
    print(f"[sikl] /start_profile failed: {e}")

# ── 3. Send decode-heavy requests ────────────────────────────────────────────
import json as _json
model_name = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V4.1-Flash")
prompt = "A" * isl  # approximate ISL

for _ in range(repeat):
    req_data = _json.dumps({
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0,
    }).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{base}/v1/completions", method="POST",
            headers={"Content-Type": "application/json"},
            data=req_data
        ), timeout=120)
    except Exception as e:
        print(f"[sikl] request failed: {e}")

# ── 4. Stop profiling ────────────────────────────────────────────────────────
trace_path = None
try:
    resp = urllib.request.urlopen(urllib.request.Request(
        f"{base}/stop_profile", method="POST",
        headers={"Content-Type": "application/json"}, data=b'{}'
    ), timeout=30)
    result = _json.loads(resp.read())
    trace_path = result.get("trace_file")
    print(f"[sikl] Trace at: {trace_path}")
except Exception as e:
    print(f"[sikl] /stop_profile failed: {e}")

srv.terminate()
srv.wait(timeout=30)

# ── 5. Parse trace ───────────────────────────────────────────────────────────
ops = []
if trace_path and os.path.exists(trace_path):
    import collections
    trace = _json.load(open(trace_path))
    by_name = collections.defaultdict(lambda: {"dur": 0, "count": 0})
    for e in trace.get("traceEvents", []):
        if e.get("cat") in ("kernel", "gpu_memcpy") and "dur" in e:
            by_name[e["name"]]["dur"] += e["dur"]
            by_name[e["name"]]["count"] += 1
    total = sum(v["dur"] for v in by_name.values()) or 1
    for name, v in sorted(by_name.items(), key=lambda x: -x[1]["dur"])[:20]:
        ops.append({
            "op": name[:60],
            "ms": v["dur"] / 1000,
            "pct": v["dur"] / total * 100,
            "count": v["count"],
        })
    print("[sikl] Top GPU ops:")
    for op in ops[:8]:
        print(f"  {op['op'][:50]:50s}  {op['ms']:7.2f}ms  {op['pct']:5.1f}%")
else:
    # No trace — emit placeholder so Arbor can proceed
    print("[sikl] No trace file. Emitting placeholder result.")
    ops = [{"op": "unknown_no_trace", "ms": 0, "pct": 100, "count": 0}]

result = {
    "status": "ok" if trace_path else "no_trace",
    "model": model_path,
    "tp": tp,
    "conc": conc,
    "isl": isl,
    "ops": ops,
    "top_bottleneck": ops[0]["op"] if ops else None,
}
_json.dump(result, open(out_path, "w"), indent=2)
print(f"[sikl] Written to {out_path}")
PYEOF
