#!/usr/bin/env python3
"""Closed-loop throughput probe against a vLLM OpenAI endpoint.
Usage: bench_one.py <port> <label> [conc] [nreq] [isl] [osl]
Prints one JSON line with output tok/s, req/s, TTFT-ish, p50 latency.
Identical prompt/params across variants => valid relative comparison."""
import time, json, threading, urllib.request, sys, statistics

port  = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
label = sys.argv[2] if len(sys.argv) > 2 else f"port{port}"
CONC  = int(sys.argv[3]) if len(sys.argv) > 3 else 32
NREQ  = int(sys.argv[4]) if len(sys.argv) > 4 else 64
ISL   = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
OSL   = int(sys.argv[6]) if len(sys.argv) > 6 else 128

BASE  = f"http://127.0.0.1:{port}"
MODEL = "gpt-oss-120b"

# Unique prompt per request to defeat automatic prefix caching (which would
# otherwise make prefill free and inflate throughput unrealistically). Weave a
# per-request integer stream so every request's KV prefix is distinct, matching
# Magpie's random-token methodology.
def make_prompt(idx):
    words = [f"w{(idx * 7919 + i * 104729) % 100000}" for i in range(ISL)]
    return " ".join(words)

results = {}
def one(idx):
    prompt = make_prompt(idx)
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": OSL,
                       "temperature": 0.0, "ignore_eos": True}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
        dt = time.time() - t0
        ct = resp.get("usage", {}).get("completion_tokens", 0)
        results[idx] = (dt, ct)
    except Exception as e:
        results[idx] = (None, str(e)[:100])

sem = threading.Semaphore(CONC)
def worker(i):
    with sem:
        one(i)

t0 = time.time()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(NREQ)]
for th in threads: th.start()
for th in threads: th.join()
wall = time.time() - t0

ok  = [(d, c) for d, c in results.values() if d is not None and isinstance(c, int)]
err = [c for d, c in results.values() if d is None]
toks = sum(c for _, c in ok)
lat  = sorted(d for d, _ in ok)
p50  = round(statistics.median(lat), 2) if lat else None
print(json.dumps({
    "variant": label, "port": port, "conc": CONC, "nreq": NREQ,
    "isl_approx": ISL, "osl": OSL, "ok": len(ok), "err": len(err),
    "wall_s": round(wall, 1), "out_tok": toks,
    "out_tok_s": round(toks / wall, 1) if wall else 0,
    "req_s": round(len(ok) / wall, 2) if wall else 0,
    "p50_req_latency_s": p50,
    "err_sample": err[0] if err else None,
}))
