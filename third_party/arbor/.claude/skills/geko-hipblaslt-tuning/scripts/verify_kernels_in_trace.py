#!/usr/bin/env python3
"""Verify the e2e serving run actually invoked the geko-tuned kernels: check whether the tuned kernel
names (from final_results.csv) appear as dispatched Cijk_ kernels in the merged-lib torch-profiler trace."""
import gzip, json, glob, os
from collections import defaultdict

WS = os.environ["WS"]  # geko_agentx workspace root
D = os.environ.get("TORCHPROF_DIR", WS + "/experiments/torchprof")
kt = set(x for x in open(D+"/kt.txt").read().split("\n") if x)
kr = set(x for x in open(D+"/kr.txt").read().split("\n") if x)
f = sorted(glob.glob(D+"/trace/*rank0*.json.gz"))[0]
ev = json.load(gzip.open(f))["traceEvents"]

def norm(n): return n.split(".kd")[0]
cij = defaultdict(float); cnt = defaultdict(int)
for e in ev:
    if e.get("cat") == "kernel":
        nm = e.get("name","")
        if nm.startswith("Cijk_"):
            cij[norm(nm)] += float(e.get("dur",0)); cnt[norm(nm)] += 1

trace = set(cij)
kt_n = {norm(k) for k in kt}; kr_n = {norm(k) for k in kr}
tuned_hit = kt_n & trace
ref_hit = kr_n & trace
print(f"distinct Cijk_ kernels in e2e trace: {len(trace)}")
print(f"TUNED kernels present in e2e trace:     {len(tuned_hit)}/{len(kt_n)}")
print(f"REFERENCE kernels present in e2e trace: {len(ref_hit)}/{len(kr_n)}")
tt = sum(cij[k] for k in tuned_hit); tc = sum(cnt[k] for k in tuned_hit)
print(f"tuned-kernel GPU time in trace: {tt/1000:.2f} ms over {tc} dispatches")
print("\nsample tuned kernels PRESENT in e2e trace:")
for k in sorted(tuned_hit, key=lambda x:-cij[x])[:5]:
    print(f"  + {cij[k]/1000:6.2f} ms ({cnt[k]:5d}x)  {k[:88]}")
miss = sorted(kt_n - trace)
if miss:
    print(f"\ntuned kernels NOT seen ({len(miss)}) — likely their shape wasn't exercised in this window:")
    for k in miss[:3]: print("  -", k[:88])
