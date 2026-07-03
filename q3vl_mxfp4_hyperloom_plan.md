# Q3VL MXFP4 (W4A4 + UA + fuse_quant) — Hyperloom run plan & asks

> Target: reproduce/optimize the ml-perf best MXFP4 config for Qwen3-VL-235B-A22B-Instruct
> on MI355X (gfx950), TP1, using Hyperloom + Magpie, against the multimodal workload.
> Budget: ~2h. TP1 ⇒ up to 8 parallel single-GPU experiments.
>
> Status: **BLOCKED** — cannot start the target server from inside the Hyperloom
> container (no docker, checkpoint not reachable). Needs one of the asks below.
>
> NOTE: `/mnt/ml-perf` is read-only inside the container, so this file lives at
> `/workspace/Hyperloom/q3vl_mxfp4_hyperloom_plan.md` (bind-mounted → visible on host).
> Copy it into the ml-perf repo from the host if you want it tracked there.

---

## 1. Goal

Run the ml-perf "best" MXFP4 serve config under Hyperloom and optimize it for the
multimodal workload below.

**Target serve config** (`ml-perf qwen3-vl/configs/server/tp1_mxfp4.yaml`, branch
`feat/jakki-quant-mxfp4`) — the "W4A4 + UA + fuse_quant + FULL_AND_PIECEWISE" entry
(results-tracker: IMG-C vllm22, 9.20 QPS, F1 0.7662 @8K):

```yaml
tensor_parallel_size: 1
max_model_len: 32768
max_number_of_batched_tokens: 32768
max_num_seqs: 2048
enable_expert_parallel: false          # FP4 MoE + EP faults on gfx950 (aiter#2343)
mm_encoder_tp_mode: "data"
async_scheduling: true
enable_chunked_prefill: true
enable_prefix_caching: false
gpu_memory_utilization: 0.90
compilation_config: '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["+rms_norm", "+quant_fp8"]}'
kv_cache_dtype: fp8_e4m3
quantization: quark
attention_backend: ROCM_AITER_UNIFIED_ATTN   # "UA" — vs ROCM_AITER_FA baseline
# MANDATORY serve-time env: AITER_ONLINE_TUNE=0
```

**Workload (multimodal, dataset-driven on the real ml-perf client; for Magpie's
`vllm_mi355x_mm.sh` synthetic proxy use):**
- `CONC (mc) = 32`
- `ISL = 1024`
- `OSL = 150`
- 1 image/request, default 512×512 (`random-mm`), `--ignore-eos`
- Optimize *this exact* workload (latency/throughput at mc=32).

---

## 2. Environment found (inside Hyperloom container `miramar350-...-d03-4`)

| Item | Value |
|---|---|
| GPUs | 8× gfx950 (MI355X), visible to torch |
| vLLM | `0.23.0+rocm723` (root-owned, **not writable**, no sudo) |
| aiter | `amd-aiter 0.1.13.post1` (the *good* one per mxfp4-v2 §5; not regressed 0.1.15-rc0) |
| quark | `amd-quark 0.11.2` |
| Magpie | importable; `scripts/benchmark/vllm_mi355x_mm.sh` present (MM client) |
| Disk | /models 120G free, /dev/shm 64G, / 120G |
| Docker | **NONE** — no docker/podman/nerdctl/ctr, no /var/run/docker.sock, PID1=`tail` |
| HF token | works (user `Vorapolamd`, org `amd`, **fine-grained**) |

---

## 3. Blockers (why it can't run as-is)

1. **Cannot launch the target image from here.** The image
   `amdsiloai/vllm-private:mlperf6.1-q3vl-r72-w4a4-fusemoe-20260620` (carries the
   patched vLLM r72 + patches 0005/0006/fuse_moe **and** the baked W4A4 quark
   checkpoint) can only be pulled/run on the **Docker host**. This Hyperloom
   container has **no container runtime and no docker socket**, so I cannot
   `docker pull`/`run`/`exec` it.

2. **Checkpoint not reachable via token.** The fine-grained token lists 568
   `amd/` models but **no MXFP4 Qwen3-VL** among them (only
   `amd/Qwen3-VL-235B-A22B-Instruct-ptpc`, which is FP8 PTPC, not MXFP4).
   `amd/Qwen3-VL-235B-A22B-Instruct-MXFP4-quark` and the `…-W4A4-quark` variant
   both **404** (not in the token's allow-list). ⇒ the W4A4 quark weights
   effectively live **only inside the image**.

3. **Stack/patch mismatch if run in-place.** `tp1_mxfp4.yaml` requires the
   patched **v0.22 / a65093c / ROCm-7.2** image (patches 0005 weight-layout +
   0006 fuse_quant). This container is **0.23**, vLLM is **not writable**, and
   `mxfp4-v2.md §5` warns the 0.23/AITER-0.1.15 path regressed. (Userspace
   shadow-patching is possible but risky and needs the weights anyway.)

> Hard gate = (1)+(2): without a running server or the weights, nothing runs.

---

## 4. Asks (pick one) — what unblocks me

### Option A — You launch the server on the host (simplest)
Start the image on the host with `--network host` and bring up vLLM with the
`tp1_mxfp4` knobs + `AITER_ONLINE_TUNE=0`, then give me the **base URL/port**.
Since it's host-networked I can reach `localhost:PORT` from inside Hyperloom and
drive the **Magpie MM client** (`MAGPIE_RUN_PHASE=client`,
`BENCHMARK_BASE_URL=...`) + the optimization sweep.
- Limitation: I optimize against whatever server(s) you start. For full 8-way
  parallel TP1 experiments you'd start up to 8 replicas on different ports/GPUs.

Example host launch (server only, one replica on GPU 0):
```bash
docker run -d --name q3vl-mxfp4-0 --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --shm-size 64g -e ROCR_VISIBLE_DEVICES=0 -e AITER_ONLINE_TUNE=0 \
  amdsiloai/vllm-private:mlperf6.1-q3vl-r72-w4a4-fusemoe-20260620 \
  bash -lc 'vllm serve <baked-ckpt-path> --port 8000 \
    --tensor-parallel-size 1 --quantization quark \
    --max-model-len 32768 --max-num-seqs 2048 \
    --max-num-batched-tokens 32768 --gpu-memory-utilization 0.90 \
    --kv-cache-dtype fp8_e4m3 --no-enable-prefix-caching \
    --async-scheduling --enable-chunked-prefill \
    --mm-encoder-tp-mode data --trust-remote-code \
    --compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"+rms_norm\",\"+quant_fp8\"]}" \
    --attention-backend ROCM_AITER_UNIFIED_ATTN'
# tell me: base-url (e.g. http://localhost:8000) and the baked ckpt path
```

### Option B — Expose docker to me (best for the 2h budget)
Mount the socket into this Hyperloom container (or give me a host shell with
docker). Then I do everything end-to-end: pull, run, serve **8× TP1 replicas**,
sweep configs in parallel, optimize, collect results.
```bash
# on host: restart the Hyperloom container adding:
#   -v /var/run/docker.sock:/var/run/docker.sock
# (and ensure my uid can use it, or run docker via the mounted sock)
```

### Option C — Give me the weights, I patch here (fallback, riskier)
Export the baked checkpoint from the image to `/data2/hf_hub_cache` (→ `/models`),
or add the repo to the token's allow-list. I then userspace-shadow-patch this
0.23 vLLM with patches 0005/0006 (`PYTHONPATH` overlay), run the coherence gate,
and benchmark. Risk: patches authored for v0.22 may not apply cleanly to 0.23;
coherence not guaranteed.

**Recommendation: Option B** (full parallel use of 8 GPUs within the budget),
else Option A.

---

## 5. Plan once unblocked

1. **Bring up server(s)** — TP1, `tp1_mxfp4` knobs + `AITER_ONLINE_TUNE=0`.
   Option B: 8 replicas (GPU 0–7) for parallel sweeps.
2. **Coherence gate** (mandatory per mxfp4-v2 §4): one normal prompt must return
   coherent text; server log must show `Q3VL native MXFP4 MoE layout fix active`
   and backend `AITER_MXFP4_BF16`. Abort/flag if garbled.
3. **Baseline MM benchmark** via `vllm_mi355x_mm.sh` (client phase) at the target
   workload: `CONC=32 ISL=1024 OSL=150`, 1×512×512 image, `--ignore-eos`.
   Capture `output_throughput, mean/p99 TTFT, TPOT, E2EL, completed, duration`.
4. **Optimize for mc=32 / isl1k / osl150** — parallel TP1 experiments over, e.g.:
   - `max_num_seqs` (2048 is large for mc=32 → try 32/64/128/256)
   - `max_num_batched_tokens` (32768 → 4096/8192/16384)
   - `cudagraph_mode` FULL_AND_PIECEWISE vs FULL vs PIECEWISE
   - `attention_backend` UA vs AITER_FA (UA is the target; confirm it wins here)
   - `gpu_memory_utilization` 0.90→0.92/0.95 headroom check
   - `NUM_PROMPTS`/warmups sizing for stable steady-state at mc=32
5. **Rank** by the workload SLA (throughput at mc=32, plus TTFT/TPOT), keep the
   winner, write a short results table + the exact winning serve cmd.
6. (If time) sanity vs FP8 baseline (`tp1_fp8_baseline.yaml`) for context.

---

## 6. Key references
- `ml-perf qwen3-vl/configs/server/tp1_mxfp4.yaml` (target) & `tp1_fp8_baseline.yaml`
- `ml-perf qwen3-vl/docs/mxfp4-v2.md` (patch 0005 + `AITER_ONLINE_TUNE=0` rationale, coherence gate)
- `ml-perf qwen3-vl/docker/patches/000{5,6}-*.patch` (W4A4 layout + fuse_quant)
- Magpie `scripts/benchmark/vllm_mi355x_mm.sh` (MM client; env: MODEL/TP/CONC/ISL/OSL/IMAGE_*/NUM_PROMPTS/MAGPIE_RUN_PHASE/BENCHMARK_BASE_URL)
- results-tracker.md (best = W4A4+UA+fuse_quant+FULL_AND_PIECEWISE, 9.20 QPS)
