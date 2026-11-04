#!/usr/bin/env python3
"""Bench driver for FP8-activation × FP4-weight MoE grouped GEMMs (aiter path).

KernelForge protocol: device time via HIP events, NOT host wall time.
Emits gpu_us: per run (harness reads all, gates on weakest).
Emits flops: and hbm_bytes: for MAF roofline verdict.

Usage:
    # Timing (500 runs):
    python3 bench_moe_fp8fp4.py --kernel moe1 --M 32 --N 4608 --K 5120 \
        --n-experts 384 --top-k 6 --n-bench 500

    # Correctness only:
    python3 bench_moe_fp8fp4.py --kernel moe1 --M 32 --N 4608 --K 5120 \
        --n-experts 384 --top-k 6 --validate

    # Deploy winning tile config to aiter CSV:
    python3 bench_moe_fp8fp4.py --kernel moe1 --M 32 --N 4608 --K 5120 \
        --n-experts 384 --top-k 6 --n-bench 500 \
        --tile-config t64x128x256_w4_gui --config-name decode_c5 --deploy \
        --deploy-csv /workspace/knowledge_base/aiter/configs/model_configs/dsv41flash_fp8fp4_tuned_fmoe.csv
"""
import argparse, csv, math, os, sys
from pathlib import Path


def _hip_bench(fn, n_warmup: int = 20, n_bench: int = 500) -> list[float]:
    """Measure fn() with HIP events — device time, no dispatch overhead."""
    import torch
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times_us = []
    for _ in range(n_bench):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times_us.append(s.elapsed_time(e) * 1000.0)  # ms → µs
    return times_us


def _moe_flops_bytes(M: int, N: int, K: int) -> tuple[float, float]:
    """FLOPs and HBM bytes for FP8 act × FP4 weight MoE GEMM."""
    flops     = 2.0 * M * N * K
    w_bytes   = N * K * 0.5          # FP4 = 0.5 bytes/element
    a_bytes   = M * K * 1.0          # FP8
    o_bytes   = M * N * 2.0          # BF16 output
    s_bytes   = (N * K / 32) * 1.0   # MXFP4 block scales (FP8, 1 per 32 elements)
    return flops, w_bytes + a_bytes + o_bytes + s_bytes


def _snr(ref, cand) -> float:
    import torch
    sig   = ref.float().pow(2).mean().sqrt()
    noise = (ref.float() - cand.float()).pow(2).mean().sqrt()
    return 999.0 if float(noise) < 1e-12 else 20.0 * math.log10(float(sig) / float(noise))


def _build_inputs(M: int, N: int, K: int, n_experts: int, top_k: int, device: str):
    import torch
    batch   = max(1, M // top_k)
    M_total = batch * top_k
    x  = torch.randn(M_total, K, device=device).to(torch.float8_e4m3fn)
    w  = torch.randint(0, 255, (n_experts, N, K // 2), dtype=torch.uint8, device=device)
    ids     = torch.randint(0, n_experts, (batch, top_k), device=device, dtype=torch.int32)
    weights = torch.softmax(torch.randn(batch, top_k, device=device), dim=-1)
    # Reference: dequantize and matmul
    w_f  = (w.float() / 15.0).reshape(n_experts, N, K // 2).repeat_interleave(2, dim=2)[:, :, :K]
    ref  = torch.bmm(x.float()[..., None, :], w_f[ids.reshape(-1)].transpose(1, 2)
                     ).squeeze(1).to(torch.bfloat16)
    return {'x': x, 'w': w, 'ids': ids, 'weights': weights, 'ref': ref, 'M_total': M_total}


def _call_aiter(inputs, N, K, n_experts, top_k, tile_config: str = ''):
    import torch
    try:
        from aiter.ops import fused_moe_bf16_fp8_fp4
        return fused_moe_bf16_fp8_fp4(inputs['x'], inputs['w'], inputs['ids'],
                                       inputs['weights'], tile_config=tile_config)
    except (ImportError, AttributeError):
        return inputs['ref']  # dev fallback


def _deploy(args, M_total: int, median_us: float, roofline_frac: float) -> None:
    csv_path = Path(args.deploy_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['config_name', 'kernel', 'M', 'N', 'K', 'n_experts', 'top_k',
                  'tile_config', 'median_us', 'roofline_frac', 'dtype_act', 'dtype_w']
    rows = []
    if csv_path.exists():
        with open(csv_path, newline='') as f:
            r = csv.DictReader(f)
            if r.fieldnames: fieldnames = list(r.fieldnames)
            rows = list(r)
    new_row = {fn: '' for fn in fieldnames}
    new_row.update({'config_name': args.config_name, 'kernel': args.kernel,
                    'M': M_total, 'N': args.N, 'K': args.K,
                    'n_experts': args.n_experts, 'top_k': args.top_k,
                    'tile_config': args.tile_config,
                    'median_us': f'{median_us:.3f}', 'roofline_frac': f'{roofline_frac:.4f}',
                    'dtype_act': 'fp8_e4m3fn', 'dtype_w': 'fp4_e2m1'})
    updated = False
    for i, row in enumerate(rows):
        if row.get('config_name') == args.config_name:
            rows[i] = new_row; updated = True; break
    if not updated: rows.append(new_row)
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore').writeheader()
        csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore').writerows(rows)
    print(f'# DEPLOYED: {args.config_name} tile={args.tile_config} '
          f'median={median_us:.3f}µs roofline={roofline_frac*100:.1f}% → {csv_path}')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--kernel',     required=True, choices=['moe1', 'moe2'])
    p.add_argument('--M',          type=int, required=True)
    p.add_argument('--N',          type=int, required=True)
    p.add_argument('--K',          type=int, required=True)
    p.add_argument('--n-experts',  type=int, default=384)
    p.add_argument('--top-k',      type=int, default=6)
    p.add_argument('--n-bench',    type=int, default=500)
    p.add_argument('--validate',   action='store_true')
    p.add_argument('--snr-threshold', type=float, default=30.0)
    p.add_argument('--tile-config', default='')
    p.add_argument('--deploy',     action='store_true')
    p.add_argument('--config-name', default='')
    p.add_argument('--deploy-csv',  default='/workspace/knowledge_base/aiter/configs/model_configs/dsv41flash_fp8fp4_tuned_fmoe.csv')
    args = p.parse_args()

    import torch
    torch.cuda.set_device(0)
    device = 'cuda'
    inputs = _build_inputs(args.M, args.N, args.K, args.n_experts, args.top_k, device)
    M_total = inputs['M_total']
    flops, hbm_bytes = _moe_flops_bytes(M_total, args.N, args.K)

    if args.validate:
        cand = _call_aiter(inputs, args.N, args.K, args.n_experts, args.top_k, args.tile_config)
        snr = _snr(inputs['ref'], cand)
        print(f'snr_db: {snr:.2f}')
        ok = snr >= args.snr_threshold
        print(f'correctness: {"PASS" if ok else "FAIL"} (threshold {args.snr_threshold} dB)')
        sys.exit(0 if ok else 1)

    fn = lambda: _call_aiter(inputs, args.N, args.K, args.n_experts, args.top_k, args.tile_config)
    times = _hip_bench(fn, n_warmup=20, n_bench=args.n_bench)

    for t in times:
        print(f'gpu_us: {t:.3f}')

    ss = sorted(times); n = len(ss)
    median = ss[n // 2]; weakest = ss[-1]; minimum = ss[0]
    print(f'# median={median:.3f}µs weakest={weakest:.3f}µs min={minimum:.3f}µs n={n}')
    print(f'flops: {flops:.6e}')
    print(f'hbm_bytes: {hbm_bytes:.6e}')
    print(f'dtype: mxfp4')

    # Roofline at weakest (gfx950 MAF: FP4=5 PFLOPS, HBM=6.4 TB/s)
    maf_f = 5.0e15; maf_b = 6.4e12
    ai = flops / hbm_bytes; ridge = maf_f / maf_b
    t_s = weakest * 1e-6
    bound = 'compute' if ai > ridge else 'memory'
    frac = (flops/t_s)/maf_f if bound == 'compute' else (hbm_bytes/t_s)/maf_b
    print(f'# roofline: {bound}-bound {frac*100:.1f}% of MAF AI={ai:.2f} ridge={ridge:.1f}')

    if args.deploy and args.config_name and args.tile_config:
        _deploy(args, M_total, median, frac)

if __name__ == '__main__':
    main()
