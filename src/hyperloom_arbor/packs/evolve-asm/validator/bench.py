"""Self-contained ASM kernel bench harness for the evolve-asm pack."""

from __future__ import annotations
import re, subprocess, time
from dataclasses import dataclass, field
from typing import Optional

# gfx950 MAF peaks (self-contained, no KernelForge import)
_GFX950_MAF_FLOPS_FP8  = 5.0e15 * 0.5
_GFX950_MAF_FLOPS_BF16 = 2.5e15 * 0.5
_GFX950_MAF_FLOPS_FP4  = 10.0e15 * 0.5
_GFX950_MAF_BW         = 8.0e12 * 0.8
_DTYPE_BYTES = {"fp4":0.5,"mxfp4":0.5,"fp8":1.0,"mxfp8":1.0,"int8":1.0,"bf16":2.0,"fp16":2.0}


def _roofline_verdict(flops: float, hbm_bytes: float, us: float, dtype: str = "bf16") -> dict:
    """Compute MAF roofline verdict from analytic FLOPs + analytic HBM bytes.

    IMPORTANT: hbm_bytes here is analytic (compulsory traffic), NOT measured.
    For small-M MoE decode, real HBM traffic is much lower (L2 cache hits),
    so frac can exceed 1.0. When frac > 1.0 we cap it and flag L2-resident.
    Use rocprofv3 FETCH_SIZE+WRITE_SIZE for accurate fracs (see KernelForge
    analyze_measured). This analytic version is direction-correct but not
    suitable for acceptance testing when data is L2-resident.
    """
    w = _DTYPE_BYTES.get(dtype.lower(), 2.0)
    if w <= 0.5:   maf_flops = _GFX950_MAF_FLOPS_FP4
    elif w <= 1.0: maf_flops = _GFX950_MAF_FLOPS_FP8
    else:          maf_flops = _GFX950_MAF_FLOPS_BF16
    t = us * 1e-6
    ai = flops / hbm_bytes
    ridge = maf_flops / _GFX950_MAF_BW
    bound = "compute" if ai > ridge else "memory"
    ach_flops = flops / t
    ach_bw    = hbm_bytes / t
    frac = (ach_flops / maf_flops) if bound == "compute" else (ach_bw / _GFX950_MAF_BW)

    # frac > 1.0 means analytic hbm_bytes overestimates real HBM traffic
    # (data is L2-resident or fewer experts activated than assumed).
    # Cap at 1.0 and flag; use PMC-measured bytes for accurate fracs.
    l2_resident = frac > 1.0
    frac = min(frac, 1.0)

    if bound == "memory":
        if l2_resident:
            directive = ("L2-RESIDENT: analytic HBM estimate exceeds peak — "
                         "real HBM traffic lower than compulsory model (cache hits). "
                         "Use rocprofv3 FETCH_SIZE+WRITE_SIZE for accurate roofline.")
        elif frac >= 0.6:
            directive = "AT HBM ROOFLINE — only lever is FEWER BYTES."
        elif ai < ridge / 8:
            directive = f"OCCUPANCY-LIMITED (AI={ai:.2f}) — expose concurrency."
        else:
            directive = "MEMORY-BOUND — vectorize loads, raise num_stages."
    else:
        directive = ("NEAR FLOP ROOFLINE." if frac >= 0.6 else
                     "COMPUTE-BOUND — tune BLOCK_M/N/K, feed accumulator.")

    return {"bound": bound, "roofline_frac": round(frac, 3),
            "l2_resident": l2_resident,
            "achieved_tflops": round(ach_flops/1e12, 1),
            "achieved_bw_tbs": round(ach_bw/1e12, 2), "directive": directive}


def check_rebuild_guard(container: str, co_path: str, s_path: str) -> tuple[bool, str]:
    cmd = (f"python3 -c \"import os; co=os.path.getmtime('{co_path}'); "
           f"s=os.path.getmtime('{s_path}') if os.path.exists('{s_path}') else 0; "
           f"print('stale' if s>0 and co<s else 'ok')\"")
    r = subprocess.run(["docker","exec",container,"bash","-c",cmd],
                       capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        return False, f"mtime check failed: {r.stderr.strip()[:200]}"
    status = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else "unknown"
    if status == "stale":
        return False, ".co is OLDER than .s — edit was never compiled. Run patch_co.py."
    return True, "ok"


def check_snr(container: str, snr_cmd: str, threshold_db: float = 30.0) -> tuple[bool, float, str]:
    r = subprocess.run(["docker","exec",container,"bash","-c",snr_cmd],
                       capture_output=True, text=True, timeout=120)
    output = r.stdout + r.stderr
    m = re.search(r"snr_db[=:\s]+([\d.]+)", output)
    if m:
        snr_db = float(m.group(1))
        passed = snr_db >= threshold_db
        return passed, snr_db, f"SNR {snr_db:.1f} dB {'≥' if passed else '<'} {threshold_db} dB"
    m = re.search(r"cos[=_]sim[=:\s]+([\d.]+)", output, re.IGNORECASE)
    if m:
        cos = float(m.group(1))
        if cos >= 0.9999: return True, 999.0, f"cos_sim={cos:.6f} (exact)"
        return False, 0.0, f"cos_sim={cos:.6f} < 0.9999"
    if r.returncode != 0:
        return False, 0.0, f"SNR command failed: {output[-300:]}"
    return False, 0.0, f"No SNR data: {output[-200:]}"


@dataclass
class BenchResult:
    success: bool
    samples_us: list[float] = field(default_factory=list)
    weakest_us: float = 0.0
    median_us: float = 0.0
    min_us: float = 0.0
    n_samples: int = 0
    roofline: Optional[dict] = None
    error: str = ""
    raw_output: str = ""


def bench_co(container: str, bench_cmd: str, n_runs: int = 5,
             timeout_sec: int = 900) -> BenchResult:
    samples: list[float] = []
    all_output: list[str] = []
    for i in range(n_runs):
        r = subprocess.run(["docker","exec",container,"bash","-c",bench_cmd],
                           capture_output=True, text=True, timeout=timeout_sec)
        out = r.stdout + r.stderr
        all_output.append(out)
        if r.returncode != 0:
            return BenchResult(success=False,
                               error=f"bench run {i+1} failed (exit {r.returncode}): {out[-400:]}",
                               raw_output="\n---\n".join(all_output))
        m = re.search(r"gpu_us[=:\s]+([\d.]+)", out)
        if not m:
            m = re.search(r"(?:wall_ms|median_ms)[=:\s]+([\d.]+)", out)
            if m: samples.append(float(m.group(1)) * 1000.0)
            else: return BenchResult(success=False,
                                     error=f"run {i+1}: no gpu_us in output: {out[-300:]}",
                                     raw_output="\n---\n".join(all_output))
        else:
            samples.append(float(m.group(1)))
    if not samples:
        return BenchResult(success=False, error="no timing samples")
    combined = "\n".join(all_output)
    ss = sorted(samples)
    result = BenchResult(success=True, samples_us=samples,
                         weakest_us=round(ss[-1], 3),
                         median_us=round(ss[len(ss)//2], 3),
                         min_us=round(ss[0], 3),
                         n_samples=len(samples), raw_output=combined[-2000:])
    fm = re.search(r"flops[=:\s]+([\d.eE+]+)", combined)
    bm = re.search(r"hbm_bytes[=:\s]+([\d.eE+]+)", combined)
    if fm and bm:
        try:
            flops = float(fm.group(1)); hbm = float(bm.group(1))
            dm = re.search(r"\bdtype[=:\s]+([a-zA-Z0-9]+)", combined)
            dtype = dm.group(1).lower() if dm else "bf16"
            result.roofline = _roofline_verdict(flops, hbm, ss[-1], dtype)
        except (ValueError, ZeroDivisionError):
            pass
    return result
