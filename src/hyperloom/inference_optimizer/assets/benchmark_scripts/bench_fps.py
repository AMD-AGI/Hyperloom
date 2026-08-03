#!/usr/bin/env python3
# Copyright 2026. MIT-licensed (this file is part of hyperloom-hy).
"""
Generated-FPS benchmark for HY-WorldPlay (HunyuanWorld 1.5) on AMD / ROCm.

This measures how fast the model *generates video frames* (the metric that
matters for a real-time streaming world model), for the autoregressive baseline
with no quantization. It does NOT modify the upstream HY-WorldPlay repo; it
imports its pipeline and wraps timing around generation.

Two FPS numbers are reported:
  * overall_fps     = total_output_frames / end-to-end wall time of one pipe()
                      call (includes text/vision encode + VAE decode).
  * steadystate_fps = frames_per_chunk / median(per-chunk time, excluding the
                      first chunk). This is the streaming throughput and the
                      number to compare against the model's 24 FPS target.

Per-chunk timing is obtained with a low-touch hook: the AR rollout calls
`pipe.progress_bar(...)` exactly once per chunk, so we wrap it to record a
CUDA-synchronized timestamp at each chunk boundary.

Run under torchrun; sequence-parallel degree = --nproc_per_node:

  torchrun --nproc_per_node=8 benchmark/bench_fps.py \
      --worldplay-dir ~/HY-WorldPlay \
      --model_type ar --video_length 125 --num_inference_steps 50 \
      --repeats 3 --out results/ar_baseline_sp8.json
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from argparse import Namespace

# Must be set before torch is imported (mirrors hyvideo/generate.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402


def rank0(*a, **k):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, **k, flush=True)


def git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def build_args(cli) -> Namespace:
    """Assemble the full args Namespace HY-WorldPlay's pipeline expects.

    Phase 2 quantization flags are env-driven (default OFF so the BF16 baseline is
    byte-for-byte unchanged). Set by scripts/arbor_worldplay_fp8sage*.sbatch:
      WORLDPLAY_USE_SAGEATTN, WORLDPLAY_USE_FP8_GEMM,
      WORLDPLAY_QUANT_TYPE, WORLDPLAY_INCLUDE_PATTERNS, WORLDPLAY_SAGE_BLOCKS_RANGE
    The sageattention/angelslim imports these trigger resolve to the ROCm shims on
    PYTHONPATH (Arbor/phase2/shims).
    """
    use_sageattn = _env_bool("WORLDPLAY_USE_SAGEATTN", False)
    use_fp8_gemm = _env_bool("WORLDPLAY_USE_FP8_GEMM", False)
    quant_type = os.environ.get("WORLDPLAY_QUANT_TYPE", "fp8-per-block")
    include_patterns = os.environ.get("WORLDPLAY_INCLUDE_PATTERNS", "double_blocks")
    sage_blocks_range = os.environ.get("WORLDPLAY_SAGE_BLOCKS_RANGE", "0-53")
    if use_sageattn or use_fp8_gemm:
        print(f"[bench] Phase2 quant: sageattn={use_sageattn} fp8_gemm={use_fp8_gemm} "
              f"quant_type={quant_type} include={include_patterns} "
              f"sage_blocks={sage_blocks_range}", flush=True)
    return Namespace(
        prompt=cli.prompt,
        negative_prompt=cli.negative_prompt,
        image_path=cli.image_path,
        pose=cli.pose,
        resolution=cli.resolution,
        aspect_ratio=cli.aspect_ratio,
        model_path=cli.model_path,
        action_ckpt=cli.action_ckpt,
        num_inference_steps=cli.num_inference_steps,
        video_length=cli.video_length,
        seed=cli.seed,
        dtype=cli.dtype,
        width=cli.width,
        height=cli.height,
        model_type=cli.model_type,
        few_step=cli.few_step,
        # Everything below is disabled for the no-quantization baseline.
        sr=False,
        save_pre_sr_video=False,
        rewrite=False,
        offloading=cli.offloading,
        group_offloading=cli.group_offloading,
        enable_torch_compile=cli.enable_torch_compile,
        with_ui=False,
        use_sageattn=use_sageattn,
        sage_blocks_range=sage_blocks_range,
        use_vae_parallel=False,
        use_fp8_gemm=use_fp8_gemm,
        quant_type=quant_type,
        include_patterns=include_patterns,
        transformer_resident_ar_rollout=cli.transformer_resident_ar_rollout,
        output_path=None,
    )


def n_frames(videos) -> int:
    """Number of frames in the pipeline output tensor."""
    v = videos
    if hasattr(v, "shape"):
        s = tuple(v.shape)
        if len(s) == 5:      # [B, C, F, H, W]
            return int(s[2])
        if len(s) == 4:      # [C, F, H, W]
            return int(s[1])
    raise ValueError(f"Unexpected video tensor shape: {getattr(v, 'shape', None)}")


class ChunkTimer:
    """Wrap pipe.progress_bar to record a synced timestamp per chunk boundary."""

    def __init__(self, pipe):
        self.pipe = pipe
        self._orig = pipe.progress_bar
        self.stamps = []

    def __enter__(self):
        def wrapped(*a, **k):
            torch.cuda.synchronize()
            self.stamps.append(time.perf_counter())
            return self._orig(*a, **k)

        self.pipe.progress_bar = wrapped
        self.stamps = []
        return self

    def __exit__(self, *exc):
        self.pipe.progress_bar = self._orig
        return False

    def per_chunk_times(self):
        # Interval between consecutive chunk starts = marginal time for a chunk.
        return [self.stamps[i + 1] - self.stamps[i] for i in range(len(self.stamps) - 1)]


def _extract_frames_uint8(videos, k):
    """Sample ``k`` evenly-spaced frames from a pipeline video tensor.

    Returns ``(frames_uint8[K,H,W,3], indices)`` normalized to [0,255]. Handles
    both [B,C,F,H,W] and [C,F,H,W] layouts and [-1,1] or [0,1] value ranges.
    """
    import numpy as np

    v = videos
    if hasattr(v, "detach"):
        v = v.detach().to("cpu", dtype=torch.float32)
    else:
        v = torch.as_tensor(v, dtype=torch.float32)
    s = tuple(v.shape)
    if len(s) == 5:            # [B, C, F, H, W] -> drop batch
        v = v[0]
        s = tuple(v.shape)
    if len(s) != 4:
        raise ValueError(f"unexpected video shape {s}")
    v = v.permute(1, 2, 3, 0)  # [C,F,H,W] -> [F,H,W,C]
    arr = v.numpy()
    if float(arr.min()) < -0.01:  # [-1,1] -> [0,1]
        arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    f = arr.shape[0]
    k = max(1, min(int(k), f))
    idx = np.unique(np.linspace(0, f - 1, k).round().astype(int)).tolist()
    frames = (arr[idx] * 255.0).round().astype("uint8")
    return frames, idx


def _save_montage_png(path, frames):
    """Best-effort human-inspectable strip of the first few reference frames."""
    try:
        import numpy as np
        from skimage.io import imsave

        k = min(4, frames.shape[0])
        strip = np.concatenate([frames[i] for i in range(k)], axis=1)
        imsave(path, strip)
    except Exception:  # noqa: BLE001 — montage is cosmetic; never fail the run
        pass


def _ssim_mse(ref_f, cur_f):
    """Mean per-frame SSIM (skimage) and mean pixel MSE on [0,1]."""
    import numpy as np
    from skimage.metrics import structural_similarity as ssim

    scores = [
        ssim(ref_f[i], cur_f[i], data_range=255, channel_axis=2)
        for i in range(ref_f.shape[0])
    ]
    a = ref_f.astype(np.float64) / 255.0
    b = cur_f.astype(np.float64) / 255.0
    return float(np.mean(scores)), float(np.mean((a - b) ** 2))


def _lpips_best_effort(ref_f, cur_f):
    """Mean LPIPS distance (AlexNet) on CPU; None if lpips/weights unavailable."""
    try:
        import lpips

        net = lpips.LPIPS(net="alex", verbose=False)

        def _to_t(x):
            t = torch.from_numpy(x.astype("float32") / 255.0)  # [K,H,W,C]
            return t.permute(0, 3, 1, 2) * 2.0 - 1.0            # [-1,1], [K,3,H,W]

        with torch.no_grad():
            d = net(_to_t(ref_f), _to_t(cur_f))
        return float(d.mean().item())
    except Exception:  # noqa: BLE001 — LPIPS is optional (offline weights, etc.)
        return None


class _LatentPerturb:
    """Add a tiny, deterministic, rank-consistent perturbation to the initial
    latent, for band calibration.

    Wraps ``pipe.prepare_latents`` (the single point where the initial noise is
    drawn, worldplay_video_pipeline.py:500) and multiplies the returned latents
    by ``(1 + eps * noise)``. The perturbation uses a fixed seed so EVERY
    sequence-parallel rank applies the identical delta (the base latents are
    already identical across SP ranks for a given seed) — no cross-rank
    divergence, no collective mismatch. This emulates the trajectory nudge a
    numerically-faithful (non-bit-identical) kernel introduces.
    """

    def __init__(self, pipe, eps, seed):
        self.pipe = pipe
        self.eps = float(eps)
        self.seed = int(seed)
        self._orig = pipe.prepare_latents

    def __enter__(self):
        orig, eps, seed = self._orig, self.eps, self.seed

        def wrapped(*a, **k):
            lat = orig(*a, **k)
            g = torch.Generator(device=lat.device).manual_seed(seed)
            noise = torch.randn(lat.shape, generator=g, device=lat.device, dtype=lat.dtype)
            return lat * (1.0 + eps * noise)

        self.pipe.prepare_latents = wrapped
        return self

    def __exit__(self, *exc):
        self.pipe.prepare_latents = self._orig
        return False


def _calibrate_band(cli, ref_out, run_once, barrier, pipe):
    """Measure the pipeline's self-calibrating equivalence band.

    ALL ranks must run the perturbed generations (the pipeline issues
    sequence-parallel collectives that must stay in lockstep); only rank 0
    extracts frames and computes drift. Returns the band dict on rank 0, else
    ``None``.

    The band is the envelope of SSIM/MSE/LPIPS drift between the unperturbed
    reference (``ref_out``) and ``samples`` generations each under an
    ``eps``-scale innocuous latent perturbation, widened by ``margin`` in
    drift-space. This is what a provably-innocuous numerical change does to the
    end-to-end video — the accept band for numerically-faithful kernels.
    """
    rank = int(os.environ.get("RANK", "0"))
    eps = cli.quality_calib_eps
    samples = max(1, int(cli.quality_calib_samples))
    margin = cli.quality_calib_margin

    ref_frames = None
    if rank == 0:
        try:
            ref_frames, _ = _extract_frames_uint8(ref_out.videos, cli.quality_frames)
        except Exception as exc:  # noqa: BLE001
            rank0(f"[bench][warn] calib: reference frame extract failed: {exc!r}")
            ref_frames = None

    drifts = []
    for s in range(samples):
        barrier()
        torch.cuda.synchronize()
        try:
            with _LatentPerturb(pipe, eps, seed=9000 + s):
                out_p = run_once()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 — a calib hiccup must not sink the run
            rank0(f"[bench][warn] calib sample {s + 1} failed: {type(exc).__name__}: {exc}")
            out_p = None
        barrier()
        if rank == 0 and out_p is not None and ref_frames is not None:
            try:
                cur, _ = _extract_frames_uint8(out_p.videos, cli.quality_frames)
                n = int(min(ref_frames.shape[0], cur.shape[0]))
                ssim_v, mse_v = _ssim_mse(ref_frames[:n], cur[:n])
                lpips_v = _lpips_best_effort(ref_frames[:n], cur[:n])
                drifts.append((ssim_v, mse_v, lpips_v))
                rank0(f"[bench] calib sample {s + 1}/{samples} eps={eps}: "
                      f"ssim={ssim_v:.4f} mse={mse_v:.6f} lpips={lpips_v}")
            except Exception as exc:  # noqa: BLE001
                rank0(f"[bench][warn] calib measure {s + 1} failed: {exc!r}")

    if rank != 0 or not drifts:
        return None
    min_ssim = min(d[0] for d in drifts)
    max_mse = max(d[1] for d in drifts)
    lp = [d[2] for d in drifts if d[2] is not None]
    max_lpips = max(lp) if lp else None
    band = {
        "ssim_min": round(1.0 - (1.0 - min_ssim) * margin, 6),
        "mse_max": round(max_mse * margin, 8),
        "eps": eps, "samples": len(drifts), "margin": margin,
        "raw_ssim_min": round(min_ssim, 6), "raw_mse_max": round(max_mse, 8),
    }
    if max_lpips is not None:
        band["lpips_max"] = round(max_lpips * margin, 6)
        band["raw_lpips_max"] = round(max_lpips, 6)
    rank0(f"[bench] calibrated equivalence band: {json.dumps(band)}")
    return band


def _compute_quality_gate(videos, cli, band=None):
    """Establish or compare a BF16 reference clip -> Hyperloom quality_gate dict.

    Reference frames are stored as ``<ref>.npz`` next to the path Hyperloom
    hands us (``baseline.png``); WRITE and COMPARE apply the same transform so
    they always agree. Emits the ``quality_gate`` contract that
    ``_accuracy_gate.parse_quality_gate`` maps onto accuracy (passed -> 1.0).
    """
    import numpy as np

    ref = (cli.quality_ref or "").strip()
    ref_write = (cli.quality_ref_write or "").strip()
    if not ref and not ref_write:
        return {"skipped": True, "reason": "no_reference_or_image"}
    try:
        frames, idx = _extract_frames_uint8(videos, cli.quality_frames)
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"frame_extract_failed:{type(exc).__name__}"}

    if ref_write:
        npz = ref_write + ".npz"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(npz)), exist_ok=True)
            save_kw = dict(frames=frames, indices=np.array(idx))
            if band is not None:
                # Persist the self-measured band alongside the reference so the
                # compare leg (a separate process) can read it.
                save_kw["band_json"] = json.dumps(band)
            np.savez_compressed(npz, **save_kw)
            _save_montage_png(ref_write, frames)
        except Exception as exc:  # noqa: BLE001
            return {"skipped": True, "reason": f"reference_write_failed:{type(exc).__name__}"}
        # ``passed: True`` keeps Magpie's own result layer from flagging the
        # baseline as failed; Hyperloom's _accuracy_gate independently treats
        # ``reason == reference_established`` as a pass regardless.
        out = {"passed": True, "skipped": True, "reason": "reference_established",
               "reference": npz, "n_frames": int(frames.shape[0])}
        if band is not None:
            out["band"] = band
            out["calibrated"] = True
        return out

    npz = ref + ".npz"
    if not os.path.exists(npz):
        return {"skipped": True, "reason": "reference_missing", "reference": npz}
    try:
        z = np.load(npz)
        ref_frames = z["frames"]
        cal_band = json.loads(str(z["band_json"])) if "band_json" in z.files else None
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"reference_unreadable:{type(exc).__name__}"}
    n = int(min(ref_frames.shape[0], frames.shape[0]))
    if n == 0 or ref_frames.shape[1:] != frames.shape[1:]:
        return {"skipped": True, "reason": "reference_shape_mismatch"}
    ref_f, cur_f = ref_frames[:n], frames[:n]

    ssim_v, mse_v = _ssim_mse(ref_f, cur_f)
    lpips_v = _lpips_best_effort(ref_f, cur_f)
    # Bit-identical output is the hard (pixel_exact) accept — trusted like
    # GEAK's byte_exact greedy parity, regardless of any band.
    pixel_exact = (
        ssim_v >= 0.999995 and mse_v <= 1e-9
        and (lpips_v is None or lpips_v <= 1e-6)
    )
    gate = {
        "skipped": False,
        "ssim": round(ssim_v, 6),
        "mse": round(mse_v, 8),
        "n_frames": n,
    }
    if lpips_v is not None:
        gate["lpips"] = round(lpips_v, 6)

    if cal_band:
        # Self-calibrating (soft) gate: accept when the drift is within the
        # band a provably-innocuous numerical change produces.
        gate["calibrated"] = True
        gate["band"] = cal_band
        gate["ssim_min"] = float(cal_band.get("ssim_min", cli.quality_ssim_min))
        gate["mse_max"] = float(cal_band.get("mse_max", cli.quality_mse_max))
        ok = (ssim_v >= gate["ssim_min"]) and (mse_v <= gate["mse_max"])
        if lpips_v is not None and "lpips_max" in cal_band:
            gate["lpips_max"] = float(cal_band["lpips_max"])
            ok = ok and (lpips_v <= gate["lpips_max"])
        gate["parity_kind"] = "pixel_exact" if pixel_exact else "perceptual"
        gate["passed"] = bool(pixel_exact or ok)
    else:
        # Fallback: fixed thresholds (backward compatible with the old gate).
        gate["calibrated"] = False
        gate["ssim_min"] = float(cli.quality_ssim_min)
        gate["mse_max"] = float(cli.quality_mse_max)
        ok = (ssim_v >= cli.quality_ssim_min) and (mse_v <= cli.quality_mse_max)
        if lpips_v is not None:
            gate["lpips_max"] = float(cli.quality_lpips_max)
            ok = ok and (lpips_v <= cli.quality_lpips_max)
        gate["parity_kind"] = "pixel_exact" if pixel_exact else "threshold"
        gate["passed"] = bool(ok)
    return gate


def main():
    p = argparse.ArgumentParser(description="HY-WorldPlay generated-FPS benchmark")
    p.add_argument("--worldplay-dir", default=os.path.expanduser("~/HY-WorldPlay"))
    # Model paths: if omitted, resolve from the local HuggingFace cache.
    p.add_argument("--model_path", default=None)
    p.add_argument("--action_ckpt", default=None)
    # Workload
    p.add_argument("--model_type", choices=["ar", "bi"], default="ar")
    p.add_argument("--video_length", type=int, default=125)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--pose", default="w-31")
    p.add_argument("--prompt", default=(
        "A paved pathway leads towards a stone arch bridge spanning a calm body of "
        "water. Lush green trees line the path. Soft natural light, tranquil atmosphere."
    ))
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--image_path", default=None,
                   help="I2V reference image (defaults to WorldPlay assets/img/test.png)")
    p.add_argument("--resolution", default="480p")
    p.add_argument("--aspect_ratio", default="16:9")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--few_step", action="store_true")
    # Perf knobs (defaults = documented AR baseline; both are legit first opts)
    p.add_argument("--offloading", type=int, default=1, help="1=on (shipped default), 0=off")
    p.add_argument("--group_offloading", type=int, default=None)
    p.add_argument("--transformer_resident_ar_rollout", type=int, default=0)
    p.add_argument("--enable_torch_compile", action="store_true")
    # Benchmark control
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=None, help="Path to write JSON result (rank 0)")
    p.add_argument("--tag", default="ar_baseline")
    # ── torch.profiler capture (roofline / kernel-agent) ──
    # When set, capture ONE generation under torch.profiler and export a
    # chrome trace (*.trace.json.gz) into this dir. Hyperloom's profile phase
    # probes <workspace>/torch_trace for it, then TraceLens attributes GPU time
    # across kernels to feed the roofline + kernel agent. Env fallback lets the
    # Magpie wrapper wire it from PROFILE=1.
    p.add_argument("--torch_profiler_dir",
                   default=(os.environ.get("WORLDPLAY_TORCH_PROFILER_DIR", "").strip() or None),
                   help="Capture a torch.profiler trace of one generation into this dir.")
    # A full 50-step x N-chunk generation traces to >1 GB (slow NFS write + slow
    # TraceLens parse). The roofline only needs representative kernel shapes, so
    # the profiled capture uses a reduced denoise-step count. 0 = use full steps.
    p.add_argument("--profile_steps", type=int,
                   default=int(os.environ.get("WORLDPLAY_PROFILE_STEPS", "12") or 12),
                   help="Denoise steps for the profiled capture (bounds trace size); 0=full.")
    # ── Video-quality gate (BF16 reference clip; scriptable correctness) ──
    # Hyperloom's scriptable plumbing injects XDIT_QUALITY_REF (compare) on
    # explore variants and XDIT_QUALITY_REF_WRITE (establish) on the baseline;
    # honor both here so a speedup can never "win" by degrading the video. Env
    # defaults let the gate also work when invoked outside Hyperloom.
    p.add_argument("--quality-ref", default=(os.environ.get("XDIT_QUALITY_REF", "").strip() or None),
                   help="Reference clip to COMPARE against (SSIM/MSE/LPIPS).")
    p.add_argument("--quality-ref-write", default=(os.environ.get("XDIT_QUALITY_REF_WRITE", "").strip() or None),
                   help="Path to WRITE the reference clip (baseline establishes it).")
    p.add_argument("--quality-frames", type=int,
                   default=int(os.environ.get("WORLDPLAY_QUALITY_FRAMES", "8") or 8),
                   help="Evenly-spaced frames sampled for the quality comparison.")
    p.add_argument("--quality-ssim-min", type=float,
                   default=float(os.environ.get("WORLDPLAY_QUALITY_SSIM_MIN", "0.85") or 0.85))
    p.add_argument("--quality-lpips-max", type=float,
                   default=float(os.environ.get("WORLDPLAY_QUALITY_LPIPS_MAX", "0.15") or 0.15))
    p.add_argument("--quality-mse-max", type=float,
                   default=float(os.environ.get("WORLDPLAY_QUALITY_MSE_MAX", "0.006") or 0.006))
    # ── Self-calibrating perceptual gate ──────────────────────────────────
    # A fixed SSIM/MSE threshold is the WRONG bar for a chaotic autoregressive
    # diffusion model: a numerically-faithful kernel (correct to its dtype
    # tolerance, but not bit-identical) nudges the initial trajectory and, over
    # many denoise steps + AR chunks, drifts to a visibly-similar-but-not-
    # pixel-identical video (baseline-vs-baseline is exactly SSIM 1.0, so the
    # drift is real, not noise). Instead of hardcoding a threshold, we MEASURE
    # the pipeline's own equivalence band at establish time: re-generate the
    # baseline under a tiny, rank-consistent numerical perturbation of the
    # initial latent (~ the size of a faithful kernel's error) and record how
    # far SSIM/MSE/LPIPS move. That measured drift (× a safety margin) BECOMES
    # the accept band. A candidate then passes when its output is within the
    # band a provably-innocuous change produces (parity_kind="perceptual"), or
    # is bit-identical (parity_kind="pixel_exact"). Falls back to the fixed
    # thresholds when calibration is off. Shared with the FP8/quant path (#5).
    p.add_argument("--quality-calibrate", type=int,
                   default=int(os.environ.get("WORLDPLAY_QUALITY_CALIBRATE", "0") or 0),
                   help="At establish: measure the self-calibrating equivalence band.")
    p.add_argument("--quality-calib-eps", type=float,
                   default=float(os.environ.get("WORLDPLAY_QUALITY_CALIB_EPS", "0.004") or 0.004),
                   help="Relative magnitude of the innocuous latent perturbation "
                        "(~bf16 tolerance) used to measure the band.")
    p.add_argument("--quality-calib-samples", type=int,
                   default=int(os.environ.get("WORLDPLAY_QUALITY_CALIB_SAMPLES", "2") or 2),
                   help="Number of perturbed generations to measure the band from.")
    p.add_argument("--quality-calib-margin", type=float,
                   default=float(os.environ.get("WORLDPLAY_QUALITY_CALIB_MARGIN", "1.25") or 1.25),
                   help="Safety factor widening the measured band (drift-space).")
    cli = p.parse_args()

    cli.offloading = bool(cli.offloading)
    cli.transformer_resident_ar_rollout = bool(cli.transformer_resident_ar_rollout)
    if cli.group_offloading is not None:
        cli.group_offloading = bool(cli.group_offloading)

    sys.path.insert(0, cli.worldplay_dir)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Mirror hyvideo/generate.py bootstrap.
    from hyvideo.commons.parallel_states import initialize_parallel_state
    from hyvideo.commons.infer_state import initialize_infer_state
    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
    from hyvideo.generate import pose_to_input

    initialize_parallel_state(sp=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Resolve model paths from the HF cache if not given.
    if cli.model_path is None or cli.action_ckpt is None:
        from huggingface_hub import snapshot_download
        hunyuan = snapshot_download("tencent/HunyuanVideo-1.5", local_files_only=True)
        wp = snapshot_download("tencent/HY-WorldPlay", local_files_only=True)
        sub = {"ar": "ar_model", "bi": "bidirectional_model"}[cli.model_type]
        cli.model_path = cli.model_path or hunyuan
        cli.action_ckpt = cli.action_ckpt or os.path.join(
            wp, sub, "diffusion_pytorch_model.safetensors")
    if cli.image_path is None:
        cli.image_path = os.path.join(cli.worldplay_dir, "assets/img/test.png")

    args = build_args(cli)
    assert ((args.video_length - 1) // 4 + 1) % 4 == 0, \
        "num latents must be divisible by 4"
    initialize_infer_state(args)

    transformer_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    transformer_version = f"{args.resolution}_i2v"

    rank0(f"[bench] building pipeline (sp={world_size}, model_type={args.model_type}, "
          f"offloading={args.offloading}, resident={args.transformer_resident_ar_rollout})")
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=transformer_dtype,
        action_ckpt=args.action_ckpt,
    )

    viewmats, Ks, action = pose_to_input(args.pose, (args.video_length - 1) // 4 + 1)

    def run_once():
        return pipe(
            enable_sr=False,
            prompt=args.prompt,
            aspect_ratio=args.aspect_ratio,
            num_inference_steps=args.num_inference_steps,
            sr_num_inference_steps=None,
            video_length=args.video_length,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            output_type="pt",
            prompt_rewrite=False,
            return_pre_sr_video=False,
            viewmats=viewmats.unsqueeze(0),
            Ks=Ks.unsqueeze(0),
            action=action.unsqueeze(0),
            few_step=args.few_step,
            chunk_latent_frames=4 if args.model_type == "ar" else 16,
            model_type=args.model_type,
            user_height=args.height,
            user_width=args.width,
            transformer_resident_ar_rollout=args.transformer_resident_ar_rollout,
            reference_image=args.image_path,
        )

    def barrier():
        if world_size > 1:
            torch.distributed.barrier()

    # ---- Warmup (pays JIT/autotune/cache costs; not timed) ----
    for w in range(cli.warmup):
        rank0(f"[bench] warmup {w + 1}/{cli.warmup} ...")
        run_once()
        torch.cuda.synchronize()
    barrier()

    # ---- Timed repeats ----
    runs = []
    for r in range(cli.repeats):
        torch.cuda.reset_peak_memory_stats()
        barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with ChunkTimer(pipe) as ct:
            out = run_once()
        torch.cuda.synchronize()
        e2e = time.perf_counter() - t0
        barrier()

        frames = n_frames(out.videos)
        chunk_times = ct.per_chunk_times()
        num_chunks = len(ct.stamps)
        frames_per_chunk = frames / num_chunks if num_chunks else float("nan")
        overall_fps = frames / e2e
        ss_times = chunk_times[1:] if len(chunk_times) > 1 else chunk_times
        steadystate_fps = (frames_per_chunk / statistics.median(ss_times)
                           if ss_times else float("nan"))
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        rank0(f"[bench] run {r + 1}/{cli.repeats}: frames={frames} e2e={e2e:.2f}s "
              f"overall={overall_fps:.3f} fps  steady={steadystate_fps:.3f} fps  "
              f"chunks={num_chunks} peakVRAM={peak_gb:.1f}GB")
        runs.append(dict(
            frames=frames, e2e_s=e2e, overall_fps=overall_fps,
            steadystate_fps=steadystate_fps, num_chunks=num_chunks,
            frames_per_chunk=frames_per_chunk, peak_vram_gb=peak_gb,
            per_chunk_times_s=chunk_times,
        ))
        # Persist after every repeat so a killed reservation never loses data.
        if int(os.environ.get("RANK", "0")) == 0:
            _finalize_and_write(cli, args, runs, world_size,
                                done=(r == cli.repeats - 1))

    # ---- Optional torch.profiler capture (roofline / kernel-agent) ----
    # Runs after the timed repeats so it never perturbs the fps numbers. All
    # ranks execute the generation (collectives stay in lockstep); only rank 0
    # records + exports the chrome trace the profile executor discovers.
    if cli.torch_profiler_dir:
        _run_profiled_capture(cli, args, run_once, barrier)

    # ---- Self-calibrating equivalence band (establish only) ----
    # Runs extra perturbed generations; ALL ranks must participate (collectives
    # stay in lockstep). Only meaningful when establishing the reference.
    band = None
    if cli.quality_ref_write and cli.quality_calibrate:
        band = _calibrate_band(cli, out, run_once, barrier, pipe)

    if int(os.environ.get("RANK", "0")) != 0:
        return

    quality_gate = None
    if cli.quality_ref or cli.quality_ref_write:
        try:
            quality_gate = _compute_quality_gate(out.videos, cli, band=band)
        except Exception as exc:  # noqa: BLE001 — a gate error must not lose fps data
            quality_gate = {"skipped": True, "reason": f"quality_error:{type(exc).__name__}"}
        rank0(f"[bench] quality_gate: {json.dumps(quality_gate)}")

    _finalize_and_write(cli, args, runs, world_size, done=True, quality_gate=quality_gate)


def _run_profiled_capture(cli, args, run_once, barrier):
    """Capture a torch.profiler trace of a single generation for TraceLens.

    All ranks execute ``run_once`` (the pipeline issues collective ops that must
    stay in lockstep across sequence-parallel ranks); only rank 0 wraps it in a
    profiler and exports a chrome trace. Hyperloom's ProfileExecutor discovers
    ``*.trace.json.gz`` under ``<workspace>/torch_trace`` and hands it to
    TraceLens, which attributes GPU time per kernel for the roofline + kernel
    agent. Best-effort: a profiler/export failure must never lose the fps data.

    The profiled generation uses ``cli.profile_steps`` denoise steps (when > 0)
    instead of the full count, keeping the trace small enough to write over NFS
    and parse quickly while still exercising every kernel type.
    """
    import contextlib

    rank = int(os.environ.get("RANK", "0"))
    prof_dir = cli.torch_profiler_dir
    if rank == 0:
        os.makedirs(prof_dir, exist_ok=True)

    from torch.profiler import ProfilerActivity, profile

    # TraceLens derives a kernel's editable source (entry_point / launcher_path)
    # from the per-kernel Python call stack (call_stack_full). That is only
    # populated when the profiler captures CPU-side stacks, so with_stack must be
    # ON for the kernel agent to resolve a repo-resident source instead of
    # "Not found". with_modules adds the nn.Module hierarchy used by the module
    # chain fallback. Overridable in case a torch build regresses stack capture.
    with_stack = os.environ.get("WORLDPLAY_PROFILER_WITH_STACK", "1") != "0"
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    prof_cm = (
        profile(activities=activities, record_shapes=True,
                with_stack=with_stack, with_modules=with_stack)
        if rank == 0
        else contextlib.nullcontext()
    )
    # Bound the profiled workload (all ranks must agree on step count).
    orig_steps = args.num_inference_steps
    if cli.profile_steps and cli.profile_steps > 0:
        args.num_inference_steps = cli.profile_steps
    rank0(f"[bench] capturing torch.profiler trace "
          f"(1 generation, {args.num_inference_steps} steps) ...")
    barrier()
    torch.cuda.synchronize()
    try:
        with prof_cm as prof:
            run_once()
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 — profiling must not sink the run
        rank0(f"[bench][warn] profiled generation failed: {type(exc).__name__}: {exc}")
        prof = None
    finally:
        args.num_inference_steps = orig_steps
    barrier()

    if rank == 0 and prof is not None:
        out_trace = os.path.join(prof_dir, "worldplay-TP-0.pt.trace.json.gz")
        try:
            prof.export_chrome_trace(out_trace)
            rank0(f"[bench] wrote torch profiler trace -> {out_trace}")
        except Exception as exc:  # noqa: BLE001
            rank0(f"[bench][warn] export_chrome_trace failed: {type(exc).__name__}: {exc}")


def _agg(runs, key):
    vals = [r[key] for r in runs if r[key] == r[key]]  # drop NaN
    if not vals:
        return None
    return dict(mean=statistics.mean(vals),
                std=(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
                min=min(vals), max=max(vals))


def _sanitize(obj):
    """Replace NaN/inf floats with None so the JSON is spec-valid."""
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _finalize_and_write(cli, args, runs, world_size, done, quality_gate=None):
    """Assemble the result dict and (over)write the JSON. Safe to call after
    every repeat so partial progress is always persisted (deadline safety)."""
    result = dict(
        tag=cli.tag,
        model="HY-WorldPlay (HunyuanWorld-1.5)",
        config=dict(
            model_type=args.model_type, video_length=args.video_length,
            num_inference_steps=args.num_inference_steps, pose=args.pose,
            resolution=args.resolution, width=args.width, height=args.height,
            dtype=args.dtype, sp_size=world_size, few_step=args.few_step,
            offloading=args.offloading,
            transformer_resident_ar_rollout=args.transformer_resident_ar_rollout,
            attn_backend="torch_sdpa", quantization="none",
        ),
        env=dict(
            gpu=torch.cuda.get_device_name(0),
            gcn_arch=torch.cuda.get_device_properties(0).gcnArchName,
            torch=torch.__version__, hip=torch.version.hip,
            python=platform.python_version(),
            worldplay_sha=git_sha(cli.worldplay_dir),
        ),
        summary=dict(
            overall_fps=_agg(runs, "overall_fps"),
            steadystate_fps=_agg(runs, "steadystate_fps"),
            e2e_s=_agg(runs, "e2e_s"),
            peak_vram_gb=_agg(runs, "peak_vram_gb"),
        ),
        completed_repeats=len(runs),
        planned_repeats=cli.repeats,
        done=done,
        runs=runs,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    if quality_gate is not None:
        result["quality_gate"] = quality_gate
    result = _sanitize(result)

    if done:
        print("\n===== HY-WorldPlay FPS baseline =====", flush=True)
        print(json.dumps(result["summary"], indent=2), flush=True)

    if cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
        tmp = cli.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        os.replace(tmp, cli.out)  # atomic
        state = "final" if done else f"partial {len(runs)}/{cli.repeats}"
        print(f"[bench] wrote {cli.out} ({state})", flush=True)


if __name__ == "__main__":
    main()
