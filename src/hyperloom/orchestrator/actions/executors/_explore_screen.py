"""Drop EXPLORE's hopeless variants on a cheap probe before benchmarking them.

EXPLORE benchmarks every proposed variant on the deployment configuration, which
for a TP=8 target means eight GPUs held for the whole run. Most of those runs buy
nothing: the variant loses, and the only thing the benchmark had to establish was
that it loses.

A screen answers that question far more cheaply by running the SAME engine with
the SAME flags on fewer GPUs. It is a real engine launch, not a projection, so
kernel-level levers -- attention backend, aiter, cudagraph -- actually execute
and actually show up; an analytical model reports 0.0% for every one of them.
What it needs to transfer is the ORDER, not the latency.

On the one grid this has been measured against, it does not. Kernel dispatch
(below) is part of the reason but not all of it: a reduced-parallelism server
resolves exactly the deployment's kernels and the order still does not hold.
What is left is that fewer GPUs is a different machine, and the levers do not
keep their order across it. Hence the default below is off, and turning it on
wants evidence from the deployment in front of you.

The catch, and the reason for most of the code below: the probe does not run the
deployment's kernels just because it was handed the deployment's flags. vLLM
resolves an attention and a MoE implementation per engine, and the offline engine
the probe builds resolves them differently from the server EXPLORE benchmarks.
Measured on gpt-oss-120b/MI355X at the SAME TP=8, same model, same flags: the
server settles on ROCM_AITER_UNIFIED_ATTN with a Triton MXFP4 MoE, the probe on
ROCM_AITER_FA with an AITER one. Across that gap the screen is not noisy, it is
confidently wrong: it ranks a stack the deployment never runs.

So the screen does not assume its regime, it checks it: both sides say in their
own logs which kernels they resolved, the deployment's from a benchmark the
session has already paid for, and nothing is pruned unless they agree. The target
backend is pinned onto the probe to give that check a chance of passing; it is
pinned first, so a variant naming the backend itself still overrides it.

A matching regime is necessary and not sufficient, which is why this is off by
default. Forcing the probe all the way onto the server's kernels lines one lever
up and lines the next one up by deleting it: with the MoE implementation pinned,
VLLM_ROCM_USE_AITER_MOE=0 becomes a no-op in the probe and reads flat, where on
the server it is not. Kernel dispatch is therefore verified and not forced past
the attention backend, and a screen that cannot reach the deployment's regime
returns the grid untouched rather than a ranking of a different stack.

The screen only prunes, never promotes, and a variant whose probe fails is kept:
the cost of a wasted benchmark is minutes, the cost of silently discarding the
round's winner is the session.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ._grid_base import GridVariant

log = logging.getLogger(__name__)

ENV_ENABLED = "HYPERLOOM_EXPLORE_SCREEN"
ENV_MARGIN_PCT = "HYPERLOOM_EXPLORE_SCREEN_MARGIN_PCT"
ENV_GPUS = "HYPERLOOM_EXPLORE_SCREEN_GPUS"
ENV_MODEL = "HYPERLOOM_EXPLORE_SCREEN_MODEL"
ENV_LAYERS = "HYPERLOOM_EXPLORE_SCREEN_LAYERS"
ENV_BACKEND = "HYPERLOOM_EXPLORE_SCREEN_ATTENTION_BACKEND"
ENV_TIMEOUT = "HYPERLOOM_EXPLORE_SCREEN_TIMEOUT_SEC"
ENV_INFERA_ROOT = "HYPERLOOM_INFERSIM_ROOT"

BENCH_REL = "infera/projection/core/projection/inference_projection/benchmark_vllm.py"
# A decode step is differenced between a K and a K/2 run, so a short K leaves
# call-to-call jitter in the estimate. Measured on MI355X: K=256 repeats to 18%
# sd on an identical config, K=1024 to 3.5%.
DECODE_STEPS = 1024
SEEDS = "0,1,2"
# The screen may only act on gaps far wider than its own noise. Measured on eight
# independent replicas, a gap this wide is one every replica reproduces and
# narrower ones are not, which makes this a floor on what may be pruned rather
# than a width at which pruning becomes safe.
DEFAULT_MARGIN_PCT = 10.0
DEFAULT_TIMEOUT_SEC = 1800
# The reading the survivors are judged against: the stack as it stands, probed
# the same way on the same device, so device and drift are common to both sides.
BASELINE = "__screen_baseline__"
BACKEND_FLAG = "--attention-backend"
# vLLM names every kernel family it settled on as it builds the engine, whether
# it was told which to use or fell back to one. These are the lines that say
# which stack a run actually measured.
_KERNEL_RES = {
    "attention": re.compile(r"Using (\w+) backend|Overriding with (\w+)"),
    "moe": re.compile(r"Using '([\w]+)' Mxfp4 MoE backend"),
}


def screen_enabled() -> bool:
    """Off unless explicitly turned on."""
    return str(os.environ.get(ENV_ENABLED, "0")).strip().lower() in ("1", "true", "yes")


def kernels_from_log(text: str) -> dict[str, str]:
    """The kernel families a vLLM log says the run actually resolved to.

    This is read from both sides -- the deployment's server log and the probe's
    own output -- because it is the only honest way to know they are the same
    stack. They are not the same by construction: the offline engine the probe
    builds and the server EXPLORE benchmarks select different kernels from the
    identical model and flags (measured on gpt-oss-120b/MI355X at TP8: the probe
    resolves ROCM_AITER_FA with an AITER MXFP4 MoE, the server
    ROCM_AITER_UNIFIED_ATTN with a Triton one). Ranked across that gap the screen
    is measuring a stack the deployment never runs.
    """
    found = {}
    for family, pattern in _KERNEL_RES.items():
        match = pattern.search(text or "")
        if match:
            found[family] = next(g for g in match.groups() if g)
    return found


def _target_kernels(session_dir: Path | None) -> dict[str, str]:
    """The kernels the DEPLOYMENT runs, read from a benchmark already paid for.

    EXPLORE has booted the stack at full parallelism before it proposes anything,
    so the answer is already on disk and costs nothing to look up.
    """
    if session_dir is None:
        return {}
    logs = sorted(Path(session_dir).rglob("server.log"),
                  key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for path in logs[:8]:
        try:
            kernels = kernels_from_log(path.read_text(errors="ignore"))
        except OSError:
            continue
        if kernels:
            log.info("explore screen: target kernels %s (from %s)", kernels, path)
            return kernels
    return {}


def _target_backend(bench: dict[str, Any], session_dir: Path | None) -> str | None:
    """The attention backend to pin onto the probe, or None to leave it alone."""
    override = os.environ.get(ENV_BACKEND, "").strip()
    if override:
        return override

    args = str((bench.get("envs") or {}).get("EXTRA_VLLM_ARGS") or "")
    tokens = args.split()
    for i, tok in enumerate(tokens):
        if tok == BACKEND_FLAG and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(BACKEND_FLAG + "="):
            return tok.split("=", 1)[1]

    return _target_kernels(session_dir).get("attention")


def _probe_command(variant: GridVariant, bench: dict[str, Any], out_path: str,
                   backend: str | None) -> list[str]:
    """The ``benchmark_vllm`` invocation that screens one variant."""
    envs = bench.get("envs") or {}
    root = os.environ.get(ENV_INFERA_ROOT, "")
    model = os.environ.get(ENV_MODEL) or str(bench.get("model") or envs.get("MODEL", ""))
    layers = os.environ.get(ENV_LAYERS, "").strip()

    cmd = [
        "python", str(Path(root) / BENCH_REL),
        "--model", model,
        "--tp", str(_as_int(envs.get("TP"), 1)),
        "--benchmark-gpus", str(_probe_gpus(envs)),
        "--batches", str(_as_int(envs.get("CONC"), 32)),
        "--input-len", str(_as_int(envs.get("ISL"), 1024)),
        "--decode-steps", str(DECODE_STEPS),
        "--seeds", SEEDS,
        "--load-format", "auto", "--routing-dist", "none",
        # The screen is a ranking probe, not an anchor, and it depends on things
        # only the offline entrypoint offers: truncated layers, a fixed decode
        # step count and a seed sweep. Anchors take the serving default instead.
        "--offline",
        "--save", out_path,
    ]
    if layers:
        cmd += ["--num-hidden-layers", layers]
    # The pin goes first so a variant that names the backend itself still wins:
    # it holds the rest of the stack at the target's regime, it does not override
    # the lever under test.
    pin = [BACKEND_FLAG, backend] if backend else []
    server_args = " ".join([*pin, variant.extra_server_args or ""]).strip()
    if server_args:
        cmd += ["--server-args=" + server_args]
    for key, value in (variant.extra_envs or {}).items():
        cmd += ["--env", f"{key}={value}"]
    return cmd


DEFAULT_PROBE_GPUS = 1


def _probe_gpus(envs: dict[str, Any]) -> int:
    """How many GPUs the probe runs on, never more than the target's TP.

    One by default: this is where nearly all the saving is, and the regime guard
    above is what makes taking it safe.
    """
    target = _as_int(envs.get("TP"), 1)
    return min(target, _as_int(os.environ.get(ENV_GPUS), DEFAULT_PROBE_GPUS))


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _probe(variant: GridVariant, bench: dict[str, Any], timeout_sec: int,
           backend: str | None) -> tuple[float | None, dict[str, str]]:
    """One variant's decode step latency (ms) and the kernels that produced it.

    A reading of None means the probe could not answer, which is always resolved
    in the variant's favour by the caller.
    """
    with tempfile.TemporaryDirectory(prefix="explore-screen-") as tmp:
        out_path = os.path.join(tmp, "probe.json")
        cmd = _probe_command(variant, bench, out_path, backend)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("explore screen: probe for %r did not run (%s)", variant.name, exc)
            return None, {}
        kernels = kernels_from_log((proc.stdout or "") + (proc.stderr or ""))
        if proc.returncode != 0 or not os.path.exists(out_path):
            # vLLM raises rather than falling back when a pinned backend is not
            # valid for the probe's shape, so this is also how a screen that
            # could not hold the target's regime fails.
            log.warning("explore screen: probe for %r failed rc=%s: %s",
                        variant.name, proc.returncode, (proc.stderr or "")[-400:])
            return None, kernels
        try:
            sweep = json.load(open(out_path))["sweep"]
            return float(sweep[0]["decode_ms"]), kernels
        except (OSError, ValueError, KeyError, IndexError) as exc:
            log.warning("explore screen: probe for %r produced no reading (%s)",
                        variant.name, exc)
            return None, kernels


def screen_variants(
    variants: list[GridVariant],
    config_path: Path,
    *,
    session_dir: Path | None = None,
    margin_pct: float | None = None,
) -> tuple[list[GridVariant], list[dict[str, str]]]:
    """Return the variants still worth benchmarking, plus a record of what was cut.

    A variant is cut only when the screen puts it more than ``margin_pct`` behind
    the screened baseline -- a gap far wider than the screen's own noise. The
    screen is not asked to pick a winner: the differences EXPLORE keeps on are a
    few percent, which is inside its error. It is asked to recognise the variants
    that are decisively worse.

    Nothing is cut unless the probe ran in the target's regime, and anything the
    probe cannot read is kept, so a broken or mismatched screen degrades to
    today's behaviour rather than to a silently smaller grid.
    """
    if not screen_enabled() or len(variants) < 3:
        return list(variants), []
    if not os.environ.get(ENV_INFERA_ROOT):
        log.warning("explore screen: %s is not set; skipping the screen", ENV_INFERA_ROOT)
        return list(variants), []

    try:
        with open(config_path) as fh:
            bench = (yaml.safe_load(fh) or {}).get("benchmark") or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("explore screen: could not read %s (%s); skipping", config_path, exc)
        return list(variants), []

    target = _target_kernels(session_dir)
    if not target:
        log.warning("explore screen: no benchmark in this session says which kernels "
                    "the deployment runs, so a probe cannot be checked against it; "
                    "skipping the screen")
        return list(variants), []

    timeout_sec = _as_int(os.environ.get(ENV_TIMEOUT), DEFAULT_TIMEOUT_SEC)
    backend = _target_backend(bench, session_dir)
    baseline, probed = _probe(GridVariant(name=BASELINE), bench, timeout_sec, backend)
    if baseline is None:
        log.warning("explore screen: baseline probe failed; benchmarking the full grid")
        return list(variants), []

    mismatch = {k: (v, probed.get(k)) for k, v in target.items() if probed.get(k) != v}
    if mismatch:
        log.warning(
            "explore screen: the probe is not running the deployment's kernels "
            "(%s), so its ordering is about a different stack; skipping the screen",
            ", ".join(f"{k}: target {t}, probe {p}" for k, (t, p) in mismatch.items()))
        return list(variants), []

    margin = margin_pct if margin_pct is not None else _margin_pct()
    cut_at = baseline * (1.0 + margin / 100.0)

    survivors, dropped = [], []
    for variant in variants:
        reading, _ = _probe(variant, bench, timeout_sec, backend)
        if reading is not None and reading > cut_at:
            dropped.append({
                "name": variant.name,
                "reason": "screen_decisively_slower",
                "detail": f"probe decode {reading:.3f} ms vs baseline "
                          f"{baseline:.3f} ms (+{(reading / baseline - 1) * 100:.0f}%)",
            })
        else:
            survivors.append(variant)

    log.info("explore screen: %d/%d variants forwarded to benchmark; cut %s",
             len(survivors), len(variants),
             ", ".join(d["name"] for d in dropped) or "nothing")
    return survivors, dropped


def _margin_pct() -> float:
    try:
        margin = float(os.environ.get(ENV_MARGIN_PCT, DEFAULT_MARGIN_PCT))
    except (TypeError, ValueError):
        return DEFAULT_MARGIN_PCT
    return margin if margin > 0 else DEFAULT_MARGIN_PCT
