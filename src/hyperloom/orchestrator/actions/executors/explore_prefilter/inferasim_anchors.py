# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The InferaSim anchor store: which measured anchor calibrates a projection.

Benchmark mode calibrates a projection against the nearest in-regime anchor in
``HYPERLOOM_INFERASIM_ANCHOR_STORE`` (or the one ``HYPERLOOM_INFERASIM_ANCHOR``
pins). An anchor is only used for its own model, engine and GPU, when its
decode curve is physically possible, and when the image, versions and launch
levers it recorded do not contradict the candidate's. On a miss, explicit
benchmark mode harvests one with Infera's served-anchor benchmark.

:func:`record_measured_anchor` turns an EXPLORE decision round into a
single-point served anchor, so the measurements a session already paid for
calibrate the next projection.

:func:`project` is the entry point: it picks the anchor for the mode and runs
the projection through :mod:`inferasim_bridge`.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from . import inferasim_bridge as bridge
from ._reuse import LAUNCH_UNPINNED
from .inferasim_bridge import (
    ATTENTION_BACKEND_FLAG,
    AnchorChoice,
    ENV_ANCHOR,
    ENV_ANCHOR_STORE,
    ENV_HARVEST,
    ENV_HARVEST_GPUS,
    ENV_HARVEST_PYTHON,
    ENV_HARVEST_TIMEOUT,
    ENV_ROOT,
    ESTIMATOR_DES,
    MODE_AUTO,
    MODE_BENCHMARK,
    MODE_SIMULATE,
    InferasimBridgeError,
    ProjMetrics,
    ServingSpec,
    _as_int,
    _parse_server_arg_str,
    estimator,
    gpu_arch,
    parse_speculative,
    projection_mode,
    recipe_from_spec,
    resolve_preset,
)

log = logging.getLogger(__name__)


def select_anchor(spec: ServingSpec) -> AnchorChoice | None:
    """Pick the closest in-regime warmup anchor for ``spec``.

    Precedence: an explicit ``HYPERLOOM_INFERASIM_ANCHOR`` always wins; otherwise
    an anchor store directory is searched for the nearest anchor in regime space.
    Returns ``None`` when neither is configured (pure-analytical projection).
    """
    explicit = os.environ.get(ENV_ANCHOR)
    if explicit and Path(explicit).is_file():
        # Pinned by an operator, but still gated: a corrupt curve would silently
        # propagate into every projection made from it.
        if not anchor_curve_is_sane(explicit):
            return None
        return AnchorChoice(path=explicit, regime_distance=0, model=spec.model_path)

    store_root = os.environ.get(ENV_ANCHOR_STORE)
    if not store_root or not Path(store_root).is_dir():
        return None

    bridge._ensure_infera_importable()
    try:
        from infera.projection.core.projection.inference_projection.search.anchor_store import (
            AnchorStore,
        )
    except Exception as exc:
        raise InferasimBridgeError(f"cannot import InferaSim AnchorStore: {exc}") from exc

    store = AnchorStore(store_root)
    recipe = recipe_from_spec(spec)

    from infera.projection.core.projection.inference_projection.search import regime

    # Only this model's warmups may calibrate it. The spec carries a checkout
    # path and a preset, the artifact a HuggingFace id, so either name may match
    # loosely; the preset alone would reject DeepSeek-R1's own warmup. An anchor
    # naming no model (a structural warmup) still has to agree on the regime
    # axes. When none is for this model there is no anchor: falling back to any
    # anchor calibrated one model against another.
    names = [n for n in (spec.model_path, resolve_preset(spec.model_path)) if n]
    entries = [
        e for e in store.entries() if not e.get("model") or any(regime.models_match(n, e["model"]) for n in names)
    ]
    if not entries:
        return None

    def rank(entry: dict[str, Any]) -> tuple[int, int, int, float]:
        """Sort key: regime distance, fidelity, provenance, transport closeness.

        Fidelity matters as much as regime here: a dummy-weight anchor runs the
        same kernels but with synthetic MoE routing, so its decode curve is much
        flatter than a real-weights run. Ranking it below a real-weights anchor
        in the same regime is the difference between ~2% and ~30% error against
        measured serving. Provenance is the same argument one level down -- an
        offline anchor does not even run the served kernels.
        """
        dist = regime.regime_distance(recipe, dict(entry.get("regime") or {}))
        real = _anchor_is_real_weights(entry["path"])
        served = _anchor_is_served(entry["path"])
        transport = entry.get("transport") or {}
        gap = 0.0
        for axis in ("tp", "ep", "pp"):
            av, rv = transport.get(axis), recipe.get(axis)
            if av and rv:
                gap += abs(float(av) - float(rv))
        return (dist, 0 if real else 1, 0 if served else 1, gap)

    usable = [
        e for e in entries if anchor_curve_is_sane(e["path"]) and not _runtime_conflicts(e["path"], _runtime_of(spec))
    ]
    if not usable:
        return None
    best = min(usable, key=rank)
    dist, fidelity_rank, served_rank, _ = rank(best)
    return AnchorChoice(
        path=best["path"],
        regime_distance=int(dist),
        model=best.get("model"),
        needs_warmup=bool(dist),
        real_weights=(fidelity_rank == 0),
        served=(served_rank == 0),
    )


# A decode step at a larger batch does strictly more work, so measured decode
# latency must not fall as batch rises. Small drops are ordinary run-to-run
# noise; a large one means the measurement itself is broken (the harness times
# decode by differencing two generate() calls, which degenerates when the longer
# call is not actually longer). Such an artifact is not merely imprecise -- it
# poisons every projection that anchors on it, so it is rejected outright.
_ANCHOR_MONOTONIC_TOLERANCE = 0.15


def anchor_curve_is_sane(path: str) -> bool:
    """False when an artifact's measured decode curve is physically impossible."""
    try:
        with open(path) as fh:
            doc = json.load(fh) or {}
    except (OSError, ValueError):
        return False
    return anchor_curve_is_sane_doc(doc)


def anchor_curve_is_sane_doc(doc: Any) -> bool:
    """:func:`anchor_curve_is_sane` on an artifact already parsed."""
    if not isinstance(doc, dict):
        return False
    points = []
    for entry in doc.get("sweep") or []:
        try:
            batch, decode_ms = int(entry["batch"]), float(entry["decode_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if decode_ms <= 0.0:
            return False
        points.append((batch, decode_ms))
    if not points:
        return False
    # A single-batch anchor is narrow, not invalid: there is no monotonicity to
    # check, and it still calibrates the one operating point it measured.
    points.sort()
    peak = points[0][1]
    for _, decode_ms in points[1:]:
        if decode_ms < peak * (1.0 - _ANCHOR_MONOTONIC_TOLERANCE):
            return False
        peak = max(peak, decode_ms)
    return True


def _anchor_is_served(path: str) -> bool:
    """True when an anchor was measured against a real server, not offline vLLM.

    The distinction is not cosmetic. Given identical flags the offline ``LLM()``
    entrypoint and ``vllm serve`` resolve different attention and MoE kernels,
    and the served decode step ran 1.9x the offline one at concurrency 8 and
    5.5x at 128. Calibrating a served target from an offline anchor scored no
    better than not calibrating at all, where a served anchor scored 2.2%.
    """
    try:
        with open(path) as fh:
            meta = (json.load(fh) or {}).get("meta") or {}
    except (OSError, ValueError):
        return False
    return "serving" in str(meta.get("derived_from") or "").lower()


def _runtime_conflicts(path: str, runtime: dict[str, str]) -> bool:
    """True when an anchor recorded an image or engine version ``runtime`` contradicts.

    Only measured anchors record one. An axis either side leaves blank is not
    a conflict.
    """
    if not runtime:
        return False
    try:
        with open(path) as fh:
            recorded = ((json.load(fh) or {}).get("meta") or {}).get("runtime") or {}
    except (OSError, ValueError):
        return False
    return any(recorded.get(k) and v and str(recorded[k]) != str(v) for k, v in runtime.items())


def _anchor_is_real_weights(path: str) -> bool:
    """True when an anchor artifact was measured with real checkpoint weights."""
    try:
        with open(path) as fh:
            meta = (json.load(fh) or {}).get("meta") or {}
    except (OSError, ValueError):
        return False
    if meta.get("real_weights") is not None:
        return bool(meta["real_weights"])
    return str(meta.get("load_format") or "").lower() not in ("dummy", "")


_HARVEST_SCRIPT_REL = "infera/projection/core/projection/inference_projection/benchmark_vllm.py"
_DEFAULT_HARVEST_GPUS = 4
_DEFAULT_HARVEST_TIMEOUT_SEC = 1800
# Parallelism is chosen by the harvest itself (``--tp`` / ``--benchmark-gpus``);
# a width pinned in the deployment's server args would fight it.
_PARALLEL_FLAGS = frozenset(
    {
        "--tensor-parallel-size",
        "-tp",
        "--tp",
        "--tp-size",
        "--pipeline-parallel-size",
        "-pp",
        "--pp-size",
        "--data-parallel-size",
        "-dp",
        "--dp-size",
    }
)


def harvest_enabled() -> bool:
    """Harvest on a miss unless ``HYPERLOOM_INFERASIM_HARVEST`` turns it off."""
    raw = str(os.environ.get(ENV_HARVEST) or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _harvest_engine(framework: str) -> str | None:
    """The ``--serving-backend`` Infera's harvest launches for ``framework``.

    The harvest only knows vLLM, SGLang and ATOM; a build of one of them
    ("mori-sglang") is launched as its family.
    """
    fw = (framework or "").lower()
    for family in ("sglang", "atom", "vllm"):
        if family in fw:
            return family
    return None


def _strip_parallel_flags(server_args: str) -> str:
    toks = (server_args or "").split()
    out: list[str] = []
    skip = False
    for tok in toks:
        if skip:
            skip = False
            continue
        flag = tok.split("=", 1)[0]
        if flag in _PARALLEL_FLAGS:
            skip = "=" not in tok
            continue
        out.append(tok)
    return " ".join(out)


def _regime_key(spec: ServingSpec) -> str:
    """One harvest per regime and model; concurrent misses on it share the lock."""
    recipe = recipe_from_spec(spec)
    axes = {k: recipe.get(k) for k in ("model", "engine", "weight_dtype", "kv_cache_dtype", "attention_backend")}
    axes["speculative"] = recipe.get("speculative")
    return hashlib.sha256(json.dumps(axes, sort_keys=True).encode()).hexdigest()[:16]


@contextlib.contextmanager
def _regime_lock(store_root: str, key: str) -> Iterator[None]:
    lock_path = Path(store_root) / f".harvest-{key}.lock"
    with lock_path.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def harvest_command(spec: ServingSpec, save_path: str) -> list[str] | None:
    """The served-anchor harvest for ``spec``'s regime, or None if it cannot run."""
    root = os.environ.get(ENV_ROOT, "")
    script = Path(root) / _HARVEST_SCRIPT_REL
    engine = _harvest_engine(spec.framework)
    if not root or not script.is_file() or not engine or not spec.model_path:
        return None
    gpus = _as_int(os.environ.get(ENV_HARVEST_GPUS), 0) or _DEFAULT_HARVEST_GPUS
    gpus = max(1, min(gpus, spec.tp * spec.pp))
    python = os.environ.get(ENV_HARVEST_PYTHON) or sys.executable or "python3"
    cmd = [
        python,
        str(script),
        "--model",
        spec.model_path,
        "--tp",
        str(spec.tp),
        "--pp",
        str(spec.pp),
        "--benchmark-gpus",
        str(gpus),
        "--input-len",
        str(spec.isl),
        "--output-len",
        str(spec.osl),
        # Capture mode: the sweep follows the engine's CUDA-graph sizes up to
        # here, so the concurrency variants of this regime interpolate rather
        # than extrapolate from it.
        "--concurrency",
        str(min(1024, max(64, 2 * spec.conc))),
        # Real weights: dummy routing flattens the MoE decode curve, the
        # difference between ~2% and ~30% TPOT error on gpt-oss-120b.
        "--load-format",
        "auto",
        "--routing-dist",
        "none",
        "--serving-backend",
        engine,
        "--save",
        save_path,
    ]
    if spec.ep > 1:
        cmd.append("--enable-expert-parallel")
    server_args = _strip_parallel_flags(spec.extra_server_args)
    if server_args:
        cmd.append("--server-args=" + server_args)
    return cmd


def harvest_anchor(spec: ServingSpec) -> AnchorChoice | None:
    """Measure a served anchor for ``spec``'s regime, index it, and select it.

    Only runs in benchmark mode with an anchor store to write into. Holds a
    per-regime lock so concurrent misses on one regime boot one server, and
    re-checks the store under it in case another caller already harvested.
    Returns None whenever the harvest cannot run or does not produce an
    in-regime anchor; the caller then fails closed.
    """
    store_root = os.environ.get(ENV_ANCHOR_STORE)
    if not store_root or not harvest_enabled():
        return None
    Path(store_root).mkdir(parents=True, exist_ok=True)
    key = _regime_key(spec)
    with _regime_lock(store_root, key):
        existing = select_anchor(spec)
        if existing is not None and existing.regime_distance == 0:
            return existing
        save_path = str(Path(store_root) / "harvested" / f"{key}-tp{spec.tp}-c{spec.conc}.json")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = harvest_command(spec, save_path)
        if cmd is None:
            log.warning(
                "inferasim: cannot harvest an anchor for %s on %s (needs %s, a vllm/sglang/atom engine and a model path)",
                spec.model_path,
                spec.framework,
                ENV_ROOT,
            )
            return None
        timeout = _as_int(os.environ.get(ENV_HARVEST_TIMEOUT), 0) or _DEFAULT_HARVEST_TIMEOUT_SEC
        log.info("inferasim: no in-regime anchor for %s; harvesting one: %s", spec.model_path, " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("inferasim: anchor harvest did not run (%s)", exc)
            return None
        if proc.returncode != 0 or not Path(save_path).is_file():
            log.warning(
                "inferasim: anchor harvest failed rc=%s: %s", proc.returncode, (proc.stderr or proc.stdout or "")[-600:]
            )
            return None
        if not anchor_curve_is_sane(save_path):
            log.warning("inferasim: harvested anchor %s failed the decode-curve sanity check", save_path)
            return None
        _index_artifact(store_root, save_path)
        choice = select_anchor(spec)
    if choice is None or choice.regime_distance != 0:
        log.warning("inferasim: harvested anchor %s is not selected in-regime for %s", save_path, spec.model_path)
        return None
    return choice


def resolve_anchor(spec: ServingSpec, *, harvest: bool = True) -> AnchorChoice:
    """The in-regime anchor benchmark mode calibrates ``spec`` against.

    Harvests one on a miss when ``harvest``. Raises when none can be had:
    benchmark mode never falls back to an analytical number while still being
    labelled calibrated.
    """
    anchor = select_anchor(spec)
    if anchor is not None and anchor.regime_distance == 0:
        return anchor
    harvested = harvest_anchor(spec) if harvest else None
    if harvested is not None:
        return harvested
    raise InferasimBridgeError(
        f"benchmark mode but no in-regime anchor for {spec.model_path!r} "
        f"(engine={spec.framework}) could be found{' or harvested' if harvest else ''}"
    )


MEASURED_DERIVED_FROM = "serving benchmark (Hyperloom EXPLORE decision round; mean TPOT)"
_MEASURED_SOURCES_KEPT = 32


def _runtime_of(spec: ServingSpec) -> dict[str, str]:
    """The deployment identity an anchor must not contradict: image, versions, launch levers."""
    runtime = dict(spec.runtime or {})
    if spec.launch:
        runtime["launch"] = spec.launch
    return runtime


def _measured_group_key(spec: ServingSpec, arch: str) -> str:
    """One artifact per launch config; its concurrencies become one ladder."""
    ident = {
        "model": spec.model_path,
        "engine": spec.framework,
        "tp": spec.tp,
        "ep": spec.ep,
        "pp": spec.pp,
        "isl": spec.isl,
        "osl": spec.osl,
        "weight_dtype": spec.weight_dtype,
        "kv_cache_dtype": spec.kv_cache_dtype,
        "server_args": " ".join(sorted((spec.extra_server_args or "").split())),
        "gpu": arch,
        "runtime": dict(sorted(_runtime_of(spec).items())),
    }
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:20]


def _measured_meta(spec: ServingSpec, arch: str, ref: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    method, k = parse_speculative(spec.extra_server_args)
    return {
        "model": spec.model_path,
        "batch": ref["batch"],
        "concurrency": ref["batch"],
        "input_len": spec.isl,
        "output_len": spec.osl,
        "tp": spec.tp,
        "ep": spec.ep,
        "pp": spec.pp,
        "target_tp": spec.tp,
        "target_pp": spec.pp,
        "benchmark_gpus": spec.tp * spec.pp,
        "weight_dtype": spec.weight_dtype,
        "kv_cache_dtype": spec.kv_cache_dtype,
        "attention_backend": _parse_server_arg_str(spec.extra_server_args, ATTENTION_BACKEND_FLAG),
        "enforce_eager": "--enforce-eager" in (spec.extra_server_args or "").split(),
        "speculative_method": method or "",
        "speculative_num_tokens": k or None,
        "server_args": spec.extra_server_args,
        "load_format": "auto",
        "real_weights": True,
        "gpu_arch": arch,
        "runtime": _runtime_of(spec),
        "derived_from": MEASURED_DERIVED_FROM,
        "sources": sources,
    }


def _extend_measured(path: Path, spec: ServingSpec, tpot_mean_ms: float, source: dict[str, Any] | None) -> dict:
    """The artifact at ``path`` with this run's point added; a re-measured batch replaces its point."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = {}
    points = {int(p["batch"]): p for p in (doc.get("sweep") or []) if p.get("batch")}
    points[spec.conc] = {"batch": spec.conc, "decode_ms": float(tpot_mean_ms)}
    sweep = [points[b] for b in sorted(points)]
    sources = list((doc.get("meta") or {}).get("sources") or [])
    if source:
        sources = (sources + [{"batch": spec.conc, **source}])[-_MEASURED_SOURCES_KEPT:]
    return {
        "backend": spec.framework,
        "client": "hyperloom-explore",
        "measured": {"model": {"decode_ms": sweep[-1]["decode_ms"]}},
        "sweep": sweep,
        "meta": _measured_meta(spec, gpu_arch(spec.gpu), sweep[-1], sources),
    }


def record_measured_anchor(spec: ServingSpec, *, tpot_mean_ms: float | None, source: dict | None = None) -> str | None:
    """Index a real decision round as a served anchor; return its path.

    The point is the run's mean TPOT at the batch it decoded at, which is how
    Infera's own served harvest derives a decode step. Prefill is left to the
    model: a TTFT under a full closed loop is mostly queue. Measurements of the
    same launch at different concurrencies extend one artifact into a ladder.
    No-op without an anchor store, a usable TPOT, or a launch whose envs and
    kernel overlays could be pinned.
    """
    store_root = os.environ.get(ENV_ANCHOR_STORE)
    if not store_root or not tpot_mean_ms or tpot_mean_ms <= 0 or not spec.model_path:
        return None
    if spec.launch == LAUNCH_UNPINNED:
        return None
    path = Path(store_root) / "measured" / f"{_measured_group_key(spec, gpu_arch(spec.gpu))}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _regime_lock(store_root, "measured-" + path.stem):
            doc = _extend_measured(path, spec, tpot_mean_ms, source)
            if not anchor_curve_is_sane_doc(doc):
                log.info("inferasim: not indexing %s; its decode ladder is not monotonic", path)
                return None
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
            tmp.replace(path)
        _index_artifact(store_root, str(path))
    except OSError as exc:
        log.warning("inferasim: could not record a measured anchor under %s (%s)", store_root, exc)
        return None
    return str(path)


def _index_artifact(store_root: str, path: str) -> None:
    """Add ``path`` to the store's index; a store that cannot be opened finds it later."""
    try:
        bridge._ensure_infera_importable()
        from infera.projection.core.projection.inference_projection.search.anchor_store import AnchorStore
    except (bridge.InferasimBridgeError, ImportError):
        # AnchorStore.discover() indexes it on next open.
        return
    with _regime_lock(store_root, "index"):
        AnchorStore(store_root, discover=False).add_artifact(path)


def resolve_mode(spec: ServingSpec) -> str:
    """The concrete mode ``spec`` is projected in: ``auto`` picks per anchor.

    ``auto`` calibrates when the store already holds an in-regime anchor for
    ``spec`` and simulates otherwise. Callers comparing several specs resolve
    it once, on the reference, and project every spec in that mode.
    """
    mode = projection_mode()
    if mode != MODE_AUTO:
        return mode
    try:
        anchor = select_anchor(spec)
    except InferasimBridgeError:
        return MODE_SIMULATE
    return MODE_BENCHMARK if anchor is not None and anchor.regime_distance == 0 else MODE_SIMULATE


def project(spec: ServingSpec, *, mode: str | None = None) -> ProjMetrics:
    """Run the InferaSim projection for ``spec`` and return mapped metrics.

    ``mode`` defaults to :func:`resolve_mode` for ``spec``. Benchmark mode
    harvests on a miss only when it was asked for explicitly; ``auto`` never
    boots a server.
    """
    bridge._ensure_infera_importable()
    workload, extra_env = bridge._resolve_workload_and_env(spec)

    from infera.projection.cli import build_parser
    from infera.projection.core.projection.inference_projection import (
        launch_projection_from_cli,
    )

    configured = projection_mode()
    mode = mode or resolve_mode(spec)
    if mode == MODE_AUTO:
        mode = resolve_mode(spec)
    est = estimator()
    anchor = resolve_anchor(spec, harvest=configured == MODE_BENCHMARK) if mode == MODE_BENCHMARK else None
    argv = bridge._build_argv(spec, workload, anchor, mode=mode, estimator_name=est)
    # Template reads INFERASIM_* env; also expose TP/PP/EP for template default
    # interpolation (overrides above still win for explicit workloads).
    prev_env: dict[str, str | None] = {}
    inject = dict(extra_env)
    inject.setdefault("INFERASIM_TP", str(spec.tp))
    inject.setdefault("INFERASIM_PP", str(spec.pp))
    inject.setdefault("INFERASIM_EP", str(spec.ep))
    for key, val in inject.items():
        prev_env[key] = os.environ.get(key)
        os.environ[key] = val
    try:
        args, overrides = build_parser().parse_known_args(argv)
        results = launch_projection_from_cli(args, overrides)
    except InferasimBridgeError:
        raise
    except Exception as exc:
        raise InferasimBridgeError(f"InferaSim projection failed: {exc}") from exc
    finally:
        for key, val in prev_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    if results.get("performance") is None:
        raise InferasimBridgeError("InferaSim returned no performance projection")
    if est == ESTIMATOR_DES and not (results.get("des") or {}).get("point"):
        # Ranking a replayed stack against a closed-form variant would compare
        # two estimators, not two configurations.
        raise InferasimBridgeError("InferaSim ran no closed-loop replay")
    if est != ESTIMATOR_DES:
        results = {**results, "des": None}
    return bridge._metrics_from_results(spec, results, anchor, mode)
