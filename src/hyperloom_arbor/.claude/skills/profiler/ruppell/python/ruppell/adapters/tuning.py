"""Run-time tuning pinned across processes (RUPPELL_TUNING=FILE).

Some kernels are chosen at run time by timing: Inductor's pointwise and
reduction kernels (`CachingAutotuner`: vLLM's `get_masked_input_and_mask`
is `torch.compile`d even with vLLM's compilation off) and `@triton.autotune`
kernels. Each process benchmarks its candidates at the first call and keeps
the fastest, so two servers, or two ranks of one, can run different block
sizes of the same kernel (job 49826: grid 128 and 512 traced, 256 untraced).
A traced and an untraced server then run different GPU programs for no
reason of ruppell's, and the comparison cannot tell.

With RUPPELL_TUNING=FILE every process that tunes a kernel replays the choice
recorded in FILE, or benchmarks as usual and records its choice. The first
process to record a kernel decides it for every later server pointed at the
same FILE, and for every rank that tunes it after. A lock on FILE.lock guards
only the reads and writes of FILE, never a benchmark: a benchmark synchronizes
the device, and a rank holding the lock there can wait on a collective that a
rank blocked on the lock has not joined (model-w TP8, MI350X, 2026-10-23: the
compiled start hung at graph capture). So two processes can tune one kernel at
once; the first to record decides FILE, and the other keeps its own choice for
its run and says so ("raced"). Inductor's own
autotune disk cache is bypassed while pinning (a hit there skips tuning,
and so the pin). Only the tuning path is wrapped; launches are not.

Each record and replay is logged ("tuning: replayed ...") so a run shows
which case it got. A pinned config the kernel no longer offers is benchmarked
and reported ("unmatched"); FILE is not rewritten for it.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
from typing import Any, Dict

_path = os.environ.get("RUPPELL_TUNING") or None
stats = {"recorded": 0, "replayed": 0, "unmatched": 0, "raced": 0}


def _log(msg: str) -> None:
    print(f"[ruppell] pid {os.getpid()}: tuning: {msg}", file=sys.stderr, flush=True)


def enabled() -> bool:
    return _path is not None


def _load() -> Dict[str, Any]:
    try:
        with open(_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save(pins: Dict[str, Any]) -> None:
    tmp = f"{_path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(pins, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, _path)


class _Lock:
    """An exclusive lock on FILE.lock, held while FILE is read or written (never across a benchmark)."""

    def __init__(self):
        self.f = open(_path + ".lock", "a")
        fcntl.flock(self.f, fcntl.LOCK_EX)

    def release(self):
        if self.f is not None:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
            self.f = None


@contextlib.contextmanager
def _locked():
    lk = _Lock()
    try:
        yield
    finally:
        lk.release()


def config_dict(cfg: Any) -> Dict[str, Any]:
    """A Triton Config as data: its kwargs (block sizes and such) and launch options."""
    d: Dict[str, Any] = {"kwargs": {k: cfg.kwargs[k] for k in sorted(cfg.kwargs)}}
    for k in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
        v = getattr(cfg, k, None)
        if v is not None:
            d[k] = v
    return d


def _same(cfg: Any, pinned: Dict[str, Any]) -> bool:
    return json.dumps(config_dict(cfg), sort_keys=True, default=str) == json.dumps(pinned, sort_keys=True, default=str)


def _record(pins: Dict[str, Any], key: str, cfg: Any) -> None:
    pins[key] = config_dict(cfg)
    _save(pins)
    stats["recorded"] += 1
    _log(f"recorded {key} {pins[key]}")


def _pinned(key: str) -> Any:
    with _locked():
        return _load().get(key)


def _settle(key: str, cfg: Any) -> None:
    """After a benchmark: record its choice, unless another process recorded the kernel meanwhile."""
    with _locked():
        pins = _load()
        now = pins.get(key)
        if now is None:
            _record(pins, key, cfg)
        elif not _same(cfg, now):
            stats["raced"] += 1
            _log(f"raced {key}: another process recorded {now} while this one tuned; keeping {config_dict(cfg)} in this run")


# ---- Inductor ---------------------------------------------------------------------------------------

def patch_inductor(mod: Any) -> None:
    """torch._inductor.runtime.triton_heuristics: pin CachingAutotuner's choice."""
    C = mod.CachingAutotuner
    if getattr(C, "_ruppell_pinned", False):
        return
    orig = C.autotune_to_one_config

    def autotune_to_one_config(self, *args, **kwargs):
        key = "inductor:" + (getattr(self, "kernel_hash", "") or self.fn.__name__)
        want = _pinned(key)
        if want is not None:
            hit = [l for l in self.launchers if _same(l.config, want)]
            if hit:
                best = hit[0]
                release = getattr(self, "_release_static_launchers_except", None)
                if release is not None:
                    release(best)
                self.launchers = [best]
                prune = getattr(self, "_prune_compile_results_to_launcher", None)
                if prune is not None:
                    prune(best)
                stats["replayed"] += 1
                _log(f"replayed {key} {want}")
                return None
            stats["unmatched"] += 1
            _log(f"unmatched {key}: pinned {want} not among {len(self.launchers)} candidates; benchmarking")
        r = orig(self, *args, **kwargs)   # unlocked: a benchmark synchronizes the device
        if want is None:
            _settle(key, self.launchers[0].config)
        return r

    C.autotune_to_one_config = autotune_to_one_config
    # Inductor's autotune disk cache: a hit loads one config and skips tuning
    check = getattr(mod, "check_autotune_cache", None)
    if check is not None:
        def check_autotune_cache(configs, filename, inductor_meta):
            return configs, None, None
        mod.check_autotune_cache = check_autotune_cache
    C._ruppell_pinned = True


# ---- Triton -----------------------------------------------------------------------------------------

def patch_triton(mod: Any) -> None:
    """triton.runtime.autotuner: pin Autotuner's choice. prune_configs runs only
    when the tuning key is not yet in the autotuner's cache (Autotuner.run), so
    the lookup costs nothing on a launch that is already tuned."""
    A = mod.Autotuner
    if getattr(A, "_ruppell_pinned", False):
        return
    orig_prune, orig_run = A.prune_configs, A.run

    def prune_configs(self, kwargs):
        configs = orig_prune(self, kwargs)
        all_args = {**(self.nargs or {}), **kwargs}
        _args = {k: v for k, v in all_args.items() if k in self.arg_names}
        tkey = [_args[k] for k in self.keys if k in _args]
        for v in _args.values():
            if hasattr(v, "dtype"):
                tkey.append(str(v.dtype))
        fn = getattr(self, "base_fn", None) or self.fn
        key = f"triton:{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', fn))}:{tuple(tkey)!r}"
        want = _pinned(key)
        self._ruppell_pin = (key, want, tuple(tkey))
        if want is not None:
            hit = [c for c in configs if _same(c, want)]
            if hit:
                stats["replayed"] += 1
                _log(f"replayed {key} {want}")
                return hit[:1]
            stats["unmatched"] += 1
            _log(f"unmatched {key}: pinned {want} not among {len(configs)} candidates; benchmarking")
        return configs

    def run(self, *args, **kwargs):
        try:
            return orig_run(self, *args, **kwargs)
        finally:
            pin = self.__dict__.pop("_ruppell_pin", None)
            if pin is not None:
                key, want, tkey = pin
                if want is None and tkey in self.cache:
                    _settle(key, self.cache[tkey])

    A.prune_configs = prune_configs
    A.run = run
    A._ruppell_pinned = True


INDUCTOR = "torch._inductor.runtime.triton_heuristics"
TRITON = "triton.runtime.autotuner"
PATCHES = {INDUCTOR: patch_inductor, TRITON: patch_triton}
