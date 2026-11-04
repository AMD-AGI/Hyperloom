"""The scope a measurement holds in: the image it ran on and its operating point.

A kill recorded on one image is stale on the next: defaults and winners move
with the image (serving-playbook amd-knobs.md §0, rule 4). And most big
settings are policies that flip sign with concurrency or sequence length
(regime-tables.md), so a kill at one operating point says nothing about
another. The playbook's rule is to record both with every kill; this makes it
mechanical.

``capture`` runs at the start of every iteration: the container's image id
(``docker inspect``) and the operating point (concurrency, ISL, OSL, TP) from
the config and the live flags, refined from ``bench_output.json`` once the
bench has run. It is written to ``iter_N/scope.json``, stamped on the
iteration's gate row, on the DEAD_ENDs and FACTs it records, and on the map's
kills; ``ledger.render`` marks a dead end from another image as stale.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

KEYS = ("image", "concurrency", "isl", "osl", "tp")
_TP_FLAGS = ("--tensor-parallel-size", "-tp", "--tp", "--tp-size")
# bench_output.json spellings (the Model M packs write `conc`).
_BENCH_KEYS = {"concurrency": ("concurrency", "conc"), "isl": ("isl",), "osl": ("osl",),
               "tp": ("tp", "tensor_parallel_size")}

# Runs argv on the host; returns (returncode, stdout).
Runner = Callable[[list], tuple]


def _host_runner(argv: list) -> tuple:
    r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def image_of(config: dict, runner: Runner | None = None) -> str | None:
    """The container's image id (``sha256:...``), or None when it cannot be read."""
    from arbor.execute.workspace import HOST
    container = config.get("docker_container", "")
    if not container or container == HOST:
        return None
    try:
        rc, out = (runner or _host_runner)(["docker", "inspect", "--format", "{{.Image}}", container])
    except Exception:  # noqa: BLE001 - a scope is bookkeeping
        return None
    out = (out or "").strip()
    return out if rc == 0 and out else None


def operating_point(config: dict, flags: str | None = None) -> dict:
    """Concurrency, ISL, OSL and TP from the config, TP from the live flags first."""
    from arbor.measure.armcheck import flags_of
    if flags is None:
        from arbor.execute.bench import _live_flags
        try:
            flags = _live_flags(config)
        except Exception:  # noqa: BLE001
            flags = str(config.get("server_flags", "") or "")
    parsed = flags_of(flags)
    tp = next((parsed[f] for f in _TP_FLAGS if f in parsed), None)
    return {
        "concurrency": _int(config.get("concurrency")),
        "isl": _int(config.get("isl")),
        "osl": _int(config.get("osl")),
        "tp": _int(tp) or _int(config.get("tp", config.get("tensor_parallel_size"))),
    }


def capture(config: dict, iter_dir=None, runner: Runner | None = None) -> dict:
    """The iteration's scope. Never raises; an unreadable part is None."""
    try:
        op = operating_point(config)
    except Exception:  # noqa: BLE001
        op = {k: None for k in KEYS[1:]}
    scope = {"image": image_of(config, runner), **op}
    write(scope, iter_dir)
    return scope


def refine(scope: dict | None, bench_output: dict | None) -> dict | None:
    """The operating point the bench says it ran, over the configured one."""
    if scope is None:
        return None
    out = dict(scope)
    for key, names in _BENCH_KEYS.items():
        for n in names:
            v = _int((bench_output or {}).get(n))
            if v is not None:
                out[key] = v
                break
    return out


def write(scope: dict | None, iter_dir) -> None:
    if scope is None or iter_dir is None:
        return
    try:
        Path(iter_dir).mkdir(parents=True, exist_ok=True)
        (Path(iter_dir) / "scope.json").write_text(json.dumps(scope, indent=1) + "\n")
    except OSError:
        pass


def short_image(image: str | None) -> str:
    return (image or "").split(":")[-1][:12]


def describe(scope: dict | None) -> str:
    """``image 58cb6b8618bd, C32, ISL 8192, OSL 1024, TP8``; the parts that are known."""
    s = scope or {}
    parts = []
    if s.get("image"):
        parts.append(f"image {short_image(s['image'])}")
    if s.get("concurrency") is not None:
        parts.append(f"C{s['concurrency']}")
    if s.get("isl") is not None:
        parts.append(f"ISL {s['isl']}")
    if s.get("osl") is not None:
        parts.append(f"OSL {s['osl']}")
    if s.get("tp") is not None:
        parts.append(f"TP{s['tp']}")
    return ", ".join(parts)
