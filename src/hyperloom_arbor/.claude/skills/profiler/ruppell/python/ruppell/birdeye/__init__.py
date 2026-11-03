"""birdeye: a built-in "micro TraceLens" report.

    python -m ruppell birdeye SRC [--format auto|ruppell|nsys|rocprof|kineto] [--steps A:B]
                                  [--phase NAME=START:END] [--rank N] [--out DIR] [--short-us 10] [--top 30]
                                  [--comm ruppell|tracelens]
    python -m ruppell birdeye-compare OUT_DIR SRC_OR_DIR... [--tags a,b,...]

SRC is a ruppell run directory, ruppell's Chrome trace, a PyTorch profiler
(Kineto) trace (`.json`, `.json.gz`, or a directory of per-rank files), an
Nsight Systems SQLite export, or a rocprofv3 kernel-trace CSV (or the
directory holding them). Every source becomes one normalized kernel table
(`readers.py`); the tables (`analyses.py`) carry TraceLens's sheet and column
names where the meaning is the same, and the report (`report.py`) writes
`OUT/<sheet>.csv`, `OUT/birdeye.json` and `OUT/REPORT.md`. `compare.py`
merges several reports. docs/birdeye.md says what each table means.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple


def default_out(src: str) -> str:
    src = src.rstrip("/\\")
    if os.path.isdir(src):
        return os.path.join(src, "birdeye")
    base = src
    for ext in (".gz", ".json", ".sqlite", ".csv"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    return base + "_birdeye"


def analysis_commit() -> Optional[str]:
    from ruppell.cli import _commit
    return _commit()


def _without_counter_steps(t):
    """The table without counter steps (gray-parrot: steps whose fields name a counter group),
    whose kernels counting serialized, so every time column leaves them out, as the engine does;
    with RUPPELL_UNTIMED_AFTER_COUNTER=1 the step after each too. Step ids are per rank."""
    import numpy as np

    st = t.steps
    if not len(st["step"]):
        return t
    cs = np.array(["counters" in f for f in t.step_fields], bool)
    if not cs.any():
        return t
    drop = cs.copy()
    if os.environ.get("RUPPELL_UNTIMED_AFTER_COUNTER") == "1":
        for r in np.unique(st["rank"]).tolist():
            idx = np.nonzero(st["rank"] == r)[0]
            idx = idx[np.argsort(st["start"][idx], kind="stable")]
            for a, b in zip(idx.tolist(), idx[1:].tolist()):
                if cs[a] and not cs[b]:
                    drop[b] = True
    key_s = st["step"].astype(np.int64) * 4096 + st["rank"]
    key_k = t.step.astype(np.int64) * 4096 + t.rank
    km = ~(np.isin(key_k, key_s[drop]) & (t.step >= 0))
    t = t.select(km, ~drop)
    t.info["counter_steps_left_out"] = int(drop.sum())
    return t


def load(src: str, fmt: str = "auto", steps: Optional[Tuple[int, int]] = None):
    """The normalized table of a source, cut to steps A..B (step ids) if given."""
    from ruppell.birdeye.readers import read

    t = read(src, fmt)
    t.info["t0_ns"] = int(t.start.min()) if len(t) else 0
    t = _without_counter_steps(t)
    if steps is not None:
        if not len(t.steps["step"]):
            raise ValueError(f"{src}: --steps given, but the source has no steps")
        lo, hi = steps
        km = (t.step >= lo) & (t.step <= hi)
        sm = (t.steps["step"] >= lo) & (t.steps["step"] <= hi)
        if not km.any():
            raise ValueError(f"{src}: no activity in steps {lo}:{hi}")
        t = t.select(km, sm)
    return t


def run(src: str, out: Optional[str] = None, fmt: str = "auto", steps: Optional[Tuple[int, int]] = None,
        phases: Optional[Dict[str, Tuple[float, float]]] = None, rank: Optional[int] = None,
        short_us: float = 10.0, top: int = 30, comm: str = "ruppell") -> dict:
    """Read SRC, compute every table, write OUT. Returns the summary, the
    output directory and the files written."""
    from ruppell.birdeye.analyses import analyze
    from ruppell.birdeye.report import write

    if comm not in ("ruppell", "tracelens"):
        raise ValueError(f"--comm {comm}: ruppell or tracelens")
    t = load(src, fmt, steps)
    res = analyze(t, rank=rank, short_us=short_us, comm=comm, phases=phases)
    out = out or default_out(src)
    opts = {"format": fmt, "steps": f"{steps[0]}:{steps[1]}" if steps else None, "phases": phases, "rank": rank,
            "short_us": short_us, "top": top, "comm": comm}
    files = write(out, t.info, res, opts, analysis_commit())
    return {"out": out, "files": files, "summary": res["summary"], "skipped": res["skipped"],
            "tables": {k: len(v.rows) for k, v in res["sheets"].items()}}
