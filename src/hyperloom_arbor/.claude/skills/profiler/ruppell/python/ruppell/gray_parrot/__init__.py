"""gray-parrot: ruppell's hardware counters (docs/gray-parrot/README.md).

Off by default: with RUPPELL_COUNTERS and RUPPELL_DEVICE_COUNTERS unset, the tool creates nothing
and a run has none of the files below.

- **Sets** per architecture (``sets/<arch>.yaml``): a set is one or more groups, each collectable
  in one pass. `ruppell run` resolves the sets RUPPELL_COUNTERS names into the groups the tool
  library reads (``RUPPELL_COUNTER_GROUPS_<ARCH>``), one variable per architecture with a set file,
  so the tool picks each GPU's own. ``raw:A,B;C,D`` names groups in the vendor's names directly.
- **Records** (docs/gray-parrot/records.md, the contract both platforms write): ``counters.bin``
  (per counted kernel and counter; on ROCm key0 is the dispatch id, joined to dispatch.bin),
  ``counters.tsv`` (each group's counters: names, sets, how instances were reduced) and
  ``device_counters.bin`` (the device service's readings).
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

SETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sets")

# counters.bin and device_counters.bin: the engine's dtypes, the contract's (docs/gray-parrot/records.md)
from ruppell.engine import COUNTER_DT, DEVICE_COUNTER_DT  # noqa: E402

Group = Tuple[str, str, List[str]]   # (set, group, counters)


# ---- sets ----------------------------------------------------------------------------------------

def architectures() -> List[str]:
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(SETS_DIR, "*.yaml")))


def load_sets(arch: str) -> dict:
    """An architecture's set file, as written (version, slots, common, sets)."""
    import yaml

    path = os.path.join(SETS_DIR, f"{arch}.yaml")
    if not os.path.exists(path):
        raise ValueError(f"no counter sets for {arch} (have: {', '.join(architectures()) or 'none'})")
    with open(path) as f:
        return yaml.safe_load(f)


def block_of(counter: str) -> str:
    """The hardware block a base counter's name starts with (SQ, TCC, TCP, SPI, GRBM, ...)."""
    return counter.split("_", 1)[0]


def slot_use(counters: List[str]) -> Dict[str, int]:
    use: Dict[str, int] = {}
    for c in counters:
        use[block_of(c)] = use.get(block_of(c), 0) + 1
    return use


def resolve(spec: str, arch: str) -> List[Group]:
    """The groups ``spec`` asks for on ``arch``: sets by name ("memory,compute"), each group with the
    file's ``common`` counters added, or ``raw:A,B;C,D`` as written (groups raw/r1, raw/r2)."""
    spec = spec.strip()
    if spec.startswith("raw:"):
        out = []
        for i, g in enumerate((x for x in spec[4:].split(";") if x.strip()), 1):
            out.append(("raw", f"r{i}", [c.strip() for c in g.split(",") if c.strip()]))
        if not out:
            raise ValueError(f"'{spec}': no counters")
        return out
    doc = load_sets(arch)
    sets = doc.get("sets") or {}
    common = list(doc.get("common") or [])
    out = []
    for name in (x.strip() for x in spec.split(",") if x.strip()):
        if name not in sets:
            raise ValueError(f"no set '{name}' for {arch} (have: {', '.join(sorted(sets))})")
        for g, counters in sets[name].items():
            cs = list(counters) + [c for c in common if c not in counters]
            out.append((name, str(g), cs))
    if not out:
        raise ValueError(f"'{spec}': no sets")
    return out


def groups_value(groups: List[Group]) -> str:
    """The tool's form: "set/group=A,B;set/group=C"."""
    return ";".join(f"{s}/{g}=" + ",".join(cs) for s, g, cs in groups)


def check_fits(groups: List[Group], arch: str) -> List[str]:
    """Groups whose counters exceed the file's per-block slots, by slot arithmetic on the names
    (base counters only; derived ones are left to the device check). Empty when all fit."""
    slots = (load_sets(arch).get("slots") or {})
    bad = []
    for s, g, cs in groups:
        for block, n in slot_use(cs).items():
            if block in slots and n > slots[block]:
                bad.append(f"{s}/{g}: {block} {n} > {slots[block]}")
    return bad


def run_env(environ: Dict[str, str]) -> Dict[str, str]:
    """The variables `ruppell run` adds for the tool: RUPPELL_COUNTERS's and RUPPELL_DEVICE_COUNTERS's
    sets resolved for every architecture with a set file. Nothing when both are unset, or given as
    raw:. Raises ValueError on a malformed switch, before the program starts."""
    out: Dict[str, str] = {}
    from ruppell.api import _rate

    spec = environ.get("RUPPELL_COUNTERS", "").strip()
    if spec:
        _rate("RUPPELL_COUNTER_STEPS", "1/100", environ)
        if not spec.startswith("raw:"):
            got = False
            for arch in architectures():
                try:
                    out[f"RUPPELL_COUNTER_GROUPS_{arch.upper()}"] = groups_value(resolve(spec, arch))
                    got = True
                except ValueError:
                    continue   # this architecture lacks a set: its GPUs are refused at start
            if not got:
                raise ValueError(f"RUPPELL_COUNTERS={spec}: no architecture has these sets")
    dev = environ.get("RUPPELL_DEVICE_COUNTERS", "").strip()
    if dev:
        dspec, at, rate = dev.rpartition("@")
        if not at or not dspec or not rate:
            raise ValueError(f"RUPPELL_DEVICE_COUNTERS={dev}: want SPEC@RATE (activity@1kHz)")
        if not dspec.startswith("raw:"):
            for arch in architectures():
                try:
                    groups = resolve(dspec, arch)
                except ValueError:
                    continue
                if len(groups) != 1:
                    raise ValueError(f"RUPPELL_DEVICE_COUNTERS={dev}: the device service counts one group; "
                                     f"{dspec} has {len(groups)} on {arch}")
                out[f"RUPPELL_DEVICE_COUNTER_GROUPS_{arch.upper()}"] = groups_value(groups)
    return out


# ---- records (docs/gray-parrot/records.md) ---------------------------------------------------------

def read_tsv(proc_dir: str) -> List[dict]:
    """counters.tsv, one line per group and counter: group, counter (a number unique across groups),
    name (NAME[xcc=i] for a counter split by XCC), set, kind (base or derived), source (a base
    counter's block, a derived one's formula), reduce (sum, max, or none when split), unit."""
    path = os.path.join(proc_dir, "counters.tsv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            x = line.rstrip("\n").split("\t")
            if len(x) < 8 or not x[0].isdigit():
                continue   # the header line, or a partial line
            rows.append({"group": int(x[0]), "counter": int(x[1]), "name": x[2], "set": x[3], "kind": x[4],
                         "source": x[5], "reduce": x[6], "unit": x[7]})
    return rows


def base_name(name: str) -> str:
    """A split counter's vendor name: TCC_HIT[xcc=3] -> TCC_HIT."""
    return name.split("[", 1)[0]


def _bin(path: str, dt: np.dtype) -> np.ndarray:
    if not os.path.exists(path):
        return np.zeros(0, dtype=dt)
    size = os.path.getsize(path)
    return np.fromfile(path, dtype=dt, count=size // dt.itemsize)


class Counters:
    """One process's counter records. ``values``: counters.bin with the records a dispatch's values
    were split into summed, one row per (key0, agent, step, group, counter); on ROCm key0 is the
    dispatch id. ``device``: device_counters.bin. ``device_groups``: the groups meta.json names as
    the device service's; the others are counted per kernel."""

    def __init__(self, proc_dir: str):
        import json

        self.path = proc_dir
        self.rows = read_tsv(proc_dir)
        self.names = {r["counter"]: r["name"] for r in self.rows}
        self.meta = {}
        try:
            with open(os.path.join(proc_dir, "meta.json")) as f:
                m = json.load(f).get("counters")
            self.meta = m if isinstance(m, dict) else {}
        except (OSError, ValueError):
            pass
        dev = self.meta.get("device")
        self.device_groups = {int(dev["group"])} if isinstance(dev, dict) and "group" in dev else set()
        self.recorded = os.path.exists(os.path.join(proc_dir, "counters.bin"))
        self.values = _sum_by(_bin(os.path.join(proc_dir, "counters.bin"), COUNTER_DT),
                              ("key0", "agent", "step", "group", "counter"))
        self.device_recorded = os.path.exists(os.path.join(proc_dir, "device_counters.bin"))
        self.device = _bin(os.path.join(proc_dir, "device_counters.bin"), DEVICE_COUNTER_DT)

    def dispatch_rows(self) -> List[dict]:
        return [r for r in self.rows if r["group"] not in self.device_groups]

    def device_rows(self) -> List[dict]:
        return [r for r in self.rows if r["group"] in self.device_groups]

    def name(self, counter: int) -> str:
        return self.names.get(int(counter), f"counter_{int(counter)}")

    def counted(self) -> np.ndarray:
        """The counted dispatches (key0), each once."""
        return np.unique(self.values["key0"]) if len(self.values) else np.zeros(0, np.uint64)

    def group_sizes(self) -> Dict[int, int]:
        """Per counted group, its number of counters (values a fully counted dispatch has)."""
        n: Dict[int, int] = {}
        for r in self.dispatch_rows():
            n[r["group"]] = n.get(r["group"], 0) + 1
        return n

    def per_dispatch(self, counter: str) -> Dict[int, float]:
        """key0 (the dispatch id) -> the value of the counter named ``counter`` (an XCC split summed)."""
        nums = np.array(sorted({r["counter"] for r in self.dispatch_rows() if base_name(r["name"]) == counter}),
                        dtype=np.uint16)
        v = self.values[np.isin(self.values["counter"], nums)]
        out: Dict[int, float] = {}
        for d, x in zip(v["key0"].tolist(), v["value"].tolist()):
            out[d] = out.get(d, 0.0) + x
        return out


def _sum_by(a: np.ndarray, keys: Tuple[str, ...]) -> np.ndarray:
    """Rows of ``a`` with equal ``keys`` merged, their values summed, ordered by the keys."""
    if not len(a):
        return a
    order = np.lexsort(tuple(a[k] for k in reversed(keys)))
    a = a[order]
    new = np.ones(len(a), bool)
    new[1:] = np.zeros(len(a) - 1, bool)
    for k in keys:
        new[1:] |= a[k][1:] != a[k][:-1]
    starts = np.nonzero(new)[0]
    out = a[starts].copy()
    out["value"] = np.add.reduceat(a["value"], starts)
    return out
