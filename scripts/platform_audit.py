#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Audit the AMD EPYC CPU tuning that affects inference benchmark results.

Reads only what the operating system exposes -- ``/sys``, ``/proc`` and, when
run as root, the HWCR MSR.

Judges Core Performance Boost and the cpufreq governor, which have an
unambiguous right answer here. Records determinism, SMT and nodes-per-socket
without a verdict, because AMD's own tuning guide varies them by workload and
this tool should not invent a recommendation for LLM inference that AMD does
not publish. See ``SOURCE`` below for the citation and the per-knob reasoning.

Why check at all, given a session's A/B is same-machine: host tuning applies to
baseline and candidates alike, so it cancels out of the *delta*. It does not
cancel out of what the session exports. On a de-tuned node the absolute
throughput is low, and the optimizer is searching around a CPU-side bottleneck
that will not exist on a correct machine -- so the configuration it selects can
be tuned against a phantom constraint and is then filed in the recipe KB for
other people to use.

Scope: OS layer only. Three further knobs -- APBDIS, DF C-states and the
platform's High Performance profile -- are not readable this way on kernels
without ``amd_hsmp`` and require Redfish against the BMC. Reaching them means
minting a temporary privileged account on the service processor, which is a
different risk class from anything here, so it lives in a separate tool and its
own review rather than being smuggled in behind a ``--bmc-host`` flag.

Deliberately self-contained: the point of this script is to audit a node that
may not have Hyperloom installed, so it copies a little sysfs plumbing rather
than importing ``hyperloom.common.platform_probe``. That duplication is a
choice for portability, not drift -- the in-repo callers all share one helper.

Usage::

    python3 scripts/platform_audit.py            # full, ~10s of measurement
    python3 scripts/platform_audit.py --quick    # no load generation
    sudo python3 scripts/platform_audit.py       # adds the MSR reading
    python3 scripts/platform_audit.py --json

Exit codes:

    0  every checked knob is on target
    1  a knob is definitively wrong
    2  a knob could not be resolved on this host
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess  # nosec B404 - fixed argv, no shell, load generation only.
import sys
import time

# --------------------------------------------------------------------------
# Target profile
# --------------------------------------------------------------------------
#: AMD, "BIOS & Workload Tuning Guide for AMD EPYC 9004 Series Processors",
#: publication 58011 rev 1.0 (2025-05-09).
#: https://docs.amd.com/v/u/en-US/58011-epyc-9004-tg-bios-and-workload
#:
#: Chapter 5 ("Workload-Specific BIOS Settings") is the reference for the target
#: profile, and reading it closely is also why this tool judges less than it
#: originally did. Chapter 5 does not publish one performance profile: it
#: publishes a different column per workload -- CPU-intensive, Java throughput,
#: Java latency, virtualization, database, HPC/Telco -- and most rows differ
#: between them. Where chapter 5 itself varies a knob by workload, this tool
#: records it instead of asserting an answer for LLM inference that AMD does not
#: publish. The guide covers 9004 (Genoa); these knobs and their semantics carry
#: forward to 9005 (Turin), which is what the fleet actually runs.
SOURCE = (
    "AMD BIOS & Workload Tuning Guide for EPYC 9004 (pub. 58011 rev 1.0), ch. 5"
)

CHECKED: dict[str, dict] = {
    "core_performance_boost": {
        "label": "Core Performance Boost",
        "target": ("enabled",),
        "basis": (
            "Boost is Enable/Auto by default and no chapter 5 profile disables it "
            "(58011 §4.1.3). Off, the CPU-side work in a serving benchmark -- "
            "sampling, scheduling, tokenization, dispatch -- runs below rated "
            "clocks, and the effect is directly measurable here as achieved MHz."
        ),
    },
    "cpufreq_governor": {
        "label": "cpufreq governor",
        "target": ("performance",),
        "basis": (
            "An OS setting, not a BIOS one, so it is outside 58011: the basis is "
            "that a ramping governor distorts latency percentiles, because the "
            "early requests of a burst are served at a lower clock than the later "
            "ones. This is the knob most often wrong on a freshly imaged node."
        ),
    },
}

#: Recorded for comparability, never judged.
#:
#: ``determinism``: 58011 §4.2.2 frames this as a deployment choice, not a
#: correctness one. Performance determinism gives "uniform performance across
#: identically configured systems"; Power determinism gives "maximum performance
#: of any individual system ... resulting in a varying performance range across
#: the datacenter". Chapter 5 leaves it at default for general-purpose and most
#: HPC columns and selects Power only for the database profile. Hyperloom wants
#: both properties at once -- the highest number on this node, and comparability
#: across nodes -- so asserting either value would contradict one of its own
#: goals. It is measured and recorded; the operator picks.
#: ``smt``: on by default across the EPYC fleet, and chapter 5 leaves SMT Control
#: at default in every general-purpose column. Judging it would fail nearly every
#: node from day one, which is why the preflight check stopped warning on it too.
#: ``nps``: NPS1 and NPS4 are opposite, both defensible tradeoffs -- NPS1
#: interleaves for bandwidth, NPS4 favours locality -- and chapter 5's own NUMA
#: rows differ per workload.
RECORDED = ("determinism", "smt", "nps")

#: Frequency spread, in MHz, above which cores are judged to be running to their
#: own limits rather than a common one. Chosen as a threshold comfortably above
#: sampling jitter observed on an idle-ish EPYC node, not from a vendor spec;
#: it is a heuristic, and a run that lands near it reports UNKNOWN rather than
#: picking a side.
DETERMINISM_SPREAD_MHZ = 8.0
DETERMINISM_AMBIGUOUS_MHZ = 4.0


def read(path: str) -> str:
    """Read a ``/sys`` or ``/proc`` file, returning ``""`` when unreadable."""
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return ""


# --------------------------------------------------------------------------
# CPU identity and topology
# --------------------------------------------------------------------------

_EPYC_FAMILIES = {
    "9005": "Turin",
    "9004": "Genoa",
    "8004": "Siena",
    "7003": "Milan",
    "7002": "Rome",
    "7001": "Naples",
}


def epyc_generation(model: str) -> str:
    """Family name from an EPYC model number, or ``"unknown"``.

    The series lives in the first and last digits of a four-digit part (9575F ->
    9005), not the middle ones. Parts with a non-numeric model such as the
    cloud-specific ``EPYC 9V84`` do not match and return ``"unknown"``: that is
    a refusal to guess, and it only costs the generation-keyed boost-ceiling
    expectation, which simply is not available on those hosts.
    """
    m = re.search(r"EPYC\s+(\d{4})", model)
    if not m:
        return "unknown"
    d = m.group(1)
    return f"EPYC {d[0]}00{d[3]} ({_EPYC_FAMILIES.get(f'{d[0]}00{d[3]}', 'unknown')})"


def cpu_identity() -> dict:
    """Model, socket and NUMA counts, and derived nodes-per-socket."""
    model = "unknown"
    for line in read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    sockets = len(
        {
            read(f"/sys/devices/system/cpu/{d}/topology/physical_package_id")
            for d in os.listdir("/sys/devices/system/cpu")
            if re.fullmatch(r"cpu\d+", d)
        }
        - {""}
    )
    try:
        nodes = len(
            [d for d in os.listdir("/sys/devices/system/node") if re.fullmatch(r"node\d+", d)]
        )
    except OSError:
        nodes = 0
    # Both counts are required. Dividing by a missing node tree previously
    # produced "NPS0" -- a value no BIOS can hold, which then read downstream as
    # a real misconfiguration rather than an unanswerable question.
    nps = f"NPS{nodes // sockets}" if sockets and nodes else "unknown"
    return {
        "model": model,
        "generation": epyc_generation(model),
        "sockets": sockets or None,
        "numa_nodes": nodes or None,
        "nps": nps,
    }


def physical_cores() -> list[int]:
    """One online CPU id per physical core, so SMT siblings are not sampled twice."""
    seen: dict[tuple[str, str], int] = {}
    try:
        entries = sorted(
            (d for d in os.listdir("/sys/devices/system/cpu") if re.fullmatch(r"cpu\d+", d)),
            key=lambda d: int(d[3:]),
        )
    except OSError:
        return []
    for d in entries:
        cpu = int(d[3:])
        topo = f"/sys/devices/system/cpu/{d}/topology"
        pkg, core = read(f"{topo}/physical_package_id"), read(f"{topo}/core_id")
        if not pkg or not core:
            continue
        seen.setdefault((pkg, core), cpu)
    return sorted(seen.values())


def sample_cores(count: int) -> list[int]:
    """Spread ``count`` sample cores across the topology, not the first N."""
    cores = physical_cores()
    if not cores:
        return []
    if len(cores) <= count:
        return cores
    step = len(cores) / count
    return [cores[int(i * step)] for i in range(count)]


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def _spinner(seconds: int) -> subprocess.Popen:
    """A busy process used purely to raise one core's clock."""
    return subprocess.Popen(  # nosec B603 - fixed argv, no shell.
        [sys.executable, "-c", f"import time\nt=time.time()+{seconds}\nwhile time.time()<t: pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def core_freq_mhz(cpu: int) -> float | None:
    """Current frequency of one specific CPU, or ``None`` if unreadable."""
    v = read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
    if v.isdigit():
        return float(v) / 1000.0
    current = None
    for line in read("/proc/cpuinfo").splitlines():
        if line.startswith("processor"):
            try:
                current = int(line.split(":")[1])
            except (IndexError, ValueError):
                current = None
        elif line.startswith("cpu MHz") and current == cpu:
            try:
                return float(line.split(":")[1])
            except (IndexError, ValueError):
                return None
    return None


def _sleepy(times: int, gap: float):
    for i in range(times):
        if i:
            time.sleep(gap)
        yield i


def measure_peak_mhz(core: int | None = None, seconds: int = 4) -> float | None:
    """Load one core and sample *that core's* achieved frequency.

    Returns ``None`` when the measurement cannot be trusted -- affinity refused,
    or the frequency unreadable -- so the caller reports UNKNOWN rather than a
    verdict built on a number that never described the pinned core.
    """
    cores = sample_cores(1) if core is None else [core]
    if not cores:
        return None
    target = cores[0]
    proc = _spinner(seconds)
    try:
        try:
            os.sched_setaffinity(proc.pid, {target})
        except (OSError, AttributeError):
            return None
        time.sleep(1.5)
        samples = [f for f in (core_freq_mhz(target) for _ in _sleepy(4, 0.4)) if f]
        return max(samples) if samples else None
    finally:
        proc.kill()
        proc.wait()


def per_core_spread(cores: list[int] | None = None, seconds: int = 3) -> float | None:
    """Frequency spread across loaded physical cores, or ``None`` if untrustworthy.

    A partial sample cannot distinguish a uniform part from a failed
    measurement, so anything short of a reading per loaded core returns None.
    """
    cores = sample_cores(4) if cores is None else cores
    if len(cores) < 2:
        return None
    procs = []
    try:
        for c in cores:
            p = _spinner(seconds)
            procs.append(p)
            try:
                os.sched_setaffinity(p.pid, {c})
            except (OSError, AttributeError):
                return None
        time.sleep(1.5)
        vals = [core_freq_mhz(c) for c in cores]
        if any(v is None for v in vals):
            return None
        return max(vals) - min(vals)  # type: ignore[type-var]
    finally:
        for p in procs:
            p.kill()
            p.wait()


def read_hwcr() -> int | None:
    """HWCR (MSR 0xC0010015); bit 25 CpbDis gates Core Performance Boost."""
    if os.geteuid() != 0:
        return None
    try:
        with open("/dev/cpu/0/msr", "rb") as fh:
            fh.seek(0xC0010015)
            return struct.unpack("<Q", fh.read(8))[0]
    except Exception:  # noqa: BLE001 - msr module absent or not permitted.
        return None


def infer_determinism(spread: float | None) -> tuple[str | None, str]:
    """Infer determinism from per-core spread.

    Returns ``(value, note)`` where value is the normalized ``"power"`` /
    ``"performance"`` / ``None``, and note is the human-readable caveat. These
    are deliberately separate: a previous version returned the prose
    "Performance (or Power at a uniform bin)" as the *value*, which the verdict
    matcher then substring-matched against the target "power" and passed -- so
    the single case this check exists to catch reported PASS.

    Only spread is used. An earlier overshoot test compared achieved MHz against
    ``cpuinfo_max_freq``, but under ``amd_pstate`` that file *is* the boost
    ceiling, so the comparison could never fire.
    """
    if spread is None:
        return None, "no trustworthy frequency sample"
    if spread > DETERMINISM_SPREAD_MHZ:
        return "power", f"cores differ by {spread:.1f} MHz under load"
    if spread < DETERMINISM_AMBIGUOUS_MHZ:
        return "performance", (
            f"cores agree within {spread:.1f} MHz, consistent with a common "
            f"guaranteed ceiling (a uniformly binned part would look the same)"
        )
    return None, f"spread of {spread:.1f} MHz is too close to call"


def os_layer(quick: bool = False) -> dict:
    """Collect every OS-visible knob. Never raises."""
    ident = cpu_identity()
    smt_active = read("/sys/devices/system/cpu/smt/active")
    boost = read("/sys/devices/system/cpu/cpufreq/boost")
    ceiling = read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    hwcr = read_hwcr()
    cpb_msr = None if hwcr is None else ("disabled" if (hwcr >> 25) & 1 else "enabled")

    out: dict = {
        "identity": ident,
        "smt": "enabled" if smt_active == "1" else ("disabled" if smt_active == "0" else "unknown"),
        "nps": ident["nps"],
        "cpufreq_governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        or "unknown",
        "boost_sysfs": boost or "n/a",
        "boost_ceiling_mhz": round(int(ceiling) / 1000) if ceiling.isdigit() else None,
        "hwcr": f"0x{hwcr:016x}" if hwcr is not None else None,
        "quick": quick,
    }

    # The MSR is definitive when readable; sysfs is the unprivileged fallback.
    sysfs_cpb = {"1": "enabled", "0": "disabled"}.get(boost)
    out["core_performance_boost"] = cpb_msr or sysfs_cpb or "unknown"

    if quick:
        # No load generation, so nothing that needs a measurement can be judged.
        out["peak_mhz"] = None
        out["core_spread_mhz"] = None
        out["determinism"] = "unknown"
        out["determinism_note"] = "skipped in --quick (needs load generation)"
        return out

    peak = measure_peak_mhz()
    spread = per_core_spread()
    out["peak_mhz"] = round(peak) if peak is not None else None
    out["core_spread_mhz"] = round(spread, 1) if spread is not None else None
    out["sampled_cores"] = sample_cores(4)
    value, note = infer_determinism(spread)
    out["determinism"] = value or "unknown"
    out["determinism_note"] = note
    return out


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

def normalize(value: object) -> str:
    """Lower-case, whitespace-collapsed form used for every comparison."""
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip().lower()


def verdict(key: str, value: object) -> str:
    """PASS/FAIL/UNKNOWN for a checked knob.

    Exact match after normalization. Substring matching was removed: no target
    needs it, and it silently passed any value that merely *contained* a target
    word.
    """
    v = normalize(value)
    if v in ("", "unknown", "auto", "n/a", "none"):
        return "UNKNOWN"
    return "PASS" if any(normalize(t) == v for t in CHECKED[key]["target"]) else "FAIL"


#: Exit codes, identical for --json and text output so CI can key off either.
EXIT_OK, EXIT_FAIL, EXIT_UNKNOWN = 0, 1, 2


_RECORD_LABELS = {"determinism": "Determinism control", "smt": "SMT", "nps": "NPS"}


def build_rows(osl: dict) -> list[dict]:
    """One row per checked knob, plus the recorded-only entries.

    Only the two knobs with an unambiguous right answer are judged, and neither
    needs load generation -- so ``--quick`` can reach every verdict this tool
    offers and a fast run is a usable gate. Determinism is still measured and
    reported when the run is not quick; it simply does not carry a verdict.
    """
    rows = []
    for key, spec in CHECKED.items():
        value = osl.get(key, "unknown")
        rows.append(
            {
                "knob": spec["label"],
                "key": key,
                "value": str(value),
                "target": "/".join(spec["target"]),
                "verdict": verdict(key, value),
                "note": osl.get(f"{key}_note", ""),
            }
        )
    for key in RECORDED:
        rows.append(
            {
                "knob": _RECORD_LABELS.get(key, key),
                "key": key,
                "value": str(osl.get(key, "unknown")),
                "target": "",
                "verdict": "RECORD",
                "note": osl.get(f"{key}_note", "") or "recorded for comparability; not judged",
            }
        )
    return rows


def exit_code(rows: list[dict]) -> int:
    """Worst status across the *checked* knobs.

    An unresolved knob is distinguished from one that is genuinely wrong: a
    fleet sweep chases FAILs first and treats UNKNOWNs as missing coverage.
    RECORD rows never influence the exit code -- that is what makes them
    recorded rather than checked.
    """
    checked = [r for r in rows if r["verdict"] in ("PASS", "FAIL", "UNKNOWN")]
    if any(r["verdict"] == "FAIL" for r in checked):
        return EXIT_FAIL
    if any(r["verdict"] == "UNKNOWN" for r in checked):
        return EXIT_UNKNOWN
    return EXIT_OK


def render(osl: dict, rows: list[dict]) -> None:
    ident = osl["identity"]
    print(f"\nHost      : {os.uname().nodename}")
    print(f"CPU       : {ident['model']}")
    print(f"Generation: {ident['generation']}")
    print(f"Topology  : {ident['sockets'] or '?'} sockets, {ident['numa_nodes'] or '?'} NUMA nodes")
    if osl.get("peak_mhz"):
        print(f"Measured  : peak {osl['peak_mhz']} MHz, spread {osl['core_spread_mhz']} MHz")
    print()
    width = max(len(r["knob"]) for r in rows)
    for r in rows:
        target = f"  (want {r['target']})" if r["target"] and r["verdict"] != "PASS" else ""
        print(f"  {r['verdict']:<7} {r['knob']:<{width}}  {r['value']}{target}")
        if r["note"] and r["verdict"] in ("UNKNOWN", "FAIL"):
            print(f"          {' ' * width}  {r['note']}")
    print()
    fails = [r for r in rows if r["verdict"] == "FAIL"]
    unknown = [r for r in rows if r["verdict"] == "UNKNOWN"]
    if fails:
        print("Off target:")
        for r in fails:
            print(f"    {r['knob']}: {r['value']} (want {r['target']}) — {CHECKED[r['key']]['basis']}")
    if unknown:
        print("Unresolved (not a pass):")
        for r in unknown:
            print(f"    {r['knob']}: {r['note'] or 'not readable on this host'}")
    if not fails and not unknown:
        print("All checked knobs on target.")
    print(f"\nTargets follow {SOURCE}.")
    print("RECORD rows are reported, not judged: the guide varies them by workload.")
    if os.geteuid() != 0:
        print("\nNote: not root — the HWCR MSR was not read, so Core Performance Boost")
        print("falls back to the sysfs view.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit OS-visible AMD EPYC tuning that affects benchmark results",
        epilog=(
            "Exit: 0 all checked knobs on target, 1 a knob is wrong, "
            "2 a knob could not be resolved. Recorded-only knobs (determinism, "
            f"SMT, NPS) never affect the exit code. Targets follow {SOURCE}."
        ),
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="skip load generation; determinism is then recorded as unmeasured",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    osl = os_layer(quick=args.quick)
    rows = build_rows(osl)
    code = exit_code(rows)

    if args.json:
        print(json.dumps({"os": osl, "rows": rows, "exit_code": code}, indent=2, sort_keys=True))
    else:
        render(osl, rows)
    return code


if __name__ == "__main__":
    sys.exit(main())
