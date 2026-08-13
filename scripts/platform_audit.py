#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Audit the AMD EPYC CPU tuning that affects inference benchmark results.

Reads only what the operating system exposes -- ``/sys``, ``/proc`` and, when
run as root, the HWCR MSR.

Judges Core Performance Boost and the cpufreq governor. Records determinism
control, SMT and nodes-per-socket without a verdict. A knob is only judged when
there is a defensible answer *and* a trustworthy way to read it here; see
``SOURCE`` below for the citation and the per-knob reasoning.

Power is still the determinism setting to want, because it maximizes what a
given platform can do. It is recorded rather than judged because the OS layer
cannot read it -- only infer it from per-core frequency spread -- and that
inference is not steady enough to gate on. The cost of Power, that platforms
then differ from each other, is paid by recording the setting: a system-to-
system delta stays explicable because the report says which mode each run
was in.

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
import textwrap
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

#: Recorded for comparability, never judged. Each entry says why there is no
#: verdict, because "we did not check this" is a claim that needs a reason.
RECORDED: dict[str, dict] = {
    "determinism": {
        "label": "Determinism control",
        "why": (
            "Power is the setting to want: 58011 §4.2.2 offers it as 'maximum "
            "performance of any individual system by leveraging the capabilities "
            "of a given CPU to the maximum', against Performance, which buys "
            "fleet uniformity by leaving headroom unused on the better parts. "
            "There is no verdict because the OS layer cannot read the setting, "
            "only infer it from per-core frequency spread, and that inference is "
            "not steady enough to gate on -- see DETERMINISM_SPREAD_MHZ. Read it "
            "from BIOS, or with platform_audit_bmc.py, when it has to be certain."
        ),
        "inferred": True,
    },
    "smt": {
        "label": "SMT",
        "why": (
            "On by default across the EPYC fleet, and chapter 5 leaves SMT "
            "Control at default in every general-purpose column. Judging it "
            "would fail nearly every node from day one."
        ),
    },
    "nps": {
        "label": "NPS",
        "why": (
            "NPS1 and NPS4 are opposite and both defensible -- NPS1 interleaves "
            "for bandwidth, NPS4 favours locality -- and chapter 5's own NUMA "
            "rows differ per workload."
        ),
    },
}

#: Frequency spread, in MHz, above which cores look like they are running to
#: their own limits rather than a common one. A heuristic, not a vendor spec.
#:
#: These thresholds classify a recorded value and nothing more. They were once
#: used for a verdict, and review showed why that could not hold: five
#: consecutive runs on one unchanged EPYC 9575F measured 7.9, 21.0, 18.1, 20.9
#: and 15.1 MHz. The host's own jitter straddles the 8.0 boundary, so a gate
#: built on a single sample flips its answer on a machine nobody touched -- and
#: a check that does that is one people learn to ignore.
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


def list_dir(path: str) -> list[str]:
    """Directory entries, or ``[]`` when the tree is absent.

    A host without ``/sys/devices/system/cpu`` -- a minimal container, usually
    -- has nothing to count rather than something to fail on. One shared helper
    is what keeps that true at every call site instead of at most of them.
    """
    try:
        return os.listdir(path)
    except OSError:
        return []


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
            for d in list_dir("/sys/devices/system/cpu")
            if re.fullmatch(r"cpu\d+", d)
        }
        - {""}
    )
    nodes = len([d for d in list_dir("/sys/devices/system/node") if re.fullmatch(r"node\d+", d)])
    # Both counts are required. Dividing into a missing node tree would yield
    # "NPS0", a value no BIOS can hold, which reads downstream as a real
    # misconfiguration rather than as an unanswerable question.
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
    entries = sorted(
        (d for d in list_dir("/sys/devices/system/cpu") if re.fullmatch(r"cpu\d+", d)),
        key=lambda d: int(d[3:]),
    )
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
    ``"performance"`` / ``None`` and note is the human-readable caveat. The two
    must stay separate: any caveat folded into the value becomes a value that
    contains the name of another setting, which is a trap for every comparison
    downstream.

    Spread is the only input. Comparing achieved MHz against ``cpuinfo_max_freq``
    cannot add anything, because under ``amd_pstate`` that file *is* the boost
    ceiling.
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
        # No load generation, so the measured knobs have nothing to report. None
        # of them is judged, so the exit code is unaffected.
        out["peak_mhz"] = None
        out["core_spread_mhz"] = None
        out["determinism"] = "unknown"
        out["determinism_note"] = "not measured in --quick (needs load generation)"
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


def build_rows(osl: dict) -> list[dict]:
    """One row per checked knob, then one per recorded knob.

    Every knob that needs load generation is recorded rather than checked, so
    ``--quick`` reaches every verdict this tool offers and needs no special
    case: a fast run and a full run return the same exit code on the same host.
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
                "inferred": bool(spec.get("inferred")),
            }
        )
    for key, spec in RECORDED.items():
        rows.append(
            {
                "knob": spec["label"],
                "key": key,
                "value": str(osl.get(key, "unknown")),
                "target": "",
                "verdict": "RECORD",
                "note": osl.get(f"{key}_note", ""),
                "inferred": bool(spec.get("inferred")),
                # Carried into --json so the record explains its own silence to
                # whoever reads it later, without the reader needing this file.
                "why": spec["why"],
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
        # An inferred row always shows its note: the value is a deduction, and a
        # reader who cannot see that will treat it as a reading.
        if r["note"] and (r["verdict"] in ("UNKNOWN", "FAIL") or r["inferred"]):
            prefix = "inferred: " if r["inferred"] else ""
            print(f"          {' ' * width}  {prefix}{r['note']}")
    print()
    fails = [r for r in rows if r["verdict"] == "FAIL"]
    unknown = [r for r in rows if r["verdict"] == "UNKNOWN"]
    if fails:
        print("Off target:")
        for r in fails:
            print(f"    {r['knob']}: {r['value']} (want {r['target']})")
            print(textwrap.fill(CHECKED[r["key"]]["basis"], width=76,
                                initial_indent="        ", subsequent_indent="        "))
    if unknown:
        print("Unresolved (not a pass):")
        for r in unknown:
            print(f"    {r['knob']}: {r['note'] or 'not readable on this host'}")
    if not fails and not unknown:
        print("All checked knobs on target.")
    print(f"\nTargets follow {SOURCE}.")
    print("RECORD rows are reported, not judged; run with --json for the reason.")
    if os.geteuid() != 0:
        print("\nNote: not root — the HWCR MSR was not read, so Core Performance Boost")
        print("falls back to the sysfs view.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit OS-visible AMD EPYC tuning that affects benchmark results",
        epilog=(
            "Exit: 0 all checked knobs on target, 1 a knob is wrong, "
            "2 a knob could not be resolved. Checked knobs are Core Performance "
            "Boost and the cpufreq governor; determinism, SMT and NPS are "
            f"recorded only and never affect the exit code. Targets follow {SOURCE}."
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
