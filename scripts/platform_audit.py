#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Audit AMD EPYC platform tuning knobs on Instinct (MI3xx) nodes.

Two independent layers, because neither is sufficient alone:

  OS layer      sysfs + MSR + a live frequency measurement. Works on any x86
                AMD host regardless of OEM, BMC, or GPU. Needs no credentials.
  Redfish layer the BIOS attribute dictionary from the BMC. Reaches knobs the
                OS cannot see at all (APBDIS, DF C-states) but reports vendor
                names, and frequently answers "Auto" -- which is not a value.

The layers are complementary. Redfish saying ``CorePerformanceBoost = Auto``
tells you nothing on its own; the OS layer measuring 5047 MHz against a 5.0 GHz
part proves boost is live. Conversely no sysfs or MSR path exposes APBDIS on
kernels without the amd_hsmp driver (upstreamed in 5.18). So the audit reports
a BIOS value, an effective value, and reconciles the two.

Attribute names are OEM-defined -- Redfish standardizes the container, not its
contents. On AMD the names largely derive from AGESA, so Supermicro, Quanta,
Gigabyte and AMD reference designs converge; Dell ("LogicalProc",
"DeterminismSlider") and Lenovo ("Processors.SMTMode") diverge more. Matching is
therefore synonym-based rather than a per-OEM table, so it degrades sensibly on
hardware it has never seen.

Usage:
    sudo python3 scripts/platform_audit.py            # OS layer + Redfish via minted BMC account
    sudo python3 scripts/platform_audit.py --no-bmc   # OS layer only, no BMC contact
    python3 scripts/platform_audit.py --bmc-host H --bmc-user U --bmc-pass P
    sudo python3 scripts/platform_audit.py --json     # machine-readable; exits non-zero on any FAIL
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import secrets
import ssl
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Canonical knob -> (regex over attribute names, target value(s)).
# Patterns are unanchored so vendor prefixes like "Processors.SMTMode" match.
KNOBS: dict[str, dict] = {
    "power_profile": {
        "label": "High Performance mode",
        "pattern": r"power.?profile|system.?profile|perf.*profile.?sel",
        "target": ["high performance mode", "high performance", "max performance"],
    },
    "core_performance_boost": {
        "label": "Core Performance Boost",
        "pattern": r"core.?perf\w*.?boost|\bcpb(?:mode|enable|control)?\b|core.?boost",
        "target": ["enabled", "auto"],
    },
    "determinism": {
        "label": "Determinism",
        "pattern": r"determinism(enable|slider|control)?",
        "target": ["power"],
    },
    "nps": {
        "label": "Nodes Per Socket",
        "pattern": r"numa.?nodes.?per.?socket|nodes.?per.?socket|\bnps(?:mode|setting)?\b",
        "target": ["nps1", "nps4"],
    },
    "apbdis": {
        "label": "APBDIS",
        "pattern": r"\bapbdis\b|apb.?dis",
        "target": ["1"],
    },
    "df_cstates": {
        "label": "DF C-states",
        "pattern": r"df.?c.?states?|data.?fabric.?c.?state",
        "target": ["disabled"],
    },
    "smt": {
        "label": "SMT Control",
        # \b sits between a word and non-word char, so a bare \bsmt\b silently
        # fails on the concatenated names vendors actually use (SMTControl,
        # Processors.SMTMode). The suffixes must be part of the alternation.
        "pattern": r"\bsmt(?:control|mode|enable|status)?\b|logical.?proc|hyper.?threading",
        "target": ["disabled"],
    },
}

# Attributes whose names collide with a knob pattern but mean something else.
NAME_BLOCKLIST = re.compile(r"sev|snp.?support|encrypt", re.I)


def sh(cmd: list[str], timeout: int = 30, stdin: str | None = None) -> str:
    """Run a command, returning stdout ('' on any failure)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=stdin)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def sudo(cmd: list[str], timeout: int = 30, stdin: str | None = None) -> str:
    if os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    return sh(cmd, timeout, stdin)


def read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------
# OS layer
# --------------------------------------------------------------------------

def cpu_identity() -> dict:
    model = ""
    for line in read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    sockets = len({read(p) for p in glob.glob(
        "/sys/devices/system/cpu/cpu*/topology/physical_package_id")} - {""})
    nodes = len(glob.glob("/sys/devices/system/node/node[0-9]*"))
    # EPYC 9005 = Turin, 9004 = Genoa/Bergamo. Governs expected boost ceiling.
    gen = "unknown"
    m = re.search(r"EPYC\s+(\d)(\d)", model)
    if m:
        gen = {"90": "Turin/Genoa"}.get(m.group(1) + "0", "unknown")
        if m.group(1) == "9":
            gen = "EPYC 9005 (Turin)" if m.group(2) == "5" else "EPYC 9004 (Genoa/Bergamo)"
    return {"model": model, "sockets": max(sockets, 1), "numa_nodes": nodes, "generation": gen}


def measure_peak_mhz(core: int = 4, seconds: int = 4) -> float:
    """Pin a spinner to one core and sample its achieved frequency.

    This is what resolves an "Auto" boost setting into a real answer.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import time\nt=time.time()+{seconds}\nwhile time.time()<t: pass"]
    )
    try:
        sh(["taskset", "-pc", str(core), str(proc.pid)])
        time.sleep(1.5)
        peak = 0.0
        for _ in range(4):
            freqs = [float(l.split(":")[1]) for l in read("/proc/cpuinfo").splitlines()
                     if l.startswith("cpu MHz")]
            if freqs:
                peak = max(peak, max(freqs))
            time.sleep(0.4)
        return peak
    finally:
        proc.kill()
        proc.wait()


def per_core_spread(cores=(4, 20, 40, 68), seconds: int = 3) -> float:
    """Frequency spread across cores.

    Power determinism lets each part run to its own silicon limit, so cores
    differ. Performance determinism pins every core to a common guaranteed
    ceiling, collapsing the spread toward zero.
    """
    procs = []
    for c in cores:
        p = subprocess.Popen(
            [sys.executable, "-c", f"import time\nt=time.time()+{seconds}\nwhile time.time()<t: pass"]
        )
        sh(["taskset", "-pc", str(c), str(p.pid)])
        procs.append(p)
    try:
        time.sleep(1.5)
        vals = []
        for c in cores:
            f = read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq")
            if f:
                vals.append(float(f) / 1000.0)
        if not vals:
            lines = [float(l.split(":")[1]) for l in read("/proc/cpuinfo").splitlines()
                     if l.startswith("cpu MHz")]
            vals = sorted(lines)[-len(cores):]
        return (max(vals) - min(vals)) if vals else 0.0
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
    except Exception:
        return None


def os_layer(quick: bool = False) -> dict:
    ident = cpu_identity()
    smt_ctl = read("/sys/devices/system/cpu/smt/control")
    smt_active = read("/sys/devices/system/cpu/smt/active")
    boost = read("/sys/devices/system/cpu/cpufreq/boost")
    ceiling = read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    hwcr = read_hwcr()

    out = {
        "identity": ident,
        "smt_effective": "Enabled" if smt_active == "1" else "Disabled",
        "smt_control": smt_ctl or "n/a",
        "nps_effective": f"NPS{ident['numa_nodes'] // ident['sockets']}" if ident["sockets"] else "n/a",
        "boost_sysfs": boost or "n/a",
        "boost_ceiling_mhz": round(int(ceiling) / 1000) if ceiling.isdigit() else None,
        "hwcr": f"0x{hwcr:016x}" if hwcr is not None else None,
        "cpb_msr": (None if hwcr is None else ("Disabled" if (hwcr >> 25) & 1 else "Enabled")),
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or "n/a",
    }

    if quick:
        out["peak_mhz"] = None
        out["core_spread_mhz"] = None
        out["cpb_effective"] = out["cpb_msr"] or ("Enabled" if boost == "1" else "Disabled")
        out["determinism_inferred"] = None
        return out

    peak = measure_peak_mhz()
    spread = per_core_spread()
    out["peak_mhz"] = round(peak)
    out["core_spread_mhz"] = round(spread, 1)

    ceil_mhz = out["boost_ceiling_mhz"] or 0
    boosting = bool(ceil_mhz and peak > ceil_mhz * 0.90)
    out["cpb_effective"] = "Enabled" if boosting or out["cpb_msr"] == "Enabled" else "Disabled"

    # Overshooting the advertised ceiling, or measurable per-core variance,
    # both indicate the part is running to its own limit rather than a
    # common guaranteed one.
    if ceil_mhz:
        if peak > ceil_mhz or spread > 8:
            out["determinism_inferred"] = "Power"
        elif peak > 0:
            out["determinism_inferred"] = "Performance (or Power at a uniform bin)"
        else:
            out["determinism_inferred"] = None
    else:
        out["determinism_inferred"] = None
    return out


# --------------------------------------------------------------------------
# BMC / Redfish layer
# --------------------------------------------------------------------------

def bmc_ip() -> str:
    """Discover the BMC address over the local KCS interface."""
    for ch in ("1", "8", "2"):
        out = sudo(["ipmitool", "lan", "print", ch])
        m = re.search(r"^IP Address\s+:\s+([0-9.]+)", out, re.M)
        if m and m.group(1) != "0.0.0.0":
            return m.group(1)
    return ""


SENTINEL_USER = "rfaudit"


def set_bmc_password(slot: int, password: str) -> bool:
    """Set a BMC account password without exposing it in the process table.

    Passing the password as an argv element would make it readable via
    /proc/<pid>/cmdline to any local user for the lifetime of the ipmitool
    process. Omitting it makes ipmitool prompt on stdin instead (twice, for
    confirmation), which keeps the secret out of argv entirely.
    """
    out = sudo(["ipmitool", "user", "set", "password", str(slot)],
               stdin=f"{password}\n{password}\n")
    return "successful" in out.lower()


def bmc_users() -> dict[int, str]:
    """Parse `ipmitool user list` into {slot: name}; empty name means free."""
    users = {}
    for line in sudo(["ipmitool", "user", "list", "1"]).splitlines():
        f = line.split()
        if not f or not f[0].isdigit():
            continue
        # A row for an unnamed slot runs straight from the ID into the
        # boolean columns, so a boolean in field 1 means the name is blank.
        name = "" if len(f) < 2 or f[1] in ("true", "false") else f[1]
        users[int(f[0])] = name
    return users


class TempBmcAccount:
    """Enable a BMC account over KCS for one audit, then revoke it.

    Root on the host already carries full BMC authority through KCS, so this
    is not an escalation -- it is the ordinary way to reach Redfish when no
    credentials are on file.

    The same sentinel slot is reused across runs. Supermicro (and others)
    refuse to blank a user name once set, so allocating a fresh slot per run
    would slowly fill the user table with dead accounts. One slot, disabled
    and re-randomized between runs, keeps the audit idempotent.
    """

    def __init__(self):
        self.slot: int | None = None
        self.user = SENTINEL_USER
        self.password = ""

    def _slot(self) -> int | None:
        users = bmc_users()
        if not users:
            return None
        for slot, name in sorted(users.items()):
            if name == SENTINEL_USER:
                return slot
        for slot, name in sorted(users.items()):
            if slot > 2 and not name:
                return slot
        return None

    def __enter__(self):
        slot = self._slot()
        if slot is None:
            return self
        pw = secrets.token_urlsafe(24)[:16]
        sudo(["ipmitool", "user", "set", "name", str(slot), self.user])
        if not set_bmc_password(slot, pw):
            return self
        sudo(["ipmitool", "user", "priv", str(slot), "4", "1"])
        sudo(["ipmitool", "user", "enable", str(slot)])
        sudo(["ipmitool", "channel", "setaccess", "1", str(slot),
              "callin=on", "ipmi=on", "link=on", "privilege=4"])
        self.slot, self.password = slot, pw
        return self

    def __exit__(self, *exc):
        if self.slot is None:
            return False
        s = str(self.slot)
        # Channel access must be revoked while the user is still enabled;
        # BMCs reject setaccess on a disabled account ("not supported in
        # present state"), which would silently leave ADMINISTRATOR rights.
        sudo(["ipmitool", "channel", "setaccess", "1", s,
              "callin=off", "ipmi=off", "link=off", "privilege=15"])
        sudo(["ipmitool", "user", "disable", s])
        # Leave the password unguessable rather than merely disabled.
        set_bmc_password(self.slot, secrets.token_urlsafe(24)[:16])
        sudo(["ipmitool", "user", "set", "name", s, SENTINEL_USER])
        self.password = ""
        return False


def rf_get(host: str, path: str, user: str, password: str, timeout: int = 30):
    url = f"https://{host}{path}"
    req = urllib.request.Request(url)
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    req.add_header("Authorization", f"Basic {token}")
    ctx = ssl._create_unverified_context()  # BMCs ship self-signed certs
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def redfish_bios(host: str, user: str, password: str) -> tuple[dict, str]:
    """Fetch BIOS attributes, discovering the system path (varies by OEM)."""
    systems = rf_get(host, "/redfish/v1/Systems", user, password)
    members = [m.get("@odata.id") for m in (systems or {}).get("Members", [])]
    for base in members or ["/redfish/v1/Systems/1", "/redfish/v1/Systems/System.Embedded.1"]:
        doc = rf_get(host, f"{base}/Bios", user, password)
        if doc and doc.get("Attributes"):
            return doc["Attributes"], f"{base}/Bios"
    return {}, ""


def resolve(attrs: dict) -> dict:
    """Map vendor attribute names onto canonical knobs."""
    found = {}
    for key, spec in KNOBS.items():
        rx = re.compile(spec["pattern"], re.I)
        hits = {k: v for k, v in attrs.items()
                if rx.search(k) and not NAME_BLOCKLIST.search(k)}
        if not hits:
            continue
        # Prefer the shortest name: "SMTControl" over "SMTControlPolicyOverride".
        name = sorted(hits, key=lambda k: (len(k), k))[0]
        found[key] = {"attribute": name, "value": str(hits[name]),
                      "all_matches": {k: str(v) for k, v in hits.items()}}
    return found


# --------------------------------------------------------------------------
# Reconciliation + report
# --------------------------------------------------------------------------

def effective(key: str, bios: dict | None, osl: dict) -> tuple[str, str]:
    """Return (value, source), resolving "Auto" with measured OS state."""
    bios_val = (bios or {}).get("value")
    ambiguous = bios_val is None or str(bios_val).strip().lower() in ("auto", "", "default")

    os_map = {
        "smt": osl.get("smt_effective"),
        "nps": osl.get("nps_effective"),
        "core_performance_boost": osl.get("cpb_effective"),
        "determinism": osl.get("determinism_inferred"),
    }
    if not ambiguous:
        return str(bios_val), "redfish"
    if os_map.get(key):
        src = "measured" if key in ("core_performance_boost", "determinism") else "os"
        if bios_val:
            src = f"redfish={bios_val} -> resolved by {src}"
        return str(os_map[key]), src
    return (str(bios_val) if bios_val else "unknown"), ("redfish" if bios_val else "unavailable")


def verdict(key: str, value: str) -> str:
    targets = KNOBS[key]["target"]
    v = value.strip().lower()
    if v in ("unknown",):
        return "UNKNOWN"
    # An explicit "Auto" that we could not resolve is not a pass.
    if v == "auto":
        return "UNKNOWN"
    return "PASS" if any(t == v or t in v for t in targets) else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit AMD EPYC platform knobs on Instinct nodes")
    ap.add_argument("--bmc-host")
    ap.add_argument("--bmc-user")
    ap.add_argument("--bmc-pass")
    ap.add_argument("--no-bmc", action="store_true", help="OS layer only")
    ap.add_argument("--quick", action="store_true", help="skip frequency measurement")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    osl = os_layer(quick=args.quick)
    bios: dict = {}
    bios_path = ""
    bmc_note = "skipped (--no-bmc)"

    if not args.no_bmc:
        host = args.bmc_host or bmc_ip()
        if not host:
            bmc_note = "BMC not discoverable (no KCS/LAN)"
        elif args.bmc_user:
            attrs, bios_path = redfish_bios(host, args.bmc_user, args.bmc_pass or "")
            bios, bmc_note = resolve(attrs), (f"{host} (supplied credentials)" if attrs
                                              else f"{host} (auth failed)")
        else:
            with TempBmcAccount() as acct:
                if acct.password:
                    attrs, bios_path = redfish_bios(host, acct.user, acct.password)
                    bios = resolve(attrs)
                    bmc_note = (f"{host} (temporary account, revoked)" if attrs
                                else f"{host} (temp account made, Redfish unavailable)")
                else:
                    bmc_note = f"{host} (could not mint account; supply --bmc-user)"

    rows = []
    for key, spec in KNOBS.items():
        val, src = effective(key, bios.get(key), osl)
        rows.append({
            "knob": spec["label"],
            "target": "/".join(spec["target"][:2]),
            "bios_attribute": (bios.get(key) or {}).get("attribute", "-"),
            "bios_value": (bios.get(key) or {}).get("value", "-"),
            "effective": val,
            "source": src,
            "verdict": verdict(key, val),
        })

    result = {"identity": osl["identity"], "os": osl, "bmc": bmc_note,
              "bios_path": bios_path, "knobs": rows}

    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if all(r["verdict"] == "PASS" for r in rows) else 1

    i = osl["identity"]
    print(f"\n  {i['model']}  |  {i['sockets']} socket(s), {i['numa_nodes']} NUMA nodes  |  {i['generation']}")
    print(f"  BMC: {bmc_note}")
    if osl.get("peak_mhz"):
        print(f"  measured peak {osl['peak_mhz']} MHz (ceiling {osl['boost_ceiling_mhz']} MHz), "
              f"per-core spread {osl['core_spread_mhz']} MHz, governor {osl['governor']}")
    print()
    w = max(len(r["knob"]) for r in rows)
    print(f"  {'KNOB'.ljust(w)}  {'TARGET':<22} {'EFFECTIVE':<22} {'VERDICT':<8} SOURCE")
    print(f"  {'-' * w}  {'-' * 22} {'-' * 22} {'-' * 8} ------")
    for r in rows:
        print(f"  {r['knob'].ljust(w)}  {r['target'][:22]:<22} {r['effective'][:22]:<22} "
              f"{r['verdict']:<8} {r['source']}")
    fails = [r for r in rows if r["verdict"] == "FAIL"]
    unknown = [r for r in rows if r["verdict"] == "UNKNOWN"]
    print(f"\n  {len(rows) - len(fails) - len(unknown)} pass, {len(fails)} fail, {len(unknown)} unknown")
    for r in fails:
        print(f"    FAIL {r['knob']}: {r['effective']} (want {r['target']})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
