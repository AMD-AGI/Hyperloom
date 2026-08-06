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
    sudo python3 scripts/platform_audit.py --ca-cert /etc/ssl/bmc-ca.pem
    sudo python3 scripts/platform_audit.py --insecure   # self-signed BMC cert
    sudo python3 scripts/platform_audit.py --no-bmc     # OS layer only, no BMC contact
    BMC_PASSWORD=... python3 scripts/platform_audit.py --bmc-host H --bmc-user U --insecure
    sudo python3 scripts/platform_audit.py --json --insecure

Exit codes (identical in both output modes): 0 all pass, 1 a knob is wrong,
2 a knob is unresolved, 3 a temporary BMC account could not be revoked.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import glob
import json
import os
import re
import secrets
import socket
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


def run(cmd: list[str], timeout: int = 30, stdin: str | None = None) -> tuple[bool, str, str]:
    """Run a command, returning ``(ok, stdout, stderr)``.

    Callers that must not swallow a failure -- anything revoking BMC access --
    use this rather than :func:`sh`.
    """
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=stdin)
        return r.returncode == 0, r.stdout, r.stderr
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller as stderr
        return False, "", str(exc)


def _as_root(cmd: list[str]) -> list[str]:
    return cmd if os.geteuid() == 0 else ["sudo", "-n"] + cmd


def sudo_run(cmd: list[str], timeout: int = 30, stdin: str | None = None) -> tuple[bool, str, str]:
    return run(_as_root(cmd), timeout, stdin)


def sh(cmd: list[str], timeout: int = 30, stdin: str | None = None) -> str:
    """Run a command, returning stdout ('' on any failure)."""
    ok, out, _ = run(cmd, timeout, stdin)
    return out if ok else ""


def sudo(cmd: list[str], timeout: int = 30, stdin: str | None = None) -> str:
    return sh(_as_root(cmd), timeout, stdin)


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
    return {
        "model": model,
        "sockets": max(sockets, 1),
        "numa_nodes": nodes,
        "generation": epyc_generation(model),
    }


#: Series code -> family. Expected boost ceilings differ per generation, so the
#: target profile is keyed on this rather than on the GPU installed.
_EPYC_FAMILIES = {
    "9005": "Turin",
    "9004": "Genoa/Bergamo",
    "8004": "Siena",
    "7003": "Milan",
    "7002": "Rome",
    "7001": "Naples",
}


def epyc_generation(model: str) -> str:
    """Map an EPYC model number to its series.

    In a 4-digit EPYC number the *last* digit carries the generation and the
    first carries the series: 9575F -> 9005 (Turin), 9654 -> 9004 (Genoa).
    Reading the second digit instead happens to work for 9575F and fails for
    parts like 9755, so index the digits explicitly.
    """
    m = re.search(r"EPYC\s+(\d{4})", model)
    if not m:
        return "unknown"
    d = m.group(1)
    series = f"{d[0]}00{d[3]}"
    family = _EPYC_FAMILIES.get(series)
    return f"EPYC {series} ({family})" if family else f"EPYC {series}"


def physical_cores() -> list[int]:
    """One online CPU id per physical core, lowest id first.

    SMT siblings share a core, so sampling both halves of a pair measures the
    same silicon twice and understates the spread across the part.
    """
    first: dict[tuple[str, str], int] = {}
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*"):
        m = re.search(r"cpu(\d+)$", path)
        if not m or read(f"{path}/online") == "0":
            continue
        cpu = int(m.group(1))
        pkg = read(f"{path}/topology/physical_package_id")
        core = read(f"{path}/topology/core_id")
        if not pkg or not core:
            continue
        key = (pkg, core)
        if key not in first or cpu < first[key]:
            first[key] = cpu
    return sorted(first.values())


def sample_cores(count: int = 4) -> list[int]:
    """Pick ``count`` physical cores spread evenly across the topology.

    Hardcoding ids assumed a part with at least 69 cores; on a smaller SKU the
    affinity call fails, the sysfs reads come back empty, and a meaningless
    spread still reaches the determinism verdict.
    """
    cores = physical_cores()
    if len(cores) <= count:
        return cores
    step = len(cores) / float(count)
    return [cores[int(i * step)] for i in range(count)]


def _spinner(seconds: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", f"import time\nt=time.time()+{seconds}\nwhile time.time()<t: pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def core_freq_mhz(cpu: int) -> float | None:
    """Current frequency of one specific CPU, or ``None`` if unreadable."""
    v = read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
    if v.isdigit():
        return float(v) / 1000.0
    # cpufreq absent (common under some hypervisors): fall back to the
    # per-processor block in /proc/cpuinfo rather than a global maximum.
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


def measure_peak_mhz(core: int | None = None, seconds: int = 4) -> float | None:
    """Load one core and sample *that core's* achieved frequency.

    Resolves an "Auto" boost setting into a real answer. Returns ``None`` when
    the measurement cannot be trusted -- affinity refused, or the core's
    frequency unreadable -- so the caller reports UNKNOWN instead of a verdict
    built on a number that never described the pinned core.
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


def _sleepy(times: int, gap: float):
    """Yield ``times`` ticks, sleeping ``gap`` between them."""
    for i in range(times):
        if i:
            time.sleep(gap)
        yield i


def per_core_spread(cores: list[int] | None = None, seconds: int = 3) -> float | None:
    """Frequency spread across physical cores, or ``None`` if untrustworthy.

    Power determinism lets each part run to its own silicon limit, so cores
    differ. Performance determinism pins every core to a common guaranteed
    ceiling, collapsing the spread toward zero. A partial sample cannot tell
    those apart, so anything short of a reading per loaded core returns None.
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
        return max(vals) - min(vals)
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
    out["peak_mhz"] = round(peak) if peak is not None else None
    out["core_spread_mhz"] = round(spread, 1) if spread is not None else None
    out["sampled_cores"] = sample_cores(4)

    ceil_mhz = out["boost_ceiling_mhz"] or 0
    if peak is not None and ceil_mhz:
        boosting = peak > ceil_mhz * 0.90
        out["cpb_effective"] = "Enabled" if boosting or out["cpb_msr"] == "Enabled" else "Disabled"
    else:
        # No trustworthy measurement: the MSR still answers definitively, and
        # without it the knob stays UNKNOWN rather than guessed.
        out["cpb_effective"] = out["cpb_msr"]

    # Overshooting the advertised ceiling, or measurable per-core variance,
    # both indicate the part is running to its own limit rather than a common
    # guaranteed one. Both readings are required: a missing sample cannot
    # distinguish a uniform part from a failed measurement.
    if ceil_mhz and peak is not None and spread is not None:
        if peak > ceil_mhz or spread > 8:
            out["determinism_inferred"] = "Power"
        else:
            out["determinism_inferred"] = "Performance (or Power at a uniform bin)"
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

#: Populated when a temporary BMC account could not be fully revoked. A failure
#: here leaves a credentialed account on the BMC, so it must reach the exit code
#: rather than scroll past in the output.
REVOKE_FAILURES: list[tuple[int, str, list[str]]] = []


def account_enabled(slot: int) -> bool:
    """Whether a BMC user slot currently reports itself enabled."""
    out = sudo(["ipmitool", "channel", "getaccess", "1", str(slot)])
    m = re.search(r"Enable Status\s*:\s*(\w+)", out)
    return bool(m and m.group(1).lower() == "enabled")


def set_bmc_password(slot: int, password: str) -> bool:
    """Set a BMC account password without exposing it in the process table.

    Passing the password as an argv element would make it readable via
    /proc/<pid>/cmdline to any local user for the lifetime of the ipmitool
    process. Omitting it makes ipmitool prompt on stdin instead (twice, for
    confirmation), which keeps the secret out of argv entirely.
    """
    ok, out, err = sudo_run(["ipmitool", "user", "set", "password", str(slot)],
                            stdin=f"{password}\n{password}\n")
    return ok and "successful" in (out + err).lower()


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
        if account_enabled(slot):
            # __exit__ cannot run if the previous invocation was SIGKILLed or
            # the box rebooted mid-audit, so a sentinel found already enabled
            # is the fingerprint of exactly that. Say so: the account has been
            # reachable over LAN since then. The password is replaced below.
            print(
                f"WARNING: BMC account '{SENTINEL_USER}' (slot {slot}) was still enabled "
                f"from an earlier run that did not shut down cleanly; it has been "
                f"reachable over the LAN channel since. Re-minting and revoking now.",
                file=sys.stderr,
            )
        pw = secrets.token_urlsafe(24)[:16]
        sudo(["ipmitool", "user", "set", "name", str(slot), self.user])
        if not set_bmc_password(slot, pw):
            return self
        # Claim the slot before granting access, so a failure between here and
        # __exit__ still triggers revocation.
        self.slot = slot
        sudo(["ipmitool", "user", "priv", str(slot), "4", "1"])
        sudo(["ipmitool", "user", "enable", str(slot)])
        sudo(["ipmitool", "channel", "setaccess", "1", str(slot),
              "callin=on", "ipmi=on", "link=on", "privilege=4"])
        self.password = pw
        return self

    def __exit__(self, *exc):
        if self.slot is None:
            return False
        s = str(self.slot)
        failures: list[str] = []

        # Channel access must be revoked while the user is still enabled; BMCs
        # reject setaccess on a disabled account ("not supported in present
        # state"), which would leave ADMINISTRATOR rights in place.
        # privilege=15 ("no access") is the right end state, but some BMCs --
        # Supermicro among them -- reject it with "not supported in present
        # state". CALLBACK is the lowest they accept and still drops
        # ADMINISTRATOR, so try the strict value first and fall back.
        revoked, last_err = False, ""
        for priv in ("15", "1"):
            revoked, _, last_err = sudo_run(
                ["ipmitool", "channel", "setaccess", "1", s,
                 "callin=off", "ipmi=off", "link=off", f"privilege={priv}"]
            )
            if revoked:
                break
        if not revoked:
            failures.append(f"revoke channel access: {last_err.strip() or 'command failed'}")

        ok, _, err = sudo_run(["ipmitool", "user", "disable", s])
        if not ok:
            failures.append(f"disable account: {err.strip() or 'command failed'}")

        # Leave the password unguessable rather than merely disabled.
        if not set_bmc_password(self.slot, secrets.token_urlsafe(24)[:16]):
            failures.append("randomize password: command failed")
        sudo(["ipmitool", "user", "set", "name", s, SENTINEL_USER])
        self.password = ""

        # Trust the read-back, not the exit codes: some BMCs report success on
        # a revoke that did not take.
        if account_enabled(self.slot):
            failures.append("account still reports Enable Status: enabled after revoke")

        if failures:
            REVOKE_FAILURES.append((self.slot, self.user, failures))
            print(
                f"\n*** SECURITY: failed to revoke BMC account '{self.user}' in slot {self.slot} "
                f"on this node.\n"
                f"*** It may remain enabled with ADMINISTRATOR privilege and be reachable\n"
                f"*** over the LAN channel. Revoke it by hand:\n"
                f"***   sudo ipmitool channel setaccess 1 {self.slot} callin=off ipmi=off "
                f"link=off privilege=15\n"
                f"***   sudo ipmitool user disable {self.slot}\n"
                + "".join(f"***   - {f}\n" for f in failures),
                file=sys.stderr,
            )
        return False


def tls_context(ca_cert: str | None, insecure: bool) -> ssl.SSLContext:
    """Build the TLS context used for Redfish.

    Verification is on by default. Most BMCs ship a self-signed certificate,
    so most sites will need ``--ca-cert`` (the BMC's own or the CA that signed
    it) or an explicit ``--insecure``; silently skipping verification would
    leave the audit open to interception on the management network, which is
    not the sort of decision a tool should make on the operator's behalf.
    """
    if insecure:
        return ssl._create_unverified_context()  # nosec B323 — operator opt-in
    return ssl.create_default_context(cafile=ca_cert) if ca_cert else ssl.create_default_context()


def rf_get(host: str, path: str, user: str, password: str,
           ctx: ssl.SSLContext, timeout: int = 30) -> tuple[dict | None, str]:
    """GET a Redfish resource, returning ``(document, error)``."""
    req = urllib.request.Request(f"https://{host}{path}")
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:  # nosec B310
            return json.loads(r.read().decode()), ""
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 — reported, never swallowed
        # urlopen wraps the verification failure in URLError, so the useful
        # error is one level down; without unwrapping, the operator sees a
        # raw OpenSSL string and no idea which flag fixes it.
        cause = getattr(exc, "reason", exc)
        if isinstance(cause, ssl.SSLCertVerificationError):
            detail = getattr(cause, "verify_message", None) or cause
            return None, (
                f"TLS verification failed ({detail}). BMCs usually ship self-signed "
                f"certificates: pass --ca-cert <file> to trust this one, or --insecure "
                f"to skip verification deliberately"
            )
        return None, str(exc)


def redfish_bios(host: str, user: str, password: str,
                 ctx: ssl.SSLContext) -> tuple[dict, str, str]:
    """Fetch BIOS attributes, discovering the system path (varies by OEM)."""
    systems, err = rf_get(host, "/redfish/v1/Systems", user, password, ctx)
    if systems is None and err:
        return {}, "", err
    members = [m.get("@odata.id") for m in (systems or {}).get("Members", [])]
    last = ""
    for base in members or ["/redfish/v1/Systems/1", "/redfish/v1/Systems/System.Embedded.1"]:
        doc, err = rf_get(host, f"{base}/Bios", user, password, ctx)
        last = err or last
        if doc and doc.get("Attributes"):
            return doc["Attributes"], f"{base}/Bios", ""
    return {}, "", last or "no BIOS attributes exposed"


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


#: Exit codes, identical for --json and text output so CI can key off either.
EXIT_OK, EXIT_FAIL, EXIT_UNKNOWN, EXIT_REVOKE = 0, 1, 2, 3


def exit_code(rows: list[dict]) -> int:
    """Worst-case status across the audit.

    An unresolved knob is distinguished from a knob that is genuinely wrong:
    a fleet sweep wants to chase the FAILs first and treat UNKNOWNs as missing
    coverage. A revoke failure outranks both -- it left credentials behind.
    """
    if REVOKE_FAILURES:
        return EXIT_REVOKE
    if any(r["verdict"] == "FAIL" for r in rows):
        return EXIT_FAIL
    if any(r["verdict"] == "UNKNOWN" for r in rows):
        return EXIT_UNKNOWN
    return EXIT_OK


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit AMD EPYC platform knobs on Instinct nodes",
        epilog=(
            "Exit: 0 all pass, 1 a knob is wrong, 2 a knob is unresolved, "
            "3 a temporary BMC account could not be revoked. "
            "Password for --bmc-user comes from $BMC_PASSWORD or a prompt; "
            "it is never taken on the command line, where the process table "
            "would expose it to every local user."
        ),
    )
    ap.add_argument("--bmc-host")
    ap.add_argument("--bmc-user")
    ap.add_argument("--ca-cert", help="CA bundle (or the BMC cert) to verify Redfish TLS against")
    ap.add_argument("--insecure", action="store_true",
                    help="skip Redfish TLS verification (self-signed BMC certs)")
    ap.add_argument("--no-bmc", action="store_true", help="OS layer only")
    ap.add_argument("--quick", action="store_true", help="skip frequency measurement")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    osl = os_layer(quick=args.quick)
    bios: dict = {}
    bios_path = ""
    bmc_note = "skipped (--no-bmc)"

    if not args.no_bmc:
        ctx = tls_context(args.ca_cert, args.insecure)
        if args.insecure:
            print("WARNING: Redfish TLS verification disabled (--insecure)", file=sys.stderr)
        host = args.bmc_host or bmc_ip()
        if not host:
            bmc_note = "BMC not discoverable (no KCS/LAN)"
        elif args.bmc_user:
            password = os.environ.get("BMC_PASSWORD")
            if password is None:
                password = getpass.getpass(f"BMC password for {args.bmc_user}@{host}: ")
            attrs, bios_path, err = redfish_bios(host, args.bmc_user, password, ctx)
            bios = resolve(attrs)
            bmc_note = f"{host} (supplied credentials)" if attrs else f"{host} ({err})"
        else:
            with TempBmcAccount() as acct:
                if acct.password:
                    attrs, bios_path, err = redfish_bios(host, acct.user, acct.password, ctx)
                    bios = resolve(attrs)
                    bmc_note = (f"{host} (temporary account, revoked)" if attrs
                                else f"{host} (temp account made; {err})")
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

    result = {"host": socket.gethostname(), "identity": osl["identity"], "os": osl,
              "bmc": bmc_note, "bios_path": bios_path, "knobs": rows,
              "revoke_failures": [{"slot": s, "user": u, "errors": e}
                                  for s, u, e in REVOKE_FAILURES]}

    if args.json:
        print(json.dumps(result, indent=2))
        return exit_code(rows)

    i = osl["identity"]
    print(f"\n  {i['model']}  |  {i['sockets']} socket(s), {i['numa_nodes']} NUMA nodes  |  {i['generation']}")
    print(f"  BMC: {bmc_note}")
    if osl.get("peak_mhz") is not None:
        spread = osl.get("core_spread_mhz")
        spread_txt = f"{spread} MHz" if spread is not None else "unmeasured"
        print(f"  measured peak {osl['peak_mhz']} MHz (ceiling {osl['boost_ceiling_mhz']} MHz) "
              f"on cpu{osl.get('sampled_cores', ['?'])[0]}, per-core spread {spread_txt}, "
              f"governor {osl['governor']}")
    elif not args.quick:
        print("  frequency measurement unavailable on this host; "
              "boost/determinism fall back to MSR or report UNKNOWN")
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
    for r in unknown:
        print(f"    UNKNOWN {r['knob']}: not resolvable from BIOS or OS on this host")
    return exit_code(rows)


if __name__ == "__main__":
    sys.exit(main())
