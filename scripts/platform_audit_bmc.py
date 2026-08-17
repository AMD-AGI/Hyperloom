#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Audit the BIOS-only EPYC tuning knobs over Redfish on the BMC.

Extends ``platform_audit.py`` to the three knobs the operating system cannot
see on kernels without ``amd_hsmp``: the platform High Performance profile,
APBDIS, and DF C-states. It also reconciles a BIOS answer of "Auto" against the
measured OS state, which is the only way those answers become meaningful.

.. warning::

   **This tool can create a temporary privileged account on the BMC.** Given
   ``--allow-account-creation`` and no ``--bmc-user``, it mints a sentinel
   ADMINISTRATOR account over the in-band KCS channel, uses it for a handful of
   HTTPS GETs, then revokes it and verifies the revocation. Root on the host
   already carries full BMC authority through KCS, so this is not an escalation
   -- but many sites prohibit creating service-processor accounts outright, so
   the flag is required rather than assumed: with neither option the audit
   refuses to start. Pass ``--bmc-user`` with an existing read-only account to
   avoid minting entirely.

Failure philosophy: every step that grants access is checked, and the read-back
that confirms revocation distinguishes "read it, disabled" from "could not read
it". A tool that cannot verify it cleaned up must say so, not report success.

Exit codes:

    0  every checked knob is on target
    1  a knob is definitively wrong
    2  a knob could not be resolved
    3  a BMC account was left enabled, or its state could not be confirmed
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import re
import signal
import ssl
import subprocess  # nosec B404 - fixed argv, no shell.
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence

SENTINEL_USER = "hlaudit"

#: AMD, "BIOS & Workload Tuning Guide for AMD EPYC 9004 Series Processors",
#: publication 58011 rev 1.0 (2025-05-09).
#: https://docs.amd.com/v/u/en-US/58011-epyc-9004-tg-bios-and-workload
#:
#: Unlike the OS-layer knobs, all three of these survive contact with chapter 5:
#: each is either the documented default or is the value chapter 5 selects for
#: its latency-sensitive columns, which is the closest published analogue to an
#: inference serving host. The guide covers 9004 (Genoa); the knobs and their
#: semantics carry forward to 9005 (Turin).
SOURCE = (
    "AMD BIOS & Workload Tuning Guide for EPYC 9004 (pub. 58011 rev 1.0), ch. 5"
)

#: BIOS knobs, matched by vendor attribute name. Names differ per OEM, so each
#: is a synonym pattern rather than a per-vendor table.
BIOS_KNOBS: dict[str, dict] = {
    "power_profile": {
        "label": "High Performance mode",
        "pattern": r"power.?profile|system.?profile|perf.*profile.?sel",
        "target": ("high performance mode", "high performance", "max performance"),
        "basis": (
            "58011 §4.2.1: High Performance mode is option 0 and the documented "
            "default; chapter 5 selects it for the CPU-intensive, Java throughput "
            "and Java latency profiles, and departs from it only for the explicit "
            "Power Efficiency column."
        ),
    },
    "apbdis": {
        "label": "APBDIS",
        "pattern": r"\bapbdis\b|apb.?dis",
        "target": ("1",),
        "basis": (
            "58011 §4.4.3: setting APBDIS to 1 with a fixed Infinity Fabric "
            "P-state 'forces the AMD Infinity Fabric and memory controllers into "
            "full-power mode and significantly reduces latency jitters'. Chapter 5 "
            "selects 1 for the NIC-latency, database and HPC columns. Serving is "
            "latency-sensitive with bursty arrivals, so it matches those."
        ),
    },
    "df_cstates": {
        "label": "DF C-states",
        "pattern": r"df.?c.?states?|data.?fabric.?c.?state",
        "target": ("disabled",),
        "basis": (
            "58011 §4.4.4: the fabric takes a delay returning to full power that "
            "'causes some latency jitter', and disabling it 'for workloads "
            "requiring low latency and/or bursty I/O will increase performance' "
            "at the cost of power. Chapter 5 disables it for the EDA column."
        ),
    },
}

#: Attribute names that collide with a knob pattern but mean something else.
#:
#: ``sev`` needs care in both directions. Unanchored it matched any name merely
#: containing those letters, so "SeverityLevel" was dropped. A plain ``\bsev\b``
#: over-corrects the other way: BIOS attributes are CamelCase, so "SevControl"
#: has no trailing word boundary and would stop being blocked. Matching "sev"
#: not followed by a lower-case letter keeps the CamelCase and SCREAMING_CASE
#: forms while letting "Severity" through. The lookahead needs its own
#: case-sensitive scope, since under re.I a bare ``[a-z]`` also matches "C".
NAME_BLOCKLIST = re.compile(r"\bsev(?!(?-i:[a-z]))|snp.?support|encrypt", re.I)


def run(cmd: list[str], timeout: int = 30, stdin: str | None = None) -> tuple[bool, str, str]:
    """Run a command, returning ``(ok, stdout, stderr)``. Never raises."""
    try:
        r = subprocess.run(  # nosec B603 - fixed argv, no shell.
            cmd, capture_output=True, text=True, timeout=timeout, input=stdin
        )
        return r.returncode == 0, r.stdout, r.stderr
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as stderr
        return False, "", str(exc)


def sudo_run(cmd: list[str], **kw) -> tuple[bool, str, str]:
    """Checked privileged run. Every access-granting step uses this."""
    return run(cmd if os.geteuid() == 0 else ["sudo", "-n", *cmd], **kw)


def ipmitool_present() -> bool:
    """Whether ipmitool is callable, so a missing tool is not read as a missing BMC."""
    return run(["ipmitool", "-V"], timeout=10)[0]


def bmc_ip() -> str:
    """Discover the BMC address over the local KCS interface."""
    for ch in ("1", "8", "2"):
        ok, out, _ = sudo_run(["ipmitool", "lan", "print", ch])
        if not ok:
            continue
        m = re.search(r"^IP Address\s+:\s+([0-9.]+)", out, re.M)
        # nosec B104 - not a bind address; this rejects the unconfigured-LAN
        # sentinel that ipmitool reports for a channel with no IP assigned.
        if m and m.group(1) != "0.0.0.0":  # nosec B104
            return m.group(1)
    return ""


def bmc_users() -> dict[int, str]:
    """Parse ``ipmitool user list`` into ``{slot: name}``; empty name means free."""
    users: dict[int, str] = {}
    ok, out, _ = sudo_run(["ipmitool", "user", "list", "1"])
    if not ok:
        return users
    for line in out.splitlines():
        f = line.split()
        if not f or not f[0].isdigit():
            continue
        # An unnamed slot runs straight from the ID into the boolean columns,
        # so a boolean in field 1 means the name is blank.
        name = "" if len(f) < 2 or f[1] in ("true", "false") else f[1]
        users[int(f[0])] = name
    return users


def account_status(slot: int) -> tuple[str | None, str]:
    """Return ``(status, detail)`` where status is ``"enabled"``/``"disabled"``/``None``.

    ``None`` means the state could not be read, which is deliberately *not* the
    same answer as "disabled". This read-back is the trust anchor for the whole
    revocation: collapsing an unreadable state into "not enabled" would hand
    back a false confirmation for any BMC that errors, times out, or formats the
    output differently -- exactly where confirmation matters most.
    """
    ok, out, err = sudo_run(["ipmitool", "channel", "getaccess", "1", str(slot)])
    if not ok:
        return None, err.strip() or "getaccess failed"
    m = re.search(r"Enable Status\s*:\s*(\w+)", out)
    if not m:
        return None, "could not parse 'Enable Status' from getaccess output"
    return m.group(1).lower(), ""


def set_bmc_password(slot: int, password: str) -> tuple[bool, bool]:
    """Set a BMC account password without exposing it in the process table.

    Passing the password in argv would make it readable through
    ``/proc/<pid>/cmdline`` to any local user. Omitting it makes ipmitool prompt
    on stdin instead (twice, for confirmation), keeping it out of argv entirely.

    Returns ``(ok, confirmed)``. The return code is ``ok`` and the primary
    signal; ``confirmed`` adds whether ipmitool printed its success string,
    which is corroboration only -- the wording varies across versions and is not
    stable under a non-English locale, so it is never the sole criterion.

    Deliberately returns no text. The secret is on this command's stdin and
    ipmitool echoes its input on some failures, while callers print their
    failures to the console and store them among the account's credential
    failures -- so handing back stderr here could publish a live password. Callers
    own the wording. Diagnosing a genuinely broken BMC means running ipmitool by
    hand, which is a fair price for keeping the secret inside the one function
    that handles it.
    """
    ok, out, err = sudo_run(
        ["ipmitool", "user", "set", "password", str(slot)],
        stdin=f"{password}\n{password}\n",
    )
    return ok, ok and "successful" in (out + err).lower()


class TempBmcAccount:
    """Mint a sentinel BMC account over KCS for one audit, then revoke it.

    The same sentinel slot is reused across runs: Supermicro and others refuse
    to blank a user name once set, so a fresh slot per run would slowly fill the
    user table with dead accounts.
    """

    def __init__(self, secret_len: int = 16):
        self.slot: int | None = None
        self.user = SENTINEL_USER
        self.password = ""
        self.note = ""
        self.mint_errors: list[str] = []
        #: Filled when the account could not be verifiably revoked, or was found
        #: already enabled from a run that did not shut down cleanly. Either way
        #: credentials were exposed for longer than this tool intended, which
        #: outranks any knob verdict. Kept on the instance rather than in a
        #: module global so ``main()`` reads it after ``__exit__`` has run --
        #: which is the only moment the list is complete.
        self.credential_failures: list[str] = []
        self._secret_len = secret_len

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
        import secrets

        slot = self._slot()
        if slot is None:
            self.note = "no reusable BMC user slot"
            return self

        # Claim the slot BEFORE any mutation, so __exit__ always runs the revoke
        # path. Revoking a slot that was never enabled is idempotent and costs
        # nothing; skipping revocation on one that was leaves a live
        # ADMINISTRATOR account.
        self.slot = slot

        status, detail = account_status(slot)
        if status is None:
            self.mint_errors.append(f"could not read initial account state ({detail})")
        elif status == "enabled":
            # __exit__ cannot run if a previous invocation was SIGKILLed or the
            # box rebooted mid-audit, so a sentinel found enabled is the
            # fingerprint of exactly that: the account has been reachable over
            # LAN since then. That is a credential exposure in its own right and
            # is recorded, not merely printed.
            msg = (
                f"BMC account '{SENTINEL_USER}' (slot {slot}) was already enabled from an "
                f"earlier run that did not shut down cleanly; it has been reachable over "
                f"the LAN channel since then"
            )
            self.credential_failures.append(msg)
            print(f"WARNING: {msg}. Re-minting and revoking now.", file=sys.stderr)

        pw = secrets.token_urlsafe(24)[: self._secret_len]

        # Abort on a failed name step, exactly as the password step does. Every
        # step after this one grants access, so running them on a slot whose
        # name is not ours enables an ADMINISTRATOR account that Redfish then
        # 401s against -- pointing the operator at authentication when the real
        # cause is a slot the audit never owned.
        ok, _, err = sudo_run(["ipmitool", "user", "set", "name", str(slot), self.user])
        if not ok:
            self.mint_errors.append(f"set account name: {err.strip() or 'command failed'}")
            self.note = "; ".join(self.mint_errors)
            return self

        ok, confirmed = set_bmc_password(slot, pw)
        if not ok:
            self.mint_errors.append("set password: ipmitool rejected the password set")
            self.note = "; ".join(self.mint_errors)
            return self
        if not confirmed:
            self.mint_errors.append(
                "set password: return code reported success but ipmitool printed "
                "no confirmation"
            )

        # Every remaining step grants access, so each is checked. A silent
        # failure here surfaces as a 401 from Redfish, which reads as an
        # authentication or OEM-compatibility problem when the real cause is an
        # account that was never enabled.
        for what, cmd in (
            ("grant privilege", ["ipmitool", "user", "priv", str(slot), "4", "1"]),
            ("enable account", ["ipmitool", "user", "enable", str(slot)]),
            (
                "grant channel access",
                ["ipmitool", "channel", "setaccess", "1", str(slot),
                 "callin=on", "ipmi=on", "link=on", "privilege=4"],
            ),
        ):
            ok, _, err = sudo_run(cmd)
            if not ok:
                self.mint_errors.append(f"{what}: {err.strip() or 'command failed'}")

        self.password = pw
        self.note = "; ".join(self.mint_errors)
        return self

    def __exit__(self, *exc):
        import secrets

        if self.slot is None:
            return False
        s = str(self.slot)
        failures: list[str] = []

        # Channel access must be revoked while the user is still enabled; BMCs
        # reject setaccess on a disabled account ("not supported in present
        # state"), which would leave ADMINISTRATOR rights in place.
        # privilege=15 ("no access") is the right end state, but Supermicro
        # among others rejects it; CALLBACK is the lowest they accept and still
        # drops ADMINISTRATOR, so try the strict value first and fall back.
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
        ok, _ = set_bmc_password(self.slot, secrets.token_urlsafe(24)[: self._secret_len])
        if not ok:
            failures.append("randomize password: ipmitool rejected the new password")
        self.password = ""

        status, detail = account_status(self.slot)
        if status is None:
            failures.append(f"could not confirm final account state ({detail})")
        elif status == "enabled":
            failures.append("account still reports Enable Status: enabled after revoke")

        if failures:
            self.credential_failures.extend(failures)
            print(
                f"\n*** SECURITY: could not verify revocation of BMC account "
                f"'{self.user}' in slot {self.slot} on this node.\n"
                f"*** It may remain enabled with ADMINISTRATOR privilege and be "
                f"reachable over the LAN channel. Revoke it by hand:\n"
                f"***   sudo ipmitool channel setaccess 1 {self.slot} callin=off "
                f"ipmi=off link=off privilege=15\n"
                f"***   sudo ipmitool user disable {self.slot}\n"
                + "".join(f"***   - {f}\n" for f in failures),
                file=sys.stderr,
            )
        return False


def install_signal_handlers() -> None:
    """Turn SIGTERM/SIGINT into an exception so ``with`` blocks still unwind.

    Without this the account outlives the process on the most ordinary
    termination there is. SIGKILL and a power cut remain uncoverable -- which is
    why a sentinel found already enabled is treated as evidence of exposure --
    but SIGTERM is what a CI timeout, a job scheduler, or an orchestration layer
    actually sends, and it is recoverable.
    """

    def _raise(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _raise)
        except (ValueError, OSError):  # not on the main thread
            pass


def tls_context(ca_cert: str | None, insecure: bool) -> ssl.SSLContext:
    """Build the TLS context used for Redfish. Verification is on by default."""
    if insecure:
        return ssl._create_unverified_context()  # nosec B323 - operator opt-in
    return ssl.create_default_context(cafile=ca_cert) if ca_cert else ssl.create_default_context()


def rf_get(host: str, path: str, user: str, password: str, ctx: ssl.SSLContext,
           timeout: int = 30) -> tuple[dict | None, str]:
    """GET a Redfish resource, returning ``(document, error)``."""
    req = urllib.request.Request(f"https://{host}{path}")
    if user:
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:  # nosec B310
            return json.loads(r.read().decode()), ""
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        # urlopen wraps a verification failure in URLError, so the useful error
        # is one level down; without unwrapping, the operator sees a raw OpenSSL
        # string and no indication of which flag fixes it.
        cause = getattr(exc, "reason", exc)
        if isinstance(cause, ssl.SSLCertVerificationError):
            detail = getattr(cause, "verify_message", None) or cause
            return None, (
                f"TLS verification failed ({detail}). BMCs usually ship self-signed "
                f"certificates: pass --ca-cert <file> to trust this one, or --insecure "
                f"to skip verification deliberately"
            )
        return None, str(exc)


def probe_reachable(host: str, ctx: ssl.SSLContext) -> str:
    """Check the service root before minting anything; ``""`` when reachable.

    ``/redfish/v1`` is unauthenticated by specification, so reachability and TLS
    can both be settled with no credentials at all. Without this the tool minted
    an ADMINISTRATOR account, opened the LAN channel, failed certificate
    verification, and revoked -- exposing an admin account on the management
    network for a request that was never going to succeed.
    """
    doc, err = rf_get(host, "/redfish/v1", "", "", ctx, timeout=15)
    if doc is not None:
        return ""
    return err or "service root unreachable"


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
    for key, spec in BIOS_KNOBS.items():
        rx = re.compile(spec["pattern"], re.I)
        hits = {k: v for k, v in attrs.items() if rx.search(k) and not NAME_BLOCKLIST.search(k)}
        if not hits:
            continue
        # Prefer the shortest name: "ApbDis" over "ApbDisPolicyOverride".
        name = sorted(hits, key=lambda k: (len(k), k))[0]
        found[key] = {
            "attribute": name,
            "value": str(hits[name]),
            "all_matches": {k: str(v) for k, v in hits.items()},
        }
    return found


def normalize(value: object) -> str:
    """Lower-case, whitespace-collapsed form used for every comparison."""
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip().lower()


def verdict(key: str, value: object) -> str:
    """PASS/FAIL/UNKNOWN for a BIOS knob. Exact match after normalization."""
    v = normalize(value)
    if v in ("", "unknown", "auto", "n/a", "none"):
        return "UNKNOWN"
    return "PASS" if any(normalize(t) == v for t in BIOS_KNOBS[key]["target"]) else "FAIL"


EXIT_OK, EXIT_FAIL, EXIT_UNKNOWN, EXIT_CREDENTIAL = 0, 1, 2, 3


def exit_code(rows: list[dict], credential_failures: Sequence[str] = ()) -> int:
    """Worst status. A credential problem outranks every knob verdict.

    ``credential_failures`` is passed in rather than read from module state so
    that callers cannot compute this before the revocation has been attempted.
    """
    if credential_failures:
        return EXIT_CREDENTIAL
    if any(r["verdict"] == "FAIL" for r in rows):
        return EXIT_FAIL
    if any(r["verdict"] == "UNKNOWN" for r in rows):
        return EXIT_UNKNOWN
    return EXIT_OK


def build_rows(bios: dict) -> list[dict]:
    rows = []
    for key, spec in BIOS_KNOBS.items():
        hit = bios.get(key)
        value = hit["value"] if hit else "unknown"
        rows.append(
            {
                "knob": spec["label"],
                "key": key,
                "value": str(value),
                "attribute": hit["attribute"] if hit else "",
                "target": "/".join(spec["target"]),
                "verdict": verdict(key, value),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit BIOS-only EPYC knobs over Redfish (creates a temporary BMC account)",
        epilog=(
            "Exit: 0 on target, 1 a knob is wrong, 2 unresolved (including no BMC "
            "access at all), 3 a BMC account was left enabled or could not be "
            "confirmed revoked. Minting the temporary account needs "
            "--allow-account-creation; without it, pass --bmc-user. The password for "
            "--bmc-user comes from $BMC_PASSWORD or a prompt, never the command line, "
            "where the process table would expose it to every local user."
        ),
    )
    ap.add_argument("--bmc-host", help="BMC address (discovered over KCS when omitted)")
    ap.add_argument("--bmc-user", help="existing BMC account; avoids minting a temporary one")
    ap.add_argument(
        "--allow-account-creation",
        action="store_true",
        help="permit minting a temporary ADMINISTRATOR account when --bmc-user is absent",
    )
    ap.add_argument("--ca-cert", help="CA file (or the BMC's own certificate) to trust")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification (deliberate)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    install_signal_handlers()
    ctx = tls_context(args.ca_cert, args.insecure)

    # The README calls minting "a deliberate choice"; this is what makes it one.
    # Refusing before any BMC contact means the default invocation cannot create
    # a privileged account by omission.
    mints_account = not args.bmc_user
    if mints_account and not args.allow_account_creation:
        print("Auditing without --bmc-user mints a temporary ADMINISTRATOR account on "
              "the BMC. Pass --bmc-user with an existing (ideally read-only) account, "
              "or --allow-account-creation to accept that.", file=sys.stderr)
        return EXIT_UNKNOWN

    # ipmitool is needed to mint an account, not only to discover the address,
    # so --bmc-host does not excuse it on the minting path.
    if (mints_account or not args.bmc_host) and not ipmitool_present():
        print("ipmitool is not on PATH, so "
              + ("the temporary account cannot be minted. Pass --bmc-user to audit "
                 "with an existing account." if mints_account
                 else "the BMC address cannot be discovered. Pass --bmc-host explicitly."),
              file=sys.stderr)
        return EXIT_UNKNOWN

    host = args.bmc_host or bmc_ip()
    if not host:
        print("No BMC address found over KCS. Pass --bmc-host.", file=sys.stderr)
        return EXIT_UNKNOWN

    # Settle reachability and TLS before creating any credential.
    err = probe_reachable(host, ctx)
    if err:
        print(f"BMC {host} is not usable: {err}", file=sys.stderr)
        return EXIT_UNKNOWN

    acct: TempBmcAccount | None = None
    attrs: dict = {}
    err = ""
    no_access = ""
    interrupted = False

    try:
        if args.bmc_user:
            password = os.environ.get("BMC_PASSWORD") or getpass.getpass(
                f"Password for {args.bmc_user}@{host}: "
            )
            attrs, _, err = redfish_bios(host, args.bmc_user, password, ctx)
        else:
            if args.insecure:
                print(
                    "WARNING: --insecure with a minted account sends freshly created "
                    "ADMINISTRATOR credentials to a peer whose certificate was not "
                    "verified. Prefer --ca-cert, or --bmc-user with a read-only account.",
                    file=sys.stderr,
                )
            # Bound outside the `with` on purpose. Nothing here may compute the
            # exit code: __exit__ is what discovers a failed revocation, and it
            # has not run yet, so a `return` from inside this block would report
            # a clean audit on the path most likely to have left an
            # ADMINISTRATOR account enabled.
            acct = TempBmcAccount()
            with acct:
                if acct.slot is None or not acct.password:
                    no_access = acct.note or "no slot"
                else:
                    if acct.mint_errors:
                        print(f"Note: {'; '.join(acct.mint_errors)}", file=sys.stderr)
                    attrs, _, err = redfish_bios(host, acct.user, acct.password, ctx)
    except KeyboardInterrupt:
        # SIGTERM and SIGINT arrive here as an exception precisely so the `with`
        # above unwinds and revokes. Letting it escape would end the run in a
        # traceback and exit 130, discarding the revocation verdict that the
        # signal handler exists to produce.
        interrupted = True
        print("\nInterrupted; see the credential report below.", file=sys.stderr)

    credential_failures = list(acct.credential_failures) if acct else []
    rows = build_rows(resolve(attrs))
    code = exit_code(rows, credential_failures)

    if no_access:
        print(f"Could not obtain BMC access: {no_access}", file=sys.stderr)
    # Never reaching a knob is unresolved, not "every checked knob is on
    # target". A credential problem still outranks it.
    if (no_access or interrupted) and code != EXIT_CREDENTIAL:
        code = EXIT_UNKNOWN

    if args.json:
        print(json.dumps(
            {"host": host, "rows": rows, "error": err,
             "credential_failures": credential_failures, "exit_code": code},
            indent=2, sort_keys=True,
        ))
    else:
        if err:
            print(f"BIOS attributes unavailable: {err}", file=sys.stderr)
        width = max((len(r["knob"]) for r in rows), default=10)
        for r in rows:
            want = f"  (want {r['target']})" if r["verdict"] != "PASS" else ""
            print(f"  {r['verdict']:<7} {r['knob']:<{width}}  {r['value']}{want}")
        if credential_failures:
            print("\nCredential problems (exit 3):", file=sys.stderr)
            for f in credential_failures:
                print(f"    {f}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
