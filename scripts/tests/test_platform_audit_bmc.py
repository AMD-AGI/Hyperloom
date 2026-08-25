# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the BMC/Redfish auditor.

Nothing here talks to a real BMC: ``sudo_run`` is replaced with a fake
``ipmitool`` whose per-command responses each case controls. That is enough to
drive the account lifecycle, which is the part of this tool that can leave a
privileged account behind on a service processor.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "platform_audit_bmc.py"


def _load():
    spec = importlib.util.spec_from_file_location("platform_audit_bmc", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bmc():
    """A freshly loaded copy of the module.

    Credential failures live on the account instance, so each test reads them
    from the account it created; nothing global needs resetting between tests.
    """
    return _load()


class FakeIpmi:
    """Records every ipmitool invocation and answers from a scripted table."""

    def __init__(self, *, enabled_after_revoke=False, initially_enabled=False, fail=(), unreadable_status=False):
        self.calls: list[list[str]] = []
        self.fail = set(fail)
        self.initially_enabled = initially_enabled
        self.enabled_after_revoke = enabled_after_revoke
        self.unreadable_status = unreadable_status
        self._getaccess_seen = 0

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        joined = " ".join(cmd)
        for token in self.fail:
            if token in joined:
                return False, "", f"fake failure: {token}"
        if "user list" in joined:
            return True, "1 root true\n3 true true\n", ""
        if "channel getaccess" in joined:
            self._getaccess_seen += 1
            if self.unreadable_status:
                return False, "", "BMC timeout"
            first = self._getaccess_seen == 1
            state = "enabled" if (first and self.initially_enabled) else "disabled"
            if not first and self.enabled_after_revoke:
                state = "enabled"
            return True, f"Enable Status   : {state}\n", ""
        if "user set password" in joined:
            return True, "Set User Password command successful\n", ""
        return True, "", ""

    def ran(self, fragment: str) -> bool:
        return any(fragment in " ".join(c) for c in self.calls)


# ------------------------------------------------------- account lifecycle


def test_revocation_runs_even_when_the_password_step_fails(bmc, monkeypatch):
    """The slot must be claimed before any mutation.

    __exit__ returns immediately when no slot was claimed, so recording the slot
    any later than the first mutation means a failure in between leaves the
    account name written -- and, on the crash-recovery path, an enabled
    ADMINISTRATOR account live -- with nothing recorded about it.
    """
    fake = FakeIpmi(initially_enabled=True, fail={"user set password"})
    monkeypatch.setattr(bmc, "sudo_run", fake)

    with bmc.TempBmcAccount() as acct:
        assert acct.slot == 3, "slot must be claimed before the password step"
        assert not acct.password

    assert fake.ran("channel setaccess 1 3 callin=off"), "revocation must still run"
    assert fake.ran("user disable 3")


def test_the_password_never_reaches_the_diagnostics(bmc, monkeypatch, capsys):
    """ipmitool echoes its stdin on some failures; that must not be republished.

    The revoke path prints every failure to the console and stores it among the
    account's credential failures, so returning the command's stderr from the
    one function that handles the secret would put a live BMC password there.
    """

    class EchoingIpmi(FakeIpmi):
        def __init__(self):
            super().__init__(initially_enabled=True)
            self.secrets: list[str] = []

        def __call__(self, cmd, **kw):
            if "user set password" in " ".join(cmd):
                self.calls.append(cmd)
                secret = (kw.get("stdin") or "").split("\n")[0]
                self.secrets.append(secret)
                return False, "", f"ipmitool: cannot set password '{secret}'"
            return super().__call__(cmd, **kw)

    fake = EchoingIpmi()
    monkeypatch.setattr(bmc, "sudo_run", fake)

    with bmc.TempBmcAccount() as acct:
        pass

    assert fake.secrets, "the fake never saw a password, so it proved nothing"
    published = "\n".join(acct.credential_failures) + capsys.readouterr().err
    assert published, "the failure was meant to be reported"
    for secret in fake.secrets:
        assert secret, "a blank password would make this test vacuous"
        assert secret not in published

    # The contract, not just the current wording: text returned from the one
    # function that holds the secret is what put a password on the print path.
    ok, confirmed = bmc.set_bmc_password(3, "a-secret-that-must-not-escape")
    assert isinstance(ok, bool) and isinstance(confirmed, bool)


def test_stale_enabled_sentinel_is_recorded_not_just_printed(bmc, monkeypatch):
    """A sentinel found enabled is evidence of a previous leak, so it must exit 3."""
    monkeypatch.setattr(bmc, "sudo_run", FakeIpmi(initially_enabled=True))
    with bmc.TempBmcAccount() as acct:
        pass
    assert any("already enabled" in f for f in acct.credential_failures)
    assert bmc.exit_code([], acct.credential_failures) == bmc.EXIT_CREDENTIAL


def test_unreadable_final_state_is_not_treated_as_revoked(bmc, monkeypatch):
    """ "Could not read" must never be recorded as "confirmed disabled".

    Collapsing an unreadable state into "not enabled" would hand back a false
    confirmation for any BMC that errors or times out -- at the exact point the
    design calls its trust anchor.
    """
    monkeypatch.setattr(bmc, "sudo_run", FakeIpmi(unreadable_status=True))
    with bmc.TempBmcAccount() as acct:
        pass
    assert any("could not confirm" in f.lower() for f in acct.credential_failures)
    assert bmc.exit_code([], acct.credential_failures) == bmc.EXIT_CREDENTIAL


def test_account_still_enabled_after_revoke_is_reported(bmc, monkeypatch):
    monkeypatch.setattr(bmc, "sudo_run", FakeIpmi(enabled_after_revoke=True))
    with bmc.TempBmcAccount() as acct:
        pass
    assert any("still reports" in f for f in acct.credential_failures)


def test_clean_lifecycle_leaves_no_credential_failures(bmc, monkeypatch):
    fake = FakeIpmi()
    monkeypatch.setattr(bmc, "sudo_run", fake)
    with bmc.TempBmcAccount() as acct:
        assert acct.password
    assert acct.credential_failures == []
    assert bmc.exit_code([], acct.credential_failures) == bmc.EXIT_OK
    # Channel access is dropped while the account is still enabled, because BMCs
    # reject setaccess on a disabled account.
    order = [" ".join(c) for c in fake.calls]
    assert order.index(next(c for c in order if "callin=off" in c)) < order.index(
        next(c for c in order if "user disable" in c)
    )


def test_failed_grant_steps_are_surfaced(bmc, monkeypatch):
    """A silent failure here surfaces as a 401 that misdirects the operator."""
    monkeypatch.setattr(bmc, "sudo_run", FakeIpmi(fail={"user enable"}))
    with bmc.TempBmcAccount() as acct:
        assert any("enable account" in e for e in acct.mint_errors)


def test_a_failed_name_step_aborts_before_anything_grants_access(bmc, monkeypatch):
    """Owning the slot's name is a precondition for granting it privilege.

    Continuing past a failed rename enables an ADMINISTRATOR account on a slot
    the audit does not control, and Redfish then 401s under the sentinel name --
    which reads as an authentication problem rather than as the rename failing.
    """
    fake = FakeIpmi(fail={"user set name"})
    monkeypatch.setattr(bmc, "sudo_run", fake)

    with bmc.TempBmcAccount() as acct:
        assert any("set account name" in e for e in acct.mint_errors)
        assert not acct.password, "no credential may be handed out after this"

    for granting in ("user priv 3", "user enable 3", "callin=on"):
        assert not fake.ran(granting), f"{granting} ran after the name step failed"
    # The slot was still claimed, so the revoke path runs regardless.
    assert fake.ran("user disable 3")


def test_signal_handlers_allow_the_context_manager_to_unwind(bmc, monkeypatch):
    """SIGTERM is what a CI timeout sends, and it must not orphan the account."""
    import signal

    fake = FakeIpmi()
    monkeypatch.setattr(bmc, "sudo_run", fake)
    bmc.install_signal_handlers()
    with pytest.raises(KeyboardInterrupt):
        with bmc.TempBmcAccount():
            signal.raise_signal(signal.SIGTERM)
    assert fake.ran("user disable 3"), "revocation must run when SIGTERM arrives"


# ------------------------------------------------------- main(): exit contract


@pytest.fixture
def audit_main(bmc, monkeypatch):
    """Run ``main()`` with everything off-box stubbed out.

    Only the ipmitool layer is left to each test, because the exit code on the
    credential paths is what these cases are about.
    """
    monkeypatch.setattr(bmc, "install_signal_handlers", lambda: None)
    monkeypatch.setattr(bmc, "ipmitool_present", lambda: True)
    monkeypatch.setattr(bmc, "bmc_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(bmc, "tls_context", lambda ca_cert, insecure: None)
    monkeypatch.setattr(bmc, "probe_reachable", lambda host, ctx: "")
    monkeypatch.setattr(bmc, "redfish_bios", lambda *a, **k: ({}, "", ""))

    def _run(*argv):
        monkeypatch.setattr(sys, "argv", ["platform_audit_bmc.py", *argv])
        return bmc.main()

    return _run


def test_a_failed_revocation_reaches_the_exit_code(audit_main, bmc, monkeypatch, capsys):
    """The verdict must be computed after ``__exit__``, not inside the ``with``.

    This is the path that prints the "may remain enabled with ADMINISTRATOR
    privilege" banner, so it is the last one that may report a clean audit.
    """
    monkeypatch.setattr(bmc, "sudo_run", FakeIpmi(fail={"user set password"}))

    code = audit_main("--allow-account-creation")

    err = capsys.readouterr().err
    assert "SECURITY" in err, "this case is only interesting with the banner"
    assert code == bmc.EXIT_CREDENTIAL, "exit 0 here contradicts the banner above it"


def test_no_bmc_access_is_unresolved_not_success(audit_main, bmc, monkeypatch, capsys):
    """Never reaching a knob cannot mean "every checked knob is on target"."""
    monkeypatch.setattr(bmc, "sudo_run", FakeIpmi(fail={"user list"}))

    code = audit_main("--allow-account-creation")

    assert "Could not obtain BMC access" in capsys.readouterr().err
    assert code == bmc.EXIT_UNKNOWN


@pytest.mark.parametrize(
    "revoke_fails,expected",
    [(False, "EXIT_UNKNOWN"), (True, "EXIT_CREDENTIAL")],
)
def test_a_signal_mid_audit_produces_an_exit_code_not_a_traceback(audit_main, bmc, monkeypatch, revoke_fails, expected):
    """SIGTERM is the case the signal handler exists for, so it must be reported.

    The handler raises KeyboardInterrupt to unwind the ``with``; letting that
    escape ends the run in a traceback and exit 130, discarding the revocation
    verdict it just produced.
    """
    fake = FakeIpmi(fail={"user disable"} if revoke_fails else ())
    monkeypatch.setattr(bmc, "sudo_run", fake)

    def _interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(bmc, "redfish_bios", _interrupted)

    code = audit_main("--allow-account-creation")

    assert fake.ran("channel setaccess 1 3 callin=off"), "the with block must unwind"
    assert code == getattr(bmc, expected)


def test_minting_an_account_requires_an_explicit_opt_in(audit_main, bmc, monkeypatch, capsys):
    """The README calls this a deliberate choice; the default must not make it."""
    calls: list[list[str]] = []
    monkeypatch.setattr(bmc, "sudo_run", lambda cmd, **kw: (calls.append(cmd), (True, "", ""))[1])

    code = audit_main()

    assert code == bmc.EXIT_UNKNOWN
    assert not calls, "the BMC must not be touched before the operator opts in"
    assert "--allow-account-creation" in capsys.readouterr().err


def test_ipmitool_is_required_to_mint_even_with_an_explicit_host(audit_main, bmc, monkeypatch, capsys):
    """--bmc-host removes the need to discover the address, not to mint."""
    monkeypatch.setattr(bmc, "ipmitool_present", lambda: False)

    assert audit_main("--bmc-host", "10.0.0.1", "--allow-account-creation") == bmc.EXIT_UNKNOWN
    assert "ipmitool" in capsys.readouterr().err

    # ... and it is not required when no account will be minted.
    monkeypatch.setenv("BMC_PASSWORD", "not-a-real-password")
    assert audit_main("--bmc-host", "10.0.0.1", "--bmc-user", "ro") == bmc.EXIT_UNKNOWN
    assert "ipmitool" not in capsys.readouterr().err


# ------------------------------------------------------- verdicts / matching


def test_verdict_is_exact_after_normalization(bmc):
    assert bmc.verdict("df_cstates", "Disabled") == "PASS"
    assert bmc.verdict("df_cstates", "Enabled") == "FAIL"
    assert bmc.verdict("apbdis", "1") == "PASS"
    assert bmc.verdict("power_profile", "High Performance Mode") == "PASS"
    # An unresolved "Auto" is not a pass.
    assert bmc.verdict("apbdis", "Auto") == "UNKNOWN"


@pytest.mark.parametrize(
    "name,blocked",
    [
        ("SevControl", True),  # CamelCase: no trailing word boundary
        ("SEV_SNP_Support", True),  # SCREAMING_CASE
        ("Sev", True),
        ("SeverityLevel", False),  # merely contains the letters
        ("DfCState", False),
    ],
)
def test_blocklist_handles_sev_in_both_directions(bmc, name, blocked):
    """Unanchored "sev" over-matched; a plain \\bsev\\b under-matches CamelCase."""
    assert bool(bmc.NAME_BLOCKLIST.search(name)) is blocked


def test_blocklist_does_not_swallow_a_real_knob(bmc):
    attrs = {"SevControl": "Enabled", "DfCState": "Disabled", "SeverityLevel": "High"}
    assert bmc.resolve(attrs)["df_cstates"]["value"] == "Disabled"


def test_resolve_prefers_the_shortest_matching_attribute(bmc):
    attrs = {"ApbDisPolicyOverride": "0", "ApbDis": "1"}
    assert bmc.resolve(attrs)["apbdis"]["attribute"] == "ApbDis"


def test_every_bios_target_cites_its_basis(bmc):
    """Unlike the OS-layer knobs, all three of these are supported by 58011 ch. 5."""
    for key, spec in bmc.BIOS_KNOBS.items():
        assert spec["basis"].strip(), f"{key} has no cited basis"
        assert "58011" in spec["basis"], f"{key} does not cite the guide"


def test_exit_code_ranks_credentials_above_everything(bmc):
    rows = [{"verdict": "FAIL"}]
    assert bmc.exit_code(rows) == bmc.EXIT_FAIL
    assert bmc.exit_code(rows, ["left enabled"]) == bmc.EXIT_CREDENTIAL
