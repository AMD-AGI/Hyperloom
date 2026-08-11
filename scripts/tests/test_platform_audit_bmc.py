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
def bmc(monkeypatch):
    """The module with a fake ipmitool, and a clean credential-failure list."""
    mod = _load()
    mod.CREDENTIAL_FAILURES.clear()
    return mod


class FakeIpmi:
    """Records every ipmitool invocation and answers from a scripted table."""

    def __init__(self, *, enabled_after_revoke=False, initially_enabled=False,
                 fail=(), unreadable_status=False):
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

    Previously the slot was recorded only after the password was set, so a
    password failure returned with slot=None, __exit__ returned immediately, and
    nothing was recorded -- while the account name had already been written and,
    on the crash-recovery path, an enabled ADMINISTRATOR account was left live.
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

    The revoke path prints every failure to the console and stores it in
    CREDENTIAL_FAILURES, so returning the command's stderr from the one function
    that handles the secret put a live BMC password on that path.
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

    with bmc.TempBmcAccount():
        pass

    assert fake.secrets, "the fake never saw a password, so it proved nothing"
    published = "\n".join(bmc.CREDENTIAL_FAILURES) + capsys.readouterr().err
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
    with bmc.TempBmcAccount():
        pass
    assert any("already enabled" in f for f in bmc.CREDENTIAL_FAILURES)
    assert bmc.exit_code([]) == bmc.EXIT_CREDENTIAL


def test_unreadable_final_state_is_not_treated_as_revoked(bmc, monkeypatch):
    """"Could not read" must never be recorded as "confirmed disabled".

    account_status used a failure-swallowing runner, so any BMC that errored or
    timed out answered "not enabled" -- a false confirmation at the exact point
    the design calls its trust anchor.
    """
    monkeypatch.setattr(bmc, "sudo_run", FakeIpmi(unreadable_status=True))
    with bmc.TempBmcAccount():
        pass
    assert any("could not confirm" in f.lower() for f in bmc.CREDENTIAL_FAILURES)
    assert bmc.exit_code([]) == bmc.EXIT_CREDENTIAL


def test_account_still_enabled_after_revoke_is_reported(bmc, monkeypatch):
    monkeypatch.setattr(bmc, "sudo_run", FakeIpmi(enabled_after_revoke=True))
    with bmc.TempBmcAccount():
        pass
    assert any("still reports" in f for f in bmc.CREDENTIAL_FAILURES)


def test_clean_lifecycle_leaves_no_credential_failures(bmc, monkeypatch):
    fake = FakeIpmi()
    monkeypatch.setattr(bmc, "sudo_run", fake)
    with bmc.TempBmcAccount() as acct:
        assert acct.password
    assert bmc.CREDENTIAL_FAILURES == []
    assert bmc.exit_code([]) == bmc.EXIT_OK
    # Channel access is dropped while the account is still enabled, because BMCs
    # reject setaccess on a disabled account.
    order = [" ".join(c) for c in fake.calls]
    assert order.index(next(c for c in order if "callin=off" in c)) < order.index(
        next(c for c in order if "user disable" in c)
    )


def test_failed_grant_steps_are_surfaced(bmc, monkeypatch):
    """A silent failure here produced a 401 that misdirected the operator."""
    monkeypatch.setattr(bmc, "sudo_run", FakeIpmi(fail={"user enable"}))
    with bmc.TempBmcAccount() as acct:
        assert any("enable account" in e for e in acct.mint_errors)


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
        ("SevControl", True),        # CamelCase: no trailing word boundary
        ("SEV_SNP_Support", True),   # SCREAMING_CASE
        ("Sev", True),
        ("SeverityLevel", False),    # merely contains the letters
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
    bmc.CREDENTIAL_FAILURES.append("left enabled")
    assert bmc.exit_code(rows) == bmc.EXIT_CREDENTIAL
