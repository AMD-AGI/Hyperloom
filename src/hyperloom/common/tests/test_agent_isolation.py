# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``common/agent_isolation``: when isolation is required, the order of the tiers, which bubblewrap is probed, the
refusal, the unconfined fallback and how it is said, and the launch command line.

No sandbox is made here: the account tier's test and bubblewrap's probe are stand-ins (the real probe and sandboxes
are ``hyperloom_common/tests/test_bwrap*.py``; a confinement launched for real is ``test_confined_launch.py``)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from hyperloom_common.bwrap import BwrapPolicy, BwrapProbe

from hyperloom.common import agent_isolation
from hyperloom.common.agent_isolation import (
    NO_LAYER_BWRAP,
    RAY_ACTOR_PATH,
    SANDBOX_ACCOUNT,
    AccountIsolation,
    BwrapIsolation,
    IsolationRefused,
    IsolationReport,
    NoIsolation,
    agent_isolation_report,
    announce_agent_isolation,
    bwrap_isolation,
    choose_agent_isolation,
    choose_sandbox_isolation,
    isolation_required,
    require_agent_isolation,
)
from hyperloom.common.env_safety import SPECIALIST_USER_ENV, RunAs, RunAsError

ACCOUNT = RunAs(user="hl-specialist", uid=1500, gid=1500, groups=(1500,), home="/home/hl", drop_tool="/usr/bin/setpriv")
LAYER = "/opt/hyperloom-host/bin/bwrap"
ON_PATH = BwrapProbe("/usr/bin/bwrap", (0, 13, 1), ())
IN_LAYER = BwrapProbe(LAYER, (0, 13, 1), ())
DENIED = BwrapProbe("/usr/bin/bwrap", (0, 11, 1), ("bwrap: setting up uid map: Permission denied",))
LAYER_DENIED = BwrapProbe(LAYER, (0, 13, 1), ("bwrap: setting up uid map: Permission denied",))
MISSING = BwrapProbe(None, None, ("bubblewrap: not found on PATH",))
NO_ANSWER = BwrapProbe(LAYER, (0, 13, 1), (f"bubblewrap at {LAYER}: no answer in 10 s",), transient=True)
NOT_SET = f"running them as an account: {SPECIALIST_USER_ENV} is not set"


class Probes:
    """A stand-in for ``bwrap_probe``: answers ``answer`` and records the path each call asked about."""

    def __init__(self, answer: BwrapProbe) -> None:
        self.answer = answer
        self.asked: list[str | None] = []

    def __call__(self, path: str | None) -> BwrapProbe:
        self.asked.append(path)
        return self.answer


def _never(path: str | None) -> BwrapProbe:
    raise AssertionError(f"bubblewrap must not be probed here (asked for {path})")


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that has not been marked hosted, with no specialist account configured."""
    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process())
    monkeypatch.delenv(SPECIALIST_USER_ENV, raising=False)


@pytest.fixture
def account(monkeypatch: pytest.MonkeyPatch):
    """``HYPERLOOM_SPECIALIST_USER`` names an account that resolves; the setter's argument is the capabilities this
    process lacks."""
    monkeypatch.setenv(SPECIALIST_USER_ENV, ACCOUNT.user)
    monkeypatch.setattr(agent_isolation, "resolve_run_as", lambda name: ACCOUNT if name == ACCOUNT.user else None)

    def missing(names: list[str]) -> None:
        monkeypatch.setattr(agent_isolation, "missing_launch_capabilities", lambda: list(names))

    missing([])
    return missing


# ---- a local run: bubblewrap where it works, else unconfined and said so ----


def test_a_local_launch_without_an_account_runs_under_the_bubblewrap_on_path():
    probes = Probes(ON_PATH)
    assert choose_agent_isolation(probe=probes) == BwrapIsolation(ON_PATH, (NOT_SET,))
    assert probes.asked == [None], "the bubblewrap on PATH"
    assert not isolation_required()


def test_a_local_launch_where_neither_tier_works_runs_unconfined_naming_both():
    found = choose_agent_isolation(probe=Probes(DENIED))
    assert found == NoIsolation(
        (NOT_SET, "bubblewrap (/usr/bin/bwrap, 0.11.1): bwrap: setting up uid map: Permission denied")
    )
    assert found.why == "; ".join(found.reasons)


def test_a_probe_line_that_names_bubblewrap_is_kept_verbatim():
    assert choose_agent_isolation(probe=Probes(MISSING)).reasons == (NOT_SET, "bubblewrap: not found on PATH")


def test_a_local_probe_that_gave_no_answer_runs_this_launch_unconfined_never_refused():
    assert choose_agent_isolation(probe=Probes(NO_ANSWER)).reasons == (NOT_SET, *NO_ANSWER.problems)


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_account_is_no_account(monkeypatch, value):
    monkeypatch.setenv(SPECIALIST_USER_ENV, value)
    assert isinstance(choose_agent_isolation(probe=Probes(DENIED)), NoIsolation)
    assert not isolation_required()


# ---- a hosted run: required, and only the hosting layer's bubblewrap ----


def test_a_hosted_run_uses_the_layers_bubblewrap_by_path_for_every_later_launch():
    probes = Probes(IN_LAYER)
    assert require_agent_isolation(bwrap=LAYER, probe=probes) == BwrapIsolation(IN_LAYER, (NOT_SET,))
    assert isolation_required()
    assert isinstance(choose_agent_isolation(probe=probes), BwrapIsolation)
    assert isinstance(choose_sandbox_isolation(probe=probes), BwrapIsolation)
    assert probes.asked == [LAYER, LAYER, LAYER], "never the one on PATH"


@pytest.mark.parametrize("named", [None, ""])
def test_a_hosted_run_that_names_no_layer_bubblewrap_refuses_without_probing_path(named):
    """An empty path, as a caller reading an unset setting passes, would send the probe to the one on PATH."""
    with pytest.raises(IsolationRefused) as refused:
        require_agent_isolation(bwrap=named, probe=_never)
    assert refused.value.reasons == (NOT_SET, NO_LAYER_BWRAP)
    for later in (choose_agent_isolation, choose_sandbox_isolation):
        with pytest.raises(IsolationRefused):
            later(probe=_never)
    assert bwrap_isolation(probe=_never) == NoIsolation((NO_LAYER_BWRAP,))


@pytest.mark.parametrize("named", ["bwrap", "bin/bwrap"])
def test_a_hosted_run_given_a_relative_layer_bubblewrap_refuses_without_probing(named):
    with pytest.raises(IsolationRefused) as refused:
        require_agent_isolation(bwrap=named, probe=_never)
    assert refused.value.reasons == (NOT_SET, f"bubblewrap at {named} is not an absolute path")


def test_a_hosted_run_where_no_tier_works_refuses_at_start_up_naming_both():
    with pytest.raises(IsolationRefused) as refused:
        require_agent_isolation(bwrap=LAYER, probe=Probes(LAYER_DENIED))
    assert isinstance(refused.value, RunAsError), "every launch path that refuses on a RunAsError refuses on this"
    assert str(refused.value) == (
        f"agents cannot be isolated on this host: running them as an account: {SPECIALIST_USER_ENV} is not set; "
        f"bubblewrap ({LAYER}, 0.13.1): bwrap: setting up uid map: Permission denied"
    )
    assert refused.value.transient is False


def test_a_hosted_probe_that_gave_no_answer_refuses_as_transient():
    with pytest.raises(IsolationRefused) as refused:
        require_agent_isolation(bwrap=LAYER, probe=Probes(NO_ANSWER))
    assert refused.value.transient is True


def test_a_hosted_run_refuses_an_unconfined_sandbox_too():
    require_agent_isolation(bwrap=LAYER, allow_account=True, probe=Probes(IN_LAYER))
    with pytest.raises(IsolationRefused) as refused:
        choose_sandbox_isolation(probe=Probes(LAYER_DENIED))
    assert refused.value.reasons[0] == f"running them as an account: {SANDBOX_ACCOUNT}"


# ---- the order of the tiers ----


def test_a_usable_account_is_the_first_tier(account):
    assert choose_agent_isolation(probe=_never) == AccountIsolation(ACCOUNT)
    assert isolation_required()


def test_an_account_that_does_not_resolve_falls_to_bubblewrap_and_warns_once(account, monkeypatch, caplog):
    def refuse(name: str) -> RunAs:
        raise RunAsError(f"{SPECIALIST_USER_ENV}={name!r}: no such user")

    monkeypatch.setattr(agent_isolation, "resolve_run_as", refuse)
    with caplog.at_level(logging.INFO, logger=agent_isolation.__name__):
        found = choose_agent_isolation(probe=Probes(ON_PATH))
        choose_agent_isolation(probe=Probes(ON_PATH))
    note = f"running them as an account: {SPECIALIST_USER_ENV}='hl-specialist': no such user"
    assert found == BwrapIsolation(ON_PATH, (note,))
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.WARNING, f"agents run under bubblewrap: {note}")
    ]


def test_an_account_this_process_cannot_signal_falls_to_bubblewrap(account):
    account(["CAP_KILL"])
    assert choose_agent_isolation(probe=Probes(ON_PATH)).notes == (
        "running them as an account: running a specialist as another user needs root or CAP_KILL",
    )


def test_no_account_configured_logs_no_warning_when_bubblewrap_is_used(caplog):
    with caplog.at_level(logging.WARNING, logger=agent_isolation.__name__):
        choose_agent_isolation(probe=Probes(ON_PATH))
    assert caplog.records == []


def test_the_ray_actor_path_never_gets_the_account_tier(account):
    found = choose_agent_isolation(allow_account=False, probe=Probes(ON_PATH))
    assert found == BwrapIsolation(ON_PATH, (f"running them as an account: {RAY_ACTOR_PATH}",))


def test_a_configured_account_makes_a_specialist_launch_required_on_the_ray_path_too(account):
    with pytest.raises(IsolationRefused) as refused:
        choose_agent_isolation(allow_account=False, probe=Probes(DENIED))
    assert refused.value.reasons == (
        f"running them as an account: {RAY_ACTOR_PATH}",
        "bubblewrap (/usr/bin/bwrap, 0.11.1): bwrap: setting up uid map: Permission denied",
    )


def test_a_configured_account_where_neither_tier_works_refuses(account):
    account(["CAP_SETUID", "CAP_KILL"])
    with pytest.raises(IsolationRefused) as refused:
        choose_agent_isolation(probe=Probes(MISSING))
    assert refused.value.reasons == (
        "running them as an account: running a specialist as another user needs root or CAP_SETUID, CAP_KILL",
        "bubblewrap: not found on PATH",
    )


def test_an_explicit_user_wins_over_the_environment(account):
    assert isinstance(choose_agent_isolation("", probe=Probes(DENIED)), NoIsolation)


# ---- sandboxes and the compile's second tier ----


def test_a_local_sandbox_is_never_refused_even_with_an_account_configured(account):
    found = choose_sandbox_isolation(probe=Probes(DENIED))
    assert isinstance(found, NoIsolation)
    assert found.reasons[0] == f"running them as an account: {SANDBOX_ACCOUNT}"


def test_a_sandbox_runs_under_bubblewrap_where_it_works(account):
    assert choose_sandbox_isolation(probe=Probes(ON_PATH)) == BwrapIsolation(
        ON_PATH, (f"running them as an account: {SANDBOX_ACCOUNT}",)
    )


def test_bubblewrap_alone_names_its_probe_s_reasons_where_it_finds_a_problem():
    assert bwrap_isolation(probe=Probes(DENIED)) == NoIsolation(
        ("bubblewrap (/usr/bin/bwrap, 0.11.1): bwrap: setting up uid map: Permission denied",)
    )
    assert bwrap_isolation(probe=Probes(MISSING)) == NoIsolation(("bubblewrap: not found on PATH",))
    assert bwrap_isolation(probe=Probes(ON_PATH)) == BwrapIsolation(ON_PATH)
    require_agent_isolation(bwrap=LAYER, probe=Probes(IN_LAYER))
    probes = Probes(IN_LAYER)
    assert bwrap_isolation(probe=probes) == BwrapIsolation(IN_LAYER)
    assert probes.asked == [LAYER]


# ---- the report and the start-up line ----


def test_the_report_of_an_unconfined_local_run_names_both_tiers():
    report = agent_isolation_report(probe=Probes(DENIED))
    reasons = (
        NOT_SET,
        "bubblewrap (/usr/bin/bwrap, 0.11.1): bwrap: setting up uid map: Permission denied",
        f"running them as an account: {SANDBOX_ACCOUNT}",
    )
    assert report == IsolationReport("none", "none", False, None, reasons)
    assert report.to_dict() == {
        "specialists": "none",
        "sandboxes": "none",
        "required": False,
        "bwrap": None,
        "reasons": list(reasons),
    }
    assert report.line() == "agent isolation: specialists none, sandboxes none: " + "; ".join(reasons)


def test_the_report_names_the_bubblewrap_in_use():
    report = agent_isolation_report(probe=Probes(ON_PATH))
    assert (report.specialists, report.sandboxes, report.bwrap) == ("bwrap", "bwrap", "0.13.1 /usr/bin/bwrap")
    assert report.line().startswith(
        "agent isolation: specialists bwrap, sandboxes bwrap (bubblewrap 0.13.1 /usr/bin/bwrap): "
    )


def test_the_report_records_a_refusal_instead_of_raising(account):
    account(["CAP_KILL"])
    report = agent_isolation_report(probe=Probes(DENIED))
    assert (report.specialists, report.sandboxes, report.required) == ("refused", "none", True)


def test_an_account_for_specialists_and_a_bubblewrap_sandbox_report_both(account):
    report = agent_isolation_report(probe=Probes(ON_PATH))
    assert (report.specialists, report.sandboxes, report.required) == ("account", "bwrap", True)
    assert report.bwrap == "0.13.1 /usr/bin/bwrap"
    assert report.reasons == (f"running them as an account: {SANDBOX_ACCOUNT}",)


def test_start_up_says_once_with_a_warning_that_agents_run_unconfined(caplog):
    with caplog.at_level(logging.INFO, logger=agent_isolation.__name__):
        report = announce_agent_isolation(probe=Probes(MISSING))
        announce_agent_isolation(probe=Probes(MISSING))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [
        "agents run unconfined on this host, as before: they can read this account's files and credentials "
        f"({'; '.join(report.reasons)})"
    ]


def test_start_up_names_only_the_unconfined_sandboxes_when_specialists_have_an_account(account, caplog):
    with caplog.at_level(logging.WARNING, logger=agent_isolation.__name__):
        announce_agent_isolation(probe=Probes(MISSING))
    assert [r.getMessage().split(",")[0] for r in caplog.records] == ["agent sandboxes run unconfined on this host"]


def test_start_up_says_one_info_line_when_every_agent_is_confined(caplog):
    with caplog.at_level(logging.INFO, logger=agent_isolation.__name__):
        report = announce_agent_isolation(probe=Probes(ON_PATH))
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [(logging.INFO, report.line())]


def test_start_up_warns_once_naming_specialists_that_would_be_refused(account, caplog):
    account(["CAP_KILL"])
    with caplog.at_level(logging.WARNING, logger=agent_isolation.__name__):
        report = announce_agent_isolation(probe=Probes(DENIED))
    assert [r.getMessage() for r in caplog.records] == [
        "agent sandboxes run unconfined on this host, as before: they can read this account's files and credentials; "
        f"specialists will be refused: they must be isolated here, and no tier can ({'; '.join(report.reasons)})"
    ]


def test_start_up_of_a_hosted_run_that_cannot_isolate_warns_that_every_agent_would_be_refused(caplog):
    with pytest.raises(IsolationRefused):
        require_agent_isolation(bwrap=LAYER, probe=Probes(LAYER_DENIED))
    with caplog.at_level(logging.WARNING, logger=agent_isolation.__name__):
        report = announce_agent_isolation(probe=Probes(LAYER_DENIED))
    assert (report.specialists, report.sandboxes, report.required) == ("refused", "refused", True)
    assert [r.getMessage().split(" (")[0] for r in caplog.records] == [
        "specialists and agent sandboxes will be refused: they must be isolated here, and no tier can"
    ]


# ---- the launch command line ----


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "rw" / "ro").mkdir(parents=True)
    return tmp_path


def test_a_launch_moves_the_marks_into_the_sandbox_and_out_of_bubblewraps_environment(tree):
    policy = BwrapPolicy(read_only=(str(tree),))
    env = {"PATH": "/usr/bin", "HYPERLOOM_REAP_MARK": "m-1", "OTHER": "kept"}
    argv, child_env = BwrapIsolation(ON_PATH).launch(["claude", "-p"], policy, env, marks=("HYPERLOOM_REAP_MARK",))
    assert argv[0] == "/usr/bin/bwrap" and argv[-3:] == ["--", "claude", "-p"]
    assert argv[argv.index("--setenv") : argv.index("--setenv") + 3] == ["--setenv", "HYPERLOOM_REAP_MARK", "m-1"]
    assert child_env == {"PATH": "/usr/bin", "OTHER": "kept"}
    assert env["HYPERLOOM_REAP_MARK"] == "m-1", "the caller's mapping is not changed"
    assert "--die-with-parent" in argv, "kept on the main thread"


def test_a_launch_off_the_main_thread_leaves_out_die_with_parent(tree):
    found: list[list[str]] = []
    worker = threading.Thread(
        target=lambda: found.append(
            BwrapIsolation(ON_PATH).launch(["true"], BwrapPolicy(read_only=(str(tree),)), {})[0]
        )
    )
    worker.start()
    worker.join(10)
    assert found and "--die-with-parent" not in found[0]


def test_a_launch_refuses_a_policy_its_bubblewrap_must_not_run(tree):
    policy = BwrapPolicy(read_only=(str(tree / "rw" / "ro"),), writable=(str(tree / "rw"),))
    BwrapIsolation(ON_PATH).launch(["true"], policy, {})
    with pytest.raises(IsolationRefused) as refused:
        BwrapIsolation(BwrapProbe("/usr/bin/bwrap", (0, 11, 1), ())).launch(["true"], policy, {})
    assert "needs bubblewrap 0.12.0 (this is 0.11.1)" in str(refused.value)


def test_a_mark_with_a_secret_shaped_name_never_reaches_the_command_line(tree):
    with pytest.raises(ValueError, match="never a secret: GH_TOKEN"):
        BwrapIsolation(ON_PATH).launch(
            ["true"], BwrapPolicy(read_only=(str(tree),)), {"GH_TOKEN": "x"}, marks=("GH_TOKEN",)
        )
