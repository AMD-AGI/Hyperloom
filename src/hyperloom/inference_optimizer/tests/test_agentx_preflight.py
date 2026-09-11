# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX preflight: AIPERF_BIN resolution + capability (weka-trace) check.

Contract:
- ``resolve_aiperf_bin`` prefers ``AIPERF_BIN``, then the managed CLI, then PATH.
- ``check_aiperf_capability`` raises ``AgentXPreflightError`` when the binary is
  missing OR lacks the AgentX (weka-trace) capability. It verifies *capability*,
  not mere existence. The probe is injectable so the check is testable offline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.agentx import preflight as pf

from hyperloom.inference_optimizer.agentx.preflight import (
    AgentXPreflightError,
    check_aiperf_capability,
    resolve_aiperf_bin,
)


def test_resolve_prefers_env():
    assert resolve_aiperf_bin({"AIPERF_BIN": "/venv/bin/aiperf"}) == "/venv/bin/aiperf"


def _managed_cli(state: Path) -> Path:
    cli = state / "aiperf-venv" / "bin" / "aiperf"
    cli.parent.mkdir(parents=True)
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    return cli


@pytest.mark.parametrize("override", ["", " \t\n"])
def test_resolve_managed_before_broken_path_aiperf(tmp_path, monkeypatch, override):
    cli = _managed_cli(tmp_path / "home" / ".hyperloom")
    monkeypatch.setattr(pf.shutil, "which", lambda *_a, **_k: "/broken/bin/aiperf")
    env = {"HOME": str(tmp_path / "home"), "AIPERF_BIN": override, "PATH": "/broken/bin"}
    original = dict(env)
    assert resolve_aiperf_bin(env) == str(cli)
    assert env == original


def test_resolve_managed_custom_state_outranks_home(tmp_path):
    cli = _managed_cli(tmp_path / "custom state")
    _managed_cli(tmp_path / "home" / ".hyperloom")
    assert resolve_aiperf_bin(
        {"HYPERLOOM_STATE_DIR": str(tmp_path / "custom state"), "HOME": str(tmp_path / "home")}
    ) == str(cli)


def test_resolve_managed_empty_state_uses_home(tmp_path):
    cli = _managed_cli(tmp_path / ".hyperloom")
    assert resolve_aiperf_bin({"HYPERLOOM_STATE_DIR": "", "HOME": str(tmp_path)}) == str(cli)


@pytest.mark.skipif(os.name != "posix", reason="POSIX user database")
@pytest.mark.parametrize("home", [None, ""])
def test_resolve_managed_missing_home_uses_system_user(tmp_path, monkeypatch, home):
    import pwd

    cli = _managed_cli(tmp_path / "system-home" / ".hyperloom")
    monkeypatch.setenv("HOME", str(tmp_path / "unrelated-parent-home"))
    seen = []

    def _getpwuid(uid):
        seen.append(uid)
        return SimpleNamespace(pw_dir=str(tmp_path / "system-home"))

    monkeypatch.setattr(pwd, "getpwuid", _getpwuid)
    env = {"HOME": home} if home is not None else {}
    assert resolve_aiperf_bin(env) == str(cli)
    assert seen == [os.getuid()]


@pytest.mark.parametrize("env", [{"HYPERLOOM_STATE_DIR": "relative"}, {"HOME": "relative-home"}])
def test_resolve_rejects_relative_state(env):
    with pytest.raises(AgentXPreflightError, match="HYPERLOOM_STATE_DIR.*absolute") as exc:
        resolve_aiperf_bin(env)
    assert not exc.value.repairable


def test_resolve_override_outranks_managed_and_invalid_state(tmp_path):
    _managed_cli(tmp_path)
    for state in (str(tmp_path), "relative"):
        env = {"HYPERLOOM_STATE_DIR": state, "AIPERF_BIN": " \t/external/bin/aiperf\n"}
        assert resolve_aiperf_bin(env) == "/external/bin/aiperf"


@pytest.mark.parametrize("candidate_kind", ["directory", "not-executable"])
def test_resolve_unusable_managed_falls_back_to_path(tmp_path, monkeypatch, candidate_kind):
    cli = _managed_cli(tmp_path)
    if candidate_kind == "directory":
        cli.unlink()
        cli.mkdir()
    else:
        if os.name != "posix":
            pytest.skip("POSIX executable permissions")
        cli.chmod(0o644)
    monkeypatch.setattr(pf.shutil, "which", lambda *_a, **_k: "/fallback/aiperf")
    assert resolve_aiperf_bin({"HYPERLOOM_STATE_DIR": str(tmp_path)}) == "/fallback/aiperf"


def test_resolve_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.agentx.preflight.shutil.which",
        lambda _n, path=None: None,
    )
    assert resolve_aiperf_bin({"HOME": str(tmp_path)}) is None


def test_resolve_path_lookup_returns_which(monkeypatch, tmp_path):
    seen = {}

    def _which(name, path=None):
        seen["name"] = name
        seen["path"] = path
        return "/opt/venv/bin/aiperf"

    monkeypatch.setattr("hyperloom.inference_optimizer.agentx.preflight.shutil.which", _which)
    # No AIPERF_BIN override -> falls back to which(), honoring the passed env PATH.
    assert resolve_aiperf_bin({"HOME": str(tmp_path), "PATH": "/opt/venv/bin"}) == "/opt/venv/bin/aiperf"
    assert seen == {"name": "aiperf", "path": "/opt/venv/bin"}


def test_missing_bin_raises():
    with pytest.raises(AgentXPreflightError) as ei:
        check_aiperf_capability(None)
    assert "AIPERF_BIN" in str(ei.value)


def test_capability_absent_raises():
    # probe returns help text WITHOUT weka-trace -> not AgentX-capable
    def _probe(_bin):
        return "usage: aiperf profile [options]\n  --public-dataset ...\n"

    with pytest.raises(AgentXPreflightError) as ei:
        check_aiperf_capability("/venv/bin/aiperf", probe=_probe, loader_probe=lambda _bin: None)
    assert "weka-trace" in str(ei.value) or "capab" in str(ei.value).lower()


_CAPABLE_HELP = (
    "usage: aiperf profile\n"
    "  --custom-dataset-type weka-trace ...\n"
    "  --scenario TEXT  Lock all benchmark invariants for a named scenario\n"
    "  --benchmark-duration FLOAT\n"
    "  --api-host TEXT\n"
    "  --api-port INTEGER\n"
)


def test_capability_present_ok():
    def _probe(_bin):
        return _CAPABLE_HELP

    # must not raise
    check_aiperf_capability("/venv/bin/aiperf", probe=_probe, loader_probe=lambda _bin: None)


def test_capability_rejects_build_without_progress_api():
    help_text = "weka-trace --scenario --benchmark-duration"
    with pytest.raises(AgentXPreflightError, match="phase progress"):
        check_aiperf_capability(
            "/venv/bin/aiperf",
            require_progress_api=True,
            probe=lambda _bin: help_text,
            loader_probe=lambda _bin: _NEW,
        )


def test_capability_rejects_pre_scenario_build():
    """weka-trace alone is stale: those builds predate the 062126 corpus.

    Their scenario allowlist rejects the corpus the client now requests and
    they have no ``--benchmark-duration``, so accepting them would defer the
    failure to an hour into a run instead of surfacing it at startup.
    """

    def _probe(_bin):
        return "usage: aiperf profile\n  --custom-dataset-type weka-trace ...\n"

    with pytest.raises(AgentXPreflightError) as ei:
        check_aiperf_capability("/venv/bin/aiperf", probe=_probe, loader_probe=lambda _bin: None)
    assert "--scenario" in str(ei.value)


def test_probe_failure_raises_not_crash():
    def _probe(_bin):
        raise OSError("cannot exec")

    with pytest.raises(AgentXPreflightError):
        check_aiperf_capability("/venv/bin/aiperf", probe=_probe, loader_probe=lambda _bin: None)


# --- loader-allowlist assertion ------------------------------------------------
#
# Flag presence cannot separate the pinned build from the previous one: aiperf
# 0.8.0 carries weka-trace, --scenario and --benchmark-duration, and defines a
# scenario by the same name, but locks different invariants and predates the
# current corpus. The allowlist is the discriminator.

_NEW = [
    "semianalysis_cc_traces_weka_with_subagents",
    "semianalysis_cc_traces_weka_with_subagents_256k",
    "semianalysis_cc_traces_weka_062126",
    "semianalysis_cc_traces_weka_062126_256k",
    "weka_trace",
]
_OLD = [  # the pre-062126 allowlist: same flags, older corpora
    "semianalysis_cc_traces_weka_with_subagents",
    "semianalysis_cc_traces_weka_with_subagents_256k",
    "weka_trace",
]


def _check(loaders, env=None):
    check_aiperf_capability(
        "/venv/bin/aiperf",
        loader_probe=lambda _b: loaders,
        probe=lambda _b: _CAPABLE_HELP,
        env=env or {},
    )


def test_pinned_allowlist_passes():
    _check(_NEW)


def test_pinned_allowlist_does_not_require_help_probe():
    check_aiperf_capability(
        "/venv/bin/aiperf",
        loader_probe=lambda _b: _NEW,
        probe=lambda _b: (_ for _ in ()).throw(OSError("help unavailable")),
        env={},
    )


def test_stale_build_is_rejected():
    """The exact case a flag probe waves through."""
    with pytest.raises(AgentXPreflightError) as ei:
        _check(_OLD)
    msg = str(ei.value)
    assert "stale build" in msg
    assert "semianalysis_cc_traces_weka_062126" in msg


def test_stale_build_is_rejected_even_when_the_pinned_corpus_is_admitted():
    """The silent path a run-scoped check alone leaves open.

    Upstream's own H100/H200 recipes pin an older corpus via
    WEKA_LOADER_OVERRIDE. A stale aiperf DOES admit that corpus, so asking only
    "is this run's corpus allowed" waves the stale build through and it replays
    under the wrong invariants. Build currency has to be asserted separately.
    """
    with pytest.raises(AgentXPreflightError) as ei:
        _check(_OLD, env={"WEKA_LOADER_OVERRIDE": "semianalysis_cc_traces_weka_with_subagents"})
    assert "stale build" in str(ei.value)


def test_current_build_accepts_an_older_corpus_pin():
    """On a current build an older corpus is a legitimate operator choice."""
    _check(_NEW, env={"WEKA_LOADER_OVERRIDE": "semianalysis_cc_traces_weka_with_subagents"})


def test_unknown_corpus_pin_is_rejected_before_the_server_boots():
    with pytest.raises(AgentXPreflightError) as ei:
        _check(_NEW, env={"WEKA_LOADER_OVERRIDE": "semianalysis_cc_traces_weka_nonexistent"})
    assert "not in the" in str(ei.value)


def test_agentx_dataset_outranks_weka_loader_override():
    with pytest.raises(AgentXPreflightError):
        _check(
            _NEW,
            env={
                "AGENTX_DATASET": "semianalysis_cc_traces_weka_nonexistent",
                "WEKA_LOADER_OVERRIDE": "semianalysis_cc_traces_weka_062126",
            },
        )


def test_unreadable_allowlist_falls_back_and_says_so(capsys):
    """Refusing outright would break setups that work today over what may be an
    unusual install layout -- but the weaker check must not pass silently."""
    check_aiperf_capability(
        "/venv/bin/aiperf",
        loader_probe=lambda _b: None,
        probe=lambda _b: _CAPABLE_HELP,
        env={},
    )
    assert "could not read" in capsys.readouterr().err


def test_unreadable_allowlist_still_rejects_a_flagless_build():
    with pytest.raises(AgentXPreflightError):
        check_aiperf_capability(
            "/venv/bin/aiperf",
            loader_probe=lambda _b: None,
            probe=lambda _b: "nothing useful here",
            env={},
        )


def test_loader_probe_survives_a_hung_interpreter(monkeypatch):
    """A timeout must degrade to the flag probe, not escape the check.

    ``subprocess.run(timeout=...)`` raises ``TimeoutExpired``, which descends
    from ``SubprocessError`` rather than ``OSError`` -- so catching only the
    latter let it propagate out of ``check_aiperf_capability`` and become a hard
    preflight failure, on exactly the input the timeout exists to handle.
    """
    import subprocess

    from hyperloom.inference_optimizer.agentx import preflight as pf

    def _hang(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="python", timeout=60)

    monkeypatch.setattr(pf.subprocess, "run", _hang)
    assert pf._default_loader_probe("/venv/bin/aiperf") is None


@pytest.mark.parametrize("external_override", [False, True])
def test_default_probes_scope_python_environment_to_managed_cli(tmp_path, monkeypatch, external_override):
    cli = _managed_cli(tmp_path)
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "PYTHONPLATLIBDIR", "__PYVENV_LAUNCHER__"):
        monkeypatch.setenv(key, "parent-pollution")
    monkeypatch.setenv("HYPERLOOM_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("AIPERF_BIN", raising=False)
    if external_override:
        monkeypatch.setenv("AIPERF_BIN", str(cli))
    original = dict(os.environ)
    calls = []

    def _run(argv, **kwargs):
        calls.append((argv, kwargs.get("env")))
        stdout = _CAPABLE_HELP if "profile" in argv else json.dumps(_NEW)
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(pf.subprocess, "run", _run)
    check_aiperf_capability(str(cli), env=os.environ, require_progress_api=True)
    assert calls[0][0][0] == str(cli.resolve().parent / "python")
    assert calls[1][0] == [str(cli), "profile", "--help"]
    for _, child_env in calls:
        assert child_env is not None
        assert child_env["PATH"] == original["PATH"]
        assert child_env.get("AIPERF_BIN") == original.get("AIPERF_BIN")
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "PYTHONPLATLIBDIR", "__PYVENV_LAUNCHER__"):
            assert child_env.get(key) == (original[key] if external_override else None)
    assert dict(os.environ) == original


@pytest.mark.parametrize("install_kind", ["legacy", "explicit"])
def test_loader_fallback_keeps_runtime_pythonpath(tmp_path, monkeypatch, install_kind):
    cli = _managed_cli(tmp_path)
    env = {"HYPERLOOM_STATE_DIR": str(tmp_path / "other-state"), "PYTHONPATH": "/host/packages"}
    if install_kind == "explicit":
        env.update(HYPERLOOM_STATE_DIR=str(tmp_path), AIPERF_BIN=str(cli))
    calls = []

    def _run(argv, **kwargs):
        calls.append((argv, kwargs.get("env")))
        return subprocess.CompletedProcess(argv, 1 if len(calls) == 1 else 0, json.dumps(_NEW), "")

    monkeypatch.setattr(pf.subprocess, "run", _run)
    check_aiperf_capability(str(cli), env=env)
    assert len(calls) == 2
    assert calls[0][1] == env
    assert calls[1][0][0] == sys.executable
    assert "-I" not in calls[1][0]
    assert calls[1][1] == env


@pytest.mark.parametrize("failure", ["missing", "timeout", "nonzero", "empty", "invalid-json", "not-a-list"])
def test_managed_loader_failure_cannot_be_blessed_by_host_python(tmp_path, monkeypatch, failure):
    cli = _managed_cli(tmp_path)
    env = {"HYPERLOOM_STATE_DIR": str(tmp_path), "PYTHONPATH": "/host/packages"}
    before = dict(env)
    sibling = str(cli.resolve().parent / "python")
    calls = []

    def _run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == sibling:
            if failure == "missing":
                raise FileNotFoundError(sibling)
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, 60)
            stdout = {"empty": "", "invalid-json": "broken import", "not-a-list": "{}"}.get(failure, json.dumps(_NEW))
            return subprocess.CompletedProcess(argv, 1 if failure == "nonzero" else 0, stdout, "")
        return subprocess.CompletedProcess(argv, 0, json.dumps(_NEW), "")

    monkeypatch.setattr(pf.subprocess, "run", _run)
    with pytest.raises(AgentXPreflightError, match="managed.*loader allowlist") as exc:
        check_aiperf_capability(str(cli), env=env, probe=lambda _bin: _CAPABLE_HELP)
    assert exc.value.repairable is True
    assert [argv[0] for argv in calls] == [sibling]
    assert env == before


def test_managed_unreadable_allowlist_does_not_degrade_to_flag_probe(tmp_path):
    cli = _managed_cli(tmp_path)
    help_calls = []

    def _probe(aiperf_bin):
        help_calls.append(aiperf_bin)
        return _CAPABLE_HELP

    with pytest.raises(AgentXPreflightError, match="managed.*loader allowlist") as exc:
        check_aiperf_capability(
            str(cli),
            env={"HYPERLOOM_STATE_DIR": str(tmp_path)},
            loader_probe=lambda _bin: None,
            probe=_probe,
        )
    assert exc.value.repairable is True
    assert help_calls == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX venv CLI shebang")
def test_managed_sibling_python_ignores_host_packages(tmp_path, monkeypatch):
    venv = tmp_path / "state" / "aiperf-venv"
    subprocess.run([sys.executable, "-I", "-m", "venv", "--without-pip", str(venv)], check=True, capture_output=True)
    python = venv / "bin" / "python"
    purelib = Path(
        subprocess.check_output(
            [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
        ).strip()
    )
    poison = tmp_path / "host-python310-packages"
    for root, loaders in ((purelib, _NEW), (poison, _OLD)):
        package = root / "aiperf" / "common"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").write_text("", encoding="utf-8")
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "scenario.py").write_text(
            f"from types import SimpleNamespace\ndef get_scenario(name):\n    return SimpleNamespace(require_loader={loaders!r})\n",
            encoding="utf-8",
        )
    cli = python.with_name("aiperf")
    cli.write_text(
        f"#!{python}\nfrom aiperf.common.scenario import get_scenario\n"
        f"assert get_scenario('test').require_loader == {_NEW!r}\nprint({_CAPABLE_HELP!r})\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    monkeypatch.setenv("PYTHONPATH", str(poison))
    env = dict(os.environ, HYPERLOOM_STATE_DIR=str(venv.parent), AIPERF_BIN="")
    env.update(PYTHONHOME="/missing-python-home", PYTHONUSERBASE=str(poison), PYTHONPLATLIBDIR="missing-lib")
    before = dict(env)
    check_aiperf_capability(str(cli), env=env, require_progress_api=True)
    assert env == before
    assert os.environ["PYTHONPATH"] == str(poison)


def test_hung_interpreter_reaches_the_flag_fallback(monkeypatch, capsys):
    """End to end: a hung probe must land on the documented fallback path."""
    import subprocess

    from hyperloom.inference_optimizer.agentx import preflight as pf

    monkeypatch.setattr(
        pf.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(subprocess.TimeoutExpired("python", 60)),
    )
    check_aiperf_capability(
        "/venv/bin/aiperf",
        probe=lambda _b: _CAPABLE_HELP,
        env={},
    )
    assert "could not read" in capsys.readouterr().err
