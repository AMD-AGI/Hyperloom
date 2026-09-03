# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the runtime repair of the AgentX aiperf dependency.

The defect these pin down: ``HYPERLOOM_AGENTX`` declares aiperf as a required,
version-pinned dependency and ``install.sh`` already knows how to install it,
but the install was gated on a *runtime* flag being true in the *installer's*
process. Provision without the flag, turn AgentX on later, and the preflight
fails with advice ("install it via install.sh") that no one on the path is able
to act on -- so a known dependency gap was routed into the enablement lane and
re-derived by an LLM specialist.
"""

from __future__ import annotations

import subprocess

import pytest
import yaml

from hyperloom.inference_optimizer.agentx import repair, runtime
from hyperloom.inference_optimizer.agentx.preflight import AgentXPreflightError

_DEPLOY = "hyperloom.inference_optimizer.agentx.deploy.deploy_agentx_assets"
_RESOLVE = "hyperloom.inference_optimizer.agentx.preflight.resolve_aiperf_bin"
_CHECK = "hyperloom.inference_optimizer.agentx.preflight.check_aiperf_capability"
_INSTALL = "hyperloom.inference_optimizer.agentx.repair.ensure_aiperf_installed"


@pytest.fixture(autouse=True)
def _clear_memos():
    runtime._PREFLIGHTED_BINS.clear()
    repair._REPAIR_RESULT.clear()
    yield
    runtime._PREFLIGHTED_BINS.clear()
    repair._REPAIR_RESULT.clear()


def _cfg(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "benchmark_script": "aiperf_client.sh"}}),
        encoding="utf-8",
    )
    return p


def _raiser(*errors):
    """Return a check_aiperf_capability stub raising ``errors`` in order.

    A ``None`` entry means "this call passes".
    """
    calls = {"n": 0}
    queue = list(errors)

    def _check(_bin, **_kwargs):
        calls["n"] += 1
        exc = queue.pop(0) if queue else None
        if exc is not None:
            raise exc

    _check.calls = calls
    return _check


# ── the installer contract: Python and the shell must agree on the flag ──────
def test_install_script_ships_the_only_aiperf_entrypoint():
    """The flag repair.py shells out with must exist in the packaged installer.

    Python and the shell are edited in different files; without this the two
    can drift and the repair would fail with an "unknown option" that looks
    like an environment problem rather than a packaging one.
    """
    script = repair.install_script_path()
    assert script.is_file(), f"packaged installer missing at {script}"
    text = script.read_text(encoding="utf-8")
    assert repair.ONLY_AIPERF_FLAG in text


def test_install_script_marks_an_explicitly_requested_aiperf_as_required():
    """A dependency the operator asked for must not fail soft.

    ``ensure_aiperf`` warned and returned on a failed install, so even the
    opt-in path left aiperf absent silently.
    """
    text = repair.install_script_path().read_text(encoding="utf-8")
    assert "AIPERF_REQUIRED" in text


def test_install_script_prewarms_aiperf_when_the_build_ships_the_client():
    """A default provision must install aiperf, not skip it.

    The gate used to read INSTALL_AIPERF / HYPERLOOM_AGENTX -- runtime mode
    flags -- to decide an install-time question. Nobody sets them while
    provisioning, because the mode is chosen later, per session. Measured on the
    incident cluster: 11 of 13 provisioning runs logged "aiperf (AgentX)
    skipped" and left a box that could not run AgentX at all.

    The presence of the shipped client is the install-time-knowable signal, so
    the default branch installs on it. Asserted against the packaged installer
    because this is the one behaviour the incident turned on.
    """
    text = repair.install_script_path().read_text(encoding="utf-8")
    assert "AGENTX_ASSET_DIR" in text
    # The default branch must reach ensure_aiperf, not just log a skip.
    default_branch = text.split('case "${_agx_want}:${_agx_sw}" in', 1)[1].split("esac", 1)[0]
    assert "pre-warming" in default_branch
    assert default_branch.count("ensure_aiperf") >= 2, "the no-flag branch never installs"


def test_agentx_assets_exist_so_a_source_checkout_prewarms():
    """The signal the installer keys on must actually be present in-tree.

    A rename of ``assets/agentx`` would silently turn the pre-warm back into a
    skip -- the same silent regression, one directory along.
    """
    from hyperloom.inference_optimizer.agentx.deploy import agentx_asset_dir

    assets = agentx_asset_dir()
    assert assets.is_dir(), f"AgentX assets missing at {assets}"
    # install.sh derives the same directory from its own location.
    assert assets == repair.install_script_path().parent / "agentx"


# ── repair mechanics ─────────────────────────────────────────────────────────
def test_repair_invokes_the_packaged_installer_with_only_aiperf(monkeypatch):
    seen = {}

    def _run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env") or {}
        seen["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(cmd, 0, "aiperf installed OK", "")

    monkeypatch.setattr(subprocess, "run", _run)
    assert repair.ensure_aiperf_installed(env={"PATH": "/opt/venv/bin"}) is None
    assert seen["cmd"][0] == "bash"
    assert seen["cmd"][1] == str(repair.install_script_path())
    assert seen["cmd"][2] == repair.ONLY_AIPERF_FLAG
    # The installer must see the opt-in so its own logs say why it ran.
    assert seen["env"]["INSTALL_AIPERF"] == "1"
    assert seen["timeout"] == repair.REPAIR_TIMEOUT_SEC


def test_repair_reports_a_nonzero_installer_with_its_output(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 3, "", "could not resolve host: github.com"),
    )
    err = repair.ensure_aiperf_installed(env={})
    assert err is not None
    assert "exited 3" in err
    assert "could not resolve host" in err


def test_repair_redacts_secrets_from_installer_output(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_TEST_TOKEN_FOR_REDACTION", "sk-supersecretvalue")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "pip failed with token sk-supersecretvalue"),
    )
    err = repair.ensure_aiperf_installed(env={})
    assert err is not None
    assert "sk-supersecretvalue" not in err


def test_repair_keeps_the_cause_when_warnings_push_it_out_of_the_tail(monkeypatch):
    """The line that gives the reason must survive a noisy installer.

    Shape taken from a measured run on a Python 3.10 box: the pip line naming
    the actual cause sat behind nine torch-gate warnings. A plain tail would
    have dropped exactly the sentence this summary exists to carry.
    """
    noise = "\n".join(f"[inference-optimizer WARN] torch gate note {i}" for i in range(20))
    cause = "ERROR: Package 'aiperf' requires a different Python: 3.10.12 not in '<3.14,>=3.11'"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, f"{cause}\n{noise}", ""),
    )
    err = repair.ensure_aiperf_installed(env={})
    assert err is not None
    assert "requires a different Python" in err, "the cause was truncated away"


def test_repair_caps_the_rescued_error_lines(monkeypatch):
    """Rescuing failure lines must not turn the summary back into a log dump."""
    many = "\n".join(f"ERROR: failure number {i}" for i in range(40))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, many, ""),
    )
    err = repair.ensure_aiperf_installed(env={})
    assert err is not None
    assert err.count("failure number") <= repair._OUTPUT_TAIL_LINES + repair._ERROR_LINE_BUDGET


def test_repair_reports_a_timeout_rather_than_hanging(monkeypatch):
    def _run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 0)

    monkeypatch.setattr(subprocess, "run", _run)
    err = repair.ensure_aiperf_installed(env={}, timeout_sec=7)
    assert err is not None and "within 7s" in err


def test_repair_reports_a_missing_installer(monkeypatch, tmp_path):
    monkeypatch.setattr(repair, "install_script_path", lambda: tmp_path / "absent.sh")
    err = repair.ensure_aiperf_installed(env={})
    assert err is not None and "missing" in err


def test_repair_runs_at_most_once_per_process(monkeypatch):
    n = {"runs": 0}

    def _run(cmd, **kwargs):
        n["runs"] += 1
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(subprocess, "run", _run)
    first = repair.ensure_aiperf_installed(env={})
    second = repair.ensure_aiperf_installed(env={})
    assert first == second
    assert n["runs"] == 1  # a retry cannot succeed where the first attempt failed


# ── preflight integration: the gap the incident actually walked through ──────
def test_missing_aiperf_is_installed_then_rechecked(tmp_path, monkeypatch):
    """The whole point: a self-declared dependency repairs itself."""
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/opt/venv/bin/aiperf")
    check = _raiser(AgentXPreflightError("aiperf was not found", repairable=True), None)
    monkeypatch.setattr(_CHECK, check)
    installs = {"n": 0}
    monkeypatch.setattr(_INSTALL, lambda **kw: installs.__setitem__("n", installs["n"] + 1))

    assert runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=_cfg(tmp_path)) is True
    assert installs["n"] == 1
    assert check.calls["n"] == 2  # re-checked after the install, not assumed fixed


def test_stale_build_is_reinstalled(tmp_path, monkeypatch):
    """A stale pin is a dependency gap too; ensure_aiperf force-reinstalls it."""
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/opt/venv/bin/aiperf")
    check = _raiser(AgentXPreflightError("is a stale build", repairable=True), None)
    monkeypatch.setattr(_CHECK, check)
    installs = {"n": 0}
    monkeypatch.setattr(_INSTALL, lambda **kw: installs.__setitem__("n", installs["n"] + 1))

    runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=_cfg(tmp_path))
    assert installs["n"] == 1


def test_operator_config_error_is_not_reinstalled(tmp_path, monkeypatch):
    """A corpus pin outside the allowlist is the operator's, not the build's.

    Reinstalling the same pinned build cannot change the verdict, so spending
    minutes on a pip install before failing would only delay the diagnosis.
    """
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/opt/venv/bin/aiperf")
    check = _raiser(AgentXPreflightError("the corpus pin 'typo' is not in the allowlist"))
    monkeypatch.setattr(_CHECK, check)
    installs = {"n": 0}
    monkeypatch.setattr(_INSTALL, lambda **kw: installs.__setitem__("n", installs["n"] + 1))

    with pytest.raises(AgentXPreflightError, match="corpus pin"):
        runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=_cfg(tmp_path))
    assert installs["n"] == 0
    assert check.calls["n"] == 1


def test_repair_failure_keeps_the_original_diagnosis(tmp_path, monkeypatch):
    """Both halves matter: what was missing, and why the fix did not land."""
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/opt/venv/bin/aiperf")
    monkeypatch.setattr(_CHECK, _raiser(AgentXPreflightError("aiperf was not found", repairable=True)))
    monkeypatch.setattr(_INSTALL, lambda **kw: "install.sh --only-aiperf exited 3: no network")

    with pytest.raises(AgentXPreflightError) as ei:
        runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=_cfg(tmp_path))
    message = str(ei.value)
    assert "aiperf was not found" in message
    assert "no network" in message
    # A repair that already failed must not invite another repair attempt.
    assert ei.value.repairable is False


def test_a_repaired_binary_is_memoized_like_any_other(tmp_path, monkeypatch):
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/opt/venv/bin/aiperf")
    check = _raiser(AgentXPreflightError("aiperf was not found", repairable=True), None)
    monkeypatch.setattr(_CHECK, check)
    monkeypatch.setattr(_INSTALL, lambda **kw: None)

    cfg = _cfg(tmp_path)
    runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg)
    runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg)
    assert check.calls["n"] == 2  # the second round reuses the post-repair verdict
