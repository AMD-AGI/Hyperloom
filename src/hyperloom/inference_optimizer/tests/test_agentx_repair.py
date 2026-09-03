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

import os
import subprocess
from pathlib import Path

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


def test_aiperf_bin_override_is_not_repaired(tmp_path, monkeypatch):
    """An operator override is not a supply gap, and installing cannot close it.

    ``ensure_aiperf`` returns 0 without doing anything when AIPERF_BIN is set,
    and ``resolve_aiperf_bin`` hands back that same binary afterwards. Repairing
    would report an install that never happened and point the reader away from
    the only thing that fixes it.
    """
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/custom/aiperf")
    monkeypatch.setattr(_CHECK, _raiser(AgentXPreflightError("aiperf was not found", repairable=True)))
    installs = {"n": 0}
    monkeypatch.setattr(_INSTALL, lambda **kw: installs.__setitem__("n", installs["n"] + 1))

    with pytest.raises(AgentXPreflightError) as ei:
        runtime.maybe_prepare_agentx(
            env={"AIPERF_BIN": "/custom/aiperf"},
            inferencex_path=str(tmp_path),
            config_path=_cfg(tmp_path),
        )
    assert installs["n"] == 0, "an install was attempted that cannot help"
    assert "AIPERF_BIN" in str(ei.value)
    assert ei.value.repairable is False


def test_capability_check_sees_the_child_env(tmp_path, monkeypatch):
    """The corpus pin lives in the benchmark env, not this process's.

    Without it the loader-allowlist admission check reads an empty override and
    a misspelled corpus sails through preflight to die after the server boots.
    """
    seen = {}
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/b/aiperf")
    monkeypatch.setattr(_CHECK, lambda b, **kw: seen.update(kw))

    child = {"WEKA_LOADER_OVERRIDE": "semianalysis_cc_traces_weka_062126"}
    runtime.maybe_prepare_agentx(env=child, inferencex_path=str(tmp_path), config_path=_cfg(tmp_path))
    assert seen.get("env") == child


def test_repair_supplies_home_to_the_installer(monkeypatch):
    """install.sh runs under ``set -u`` and expands ${HOME} for its state dir."""
    seen = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: (seen.update(kw.get("env") or {}), subprocess.CompletedProcess(cmd, 0, "", ""))[1],
    )
    repair.ensure_aiperf_installed(env={"PATH": "/usr/bin"})
    assert seen.get("HOME"), "the installer would die on HOME: unbound variable"


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


# ── the installer gate, executed rather than read ────────────────────────────
# The tests above assert on the TEXT of install.sh, which is what a packaging
# drift check can do. Both defects this gate has had were behavioural and
# invisible to a text match: a falsy INSTALL_AIPERF started installing once the
# pre-warm became the default arm, and a bare $ONLY_AIPERF in a sibling function
# broke every preflight test under `set -u`. So run the real block.


def _bash(script: Path, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run ``script`` by bare name from its own directory.

    Absolute paths are the one thing bash flavours disagree about -- WSL wants
    ``/mnt/c/...`` and Git Bash wants ``C:/...`` -- and a relative name works in
    every one of them, including the Linux CI runner.
    """
    return subprocess.run(["bash", script.name], cwd=str(script.parent), capture_output=True, text=True, env=env)


def _run_aiperf_gate(tmp_path: Path, *, env: dict[str, str], ships_client: bool = True) -> str:
    """Execute install.sh's aiperf gate in isolation and return what it logged.

    The block is sliced out of the packaged installer rather than restated, so a
    change to the real gate that nobody mirrors here fails the assertions
    instead of quietly passing against a stale copy.
    """
    text = repair.install_script_path().read_text(encoding="utf-8")
    start = text.index('if [ "$HYPERLOOM_BENCHMARK_BACKEND_LC" != "bypass" ]; then')
    end = text.index("\nensure_bench_serving_deps", start)

    asset_dir = tmp_path / "agentx"
    if ships_client:
        asset_dir.mkdir()

    runner = tmp_path / "gate.sh"
    with runner.open("w", encoding="utf-8", newline="\n") as f:
        f.write(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    # `set -u` is what install.sh itself runs under, and is the
                    # only reason the ONLY_AIPERF defect was a hard failure
                    # rather than a silently-false branch.
                    "set -uo pipefail",
                    f'HYPERLOOM_BENCHMARK_BACKEND_LC="{env.get("BACKEND", "vllm")}"',
                    f'INSTALL_AIPERF="{env.get("INSTALL_AIPERF", "")}"',
                    f'HYPERLOOM_AGENTX="{env.get("HYPERLOOM_AGENTX", "")}"',
                    f'AGENTX_ASSET_DIR="{asset_dir.name}"',
                    "AIPERF_REQUIRED=0",
                    'log() { echo "$*"; }',
                    'ensure_aiperf() { echo "RAN ensure_aiperf required=${AIPERF_REQUIRED}"; }',
                    text[start:end],
                ]
            )
            + "\n"
        )
    proc = _bash(runner)
    assert proc.returncode == 0, f"the gate itself failed: {proc.stderr}"
    return proc.stdout


@pytest.mark.parametrize(
    ("flags", "expect_install"),
    [
        # Nobody set anything: the provisioning case the incident was measured
        # in. This must install, and it is the whole point of the change.
        ({}, True),
        ({"INSTALL_AIPERF": "1"}, True),
        ({"HYPERLOOM_AGENTX": "on"}, True),
        ({"HYPERLOOM_AGENTX": " ON "}, True),
        # Declining by name has to keep working. Before the falsy arm existed
        # these fell through to the pre-warm and installed anyway.
        ({"INSTALL_AIPERF": "0"}, False),
        ({"INSTALL_AIPERF": "false"}, False),
        ({"INSTALL_AIPERF": " OFF "}, False),
        ({"HYPERLOOM_AGENTX": "no"}, False),
        # An unparseable value is not a decline; it falls to the default arm,
        # which is the safe direction -- a typo must not silently disarm AgentX.
        ({"INSTALL_AIPERF": "bogus"}, True),
    ],
)
def test_installer_aiperf_gate_truth_table(tmp_path, flags, expect_install):
    out = _run_aiperf_gate(tmp_path, env=flags)
    assert ("RAN ensure_aiperf" in out) is expect_install, out


def test_installer_gate_makes_an_explicit_request_fatal(tmp_path):
    """Asked for by name means AIPERF_REQUIRED=1, which is what makes
    ``ensure_aiperf`` die instead of warn."""
    out = _run_aiperf_gate(tmp_path, env={"INSTALL_AIPERF": "1"})
    assert "RAN ensure_aiperf required=1" in out


def test_installer_gate_prewarm_stays_non_fatal(tmp_path):
    """The default arm must not raise the flag: an interpreter that cannot
    supply aiperf must not block a provision that was never going to use it."""
    out = _run_aiperf_gate(tmp_path, env={})
    assert "RAN ensure_aiperf required=0" in out
    assert "pre-warming" in out


def test_installer_gate_skips_a_build_without_the_client(tmp_path):
    """The pre-warm keys on the shipped client, so a build without it says so
    rather than installing a dependency it has no use for."""
    out = _run_aiperf_gate(tmp_path, env={}, ships_client=False)
    assert "RAN ensure_aiperf" not in out
    assert "ships no" in out


def test_installer_gate_leaves_the_bypass_backend_alone(tmp_path):
    """A bypass run starts no server and benchmarks nothing."""
    out = _run_aiperf_gate(tmp_path, env={"BACKEND": "bypass", "INSTALL_AIPERF": "1"})
    assert out.strip() == ""


# ── the --only-aiperf preambles, also executed ───────────────────────────────
def _run_sliced_function(
    tmp_path: Path, opener: str, *, preamble: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run one install.sh function standalone, the way test_setup_cli does.

    That harness is the reason `${ONLY_AIPERF:-0}` has to be defaulted: it
    declares a handful of variables and nothing else, under `set -u`.
    """
    text = repair.install_script_path().read_text(encoding="utf-8")
    start = text.index(opener)
    name = opener.split("(", 1)[0]
    end = text.index(f"\n{name}", start) if f"\n{name}" in text[start:] else len(text)
    body = text[start : text.index("\n}\n", start) + 3]

    runner = tmp_path / f"{name}.sh"
    with runner.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(["#!/usr/bin/env bash", "set -uo pipefail", *preamble, body, name]) + "\n")
    return _bash(runner, env=env)


@pytest.mark.parametrize(
    "opener",
    ["preflight_validate_credentials() {", "ensure_torch_compatible_with_gpu() {"],
)
def test_only_aiperf_guards_tolerate_an_undeclared_flag(tmp_path, opener):
    """The ``--only-aiperf`` early returns must not require ONLY_AIPERF to exist.

    ``ONLY_AIPERF`` is assigned at the top of install.sh, but these functions
    are also run standalone: test_setup_cli.py slices
    ``preflight_validate_credentials`` into a generated script whose preamble
    declares a handful of variables and nothing else, under ``set -u``. A bare
    ``$ONLY_AIPERF`` there is an unbound variable, which took out all eight CI
    test shards -- five preflight tests died before reaching what they assert.

    So the preamble here deliberately does NOT declare it.
    """
    proc = _run_sliced_function(
        tmp_path,
        opener,
        preamble=[
            "REPO_ROOT=.",
            "CHECK_ONLY=0",
            "DRY_RUN=0",
            "PYTHON=python3",
            "CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-test",
            "log() { :; }",
            'warn() { echo "$*" >&2; }',
            'die() { echo "$*" >&2; exit 99; }',
            "preflight_load_dotenv() { :; }",
            "normalize_legacy_deepseek_env() { :; }",
            "preflight_reject_cross_provider() { :; }",
        ],
        env={k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "OPENAI_", "DEEPSEEK_"))},
    )
    assert "unbound variable" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, proc.stderr


def test_repair_reports_an_unrunnable_installer(monkeypatch):
    """A packaged install.sh that cannot be executed is a packaging fault.

    Real shape: a wheel unpacked without the executable bit, or a $TMPDIR
    mounted noexec. The OSError carries the only useful detail, and swallowing
    it would leave "repair failed" with nothing to act on.
    """

    def _boom(cmd, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(subprocess, "run", _boom)
    err = repair.ensure_aiperf_installed(env={})
    assert err is not None
    assert "could not run" in err
    assert "PermissionError" in err


def test_repair_says_so_when_a_failed_installer_printed_nothing(monkeypatch):
    """A silent non-zero exit must still name itself.

    Without the fallback the message ends at "exited 2: " -- a colon and a
    blank, which reads as a truncated log rather than as "the installer said
    nothing", and sends the reader looking for output that does not exist.
    """
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 2, "", "   \n\n"))
    err = repair.ensure_aiperf_installed(env={})
    assert err is not None
    assert err.endswith("(no output)"), err


def test_repair_keeps_stdout_and_stderr_on_separate_lines(monkeypatch):
    """A stdout tail without a trailing newline must not fuse into stderr.

    The first stderr line is usually the pip error this summary exists to
    carry; concatenating the streams glued it onto the end of an unrelated
    progress line, where a reader scanning for "ERROR:" at line start misses it.
    """
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "Collecting aiperf", "ERROR: no matching distribution"),
    )
    err = repair.ensure_aiperf_installed(env={})
    assert err is not None
    assert "aiperfERROR" not in err, "the two streams were concatenated without a separator"


def test_repair_replaces_an_empty_home_not_just_a_missing_one(monkeypatch):
    """``HOME=""`` is not the same as unset, and setdefault treats it as set.

    install.sh expands ``${HOME}/.hyperloom`` for its install stamp; an empty
    HOME makes that ``/.hyperloom``, which is unwritable, so the stamp never
    lands and every later provision redoes the install it was meant to record.
    """
    seen = {}

    def _run(cmd, **kw):
        seen["env"] = kw.get("env") or {}
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _run)
    repair.ensure_aiperf_installed(env={"HOME": ""})
    assert seen["env"]["HOME"], "an empty HOME was passed through to the installer"


def test_post_repair_failure_is_marked_unrepairable(tmp_path, monkeypatch):
    """The installer reported success and the build is still unusable.

    This is no longer a supply gap this process can close, and the distinction
    is load-bearing: the repair result is memoized as a success, so a caller
    that trusted ``repairable`` would re-enter the repair branch, get that
    memoized success handed straight back, and arrive here again having done
    nothing at all.
    """
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/opt/venv/bin/aiperf")
    monkeypatch.setattr(_INSTALL, lambda **kw: None)  # the install "succeeds"
    monkeypatch.setattr(
        _CHECK,
        _raiser(
            AgentXPreflightError("aiperf was not found", repairable=True),
            AgentXPreflightError("aiperf is not AgentX-capable", repairable=True),
        ),
    )

    with pytest.raises(AgentXPreflightError) as ei:
        runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=_cfg(tmp_path))

    assert ei.value.repairable is False, "a second repair would be attempted for nothing"
    assert "installed during this run and the check still fails" in str(ei.value)


def test_an_unreadable_config_is_left_to_magpie(tmp_path, monkeypatch):
    """AgentX preparation must not be the thing that reports a broken config.

    Magpie parses the same file moments later and says so with the context this
    module does not have; failing here first would replace that diagnosis with
    a traceback from the AgentX path, pointing at the wrong subsystem.
    """
    called = {"deploy": False}
    monkeypatch.setattr(_DEPLOY, lambda d: called.__setitem__("deploy", True))
    bad = tmp_path / "cfg.yaml"
    bad.write_text("benchmark: [unclosed", encoding="utf-8")

    assert runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=bad) is False
    assert called["deploy"] is False
