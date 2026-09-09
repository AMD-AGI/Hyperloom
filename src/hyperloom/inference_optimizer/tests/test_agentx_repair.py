# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the runtime repair of the AgentX aiperf dependency."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
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
    """Return a check_aiperf_capability stub raising ``errors`` in order."""
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
    """The flag repair.py shells out with must exist in the packaged installer."""
    script = repair.install_script_path()
    assert script.is_file(), f"packaged installer missing at {script}"
    text = script.read_text(encoding="utf-8")
    assert repair.ONLY_AIPERF_FLAG in text


def test_install_script_marks_an_explicitly_requested_aiperf_as_required():
    """A dependency the operator asked for must not fail soft."""
    text = repair.install_script_path().read_text(encoding="utf-8")
    assert "AIPERF_REQUIRED" in text


def test_install_script_prewarms_aiperf_when_the_build_ships_the_client():
    """A default provision must install aiperf, not skip it."""
    text = repair.install_script_path().read_text(encoding="utf-8")
    assert "AGENTX_ASSET_DIR" in text
    # The default branch must reach ensure_aiperf, not just log a skip.
    default_branch = text.split('case "${_agx_want}:${_agx_sw}" in', 1)[1].split("esac", 1)[0]
    assert "pre-warming" in default_branch
    assert default_branch.count("ensure_aiperf") >= 2, "the no-flag branch never installs"


def test_agentx_assets_exist_so_a_source_checkout_prewarms():
    """The signal the installer keys on must actually be present in-tree."""
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
    """The line that gives the reason must survive a noisy installer."""
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
    """An operator override is not a supply gap, and installing cannot close it."""
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
    """The corpus pin lives in the benchmark env, not this process's."""
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
    """A corpus pin outside the allowlist is the operator's, not the build's."""
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


# ── the installer gate, executed rather than read ──────────────────────────── The tests above assert on the TEXT of
# install.sh, which is what a packaging drift check can do.


def _bash(script: Path, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run ``script`` by bare name from its own directory."""
    return subprocess.run(["bash", script.name], cwd=str(script.parent), capture_output=True, text=True, env=env)


def _run_aiperf_gate(tmp_path: Path, *, env: dict[str, str], ships_client: bool = True) -> str:
    """Execute install.sh's aiperf gate in isolation and return what it logged."""
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
                    # `set -u` is what install.sh itself runs under, and is the only reason the ONLY_AIPERF defect was
                    # a hard failure rather than a silently-false branch.
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
        # Nobody set anything: the provisioning case the incident was measured in.
        ({}, True),
        ({"INSTALL_AIPERF": "1"}, True),
        ({"HYPERLOOM_AGENTX": "on"}, True),
        ({"HYPERLOOM_AGENTX": " ON "}, True),
        # Declining by name has to keep working.
        ({"INSTALL_AIPERF": "0"}, False),
        ({"INSTALL_AIPERF": "false"}, False),
        ({"INSTALL_AIPERF": " OFF "}, False),
        ({"HYPERLOOM_AGENTX": "no"}, False),
        # An unparseable value is not a decline; it falls to the default arm, which is the safe direction -- a typo
        # must not silently disarm AgentX.
        ({"INSTALL_AIPERF": "bogus"}, True),
    ],
)
def test_installer_aiperf_gate_truth_table(tmp_path, flags, expect_install):
    out = _run_aiperf_gate(tmp_path, env=flags)
    assert ("RAN ensure_aiperf" in out) is expect_install, out


def test_installer_gate_makes_an_explicit_request_fatal(tmp_path):
    """Asked for by name means AIPERF_REQUIRED=1, which is what makes ``ensure_aiperf`` die instead of warn."""
    out = _run_aiperf_gate(tmp_path, env={"INSTALL_AIPERF": "1"})
    assert "RAN ensure_aiperf required=1" in out


def test_installer_gate_prewarm_stays_non_fatal(tmp_path):
    """The default arm must not raise the flag: an interpreter that cannot supply aiperf must not block a provision that was never going to use it."""
    out = _run_aiperf_gate(tmp_path, env={})
    assert "RAN ensure_aiperf required=0" in out
    assert "pre-warming" in out


def test_installer_gate_skips_a_build_without_the_client(tmp_path):
    """The pre-warm keys on the shipped client, so a build without it says so rather than installing a dependency it has no use for."""
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
    """Run one install.sh function standalone, the way test_setup_cli does."""
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
    """The ``--only-aiperf`` early returns must not require ONLY_AIPERF to exist."""
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
    """A packaged install.sh that cannot be executed is a packaging fault."""

    def _boom(cmd, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(subprocess, "run", _boom)
    err = repair.ensure_aiperf_installed(env={})
    assert err is not None
    assert "could not run" in err
    assert "PermissionError" in err


def test_repair_says_so_when_a_failed_installer_printed_nothing(monkeypatch):
    """A silent non-zero exit must still name itself."""
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 2, "", "   \n\n"))
    err = repair.ensure_aiperf_installed(env={})
    assert err is not None
    assert err.endswith("(no output)"), err


def test_repair_keeps_stdout_and_stderr_on_separate_lines(monkeypatch):
    """A stdout tail without a trailing newline must not fuse into stderr."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "Collecting aiperf", "ERROR: no matching distribution"),
    )
    err = repair.ensure_aiperf_installed(env={})
    assert err is not None
    assert "aiperfERROR" not in err, "the two streams were concatenated without a separator"


def test_repair_replaces_an_empty_home_not_just_a_missing_one(monkeypatch):
    """``HOME=\"\"`` is not the same as unset, and setdefault treats it as set."""
    seen = {}

    def _run(cmd, **kw):
        seen["env"] = kw.get("env") or {}
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _run)
    repair.ensure_aiperf_installed(env={"HOME": ""})
    assert seen["env"]["HOME"], "an empty HOME was passed through to the installer"


def test_post_repair_failure_is_marked_unrepairable(tmp_path, monkeypatch):
    """The installer reported success and the build is still unusable."""
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
    """AgentX preparation must not be the thing that reports a broken config."""
    called = {"deploy": False}
    monkeypatch.setattr(_DEPLOY, lambda d: called.__setitem__("deploy", True))
    bad = tmp_path / "cfg.yaml"
    bad.write_text("benchmark: [unclosed", encoding="utf-8")

    assert runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=bad) is False
    assert called["deploy"] is False


# Run the real installer block with shell executables, not the host's Python/uv.
_FAKE_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail
printf 'python|%s' "$0" >> "$FAKE_CALLS"
printf '|%s' "$@" >> "$FAKE_CALLS"
printf '\n' >> "$FAKE_CALLS"
if [ "${1:-}" = -I ]; then shift; fi
case "${1:-}" in
  -c)
    case "$2" in
      *version_info*)
        version="$(cat "$0.version")"
        case "$version" in 3.11|3.12|3.13) exit 0 ;; *) exit 1 ;; esac ;;
      *expanduser*|*getpwuid*) printf '%s\n' "$FAKE_USER_HOME"; exit 0 ;;
    esac ;;
  -)
    source="$(cat)"
    case "$source" in
      *sys.base_prefix*) exit "$FAKE_PRIMARY_VENV" ;;
      *direct_url*)
        printf 'metadata|direct_url\n' >> "$FAKE_CALLS"
        root="${0%/bin/python}"
        test -f "$root/healthy" && { [ -z "${2:-}" ] || test "$(cat "$root/installed-ref")" = "$2"; }
        exit $? ;;
    esac ;;
  -m)
    [ "$2" = pip ] || exit 71
    [ "$FAKE_PIP" = 1 ] || { printf 'ERROR: No module named pip\n' >&2; exit 72; }
    case "$3" in
      --version) printf 'pip 22.0 from primary\n'; exit 0 ;;
      install)
        for name in PIP_CONFIG_FILE PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_NO_INDEX PIP_FIND_LINKS; do
          printf 'pip-policy|%s|%s\n' "$name" "${!name:-}" >> "$FAKE_CALLS"
        done
        target=''
        while [ "$#" -gt 0 ]; do
          case "$1" in --target) target="$2"; shift ;; esac
          shift
        done
        [ -n "$target" ] || { printf 'ERROR: refused to modify primary Python\n' >&2; exit 73; }
        [ "$FAKE_FAILURE" != bootstrap ] || { printf 'ERROR: uv wheel download blocked\n' >&2; exit 74; }
        mkdir -p "$target/bin"
        cp "$FAKE_ROOT/fake-uv" "$target/bin/uv"
        exit 0 ;;
    esac ;;
esac
printf 'ERROR: unexpected Python invocation\n' >&2
exit 75
"""

_FAKE_UV = r"""#!/usr/bin/env bash
set -euo pipefail
[ "${1:-}" = --no-config ] || { printf 'ERROR: uv config files were not disabled\n' >&2; exit 79; }
shift
printf 'uv' >> "$FAKE_CALLS"
printf '|%s' "$@" >> "$FAKE_CALLS"
printf '\n' >> "$FAKE_CALLS"
for name in UV_OFFLINE UV_PYTHON_DOWNLOADS UV_PYTHON_INSTALL_MIRROR UV_DEFAULT_INDEX UV_INDEX_URL UV_INDEX UV_EXTRA_INDEX_URL UV_FIND_LINKS SSL_CERT_FILE HTTPS_PROXY; do
  printf 'uv-policy|%s|%s\n' "$name" "${!name:-}" >> "$FAKE_CALLS"
done
for name in PYTHONHOME PYTHONPATH PYTHONUSERBASE PYTHONPLATLIBDIR __PYVENV_LAUNCHER__ PIP_TARGET PIP_PREFIX PIP_ROOT PIP_USER UV_SYSTEM_PYTHON UV_PYTHON UV_TARGET UV_PREFIX UV_PROJECT_ENVIRONMENT UV_MANAGED_PYTHON UV_NO_MANAGED_PYTHON UV_PYTHON_PREFERENCE; do
  [ -z "${!name:-}" ] || { printf 'ERROR: leaked %s\n' "$name" >&2; exit 80; }
done
printf 'uv-env|%s|%s|%s\n' "${UV_PYTHON_INSTALL_DIR:-}" "${UV_PYTHON_BIN_DIR:-}" "${UV_CACHE_DIR:-}" >> "$FAKE_CALLS"
case "$1 $2" in
  'python install')
    [ "$FAKE_FAILURE" != download ] || { printf 'ERROR: managed Python download blocked\n' >&2; exit 81; }
    mkdir -p "$UV_PYTHON_INSTALL_DIR/cpython/bin"
    cp "$FAKE_ROOT/fake-python" "$UV_PYTHON_INSTALL_DIR/cpython/bin/python"
    printf '3.11\n' > "$UV_PYTHON_INSTALL_DIR/cpython/bin/python.version" ;;
  'python find') printf '%s\n' "$UV_PYTHON_INSTALL_DIR/cpython/bin/python" ;;
  'venv '*)
    target="${!#}"
    mkdir -p "$target/bin"
    [ "$FAKE_FAILURE" != venv ] || { printf 'ERROR: venv creation failed\n' >&2; exit 82; }
    cp "$FAKE_ROOT/fake-python" "$target/bin/python"
    printf '3.11\n' > "$target/bin/python.version" ;;
  'pip install')
    if [ "$FAKE_FAILURE" = slow ]; then
      touch "$FAKE_ROOT/install-started"
      sleep 1
    fi
    [ "$FAKE_FAILURE" != install ] || { printf 'ERROR: aiperf download blocked\n' >&2; exit 83; }
    target=''
    while [ "$#" -gt 0 ]; do
      case "$1" in --python) target="$2"; shift ;; esac
      shift
    done
    [ -n "$target" ] || exit 84
    root="${target%/bin/python}"
    cp "$FAKE_ROOT/fake-aiperf" "$root/bin/aiperf"
    printf '%s\n' "$FAKE_REF" > "$root/installed-ref"
    if [ "$FAKE_FAILURE" = metadata ]; then printf 'no-vcs\n' > "$root/installed-ref"; fi
    touch "$root/healthy" ;;
  *) printf 'ERROR: unexpected uv invocation\n' >&2; exit 85 ;;
esac
"""

_FAKE_AIPERF = r"""#!/usr/bin/env bash
set -euo pipefail
printf 'aiperf|%s|%s\n' "$0" "$*" >> "$FAKE_CALLS"
[ -z "${PYTHONHOME:-}${PYTHONPATH:-}" ] || exit 86
[ ! -f "${0%/bin/aiperf}/broken-cli" ] || exit 87
printf 'weka-trace --scenario --benchmark-duration --api-host --api-port\n'
"""


@pytest.fixture
def isolated_aiperf(tmp_path):
    """A shell-only fixture safe on Windows and Linux, with no network access."""
    for name, body in (("fake-python", _FAKE_PYTHON), ("fake-uv", _FAKE_UV), ("fake-aiperf", _FAKE_AIPERF)):
        path = tmp_path / name
        path.write_text(body, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    primary = bin_dir / "primary"
    primary.write_text(_FAKE_PYTHON, encoding="utf-8", newline="\n")
    primary.chmod(0o755)
    (bin_dir / "primary.version").write_text("3.10", encoding="utf-8")

    def run(
        *,
        primary_version="3.10",
        primary_venv=False,
        candidates=None,
        uv_on_path=True,
        pip=True,
        failure="",
        required=True,
        mode="",
        override=False,
        state='"$PWD/state with spaces"',
        home='"$PWD/home"',
        polluted=False,
        package_spec=None,
        policy=None,
        real_flock=False,
        prepare_only=False,
    ):
        (bin_dir / "primary.version").write_text(primary_version, encoding="utf-8")
        for version, actual in (candidates or {}).items():
            candidate = bin_dir / f"python{version}"
            candidate.write_text(_FAKE_PYTHON, encoding="utf-8", newline="\n")
            candidate.chmod(0o755)
            (bin_dir / f"python{version}.version").write_text(actual, encoding="utf-8")
        uv_path = bin_dir / "uv"
        if uv_on_path:
            uv_path.write_text(_FAKE_UV, encoding="utf-8", newline="\n")
            uv_path.chmod(0o755)
        else:
            uv_path.unlink(missing_ok=True)
        text = repair.install_script_path().read_text(encoding="utf-8")
        pip_gate = text[text.index("PIP_EXTRA=()") : text.index("# --- 1. inference_optimizer")]
        block = text[text.index("# --- 2a. aiperf") : text.index("# --- 2b. Atomic-write")]
        polluted_vars = (
            "PYTHONHOME PYTHONPATH PYTHONUSERBASE PYTHONPLATLIBDIR __PYVENV_LAUNCHER__ "
            "PIP_TARGET PIP_PREFIX PIP_ROOT PIP_USER UV_SYSTEM_PYTHON UV_PYTHON "
            "UV_TARGET UV_PREFIX UV_PROJECT_ENVIRONMENT UV_CONFIG_FILE "
            "UV_MANAGED_PYTHON UV_NO_MANAGED_PYTHON UV_PYTHON_PREFERENCE"
        )
        policy_vars = (
            "UV_OFFLINE UV_PYTHON_DOWNLOADS UV_PYTHON_INSTALL_MIRROR UV_DEFAULT_INDEX UV_INDEX_URL "
            "UV_INDEX UV_EXTRA_INDEX_URL UV_FIND_LINKS SSL_CERT_FILE HTTPS_PROXY PIP_CONFIG_FILE "
            "PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_NO_INDEX PIP_FIND_LINKS"
        )
        runner = tmp_path / "isolated-install.sh"
        runner.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    f"unset {polluted_vars} {policy_vars}",
                    'export FAKE_ROOT="$PWD" FAKE_CALLS="${FAKE_CALLS:-$PWD/calls}" FAKE_USER_HOME="$PWD/fallback-home"',
                    ': > "$FAKE_CALLS"',
                    f"export FAKE_PIP={int(pip)} FAKE_FAILURE='{failure}' FAKE_REF=754356e9a39acc6cc6afb242d123bb57c3fb6f75",
                    f"export FAKE_PRIMARY_VENV={int(primary_venv)}",
                    'PYTHON="$PWD/bin/primary"',
                    f"export HYPERLOOM_STATE_DIR={state}",
                    f"export HOME={home}" if home is not None else "unset HOME",
                    f"ONLY_AIPERF=1; AIPERF_REQUIRED={int(required)}",
                    f"DRY_RUN={int(mode == 'dry')}; CHECK_ONLY={int(mode == 'check')}",
                    'AIPERF_REF="$FAKE_REF"; AIPERF_REPO="https://example.test/aiperf.git"',
                    f"AIPERF_PACKAGE_SPEC='{package_spec}'"
                    if package_spec
                    else 'AIPERF_PACKAGE_SPEC="aiperf @ git+${AIPERF_REPO}@$AIPERF_REF"',
                    'AIPERF_BIN="/operator/aiperf"' if override else "unset AIPERF_BIN",
                    'log() { printf "%s\\n" "$*"; }',
                    'warn() { printf "%s\\n" "$*" >&2; }',
                    'die() { warn "$*"; exit 99; }',
                    # Hide host candidates without hiding the shell's core utilities.
                    'command() { if [ "${1:-}" = -v ]; then case "$2" in uv|python3.11|python3.12|python3.13|aiperf) '
                    '[ -x "$PWD/bin/$2" ] || return 1; printf "%s\\n" "$PWD/bin/$2"; return ;; esac; fi; builtin command "$@"; }',
                    ":" if real_flock else 'flock() { printf "flock|%s\\n" "$*" >> "$FAKE_CALLS"; }',
                    pip_gate,
                    f"export {' '.join(name + '=polluted' for name in polluted_vars.split())}" if polluted else ":",
                    *[f"export {name}={shlex.quote(value)}" for name, value in (policy or {}).items()],
                    block,
                    "ensure_aiperf",
                    '[ "$PYTHON" = "$PWD/bin/primary" ]',
                    '[ "${PYTHONHOME:-}" = polluted ]' if polluted else ":",
                ]
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if prepare_only:
            return runner
        proc = _bash(runner)
        return proc, (tmp_path / "calls").read_text(encoding="utf-8")

    return run


def _assert_isolated_success(tmp_path, proc, calls):
    assert proc.returncode == 0, proc.stdout + proc.stderr
    state = tmp_path / "state with spaces"
    assert (state / "aiperf-venv/bin/aiperf").is_file()
    stamp = (state / "aiperf_installed_ref").read_text().splitlines()
    assert stamp == [
        "754356e9a39acc6cc6afb242d123bb57c3fb6f75",
        "aiperf @ git+https://example.test/aiperf.git@754356e9a39acc6cc6afb242d123bb57c3fb6f75",
    ]
    installs = [line for line in calls.splitlines() if line.startswith("uv|pip|install|")]
    assert len(installs) == 1, calls
    assert "/aiperf-venv/bin/python" in installs[0]
    assert "--no-deps" not in installs[0]
    assert "--seed" not in calls and "ensurepip" not in calls
    assert "--break-system-packages" not in calls


@pytest.mark.parametrize("primary_venv", [False, True])
def test_isolated_aiperf_uses_managed_python_for_310(tmp_path, isolated_aiperf, primary_venv):
    proc, calls = isolated_aiperf(primary_venv=primary_venv)
    _assert_isolated_success(tmp_path, proc, calls)
    assert "uv|python|install|" in calls and "|3.11" in calls
    assert "uv|python|find|" in calls
    assert "/aiperf-python/" in calls
    assert "/aiperf-python-bin|" in calls and "/aiperf-cache" in calls


@pytest.mark.parametrize("version", ["3.11", "3.12", "3.13"])
def test_isolated_aiperf_reuses_compatible_primary(tmp_path, isolated_aiperf, version):
    proc, calls = isolated_aiperf(primary_version=version)
    _assert_isolated_success(tmp_path, proc, calls)
    assert "uv|python|install" not in calls
    assert any("/bin/primary" in line for line in calls.splitlines() if line.startswith("uv|venv|"))


def test_isolated_aiperf_validates_path_candidate_versions(tmp_path, isolated_aiperf):
    proc, calls = isolated_aiperf(primary_version="3.14", candidates={"3.11": "3.10", "3.12": "3.12"})
    _assert_isolated_success(tmp_path, proc, calls)
    assert "uv|python|install" not in calls
    assert any("/bin/python3.12" in line for line in calls.splitlines() if line.startswith("uv|venv|"))


def test_isolated_aiperf_bootstraps_uv_privately_without_ensurepip(tmp_path, isolated_aiperf):
    proc, calls = isolated_aiperf(uv_on_path=False)
    _assert_isolated_success(tmp_path, proc, calls)
    bootstraps = [line for line in calls.splitlines() if "|-m|pip|install|" in line]
    assert len(bootstraps) == 1
    assert "--target|" in bootstraps[0] and "/aiperf-tools|" in bootstraps[0]
    assert "uv==0.12.3" in bootstraps[0]
    assert (tmp_path / "state with spaces/aiperf-tools/bin/uv").is_file()


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize("failure", ["bootstrap", "download", "venv", "install"])
def test_isolated_aiperf_failure_has_no_success_stamp(tmp_path, isolated_aiperf, required, failure):
    proc, calls = isolated_aiperf(uv_on_path=failure != "bootstrap", failure=failure, required=required)
    assert (proc.returncode != 0) is required, proc.stdout + proc.stderr
    assert "ERROR:" in proc.stderr
    assert "aiperf" in proc.stderr.lower()
    assert not (tmp_path / "state with spaces/aiperf_installed_ref").exists()
    assert "ensurepip" not in calls


def test_isolated_aiperf_missing_pip_does_not_bootstrap_system(tmp_path, isolated_aiperf):
    proc, calls = isolated_aiperf(uv_on_path=False, pip=False)
    assert proc.returncode != 0
    assert "pip" in proc.stderr and "uv" in proc.stderr
    assert "ensurepip" not in calls and "apt-get" not in calls
    assert not (tmp_path / "state with spaces/aiperf_installed_ref").exists()


def test_isolated_aiperf_path_uv_does_not_need_primary_pip(tmp_path, isolated_aiperf):
    proc, calls = isolated_aiperf(pip=False)
    _assert_isolated_success(tmp_path, proc, calls)
    assert "|-m|pip|" not in calls


def test_isolated_aiperf_same_pin_is_healthy_before_skip(tmp_path, isolated_aiperf):
    first, calls = isolated_aiperf()
    _assert_isolated_success(tmp_path, first, calls)
    second, calls = isolated_aiperf(pip=False, uv_on_path=False)
    assert second.returncode == 0, second.stderr
    assert "uv|" not in calls and "|-m|pip|" not in calls
    assert "aiperf|" in calls and "profile --help" in calls
    assert "direct_url" in calls


@pytest.mark.parametrize("broken", ["healthy", "bin/python", "bin/aiperf", "installed-ref", "broken-cli"])
def test_isolated_aiperf_repairs_broken_same_pin(tmp_path, isolated_aiperf, broken):
    first, calls = isolated_aiperf()
    _assert_isolated_success(tmp_path, first, calls)
    target = tmp_path / "state with spaces/aiperf-venv" / broken
    if broken == "broken-cli":
        target.touch()
    else:
        target.unlink()
    second, calls = isolated_aiperf()
    _assert_isolated_success(tmp_path, second, calls)


def test_isolated_aiperf_failed_repair_removes_stale_stamp(tmp_path, isolated_aiperf):
    first, calls = isolated_aiperf()
    _assert_isolated_success(tmp_path, first, calls)
    (tmp_path / "state with spaces/aiperf-venv/healthy").unlink()
    second, _ = isolated_aiperf(failure="install")
    assert second.returncode != 0
    assert not (tmp_path / "state with spaces/aiperf_installed_ref").exists()


def test_isolated_aiperf_leaves_unknown_venv_untouched(tmp_path, isolated_aiperf):
    unknown = tmp_path / "state with spaces/aiperf-venv"
    unknown.mkdir(parents=True)
    sentinel = unknown / "operator-data"
    sentinel.write_text("keep", encoding="utf-8")
    proc, calls = isolated_aiperf()
    assert proc.returncode != 0
    assert "refus" in proc.stderr.lower() and "aiperf-venv" in proc.stderr
    assert sentinel.read_text() == "keep"
    assert "uv|venv|" not in calls


@pytest.mark.parametrize("mode,override", [("dry", False), ("check", False), ("", True)])
def test_isolated_aiperf_readonly_and_override_do_not_install(tmp_path, isolated_aiperf, mode, override):
    proc, calls = isolated_aiperf(mode=mode, override=override, pip=False, uv_on_path=False)
    assert proc.returncode == 0, proc.stderr
    assert "uv|" not in calls and "|-m|pip|" not in calls
    assert not (tmp_path / "state with spaces").exists()


def test_isolated_aiperf_rejects_relative_state(tmp_path, isolated_aiperf):
    proc, calls = isolated_aiperf(state="relative-state")
    assert proc.returncode != 0
    assert "absolute" in proc.stderr and "HYPERLOOM_STATE_DIR" in proc.stderr
    assert not (tmp_path / "relative-state").exists()
    assert "uv|" not in calls


@pytest.mark.parametrize("home", ['""', None])
def test_isolated_aiperf_empty_or_missing_home_uses_user_home(tmp_path, isolated_aiperf, home):
    proc, _ = isolated_aiperf(state='""', home=home)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "fallback-home/.hyperloom/aiperf-venv/bin/aiperf").is_file()


def test_isolated_aiperf_cleans_child_environment_only(tmp_path, isolated_aiperf):
    proc, calls = isolated_aiperf(polluted=True)
    _assert_isolated_success(tmp_path, proc, calls)


def test_isolated_aiperf_does_not_trust_old_path_binary(tmp_path, isolated_aiperf):
    stale = tmp_path / "bin/aiperf"
    stale.write_text('#!/usr/bin/env bash\nprintf "stale PATH aiperf invoked\\n" >&2\nexit 0\n', encoding="utf-8")
    stale.chmod(0o755)
    proc, calls = isolated_aiperf(required=False)
    _assert_isolated_success(tmp_path, proc, calls)
    assert "stale PATH aiperf invoked" not in proc.stderr


def test_isolated_aiperf_preserves_package_spec_override(tmp_path, isolated_aiperf):
    spec = "aiperf @ https://example.test/custom.whl"
    first, calls = isolated_aiperf(package_spec=spec, failure="metadata")
    assert first.returncode == 0, first.stderr
    assert spec in calls
    second, calls = isolated_aiperf(package_spec=spec, failure="metadata")
    assert second.returncode == 0, second.stderr
    assert "uv|pip|install" not in calls
    changed, calls = isolated_aiperf(package_spec="aiperf @ https://example.test/other.whl", failure="metadata")
    assert changed.returncode == 0, changed.stderr
    assert "uv|pip|install" in calls


def test_isolated_aiperf_default_source_requires_real_pin(tmp_path, isolated_aiperf):
    proc, _ = isolated_aiperf(failure="metadata")
    assert proc.returncode != 0
    assert not (tmp_path / "state with spaces/aiperf_installed_ref").exists()


@pytest.mark.parametrize("layout", ["unknown", "symlink", "owned-symlink", "unknown-uv"])
def test_isolated_aiperf_rejects_unowned_tools(tmp_path, isolated_aiperf, layout):
    tools = tmp_path / "state with spaces/aiperf-tools"
    tools.parent.mkdir()
    target = tmp_path / "external" if "symlink" in layout else tools
    (target / "bin").mkdir(parents=True)
    sentinel = target / "bin/keep-me"
    sentinel.write_text("operator data", encoding="utf-8")
    if layout == "owned-symlink":
        (target / ".hyperloom-aiperf-tools").write_text("hyperloom-aiperf-v1\n", encoding="utf-8")
    if "symlink" in layout:
        if os.name != "posix":
            pytest.skip("requires POSIX symlinks")
        tools.symlink_to(target, target_is_directory=True)
    if layout == "unknown-uv":
        (target / "bin/uv").write_text(_FAKE_UV, encoding="utf-8", newline="\n")
        (target / "bin/uv").chmod(0o755)
    before = sorted(path.relative_to(target).as_posix() for path in target.rglob("*"))
    proc, calls = isolated_aiperf(uv_on_path=False)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "refus" in proc.stderr.lower() and "aiperf-tools" in proc.stderr
    assert "|-m|pip|install|" not in calls and "uv|" not in calls
    assert sentinel.read_text() == "operator data"
    assert sorted(path.relative_to(target).as_posix() for path in target.rglob("*")) == before


def test_isolated_aiperf_retries_owned_tools_bootstrap(tmp_path, isolated_aiperf):
    failed, _ = isolated_aiperf(uv_on_path=False, failure="bootstrap")
    assert failed.returncode != 0
    assert (tmp_path / "state with spaces/aiperf-tools/.hyperloom-aiperf-tools").is_file()
    retried, calls = isolated_aiperf(uv_on_path=False)
    _assert_isolated_success(tmp_path, retried, calls)


def test_isolated_aiperf_preserves_network_policy(tmp_path, isolated_aiperf):
    policy = {
        "UV_OFFLINE": "true",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_PYTHON_INSTALL_MIRROR": "https://python.example.test",
        "UV_DEFAULT_INDEX": "https://packages.example.test/simple",
        "UV_EXTRA_INDEX_URL": "https://extra.example.test/simple",
        "SSL_CERT_FILE": "/etc/company-ca.pem",
        "HTTPS_PROXY": "http://proxy.example.test:8080",
    }
    proc, calls = isolated_aiperf(policy=policy, primary_version="3.12")
    _assert_isolated_success(tmp_path, proc, calls)
    for name, value in policy.items():
        assert f"uv-policy|{name}|{value}\n" in calls


def test_isolated_aiperf_bootstrap_preserves_pip_config(tmp_path, isolated_aiperf):
    policy = {"PIP_CONFIG_FILE": "/etc/company-pip.conf", "PIP_INDEX_URL": "https://packages.example.test/simple"}
    proc, calls = isolated_aiperf(policy=policy, uv_on_path=False)
    _assert_isolated_success(tmp_path, proc, calls)
    for name, value in policy.items():
        assert f"pip-policy|{name}|{value}\n" in calls


@pytest.mark.parametrize("offline", ["true", "1"])
def test_isolated_aiperf_offline_cannot_bootstrap_uv(tmp_path, isolated_aiperf, offline):
    proc, calls = isolated_aiperf(policy={"UV_OFFLINE": offline}, uv_on_path=False)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "UV_OFFLINE" in proc.stderr
    assert "|-m|pip|install|" not in calls
    assert not (tmp_path / "state with spaces/aiperf_installed_ref").exists()


@pytest.mark.parametrize(
    "uv_indexes", [{}, {"UV_DEFAULT_INDEX": "https://uv.test/simple", "UV_INDEX": "https://extra-uv.test"}]
)
def test_isolated_aiperf_maps_pip_network_env_without_overriding_uv(tmp_path, isolated_aiperf, uv_indexes):
    pip_policy = {
        "PIP_INDEX_URL": "https://pip.test/simple",
        "PIP_EXTRA_INDEX_URL": "https://extra-pip.test/simple",
        "PIP_FIND_LINKS": "/company/wheelhouse",
    }
    proc, calls = isolated_aiperf(policy={**pip_policy, **uv_indexes})
    _assert_isolated_success(tmp_path, proc, calls)
    assert f"uv-policy|UV_DEFAULT_INDEX|{uv_indexes.get('UV_DEFAULT_INDEX', pip_policy['PIP_INDEX_URL'])}\n" in calls
    if uv_indexes:
        assert f"uv-policy|UV_INDEX|{uv_indexes['UV_INDEX']}\n" in calls
        assert "uv-policy|UV_EXTRA_INDEX_URL|\n" in calls
    else:
        assert f"uv-policy|UV_EXTRA_INDEX_URL|{pip_policy['PIP_EXTRA_INDEX_URL']}\n" in calls
    assert "uv-policy|UV_FIND_LINKS|/company/wheelhouse\n" in calls


@pytest.mark.parametrize("no_index", ["1", "true", "false", ""])
def test_isolated_aiperf_maps_pip_no_index_to_cli_only(tmp_path, isolated_aiperf, no_index):
    proc, calls = isolated_aiperf(policy={"PIP_NO_INDEX": no_index})
    _assert_isolated_success(tmp_path, proc, calls)
    installs = [line for line in calls.splitlines() if line.startswith("uv|pip|install|")]
    assert ("--no-index" in installs[0].split("|")) is (no_index in {"1", "true"})
    assert all("--no-index" not in line for line in calls.splitlines() if line.startswith("uv|python|"))


@pytest.mark.parametrize("home", [None, ""])
def test_repair_state_resolution_precedes_parent_home_fallback(tmp_path, monkeypatch, home):
    from hyperloom.inference_optimizer.agentx import preflight

    child_env = {"PATH": "/usr/bin"}
    if home is not None:
        child_env["HOME"] = home
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("HOME", str(tmp_path / "parent-home"))
    if os.name == "posix":
        import pwd
        from types import SimpleNamespace

        monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(user_home)))
    else:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user_home))
    # The installer must target the state directory the preflight will re-check.
    expected = preflight._aiperf_state_dir(child_env)
    seen = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: (seen.update(kw["env"]), subprocess.CompletedProcess(cmd, 0, "", ""))[1],
    )
    assert repair.ensure_aiperf_installed(env=child_env) is None
    assert preflight._aiperf_state_dir(seen) == expected
    assert seen.get("HYPERLOOM_STATE_DIR") == str(expected)
    assert child_env.get("HOME") == home


@pytest.mark.skipif(os.name != "posix" or not shutil.which("flock"), reason="requires POSIX flock")
def test_isolated_aiperf_real_flock_serializes_shared_state(tmp_path, isolated_aiperf):
    runner = isolated_aiperf(primary_version="3.12", failure="slow", real_flock=True, prepare_only=True)
    processes = []
    try:
        for index in range(2):
            processes.append(
                subprocess.Popen(
                    ["bash", runner.name],
                    cwd=tmp_path,
                    env={**os.environ, "FAKE_CALLS": str(tmp_path / f"calls-{index}")},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
            if index == 0:
                deadline = time.monotonic() + 10
                while not (tmp_path / "install-started").exists():
                    assert processes[0].poll() is None, "first installer exited before reaching installation"
                    assert time.monotonic() < deadline, "first installer never reached installation"
                    time.sleep(0.01)
        outputs = [process.communicate(timeout=20) for process in processes]
        assert all(process.returncode == 0 for process in processes), outputs
        calls = "".join((tmp_path / f"calls-{index}").read_text() for index in range(2))
        _assert_isolated_success(tmp_path, processes[0], calls)
        assert "skipping install" in outputs[1][0], outputs
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
