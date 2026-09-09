# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioral tests for the aiperf_client.sh asset, driven via bash with fakes."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.agentx.deploy import agentx_asset_dir

pytestmark = pytest.mark.skipif(os.name != "posix", reason="bash-driven; POSIX only")


def _write_exec(path: Path, content: str):
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _fake_builtin(write_pid: bool) -> str:
    # Emulates the builtin MAGPIE_RUN_PHASE=server phase: (optionally) record a tearable bg pid, then return.
    pid_line = 'sleep 300 & echo $! > "$MAGPIE_SERVER_PID_FILE"\n' if write_pid else ": no pid written\n"
    dump = (
        "{\n"
        '  echo "VLLM_HTTP_TIMEOUT_KEEP_ALIVE=${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-UNSET}"\n'
        '  echo "SGLANG_TIMEOUT_KEEP_ALIVE=${SGLANG_TIMEOUT_KEEP_ALIVE:-UNSET}"\n'
        '  echo "PATH=$PATH"\n'
        '  echo "PYTHON=${PYTHON:-UNSET}"\n'
        '  echo "PYTHONPATH=${PYTHONPATH:-UNSET}"\n'
        '  echo "PYTHONHOME=${PYTHONHOME:-UNSET}"\n'
        '  echo "PYTHONUSERBASE=${PYTHONUSERBASE:-UNSET}"\n'
        '} > "${AGENTX_TEST_SERVER_MARKER:-/dev/null}"\n'
    )
    return "#!/usr/bin/env bash\nset -e\n" + dump + pid_line + "exit 0\n"


_FAKE_AIPERF = r"""#!/usr/bin/env bash
# Record env markers, write a minimal export into --artifact-dir, exit rc.
# FAKE_AIPERF_SLEEP keeps the process alive long enough for the PROFILE branch
# to find it running; it is 0 for every other test.
sleep "${FAKE_AIPERF_SLEEP:-0}"
art=""
prev=""
for a in "$@"; do
  [ "$prev" = "--artifact-dir" ] && art="$a"
  prev="$a"
done
mkdir -p "$art"
echo '{"output_token_throughput":{"avg":1.0},"request_count":{"avg":1}}' > "$art/profile_export_aiperf.json"
printf '%s\n' "$@" > "$art/aiperf_args.txt"
{
  echo "CLI=$0"
  echo "PATH=$PATH"
  echo "PYTHON=${PYTHON:-UNSET}"
  echo "PYTHONPATH=${PYTHONPATH:-UNSET}"
  echo "PYTHONHOME=${PYTHONHOME:-UNSET}"
  echo "PYTHONUSERBASE=${PYTHONUSERBASE:-UNSET}"
  echo "PYTHONPLATLIBDIR=${PYTHONPLATLIBDIR:-UNSET}"
  echo "__PYVENV_LAUNCHER__=${__PYVENV_LAUNCHER__:-UNSET}"
  echo "AIPERF_BIN=${AIPERF_BIN:-UNSET}"
  echo "AIPERF_FOO=${AIPERF_FOO:-UNSET}"
  echo "AIPERF_DATASET_CONFIGURATION_TIMEOUT=${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-UNSET}"
  echo "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-UNSET}"
  echo "AIPERF_DATASET_MMAP_CACHE_DIR=${AIPERF_DATASET_MMAP_CACHE_DIR:-UNSET}"
  echo "AIPERF_HTTP_TCP_USER_TIMEOUT=${AIPERF_HTTP_TCP_USER_TIMEOUT:-UNSET}"
  echo "AIPERF_UI_REALTIME_METRICS_ENABLED=${AIPERF_UI_REALTIME_METRICS_ENABLED:-UNSET}"
} > "${AGENTX_TEST_MARKER}"
exit "${FAKE_RC:-0}"
"""

_FAKE_CURL = r"""#!/usr/bin/env bash
# /v1/models -> model json; profile endpoints -> ok. Records the argv of any
# /start_profile call so tests can assert what was (or was not) forwarded.
for a in "$@"; do case "$a" in *v1/models*) echo '{"data":[{"id":"m"}]}'; exit 0;; esac; done
for a in "$@"; do
  case "$a" in
    *start_profile*) printf '%s\n' "$@" > "${AGENTX_CURL_MARKER:-/dev/null}";;
    *stop_profile*) printf '%s\n' "$@" > "${AGENTX_CURL_STOP_MARKER:-/dev/null}";;
  esac
done
exit 0
"""

_FAKE_PHASE_GATE = r"""#!/usr/bin/env python3
import os
import sys
import json

if sys.argv[1] == "pick-port":
    print("19090")
    raise SystemExit(0)
if sys.argv[1] == "wait-phase":
    marker = os.environ.get("AGENTX_PHASE_GATE_ARGS_MARKER")
    if marker:
        with open(marker, "w", encoding="utf-8") as handle:
            json.dump(sys.argv, handle)
    if os.environ.get("FAKE_PHASE_GATE_FAIL") == "1":
        print("fake phase gate failure", file=sys.stderr)
        raise SystemExit(1)
    print("123456789")
    raise SystemExit(0)
if sys.argv[1] == "wait-capture-stop":
    if os.environ.get("FAKE_CAPTURE_GATE_FAIL") == "1":
        print("fake capture gate failure", file=sys.stderr)
        raise SystemExit(1)
    print('{"stop_reason":"wall_clock_limit","elapsed_seconds":0}')
    raise SystemExit(0)
if sys.argv[1] == "write-capture-status":
    def value(name):
        return sys.argv[sys.argv.index(name) + 1]
    with open(value("--output"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "capture_id": value("--capture-id"),
                "status": value("--status"),
                "reason": value("--reason"),
                "phase": "profiling",
                "phase_start_ns": int(value("--phase-start-ns")),
                "requested_window_seconds": float(value("--requested-window-seconds")),
                "decision": json.loads(value("--decision-json") or "{}"),
                "recorded_at_ns": 1,
            },
            handle,
        )
    raise SystemExit(0)
raise SystemExit(2)
"""

_FAKE_FUSER = "#!/usr/bin/env bash\nexit 0\n"


def _sandbox(tmp_path, *, write_pid=True, make_builtin=True):
    bench = tmp_path / "benchmarks"
    bind = tmp_path / "bin"
    res = tmp_path / "res"
    bench.mkdir()
    bind.mkdir()
    res.mkdir()
    shutil.copy2(agentx_asset_dir() / "aiperf_client.sh", bench / "aiperf_client.sh")
    shutil.copy2(agentx_asset_dir() / "map_aiperf.py", bench / "map_aiperf.py")
    (bench / "aiperf_phase_gate.py").write_text(_FAKE_PHASE_GATE, encoding="utf-8")
    if make_builtin:
        _write_exec(bench / "vllm_mi300x.sh", _fake_builtin(write_pid))
    _write_exec(bind / "aiperf", _FAKE_AIPERF)
    _write_exec(bind / "curl", _FAKE_CURL)
    _write_exec(bind / "fuser", _FAKE_FUSER)
    return bench, bind, res


def _run(bench, bind, res, tmp_path, **extra_env):
    env = dict(os.environ)
    # Drop the knobs under test before overlaying: a developer or CI box with WEKA_LOADER_OVERRIDE / AGENTX_DATASET
    # exported (both documented operator knobs) would otherwise fail the canonical-run assertions, and an inherited
    # AGENTX_NONCANONICAL_REASONS would make the deviation tests pass vacuously.
    for _k in [k for k in env if k.startswith("AGENTX_")]:
        env.pop(_k, None)
    env.pop("WEKA_LOADER_OVERRIDE", None)
    env["PATH"] = f"{bind}:{env.get('PATH', '')}"
    env.update(
        MODEL="/m",
        TP="1",
        PORT="8199",
        CONC="2",
        MAX_MODEL_LEN="4096",
        RESULT_DIR=str(res),
        RESULT_FILENAME="inferencex_result",
        FRAMEWORK="vllm",
        GPU_TYPE="mi300x",
        AIPERF_BIN=str(bind / "aiperf"),
        AGENTX_TEST_MARKER=str(tmp_path / "marker.txt"),
    )
    env.update(extra_env)
    env = {key: value for key, value in env.items() if value is not None}
    return subprocess.run(
        ["bash", str(bench / "aiperf_client.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_happy_path_writes_result(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    assert (res / "inferencex_result.json").exists()


def _managed_aiperf(state: Path) -> Path:
    cli = state / "aiperf-venv" / "bin" / "aiperf"
    cli.parent.mkdir(parents=True)
    _write_exec(cli, _FAKE_AIPERF)
    return cli


@pytest.mark.parametrize("custom_state", [False, True])
def test_managed_client_outranks_broken_path_without_changing_parent_env(tmp_path, custom_state):
    bench, bind, res = _sandbox(tmp_path)
    home = tmp_path / "home"
    state = tmp_path / "custom state" if custom_state else home / ".hyperloom"
    cli = _managed_aiperf(state)
    _write_exec(bind / "aiperf", "#!/bin/sh\nexit 91\n")
    before = dict(os.environ)
    r = _run(
        bench,
        bind,
        res,
        tmp_path,
        AIPERF_BIN=" \t\n",
        HOME=str(home),
        HYPERLOOM_STATE_DIR=str(state) if custom_state else "",
        AGENTX_TEST_SERVER_MARKER=str(tmp_path / "server.txt"),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    client = _server_env(tmp_path, tmp_path / "marker.txt")
    server = _server_env(tmp_path, tmp_path / "server.txt")
    assert client["CLI"] == str(cli)
    assert client["PATH"] == server["PATH"] == f"{bind}:{before.get('PATH', '')}"
    assert client["PYTHON"] == server["PYTHON"] == before.get("PYTHON", "UNSET")
    assert dict(os.environ) == before


@pytest.mark.parametrize("home", [None, ""])
def test_managed_client_missing_home_uses_isolated_system_user_lookup(tmp_path, home):
    bench, bind, res = _sandbox(tmp_path)
    cli = _managed_aiperf(tmp_path / "system-home" / ".hyperloom")
    user_python = bind / "user-python"
    _write_exec(
        user_python,
        f'#!/bin/sh\n[ "$1" = "-I" ] || exit 88\nprintf \'%s\\n\' {shlex.quote(str(tmp_path / "system-home"))}\n',
    )
    _write_exec(bind / "aiperf", "#!/bin/sh\nexit 91\n")
    r = _run(bench, bind, res, tmp_path, AIPERF_BIN=None, HOME=home, HYPERLOOM_STATE_DIR="", PYTHON=str(user_python))
    assert r.returncode == 0, r.stdout + r.stderr
    assert _server_env(tmp_path, tmp_path / "marker.txt")["CLI"] == str(cli)


@pytest.mark.parametrize(
    "state_env", [{"HYPERLOOM_STATE_DIR": "relative"}, {"HOME": "relative-home", "HYPERLOOM_STATE_DIR": ""}]
)
def test_client_rejects_relative_state(tmp_path, state_env):
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AIPERF_BIN="", **state_env)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "HYPERLOOM_STATE_DIR must be an absolute path" in r.stdout + r.stderr
    assert not (tmp_path / "marker.txt").exists()


@pytest.mark.parametrize("candidate_kind", ["missing", "directory", "not-executable"])
def test_managed_client_unusable_falls_back_to_path(tmp_path, candidate_kind):
    bench, bind, res = _sandbox(tmp_path)
    state = tmp_path / "state"
    if candidate_kind != "missing":
        cli = _managed_aiperf(state)
        if candidate_kind == "directory":
            cli.unlink()
            cli.mkdir()
        else:
            cli.chmod(0o644)
    r = _run(bench, bind, res, tmp_path, AIPERF_BIN=None, HYPERLOOM_STATE_DIR=str(state))
    assert r.returncode == 0, r.stdout + r.stderr
    assert _server_env(tmp_path, tmp_path / "marker.txt")["CLI"] == str(bind / "aiperf")


@pytest.mark.parametrize("broken_override", [False, True])
def test_managed_client_explicit_override_wins_and_keeps_pythonpath(tmp_path, broken_override):
    bench, bind, res = _sandbox(tmp_path)
    _managed_aiperf(tmp_path / "state")
    override = bind / "operator-aiperf"
    _write_exec(override, "#!/bin/sh\nexit 79\n" if broken_override else _FAKE_AIPERF)
    r = _run(
        bench,
        bind,
        res,
        tmp_path,
        AIPERF_BIN=f" \t{override}\n",
        HYPERLOOM_STATE_DIR=str(tmp_path / "state"),
        PYTHONPATH="/operator/python-packages",
    )
    assert r.returncode == (79 if broken_override else 0), r.stdout + r.stderr
    if not broken_override:
        client = _server_env(tmp_path, tmp_path / "marker.txt")
        assert client["CLI"] == str(override)
        assert client["PYTHONPATH"] == "/operator/python-packages"
    else:
        assert not (res / "inferencex_result.json").exists()


@pytest.mark.parametrize("profile", [False, True])
def test_managed_client_sanitizes_only_aiperf_python_environment(tmp_path, profile):
    bench, bind, res = _sandbox(tmp_path)
    cli = _managed_aiperf(tmp_path / "state")
    _write_exec(
        bind / "python3",
        '#!/bin/sh\nprintf "%s\\n" "${PYTHONPATH:-UNSET}" >> "$AGENTX_TEST_PYTHON_MARKER"\n'
        "exec env -u PYTHONHOME -u PYTHONPATH -u PYTHONUSERBASE -u PYTHONPLATLIBDIR -u __PYVENV_LAUNCHER__ "
        f'{shlex.quote(sys.executable)} "$@"\n',
    )
    pollution = {
        "PYTHONHOME": "/host/python310",
        "PYTHONPATH": "/host/python310/site-packages",
        "PYTHONUSERBASE": "/host/user-packages",
        "PYTHONPLATLIBDIR": "host-lib",
        "__PYVENV_LAUNCHER__": "/host/python310/bin/python",
    }
    run = _run_profile if profile else _run
    r = run(
        bench,
        bind,
        res,
        tmp_path,
        AIPERF_BIN="",
        HYPERLOOM_STATE_DIR=str(tmp_path / "state"),
        AGENTX_TEST_SERVER_MARKER=str(tmp_path / "server.txt"),
        AGENTX_TEST_PYTHON_MARKER=str(tmp_path / "python.txt"),
        **pollution,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    client = _server_env(tmp_path, tmp_path / "marker.txt")
    server = _server_env(tmp_path, tmp_path / "server.txt")
    assert client["CLI"] == str(cli)
    for key in pollution:
        assert client[key] == "UNSET"
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
        assert server[key] == pollution[key]
    assert set((tmp_path / "python.txt").read_text().splitlines()) == {pollution["PYTHONPATH"]}
    assert (res / "inferencex_result.json").exists()


def test_missing_builtin_exit_2(tmp_path):
    bench, bind, res = _sandbox(tmp_path, make_builtin=False)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 2


def test_no_pidfile_fail_loud_exit_3(tmp_path):
    bench, bind, res = _sandbox(tmp_path, write_pid=False)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 3
    assert not (res / "inferencex_result.json").exists()


def test_aiperf_failure_not_mapped(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, FAKE_RC="7")
    assert r.returncode == 7
    assert not (res / "inferencex_result.json").exists()


def test_scrub_keeps_aiperf_bin_drops_others(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AIPERF_FOO="leak")
    assert r.returncode == 0, r.stderr
    marker = (tmp_path / "marker.txt").read_text()
    assert "AIPERF_BIN=" in marker and "UNSET" not in marker.split("AIPERF_BIN=")[1].splitlines()[0]
    assert "AIPERF_FOO=UNSET" in marker  # stray AIPERF_* scrubbed


def test_tcp_user_timeout_survives_the_scrub(tmp_path):
    """The scrub must not leave aiperf on its 30s stock TCP_USER_TIMEOUT."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "AIPERF_HTTP_TCP_USER_TIMEOUT=900000" in (tmp_path / "marker.txt").read_text()


def test_tcp_user_timeout_is_tunable_through_the_agentx_name(tmp_path):
    """Operators tune it through AGENTX_, like every other knob in this file."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_HTTP_TCP_USER_TIMEOUT="1200000")
    assert r.returncode == 0, r.stderr
    assert "AIPERF_HTTP_TCP_USER_TIMEOUT=1200000" in (tmp_path / "marker.txt").read_text()


def test_inherited_tcp_user_timeout_does_not_win(tmp_path):
    """An inherited AIPERF_ copy is scrubbed; ours is authoritative."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AIPERF_HTTP_TCP_USER_TIMEOUT="30000")
    assert r.returncode == 0, r.stderr
    assert "AIPERF_HTTP_TCP_USER_TIMEOUT=900000" in (tmp_path / "marker.txt").read_text()


def test_gpu_type_uppercase_resolves_builtin(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, GPU_TYPE="MI300X")
    assert r.returncode == 0, r.stderr  # lowercased -> vllm_mi300x.sh found
    assert (res / "inferencex_result.json").exists()


def _aiperf_args(res):
    return (res / "aiperf_artifacts" / "aiperf_args.txt").read_text()


def test_no_max_context_length_flag(tmp_path):
    """AgentX must never cap the replay context from ``$MAX_MODEL_LEN``."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "--max-context-length" not in _aiperf_args(res)


def test_failed_request_threshold_is_passed(tmp_path):
    """A partial error storm must fail the run, not be scored as a clean result."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "--failed-request-threshold" in _aiperf_args(res)


def test_progress_api_is_enabled_on_an_ordinary_measurement_round(tmp_path):
    """Not just under PROFILE=1.

    ``phases.<name>.start_ns`` is the only authoritative phase boundary; the log
    line the KV collector otherwise greps for is written after the fact. Leaving
    the endpoint off outside trace capture meant every measured round attributed
    its warmup traffic by a timestamp that lagged the real transition.
    """
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr

    argv = _aiperf_args(res).splitlines()
    assert "--api-host" in argv
    assert "--api-port" in argv
    assert "19090" in argv  # the fake gate's pick-port


def test_progress_api_address_is_published_for_the_watchdog(tmp_path):
    """The reader is a different process and only sees this round's result dir."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr

    published = json.loads((res / "aiperf_artifacts" / "progress_api.json").read_text(encoding="utf-8"))
    assert published["url"] == "http://127.0.0.1:19090"


def test_run_survives_a_phase_gate_that_cannot_allocate_a_port(tmp_path):
    """Phase timing is observational: losing it must not cost the measurement."""
    bench, bind, res = _sandbox(tmp_path)
    (bench / "aiperf_phase_gate.py").write_text(
        "#!/usr/bin/env python3\nimport sys\nraise SystemExit(1)\n",
        encoding="utf-8",
    )
    r = _run(bench, bind, res, tmp_path)

    assert r.returncode == 0, r.stderr
    assert (res / "inferencex_result.json").exists()
    argv = _aiperf_args(res).splitlines()
    assert "--api-host" not in argv
    assert not (res / "aiperf_artifacts" / "progress_api.json").exists()


def test_run_survives_a_missing_phase_gate(tmp_path):
    """The asset can be absent on an older deployment; the round still measures."""
    bench, bind, res = _sandbox(tmp_path)
    (bench / "aiperf_phase_gate.py").unlink()
    r = _run(bench, bind, res, tmp_path)

    assert r.returncode == 0, r.stderr
    assert (res / "inferencex_result.json").exists()
    assert "--api-host" not in _aiperf_args(res).splitlines()


# The upstream contract, flag by flag.
_UPSTREAM_FLAGS = (
    ("--scenario", "inferencex-agentx-mvp"),
    ("--url", "http://localhost:8199"),
    ("--endpoint", "/v1/chat/completions"),
    ("--endpoint-type", "chat"),
    ("--model", "m"),  # probed from /v1/models, not $MODEL
    ("--tokenizer", "/m"),
    ("--public-dataset", "semianalysis_cc_traces_weka_062126_256k"),
    ("--num-dataset-entries", "393"),
    ("--concurrency", "2"),
    ("--benchmark-duration", "3600"),
    ("--random-seed", "42"),
    ("--trajectory-start-min-ratio", "0.25"),
    ("--trajectory-start-max-ratio", "0.75"),
    ("--warmup-requests-per-lane", "10"),
    ("--warmup-grace-period", "1800"),
    # Not scenario-locked, so nothing downstream would notice its removal: a trace carrying a 20-minute recorded idle
    # gap would replay it in full and, against a fixed duration window, silently cost measured requests.
    ("--trace-idle-gap-cap-seconds", "300"),
    ("--failed-request-threshold", "0.10"),
    ("--stats-interval", "30"),
    ("--slice-duration", "1.0"),
)

_UPSTREAM_BARE_FLAGS = (
    "--streaming",
    "--use-server-token-count",
    "--no-gpu-telemetry",
    "--tokenizer-trust-remote-code",
)


def test_upstream_flag_contract(tmp_path):
    """Every leaderboard-defining flag is present with the upstream value."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    argv = _aiperf_args(res).splitlines()
    for flag, value in _UPSTREAM_FLAGS:
        assert flag in argv, f"missing {flag}"
        assert argv[argv.index(flag) + 1] == value, f"{flag} != {value}"
    for flag in _UPSTREAM_BARE_FLAGS:
        assert flag in argv, f"missing {flag}"


def test_removed_warmup_flags_are_gone(tmp_path):
    """The old warmup pair measured a different thing; the scenario rejects it."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    argv = _aiperf_args(res)
    assert "--warmup-duration" not in argv
    assert "--num-warmup-sessions" not in argv


def test_corpus_defaults_to_256k_variant_for_unlisted_family(tmp_path):
    """An unmatched model family gets the capped corpus, like upstream."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)  # MODEL=/m -> not in the whitelist
    assert r.returncode == 0, r.stderr
    assert "semianalysis_cc_traces_weka_062126_256k" in _aiperf_args(res)


def test_corpus_full_variant_for_whitelisted_family(tmp_path):
    """The 1M-context families replay the unfiltered corpus."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, MODEL="/models/Kimi-K3")
    assert r.returncode == 0, r.stderr
    argv = _aiperf_args(res)
    assert "semianalysis_cc_traces_weka_062126" in argv
    assert "semianalysis_cc_traces_weka_062126_256k" not in argv


def test_corpus_override_wins(tmp_path):
    """WEKA_LOADER_OVERRIDE pins the loader regardless of family."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, WEKA_LOADER_OVERRIDE="weka_trace")
    assert r.returncode == 0, r.stderr
    assert "weka_trace" in _aiperf_args(res)


def test_aiperf_env_contract_survives_the_scrub(tmp_path):
    """The scrub must not eat the timeouts the corpus load needs."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    marker = (tmp_path / "marker.txt").read_text()
    assert "AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800" in marker
    assert "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800" in marker


def test_framework_sglang_delegates_to_sglang_builtin(tmp_path):
    """FRAMEWORK=sglang must delegate to sglang_{gpu}.sh, not the vllm default."""
    bench, bind, res = _sandbox(tmp_path, make_builtin=False)
    _write_exec(bench / "sglang_mi300x.sh", _fake_builtin(True))
    r = _run(bench, bind, res, tmp_path, FRAMEWORK="sglang")
    assert r.returncode == 0, r.stderr
    assert (res / "inferencex_result.json").exists()


def test_missing_framework_fail_loud(tmp_path):
    """FRAMEWORK unset must fail loud (exit 2), never silently boot the vllm builtin — the switch always injects FRAMEWORK from benchmark.framework."""
    bench, bind, res = _sandbox(tmp_path)  # vllm_mi300x.sh present
    r = _run(bench, bind, res, tmp_path, FRAMEWORK="")
    assert r.returncode == 2
    assert not (res / "inferencex_result.json").exists()


# --- smoke escape hatch ---------------------------------------------------------


def _result(res):
    import json

    return json.loads((res / "inferencex_result.json").read_text())


def test_default_run_is_not_flagged_unsafe(tmp_path):
    """The canonical 3600s run must stay submittable."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "--unsafe-override" not in _aiperf_args(res)
    assert not _result(res)["submission_invalid_reasons"]


def test_missing_conc_fails_loud(tmp_path):
    """A missing CONC must abort, not silently pick a concurrency."""
    bench, bind, res = _sandbox(tmp_path)
    env_without_conc = {"CONC": ""}
    r = _run(bench, bind, res, tmp_path, **env_without_conc)
    assert r.returncode != 0
    assert "CONC required" in (r.stderr + r.stdout)
    assert not (res / "inferencex_result.json").exists()


# --- non-canonical workloads may run, but may never be submittable -------------


def test_shrunken_corpus_cannot_keep(tmp_path):
    """A reduced trace count is a smoke, not a leaderboard measurement."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_NUM_ENTRIES="50")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert out["submission_valid"] is False
    assert any("entries=50" in x for x in out["submission_invalid_reasons"])


def test_forced_unsafe_override_at_canonical_duration_cannot_keep(tmp_path):
    """``--unsafe-override`` alone does NOT invalidate a run."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_UNSAFE_OVERRIDE="true")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert out["submission_valid"] is False
    assert any("unsafe_override_forced" in x for x in out["submission_invalid_reasons"])


def test_client_side_context_cap_cannot_keep(tmp_path):
    """An opt-in ``AGENTX_MAX_CTX`` drops traces, so it is non-canonical too."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_MAX_CTX="32768")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert out["submission_valid"] is False
    assert any("client_context_cap" in x for x in out["submission_invalid_reasons"])


def test_short_duration_opts_into_unsafe_override(tmp_path):
    """A sub-900s duration must be runnable as a smoke, not a startup abort."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_DURATION="120")
    assert r.returncode == 0, r.stderr
    argv = _aiperf_args(res).splitlines()
    assert "--unsafe-override" in argv
    assert argv[argv.index("--benchmark-duration") + 1] == "120"


def test_unsafe_override_can_be_forced_at_full_duration(tmp_path):
    """The operator escape hatch works independently of the duration."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_UNSAFE_OVERRIDE="true")
    assert r.returncode == 0, r.stderr
    assert "--unsafe-override" in _aiperf_args(res)


def test_realtime_metrics_survive_the_scrub(tmp_path):
    """Without this env the rolling stats block is skipped and ``--stats-interval`` is inert -- a 60-minute window emits nothing until it ends, so a merely slow run looks identical to a wedged one."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AIPERF_UI_REALTIME_METRICS_ENABLED="false")
    assert r.returncode == 0, r.stderr
    marker = (tmp_path / "marker.txt").read_text()
    assert "AIPERF_UI_REALTIME_METRICS_ENABLED=true" in marker


# --- PROFILE=1 self-bracketing ------------------------------------------------


def _run_profile(bench, bind, res, tmp_path, **extra_env):
    """PROFILE=1 with the window collapsed, so the branch runs in seconds."""
    capture_dir = res / "agentx-profile" / "test-capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    return _run(
        bench,
        bind,
        res,
        tmp_path,
        PROFILE="1",
        AGENTX_PROFILE_WINDOW_S="0",
        AGENTX_CAPTURE_ID="test-capture",
        AGENTX_CAPTURE_STATUS_PATH=str(capture_dir / "capture-status.json"),
        FAKE_AIPERF_SLEEP="6",
        AGENTX_CURL_MARKER=str(tmp_path / "curl.txt"),
        **extra_env,
    )


def _capture_status_path(res: Path) -> Path:
    return res / "agentx-profile" / "test-capture" / "capture-status.json"


def test_profile_forwards_capture_bounds_to_start_profile(tmp_path):
    """SGLang takes its capture bounds in the POST body, not on the serve line."""
    bench, bind, res = _sandbox(tmp_path)
    body = '{"start_step":0,"num_steps":128,"with_stack":true}'
    r = _run_profile(bench, bind, res, tmp_path, PROFILE_EXTRA_BODY=body)
    assert r.returncode == 0, r.stderr
    argv = (tmp_path / "curl.txt").read_text().splitlines()
    assert "-d" in argv
    assert argv[argv.index("-d") + 1] == body
    assert "Content-Type: application/json" in argv


@pytest.mark.parametrize("env", [{"PROFILE_EXTRA_BODY": "{}"}, {}])
def test_profile_posts_bare_when_there_are_no_bounds(tmp_path, env):
    """vLLM carries its bounds on --profiler-config; an empty body must not be posted as one, or the endpoint gets a meaningless payload."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run_profile(bench, bind, res, tmp_path, **env)
    assert r.returncode == 0, r.stderr
    argv = (tmp_path / "curl.txt").read_text().splitlines()
    assert "start_profile" in " ".join(argv)  # the call still happened
    assert "-d" not in argv


def test_profile_enables_the_aiperf_progress_api(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    r = _run_profile(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    argv = _aiperf_args(res).splitlines()
    assert argv[argv.index("--api-host") + 1] == "127.0.0.1"
    assert argv[argv.index("--api-port") + 1] == "19090"
    assert "AIPerf measured phase started" in (r.stdout + r.stderr)
    assert '"stop_reason":"wall_clock_limit"' in (r.stdout + r.stderr)


def test_profile_phase_wait_has_bounded_fallback(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    marker = tmp_path / "phase-gate-args.json"
    r = _run_profile(
        bench,
        bind,
        res,
        tmp_path,
        AGENTX_DATASET_CONFIG_TIMEOUT="7",
        AGENTX_WARMUP_GRACE_PERIOD="11",
        AGENTX_PHASE_GATE_ARGS_MARKER=str(marker),
    )
    assert r.returncode == 0, r.stderr
    argv = json.loads(marker.read_text())
    assert argv[argv.index("--timeout-seconds") + 1] == "3618"


def test_legacy_profile_warmup_delay_is_ignored(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    r = _run_profile(bench, bind, res, tmp_path, AGENTX_PROFILE_WARMUP_S="not-a-duration")
    assert r.returncode == 0, r.stderr
    assert "AGENTX_PROFILE_WARMUP_S is ignored" in (r.stdout + r.stderr)


def test_phase_gate_failure_keeps_measurement_but_skips_capture(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    marker = tmp_path / "curl.txt"
    r = _run_profile(
        bench,
        bind,
        res,
        tmp_path,
        FAKE_PHASE_GATE_FAIL="1",
    )
    assert r.returncode == 0, r.stderr
    assert (res / "inferencex_result.json").exists()
    assert not marker.exists()
    assert "without trace capture" in (r.stdout + r.stderr)
    capture = json.loads(_capture_status_path(res).read_text())
    assert capture["status"] == "failed"
    assert capture["reason"] == "profiling_phase_unavailable"


def test_capture_gate_failure_still_stops_profiler(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    stop_marker = tmp_path / "stop.txt"
    r = _run_profile(
        bench,
        bind,
        res,
        tmp_path,
        FAKE_CAPTURE_GATE_FAIL="1",
        AGENTX_CURL_STOP_MARKER=str(stop_marker),
    )
    assert r.returncode == 0, r.stderr
    assert stop_marker.exists()
    assert "stopping the profiler immediately" in (r.stdout + r.stderr)


def test_missing_phase_gate_leaves_status_for_executor_to_mark_missing(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    (bench / "aiperf_phase_gate.py").unlink()
    r = _run_profile(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    assert not _capture_status_path(res).exists()
    assert "cannot write trace-capture status" in (r.stdout + r.stderr)


def test_agentx_server_script_override_without_framework(tmp_path):
    """An explicit AGENTX_SERVER_SCRIPT still resolves when FRAMEWORK is unset."""
    bench, bind, res = _sandbox(tmp_path, make_builtin=False)
    _write_exec(bench / "custom_server.sh", _fake_builtin(True))
    r = _run(bench, bind, res, tmp_path, FRAMEWORK="", AGENTX_SERVER_SCRIPT="custom_server.sh")
    assert r.returncode == 0, r.stderr
    assert (res / "inferencex_result.json").exists()


def test_pinned_corpus_cannot_keep(tmp_path):
    """A different corpus is a different workload, and the scenario cannot object."""
    bench, bind, res = _sandbox(tmp_path)
    older = "semianalysis_cc_traces_weka_with_subagents_256k"
    r = _run(bench, bind, res, tmp_path, WEKA_LOADER_OVERRIDE=older)
    assert r.returncode == 0, r.stderr
    assert older in _aiperf_args(res)  # the pin is honoured
    out = _result(res)
    assert out["submission_valid"] is False  # but it cannot be submitted
    assert any("corpus=" in x for x in out["submission_invalid_reasons"])


def test_agentx_dataset_pin_cannot_keep(tmp_path):
    """The Hyperloom-side alias for the same knob gets the same treatment."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_DATASET="semianalysis_cc_traces_weka_062126")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert out["submission_valid"] is False
    assert any("corpus=" in x for x in out["submission_invalid_reasons"])


def test_default_corpus_is_canonical_and_submittable(tmp_path):
    """The unpinned path must not be demoted by the new check."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert not out["submission_invalid_reasons"]
    assert "semianalysis_cc_traces_weka_062126_256k" in _aiperf_args(res)


def test_canonical_pin_can_be_declared(tmp_path):
    """The family whitelist is a derivation, not a registry."""
    bench, bind, res = _sandbox(tmp_path)
    full = "semianalysis_cc_traces_weka_062126"
    r = _run(
        bench,
        bind,
        res,
        tmp_path,
        WEKA_LOADER_OVERRIDE=full,
        AGENTX_CANONICAL_DATASET=full,
    )
    assert r.returncode == 0, r.stderr
    assert full in _aiperf_args(res)
    out = _result(res)
    assert not out["submission_invalid_reasons"]
    assert out["submission_valid"] is not False


def test_declaring_canonical_does_not_excuse_a_different_pin(tmp_path):
    """Declaring one corpus canonical must not bless replaying another."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(
        bench,
        bind,
        res,
        tmp_path,
        AGENTX_CANONICAL_DATASET="semianalysis_cc_traces_weka_062126",
        WEKA_LOADER_OVERRIDE="semianalysis_cc_traces_weka_with_subagents_256k",
    )
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert out["submission_valid"] is False
    assert any("corpus=" in x for x in out["submission_invalid_reasons"])


def test_inherited_noncanonical_marker_does_not_leak_in(tmp_path):
    """The switch forwards every AGENTX_* key, so a stale marker must be cleared."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_NONCANONICAL_REASONS="entries=7(stale)")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert not out["submission_invalid_reasons"]
    assert out["submission_valid"] is not False


def test_reduced_warmup_is_flagged_non_canonical(tmp_path):
    """Warmup is measurement-defining, so trimming it must void submittability."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_WARMUP_REQUESTS_PER_LANE="1")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert out["submission_valid"] is False
    assert any("warmup_per_lane=1" in x for x in out["submission_invalid_reasons"]), out["submission_invalid_reasons"]


def test_reduced_warmup_grace_is_flagged_non_canonical(tmp_path):
    """Same for the drain window: a shorter grace truncates the warmup it gates."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_WARMUP_GRACE_PERIOD="60")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert out["submission_valid"] is False
    assert any("warmup_grace=60s" in x for x in out["submission_invalid_reasons"]), out["submission_invalid_reasons"]


def test_canonical_warmup_is_not_flagged(tmp_path):
    """The canonical values must not trip the new check (no false positive)."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(
        bench,
        bind,
        res,
        tmp_path,
        AGENTX_WARMUP_REQUESTS_PER_LANE="10",
        AGENTX_WARMUP_GRACE_PERIOD="1800",
    )
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert not out["submission_invalid_reasons"]
    assert out["submission_valid"] is not False


def test_raised_warmup_grace_is_not_flagged_non_canonical(tmp_path):
    """A *longer* grace period is more warmup, not less, and must not be flagged."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_WARMUP_GRACE_PERIOD="14400")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert not out["submission_invalid_reasons"]
    assert out["submission_valid"] is not False


def test_raised_warmup_per_lane_is_not_flagged_non_canonical(tmp_path):
    """Symmetric with the grace period: more warmup requests is not a deviation."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_WARMUP_REQUESTS_PER_LANE="20")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert not out["submission_invalid_reasons"]
    assert out["submission_valid"] is not False


def test_raised_failed_request_threshold_is_flagged_non_canonical(tmp_path):
    """Loosening the abort threshold is measurement-defining and carries no marker."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_FAILED_REQUEST_THRESHOLD="0.5")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert out["submission_valid"] is False
    assert any("failed_request_threshold=0.5" in x for x in out["submission_invalid_reasons"]), out[
        "submission_invalid_reasons"
    ]


def test_tightened_failed_request_threshold_is_not_flagged(tmp_path):
    """A stricter threshold measures a cleaner run, so it is not a deviation."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_FAILED_REQUEST_THRESHOLD="0.01")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert not out["submission_invalid_reasons"]
    assert out["submission_valid"] is not False


def test_canonical_failed_request_threshold_is_not_flagged(tmp_path):
    """Restating the canonical ratio must not trip the check, in either spelling."""
    for spelling in ("0.10", "0.1"):
        base = tmp_path / spelling.replace(".", "_")
        base.mkdir()
        bench, bind, res = _sandbox(base)
        r = _run(bench, bind, res, tmp_path, AGENTX_FAILED_REQUEST_THRESHOLD=spelling)
        assert r.returncode == 0, r.stderr
        out = _result(res)
        assert not out["submission_invalid_reasons"], spelling
        assert out["submission_valid"] is not False, spelling


def test_failed_request_threshold_cannot_inject_awk_code(tmp_path):
    """FRT reaches an awk program; it must be DATA, never program text."""
    canary = tmp_path / "pwned.txt"
    bench, bind, res = _sandbox(tmp_path)
    r = _run(
        bench,
        bind,
        res,
        tmp_path,
        AGENTX_FAILED_REQUEST_THRESHOLD=f'system("touch {canary}")',
    )
    assert not canary.exists(), "awk executed injected code"
    # And it must be rejected outright rather than silently treated as canonical.
    assert r.returncode != 0
    assert not res.joinpath("inferencex_result.json").exists()


@pytest.mark.parametrize(
    "knob,value",
    [
        ("AGENTX_WARMUP_REQUESTS_PER_LANE", "1.5"),
        ("AGENTX_WARMUP_REQUESTS_PER_LANE", "0x2"),
        ("AGENTX_WARMUP_GRACE_PERIOD", "1.5"),
        ("AGENTX_WARMUP_GRACE_PERIOD", "abc"),
        ("AGENTX_FAILED_REQUEST_THRESHOLD", "0.1.2"),
    ],
)
def test_non_integer_measurement_knobs_fail_loud(tmp_path, knob, value):
    """A malformed measurement-defining knob must stop the round, not be stamped."""
    base = tmp_path / f"{knob}_{value}".replace(".", "_").replace("/", "_")
    base.mkdir()
    bench, bind, res = _sandbox(base)
    r = _run(bench, bind, res, tmp_path, **{knob: value})
    assert r.returncode != 0, f"{knob}={value} was accepted"
    assert not res.joinpath("inferencex_result.json").exists()


def _server_env(tmp_path, marker: Path) -> dict[str, str]:
    """Parse the keep-alive env the fake builtin server phase inherited."""
    return dict(line.split("=", 1) for line in marker.read_text(encoding="utf-8").splitlines() if "=" in line)


def test_server_keep_alive_defaults_to_the_client_tolerance(tmp_path):
    """The server idle timeout must be raised before the server boots."""
    marker = tmp_path / "srv.txt"
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_TEST_SERVER_MARKER=str(marker))
    assert r.returncode == 0, r.stderr
    env = _server_env(tmp_path, marker)
    assert env["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"] == "900"
    # A vllm run must not carry the sglang spelling.
    assert env["SGLANG_TIMEOUT_KEEP_ALIVE"] == "UNSET"


def test_server_keep_alive_is_operator_overridable(tmp_path):
    """AGENTX_HTTP_KEEP_ALIVE_S sets it; an explicit framework knob wins outright."""
    marker = tmp_path / "knob.txt"
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_TEST_SERVER_MARKER=str(marker), AGENTX_HTTP_KEEP_ALIVE_S="120")
    assert r.returncode == 0, r.stderr
    assert _server_env(tmp_path, marker)["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"] == "120"

    pinned = tmp_path / "pinned"
    pinned.mkdir()
    marker2 = tmp_path / "pinned.txt"
    bench, bind, res = _sandbox(pinned)
    r = _run(
        bench,
        bind,
        res,
        tmp_path,
        AGENTX_TEST_SERVER_MARKER=str(marker2),
        AGENTX_HTTP_KEEP_ALIVE_S="120",
        VLLM_HTTP_TIMEOUT_KEEP_ALIVE="77",
    )
    assert r.returncode == 0, r.stderr
    assert _server_env(tmp_path, marker2)["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"] == "77"


def test_server_keep_alive_uses_the_frameworks_own_knob(tmp_path):
    """sglang names it differently; exporting the vllm spelling would be a no-op."""
    marker = tmp_path / "sg.txt"
    bench, bind, res = _sandbox(tmp_path, make_builtin=False)
    _write_exec(bench / "sglang_mi300x.sh", _fake_builtin(True))
    r = _run(bench, bind, res, tmp_path, FRAMEWORK="sglang", AGENTX_TEST_SERVER_MARKER=str(marker))
    assert r.returncode == 0, r.stderr
    env = _server_env(tmp_path, marker)
    assert env["SGLANG_TIMEOUT_KEEP_ALIVE"] == "900"
    assert env["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"] == "UNSET"


def test_keep_alive_follows_the_server_script_not_a_concatenation(tmp_path):
    """The exact mismatch that reached production."""
    marker = tmp_path / "mix.txt"
    bench, bind, res = _sandbox(tmp_path, make_builtin=False)
    _write_exec(bench / "sglang_mi300x.sh", _fake_builtin(True))
    r = _run(
        bench,
        bind,
        res,
        tmp_path,
        FRAMEWORK="vllm",
        AGENTX_SERVER_SCRIPT="sglang_mi300x.sh",
        AGENTX_TEST_SERVER_MARKER=str(marker),
    )
    assert r.returncode == 0, r.stderr
    env = _server_env(tmp_path, marker)
    assert env["SGLANG_TIMEOUT_KEEP_ALIVE"] == "900"
    assert env["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"] == "UNSET"


def test_a_framework_script_disagreement_is_said_out_loud(tmp_path):
    """Resolving it silently in either direction hides a misconfigured round."""
    marker = tmp_path / "warn.txt"
    bench, bind, res = _sandbox(tmp_path, make_builtin=False)
    _write_exec(bench / "sglang_mi300x.sh", _fake_builtin(True))
    r = _run(
        bench,
        bind,
        res,
        tmp_path,
        FRAMEWORK="vllm",
        AGENTX_SERVER_SCRIPT="sglang_mi300x.sh",
        AGENTX_TEST_SERVER_MARKER=str(marker),
    )
    assert r.returncode == 0, r.stderr
    assert "disagrees with the server script" in (r.stdout + r.stderr)


@pytest.mark.parametrize(
    "knob,value",
    [
        ("AGENTX_PROFILE_WINDOW_S", "20.5"),
        ("AGENTX_DURATION", "3600.0"),
    ],
)
def test_profile_window_knobs_fail_loud_rather_than_two_silent_ways(tmp_path, knob, value):
    """Both downstream constructs mishandle a non-integer, in opposite directions."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, PROFILE="1", **{knob: value})
    assert r.returncode == 2, r.stdout + r.stderr
    assert knob in (r.stdout + r.stderr)


# --- the trace has to finish writing before the server is torn down -----------


def test_the_client_waits_for_the_trace_to_stop_growing(tmp_path):
    """A 200 from /stop_profile means \"told to stop\", not \"written to disk\"."""
    bench, bind, res = _sandbox(tmp_path)
    trace = res / "torch_trace"
    trace.mkdir()
    grower = tmp_path / "grow.sh"
    grower.write_text(
        "#!/usr/bin/env bash\n"
        f"for i in 1 2 3 4 5 6; do printf 'x%.0s' $(seq 1 2000) >> '{trace}/r0.trace.json'; sleep 2; done\n",
        encoding="utf-8",
    )
    grower.chmod(0o755)
    subprocess.Popen(["bash", str(grower)])

    r = _run_profile(bench, bind, res, tmp_path, TP="1", AGENTX_TRACE_FLUSH_TIMEOUT_S="120")
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "trace flush complete" in out, out[-1500:]
    # It must not have declared completion on the first sample, while the file was still being appended to.
    assert "waiting for the profiler trace" in out


def test_a_stalled_flush_says_the_files_are_probably_truncated(tmp_path):
    """Timing out must be loud, and must name the knob."""
    bench, bind, res = _sandbox(tmp_path)
    trace = res / "torch_trace"
    trace.mkdir()
    # Two ranks expected, only one ever appears -> the count gate never passes.
    (trace / "r0.trace.json").write_text("partial", encoding="utf-8")

    r = _run_profile(bench, bind, res, tmp_path, TP="2", AGENTX_TRACE_FLUSH_TIMEOUT_S="20")
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "trace flush did not settle" in out, out[-1500:]
    assert "TRUNCATED" in out
    assert "AGENTX_TRACE_FLUSH_TIMEOUT_S" in out
    capture = json.loads(_capture_status_path(res).read_text())
    assert capture["reason"] == "trace_flush_timeout"


def test_a_missing_rank_is_not_accepted_as_settled(tmp_path):
    """Ranks serialise one at a time, so \"not growing\" is not \"complete\"."""
    bench, bind, res = _sandbox(tmp_path)
    trace = res / "torch_trace"
    trace.mkdir()
    (trace / "r0.trace.json").write_text("done", encoding="utf-8")

    r = _run_profile(bench, bind, res, tmp_path, TP="8", AGENTX_TRACE_FLUSH_TIMEOUT_S="20")
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "trace flush did not settle" in out, out[-1500:]
    assert "expected 8 ranks" in out


def test_the_wait_is_skipped_when_not_profiling(tmp_path):
    """Measurement rounds must not pay for a capture they never took."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "waiting for the profiler trace" not in (r.stdout + r.stderr)


def test_no_configured_trace_dir_is_not_waited_on(tmp_path):
    """With no profiler output directory there is nothing that can ever settle."""
    bench, bind, res = _sandbox(tmp_path)
    assert not (res / "torch_trace").exists()

    r = _run_profile(bench, bind, res, tmp_path, TP="8")
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "no profiler output directory is configured" in out, out[-1500:]
    # Crucially, it must not have entered the polling loop at all: the default AGENTX_TRACE_FLUSH_TIMEOUT_S is 1800s
    # and this test does not lower it.
    assert "waiting for the profiler trace" not in out
    capture = json.loads(_capture_status_path(res).read_text())
    assert capture["reason"] == "profiler_output_unconfigured"


def test_configured_trace_dir_is_waited_on_before_it_exists(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    configured = tmp_path / "profiler-output-created-during-flush"
    r = _run_profile(
        bench,
        bind,
        res,
        tmp_path,
        SGLANG_TORCH_PROFILER_DIR=str(configured),
        AGENTX_TRACE_FLUSH_TIMEOUT_S="1",
        AGENTX_TRACE_FIRST_FILE_TIMEOUT_S="1",
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "waiting for the profiler trace" in out
    assert "no trace file appeared" in out
    assert "no profiler output directory is configured" not in out
    capture = json.loads(_capture_status_path(res).read_text())
    assert capture["reason"] == "trace_files_missing"


def test_a_capture_that_produces_nothing_gives_up_early(tmp_path):
    """Zero files is a failed capture, not a slow one; bound it separately."""
    bench, bind, res = _sandbox(tmp_path)
    (res / "torch_trace").mkdir()  # exists, but nothing ever lands in it

    r = _run_profile(
        bench,
        bind,
        res,
        tmp_path,
        TP="8",
        AGENTX_TRACE_FLUSH_TIMEOUT_S="600",
        AGENTX_TRACE_FIRST_FILE_TIMEOUT_S="15",
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "no trace file appeared within 15s" in out, out[-1500:]
    # The shorter first-file bound must win over the flush budget, not the other way round.
    assert "trace flush did not settle" not in out


def test_the_first_file_bound_never_exceeds_the_flush_budget(tmp_path):
    """An operator who lowers only the flush budget must still get that bound."""
    bench, bind, res = _sandbox(tmp_path)
    (res / "torch_trace").mkdir()

    r = _run_profile(bench, bind, res, tmp_path, TP="8", AGENTX_TRACE_FLUSH_TIMEOUT_S="15")
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "first-file bound 15s" in out, out[-1500:]
    assert "no trace file appeared within 15s" in out
