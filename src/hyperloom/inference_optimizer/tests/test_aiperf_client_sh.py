# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioral tests for the aiperf_client.sh asset, driven via bash with fakes.

Covers the shell invariants unit tests can't reach: missing-builtin exit, no-pid
fail-loud, aiperf rc gating, happy-path mapping, AIPERF_* scrub keeping
AIPERF_BIN, GPU_TYPE lowercasing, warmup-flag gating, and builtin resolution
from FRAMEWORK / AGENTX_SERVER_SCRIPT. POSIX-only (skipped elsewhere).
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.agentx.deploy import agentx_asset_dir

pytestmark = pytest.mark.skipif(os.name != "posix", reason="bash-driven; POSIX only")


def _write_exec(path: Path, content: str):
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _fake_builtin(write_pid: bool) -> str:
    # Emulates the builtin MAGPIE_RUN_PHASE=server phase: (optionally) record a
    # tearable bg pid, then return. Only the server phase is exercised.
    pid_line = 'sleep 300 & echo $! > "$MAGPIE_SERVER_PID_FILE"\n' if write_pid else ": no pid written\n"
    return "#!/usr/bin/env bash\nset -e\n" + pid_line + "exit 0\n"


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
  echo "AIPERF_BIN=${AIPERF_BIN:-UNSET}"
  echo "AIPERF_FOO=${AIPERF_FOO:-UNSET}"
  echo "AIPERF_DATASET_CONFIGURATION_TIMEOUT=${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-UNSET}"
  echo "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-UNSET}"
  echo "AIPERF_DATASET_MMAP_CACHE_DIR=${AIPERF_DATASET_MMAP_CACHE_DIR:-UNSET}"
  echo "AIPERF_UI_REALTIME_METRICS_ENABLED=${AIPERF_UI_REALTIME_METRICS_ENABLED:-UNSET}"
} > "${AGENTX_TEST_MARKER}"
exit "${FAKE_RC:-0}"
"""

_FAKE_CURL = r"""#!/usr/bin/env bash
# /v1/models -> model json; profile endpoints -> ok. Records the argv of any
# /start_profile call so tests can assert what was (or was not) forwarded.
for a in "$@"; do case "$a" in *v1/models*) echo '{"data":[{"id":"m"}]}'; exit 0;; esac; done
for a in "$@"; do
  case "$a" in *start_profile*) printf '%s\n' "$@" > "${AGENTX_CURL_MARKER:-/dev/null}";; esac
done
exit 0
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
    if make_builtin:
        _write_exec(bench / "vllm_mi300x.sh", _fake_builtin(write_pid))
    _write_exec(bind / "aiperf", _FAKE_AIPERF)
    _write_exec(bind / "curl", _FAKE_CURL)
    _write_exec(bind / "fuser", _FAKE_FUSER)
    return bench, bind, res


def _run(bench, bind, res, tmp_path, **extra_env):
    env = dict(os.environ)
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


def test_gpu_type_uppercase_resolves_builtin(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, GPU_TYPE="MI300X")
    assert r.returncode == 0, r.stderr  # lowercased -> vllm_mi300x.sh found
    assert (res / "inferencex_result.json").exists()


def _aiperf_args(res):
    return (res / "aiperf_artifacts" / "aiperf_args.txt").read_text()


def test_no_max_context_length_flag(tmp_path):
    """AgentX must never cap the replay context from ``$MAX_MODEL_LEN``.

    ``--max-context-length`` makes aiperf DROP every trace whose peak exceeds
    it (not truncate), and ``$MAX_MODEL_LEN`` is itself derived from the
    synthetic ISL+OSL shape the agentic corpus never uses. Emitting the flag
    therefore shrinks the 393-trace corpus to its short-trace tail while every
    status marker still reports a clean run. Upstream's agentic path unsets
    ``MAX_MODEL_LEN`` and never emits the flag; the server's own context window
    is the only limit that may apply.
    """
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "--max-context-length" not in _aiperf_args(res)


def test_failed_request_threshold_is_passed(tmp_path):
    """A partial error storm must fail the run, not be scored as a clean result.

    aiperf defaults ``--failed-request-threshold`` to None, which DISABLES the
    check, so without the flag a run whose requests mostly 4xx still exits 0
    and is mapped as a normal measurement. ``map_aiperf.py`` carries no error
    counters, so nothing downstream can notice.
    """
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "--failed-request-threshold" in _aiperf_args(res)


# The upstream contract, flag by flag. A golden list rather than scattered
# substring checks: the failure mode this guards against is a flag quietly
# going missing, which no individual assertion would notice.
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
    # Not scenario-locked, so nothing downstream would notice its removal: a
    # trace carrying a 20-minute recorded idle gap would replay it in full and,
    # against a fixed duration window, silently cost measured requests.
    ("--trace-idle-gap-cap-seconds", "300"),
    ("--failed-request-threshold", "0.10"),
    ("--stats-interval", "30"),
    ("--slice-duration", "1.0"),
)

_UPSTREAM_BARE_FLAGS = ("--streaming", "--use-server-token-count", "--no-gpu-telemetry",
                        "--tokenizer-trust-remote-code")


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
    """The old warmup pair measured a different thing; the scenario rejects it.

    Kept as an explicit assertion rather than deleting the coverage outright,
    so a re-introduction has to argue with a red test.
    """
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
    """FRAMEWORK unset must fail loud (exit 2), never silently boot the vllm
    builtin — the switch always injects FRAMEWORK from benchmark.framework."""
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
    """A missing CONC must abort, not silently pick a concurrency.

    Concurrency is measurement-defining and upstream makes it a hard requirement.
    A default would produce a full scenario-locked run at a concurrency nobody
    chose, and the mapped result records no concurrency at all, so the mismatch
    would be invisible afterwards.
    """
    bench, bind, res = _sandbox(tmp_path)
    env_without_conc = {"CONC": ""}
    r = _run(bench, bind, res, tmp_path, **env_without_conc)
    assert r.returncode != 0
    assert "CONC required" in (r.stderr + r.stdout)
    assert not (res / "inferencex_result.json").exists()


# --- non-canonical workloads may run, but may never be submittable -------------
#
# aiperf cannot judge these: the scenario has no concept of corpus size, and it
# stamps a False verdict only when --unsafe-override actually suppressed a
# violation. So the client reports the deviation and map_aiperf forces it.


def test_shrunken_corpus_cannot_keep(tmp_path):
    """A reduced trace count is a smoke, not a leaderboard measurement.

    Without this the corpus could be cut to a handful of traces while
    ``submission_valid`` stayed true -- exactly the failure this whole path
    exists to prevent, arriving through the one knob the scenario cannot see.
    """
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_NUM_ENTRIES="50")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert out["submission_valid"] is False
    assert any("entries=50" in x for x in out["submission_invalid_reasons"])


def test_forced_unsafe_override_at_canonical_duration_cannot_keep(tmp_path):
    """``--unsafe-override`` alone does NOT invalidate a run.

    aiperf stamps the verdict false only when the override suppressed a real
    violation, so forcing it at 3600s -- where there is nothing to suppress --
    would otherwise leave a fully KEEP-able result while the log claimed the
    opposite.
    """
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
    """A sub-900s duration must be runnable as a smoke, not a startup abort.

    The scenario enforces a 900s floor, so without the flag ``AGENTX_DURATION``
    below it aborts before the first request and this path cannot be smoke
    tested at all. Upstream opts in below the floor; the scenario then stamps
    ``submission_valid`` false, which ``benchmark_result.py`` rejects -- so the
    escape hatch cannot be mistaken for a leaderboard measurement.
    """
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
    """Without this env the rolling stats block is skipped and
    ``--stats-interval`` is inert -- a 60-minute window emits nothing until it
    ends, so a merely slow run looks identical to a wedged one."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AIPERF_UI_REALTIME_METRICS_ENABLED="false")
    assert r.returncode == 0, r.stderr
    marker = (tmp_path / "marker.txt").read_text()
    assert "AIPERF_UI_REALTIME_METRICS_ENABLED=true" in marker


# --- PROFILE=1 self-bracketing ------------------------------------------------


def _run_profile(bench, bind, res, tmp_path, **extra_env):
    """PROFILE=1 with the window collapsed, so the branch runs in seconds."""
    return _run(
        bench,
        bind,
        res,
        tmp_path,
        PROFILE="1",
        AGENTX_PROFILE_WARMUP_S="1",
        AGENTX_PROFILE_WINDOW_S="0",
        FAKE_AIPERF_SLEEP="6",
        AGENTX_CURL_MARKER=str(tmp_path / "curl.txt"),
        **extra_env,
    )


def test_profile_forwards_capture_bounds_to_start_profile(tmp_path):
    """SGLang takes its capture bounds in the POST body, not on the serve line.

    A bare POST leaves the capture unbounded and the worker accumulates profiler
    events in host RAM until the cgroup OOM-killer takes it out mid-run, which
    surfaces as an unexplained server death rather than a profiling bug.
    """
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
    """vLLM carries its bounds on --profiler-config; an empty body must not be
    posted as one, or the endpoint gets a meaningless payload."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run_profile(bench, bind, res, tmp_path, **env)
    assert r.returncode == 0, r.stderr
    argv = (tmp_path / "curl.txt").read_text().splitlines()
    assert "start_profile" in " ".join(argv)  # the call still happened
    assert "-d" not in argv


def test_agentx_server_script_override_without_framework(tmp_path):
    """An explicit AGENTX_SERVER_SCRIPT still resolves when FRAMEWORK is unset."""
    bench, bind, res = _sandbox(tmp_path, make_builtin=False)
    _write_exec(bench / "custom_server.sh", _fake_builtin(True))
    r = _run(bench, bind, res, tmp_path, FRAMEWORK="", AGENTX_SERVER_SCRIPT="custom_server.sh")
    assert r.returncode == 0, r.stderr
    assert (res / "inferencex_result.json").exists()


def test_pinned_corpus_cannot_keep(tmp_path):
    """A different corpus is a different workload, and the scenario cannot object.

    Its allowlist admits every dated weka variant, so replaying an older set --
    which upstream's own H100/H200 recipes pin via WEKA_LOADER_OVERRIDE -- comes
    back submission_valid=true against a row measured on 062126.
    """
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
