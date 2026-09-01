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
    # Also dumps the keep-alive env it inherited, so a test can assert what the
    # client exported BEFORE the server booted. Defaults to /dev/null so every
    # pre-existing test is unaffected.
    pid_line = 'sleep 300 & echo $! > "$MAGPIE_SERVER_PID_FILE"\n' if write_pid else ": no pid written\n"
    dump = (
        "{\n"
        '  echo "VLLM_HTTP_TIMEOUT_KEEP_ALIVE=${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-UNSET}"\n'
        '  echo "SGLANG_TIMEOUT_KEEP_ALIVE=${SGLANG_TIMEOUT_KEEP_ALIVE:-UNSET}"\n'
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
    # Drop the knobs under test before overlaying: a developer or CI box with
    # WEKA_LOADER_OVERRIDE / AGENTX_DATASET exported (both documented operator
    # knobs) would otherwise fail the canonical-run assertions, and an inherited
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


def test_tcp_user_timeout_survives_the_scrub(tmp_path):
    """The scrub must not leave aiperf on its 30s stock TCP_USER_TIMEOUT.

    That bound is how long Linux tolerates an established connection making no
    progress, and an agentic turn against a long-context model makes none for
    as long as the server is prefill-bound. Upstream's Kimi-K3 and DSv4 recipes
    export 900000; because the scrub above drops any inherited copy, this file
    has to re-state it or the connection dies mid-prefill and the round fails
    as a warmup error with no matching server-side fault.
    """
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


def test_profile_delay_safe_bound_accounts_for_warmup_grace(tmp_path):
    """The `_pmax` clamp must not treat DURATION as the whole round's clock.

    A large model's warmup drain is bounded by WARMGRACE, not DURATION, and
    happens *before* the measurement window opens. With WARMGRACE=65,
    DURATION=5, PWIN=0: the old DURATION-only bound (5 - 0 - 60 = -55, clamped
    to 0) would force *any* positive PWARM to clamp to 0 and fire immediately
    -- squarely inside warmup. The WARMGRACE-inclusive bound (65 + 5 - 0 - 60
    = 10) correctly leaves room for a delay that clears the drain first.
    """
    bench, bind, res = _sandbox(tmp_path)
    r = _run(
        bench,
        bind,
        res,
        tmp_path,
        PROFILE="1",
        AGENTX_PROFILE_WARMUP_S="8",
        AGENTX_PROFILE_WINDOW_S="0",
        AGENTX_DURATION="5",
        AGENTX_WARMUP_GRACE_PERIOD="65",
        FAKE_AIPERF_SLEEP="0.2",
        AGENTX_CURL_MARKER=str(tmp_path / "curl.txt"),
    )
    assert r.returncode == 0, r.stderr
    assert "clamping to" not in (r.stdout + r.stderr)


def test_profile_delay_is_still_clamped_when_it_exceeds_the_full_bound(tmp_path):
    """A PWARM that exceeds even WARMGRACE + DURATION is still clamped early."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(
        bench,
        bind,
        res,
        tmp_path,
        PROFILE="1",
        AGENTX_PROFILE_WARMUP_S="30",
        AGENTX_PROFILE_WINDOW_S="0",
        AGENTX_DURATION="5",
        AGENTX_WARMUP_GRACE_PERIOD="65",
        FAKE_AIPERF_SLEEP="0.2",
        AGENTX_CURL_MARKER=str(tmp_path / "curl.txt"),
    )
    assert r.returncode == 0, r.stderr
    assert "clamping to 10s" in (r.stdout + r.stderr)


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


def test_canonical_pin_can_be_declared(tmp_path):
    """The family whitelist is a derivation, not a registry.

    A model upstream runs on the full corpus but whose slug does not match the
    whitelist falls back to the 256k set, and the corpus log line tells the
    operator to pin the right one. Treating that pin as a deviation would make
    the *correct* run permanently non-submittable, so the operator can declare
    which corpus is canonical here.
    """
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
    """The switch forwards every AGENTX_* key, so a stale marker must be cleared.

    Left alone it would stamp submission_valid=false, with reasons from a
    previous run, onto a round that deviated in nothing.
    """
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_NONCANONICAL_REASONS="entries=7(stale)")
    assert r.returncode == 0, r.stderr
    out = _result(res)
    assert not out["submission_invalid_reasons"]
    assert out["submission_valid"] is not False


def test_reduced_warmup_is_flagged_non_canonical(tmp_path):
    """Warmup is measurement-defining, so trimming it must void submittability.

    Measured on a 743B model: the canonical 10 requests/lane is a ~2h warmup, so
    an operator reaches for this knob under real time pressure. aiperf has no
    concept of "enough warmup", so the scenario stamps submission_valid=true and
    the round looks publishable while having measured a materially emptier cache.
    Only the client knows the canonical value, so only the client can object.
    """
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
    """A *longer* grace period is more warmup, not less, and must not be flagged.

    An operator raising this so a large model's warmup has room to fully drain
    (e.g. 4h for Kimi-K3-scale warmup) does not change what gets replayed --
    only how long the client is willing to wait for it. Flagging it the same
    as a truncated drain would make the correct run non-submittable.
    """
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
    """Loosening the abort threshold is measurement-defining and carries no marker.

    Raising it keeps alive a run that upstream's 0.10 would have aborted, and
    the requests that did survive are then mapped as an ordinary measurement.
    aiperf stamps nothing for this -- the threshold is the client's own safety
    net, not part of the scenario -- so without the client objecting the round
    comes back submission_valid=true.
    """
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
    """FRT reaches an awk program; it must be DATA, never program text.

    ``awk "BEGIN{exit !(($FRT) > ($CANON_FRT))}"`` interpolates the value into
    the program body, so ``AGENTX_FAILED_REQUEST_THRESHOLD='system("...")'``
    executes inside the container. The switch forwards every AGENTX_* key from
    the orchestrator's environment verbatim, so anything that can write a config
    or recipe gets command execution. Worse, the injected program supplies its
    own exit status, so the non-canonical guard silently stops firing too.
    """
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
    """A malformed measurement-defining knob must stop the round, not be stamped.

    ``[ "$WARMLANE" -lt N ]`` exits 2 on a non-integer, and on the left of ``&&``
    that status is exempt from ``set -e``: the guard silently does not fire and
    NONCANON stays empty, so an illegal configuration comes back
    submission_valid=true. A non-integer grace additionally reaches the ``$(( ))``
    in the PROFILE branch and aborts an otherwise-complete round there instead.
    """
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
    """The server idle timeout must be raised before the server boots.

    Regression: AIPerf pins one pooled keep-alive connection per agentic session
    and reuses it across turns. vLLM's default idle timeout is 5s
    (``envs.py: VLLM_HTTP_TIMEOUT_KEEP_ALIVE: int = 5``) while the client is
    already given 900s via AIPERF_HTTP_TCP_USER_TIMEOUT -- a 180x disagreement.
    An inter-turn think-time past 5s lets the server close the socket exactly as
    the client reuses it; aiohttp raises ServerDisconnectedError and AIPerf
    escalates it to a terminal warmup failure against a healthy server. Measured
    on a conc=16 K3 round: orderly "Application shutdown complete" at warmup
    64/177, no error in the server log at all.
    """
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
