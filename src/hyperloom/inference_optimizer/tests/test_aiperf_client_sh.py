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
} > "${AGENTX_TEST_MARKER}"
exit "${FAKE_RC:-0}"
"""

_FAKE_CURL = r"""#!/usr/bin/env bash
# /v1/models -> model json; profile endpoints -> ok.
for a in "$@"; do case "$a" in *v1/models*) echo '{"data":[{"id":"m"}]}'; exit 0;; esac; done
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
    ("--endpoint", "/v1/chat/completions"),
    ("--endpoint-type", "chat"),
    ("--num-dataset-entries", "393"),
    ("--benchmark-duration", "3600"),
    ("--random-seed", "42"),
    ("--trajectory-start-min-ratio", "0.25"),
    ("--trajectory-start-max-ratio", "0.75"),
    ("--warmup-requests-per-lane", "10"),
    ("--warmup-grace-period", "1800"),
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


def test_agentx_server_script_override_without_framework(tmp_path):
    """An explicit AGENTX_SERVER_SCRIPT still resolves when FRAMEWORK is unset."""
    bench, bind, res = _sandbox(tmp_path, make_builtin=False)
    _write_exec(bench / "custom_server.sh", _fake_builtin(True))
    r = _run(bench, bind, res, tmp_path, FRAMEWORK="", AGENTX_SERVER_SCRIPT="custom_server.sh")
    assert r.returncode == 0, r.stderr
    assert (res / "inferencex_result.json").exists()
