# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioral tests for the aiperf_client.sh asset, driven via bash with fakes.

Covers the shell invariants unit tests can't reach: missing-builtin exit, no-pid
fail-loud, aiperf rc gating, happy-path mapping, AIPERF_* scrub keeping
AIPERF_BIN, and GPU_TYPE lowercasing. POSIX-only (skipped elsewhere).
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
{ echo "AIPERF_BIN=${AIPERF_BIN:-UNSET}"; echo "AIPERF_FOO=${AIPERF_FOO:-UNSET}"; } > "${AGENTX_TEST_MARKER}"
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
        AGENTX_NUM_ENTRIES="2",
        AGENTX_WARMUP_DURATION="0",
        AGENTX_NUM_WARMUP_SESSIONS="1",
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


def test_warmup_on_passes_flags(tmp_path):
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_WARMUP_DURATION="20", AGENTX_NUM_WARMUP_SESSIONS="2")
    assert r.returncode == 0, r.stderr
    args = _aiperf_args(res)
    assert "--warmup-duration" in args and "--num-warmup-sessions" in args


def test_warmup_off_omits_flags(tmp_path):
    """aiperf rejects an explicit 0 warmup; disabling must OMIT the flags."""
    bench, bind, res = _sandbox(tmp_path)
    r = _run(bench, bind, res, tmp_path, AGENTX_WARMUP_DURATION="0", AGENTX_NUM_WARMUP_SESSIONS="0")
    assert r.returncode == 0, r.stderr
    args = _aiperf_args(res)
    assert "--warmup-duration" not in args
    assert "--num-warmup-sessions" not in args
