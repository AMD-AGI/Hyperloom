# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for build_utils — all subprocess-free via mocked runners."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.framework.build_utils import (
    AbiMismatchError,
    check_rocm_toolchain_alignment,
    coerce_build_argv,
    hash_artifacts,
    probe_torch_abi,
    run_argv,
    sort_tags_desc,
    verify_fresh_artifacts,
    verify_symbols,
    write_rocm_torch_constraints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


# ---------------------------------------------------------------------------
# coerce_build_argv
# ---------------------------------------------------------------------------

def test_coerce_list_passthrough():
    assert coerce_build_argv(["pip", "install", "-e", "."]) == ["pip", "install", "-e", "."]


def test_coerce_string_split():
    assert coerce_build_argv("pip install -e .") == ["pip", "install", "-e", "."]


def test_coerce_none_returns_empty():
    assert coerce_build_argv(None) == []
    assert coerce_build_argv([]) == []
    assert coerce_build_argv("") == []


def test_coerce_rejects_pipe():
    with pytest.raises(ValueError):
        coerce_build_argv(["pip", "install", "|", "tee"])


def test_coerce_rejects_semicolon():
    with pytest.raises(ValueError):
        coerce_build_argv("pip install; rm -rf /")


def test_coerce_rejects_backtick():
    with pytest.raises(ValueError):
        coerce_build_argv(["pip", "`evil`"])


def test_coerce_rejects_control_chars():
    with pytest.raises(ValueError):
        coerce_build_argv(["pip\x00install"])


def test_coerce_rejects_shell_dash_c():
    with pytest.raises(ValueError):
        coerce_build_argv(["bash", "-c", "echo hi"])


def test_coerce_allows_safe_args():
    argv = coerce_build_argv(["python", "setup.py", "develop", "--user"])
    assert argv[0] == "python"


# ---------------------------------------------------------------------------
# run_argv
# ---------------------------------------------------------------------------

def test_run_argv_ok():
    calls: list[Any] = []
    def _run(argv, *, cwd, env, capture_output, text, timeout):
        calls.append((argv, cwd))
        return _completed(stdout="hello\n", returncode=0)

    r = run_argv(["echo", "hello"], cwd="/tmp", run=_run)
    assert r.returncode == 0
    assert "hello" in r.stdout_tail
    assert calls[0][0] == ["echo", "hello"]


def test_run_argv_nonzero():
    def _run(argv, **kw):
        return _completed(stdout="err\n", returncode=1)
    r = run_argv(["false"], cwd="/tmp", run=_run)
    assert r.returncode == 1


def test_run_argv_timeout():
    import subprocess
    def _run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)
    r = run_argv(["sleep", "9999"], cwd="/tmp", run=_run)
    assert r.returncode == -1
    assert r.timed_out is True


def test_run_argv_truncates_output():
    big = "x" * 10000
    def _run(argv, **kw):
        return _completed(stdout=big, stderr=big)
    r = run_argv(["cmd"], cwd="/tmp", run=_run)
    assert len(r.stdout_tail) == 4000
    assert len(r.stderr_tail) == 4000


# ---------------------------------------------------------------------------
# write_rocm_torch_constraints
# ---------------------------------------------------------------------------

def _make_constraint_runner(hip_rc=0, torch_ver="2.10.0+git8514f05", triton_ver="3.1.0"):
    """Return a mock runner for write_rocm_torch_constraints."""
    def _run(argv, capture_output=False, text=False, timeout=60, env=None):
        cmd = " ".join(argv)
        if "hip" in cmd or "sys.exit(0" in cmd:
            return _completed(returncode=hip_rc)
        if "argv[1]" in cmd or "version" in cmd:
            if "triton" in cmd or (len(argv) > 3 and argv[-1] == "triton"):
                return _completed(stdout=triton_ver + "\n")
            return _completed(stdout=torch_ver + "\n")
        return _completed()
    return _run


def test_write_rocm_torch_constraints_success(tmp_path):
    f = tmp_path / "c.txt"
    write_rocm_torch_constraints("python3", str(f), run=_make_constraint_runner())
    content = f.read_text()
    assert "torch==2.10.0" in content
    assert "triton==" in content


def test_write_rocm_torch_constraints_non_rocm_raises(tmp_path):
    f = tmp_path / "c.txt"
    with pytest.raises(AbiMismatchError):
        write_rocm_torch_constraints("python3", str(f), run=_make_constraint_runner(hip_rc=2))


def test_write_rocm_torch_constraints_no_triton(tmp_path):
    def _run(argv, **kw):
        cmd = " ".join(argv)
        if "hip" in cmd or "sys.exit" in cmd:
            return _completed(returncode=0)
        if "triton" in " ".join(argv[-1:]):
            return _completed(stdout="", returncode=1)
        return _completed(stdout="2.10.0\n")
    f = tmp_path / "c.txt"
    write_rocm_torch_constraints("python3", str(f), run=_run)
    content = f.read_text()
    assert "torch==2.10.0" in content
    assert "triton" not in content


# ---------------------------------------------------------------------------
# check_rocm_toolchain_alignment
# ---------------------------------------------------------------------------

def _toolchain_run(hipcc_path="/opt/rocm/bin/hipcc", rocm_path="/opt/rocm",
                   header_ok=True, hip_major=7):
    """Mock runner for check_rocm_toolchain_alignment."""
    def _run(argv, capture_output=False, text=False, timeout=10, env=None):
        cmd = " ".join(argv)
        if "which hipcc" in cmd:
            return _completed(stdout=hipcc_path + "\n")
        if "dirname" in cmd:
            return _completed(stdout=rocm_path + "\n")
        if f"cd {rocm_path!r}" in cmd or f"cd '{rocm_path}'" in cmd:
            return _completed(stdout=rocm_path + "\n")
        return _completed()
    return _run


def test_toolchain_ok(tmp_path):
    # Create a fake hip_runtime_api.h with the required sentinel
    (tmp_path / "include" / "hip").mkdir(parents=True)
    (tmp_path / "include" / "hip" / "hip_runtime_api.h").write_text(
        "// hipDeviceAttributePciChipId\n"
    )
    ok, msg = check_rocm_toolchain_alignment(
        env={"ROCM_PATH": str(tmp_path), "PATH": "/opt/rocm/bin:/usr/bin"},
        run=_toolchain_run(hipcc_path=str(tmp_path / "bin" / "hipcc"), rocm_path=str(tmp_path)),
    )
    assert ok is True


def test_toolchain_no_hipcc():
    def _run(argv, **kw):
        return _completed(returncode=1, stdout="")
    ok, msg = check_rocm_toolchain_alignment(env={}, run=_run)
    assert ok is True  # warn-only, not fatal


def test_toolchain_bad_header(tmp_path):
    (tmp_path / "include" / "hip").mkdir(parents=True)
    (tmp_path / "include" / "hip" / "hip_runtime_api.h").write_text("// no sentinel here\n")

    def _run(argv, **kw):
        cmd = " ".join(argv)
        if "which hipcc" in cmd:
            return _completed(stdout=str(tmp_path / "bin" / "hipcc") + "\n")
        if "dirname" in cmd:
            return _completed(stdout=str(tmp_path) + "\n")
        return _completed()

    ok, msg = check_rocm_toolchain_alignment(
        env={"ROCM_PATH": str(tmp_path)}, run=_run
    )
    assert ok is False
    assert "compatible" in msg.lower() or "toolchain" in msg.lower()


# ---------------------------------------------------------------------------
# probe_torch_abi
# ---------------------------------------------------------------------------

def test_probe_torch_abi_rocm():
    import json
    payload = json.dumps({
        "torch_version": "2.10.0+git8514f05",
        "hip_version": "7.2.53211",
        "python_version": "3.12.13",
        "is_rocm": True,
    })
    def _run(argv, **kw):
        return _completed(stdout=payload + "\n")
    info = probe_torch_abi("python3", run=_run)
    assert info["is_rocm"] is True
    assert info["hip_version"] == "7.2.53211"


def test_probe_torch_abi_failure():
    def _run(argv, **kw):
        return _completed(returncode=1, stdout="")
    info = probe_torch_abi("python3", run=_run)
    assert info["is_rocm"] is False


# ---------------------------------------------------------------------------
# verify_fresh_artifacts
# ---------------------------------------------------------------------------

def test_verify_fresh_artifacts_found(tmp_path):
    so = tmp_path / "lib.so"
    so.write_bytes(b"\x7fELF")
    since = time.time() - 60
    result = verify_fresh_artifacts(str(tmp_path), since, ["*.so"])
    assert result["verified"] is True
    assert any("lib.so" in p for p in result["fresh"])


def test_verify_fresh_artifacts_stale(tmp_path):
    so = tmp_path / "lib.so"
    so.write_bytes(b"\x7fELF")
    import os
    old = time.time() - 3600
    os.utime(so, (old, old))
    result = verify_fresh_artifacts(str(tmp_path), time.time(), ["*.so"])
    assert result["verified"] is False


def test_verify_fresh_artifacts_missing_dir():
    result = verify_fresh_artifacts("/nonexistent/dir/xyz", time.time(), ["*.so"])
    assert result["verified"] is False


# ---------------------------------------------------------------------------
# verify_symbols
# ---------------------------------------------------------------------------

def test_verify_symbols_all_present():
    def _run(argv, **kw):
        return _completed(returncode=0)
    r = verify_symbols("python3", ["aiter.ops.fp4_moe"], run=_run)
    assert r["verified"] is True
    assert "aiter.ops.fp4_moe" in r["present"]


def test_verify_symbols_missing():
    def _run(argv, **kw):
        return _completed(returncode=1)
    r = verify_symbols("python3", ["aiter.missing_op"], run=_run)
    assert r["verified"] is False
    assert "aiter.missing_op" in r["missing"]


def test_verify_symbols_mixed():
    def _run(argv, **kw):
        # argv[2] is the -c code string
        code = argv[2]
        rc = 0 if "missing" not in code else 1
        return _completed(returncode=rc)
    r = verify_symbols("python3", ["aiter", "aiter.missing"], run=_run)
    assert "aiter" in r["present"]
    assert "aiter.missing" in r["missing"]


# ---------------------------------------------------------------------------
# hash_artifacts
# ---------------------------------------------------------------------------

def test_hash_artifacts(tmp_path):
    f = tmp_path / "lib.so"
    f.write_bytes(b"data")
    result = hash_artifacts([str(f)])
    assert str(f) in result
    assert len(result[str(f)]) == 64


def test_hash_artifacts_missing_skipped(tmp_path):
    result = hash_artifacts([str(tmp_path / "no_such.so")])
    assert result == {}


# ---------------------------------------------------------------------------
# sort_tags_desc
# ---------------------------------------------------------------------------

def test_sort_tags_desc_basic():
    tags = ["v0.1.0", "v0.3.0", "v0.2.0", "v1.0.0"]
    assert sort_tags_desc(tags)[0] == "v1.0.0"
    assert sort_tags_desc(tags)[-1] == "v0.1.0"


def test_sort_tags_desc_newest_first():
    tags = ["v0.5.1", "v0.10.0", "v0.9.0"]
    result = sort_tags_desc(tags)
    assert result.index("v0.10.0") < result.index("v0.9.0")


def test_sort_tags_desc_empty():
    assert sort_tags_desc([]) == []
