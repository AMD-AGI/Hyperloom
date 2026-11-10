# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for build_utils, through mocked runners except where the pin probe itself is under test."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.enablement.runtime.build_utils import (
    VLLM_ROCM_PROBE,
    AbiMismatchError,
    check_rocm_toolchain_alignment,
    probe_torch_abi,
    run_argv,
    sglang_rocm_install_target,
    sort_tags_desc,
    write_rocm_constraints,
)


# Helpers


def _completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


# run_argv


@pytest.mark.parametrize("argv", [["echo", "hello"], ["python", "-c", "print('a; $VALUE | >')", "path with spaces"]])
def test_run_argv_ok(argv):
    calls: list[Any] = []
    work_dir = Path("build workspace")
    build_env = {"BUILD_MODE": "release"}

    def _run(argv, *, cwd, env, capture_output, text, timeout):
        calls.append((argv, cwd, env, capture_output, text, timeout))
        return _completed(stdout="hello\n", returncode=0)

    r = run_argv(argv, cwd=work_dir, env=build_env, timeout_sec=42, run=_run)
    assert r.returncode == 0
    assert r.stdout_tail == "hello\n"
    assert r.stderr_tail == ""
    assert calls == [(argv, str(work_dir), build_env, True, True, 42)]


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


def test_run_argv_truncates_output():
    big = "x" * 10000

    def _run(argv, **kw):
        return _completed(stdout=big, stderr=big)

    r = run_argv(["cmd"], cwd="/tmp", run=_run)
    assert len(r.stdout_tail) == 4000
    assert len(r.stderr_tail) == 4000


# write_rocm_constraints


def _constraint_runner(hip_rc=0, pins="torch==2.10.0+git8514f05\ntriton==3.1.0"):
    """A runner answering the ROCm probe with ``hip_rc`` and the pin probe with ``pins``."""

    def _run(argv, **_kw):
        if "sys.exit(0" in argv[2]:
            return _completed(returncode=hip_rc)
        return _completed(stdout=pins + "\n")

    return _run


def test_write_rocm_constraints_writes_the_probed_pins(tmp_path):
    f = tmp_path / "c.txt"
    write_rocm_constraints("python3", str(f), replacing="vllm", run=_constraint_runner())
    assert f.read_text().splitlines() == ["torch==2.10.0+git8514f05", "triton==3.1.0"]


def test_write_rocm_constraints_non_rocm_raises(tmp_path):
    with pytest.raises(AbiMismatchError):
        write_rocm_constraints("python3", str(tmp_path / "c.txt"), replacing="vllm", run=_constraint_runner(hip_rc=2))


def test_write_rocm_constraints_without_torch_raises(tmp_path):
    with pytest.raises(RuntimeError, match="torch version"):
        write_rocm_constraints("python3", str(tmp_path / "c.txt"), replacing="vllm", run=_constraint_runner(pins=""))


def test_rocm_constraints_pin_torch_and_rocm_labelled_builds_but_free_the_replaced_one(tmp_path, monkeypatch):
    """The pin probe runs in a real interpreter: the first distribution of a name on sys.path is the one pinned."""
    site = tmp_path / "site"
    for name, version in (("torch", "2.0.0"), ("rocmlib", "1.0+rocm7.2"), ("plainlib", "1.0"), ("vllm", "9+rocm")):
        info = site / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(site))

    def _run(argv, **kw):
        if "sys.exit(0" in argv[2]:
            return _completed()
        return subprocess.run(argv, check=False, **kw)

    out = tmp_path / "c.txt"
    write_rocm_constraints(sys.executable, out, replacing="vllm", run=_run)
    pins = out.read_text().splitlines()
    assert "torch==2.0.0" in pins
    assert "rocmlib==1.0+rocm7.2" in pins
    assert not any(p.startswith(("plainlib==", "vllm==")) for p in pins)


# check_rocm_toolchain_alignment


def _toolchain_run(hipcc_path="/opt/rocm/bin/hipcc", rocm_path="/opt/rocm", header_ok=True, hip_major=7):
    """Mock runner for check_rocm_toolchain_alignment."""

    def _run(argv, capture_output=False, text=False, timeout=10, env=None):
        cmd = " ".join(argv)
        if "which hipcc" in cmd:
            return _completed(stdout=hipcc_path + "\n")
        if "dirname" in cmd:
            return _completed(stdout=rocm_path + "\n")
        return _completed()

    return _run


def test_toolchain_ok(tmp_path):
    # Create a fake hip_runtime_api.h with the required sentinel
    (tmp_path / "include" / "hip").mkdir(parents=True)
    (tmp_path / "include" / "hip" / "hip_runtime_api.h").write_text("// hipDeviceAttributePciChipId\n")
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

    ok, msg = check_rocm_toolchain_alignment(env={"ROCM_PATH": str(tmp_path)}, run=_run)
    assert ok is False
    assert "compatible" in msg.lower() or "toolchain" in msg.lower()


# probe_torch_abi


def test_probe_torch_abi_rocm():
    import json

    payload = json.dumps(
        {
            "torch_version": "2.10.0+git8514f05",
            "hip_version": "7.2.53211",
            "python_version": "3.12.13",
            "is_rocm": True,
        }
    )

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


# sort_tags_desc


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


# sglang_rocm_install_target

_ROCM_EXTRAS = "[project.optional-dependencies]\nsrt_hip = ['torch']\nsrt_hip_rocm724 = ['torch==2.11.0']\n"


@pytest.mark.parametrize("torch_version, extra", [("2.11.0+rocm7.2", "srt_hip_rocm724"), ("2.9.1+rocm7.0", "srt_hip")])
def test_the_rocm_pyproject_installs_with_the_extra_the_torch_satisfies(tmp_path, torch_version, extra):
    """The checkout's CUDA pyproject would replace the ROCm torch; the ROCm one stands in only for the install."""
    (tmp_path / "pyproject.toml").write_text("cuda", encoding="utf-8")
    (tmp_path / "pyproject_other.toml").write_text(_ROCM_EXTRAS, encoding="utf-8")
    with sglang_rocm_install_target(tmp_path, torch_version) as target:
        assert target == f"{tmp_path}[{extra}]"
        assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == _ROCM_EXTRAS
    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == "cuda"


def test_no_extra_admitting_the_torch_refuses_the_install(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[project.optional-dependencies]\nsrt_hip = ['torch==2.9.1']\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="admits torch 2.11.0"):
        with sglang_rocm_install_target(tmp_path, "2.11.0+rocm7.2"):
            pass


def test_the_checkout_is_restored_when_the_install_raises(tmp_path):
    (tmp_path / "pyproject.toml").write_text("cuda", encoding="utf-8")
    (tmp_path / "pyproject_other.toml").write_text(_ROCM_EXTRAS, encoding="utf-8")
    with pytest.raises(RuntimeError, match="pip died"):
        with sglang_rocm_install_target(tmp_path, "2.11.0+rocm7.2"):
            raise RuntimeError("pip died")
    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == "cuda"


def test_the_vllm_rocm_probe_fails_when_the_platform_check_raises(tmp_path):
    (tmp_path / "torch").mkdir()
    (tmp_path / "torch" / "__init__.py").write_text("class version:\n    hip = '7.0'\n", encoding="utf-8")
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "vllm" / "platforms.py").write_text(
        "class RocmPlatform:\n"
        "    def is_rocm(self):\n"
        "        raise RuntimeError('platform check broke')\n"
        "current_platform = RocmPlatform()\n",
        encoding="utf-8",
    )

    probe = subprocess.run(
        [sys.executable, "-c", VLLM_ROCM_PROBE],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(tmp_path)},
        check=False,
    )

    assert probe.returncode == 1
    assert probe.stdout == ""
    assert probe.stderr.splitlines()[-1] == "RuntimeError: platform check broke"
