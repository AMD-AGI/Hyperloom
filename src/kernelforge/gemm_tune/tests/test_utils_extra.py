# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover GPU status parsing, run_subprocess success/timeout, aiter resolution."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kernelforge.gemm_tune import aiter_script_map, utils
from kernelforge.gemm_tune.utils import (
    check_gpu_status,
    find_tuner_script,
    resolve_aiter_csrc,
    resolve_aiter_root,
    run_subprocess,
)


# ── aiter resolution ─────────────────────────────────────────────────────────
def test_resolve_aiter_root_wellknown_fallback(monkeypatch):
    monkeypatch.delenv("AITER_ROOT_DIR", raising=False)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "aiter":
            raise ImportError("no aiter")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Only the "/opt/aiter" well-known path exists.
    monkeypatch.setattr(utils.Path, "is_dir", lambda self: str(self) == "/opt/aiter")
    root = resolve_aiter_root()
    assert root == Path("/opt/aiter")


def test_resolve_aiter_root_none(monkeypatch):
    monkeypatch.delenv("AITER_ROOT_DIR", raising=False)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "aiter":
            raise ImportError("no aiter")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(utils.Path, "is_dir", lambda self: False)
    assert resolve_aiter_root() is None


def test_resolve_aiter_csrc_none_when_no_root(monkeypatch):
    monkeypatch.setattr(aiter_script_map, "resolve_aiter_root", lambda: None)
    assert resolve_aiter_csrc() is None


def test_resolve_aiter_csrc_ok(tmp_path, monkeypatch):
    (tmp_path / "csrc").mkdir()
    monkeypatch.setattr(aiter_script_map, "resolve_aiter_root", lambda: tmp_path)
    assert resolve_aiter_csrc() == tmp_path / "csrc"


def test_utils_still_exports_the_aiter_resolvers():
    # They moved to a leaf module to break the utils/script_discovery import
    # cycle; tuners import them from here and must keep working.
    assert utils.resolve_aiter_root is aiter_script_map.resolve_aiter_root
    assert utils.resolve_aiter_csrc is aiter_script_map.resolve_aiter_csrc


def test_find_tuner_script_found(tmp_path, monkeypatch):
    csrc = tmp_path / "csrc"
    rel = utils.AITER_TUNER_SCRIPTS["fmoe_ck"]
    script = csrc / rel
    script.parent.mkdir(parents=True)
    script.write_text("# tuner")
    monkeypatch.setattr(aiter_script_map, "resolve_aiter_csrc", lambda: csrc)
    assert find_tuner_script("fmoe_ck") == script


def test_find_tuner_script_no_csrc(monkeypatch):
    monkeypatch.setattr(aiter_script_map, "resolve_aiter_csrc", lambda: None)
    assert find_tuner_script("fmoe_ck") is None


# ── check_gpu_status ─────────────────────────────────────────────────────────
class _Proc:
    def __init__(self, stdout="", rc=0):
        self.stdout = stdout
        self.returncode = rc


def test_check_gpu_status_skip():
    assert check_gpu_status(skip=True) == []


def test_check_gpu_status_parses(monkeypatch):
    data = {
        "card0": {
            "GPU use (%)": "80",
            "Temperature (Sensor edge) (C)": "45",
            "Average Graphics Package Power (W)": "300",
            "VRAM Total Used Memory (B)": "1000",
            "VRAM Total Memory (B)": "2000",
        },
        "card1": {"GPU Utilization (%)": "10"},
        "system": {"ignored": "x"},
    }
    monkeypatch.setattr(utils.subprocess, "run", lambda *a, **k: _Proc(stdout=json.dumps(data)))
    gpus = check_gpu_status()
    assert len(gpus) == 2
    g0 = next(g for g in gpus if g.gpu_id == 0)
    assert g0.busy is True and g0.temperature == "45"
    g1 = next(g for g in gpus if g.gpu_id == 1)
    assert g1.busy is False


def test_check_gpu_status_nonzero_rc(monkeypatch):
    monkeypatch.setattr(utils.subprocess, "run", lambda *a, **k: _Proc(rc=1))
    assert check_gpu_status() == []


def test_check_gpu_status_not_found(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("rocm-smi")

    monkeypatch.setattr(utils.subprocess, "run", boom)
    assert check_gpu_status() == []


def test_check_gpu_status_bad_json(monkeypatch):
    monkeypatch.setattr(utils.subprocess, "run", lambda *a, **k: _Proc(stdout="not json"))
    assert check_gpu_status() == []


# ── run_subprocess ───────────────────────────────────────────────────────────
class _FakePopen:
    def __init__(self, *a, **k):
        self.returncode = 0
        self.pid = 9999

    def communicate(self, timeout=None):
        return "out-data", "err-data"

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_run_subprocess_success_writes_log(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.subprocess, "Popen", _FakePopen)
    log = tmp_path / "logs" / "run.log"
    rc, out, err = run_subprocess(["echo", "hi"], log_file=log, timeout_s=10)
    assert rc == 0 and out == "out-data" and err == "err-data"
    text = log.read_text()
    assert "STDOUT" in text and "out-data" in text


def test_run_subprocess_env_override(tmp_path, monkeypatch):
    captured = {}

    class _P(_FakePopen):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured["env"] = k.get("env")

    monkeypatch.setattr(utils.subprocess, "Popen", _P)
    run_subprocess(["x"], env_override={"MYVAR": "1"})
    assert captured["env"]["MYVAR"] == "1"


def test_run_subprocess_timeout(tmp_path, monkeypatch):
    class _TimeoutPopen:
        def __init__(self, *a, **k):
            self.returncode = None
            self.pid = 1234

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(utils.subprocess, "Popen", _TimeoutPopen)
    monkeypatch.setattr(utils.os, "killpg", lambda *a, **k: None)
    monkeypatch.setattr(utils.time, "sleep", lambda *_: None)
    log = tmp_path / "t.log"
    rc, out, err = run_subprocess(["sleep", "999"], timeout_s=1, log_file=log)
    assert rc == 124 and "Timeout" in err
    assert "TIMEOUT" in log.read_text()


def test_run_subprocess_timeout_then_wait_fails(monkeypatch):
    class _P:
        def __init__(self, *a, **k):
            self.returncode = None
            self.pid = 1234
            self._killed = False

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

        def kill(self):
            self._killed = True

    monkeypatch.setattr(utils.subprocess, "Popen", _P)
    monkeypatch.setattr(utils.os, "killpg", lambda *a, **k: None)
    monkeypatch.setattr(utils.time, "sleep", lambda *_: None)
    rc, out, err = run_subprocess(["x"], timeout_s=1)
    assert rc == 124
