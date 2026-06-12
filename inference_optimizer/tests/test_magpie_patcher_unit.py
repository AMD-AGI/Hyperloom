# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the idempotent, atomic Magpie ``benchmarker.py`` patcher
(path resolution, sentinel/legacy detection, upstream-atomic awareness, and
the classified atomic-reason outcomes)."""
from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import _magpie_patcher as mp


_LEGACY_SRC = (
    "class Benchmarker:\n"
    "    def _prepare_benchmark_scripts(self, target_dir):\n"
    "        for script in scripts:\n"
    "            shutil.copy2(script, target_file)\n"
    "            target_file.chmod(0o755)\n"
    "        return\n"
)

_SGLANG_LEGACY = (
    "#!/bin/bash\n"
    "    SERVER_MONITOR_ARGS=()\n"
    "    magpie_run_benchmark_serving_remote_direct || exit $?\n"
)


def _make_magpie(root: Path, *, benchmarker: str | None = _LEGACY_SRC,
                 sglang: str | None = _SGLANG_LEGACY) -> Path:
    if benchmarker is not None:
        bp = root / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(benchmarker, encoding="utf-8")
    if sglang is not None:
        sp = root / "Magpie" / "scripts" / "benchmark" / "sglang_mi300x.sh"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(sglang, encoding="utf-8")
    return root


# ---- path resolution ------------------------------------------------------
def test_resolve_benchmarker_none(monkeypatch):
    monkeypatch.delenv("MAGPIE_DIR", raising=False)
    assert mp._resolve_benchmarker_path(None) is None


def test_resolve_benchmarker_env(monkeypatch, tmp_path):
    _make_magpie(tmp_path)
    monkeypatch.setenv("MAGPIE_DIR", str(tmp_path))
    p = mp._resolve_benchmarker_path(None)
    assert p is not None and p.name == "benchmarker.py"


def test_resolve_benchmarker_missing_file(tmp_path):
    # dir given but file absent
    assert mp._resolve_benchmarker_path(tmp_path) is None


def test_resolve_sglang(monkeypatch, tmp_path):
    _make_magpie(tmp_path)
    assert mp._resolve_sglang_mi300x_script_path(tmp_path) is not None
    monkeypatch.delenv("MAGPIE_DIR", raising=False)
    assert mp._resolve_sglang_mi300x_script_path(None) is None


def test_resolve_sglang_env(monkeypatch, tmp_path):
    _make_magpie(tmp_path)
    monkeypatch.setenv("MAGPIE_DIR", str(tmp_path))
    assert mp._resolve_sglang_mi300x_script_path(None) is not None


# ---- file lock ------------------------------------------------------------
def test_file_lock_normal(tmp_path):
    lock = str(tmp_path / "x.lock")
    with mp._file_lock(lock):
        pass
    assert Path(lock).exists()


def test_file_lock_unopenable():
    # directory path can't be opened "w" -> warn + yield without exclusion
    with mp._file_lock("/nonexistent_dir_zzz/sub/lock"):
        pass


# ---- _is_patched ----------------------------------------------------------
def test_is_patched(tmp_path):
    f = tmp_path / "b.py"
    f.write_text("nothing here", encoding="utf-8")
    assert mp._is_patched(f) is False
    f.write_text("... Hyperloom #C1 patch ...", encoding="utf-8")
    assert mp._is_patched(f) is True
    assert mp._is_patched(tmp_path / "missing.py") is False


# ---- prepare region + upstream atomic -------------------------------------
def test_extract_prepare_region():
    region = mp._extract_prepare_region(_LEGACY_SRC)
    assert "shutil.copy2" in region
    assert mp._extract_prepare_region("no method here") == ""


def test_extract_prepare_region_blank_and_dedent():
    src = (
        "class C:\n"
        "    def _prepare_benchmark_scripts(self):\n"
        "        a = 1\n"
        "\n"                       # blank line inside body -> continue
        "        b = 2\n"
        "    def other(self):\n"   # dedent to def_indent -> break
        "        c = 3\n"
    )
    region = mp._extract_prepare_region(src)
    assert "a = 1" in region and "b = 2" in region
    assert "c = 3" not in region


def test_upstream_already_atomic_helper():
    txt = "def x():\n    _copy_benchmark_script_atomic()\n"
    assert mp._upstream_is_already_atomic(txt) is True


def test_upstream_already_atomic_inline():
    txt = (
        "    def _prepare_benchmark_scripts(self):\n"
        "        fd = tempfile.mkstemp(dir=d)\n"
        "        os.replace(tmp, target)\n"
    )
    assert mp._upstream_is_already_atomic(txt) is True


def test_upstream_not_atomic():
    assert mp._upstream_is_already_atomic(_LEGACY_SRC) is False


# ---- _apply_patch_atomic_reason -------------------------------------------
def test_apply_reason_io_error_read(tmp_path):
    # directory path -> read_text raises OSError
    assert mp._apply_patch_atomic_reason(tmp_path) == mp._ATOMIC_REASON_IO_ERROR


def test_apply_reason_already_patched(tmp_path):
    f = tmp_path / "b.py"
    f.write_text("Hyperloom #C1 patch present", encoding="utf-8")
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_ALREADY_PATCHED


def test_apply_reason_upstream_atomic(tmp_path):
    f = tmp_path / "b.py"
    f.write_text(
        "def _prepare_benchmark_scripts(self):\n"
        "    tempfile.mkstemp(dir=d)\n"
        "    os.replace(a, b)\n",
        encoding="utf-8",
    )
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_UPSTREAM_ATOMIC


def test_apply_reason_unrecognized(tmp_path):
    f = tmp_path / "b.py"
    f.write_text("totally different code\n", encoding="utf-8")
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_UNRECOGNIZED_SHAPE


def test_apply_reason_applied(tmp_path):
    f = tmp_path / "b.py"
    f.write_text(_LEGACY_SRC, encoding="utf-8")
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_APPLIED
    assert "Hyperloom #C1 patch" in f.read_text(encoding="utf-8")


def test_apply_reason_write_error(tmp_path, monkeypatch):
    f = tmp_path / "b.py"
    f.write_text(_LEGACY_SRC, encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("no space")

    monkeypatch.setattr(mp.tempfile, "mkstemp", _boom)
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_IO_ERROR


def test_apply_reason_fdopen_write_error(tmp_path, monkeypatch):
    f = tmp_path / "b.py"
    f.write_text(_LEGACY_SRC, encoding="utf-8")
    # mkstemp succeeds but os.replace fails -> fdopen-path OSError + cleanup
    monkeypatch.setattr(mp.os, "replace",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert mp._apply_patch_atomic_reason(f) == mp._ATOMIC_REASON_IO_ERROR


def test_apply_patch_atomic_bool_wrapper(tmp_path):
    f = tmp_path / "b.py"
    f.write_text(_LEGACY_SRC, encoding="utf-8")
    assert mp._apply_patch_atomic(f) is True
    bad = tmp_path / "bad.py"
    bad.write_text("nonsense", encoding="utf-8")
    assert mp._apply_patch_atomic(bad) is False


# ---- remote trust patch ---------------------------------------------------
def test_is_remote_trust_patched(tmp_path):
    f = tmp_path / "s.sh"
    f.write_text("no sentinel", encoding="utf-8")
    assert mp._is_remote_trust_patched(f) is False
    f.write_text("MAGPIE_TRUST_REMOTE_CODE here", encoding="utf-8")
    assert mp._is_remote_trust_patched(f) is True
    assert mp._is_remote_trust_patched(tmp_path / "missing") is False


def test_apply_remote_trust_already(tmp_path):
    f = tmp_path / "s.sh"
    f.write_text("MAGPIE_TRUST_REMOTE_CODE", encoding="utf-8")
    assert mp._apply_remote_trust_patch_atomic(f) is True


def test_apply_remote_trust_legacy_missing(tmp_path):
    f = tmp_path / "s.sh"
    f.write_text("unrelated", encoding="utf-8")
    assert mp._apply_remote_trust_patch_atomic(f) is False


def test_apply_remote_trust_applied(tmp_path):
    f = tmp_path / "s.sh"
    f.write_text(_SGLANG_LEGACY, encoding="utf-8")
    assert mp._apply_remote_trust_patch_atomic(f) is True
    assert "MAGPIE_TRUST_REMOTE_CODE" in f.read_text(encoding="utf-8")


def test_apply_remote_trust_read_error(tmp_path):
    assert mp._apply_remote_trust_patch_atomic(tmp_path) is False


def test_apply_remote_trust_write_error(tmp_path, monkeypatch):
    f = tmp_path / "s.sh"
    f.write_text(_SGLANG_LEGACY, encoding="utf-8")
    monkeypatch.setattr(mp.tempfile, "mkstemp",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert mp._apply_remote_trust_patch_atomic(f) is False


def test_apply_remote_trust_fdopen_write_error(tmp_path, monkeypatch):
    f = tmp_path / "s.sh"
    f.write_text(_SGLANG_LEGACY, encoding="utf-8")
    monkeypatch.setattr(mp.os, "replace",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert mp._apply_remote_trust_patch_atomic(f) is False


# ---- MagpiePatchStatus ----------------------------------------------------
def test_status_properties():
    s = mp.MagpiePatchStatus(atomic_ok=True, remote_trust_ok=True,
                             atomic_reason=mp._ATOMIC_REASON_APPLIED)
    assert s.ok is True
    assert s.atomic_genuine_failure is False
    s2 = mp.MagpiePatchStatus(atomic_ok=False, remote_trust_ok=True,
                              atomic_reason=mp._ATOMIC_REASON_IO_ERROR)
    assert s2.ok is False
    assert s2.atomic_genuine_failure is True


# ---- top-level orchestration ----------------------------------------------
def test_patch_status_missing(monkeypatch):
    monkeypatch.delenv("MAGPIE_DIR", raising=False)
    s = mp.magpie_scripts_patch_status(None)
    assert s.atomic_ok is False
    assert s.atomic_reason == mp._ATOMIC_REASON_MISSING
    assert s.remote_trust_ok is True


def test_patch_status_full_flow(tmp_path):
    _make_magpie(tmp_path)
    s = mp.magpie_scripts_patch_status(tmp_path)
    assert s.atomic_ok is True
    assert s.atomic_reason == mp._ATOMIC_REASON_APPLIED
    assert s.remote_trust_ok is True
    assert s.ok is True


def test_patch_status_no_sglang(tmp_path):
    _make_magpie(tmp_path, sglang=None)
    s = mp.magpie_scripts_patch_status(tmp_path)
    assert s.remote_trust_ok is True  # no script -> not applicable


def test_patch_status_remote_trust_fails(tmp_path):
    # sglang script present but legacy block absent -> remote_trust_ok False
    _make_magpie(tmp_path, sglang="#!/bin/bash\nunrelated content\n")
    s = mp.magpie_scripts_patch_status(tmp_path)
    assert s.remote_trust_ok is False
    assert s.ok is False


def test_ensure_wrapper(tmp_path):
    _make_magpie(tmp_path)
    assert mp.ensure_magpie_atomic_scripts_patch(tmp_path) is True
