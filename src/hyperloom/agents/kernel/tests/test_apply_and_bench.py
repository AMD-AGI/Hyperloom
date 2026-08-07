#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for apply_and_bench helpers (no serving / no GPU required).

Covers the patch-operation coverage gate (_diff_unsupported_ops), the
measurement spread/significance helper (_spread), P99 tail-latency parsing and
the peak-VRAM probe, and the vLLM orphan reap during server teardown.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"


def _load():
    spec = importlib.util.spec_from_file_location("apply_and_bench_under_test", _TOOLS_DIR / "apply_and_bench.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ab = _load()


# _diff_unsupported_ops: modify/add allowed; delete/rename/copy/mode/binary refused

_MODIFY = "diff --git a/x.cu b/x.cu\n--- a/x.cu\n+++ b/x.cu\n@@ -1 +1 @@\n-a\n+b\n"
_ADD = "diff --git a/n.cu b/n.cu\nnew file mode 100644\n--- /dev/null\n+++ b/n.cu\n@@ -0,0 +1 @@\n+z\n"


def test_unsupported_ops_allows_modify_and_add():
    assert ab._diff_unsupported_ops(_MODIFY) == []
    assert ab._diff_unsupported_ops(_ADD) == []
    # a modify + add multi-file diff is still all-clear
    assert ab._diff_unsupported_ops(_MODIFY + _ADD) == []


def test_unsupported_ops_flags_delete():
    d = "diff --git a/x.cu b/x.cu\ndeleted file mode 100644\n--- a/x.cu\n+++ /dev/null\n"
    assert ab._diff_unsupported_ops(d) == ["delete"]


def test_unsupported_ops_flags_rename_and_copy():
    assert ab._diff_unsupported_ops("diff --git a/x b/y\nrename from x\nrename to y\n") == ["rename"]
    assert ab._diff_unsupported_ops("diff --git a/x b/y\ncopy from x\ncopy to y\n") == ["copy"]


def test_unsupported_ops_flags_mode_and_binary():
    assert ab._diff_unsupported_ops("diff --git a/x b/x\nold mode 100644\nnew mode 100755\n") == ["mode-change"]
    assert ab._diff_unsupported_ops("diff --git a/x b/x\nGIT binary patch\n") == ["binary"]
    assert ab._diff_unsupported_ops("diff --git a/x b/x\nBinary files a/x and b/x differ\n") == ["binary"]


def test_unsupported_ops_dedup_and_sorted():
    mixed = (
        "diff --git a/x b/x\ndeleted file mode 100644\n"
        "diff --git a/y b/z\nrename from y\nrename to z\n"
        "diff --git a/y b/z\ndeleted file mode 100644\n"
    )
    assert ab._diff_unsupported_ops(mixed) == ["delete", "rename"]


# _spread: median + p25/p75 + stdev, None-safe


def test_spread_basic():
    s = ab._spread([10.0, 12.0, 11.0, 13.0, 9.0])
    assert s["n"] == 5
    assert s["median"] == 11.0
    assert s["p25"] is not None and s["p75"] is not None
    assert s["p25"] <= s["median"] <= s["p75"]
    assert s["stdev"] is not None and s["stdev"] > 0


def test_spread_edge_cases():
    assert ab._spread([])["median"] is None
    one = ab._spread([5.0])
    assert one["median"] == 5.0 and one["stdev"] == 0.0 and one["n"] == 1


# P99 tail-latency parsing + peak-VRAM probe (additive to the ABBA result)


def test_bench_once_parses_p99(tmp_path, monkeypatch):
    monkeypatch.setattr(ab.subprocess, "run", lambda *a, **k: None)
    (tmp_path / "baseline_rep1.json").write_text(json.dumps({
        "output_throughput": 1700.0, "median_tpot_ms": 12.0,
        "p99_tpot_ms": 30.0, "p99_e2el_ms": 900.0, "p99_itl_ms": 15.0,
    }))
    out = ab._bench_once("bs.py", "m", 8888, 1024, 1024, 64, 320, "baseline", 1, tmp_path, 0)
    assert out["output_throughput"] == 1700.0
    assert out["p99_tpot_ms"] == 30.0
    assert out["p99_e2el_ms"] == 900.0
    assert out["p99_itl_ms"] == 15.0


def test_gpu_vram_used_mb_sums_wanted(monkeypatch):
    fake = json.dumps({
        "card0": {"VRAM Total Used Memory (B)": str(2 * 1024 * 1024 * 1024)},  # 2048 MiB
        "card1": {"VRAM Total Used Memory (B)": str(1 * 1024 * 1024 * 1024)},  # 1024 MiB
    })
    monkeypatch.setattr(ab.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=fake))
    assert ab._gpu_vram_used_mb("0") == 2048.0
    assert ab._gpu_vram_used_mb("0,1") == 3072.0


def test_gpu_vram_used_mb_failure_returns_none(monkeypatch):
    def _raise(*a, **k):
        raise OSError("no rocm-smi")

    monkeypatch.setattr(ab.subprocess, "run", _raise)
    assert ab._gpu_vram_used_mb("0") is None


# Leaked VLLM::EngineCore reap: vLLM's engine worker escapes killpg (fresh session),
# so teardown must reap it by name — POSIX-only, self-match-safe, never fatal.


def test_reap_vllm_orphans_skips_non_posix(monkeypatch):
    calls = []
    monkeypatch.setattr(ab.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(ab.os, "name", "nt")
    ab._reap_vllm_orphans()
    assert calls == []  # no POSIX pkill on non-POSIX hosts


def test_reap_vllm_orphans_targets_enginecore_and_worker(monkeypatch):
    cmds = []
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, **k: cmds.append(cmd))
    monkeypatch.setattr(ab.os, "name", "posix")
    ab._reap_vllm_orphans()
    pats = [c[-1] for c in cmds if c[:3] == ["pkill", "-9", "-f"]]
    assert any("EngineCore" in p for p in pats)
    assert any("Worker" in p for p in pats)
    # bracket trick: pattern must not contain a bare 'VLLM::' (would match pkill's own argv)
    for p in pats:
        assert p.startswith("[V]") and "VLLM::" not in p


def test_reap_vllm_orphans_swallows_errors(monkeypatch):
    def _raise(*a, **k):
        raise OSError("no pkill")

    monkeypatch.setattr(ab.subprocess, "run", _raise)
    monkeypatch.setattr(ab.os, "name", "posix")
    ab._reap_vllm_orphans()  # must not raise


def test_reap_vllm_orphans_opt_out(monkeypatch):
    cmds = []
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, **k: cmds.append(cmd))
    monkeypatch.setattr(ab.os, "name", "posix")
    monkeypatch.setenv("APPLY_BENCH_NO_ORPHAN_REAP", "1")
    ab._reap_vllm_orphans()
    assert cmds == []


def test_kill_servers_reaps_orphans_for_vllm(monkeypatch):
    cmds = []
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, **k: cmds.append(cmd))
    monkeypatch.setattr(ab.os, "name", "posix")
    monkeypatch.setattr(ab.time, "sleep", lambda *_: None)
    ab._kill_servers(None, "vllm")  # gpu=None => no VRAM-drain probe
    assert any(c[:3] == ["pkill", "-9", "-f"] and "EngineCore" in c[-1] for c in cmds)


def test_kill_servers_skips_orphan_reap_for_sglang(monkeypatch):
    cmds = []
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, **k: cmds.append(cmd))
    monkeypatch.setattr(ab.os, "name", "posix")
    monkeypatch.setattr(ab.time, "sleep", lambda *_: None)
    ab._kill_servers(None, "sglang")
    assert not any("EngineCore" in c[-1] for c in cmds)


def test_wait_vram_drain_returns_when_below_threshold(monkeypatch):
    monkeypatch.setattr(ab, "_gpu_vram_used_mb", lambda gpu: 1000.0)
    monkeypatch.setattr(ab.time, "sleep", lambda *_: None)
    assert ab._wait_vram_drain("0", threshold_mb=20000.0, timeout_s=5.0) == 1000.0


def test_wait_vram_drain_none_when_no_rocm_smi(monkeypatch):
    monkeypatch.setattr(ab, "_gpu_vram_used_mb", lambda gpu: None)
    assert ab._wait_vram_drain("0") is None


def test_wait_vram_drain_warns_on_timeout(monkeypatch, tmp_path):
    # Timeout without draining must NOT be silent — the next arm would launch on
    # an occupied GPU (the contamination this guard exists to prevent).
    monkeypatch.setattr(ab, "_gpu_vram_used_mb", lambda gpu: 50000.0)
    monkeypatch.setattr(ab.time, "sleep", lambda *_: None)
    logged = []
    monkeypatch.setattr(ab, "_log", lambda out_dir, msg: logged.append(msg))
    ab._wait_vram_drain("0", out_dir=tmp_path, threshold_mb=20000.0, timeout_s=0.0)
    assert any("did not drain" in m for m in logged)


def test_gpu_vram_used_none_without_gpu_scope(monkeypatch):
    # No GPU scope -> None WITHOUT probing rocm-smi (never sum the whole host).
    import hyperloom.agents.kernel.tools.apply_and_bench as _ab

    def _fail(*a, **k):
        raise AssertionError("rocm-smi must not be called without a GPU scope")

    monkeypatch.setattr(_ab.subprocess, "run", _fail)
    assert _ab._gpu_vram_used_mb("") is None
    assert _ab._gpu_vram_used_mb(None) is None
    assert _ab._gpu_vram_used_mb("  ,  ") is None


def test_apply_and_bench_accepts_and_reports_deferred_rebuild(
    tmp_path,
    monkeypatch,
):
    patch_file = tmp_path / "kernel.py"
    target = tmp_path / "target.py"
    patch_file.write_text("VALUE = 2\n", encoding="utf-8")
    target.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    measurements = iter(
        (
            {
                "status": "ok",
                "median": 100.0,
                "tput_spread": {"p25": 99.0, "p75": 101.0},
                "tpot_spread_ms": {"median": 1.0},
                "p99_tpot_spread_ms": {"median": 2.0},
                "vram_used_mb": 1000,
                "reps": [100.0],
            },
            {
                "status": "ok",
                "median": 110.0,
                "tput_spread": {"p25": 109.0, "p75": 111.0},
                "tpot_spread_ms": {"median": 0.9},
                "p99_tpot_spread_ms": {"median": 1.8},
                "vram_used_mb": 1000,
                "reps": [110.0],
            },
        )
    )
    monkeypatch.setattr(ab, "_find_benchmark_serving", lambda: tmp_path)
    monkeypatch.setattr(
        ab,
        "_serve_and_bench",
        lambda *args, **kwargs: next(measurements),
    )
    monkeypatch.setattr(
        ab,
        "apply_kernel_patch",
        lambda **kwargs: {
            "status": "ok",
            "manifest_path": str(manifest),
            "rebuild": {
                "status": "deferred",
                "mode": "runtime_jit",
            },
        },
    )
    monkeypatch.setattr(
        ab,
        "revert_kernel_patch",
        lambda manifest_path: {"status": "ok"},
    )
    monkeypatch.setattr(
        ab,
        "_engagement_proof",
        lambda *args, **kwargs: {"engaged": True},
    )

    result = ab.apply_and_bench(
        patch_path=str(patch_file),
        target_file=str(target),
        backup_root=str(tmp_path / "backups"),
        model="model",
        out_dir=str(tmp_path / "out"),
    )

    assert result["status"] == "ok"
    assert result["applied"][0]["rebuild"] == {
        "status": "deferred",
        "mode": "runtime_jit",
    }
