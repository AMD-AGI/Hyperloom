# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for tune_robustness (fault classification, blocklist, helpers).

Hermetic: no GPU / no subprocess. run_isolated / gpu_healthy are integration
paths exercised on the pod, not here.
"""

from __future__ import annotations

from kernelforge.gemm_tune import tune_robustness as tr


class TestTaskTimeout:
    def test_injects_timeout(self):
        cmd = ["python3", "x.py", "-i", "a", "--compare"]
        out = tr.with_task_timeout(cmd, 90)
        assert "--timeout" in out and out[out.index("--timeout") + 1] == "90"

    def test_idempotent(self):
        cmd = ["python3", "x.py", "--timeout", "42"]
        assert tr.with_task_timeout(cmd, 90) == cmd  # not double-added


class TestClassifyFault:
    def test_outer_timeout(self):
        assert tr.classify_fault(124, "", "") == "outer_timeout"

    def test_hard_fault_gpu_memory(self):
        # A memory-access fault that ALSO crashed the run (non-zero exit).
        assert tr.classify_fault(134, "Memory access fault by GPU node-2", "") == "hard_fault"

    def test_hard_fault_coredump(self):
        assert tr.classify_fault(1, "", "GPU coredump failed") == "hard_fault"

    def test_recovered_memory_fault_rc0_not_hard(self):
        # rc==0 with a memory-fault string = a per-candidate fault aiter --timeout
        # recovered (merely printed under -v); must NOT be a hard fault, else a
        # shape that tuned fine gets permanently blocklisted.
        assert tr.classify_fault(0, "Memory access fault by GPU node-2", "") is None

    def test_clean_run_is_none(self):
        assert tr.classify_fault(0, "Total shapes: 1 | Would update: 1", "") is None

    def test_soft_fault_not_a_fault(self):
        # A recovered per-candidate timeout / mapping error is survivable.
        out = "[!] Task 25 timed out after 120.5s\nPool restarted."
        assert tr.classify_fault(0, out, "") is None
        assert tr.count_soft_faults(out, "") >= 1

    def test_count_soft_faults_mapping_error(self):
        # The line carries two markers ("Mapping Error" + "Process PID not in GPU
        # map"); count is a coarse diagnostic, so >=1 is what matters.
        assert tr.count_soft_faults("[aiter] [Mapping Error] Task 3 - Process PID not in GPU map", "") >= 1


class TestReadCsvAndSignature:
    def test_read_untuned_csv(self, tmp_path):
        p = tmp_path / "u.csv"
        p.write_text("M,N,K\n16,7168,5120\n64,5120,5120\n")
        header, rows = tr.read_untuned_csv(p)
        assert header == "M,N,K"
        assert rows == ["16,7168,5120", "64,5120,5120"]

    def test_read_missing_or_headeronly(self, tmp_path):
        assert tr.read_untuned_csv(tmp_path / "nope.csv") == ("", [])
        p = tmp_path / "h.csv"
        p.write_text("M,N,K\n")
        assert tr.read_untuned_csv(p) == ("M,N,K", [])

    def test_signature_stable_and_whitespace_normalized(self):
        a = tr.shape_signature("16, 7168 ,5120")
        b = tr.shape_signature("16,7168,5120")
        assert a == b  # whitespace-insensitive
        assert a != tr.shape_signature("64,7168,5120")


class TestFaultBlocklist:
    def _key(self, tuner="fmoe_ck"):
        return {"gpu_type": "mi355x", "quant_type": "a8w8_blockscale", "tp": 1, "tuner": tuner}

    def test_record_filter_roundtrip(self, tmp_path):
        p = tmp_path / "bl.json"
        bl = tr.FaultBlocklist(p, self._key())
        rows = ["16,4096,1536", "64,4096,1536"]
        bl.record(tr.shape_signature(rows[1]), "hard_fault", rows[1])
        bl.save()
        kept, skipped = bl.filter_rows(rows)
        assert kept == ["16,4096,1536"] and skipped == ["64,4096,1536"]
        # persisted + reloads
        bl2 = tr.FaultBlocklist(p, self._key())
        assert bl2.is_blocked(tr.shape_signature(rows[1]))
        assert not bl2.is_blocked(tr.shape_signature(rows[0]))

    def test_provenance_keyed_isolation(self, tmp_path):
        p = tmp_path / "bl.json"
        row = "64,4096,1536"
        sig = tr.shape_signature(row)
        # record + SAVE under one regime key (record on the saved object, else
        # the file persists an empty table and the assertion below is vacuous).
        bl_a = tr.FaultBlocklist(p, self._key())
        bl_a.record(sig, "hard_fault", row)
        bl_a.save()
        # the SAME regime must see it (proves the record actually persisted) ...
        assert tr.FaultBlocklist(p, self._key()).is_blocked(sig)
        # ... but a DIFFERENT regime (different tuner) must NOT -> real isolation,
        # not a degenerate always-empty table.
        assert not tr.FaultBlocklist(p, self._key(tuner="a8w8_blockscale")).is_blocked(sig)

    def test_corrupt_file_degrades(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        bl = tr.FaultBlocklist(p, self._key())
        assert bl.filter_rows(["1,2,3"]) == (["1,2,3"], [])


class TestIsolationSwitch:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv(tr.ISOLATE_ENV, raising=False)
        assert tr.is_isolation_enabled() is False

    def test_on(self, monkeypatch):
        monkeypatch.setenv(tr.ISOLATE_ENV, "1")
        assert tr.is_isolation_enabled() is True


class TestRunIsolatedProfileMerge:
    def test_per_shape_profiles_merged_into_shared(self, tmp_path, monkeypatch):
        # Each isolated shape writes its OWN -o2 profile; run_isolated must merge
        # them all into the shared -o2 path (else only the last shape survives and
        # the serve-safe split-K cap loses every other shape's candidates).
        from pathlib import Path as _P

        untuned = tmp_path / "untuned.csv"
        untuned.write_text("M,N,K\n16,5120,5120\n64,5120,17408\n", encoding="utf-8")
        shared_profile = tmp_path / "profile.csv"
        base_args = ["-o2", str(shared_profile), "--mp", "1", "--compare"]

        def _fake_run(cmd, cwd, timeout_s, log_file):
            i = cmd[cmd.index("-i") + 1]
            o2 = cmd[cmd.index("-o2") + 1]
            data_row = _P(i).read_text(encoding="utf-8").splitlines()[1]
            _P(o2).write_text(f"M,N,K,splitK,us\n{data_row},2,10.0\n", encoding="utf-8")
            return 0, "Total shapes: 1 | Would update: 0", ""

        monkeypatch.setattr(tr, "run_subprocess", _fake_run)
        monkeypatch.setattr(tr, "with_task_timeout", lambda cmd, t=None: cmd)
        monkeypatch.setattr(tr, "gpu_healthy", lambda gpu_ids: True)
        monkeypatch.setattr(tr, "_latest_candidate", lambda *a, **k: None)

        rc, out, err, cand = tr.run_isolated(
            script="x.py",
            base_args=base_args,
            input_csv=str(untuned),
            tuned_stem="t",
            work_dir=tmp_path,
            aiter_root=str(tmp_path),
            outer_timeout_s=60,
            task_timeout_s=30,
            gpu_ids="",
            blocklist=None,
        )
        assert rc == 0
        body = shared_profile.read_text(encoding="utf-8")
        # BOTH shapes present in the merged shared profile (not just the last)
        assert "16,5120,5120" in body and "64,5120,17408" in body


class TestLatestCandidateStemBoundary:
    """`_latest_candidate` must match the stem as a whole token, not a substring:
    the dense tuners nest by prefix (tuned_a8w8_blockscale is a prefix of
    tuned_a8w8_blockscale_bpreshuffle), so a plain `in` test would let a shorter
    tuner steal a longer sibling's candidate CSV."""

    def _touch(self, path, mtime):
        import os

        path.write_text("data", encoding="utf-8")
        os.utime(path, (mtime, mtime))

    def test_picks_own_not_longer_sibling(self, tmp_path):
        import time

        start = time.time() - 10
        own = tmp_path / "tuned_a8w8_blockscale.100.candidate.csv"
        sibling = tmp_path / "tuned_a8w8_blockscale_bpreshuffle.200.candidate.csv"
        # Sibling is NEWER, so a substring match would wrongly prefer it.
        self._touch(own, start + 1)
        self._touch(sibling, start + 5)

        got = tr._latest_candidate(tmp_path, "tuned_a8w8_blockscale", start)
        assert got == own

    def test_shortest_stem_does_not_swallow_siblings(self, tmp_path):
        import time

        start = time.time() - 10
        sibling = tmp_path / "tuned_a8w8_blockscale.200.candidate.csv"
        self._touch(sibling, start + 5)
        # No candidate actually belongs to bare "tuned_a8w8" -> None, not the sibling.
        assert tr._latest_candidate(tmp_path, "tuned_a8w8", start) is None

    def test_isolated_naming_matches(self, tmp_path):
        import time

        start = time.time() - 10
        iso = tmp_path / "_iso_tuned_a8w8_blockscale_0_tuned.candidate.csv"
        self._touch(iso, start + 1)
        assert tr._latest_candidate(tmp_path, "tuned_a8w8_blockscale", start) == iso

    def test_missing_dir_returns_none(self, tmp_path):
        import time

        assert tr._latest_candidate(tmp_path / "nope", "tuned_a8w8", time.time()) is None
