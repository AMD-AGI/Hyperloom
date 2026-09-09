# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sizing an aiter JIT build to the container rather than to the host.

The measurement behind every number here, taken on an MI355X under ROCm 7.2:
one ``hipcc`` job on a CK-tile fused-MoE source holds **1.64 GiB** and runs for
30 seconds, three samples within 0.01 GiB of each other. The box shows 236 CPUs
and a cgroup ceiling of 128 GiB, so aiter's ``-j 188`` asks for roughly 308 GiB
and the build is killed rather than slowed.

These tests inject both numbers instead of reading the machine, so they say the
same thing on a laptop as on a fleet node.
"""

from __future__ import annotations

import pytest

from kernelforge.gemm_tune import utils

GIB = 1024**3


class TestReadingTheCeiling:
    def test_cgroup_v2_reports_bytes(self, tmp_path, monkeypatch):
        limit = tmp_path / "memory.max"
        limit.write_text("137438953472\n", encoding="utf-8")
        monkeypatch.setattr(utils, "_CGROUP_MEMORY_LIMITS", (str(limit),))
        assert utils.cgroup_memory_limit() == 128 * GIB

    def test_cgroup_v2_unlimited_is_not_a_ceiling(self, tmp_path, monkeypatch):
        limit = tmp_path / "memory.max"
        limit.write_text("max\n", encoding="utf-8")
        monkeypatch.setattr(utils, "_CGROUP_MEMORY_LIMITS", (str(limit),))
        assert utils.cgroup_memory_limit() is None

    def test_cgroup_v1_sentinel_is_not_a_ceiling(self, tmp_path, monkeypatch):
        # v1 spells "no limit" as PAGE_COUNTER_MAX rounded to a page. Taken at
        # face value it would compute a job limit of several billion.
        limit = tmp_path / "memory.limit_in_bytes"
        limit.write_text("9223372036854771712\n", encoding="utf-8")
        monkeypatch.setattr(utils, "_CGROUP_MEMORY_LIMITS", (str(limit),))
        assert utils.cgroup_memory_limit() is None

    def test_no_cgroup_files_at_all(self, monkeypatch):
        # Every developer machine that is not a container, and every Mac.
        monkeypatch.setattr(utils, "_CGROUP_MEMORY_LIMITS", ("/no/such/path",))
        assert utils.cgroup_memory_limit() is None

    def test_the_first_readable_file_wins(self, tmp_path, monkeypatch):
        v2, v1 = tmp_path / "memory.max", tmp_path / "limit_in_bytes"
        v2.write_text("68719476736", encoding="utf-8")
        v1.write_text("137438953472", encoding="utf-8")
        monkeypatch.setattr(utils, "_CGROUP_MEMORY_LIMITS", (str(v2), str(v1)))
        assert utils.cgroup_memory_limit() == 64 * GIB


class TestTheJobLimit:
    def test_the_box_this_was_measured_on(self):
        # 128 GiB * 0.7 / 2 GiB = 44, against 236 CPUs.
        assert utils.build_job_limit(cpus=236, memory_bytes=128 * GIB) == 44

    def test_a_roomy_container_is_left_alone(self):
        # 2 TiB holds a job for every one of 236 CPUs, so there is nothing to
        # correct and None says so rather than restating the CPU count.
        assert utils.build_job_limit(cpus=236, memory_bytes=2048 * GIB) is None

    def test_no_ceiling_means_no_opinion(self):
        assert utils.build_job_limit(cpus=236, memory_bytes=None) is None

    def test_a_ceiling_below_one_job_still_runs_one(self):
        # Serialised and probably doomed, but a limit of 0 would be a ninja
        # invocation that never compiles anything, which is worse to debug.
        assert utils.build_job_limit(cpus=8, memory_bytes=1 * GIB) == 1

    def test_the_limit_never_exceeds_the_cpus(self):
        assert utils.build_job_limit(cpus=4, memory_bytes=512 * GIB) is None


class TestWhatItDoesToTheEnvironment:
    def test_it_sets_max_jobs_when_the_container_is_tight(self, monkeypatch):
        monkeypatch.setattr(utils, "build_job_limit", lambda: 44)
        monkeypatch.setattr(utils, "cgroup_memory_limit", lambda: 128 * GIB)
        assert utils.cap_build_parallelism({})["MAX_JOBS"] == "44"

    def test_an_operator_who_chose_a_number_keeps_it(self, monkeypatch):
        # Including through a run that would have picked a different one:
        # someone debugging a build sets this by hand and expects it to hold.
        monkeypatch.setattr(utils, "build_job_limit", lambda: 44)
        assert utils.cap_build_parallelism({"MAX_JOBS": "4"})["MAX_JOBS"] == "4"

    def test_it_leaves_a_roomy_box_untouched(self, monkeypatch):
        monkeypatch.setattr(utils, "build_job_limit", lambda: None)
        assert "MAX_JOBS" not in utils.cap_build_parallelism({})

    def test_an_empty_max_jobs_is_not_a_choice(self, monkeypatch):
        # `MAX_JOBS=` in a shell profile sets it to the empty string, which
        # ninja reads as unset. Treating it as a choice would leave the build
        # at the default and back at the OOM this exists to prevent.
        monkeypatch.setattr(utils, "build_job_limit", lambda: 44)
        monkeypatch.setattr(utils, "cgroup_memory_limit", lambda: 128 * GIB)
        assert utils.cap_build_parallelism({"MAX_JOBS": ""})["MAX_JOBS"] == "44"

    @pytest.mark.parametrize("other", ["PATH", "AITER_CONFIG_FMOE"])
    def test_nothing_else_in_the_environment_moves(self, monkeypatch, other):
        monkeypatch.setattr(utils, "build_job_limit", lambda: 44)
        monkeypatch.setattr(utils, "cgroup_memory_limit", lambda: 128 * GIB)
        env = {other: "untouched"}
        assert utils.cap_build_parallelism(env)[other] == "untouched"


class TestTheSubprocessGetsIt:
    def test_run_subprocess_passes_the_cap_to_the_child(self, monkeypatch, tmp_path):
        monkeypatch.setattr(utils, "build_job_limit", lambda: 44)
        monkeypatch.setattr(utils, "cgroup_memory_limit", lambda: 128 * GIB)
        seen: dict[str, str] = {}

        class _Proc:
            returncode = 0

            def communicate(self, timeout=None):
                return "", ""

        def _popen(cmd, **kwargs):
            seen.update(kwargs["env"])
            return _Proc()

        monkeypatch.setattr(utils.subprocess, "Popen", _popen)
        utils.run_subprocess(["true"], cwd=tmp_path)
        assert seen["MAX_JOBS"] == "44"

    def test_an_explicit_override_still_wins(self, monkeypatch, tmp_path):
        # env_override is how a caller says what this particular build needs;
        # the cap is a default and must not overrule it.
        monkeypatch.setattr(utils, "build_job_limit", lambda: 44)
        seen: dict[str, str] = {}

        class _Proc:
            returncode = 0

            def communicate(self, timeout=None):
                return "", ""

        def _popen(cmd, **kwargs):
            seen.update(kwargs["env"])
            return _Proc()

        monkeypatch.setattr(utils.subprocess, "Popen", _popen)
        utils.run_subprocess(["true"], cwd=tmp_path, env_override={"MAX_JOBS": "8"})
        assert seen["MAX_JOBS"] == "8"
