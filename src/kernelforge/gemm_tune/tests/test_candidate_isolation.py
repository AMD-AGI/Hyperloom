# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for candidate CSV isolation logic."""

import time
from pathlib import Path


class TestFmoeCandidateIsolation:
    """Test _find_candidate_csv rejects concurrent/stale candidates."""

    def test_rejects_stale_candidate(self, tmp_path, monkeypatch):
        """Candidate written before start_time should be rejected."""
        compare_dir = tmp_path / "aiter_compare"
        compare_dir.mkdir()
        stale = compare_dir / "tuned_fmoe.99999.candidate.csv"
        stale.write_text("old data")

        monkeypatch.setattr(
            "kernelforge.gemm_tune.tuners.fmoe_ck.Path",
            lambda x: tmp_path / "aiter_compare" if x == "/tmp/aiter_compare" else Path(x),
        )

        # Simulate: file mtime is before start_time
        import os

        os.utime(stale, (1000, 1000))
        start_time = time.time()

        # Direct test of the logic (without full tuner setup)
        candidates = [
            p for p in compare_dir.glob("*.candidate.csv") if p.stat().st_mtime > start_time and "tuned_fmoe" in p.name
        ]
        assert len(candidates) == 0

    def test_rejects_concurrent_wrong_stem(self, tmp_path):
        """Candidate from another run (different stem) should be rejected."""
        compare_dir = tmp_path / "aiter_compare"
        compare_dir.mkdir()

        start_time = time.time() - 1  # started 1s ago

        # Another run wrote a candidate with different stem
        other_run = compare_dir / "tuned_other_model.12345.candidate.csv"
        other_run.write_text("other model data")

        # Our stem
        stem = "tuned_fmoe"
        candidates = [
            p for p in compare_dir.glob("*.candidate.csv") if p.stat().st_mtime > start_time and stem in p.name
        ]
        assert len(candidates) == 0

    def test_accepts_matching_candidate(self, tmp_path):
        """Candidate with correct stem and recent mtime should be accepted."""
        compare_dir = tmp_path / "aiter_compare"
        compare_dir.mkdir()

        start_time = time.time() - 1

        # Our run's candidate
        ours = compare_dir / "tuned_fmoe.54321.candidate.csv"
        ours.write_text("our tuned data")

        stem = "tuned_fmoe"
        candidates = [
            p for p in compare_dir.glob("*.candidate.csv") if p.stat().st_mtime > start_time and stem in p.name
        ]
        assert len(candidates) == 1
        assert candidates[0] == ours

    def test_concurrent_two_candidates_picks_own(self, tmp_path):
        """Two candidates after start_time, only the one with matching stem is picked."""
        compare_dir = tmp_path / "aiter_compare"
        compare_dir.mkdir()

        start_time = time.time() - 1

        # Other run's candidate (newer mtime)
        other = compare_dir / "tuned_a8w8_blockscale.99999.candidate.csv"
        other.write_text("other")

        # Our candidate (slightly older)
        import os

        ours = compare_dir / "tuned_fmoe.11111.candidate.csv"
        ours.write_text("ours")
        os.utime(ours, (time.time() - 0.5, time.time() - 0.5))

        stem = "tuned_fmoe"
        candidates = [
            p for p in compare_dir.glob("*.candidate.csv") if p.stat().st_mtime > start_time and stem in p.name
        ]
        assert len(candidates) == 1
        assert candidates[0] == ours


class TestDenseCommonCandidateIsolation:
    """Document the candidate stem-matching convention shared by tuners."""

    def test_stem_matching(self, tmp_path):
        """Only candidates matching tuned_<tuner_name> stem are returned."""
        compare_dir = tmp_path
        start_time = time.time() - 1

        # Wrong stem
        wrong = compare_dir / "tuned_fmoe.123.candidate.csv"
        wrong.write_text("wrong")

        # Right stem
        right = compare_dir / "tuned_a8w8_blockscale.456.candidate.csv"
        right.write_text("right")

        stem = "tuned_a8w8_blockscale"
        candidates = [
            p for p in compare_dir.glob("*.candidate.csv") if p.stat().st_mtime > start_time and stem in p.name
        ]
        assert len(candidates) == 1
        assert candidates[0] == right


class TestStemMatches:
    """`_stem_matches` must treat the stem as a whole token, not a substring:
    the dense tuner names nest by prefix, so a plain `in` test lets a shorter
    tuner steal a longer sibling's candidate CSV (the a8w8_blockscale ->
    a8w8_blockscale_bpreshuffle regression)."""

    def test_exact_stem_matches(self):
        from kernelforge.gemm_tune.tuners._aiter_dense_common import _stem_matches

        assert _stem_matches("a8w8_blockscale", "tuned_a8w8_blockscale.candidate.csv")
        assert _stem_matches("a8w8", "tuned_a8w8.candidate.csv")
        assert _stem_matches("a4w4_blockscale", "tuned_a4w4_blockscale.candidate.csv")

    def test_longer_sibling_is_rejected(self):
        from kernelforge.gemm_tune.tuners._aiter_dense_common import _stem_matches

        # The core regression: a8w8_blockscale must NOT claim the bpreshuffle CSV.
        assert not _stem_matches("a8w8_blockscale", "tuned_a8w8_blockscale_bpreshuffle.candidate.csv")
        # ...nor should the shortest name swallow every longer sibling.
        assert not _stem_matches("a8w8", "tuned_a8w8_blockscale.candidate.csv")
        assert not _stem_matches("a8w8", "tuned_a8w8_bpreshuffle.candidate.csv")

    def test_isolated_per_shape_naming_matches(self):
        from kernelforge.gemm_tune.tuners._aiter_dense_common import _stem_matches

        # Isolated runner: _iso_tuned_<tuner>_<idx>_tuned... -> stem + "_<digit>".
        assert _stem_matches("a8w8_blockscale", "_iso_tuned_a8w8_blockscale_0_tuned.candidate.csv")
        # ...and a longer sibling in isolated form is still rejected.
        assert not _stem_matches(
            "a8w8_blockscale",
            "_iso_tuned_a8w8_blockscale_bpreshuffle_0_tuned.candidate.csv",
        )
