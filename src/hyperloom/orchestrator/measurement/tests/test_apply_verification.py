# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for tuned-artifact apply verification.

The load-bearing distinction: aiter logs a miss unconditionally but a hit only
when AITER_LOG_TUNED_CONFIG=1. "No hit lines" therefore does not mean "zero
hits", and a check that conflates them would revert every arm that ran without
the flag -- which, in a scan of 60 production logs, was all of them.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.measurement.apply_verification import verify_applied

pytest.importorskip("forge_gemm_tune", reason="apply verification parses via forge")

_MISS = (
    "[aiter] shape is M:{m}, N:4096, K:4096 dtype='torch.bfloat16' "
    "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, "
    "not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv"
)
_HIT = (
    "[aiter] shape is M:{m}, N:4096, K:4096 dtype='torch.bfloat16' "
    "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, "
    "found padded_M: {m}"
)
_MERGE = (
    "[aiter] merge tuned file under model_configs/ and configs/ "
    "/srv/cfg/bf16_tuned_gemm.csv:/srv/cfg/other.csv"
)


def _log(tmp_path, lines):
    p = tmp_path / "server.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


class TestServed:
    def test_hits_mean_served(self, tmp_path):
        p = _log(tmp_path, [_MERGE, _HIT.format(m=16), _MISS.format(m=15)])
        v = verify_applied(p, ["/work/bf16_tuned_gemm.csv"])
        assert v.verdict == "served"
        assert v.hits == 1 and v.misses == 1
        assert not v.blocks_keep and v.conclusive


class TestNotMerged:
    def test_artifact_absent_from_merge_list_blocks(self, tmp_path):
        # The artifact was written but the server loaded its bundled default.
        p = _log(tmp_path, [_MERGE, _MISS.format(m=15)])
        v = verify_applied(p, ["/work/a8w8_blockscale_tuned_gemm.csv"])
        assert v.verdict == "not_merged" and v.blocks_keep
        assert v.unmerged_artifacts == ["/work/a8w8_blockscale_tuned_gemm.csv"]

    def test_match_is_by_basename(self, tmp_path):
        # The server copies tables into its own config dir, so the deployed path
        # is never the path we wrote.
        p = _log(tmp_path, [_MERGE, _HIT.format(m=16)])
        v = verify_applied(p, ["/some/other/dir/bf16_tuned_gemm.csv"])
        assert v.verdict == "served"


class TestHitLoggingTrap:
    def test_misses_without_hit_logging_is_not_a_failure(self, tmp_path):
        # No hit lines because the flag was off -- must NOT block.
        p = _log(tmp_path, [_MERGE, _MISS.format(m=15), _MISS.format(m=17)])
        v = verify_applied(p, ["/work/bf16_tuned_gemm.csv"])
        assert v.verdict == "inconclusive_no_hit_logging"
        assert not v.blocks_keep
        assert not v.conclusive
        assert "AITER_LOG_TUNED_CONFIG" in v.detail


class TestDegraded:
    def test_missing_log(self, tmp_path):
        v = verify_applied(tmp_path / "nope.log", ["/work/x.csv"])
        assert v.verdict == "unknown" and not v.blocks_keep

    def test_no_lookups_at_all(self, tmp_path):
        p = _log(tmp_path, ["server started", "ready"])
        v = verify_applied(p, [])
        assert v.verdict == "no_lookups" and not v.blocks_keep

    def test_no_artifacts_supplied_skips_the_merge_check(self, tmp_path):
        p = _log(tmp_path, [_MERGE, _HIT.format(m=16)])
        assert verify_applied(p, None).verdict == "served"

    def test_to_dict_is_serialisable(self, tmp_path):
        p = _log(tmp_path, [_MERGE, _HIT.format(m=16)])
        d = verify_applied(p, ["/work/bf16_tuned_gemm.csv"]).to_dict()
        assert d["verdict"] == "served" and d["blocks_keep"] is False
