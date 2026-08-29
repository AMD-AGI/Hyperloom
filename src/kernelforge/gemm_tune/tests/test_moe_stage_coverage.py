# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for MoE stage detection granularity.

A model does not pick one MoE stage and keep it: aiter dispatches 1-stage ASM at
some token counts and CK 2-stage at others within the same run. The old
predicate answered "did we see 1stage anywhere?" and skipped the CK tuner on the
first sighting -- forfeiting the token range (observed 1-32) that 2-stage
actually serves and that the CK tuner can tune.
"""

from __future__ import annotations

from kernelforge.gemm_tune.router import _detect_1stage_from_log, moe_stage_coverage

_1STAGE = "[aiter] [fused_moe] using 1stage default for (304, {tok}, 4096, 1536, 256, 6)"
_2STAGE = "[aiter] [fused_moe] using 2stage default for (304, {tok}, 4096, 1536, 256, 6)"


def _log(tmp_path, lines, name="server.log"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


class TestMixedDispatch:
    def test_both_stages_means_there_is_ck_work_to_tune(self, tmp_path):
        # The regression: 2-stage covers small tokens, 1-stage covers large ones.
        # Skipping here forfeits every token 2-stage serves.
        path = _log(
            tmp_path,
            [
                *[_2STAGE.format(tok=t) for t in (1, 8, 16, 32)],
                *[_1STAGE.format(tok=t) for t in (64, 128, 256)],
            ],
        )
        assert _detect_1stage_from_log(path) is False

    def test_coverage_reports_tokens_per_stage(self, tmp_path):
        path = _log(
            tmp_path,
            [
                _2STAGE.format(tok=1),
                _2STAGE.format(tok=32),
                _1STAGE.format(tok=256),
            ],
        )
        cov = moe_stage_coverage(path)
        assert cov["tunable_ck_2stage"] is True
        assert cov["missed_ck_keys"] == 0
        assert sorted(cov["stages_seen"]) == ["1stage", "2stage"]
        toks = cov["tokens_by_stage"]
        assert toks["2stage/default"] == [1, 32]
        assert toks["1stage/default"] == [256]


class TestSingleStage:
    def test_only_1stage_still_skips(self, tmp_path):
        # The case the skip was written for: nothing CK-served, nothing to tune.
        path = _log(tmp_path, [_1STAGE.format(tok=t) for t in (1, 64, 256)])
        assert _detect_1stage_from_log(path) is True

    def test_only_2stage_does_not_skip(self, tmp_path):
        path = _log(tmp_path, [_2STAGE.format(tok=t) for t in (1, 64)])
        assert _detect_1stage_from_log(path) is False


class TestDegradedInputs:
    def test_no_moe_lines_does_not_skip(self, tmp_path):
        path = _log(tmp_path, ["nothing relevant here", "[aiter] some other line"])
        assert _detect_1stage_from_log(path) is False

    def test_missing_file(self):
        assert _detect_1stage_from_log("/nonexistent/server.log") is False
        assert moe_stage_coverage("/nonexistent/server.log") == {}

    def test_none_path(self):
        assert _detect_1stage_from_log(None) is False
        assert moe_stage_coverage(None) == {}

    def test_unparseable_format_falls_back_to_substring_probe(self, tmp_path):
        # An older/unknown log shape the structured parser cannot read must not
        # silently flip the decision to "always tune".
        path = _log(tmp_path, ["MoE kernel: using 1stage default (legacy format)"])
        assert _detect_1stage_from_log(path) is True
