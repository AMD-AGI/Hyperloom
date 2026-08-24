# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The GEMM-tuning audit row has to survive a tuner that failed.

``reports/trace/gemm_tuning.jsonl`` exists to answer one question: did this
tuner run? It used to record only ``tuner``/``best_micro_speedup``/``kept``,
which cannot separate a crash from a clean search that found nothing. Across one
campaign 38 of 337 tuner runs ended ``failed`` or ``empty_output`` and the trace
showed none of them.

Both cases below are transcribed from the runs in the issue.
"""

from __future__ import annotations

import json
from pathlib import Path

import hyperloom.orchestrator.kernel.request_handlers as krh
from hyperloom.inference_optimizer.session.session_paths import gemm_tuning_steps_path


def _rows(session_dir: Path) -> list[dict]:
    text = gemm_tuning_steps_path(session_dir).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestFailureReachesTheAuditRow:
    def test_a_crashed_tuner_is_not_indistinguishable_from_a_barren_one(self, tmp_path):
        # DeepSeek-V4-Pro/20260815T002915Z: 288s of GPU time, an aiter failure
        # and a subprocess_error class, none of which reached the trace.
        krh._trace_gemm_tuning_run(
            {
                "status": "failed",
                "backend": "forge",
                "engine": "forge",
                "decision": "REVERT",
                "micro_decision": "failed",
                "framework": "vllm-aiter",
                "precision": "mxfp4",
                "error_class": None,
                "tuners_run": [
                    {
                        "tuner": "fmoe_ck",
                        "status": "failed",
                        "elapsed_s": 288.54,
                        "error_class": "subprocess_error",
                        "error": (
                            "Tuner exited with code 1: [Tuning not Finished] some shapes are not tuned or all failed"
                        ),
                    }
                ],
            },
            session_dir=tmp_path,
        )

        (row,) = _rows(tmp_path)
        tuner = row["tuners_run"][0]
        assert tuner["status"] == "failed"
        assert tuner["elapsed_s"] == 288.54
        assert tuner["error_class"] == "subprocess_error"
        assert "Tuning not Finished" in tuner["error"]
        # The envelope named no class although the tuner did; take the tuner's.
        assert row["error_class"] == "subprocess_error"

    def test_an_argparse_rejection_no_longer_reads_as_a_clean_result(self, tmp_path):
        # Llama-3.1-8B-Instruct/20260812T091612Z: rejected in 10.8s and recorded
        # status "ok" / "no_improvement", identical in the trace to the six runs
        # that really tuned. This is the blind spot that hid #1211 for 3 weeks.
        krh._trace_gemm_tuning_run(
            {
                "status": "ok",
                "engine": "forge",
                "micro_decision": "no_improvement",
                "tuners_run": [
                    {
                        "tuner": "sglang_dense_bf16",
                        "status": "failed",
                        "elapsed_s": 10.79,
                        "error_class": "unsupported_argument",
                        "error": "gemm_tuner.py: error: unrecognized arguments: --libtype",
                    }
                ],
            },
            session_dir=tmp_path,
        )

        (row,) = _rows(tmp_path)
        tuner = row["tuners_run"][0]
        assert tuner["status"] == "failed"
        assert tuner["error_class"] == "unsupported_argument"
        assert "unrecognized arguments" in tuner["error"]
        # Envelope status stays as the handler reported it -- the row records
        # what happened, it does not re-decide it. The per-tuner fields are what
        # make the two runs distinguishable.
        assert row["status"] == "ok"
        assert row["error_class"] == "unsupported_argument"


class TestRowStaysCompact:
    def test_a_successful_run_gains_only_status_and_elapsed(self, tmp_path):
        krh._trace_gemm_tuning_run(
            {
                "status": "ok",
                "engine": "forge",
                "tuners_run": [
                    {
                        "tuner": "vllm_moe_triton",
                        "status": "ok",
                        "elapsed_s": 35.4,
                        "best_micro_speedup": 1.0973,
                    }
                ],
            },
            session_dir=tmp_path,
        )

        (row,) = _rows(tmp_path)
        assert set(row["tuners_run"][0]) == {
            "tuner",
            "best_micro_speedup",
            "kept",
            "status",
            "elapsed_s",
        }

    def test_the_three_original_keys_survive_being_null(self, tmp_path):
        # kept is null on every row observed so far; an absent key would be
        # indistinguishable from false.
        krh._trace_gemm_tuning_run(
            {"status": "ok", "engine": "forge", "tuners_run": [{"tuner": "a8w8"}]},
            session_dir=tmp_path,
        )

        (row,) = _rows(tmp_path)
        tuner = row["tuners_run"][0]
        assert tuner == {"tuner": "a8w8", "best_micro_speedup": None, "kept": None}

    def test_a_long_error_is_truncated(self, tmp_path):
        krh._trace_gemm_tuning_run(
            {
                "status": "failed",
                "engine": "forge",
                "tuners_run": [{"tuner": "fmoe_ck", "error": "x" * 5000}],
            },
            session_dir=tmp_path,
        )

        (row,) = _rows(tmp_path)
        error = row["tuners_run"][0]["error"]
        assert len(error) == krh._TRACE_TUNER_ERROR_MAXLEN + 3
        assert error.endswith("...")


class TestRobustness:
    def test_the_envelope_class_wins_when_it_has_one(self, tmp_path):
        krh._trace_gemm_tuning_run(
            {
                "status": "failed",
                "engine": "forge",
                "error_class": "handler_error",
                "tuners_run": [{"tuner": "a8w8", "error_class": "subprocess_error"}],
            },
            session_dir=tmp_path,
        )

        assert _rows(tmp_path)[0]["error_class"] == "handler_error"

    def test_the_first_named_class_is_taken_not_the_last(self, tmp_path):
        krh._trace_gemm_tuning_run(
            {
                "status": "failed",
                "engine": "forge",
                "tuners_run": [
                    {"tuner": "ok_one", "status": "ok"},
                    {"tuner": "first_bad", "error_class": "unsupported_argument"},
                    {"tuner": "second_bad", "error_class": "subprocess_error"},
                ],
            },
            session_dir=tmp_path,
        )

        assert _rows(tmp_path)[0]["error_class"] == "unsupported_argument"

    def test_non_dict_entries_are_skipped_without_losing_the_rest(self, tmp_path):
        krh._trace_gemm_tuning_run(
            {
                "status": "ok",
                "engine": "forge",
                "tuners_run": ["garbage", None, {"tuner": "a8w8", "status": "ok"}],
            },
            session_dir=tmp_path,
        )

        rows = _rows(tmp_path)[0]["tuners_run"]
        assert [t["tuner"] for t in rows] == ["a8w8"]

    def test_a_non_string_error_is_passed_through_untruncated(self, tmp_path):
        krh._trace_gemm_tuning_run(
            {
                "status": "failed",
                "engine": "forge",
                "tuners_run": [{"tuner": "a8w8", "error": {"code": 2}}],
            },
            session_dir=tmp_path,
        )

        assert _rows(tmp_path)[0]["tuners_run"][0]["error"] == {"code": 2}
