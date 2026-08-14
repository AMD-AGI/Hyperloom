# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the failure evidence pure helpers."""

from __future__ import annotations

from hyperloom.orchestrator.state.failure_evidence import (
    FAILURE_STAGE_DECISION,
    FAILURE_STAGE_WARMUP,
    failure_from_variant_outcome,
    make_failure_id,
    render_failure_line,
    tail_excerpt,
)


class TestTailExcerpt:
    def test_returns_none_for_falsy(self):
        assert tail_excerpt(None) is None
        assert tail_excerpt("") is None

    def test_returns_full_text_when_short(self):
        assert tail_excerpt("hello") == "hello"

    def test_takes_tail_not_head(self):
        text = "HEADER" + "x" * 1300
        result = tail_excerpt(text, limit=1200)
        assert result is not None
        assert result.startswith("x")
        assert "HEADER" not in result

    def test_does_not_truncate_when_within_limit(self):
        text = "a" * 100
        assert tail_excerpt(text, limit=200) == text

    def test_redacts_credential_patterns(self):
        # The bearer-token pattern should be masked.
        result = tail_excerpt("Authorization: Bearer abcdef12345678")
        assert result is not None
        assert "abcdef12345678" not in result


class TestMakeFailureId:
    def test_stable_across_calls(self):
        fid = make_failure_id(task_id="t1", fingerprint="abcdef123456")
        assert fid == make_failure_id(task_id="t1", fingerprint="abcdef123456")

    def test_uses_first_12_chars_of_fingerprint(self):
        fid = make_failure_id(task_id="t1", fingerprint="abcdef123456789")
        assert fid == "fail.t1.abcdef123456"

    def test_falls_back_to_variant_name_slug_when_no_fingerprint(self):
        fid = make_failure_id(task_id="t2", fingerprint="", variant_name="my variant!")
        assert fid.startswith("fail.t2.")
        assert " " not in fid
        assert "!" not in fid

    def test_different_fingerprints_produce_different_ids(self):
        a = make_failure_id(task_id="t1", fingerprint="aaa")
        b = make_failure_id(task_id="t1", fingerprint="bbb")
        assert a != b

    def test_different_tasks_produce_different_ids(self):
        a = make_failure_id(task_id="t1", fingerprint="abc")
        b = make_failure_id(task_id="t2", fingerprint="abc")
        assert a != b


class TestFailureFromVariantOutcome:
    def _make_vo(self, **kw):
        return {
            "variant_name": "fp8_kv",
            "outcome": "FAILED",
            "fingerprint": "abc123",
            "stage": FAILURE_STAGE_WARMUP,
            "error_class": "server_init_dead",
            "error_excerpt": "AssertionError: batch_size == 1",
            "reason": "warmup_failed",
            "server_log_path": "/runs/v00/server.log",
            "workspace": "/runs/v00",
            "raw_result_path": None,
            "variant": {"extra_server_args": "--kv-cache-dtype fp8", "extra_envs": {}, "note": ""},
            **kw,
        }

    def test_failure_id_matches_make_failure_id(self):
        vo = self._make_vo()
        fe = failure_from_variant_outcome(task_id="t1", round_id="r1", vo=vo)
        assert fe["failure_id"] == make_failure_id(task_id="t1", fingerprint="abc123")

    def test_all_fields_present(self):
        vo = self._make_vo()
        fe = failure_from_variant_outcome(task_id="t1", round_id="r1", vo=vo)
        for key in ("failure_id", "task_id", "round_id", "variant_name", "fingerprint",
                    "stage", "outcome", "error_class", "error_excerpt", "reason",
                    "server_log_path", "workspace", "variant"):
            assert key in fe, f"missing key: {key}"

    def test_stage_defaults_to_decision_when_absent(self):
        vo = self._make_vo()
        del vo["stage"]
        fe = failure_from_variant_outcome(task_id="t1", round_id="r1", vo=vo)
        assert fe["stage"] == FAILURE_STAGE_DECISION


class TestRenderFailureLine:
    def test_contains_key_fields(self):
        fe = {
            "failure_id": "fail.t1.abc",
            "variant_name": "fp8_kv",
            "stage": FAILURE_STAGE_WARMUP,
            "error_class": "server_init_dead",
            "error_excerpt": "AssertionError: batch == 1",
        }
        line = render_failure_line(fe)
        assert "fail.t1.abc" in line
        assert "fp8_kv" in line
        assert "server_init_dead" in line

    def test_falls_back_to_reason_when_no_excerpt(self):
        fe = {
            "failure_id": "fail.t1.abc",
            "variant_name": "v",
            "stage": FAILURE_STAGE_DECISION,
            "error_class": "",
            "error_excerpt": "",
            "reason": "warmup_failed",
        }
        line = render_failure_line(fe)
        assert "warmup_failed" in line

    def test_keeps_the_tail_of_a_long_body(self):
        fe = {
            "failure_id": "fail.t1.abc",
            "variant_name": "v",
            "stage": FAILURE_STAGE_WARMUP,
            "error_excerpt": "banner " * 200 + "AssertionError: batch == 1",
        }
        line = render_failure_line(fe, excerpt_chars=40)
        assert "AssertionError: batch == 1" in line
