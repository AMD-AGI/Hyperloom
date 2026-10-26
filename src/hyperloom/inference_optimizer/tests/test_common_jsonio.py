# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import re

import pytest

from hyperloom.common.jsonio import (
    extract_last_json_matching,
    extract_last_json_with_key,
    read_json,
    read_jsonl,
    read_json_strict,
)

_BARE = re.compile(r"(\{.*\})", re.DOTALL)


class TestReadJson:
    def test_tolerant_missing(self, tmp_path):
        assert read_json(tmp_path / "nope.json") is None
        assert read_json(tmp_path / "nope.json", default={"x": 1}) == {"x": 1}

    def test_tolerant_malformed(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json}")
        assert read_json(p, default="fallback") == "fallback"

    def test_require_dict_tolerant(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2]")
        assert read_json(p, require_dict=True) is None

    def test_require_dict_tolerant_reports_error(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2]")
        errors: list[BaseException] = []
        assert read_json(p, default={}, require_dict=True, on_error=errors.append) == {}
        assert isinstance(errors[0], ValueError)

    def test_strict_missing_raises(self, tmp_path):
        with pytest.raises(OSError):
            read_json_strict(tmp_path / "nope.json")

    def test_strict_malformed_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json}")
        with pytest.raises(json.JSONDecodeError):
            read_json_strict(p)

    def test_strict_require_dict_raises(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2]")
        with pytest.raises(ValueError):
            read_json_strict(p, require_dict=True)

    def test_reads_value(self, tmp_path):
        p = tmp_path / "ok.json"
        p.write_text('{"a": 1}')
        assert read_json(p) == {"a": 1}

    def test_empty_value(self, tmp_path):
        p = tmp_path / "blank.json"
        p.write_text("  ")
        assert read_json_strict(p, empty_value=None) is None

    def test_tolerant_reports_error(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json}")
        errors: list[BaseException] = []
        assert read_json(p, default={}, on_error=errors.append) == {}
        assert isinstance(errors[0], json.JSONDecodeError)


class TestReadJsonl:
    def test_reads_rows(self, tmp_path):
        p = tmp_path / "rows.jsonl"
        p.write_text('{"a": 1}\n\n{"b": 2}\n')
        assert read_jsonl(p) == [{"a": 1}, {"b": 2}]

    def test_missing_file_is_empty_and_reported(self, tmp_path):
        errors: list[BaseException] = []
        assert read_jsonl(tmp_path / "missing.jsonl", on_error=errors.append) == []
        assert isinstance(errors[0], OSError)

    def test_skip_malformed_and_non_dict_rows(self, tmp_path):
        p = tmp_path / "mixed.jsonl"
        p.write_text('{"ok": 1}\nnot-json\n[1, 2]\n{"ok": 2}\n')
        errors: list[BaseException] = []
        assert read_jsonl(p, on_error=errors.append) == [
            {"ok": 1},
            {"ok": 2},
        ]
        assert len(errors) == 2


class TestExtractLastJsonMatching:
    def test_fenced_with_key(self):
        text = 'blah ```json\n{"scores": [1]}\n``` end'
        assert extract_last_json_matching(text, "scores", _BARE) == {"scores": [1]}

    def test_bare_with_key(self):
        text = 'prefix {"intents": []} trailing prose'
        assert extract_last_json_matching(text, "intents", _BARE) == {"intents": []}

    def test_missing_key_returns_none(self):
        text = '{"other": 1}'
        assert extract_last_json_matching(text, "scores", _BARE) is None

    def test_last_object(self):
        text = '```json\n{"k": 1}\n```\n```json\n{"k": 2}\n```'
        assert extract_last_json_matching(text, "k", _BARE) == {"k": 2}

    def test_invalid_fenced_json_is_skipped(self):
        text = '```json\n{"k": 1}\n```\n```json\n{bad}\n```'
        assert extract_last_json_matching(text, "k", re.compile(r"(?!)")) == {"k": 1}

    def test_bare_candidate_trims_trailing_text(self):
        bare_with_trailing = re.compile(r"(\{.*)", re.DOTALL)
        text = 'prefix {"k": 3} trailing prose'
        assert extract_last_json_matching(text, "k", bare_with_trailing) == {"k": 3}

    def test_empty(self):
        assert extract_last_json_matching("", "k", _BARE) is None


class TestExtractLastJson:
    def test_no_match_returns_none(self):
        assert extract_last_json_with_key("", "scores") is None
        assert extract_last_json_with_key("no json here", "scores") is None
        assert extract_last_json_with_key('{"foo": 1, "scores_x": 2}', "scores") is None

    def test_fenced_and_bare_with_trailing_prose(self):
        fenced = '```json\n{"scores": {"proposal_0": {"score": 1}}}\n```'
        assert extract_last_json_with_key(fenced, "scores") == {"scores": {"proposal_0": {"score": 1}}}
        bare = 'here {"scores": {"proposal_0": {"score": 2}}} thanks'
        assert extract_last_json_with_key(bare, "scores") == {"scores": {"proposal_0": {"score": 2}}}

    def test_meta_prefix_scores(self):
        text = '{"meta": "draft", "scores": {"proposal_0": {"score": 9, "reason": "ok"}}}'
        out = extract_last_json_with_key(text, "scores")
        assert out["scores"]["proposal_0"]["score"] == 9

    def test_bare_then_fenced_prefers_later_offset(self):
        text = (
            '{"scores": {"proposal_0": {"score": 1, "reason": "old"}}} '
            '```json\n{"meta": "final", "scores": {"proposal_0": {"score": 8, "reason": "new"}}}\n```'
        )
        out = extract_last_json_with_key(text, "scores")
        assert out["scores"]["proposal_0"]["score"] == 8

    def test_many_braces_and_braces_inside_strings(self):
        drafts = " ".join(f'{{"meta": {index}, "reason": "literal {{ brace }}"}}' for index in range(800))
        final = '{"meta": "final", "scores": {"proposal_0": {"score": 7}}}'

        assert extract_last_json_with_key(f"{drafts} {final}", "scores") == {
            "meta": "final",
            "scores": {"proposal_0": {"score": 7}},
        }
