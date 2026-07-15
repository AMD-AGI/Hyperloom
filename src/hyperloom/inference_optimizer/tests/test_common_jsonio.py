# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import json
import re

import pytest

from hyperloom.common.jsonio import coerce_dict, extract_first_json_with_key, read_json, read_jsonl

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

    def test_strict_missing_raises(self, tmp_path):
        with pytest.raises(OSError):
            read_json(tmp_path / "nope.json", strict=True)

    def test_strict_malformed_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json}")
        with pytest.raises(json.JSONDecodeError):
            read_json(p, strict=True)

    def test_strict_require_dict_raises(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2]")
        with pytest.raises(ValueError):
            read_json(p, require_dict=True, strict=True)

    def test_reads_value(self, tmp_path):
        p = tmp_path / "ok.json"
        p.write_text('{"a": 1}')
        assert read_json(p) == {"a": 1}

    def test_empty_value(self, tmp_path):
        p = tmp_path / "blank.json"
        p.write_text("  ")
        assert read_json(p, strict=True, empty_value=None) is None

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

    def test_missing_default_and_error_callback(self, tmp_path):
        errors: list[BaseException] = []
        assert read_jsonl(tmp_path / "missing.jsonl", default=["fallback"], on_error=errors.append) == ["fallback"]
        assert isinstance(errors[0], OSError)

    def test_skip_malformed_and_non_dict_rows(self, tmp_path):
        p = tmp_path / "mixed.jsonl"
        p.write_text('{"ok": 1}\nnot-json\n[1, 2]\n{"ok": 2}\n')
        errors: list[BaseException] = []
        assert read_jsonl(p, require_dict=True, skip_malformed=True, on_error=errors.append) == [
            {"ok": 1},
            {"ok": 2},
        ]
        assert len(errors) == 2

    def test_skip_non_dict_but_raise_malformed(self, tmp_path):
        p = tmp_path / "mixed.jsonl"
        p.write_text('{"ok": 1}\n[1, 2]\n')
        assert read_jsonl(p, require_dict=True, skip_non_dict=True) == [{"ok": 1}]
        p.write_text("not-json\n")
        with pytest.raises(json.JSONDecodeError):
            read_jsonl(p, require_dict=True, skip_non_dict=True)

    def test_malformed_raises_by_default(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("{not json}\n")
        with pytest.raises(json.JSONDecodeError):
            read_jsonl(p)


class TestCoerceDict:
    def test_dict_returned(self):
        data = {"x": 1}
        assert coerce_dict(data) is data

    def test_path_to_dict(self, tmp_path):
        p = tmp_path / "value.json"
        p.write_text('{"x": 1}')
        assert coerce_dict(p) == {"x": 1}

    def test_invalid_or_non_dict_default(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2]")
        assert coerce_dict(p) == {}
        assert coerce_dict(None, default={"fallback": True}) == {"fallback": True}


class TestExtractFirstJson:
    def test_fenced_with_key(self):
        text = 'blah ```json\n{"scores": [1]}\n``` end'
        assert extract_first_json_with_key(text, "scores", _BARE) == {"scores": [1]}

    def test_bare_with_key(self):
        text = 'prefix {"intents": []} trailing prose'
        assert extract_first_json_with_key(text, "intents", _BARE) == {"intents": []}

    def test_missing_key_returns_none(self):
        text = '{"other": 1}'
        assert extract_first_json_with_key(text, "scores", _BARE) is None

    def test_optional_key_any_object(self):
        text = 'note ```json\n{"anything": 9}\n```'
        assert extract_first_json_with_key(text) == {"anything": 9}

    def test_last_object(self):
        text = '```json\n{"k": 1}\n```\n```json\n{"k": 2}\n```'
        assert extract_first_json_with_key(text, "k", last=True) == {"k": 2}
        assert extract_first_json_with_key(text, "k") == {"k": 1}

    def test_empty(self):
        assert extract_first_json_with_key("", "k", _BARE) is None

    def test_no_bare_re_only_fenced(self):
        text = 'bare {"k": 1} no fence'
        # bare_re None -> only fenced blocks considered
        assert extract_first_json_with_key(text, "k") is None
