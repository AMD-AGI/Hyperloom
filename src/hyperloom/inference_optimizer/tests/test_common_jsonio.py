# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import json
import re

import pytest

from hyperloom.common.jsonio import extract_first_json_with_key, read_json

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
        # bare_re defaults to None -> only fenced blocks considered
        assert extract_first_json_with_key(text, "k") is None
