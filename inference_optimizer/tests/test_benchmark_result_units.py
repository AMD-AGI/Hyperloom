"""Targeted unit tests for ``orchestrator.action_executors.benchmark_result``.

Existing end-to-end tests exercise rescue + workspace parsing through
``test_benchmark_result_rescue``; this module covers the small helpers
(``_to_float`` / ``_to_int`` / ``_first_*`` / ``_load_json``) plus a
handful of branch-only edge cases the executor tests miss.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import benchmark_result as br


# ---------------------------------------------------------------------------
# scalar coercion helpers
# ---------------------------------------------------------------------------

class TestScalarHelpers:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("3.14", 3.14),
            (42, 42.0),
            (None, None),
            (True, None),     # booleans are skipped intentionally
            ("nope", None),
            ([], None),
        ],
    )
    def test_to_float(self, value, expected):
        out = br._to_float(value)
        if expected is None:
            assert out is None
        else:
            assert out == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("7", 7),
            (3.9, 3),
            (None, None),
            (True, None),
            ("oops", None),
        ],
    )
    def test_to_int(self, value, expected):
        assert br._to_int(value) == expected

    def test_first_float_returns_first_valid(self):
        assert br._first_float(None, "x", "1.5", 9.0) == 1.5

    def test_first_int_returns_first_valid(self):
        assert br._first_int(None, "bad", "3", 5) == 3

    def test_first_helpers_return_none_when_all_invalid(self):
        assert br._first_float(None, "x") is None
        assert br._first_int(True, None) is None


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------

class TestLoadJson:
    def test_returns_dict_on_valid_json(self, tmp_path):
        path = tmp_path / "x.json"
        path.write_text(json.dumps({"a": 1}))
        assert br._load_json(path) == {"a": 1}

    def test_returns_none_on_missing_file(self, tmp_path):
        assert br._load_json(tmp_path / "ghost.json") is None

    def test_returns_none_on_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{nope")
        assert br._load_json(path) is None

    def test_returns_none_when_not_dict(self, tmp_path):
        path = tmp_path / "lst.json"
        path.write_text(json.dumps([1, 2, 3]))
        assert br._load_json(path) is None


# ---------------------------------------------------------------------------
# _candidate_raw_jsons ordering
# ---------------------------------------------------------------------------

class TestCandidateRawJsons:
    def test_orders_non_profile_first(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "inferencex_result.json").write_text("{}")
        (ws / "profile_result.json").write_text("{}")
        (ws / "benchmark_report.json").write_text("{}")
        out = br._candidate_raw_jsons(ws)
        names = [p.name for p in out]
        # benchmark_report.json filtered out, non-profile sorted first.
        assert names[0] == "inferencex_result.json"
        assert names[1] == "profile_result.json"
        assert "benchmark_report.json" not in names

    def test_returns_empty_when_no_json_files(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        assert br._candidate_raw_jsons(ws) == []


# ---------------------------------------------------------------------------
# _rescue_candidate_paths — env handling + workspace filter
# ---------------------------------------------------------------------------

class TestRescueCandidatePaths:
    def test_no_env_no_default_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", raising=False)
        monkeypatch.setattr(br, "_DEFAULT_RESCUE_PATHS", ())
        ws = tmp_path / "ws"
        ws.mkdir()
        assert br._rescue_candidate_paths(ws) == []

    def test_explicit_file_included(self, tmp_path, monkeypatch):
        leak = tmp_path / "leak" / "inferencex_result.json"
        leak.parent.mkdir()
        leak.write_text("{}")
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
        monkeypatch.setattr(br, "_DEFAULT_RESCUE_PATHS", ())
        ws = tmp_path / "ws"
        ws.mkdir()
        out = br._rescue_candidate_paths(ws)
        assert leak.resolve() in [p.resolve() for p in out]

    def test_directory_scanned_for_inferencex_pattern(self, tmp_path, monkeypatch):
        leak_dir = tmp_path / "leak"
        leak_dir.mkdir()
        a = leak_dir / "inferencex_result.json"
        a.write_text("{}")
        b = leak_dir / "inferencex_result_eval.json"
        b.write_text("{}")
        unrelated = leak_dir / "unrelated.json"
        unrelated.write_text("{}")
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_dir))
        monkeypatch.setattr(br, "_DEFAULT_RESCUE_PATHS", ())
        ws = tmp_path / "ws"
        ws.mkdir()
        out = br._rescue_candidate_paths(ws)
        names = {p.name for p in out}
        assert names == {"inferencex_result.json", "inferencex_result_eval.json"}

    def test_paths_inside_workspace_filtered_out(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        nested = ws / "inferencex_result.json"
        nested.write_text("{}")
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(nested))
        monkeypatch.setattr(br, "_DEFAULT_RESCUE_PATHS", ())
        out = br._rescue_candidate_paths(ws)
        assert nested.resolve() not in [p.resolve() for p in out]

    def test_mtime_gate_drops_stale_leak(self, tmp_path, monkeypatch):
        leak = tmp_path / "old" / "inferencex_result.json"
        leak.parent.mkdir()
        leak.write_text("{}")
        # Force leak mtime way in the past.
        import os as _os
        old = leak.stat().st_mtime - 3600.0
        _os.utime(leak, (old, old))
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
        monkeypatch.setattr(br, "_DEFAULT_RESCUE_PATHS", ())
        ws = tmp_path / "ws"
        ws.mkdir()
        out = br._rescue_candidate_paths(
            ws, subprocess_started_unix=leak.stat().st_mtime + 60.0,
        )
        assert out == []
