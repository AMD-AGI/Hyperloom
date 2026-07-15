# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ci/generate_hf_matrix.py."""

from __future__ import annotations

import io
import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import generate_hf_matrix as gm  # noqa: E402


# ── pure helpers ──


def test_slugify():
    assert gm.slugify("Qwen/Qwen3-8B") == "qwen-qwen3-8b"
    assert gm.slugify("a..b//c") == "a-b-c"
    assert gm.slugify("???") == "model"


def test_truthy():
    for v in ("1", "true", "YES", "y", "on"):
        assert gm._truthy(v) is True
    for v in ("0", "", "no", None):
        assert gm._truthy(v) is False


def test_entry_repo_and_slug():
    assert gm._entry_repo({"repo_id": "a/b"}) == "a/b"
    assert gm._entry_repo({"model": "c/d"}) == "c/d"
    assert gm._entry_repo("e/f") == "e/f"
    assert gm._entry_repo({}) == ""
    assert gm._entry_slug({"repo_id": "Org/M"}) == "org-m"


def test_parse_explicit_models():
    assert gm._parse_explicit_models("a b,c\nd") == ["a", "b", "c", "d"]
    assert gm._parse_explicit_models("  ") == []


# ── _filter_entries_by_explicit_models ──


def test_filter_entries_by_explicit_models():
    entries = [{"repo_id": "Org/A"}, {"repo_id": "Org/B"}]
    out = gm._filter_entries_by_explicit_models(entries, ["org/b"])
    assert out == [{"repo_id": "Org/B"}]


def test_filter_entries_empty_repos_returns_all():
    entries = [{"repo_id": "A"}]
    assert gm._filter_entries_by_explicit_models(entries, []) == entries


def test_filter_entries_missing_warns(capsys):
    out = gm._filter_entries_by_explicit_models([{"repo_id": "A"}], ["a", "missing"])
    assert len(out) == 1
    assert "not present" in capsys.readouterr().err


def test_filter_top_entries_only_drops_explicit_false(monkeypatch):
    monkeypatch.setenv("INPUT_TOP_ONLY", "true")
    entries = [
        {"repo_id": "top", "is_top": True},
        {"repo_id": "legacy"},
        {"repo_id": "supplement", "is_top": False},
    ]
    out = gm._filter_top_entries(entries)
    assert [gm._entry_repo(e) for e in out] == ["top", "legacy"]


def test_filter_top_entries_disabled(monkeypatch):
    monkeypatch.delenv("INPUT_TOP_ONLY", raising=False)
    entries = [{"repo_id": "supplement", "is_top": False}]
    assert gm._filter_top_entries(entries) == entries


# ── _load_candidate_entries ──


def test_load_candidate_entries(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps({"pool_id": "pool1", "candidates": [{"repo_id": "a"}, {"no_repo": 1}, {"repo_id": "b"}]}),
        encoding="utf-8",
    )
    out = gm._load_candidate_entries(p)
    assert [e["repo_id"] for e in out] == ["a", "b"]
    assert out[0]["pool_id"] == "pool1"
    assert out[0]["pool_index"] == 0


def test_load_candidate_entries_bad_file(tmp_path: Path, capsys):
    p = tmp_path / "c.json"
    p.write_text("{bad", encoding="utf-8")
    assert gm._load_candidate_entries(p) == []


# ── _resolve_batch_index ──


def test_resolve_batch_index_explicit(monkeypatch):
    monkeypatch.setenv("INPUT_BATCH_INDEX", "3")
    assert gm._resolve_batch_index(100, 10) == 3


def test_resolve_batch_index_explicit_invalid(monkeypatch):
    monkeypatch.setenv("INPUT_BATCH_INDEX", "xx")
    assert gm._resolve_batch_index(100, 10) == 0


def test_resolve_batch_index_empty_uses_rotation(monkeypatch):
    # Empty batch_index + cron_now -> max_hours-paced rotation.
    monkeypatch.delenv("INPUT_BATCH_INDEX", raising=False)
    monkeypatch.setenv("INPUT_MAX_HOURS", "6")
    # anchor + 6h -> one batch advanced.
    monkeypatch.setenv("INPUT_CRON_NOW", (gm._CRON_ANCHOR_UTC + timedelta(hours=6)).isoformat())
    assert gm._resolve_batch_index(100, 10) == 1


def test_resolve_batch_index_schedule_empty_ok(monkeypatch):
    # Schedule fire with empty batch_index/cron_now uses real-clock rotation.
    monkeypatch.delenv("INPUT_BATCH_INDEX", raising=False)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("INPUT_MAX_HOURS", "6")
    monkeypatch.setenv("INPUT_CRON_NOW", (gm._CRON_ANCHOR_UTC + timedelta(hours=6)).isoformat())
    assert gm._resolve_batch_index(100, 10) == 1


def test_resolve_batch_index_manual_no_index_no_cron_raises(monkeypatch):
    # Manual dispatch with neither batch_index nor cron_now -> refuse.
    monkeypatch.delenv("INPUT_BATCH_INDEX", raising=False)
    monkeypatch.delenv("INPUT_CRON_NOW", raising=False)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    with pytest.raises(SystemExit):
        gm._resolve_batch_index(100, 10)


def test_resolve_batch_index_zero_batch(monkeypatch):
    monkeypatch.delenv("INPUT_BATCH_INDEX", raising=False)
    assert gm._resolve_batch_index(0, 0) == 0


# ── cron rotation ──


def test_rotation_step_hours_from_env(monkeypatch):
    monkeypatch.setenv("INPUT_MAX_HOURS", "12")
    assert gm._rotation_step_hours() == 12.0


def test_rotation_step_hours_fallback(monkeypatch):
    # unset / invalid / non-positive all fall back to the 6h default
    monkeypatch.delenv("INPUT_MAX_HOURS", raising=False)
    assert gm._rotation_step_hours() == gm._DEFAULT_MAX_HOURS
    monkeypatch.setenv("INPUT_MAX_HOURS", "nan-ish")
    assert gm._rotation_step_hours() == gm._DEFAULT_MAX_HOURS
    monkeypatch.setenv("INPUT_MAX_HOURS", "0")
    assert gm._rotation_step_hours() == gm._DEFAULT_MAX_HOURS


def test_cron_batch_index_anchor_is_zero(monkeypatch):
    monkeypatch.setenv("INPUT_CRON_NOW", gm._CRON_ANCHOR_UTC.isoformat())
    monkeypatch.delenv("INPUT_MAX_HOURS", raising=False)  # 6h step
    assert gm._cron_batch_index(100, 10) == 0


def test_cron_batch_index_advances_by_max_hours(monkeypatch):
    # 6h step: 6h after the anchor -> one batch advanced.
    monkeypatch.setenv("INPUT_MAX_HOURS", "6")
    monkeypatch.setenv("INPUT_CRON_NOW", (gm._CRON_ANCHOR_UTC + timedelta(hours=6)).isoformat())
    assert gm._cron_batch_index(100, 10) == 1
    # within the same window -> still batch 0.
    monkeypatch.setenv("INPUT_CRON_NOW", (gm._CRON_ANCHOR_UTC + timedelta(hours=5, minutes=59)).isoformat())
    assert gm._cron_batch_index(100, 10) == 0


def test_cron_batch_index_step_scales_with_max_hours(monkeypatch):
    # 12h step: 12h -> one batch; 24h -> two batches.
    monkeypatch.setenv("INPUT_MAX_HOURS", "12")
    monkeypatch.setenv("INPUT_CRON_NOW", (gm._CRON_ANCHOR_UTC + timedelta(hours=12)).isoformat())  # +12h
    assert gm._cron_batch_index(100, 10) == 1
    monkeypatch.setenv("INPUT_CRON_NOW", (gm._CRON_ANCHOR_UTC + timedelta(hours=24)).isoformat())  # +24h
    assert gm._cron_batch_index(100, 10) == 2


def test_cron_batch_index_before_anchor_clamped(monkeypatch):
    monkeypatch.setenv("INPUT_CRON_NOW", (gm._CRON_ANCHOR_UTC - timedelta(days=1)).isoformat())
    assert gm._cron_batch_index(100, 10) == 0


def test_cron_batch_index_bad_now(monkeypatch):
    monkeypatch.setenv("INPUT_CRON_NOW", "not-a-date")
    assert isinstance(gm._cron_batch_index(100, 10), int)


# ── _slice_entries ──


def test_slice_entries_no_batch(monkeypatch):
    monkeypatch.delenv("INPUT_BATCH_SIZE", raising=False)
    entries = [{"repo_id": str(i)} for i in range(5)]
    assert gm._slice_entries(entries) == entries


def test_slice_entries_batch(monkeypatch):
    monkeypatch.setenv("INPUT_BATCH_SIZE", "2")
    monkeypatch.setenv("INPUT_BATCH_INDEX", "1")
    entries = [{"repo_id": str(i)} for i in range(5)]
    out = gm._slice_entries(entries)
    assert [e["repo_id"] for e in out] == ["2", "3"]
    assert out[0]["_selected_batch_index"] == 1


def test_slice_entries_manual_tail_wraps(monkeypatch):
    monkeypatch.setenv("INPUT_BATCH_SIZE", "3")
    monkeypatch.setenv("INPUT_BATCH_INDEX", "1")
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    entries = [{"repo_id": str(i)} for i in range(5)]
    out = gm._slice_entries(entries)  # [3,4] + wrap [0]
    assert [e["repo_id"] for e in out] == ["3", "4", "0"]


def test_slice_entries_schedule_no_wrap(monkeypatch):
    monkeypatch.setenv("INPUT_BATCH_SIZE", "3")
    monkeypatch.setenv("INPUT_BATCH_INDEX", "1")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    entries = [{"repo_id": str(i)} for i in range(5)]
    out = gm._slice_entries(entries)  # no wrap
    assert [e["repo_id"] for e in out] == ["3", "4"]


def test_slice_entries_bad_batch_size(monkeypatch):
    monkeypatch.setenv("INPUT_BATCH_SIZE", "abc")
    entries = [{"repo_id": "a"}]
    assert gm._slice_entries(entries) == entries


# ── _slice_entries_with_active_refill ──


def test_active_refill_no_batch(monkeypatch):
    monkeypatch.delenv("INPUT_BATCH_SIZE", raising=False)
    entries = [{"repo_id": "a"}, {"repo_id": "b"}]
    out = gm._slice_entries_with_active_refill(entries, {gm.slugify("a")})
    assert [e["repo_id"] for e in out] == ["b"]


def test_active_refill_skips_active(monkeypatch):
    monkeypatch.setenv("INPUT_BATCH_SIZE", "2")
    monkeypatch.setenv("INPUT_BATCH_INDEX", "0")
    entries = [{"repo_id": str(i)} for i in range(5)]
    out = gm._slice_entries_with_active_refill(entries, {gm.slugify("0")})
    assert [e["repo_id"] for e in out] == ["1", "2"]


def test_active_refill_empty_pool(monkeypatch):
    monkeypatch.setenv("INPUT_BATCH_SIZE", "2")
    assert gm._slice_entries_with_active_refill([], set()) == []


# ── _slice_from_candidates / _all_from_candidates ──


def test_slice_from_candidates(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("INPUT_BATCH_SIZE", raising=False)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"candidates": [{"repo_id": "a"}, {"repo_id": "b"}]}), encoding="utf-8")
    assert gm._slice_from_candidates(p) == ["a", "b"]


def test_all_from_candidates(tmp_path: Path, capsys):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"candidates": [{"repo_id": "a"}]}), encoding="utf-8")
    assert gm._all_from_candidates(p) == ["a"]


# ── _matrix_entry ──


def test_matrix_entry_string():
    assert gm._matrix_entry("Org/M") == {"model": "Org/M", "key": "org-m"}


def test_matrix_entry_dict_with_meta(monkeypatch):
    monkeypatch.setenv("INPUT_BATCH_SIZE", "10")
    entry = {
        "repo_id": "Org/M",
        "framework": "sglang",
        "tp": 8,
        "is_top": False,
        "params_b": 685,
        "_selected_batch_index": 2,
        "_selected_batch_size": 10,
    }
    out = gm._matrix_entry(entry)
    assert out["model"] == "Org/M"
    assert out["framework"] == "sglang"
    assert out["is_top"] is False
    assert out["params_b"] == 685
    assert out["batch_index"] == 2
    assert out["batch_size"] == 10


# ── collect_entries ──


def test_collect_entries_explicit_only(monkeypatch):
    monkeypatch.setenv("INPUT_MODELS", "a b")
    monkeypatch.delenv("INPUT_CANDIDATES_FILE", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    assert gm.collect_entries() == ["a", "b"]


def test_collect_entries_candidates_file(tmp_path: Path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"candidates": [{"repo_id": "a"}, {"repo_id": "b"}]}), encoding="utf-8")
    monkeypatch.delenv("INPUT_MODELS", raising=False)
    monkeypatch.setenv("INPUT_CANDIDATES_FILE", str(p))
    monkeypatch.delenv("INPUT_BATCH_SIZE", raising=False)
    monkeypatch.delenv("INPUT_EXCLUDE_LEADERBOARD", raising=False)
    monkeypatch.delenv("INPUT_EXCLUDE_ACTIVE_WORKFLOWS", raising=False)
    out = gm.collect_entries()
    assert [gm._entry_repo(e) for e in out] == ["a", "b"]


def test_collect_entries_top_only_filters_candidates(tmp_path: Path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "candidates": [
                    {"repo_id": "top", "is_top": True},
                    {"repo_id": "supplement", "is_top": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("INPUT_MODELS", raising=False)
    monkeypatch.setenv("INPUT_CANDIDATES_FILE", str(p))
    monkeypatch.setenv("INPUT_TOP_ONLY", "true")
    monkeypatch.delenv("INPUT_BATCH_SIZE", raising=False)
    monkeypatch.delenv("INPUT_EXCLUDE_LEADERBOARD", raising=False)
    monkeypatch.delenv("INPUT_EXCLUDE_ACTIVE_WORKFLOWS", raising=False)
    out = gm.collect_entries()
    assert [gm._entry_repo(e) for e in out] == ["top"]


def test_collect_entries_explicit_filter_bypasses_top_only(tmp_path: Path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps({"candidates": [{"repo_id": "supplement", "is_top": False}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("INPUT_MODELS", "supplement")
    monkeypatch.setenv("INPUT_CANDIDATES_FILE", str(p))
    monkeypatch.setenv("INPUT_TOP_ONLY", "true")
    monkeypatch.delenv("INPUT_EXCLUDE_LEADERBOARD", raising=False)
    monkeypatch.delenv("INPUT_EXCLUDE_ACTIVE_WORKFLOWS", raising=False)
    out = gm.collect_entries()
    assert [gm._entry_repo(e) for e in out] == ["supplement"]


def test_collect_entries_candidates_file_missing(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("INPUT_MODELS", raising=False)
    monkeypatch.setenv("INPUT_CANDIDATES_FILE", str(tmp_path / "nope.json"))
    assert gm.collect_entries() == []
    assert "not found" in capsys.readouterr().err


def test_collect_entries_explicit_filter_candidates(tmp_path: Path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps({"candidates": [{"repo_id": "Org/A", "framework": "sglang"}, {"repo_id": "Org/B"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("INPUT_MODELS", "org/a")
    monkeypatch.setenv("INPUT_CANDIDATES_FILE", str(p))
    monkeypatch.delenv("INPUT_EXCLUDE_LEADERBOARD", raising=False)
    monkeypatch.delenv("INPUT_EXCLUDE_ACTIVE_WORKFLOWS", raising=False)
    out = gm.collect_entries()
    assert [gm._entry_repo(e) for e in out] == ["Org/A"]


def test_collect_repos(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("INPUT_MODELS", "x y")
    monkeypatch.delenv("INPUT_CANDIDATES_FILE", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    assert gm.collect_repos() == ["x", "y"]


# ── main ──


def test_main_empty(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.delenv("INPUT_MODELS", raising=False)
    monkeypatch.delenv("INPUT_CANDIDATES_FILE", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(gm, "collect_entries", lambda: [])
    assert gm.main() == 0
    assert "empty matrix" in capsys.readouterr().err


def test_main_writes_output(monkeypatch, tmp_path: Path):
    out_file = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
    monkeypatch.setattr(gm, "collect_entries", lambda: [{"repo_id": "Org/M"}])
    assert gm.main() == 0
    content = out_file.read_text(encoding="utf-8")
    assert "matrix=" in content
    assert "count=1" in content


def test_main_stdout(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(gm, "collect_entries", lambda: ["Org/M"])
    assert gm.main() == 0
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(out)["include"][0]["model"] == "Org/M"


# ── network functions (mocked urllib) ──


class _FakeURLMap:
    """Maps a URL substring to a JSON/HTML payload."""

    def __init__(self, routes: dict, html_routes: dict | None = None):
        self.routes = routes
        self.html_routes = html_routes or {}

    def __call__(self, url_or_req, timeout=0):
        url = getattr(url_or_req, "full_url", url_or_req)
        for frag, payload in self.html_routes.items():
            if frag in url:
                return io.BytesIO(payload.encode("utf-8"))
        for frag, payload in self.routes.items():
            if frag in url:
                return io.BytesIO(json.dumps(payload).encode("utf-8"))
        return io.BytesIO(json.dumps({"results": []}).encode("utf-8"))


def test_paginate_models(monkeypatch):
    routes = {
        "/api/v1/leaderboard": {
            "results": [{"model": "Org/A", "task_id": "t1"}, {"model": "Org/B", "tasks": [{"task_id": "t2"}]}],
            "pagination": {"has_more": False},
        }
    }
    monkeypatch.setattr(gm.urllib.request, "urlopen", _FakeURLMap(routes))
    models, tids = gm._paginate_models("/api/v1/leaderboard")
    assert "org/a" in models and "org/b" in models
    assert {"t1", "t2"} <= tids


def test_paginate_models_error(monkeypatch):
    def boom(url, timeout=0):
        raise OSError("net")

    monkeypatch.setattr(gm.urllib.request, "urlopen", boom)
    try:
        gm._paginate_models("/x")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_resolve_task_models(monkeypatch):
    routes = {"/api/v1/tasks/t1": {"model": "Org/X"}}
    monkeypatch.setattr(gm.urllib.request, "urlopen", _FakeURLMap(routes))
    out = gm._resolve_task_models(["t1"], max_workers=2)
    assert out == {"org/x"}


def test_resolve_task_models_empty():
    assert gm._resolve_task_models([]) == set()


def test_dashboard_task_ids(monkeypatch):
    html = '<a href="api/v1/tasks/abc">x</a> api/v1/tasks/def'
    monkeypatch.setattr(gm.urllib.request, "urlopen", _FakeURLMap({}, html_routes={"/dashboard": html}))
    tids = gm._dashboard_task_ids()
    assert {"abc", "def"} <= tids


def test_active_workflow_slugs_no_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert gm._active_workflow_slugs() == set()


def test_active_workflow_slugs(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    routes = {
        "actions/runs": {
            "workflow_runs": [
                {"id": 1, "name": "SaFE Optimize Submit", "jobs_url": "https://api.github.com/jobs1"},
                {"id": 999, "name": "SaFE Optimize Submit", "jobs_url": "https://api.github.com/skip"},
            ]
        },
        "jobs1": {"jobs": [{"name": "optimize-org-m"}, {"name": "other"}]},
    }
    monkeypatch.setattr(gm.urllib.request, "urlopen", _FakeURLMap(routes))
    slugs = gm._active_workflow_slugs()
    assert "org-m" in slugs


def test_leaderboard_models(monkeypatch):
    routes = {
        "/api/v1/leaderboard": {"results": [{"model": "Org/A", "task_id": "t1"}], "pagination": {"has_more": False}},
        "/api/v1/tasks": {"results": [{"model": "Org/B", "task_id": "t2"}], "pagination": {"has_more": False}},
    }
    monkeypatch.setattr(gm.urllib.request, "urlopen", _FakeURLMap(routes, html_routes={"/dashboard": "no tasks here"}))
    models = gm._leaderboard_models()
    assert "org/a" in models and "org/b" in models


def test_apply_exclusions(monkeypatch):
    monkeypatch.delenv("INPUT_EXCLUDE_LEADERBOARD", raising=False)
    monkeypatch.delenv("INPUT_EXCLUDE_ACTIVE_WORKFLOWS", raising=False)
    assert gm._apply_exclusions(["a", "b"]) == ["a", "b"]


def test_apply_exclusions_to_entries_with_leaderboard(monkeypatch):
    monkeypatch.setenv("INPUT_EXCLUDE_LEADERBOARD", "1")
    monkeypatch.delenv("INPUT_EXCLUDE_ACTIVE_WORKFLOWS", raising=False)
    monkeypatch.setattr(gm, "_leaderboard_models", lambda: {"org/a"})
    out = gm._apply_exclusions_to_entries([{"repo_id": "Org/A"}, {"repo_id": "Org/B"}])
    assert [gm._entry_repo(e) for e in out] == ["Org/B"]


def test_apply_exclusions_to_entries_with_active(monkeypatch):
    monkeypatch.delenv("INPUT_EXCLUDE_LEADERBOARD", raising=False)
    monkeypatch.setenv("INPUT_EXCLUDE_ACTIVE_WORKFLOWS", "1")
    monkeypatch.setattr(gm, "_active_workflow_slugs", lambda: {gm.slugify("Org/A")})
    out = gm._apply_exclusions_to_entries([{"repo_id": "Org/A"}, {"repo_id": "Org/B"}])
    assert [gm._entry_repo(e) for e in out] == ["Org/B"]


# ── extra branch coverage for generate_hf_matrix ──


def test_resolve_batch_index_schedule(monkeypatch):
    monkeypatch.delenv("INPUT_BATCH_INDEX", raising=False)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    # anchor + 12h at the 6h default step -> two batches advanced.
    monkeypatch.setenv("INPUT_CRON_NOW", (gm._CRON_ANCHOR_UTC + timedelta(hours=12)).isoformat())
    monkeypatch.delenv("INPUT_MAX_HOURS", raising=False)
    assert gm._resolve_batch_index(100, 10) == 2


def test_cron_batch_index_uses_now_when_unset(monkeypatch):
    monkeypatch.delenv("INPUT_CRON_NOW", raising=False)
    assert isinstance(gm._cron_batch_index(100, 10), int)


def test_matrix_entry_bad_batch_size(monkeypatch):
    monkeypatch.delenv("INPUT_BATCH_SIZE", raising=False)
    entry = {"repo_id": "Org/M", "_selected_batch_size": "bad"}
    out = gm._matrix_entry(entry)
    assert out["model"] == "Org/M"
    assert "batch_size" not in out


def test_collect_entries_relative_path(tmp_path: Path, monkeypatch):
    (tmp_path / "rel.json").write_text(json.dumps({"candidates": [{"repo_id": "a"}]}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INPUT_MODELS", raising=False)
    monkeypatch.setenv("INPUT_CANDIDATES_FILE", "rel.json")
    monkeypatch.delenv("INPUT_BATCH_SIZE", raising=False)
    monkeypatch.delenv("INPUT_EXCLUDE_LEADERBOARD", raising=False)
    monkeypatch.delenv("INPUT_EXCLUDE_ACTIVE_WORKFLOWS", raising=False)
    out = gm.collect_entries()
    assert [gm._entry_repo(e) for e in out] == ["a"]


def test_collect_entries_exclude_leaderboard(tmp_path: Path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"candidates": [{"repo_id": "Org/A"}, {"repo_id": "Org/B"}]}), encoding="utf-8")
    monkeypatch.delenv("INPUT_MODELS", raising=False)
    monkeypatch.setenv("INPUT_CANDIDATES_FILE", str(p))
    monkeypatch.setenv("INPUT_EXCLUDE_LEADERBOARD", "1")
    monkeypatch.delenv("INPUT_EXCLUDE_ACTIVE_WORKFLOWS", raising=False)
    monkeypatch.delenv("INPUT_BATCH_SIZE", raising=False)
    monkeypatch.setattr(gm, "_leaderboard_models", lambda: {"org/a"})
    out = gm.collect_entries()
    assert [gm._entry_repo(e) for e in out] == ["Org/B"]


def test_collect_entries_exclude_active(tmp_path: Path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"candidates": [{"repo_id": "Org/A"}, {"repo_id": "Org/B"}]}), encoding="utf-8")
    monkeypatch.delenv("INPUT_MODELS", raising=False)
    monkeypatch.setenv("INPUT_CANDIDATES_FILE", str(p))
    monkeypatch.delenv("INPUT_EXCLUDE_LEADERBOARD", raising=False)
    monkeypatch.setenv("INPUT_EXCLUDE_ACTIVE_WORKFLOWS", "1")
    monkeypatch.delenv("INPUT_BATCH_SIZE", raising=False)
    monkeypatch.setattr(gm, "_active_workflow_slugs", lambda: {gm.slugify("Org/A")})
    out = gm.collect_entries()
    assert [gm._entry_repo(e) for e in out] == ["Org/B"]


def test_collect_entries_hf_fallback(monkeypatch):
    monkeypatch.delenv("INPUT_MODELS", raising=False)
    monkeypatch.delenv("INPUT_CANDIDATES_FILE", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.delenv("INPUT_EXCLUDE_LEADERBOARD", raising=False)
    monkeypatch.delenv("INPUT_EXCLUDE_ACTIVE_WORKFLOWS", raising=False)

    class FakeHF:
        def __init__(self, *a, **k):
            pass

        def top_models(self, n, min_params_b=7):
            return ["org/x", "org/y"]

    monkeypatch.setattr(gm, "HuggingFaceClient", FakeHF)
    assert gm.collect_entries() == ["org/x", "org/y"]


def test_resolve_task_models_failure(monkeypatch, capsys):
    def boom(url, timeout=0):
        raise OSError("net")

    monkeypatch.setattr(gm.urllib.request, "urlopen", boom)
    assert gm._resolve_task_models(["t1"], max_workers=1) == set()


def test_resolve_task_models_non_dict(monkeypatch):
    monkeypatch.setattr(gm.urllib.request, "urlopen", _FakeURLMap({"/api/v1/tasks/t1": [1, 2, 3]}))
    assert gm._resolve_task_models(["t1"], max_workers=1) == set()


def test_dashboard_task_ids_error(monkeypatch):
    def boom(url, timeout=0):
        raise OSError("net")

    monkeypatch.setattr(gm.urllib.request, "urlopen", boom)
    assert gm._dashboard_task_ids() == set()


def test_leaderboard_models_recovers_hidden(monkeypatch):
    routes = {
        "/api/v1/leaderboard": {"results": [{"model": "Org/A", "task_id": "t1"}], "pagination": {"has_more": False}},
        "/api/v1/tasks/hidden1": {"model": "Org/Hidden"},
        "/api/v1/tasks": {"results": [], "pagination": {"has_more": False}},
    }
    html = {"/dashboard": "api/v1/tasks/hidden1"}
    monkeypatch.setattr(gm.urllib.request, "urlopen", _FakeURLMap(routes, html_routes=html))
    models = gm._leaderboard_models()
    assert "org/hidden" in models


def test_active_workflow_slugs_error_branches(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)

    def boom(url_or_req, timeout=0):
        raise OSError("net down")

    monkeypatch.setattr(gm.urllib.request, "urlopen", boom)
    assert gm._active_workflow_slugs() == set()


def test_active_workflow_slugs_jobs_error(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)

    def selective(url_or_req, timeout=0):
        url = getattr(url_or_req, "full_url", url_or_req)
        if "actions/runs" in url:
            payload = {
                "workflow_runs": [
                    {"id": 1, "name": "SaFE Optimize Submit", "jobs_url": "https://api.github.com/jobs1"},
                    {"id": 2, "name": "Other Workflow", "jobs_url": "https://x/jobs2"},
                    {"id": 3, "name": "SaFE Optimize Submit"},  # missing jobs_url
                ]
            }
            return io.BytesIO(json.dumps(payload).encode())
        raise OSError("jobs listing failed")

    monkeypatch.setattr(gm.urllib.request, "urlopen", selective)
    assert gm._active_workflow_slugs() == set()
