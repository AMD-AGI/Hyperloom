# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ci/generate_hf_matrix.py."""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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


def test_resolve_batch_index_run_number(monkeypatch):
    monkeypatch.delenv("INPUT_BATCH_INDEX", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "3")
    # pool 100 / batch 10 => 10 batches; run 3 -> (3-1)%10 = 2
    assert gm._resolve_batch_index(100, 10) == 2


def test_resolve_batch_index_zero_batch(monkeypatch):
    monkeypatch.delenv("INPUT_BATCH_INDEX", raising=False)
    assert gm._resolve_batch_index(0, 0) == 0


def test_resolve_batch_index_run_number_invalid(monkeypatch):
    monkeypatch.delenv("INPUT_BATCH_INDEX", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "notint")
    assert gm._resolve_batch_index(100, 10) == 0


# ── cron rotation ──


def test_cron_fire_counter_monotonic():
    a = gm._cron_fire_counter(datetime(2026, 6, 15, 16, 0, tzinfo=timezone.utc))
    b = gm._cron_fire_counter(datetime(2026, 6, 16, 4, 0, tzinfo=timezone.utc))
    assert b > a


def test_cron_fire_counter_before_first_fire():
    # 02:00 UTC is before the 04:00 fire -> counted as prior day's last fire.
    c = gm._cron_fire_counter(datetime(2026, 6, 15, 2, 0, tzinfo=timezone.utc))
    assert isinstance(c, int)


def test_cron_batch_index_anchor_is_zero(monkeypatch):
    monkeypatch.setenv("INPUT_CRON_NOW", "2026-06-15T16:00:00+00:00")
    assert gm._cron_batch_index(100, 10) == 0


def test_cron_batch_index_advances(monkeypatch):
    monkeypatch.setenv("INPUT_CRON_NOW", "2026-06-16T04:00:00Z")
    # one fire after anchor -> batch 1
    assert gm._cron_batch_index(100, 10) == 1


def test_cron_batch_index_before_anchor_clamped(monkeypatch):
    monkeypatch.setenv("INPUT_CRON_NOW", "2026-06-14T16:00:00+00:00")
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
    out = gm._slice_entries(entries)  # start=3, end=6 -> [3,4] + wrap [0]
    assert [e["repo_id"] for e in out] == ["3", "4", "0"]


def test_slice_entries_schedule_no_wrap(monkeypatch):
    monkeypatch.setenv("INPUT_BATCH_SIZE", "3")
    monkeypatch.setenv("INPUT_BATCH_INDEX", "1")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    entries = [{"repo_id": str(i)} for i in range(5)]
    out = gm._slice_entries(entries)  # [3,4] only, no wrap
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
    # start 0, skip "0" -> picks "1","2"
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
    entry = {"repo_id": "Org/M", "framework": "sglang", "tp": 8, "_selected_batch_index": 2, "_selected_batch_size": 10}
    out = gm._matrix_entry(entry)
    assert out["model"] == "Org/M"
    assert out["framework"] == "sglang"
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
    # last stdout line is the matrix json
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(out)["include"][0]["model"] == "Org/M"


# ── network functions (mocked urllib) ──


class _FakeURLMap:
    """Monkeypatch target: maps URL substring -> JSON/HTML payload."""

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
    monkeypatch.setenv("INPUT_CRON_NOW", "2026-06-16T04:00:00Z")
    assert gm._resolve_batch_index(100, 10) == 1


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
