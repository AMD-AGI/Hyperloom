# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for scripts/report_upstream_results.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import report_upstream_results
from report_upstream_results import MergedPull, Report, attributed, check, listed, main, merged_pulls

PAGE = Path("docs/upstream-results.md")


def _page(root: Path, body: str) -> None:
    (root / PAGE.parent).mkdir(parents=True, exist_ok=True)
    (root / PAGE).write_text(body, encoding="utf-8")


def _search_reply(*items: dict, total: int | None = None) -> dict:
    return {"total_count": len(items) if total is None else total, "items": list(items)}


def _item(number: int, merged_at: str | None, title: str = "a title", state: str = "closed") -> dict:
    return {"number": number, "title": title, "state": state, "pull_request": {"merged_at": merged_at}}


def test_listed_reads_every_linked_pull(tmp_path: Path) -> None:
    _page(
        tmp_path,
        "| [vllm#58539](https://github.com/vllm-project/vllm/pull/58539) | 2026-09-30 |\n"
        "| [sglang#41071](https://github.com/sgl-project/sglang/pull/41071) | 2026-10-10 |\n"
        "See https://github.com/AMD-AGI/Hyperloom/pull/2095 for the layout.\n",
    )
    assert listed(tmp_path / PAGE) == {
        ("vllm-project/vllm", 58539),
        ("sgl-project/sglang", 41071),
        ("AMD-AGI/Hyperloom", 2095),
    }


def test_listed_names_a_missing_page(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        listed(tmp_path / PAGE)


def test_merged_pulls_keeps_only_merged_newest_first() -> None:
    items = [
        _item(1, "2026-09-16T07:26:56Z", "older"),
        _item(2, None, "still open", state="open"),
        _item(3, "2026-10-14T02:48:06Z", "newer"),
    ]
    assert merged_pulls("vllm-project/vllm", items) == [
        MergedPull("vllm-project/vllm", 3, "2026-10-14", "newer"),
        MergedPull("vllm-project/vllm", 1, "2026-09-16", "older"),
    ]


def test_attributed_reads_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = {
        1: _search_reply(*(_item(n, None) for n in range(100)), total=150),
        2: _search_reply(*(_item(n, None) for n in range(100, 150)), total=150),
    }
    asked: list[int] = []

    def reply(url: str, token: str) -> dict:
        page = int(url.rsplit("&page=", 1)[1])
        asked.append(page)
        return pages[page]

    monkeypatch.setattr(report_upstream_results, "_get_json", reply)
    assert [item["number"] for item in attributed("vllm-project/vllm", "")] == list(range(150))
    assert asked == [1, 2]


def test_attributed_refuses_more_than_the_search_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_upstream_results, "_get_json", lambda url, token: _search_reply(total=1001))
    with pytest.raises(RuntimeError, match="returns at most 1000"):
        attributed("vllm-project/vllm", "")


def test_attributed_rejects_an_unexpected_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_upstream_results, "_get_json", lambda url, token: {"message": "rate limited"})
    with pytest.raises(RuntimeError, match="unexpected reply searching"):
        attributed("vllm-project/vllm", "")


def test_check_reports_a_merged_pull_the_page_omits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _page(tmp_path, "[vllm#58539](https://github.com/vllm-project/vllm/pull/58539)\n")
    replies = {
        "vllm-project/vllm": _search_reply(
            _item(58539, "2026-09-30T07:31:52Z"),
            _item(58405, "2026-09-30T07:26:58Z"),
            _item(58546, None, state="open"),
            _item(50563, None),
        ),
        "sgl-project/sglang": _search_reply(_item(41229, "2026-10-19T00:00:00Z", "spread decode batches")),
    }
    monkeypatch.setattr(
        report_upstream_results,
        "_get_json",
        lambda url, token: replies["sgl-project/sglang" if "sgl-project" in url else "vllm-project/vllm"],
    )
    missing = [
        MergedPull("vllm-project/vllm", 58405, "2026-09-30", "a title"),
        MergedPull("sgl-project/sglang", 41229, "2026-10-19", "spread decode batches"),
    ]
    report = check(tmp_path)
    assert report == Report(missing, attributed=5, merged=3, open=1)
    assert report.closed_unmerged == 1
    assert main([str(tmp_path)]) == 1
    assert "3 of 5 attributed PRs merged, 1 still open, 1 closed unmerged" in capsys.readouterr().out


def test_check_is_quiet_when_the_page_is_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _page(
        tmp_path,
        "https://github.com/vllm-project/vllm/pull/58539\nhttps://github.com/sgl-project/sglang/pull/41229\n",
    )
    monkeypatch.setattr(
        report_upstream_results,
        "_get_json",
        lambda url, token: _search_reply(
            _item(58539 if "vllm-project" in url else 41229, "2026-09-30T07:31:52Z"),
            _item(999, None, "open, so not claimed", state="open"),
        ),
    )
    assert check(tmp_path).missing == []
    assert main([str(tmp_path)]) == 0


def test_get_json_refuses_a_non_https_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_API_URL", "http://127.0.0.1:8080")
    with pytest.raises(RuntimeError, match="the API URL must be https"):
        attributed("vllm-project/vllm", "")
