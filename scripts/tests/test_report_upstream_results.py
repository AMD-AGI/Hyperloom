# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for scripts/report_upstream_results.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import report_upstream_results
from report_upstream_results import MergedPull, check, listed, main, merged_attributed

PAGE = Path("docs/reference/upstream-results.md")


def _page(root: Path, body: str) -> None:
    (root / PAGE.parent).mkdir(parents=True, exist_ok=True)
    (root / PAGE).write_text(body, encoding="utf-8")


def _search_reply(*items: dict) -> dict:
    return {"items": list(items)}


def _item(number: int, merged_at: str | None, title: str = "a title") -> dict:
    return {"number": number, "title": title, "pull_request": {"merged_at": merged_at}}


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


def test_merged_attributed_keeps_only_merged_newest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        report_upstream_results,
        "_get_json",
        lambda url, token: _search_reply(
            _item(1, "2026-09-16T07:26:56Z", "older"),
            _item(2, None, "still open"),
            _item(3, "2026-10-14T02:48:06Z", "newer"),
        ),
    )
    assert merged_attributed("vllm-project/vllm", "") == [
        MergedPull("vllm-project/vllm", 3, "2026-10-14", "newer"),
        MergedPull("vllm-project/vllm", 1, "2026-09-16", "older"),
    ]


def test_merged_attributed_rejects_an_unexpected_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_upstream_results, "_get_json", lambda url, token: {"message": "rate limited"})
    with pytest.raises(RuntimeError, match="unexpected reply searching"):
        merged_attributed("vllm-project/vllm", "")


def test_check_reports_a_merged_pull_the_page_omits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _page(tmp_path, "[vllm#58539](https://github.com/vllm-project/vllm/pull/58539)\n")
    replies = {
        "vllm-project/vllm": _search_reply(_item(58539, "2026-09-30T07:31:52Z"), _item(58405, "2026-09-30T07:26:58Z")),
        "sgl-project/sglang": _search_reply(_item(41229, "2026-10-19T00:00:00Z", "spread decode batches")),
    }
    monkeypatch.setattr(
        report_upstream_results,
        "_get_json",
        lambda url, token: replies["sgl-project/sglang" if "sgl-project" in url else "vllm-project/vllm"],
    )
    assert check(tmp_path) == [
        MergedPull("vllm-project/vllm", 58405, "2026-09-30", "a title"),
        MergedPull("sgl-project/sglang", 41229, "2026-10-19", "spread decode batches"),
    ]
    assert main([str(tmp_path)]) == 1


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
            _item(999, None, "open, so not claimed"),
        ),
    )
    assert check(tmp_path) == []
    assert main([str(tmp_path)]) == 0


def test_get_json_refuses_a_non_https_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_API_URL", "http://127.0.0.1:8080")
    with pytest.raises(RuntimeError, match="the API URL must be https"):
        merged_attributed("vllm-project/vllm", "")
