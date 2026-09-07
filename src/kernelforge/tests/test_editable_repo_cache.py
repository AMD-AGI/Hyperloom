# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

import kernelforge.loop.editable_repo as editable_repo


@pytest.fixture(autouse=True)
def _fresh_cache():
    editable_repo._editable_roots_cached.cache_clear()
    yield
    editable_repo._editable_roots_cached.cache_clear()


def test_the_scan_runs_once_however_often_it_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpoint probe asks this every second for the length of a campaign."""
    calls = {"n": 0}

    def counted() -> tuple[str, ...]:
        calls["n"] += 1
        return ("/one-root",)

    monkeypatch.setattr(editable_repo, "_scan_editable_roots", counted)

    answers = [editable_repo.editable_roots() for _ in range(50)]

    assert calls["n"] == 1
    assert answers == [["/one-root"]] * 50


def test_a_caller_cannot_disturb_the_cached_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(editable_repo, "_scan_editable_roots", lambda: ("/one-root",))

    editable_repo.editable_roots().append("/not-a-real-root")

    assert editable_repo.editable_roots() == ["/one-root"]


def test_needs_inplace_reads_through_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(editable_repo, "_scan_editable_roots", lambda: ("/sgl-workspace/aiter",))

    assert editable_repo.needs_inplace("/sgl-workspace/aiter") is True
    assert editable_repo.needs_inplace("/sgl-workspace/aiter/aiter/ops") is True
    assert editable_repo.needs_inplace("/elsewhere") is False
    assert editable_repo.needs_inplace("") is False
