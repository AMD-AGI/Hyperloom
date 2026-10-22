# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Entry-point discovery shared by the plug registry and the kernelforge provider registry."""

from __future__ import annotations

from hyperloom.common import entry_points as ep


class _Entry:
    def __init__(self, name, loaded=None, error=None, dist=None):
        self.name = name
        self._loaded = loaded
        self._error = error
        self.dist = dist

    def load(self):
        if self._error is not None:
            raise self._error
        return self._loaded


class _Dist:
    name = "vendor-pkg"
    version = "1.2"


def _patch(monkeypatch, groups, *, fail=None):
    class _EntryPoints:
        @staticmethod
        def select(*, group):
            if fail is not None:
                raise fail
            return groups.get(group, [])

    monkeypatch.setattr(ep.metadata, "entry_points", _EntryPoints)


def test_returns_an_object_and_calls_a_zero_argument_factory(monkeypatch) -> None:
    _patch(monkeypatch, {"g": [_Entry("direct", 5), _Entry("factory", lambda: 7)]})
    found = {o.name: o.value for o in ep.iter_entry_point_objects("g")}
    assert found == {"direct": 5, "factory": 7}


def test_origin_is_the_declaring_distribution(monkeypatch) -> None:
    _patch(monkeypatch, {"g": [_Entry("a", 1, dist=_Dist()), _Entry("b", 1)]})
    origins = {o.name: o.origin for o in ep.iter_entry_point_objects("g")}
    assert origins == {"a": "vendor-pkg==1.2", "b": ""}


def test_one_failing_entry_point_does_not_hide_the_others(monkeypatch) -> None:
    _patch(monkeypatch, {"g": [_Entry("bad", error=RuntimeError("boom")), _Entry("good", 1)]})
    found = {o.name: o for o in ep.iter_entry_point_objects("g")}
    assert found["bad"].error == "RuntimeError: boom" and found["bad"].value is None
    assert found["good"].error is None and found["good"].value == 1


def test_a_failing_factory_is_recorded_like_a_failing_load(monkeypatch) -> None:
    def factory():
        raise ValueError("factory broke")

    _patch(monkeypatch, {"g": [_Entry("f", factory)]})
    (only,) = ep.iter_entry_point_objects("g")
    assert only.error == "ValueError: factory broke"


def test_unreadable_metadata_yields_one_discovery_record(monkeypatch) -> None:
    _patch(monkeypatch, {}, fail=OSError("metadata unreadable"))
    records = list(ep.iter_entry_point_objects("g", legacy_groups=("old",)))
    assert [(r.name, r.error) for r in records] == [(ep.DISCOVERY_FAILED, "OSError: metadata unreadable")]


def test_legacy_groups_are_scanned_after_the_group_and_do_not_duplicate_a_name(monkeypatch) -> None:
    _patch(monkeypatch, {"new": [_Entry("x", "new-x")], "old": [_Entry("x", "old-x"), _Entry("y", "old-y")]})
    found = [(o.name, o.value, o.group) for o in ep.iter_entry_point_objects("new", legacy_groups=("old",))]
    assert found == [("x", "new-x", "new"), ("y", "old-y", "old")]
