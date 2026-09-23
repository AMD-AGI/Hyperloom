# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Default-asdict equivalence and SharedState persistence contracts."""

from __future__ import annotations

import copy
import json
from collections import namedtuple
from dataclasses import InitVar, asdict, dataclass, field
from enum import IntEnum
from typing import ClassVar

import pytest

from hyperloom.common.dataclass_serde import fast_asdict


@dataclass
class _Record:
    value: object


@dataclass(frozen=True, slots=True)
class _Point:
    label: str
    samples: list[float]


@dataclass
class _DerivedRecord(_Record):
    ignored: InitVar[str] = "init-only"
    category: ClassVar[str] = "class-only"
    extra: list[int] = field(default_factory=list)


@dataclass
class _ListRecord(list):
    value: object


@dataclass
class _DictRecord(dict):
    value: object


class _List(list):
    pass


class _Tuple(tuple):
    pass


class _Dict(dict):
    pass


class _Text(str):
    pass


class _Status(IntEnum):
    READY = 1


_Pair = namedtuple("_Pair", ["left", "right"])


@pytest.mark.parametrize("value", [None, False, True, 10**30, 1.25, "plain", b"bytes", 2 + 3j])
def test_exact_atomic_leaves_match_deepcopy_identity(value):
    assert copy.deepcopy(value) is value
    root = _Record({"direct": value, "list": [value], "tuple": (value,), "nested": _Record(value)})
    result = fast_asdict(root)
    assert result == asdict(root)
    assert result["value"]["direct"] is value
    assert result["value"]["list"][0] is value
    assert result["value"]["tuple"][0] is value
    assert result["value"]["nested"]["value"] is value


def test_nested_dataclasses_and_exact_containers_are_rebuilt():
    point = _Point("sample", [1.25, 2.5])
    root = _DerivedRecord({"point": point, "list": [point], "tuple": (point, {3: [point]})}, extra=[7])
    result = fast_asdict(root)
    assert result == asdict(root)
    assert list(result) == ["value", "extra"]
    assert type(result) is dict
    assert type(result["value"]) is dict
    assert type(result["value"]["list"]) is list
    assert type(result["value"]["tuple"]) is tuple
    assert type(result["value"]["point"]) is dict
    result["value"]["point"]["samples"].append(99.0)
    result["extra"].append(8)
    assert point.samples == [1.25, 2.5]
    assert root.extra == [7]
    point.samples.append(3.75)
    assert result["value"]["list"][0]["samples"] == [1.25, 2.5]


@pytest.mark.parametrize("record_type", [_ListRecord, _DictRecord])
def test_dataclass_dispatch_precedes_container_dispatch(record_type):
    root = record_type(_Point("field", [1.0]))
    assert fast_asdict(root) == asdict(root) == {"value": {"label": "field", "samples": [1.0]}}
    assert fast_asdict(_Record(root)) == asdict(_Record(root))


def test_dataclass_deepcopy_hook_is_not_used():
    @dataclass
    class RecordWithHook:
        value: int

        def __deepcopy__(self, memo):
            raise AssertionError("Dataclass fields must be traversed instead")

    root = _Record(RecordWithHook(3))
    assert fast_asdict(root) == asdict(root) == {"value": {"value": 3}}


@pytest.mark.parametrize("container_type", [_List, _Tuple, _Dict, _Pair])
def test_fallback_containers_keep_type_and_convert_nested_dataclasses(container_type):
    point = _Point("nested", [2.0])
    if container_type is _Dict:
        container = container_type({"point": point, "more": [point]})
    elif container_type is _Pair:
        container = container_type(point, [point])
    else:
        container = container_type([point, [point]])
    result = fast_asdict(_Record(container))["value"]
    expected = asdict(_Record(container))["value"]
    assert result == expected
    assert type(result) is container_type
    converted = result["point"] if container_type is _Dict else result[0]
    assert type(converted) is dict
    converted["samples"].append(3.0)
    assert point.samples == [2.0]


def test_container_deepcopy_hook_does_not_override_stdlib_traversal():
    class DictWithHook(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("Dict subclasses must be traversed instead")

    root = _Record(DictWithHook(point=_Point("nested", [1.0])))
    result = fast_asdict(root)
    assert result == asdict(root)
    assert type(result["value"]) is DictWithHook
    assert type(result["value"]["point"]) is dict


def test_non_string_keys_are_recursively_copied():
    text = _Text("label")
    text.tags = ["original"]
    root = _Record({(2, "tuple"): _Record(4), _Pair(5, "key"): [6], _Status.READY: text, b"raw": 1.5})
    result = fast_asdict(root)
    assert result == asdict(root)
    assert type(next(key for key in result["value"] if isinstance(key, _Pair))) is _Pair
    assert next(key for key in result["value"] if isinstance(key, _Status)) is _Status.READY
    copied_text = result["value"][_Status.READY]
    assert type(copied_text) is _Text
    assert copied_text is not text
    copied_text.tags.append("copy")
    assert text.tags == ["original"]


def test_atomic_subclass_deepcopy_hooks_are_honored_for_keys_and_values():
    calls = []

    class CopyText(str):
        def __deepcopy__(self, memo):
            calls.append(str(self))
            return type(self)(str(self) + "-copied")

    root = _Record({CopyText("key"): [CopyText("value"), _Status.READY]})
    expected = asdict(root)
    expected_calls = calls[:]
    calls.clear()
    result = fast_asdict(root)
    assert result == expected
    assert calls == expected_calls == ["key", "value"]
    key = next(iter(result["value"]))
    assert type(key) is CopyText
    assert type(result["value"][key][0]) is CopyText
    assert result["value"][key][1] is _Status.READY


def test_custom_deepcopy_has_no_shared_memo_between_leaves():
    calls = []

    class CopyValue:
        def __deepcopy__(self, memo):
            calls.append(not memo)
            return {"copied": []}

    value = CopyValue()
    root = _Record([value, value])
    expected = asdict(root)
    expected_calls = calls[:]
    calls.clear()
    result = fast_asdict(root)
    assert result == expected
    assert calls == expected_calls == [True, True]
    assert result["value"][0] is not result["value"][1]
    assert result["value"][0]["copied"] is not result["value"][1]["copied"]


def test_unknown_objects_keep_deepcopy_semantics_inside_the_object():
    class Opaque:
        def __init__(self):
            shared = []
            self.items = [shared, shared]
            self.record = _Record([1])

    source = Opaque()
    root = _Record(source)
    result = fast_asdict(root)["value"]
    expected = asdict(root)["value"]
    assert type(result) is type(expected) is Opaque
    assert result is not source
    assert type(result.record) is type(expected.record) is _Record
    assert result.record == expected.record
    assert result.items[0] is result.items[1]
    assert result.items[0] is not source.items[0]
    result.record.value.append(2)
    assert source.record.value == [1]


def test_shared_mutable_references_are_not_preserved_across_fields():
    values = [1]
    point = _Point("shared", values)
    root = _Record({"left": point, "right": point, "values": values})
    for convert in (asdict, fast_asdict):
        result = convert(root)["value"]
        assert result["left"] is not result["right"]
        assert result["left"]["samples"] is not result["right"]["samples"]
        assert result["left"]["samples"] is not result["values"]
        result["left"]["samples"].append(2)
        assert result["right"]["samples"] == [1]
        assert values == [1]


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_json_bytes_match_for_unicode_and_special_floats(ensure_ascii):
    root = _Record(
        {
            "text": 'café λ \U0001f680\n"quoted"',
            "floats": [float("nan"), float("inf"), float("-inf"), -0.0, 1e-300, 1.25e30],
            "nested": _Point("β", [0.1, -2.5]),
        }
    )
    options = {"indent": 2, "sort_keys": True, "ensure_ascii": ensure_ascii}
    expected = json.dumps(asdict(root), **options).encode("utf-8")
    result = json.dumps(fast_asdict(root), **options).encode("utf-8")
    assert result == expected
    assert b"NaN" in result and b"-Infinity" in result and b"-0.0" in result
    with pytest.raises(ValueError):
        json.dumps(fast_asdict(root), allow_nan=False)
    with pytest.raises(ValueError):
        json.dumps(asdict(root), allow_nan=False)


@pytest.mark.parametrize("value", [b"raw", 1 + 2j])
def test_non_json_atomic_values_keep_json_failure(value):
    for convert in (asdict, fast_asdict):
        with pytest.raises(TypeError):
            json.dumps(convert(_Record(value)))


@pytest.mark.parametrize("root", [None, 1, "text", [], {}, (), _Record, _ListRecord, object()])
def test_root_must_be_a_dataclass_instance(root):
    with pytest.raises(TypeError) as expected:
        asdict(root)
    with pytest.raises(type(expected.value)):
        fast_asdict(root)


@pytest.mark.parametrize("cycle_kind", ["dataclass", "list", "dict", "tuple", "subclass"])
def test_cycles_raise_recursion_error_like_stdlib(cycle_kind):
    root = _Record(None)
    if cycle_kind == "dataclass":
        root.value = root
    elif cycle_kind == "dict":
        root.value = {}
        root.value["cycle"] = root.value
    elif cycle_kind == "tuple":
        loop = []
        root.value = (loop,)
        loop.append(root.value)
    else:
        root.value = _List() if cycle_kind == "subclass" else []
        root.value.append(root.value)
    for convert in (asdict, fast_asdict):
        with pytest.raises(RecursionError):
            convert(root)


def _rich_shared_state():
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(
        session_id="serde-session",
        start_ts="2026-09-22T12:00:00+00:00",
        framework="sglang",
        phase="FRAMEWORK_AGENT",
        tick=7,
        baseline_tput=100.0,
        current_best={"tput": 110.0, "extra_server_args": "--chunked-prefill-size 4096", "extra_envs": {"MODE": "1"}},
        attempts=[{"task_id": "attempt-1", "metrics": {"samples": [1.25, 2.5]}, "kept": True}],
        optimization_stack=[{"action": "explore", "params": {"extra_envs": {"MODE": "1"}}, "gain_pct": 10.0}],
        phase_history=[{"from": "PRELUDE", "to": "FRAMEWORK_AGENT", "ts": "2026-09-22T12:00:00+00:00"}],
        explore_search={"tested": {}, "accepted": [], "rejected": []},
        elapsed_charged_sec=5.0,
        leg_anchor_unix=100.0,
    )
    state.enablement.kept_rounds = [{"patches": ["fix.patch"], "artifacts": [{"path": "out.so"}]}]
    state.enablement.accepted_config = {"extra_envs": {"MODE": "1"}}
    state.enablement.active_runtime = {"task_id": "enable-1", "elapsed_sec": 1.5}
    return state


def test_shared_state_snapshot_matches_stdlib_and_is_isolated():
    state = _rich_shared_state()
    expected = asdict(state)
    snapshot = state.to_dict()
    assert snapshot == expected
    assert json.dumps(snapshot, indent=2, sort_keys=True).encode("utf-8") == json.dumps(
        expected, indent=2, sort_keys=True
    ).encode("utf-8")
    assert "_session_dir" not in snapshot
    assert "PROFILE_WORKLOAD_IDENTITY_KEYS" not in snapshot
    snapshot["attempts"][0]["metrics"]["samples"].append(9.0)
    snapshot["enablement"]["kept_rounds"][0]["patches"].append("other.patch")
    assert asdict(state) == expected
    state.current_best["extra_envs"]["MODE"] = "2"
    assert snapshot["current_best"]["extra_envs"]["MODE"] == "1"


def test_shared_state_from_dict_round_trip_preserves_stable_fields():
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = _rich_shared_state()
    raw = json.loads(json.dumps(state.to_dict()))
    restored = SharedState.from_dict(raw)
    snapshot = restored.to_dict()
    for name in (
        "schema_version",
        "session_id",
        "start_ts",
        "phase",
        "tick",
        "current_best",
        "attempts",
        "optimization_stack",
        "phase_history",
        "enablement",
        "elapsed_charged_sec",
    ):
        assert snapshot[name] == raw[name], name
    assert type(restored.enablement) is type(state.enablement)
    assert restored.leg_anchor_unix == 0.0


def test_shared_state_save_keeps_atomic_write_count_options_and_bytes(monkeypatch, tmp_path):
    from hyperloom.inference_optimizer.breakdown.recorder import instrument
    from hyperloom.orchestrator.state import shared_state as module
    from hyperloom.orchestrator.trace import langfuse_emitter

    monkeypatch.setattr(module.time, "time", lambda: 115.0)
    monkeypatch.setattr(instrument, "snapshot_state_sections", lambda *args: None)
    monkeypatch.setattr(langfuse_emitter, "record_status", lambda *args: None)
    writes = []

    def capture_write(path, payload, **options):
        writes.append((path, json.dumps(payload, **options).encode("utf-8"), options))

    monkeypatch.setattr(module, "atomic_write_json", capture_write)
    original_to_dict = module.SharedState.to_dict
    arms = []
    for convert in (asdict, original_to_dict):
        writes.clear()
        state = _rich_shared_state()
        with monkeypatch.context() as patch:
            patch.setattr(module.SharedState, "to_dict", convert)
            for index in range(3):
                state.tick += 1
                state.save(tmp_path)
                assert len(writes) == index + 1
                assert state.elapsed_charged_sec == 20.0
                assert state.leg_anchor_unix == 115.0
                path, payload, options = writes[-1]
                assert path == tmp_path / "state.json"
                assert options == {"indent": 2, "sort_keys": True}
                assert json.loads(payload)["tick"] == 8 + index
        arms.append(writes[:])
    assert arms[0] == arms[1]
