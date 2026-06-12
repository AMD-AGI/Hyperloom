# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for langfuse_emitter SDK-version-tolerant helpers: ns conversion,
observation start/end shims, OTEL attribute coercion, trace-attr fallback, and
JSON/JSONL loaders."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_optimizer.orchestrator.trace import langfuse_emitter as lfe


# -- _to_ns ----------------------------------------------------------------
def test_to_ns_none_and_non_datetime() -> None:
    assert lfe._to_ns(None) is None
    assert lfe._to_ns("not-a-datetime") is None


def test_to_ns_datetime() -> None:
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ns = lfe._to_ns(dt)
    assert ns == int(dt.timestamp() * 1_000_000_000)


# -- _start_obs ------------------------------------------------------------
class _ParentRejectsStartTime:
    """start_observation that rejects start_time once (v4 behaviour)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def start_observation(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        if "start_time" in kwargs:
            raise TypeError("v4 does not accept start_time")
        return "obs"


def test_start_obs_retries_without_start_time() -> None:
    parent = _ParentRejectsStartTime()
    out = lfe._start_obs(parent, name="x", start_time=datetime.now())
    assert out == "obs"
    # second (successful) call dropped start_time
    assert "start_time" not in parent.calls[-1]


def test_start_obs_passthrough_when_accepted() -> None:
    class _Parent:
        def start_observation(self, **kwargs: Any) -> str:
            return "ok"

    assert lfe._start_obs(_Parent(), name="x") == "ok"


# -- _end_time_wants_int / _end_obs ---------------------------------------
class _ObsIntEnd:
    """v4-style observation: end(end_time: int) signature."""

    def __init__(self) -> None:
        self.end_args: list[Any] = []

    def end(self, end_time: int | None = None) -> None:
        self.end_args.append(end_time)


class _ObsDatetimeEnd:
    """v2/v3-style observation: end(end_time: datetime) signature."""

    def __init__(self) -> None:
        self.end_args: list[Any] = []

    def end(self, end_time: "datetime | None" = None) -> None:
        self.end_args.append(end_time)


def test_end_time_wants_int_detects_v4() -> None:
    assert lfe._end_time_wants_int(_ObsIntEnd()) is True
    assert lfe._end_time_wants_int(_ObsDatetimeEnd()) is False


def test_end_time_wants_int_unreadable_signature_defaults_false() -> None:
    class _Weird:
        end = 123  # not callable -> signature() raises

    assert lfe._end_time_wants_int(_Weird()) is False


def test_end_obs_none_observation_is_noop() -> None:
    lfe._end_obs(None, datetime.now())  # must not raise


def test_end_obs_none_end_dt_calls_bare_end() -> None:
    obs = _ObsIntEnd()
    lfe._end_obs(obs, None)
    assert obs.end_args == [None]


def test_end_obs_int_path_converts_to_ns() -> None:
    obs = _ObsIntEnd()
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    lfe._end_obs(obs, dt)
    assert obs.end_args == [int(dt.timestamp() * 1_000_000_000)]


def test_end_obs_datetime_path_passes_datetime() -> None:
    obs = _ObsDatetimeEnd()
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    lfe._end_obs(obs, dt)
    assert obs.end_args == [dt]


def test_end_obs_typed_call_rejected_falls_back_to_bare() -> None:
    class _Rejects:
        def __init__(self) -> None:
            self.bare = 0

        def end(self, end_time: "datetime | None" = None) -> None:
            if end_time is not None:
                raise ValueError("rejected typed end")
            self.bare += 1

    obs = _Rejects()
    lfe._end_obs(obs, datetime.now())
    assert obs.bare == 1


# -- _otel_attr_value ------------------------------------------------------
def test_otel_attr_value_scalars_and_none() -> None:
    assert lfe._otel_attr_value(None) is None
    assert lfe._otel_attr_value("s") == "s"
    assert lfe._otel_attr_value(True) is True
    assert lfe._otel_attr_value(3) == 3
    assert lfe._otel_attr_value(2.5) == 2.5


def test_otel_attr_value_json_serialises_complex() -> None:
    out = lfe._otel_attr_value({"a": 1, "b": [2, 3]})
    assert out == '{"a": 1, "b": [2, 3]}'


def test_otel_attr_value_falls_back_to_str_on_unserialisable() -> None:
    class _NoJson:
        def __repr__(self) -> str:
            return "no-json-obj"

    # json.dumps default=str handles it -> string form
    out = lfe._otel_attr_value(_NoJson())
    assert "no-json-obj" in out


# -- _set_trace_attrs ------------------------------------------------------
class _SpanV3:
    """v2/v3 span exposing update_trace."""

    def __init__(self) -> None:
        self.updated: dict[str, Any] | None = None

    def update_trace(self, **kwargs: Any) -> None:
        self.updated = kwargs


def test_set_trace_attrs_v3_uses_update_trace() -> None:
    span = _SpanV3()
    lfe._set_trace_attrs(span, name="n", session_id="s", metadata={"k": "v"})
    assert span.updated == {"name": "n", "session_id": "s", "metadata": {"k": "v"}}


class _OtelSpan:
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value


class _SpanV4:
    """v4 span: no update_trace, only an underlying OTEL span."""

    def __init__(self) -> None:
        self._otel_span = _OtelSpan()

    def update_trace(self, **kwargs: Any) -> None:
        raise AttributeError("v4 removed update_trace")


def test_set_trace_attrs_v4_falls_back_to_otel_attributes() -> None:
    span = _SpanV4()
    lfe._set_trace_attrs(
        span, name="n", session_id="s",
        metadata={"good": "v", "skip_none": None, "obj": {"x": 1}},
    )
    attrs = span._otel_span.attrs
    assert attrs["langfuse.trace.name"] == "n"
    assert attrs["session.id"] == "s"
    assert attrs["langfuse.trace.metadata.good"] == "v"
    # None metadata values are dropped (OTEL rejects them)
    assert "langfuse.trace.metadata.skip_none" not in attrs
    # complex values are JSON-stringified
    assert attrs["langfuse.trace.metadata.obj"] == '{"x": 1}'


def test_set_trace_attrs_v4_no_otel_span_is_noop() -> None:
    class _SpanNoOtel:
        def update_trace(self, **kwargs: Any) -> None:
            raise AttributeError("v4")

    lfe._set_trace_attrs(_SpanNoOtel(), name="n")  # must not raise


# -- _load_jsonl / _load_json ---------------------------------------------
def test_load_jsonl_missing_file(tmp_path: Path) -> None:
    assert lfe._load_jsonl(tmp_path / "absent.jsonl") == []


def test_load_jsonl_skips_blank_and_malformed(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text(
        '{"a": 1}\n\n  \nnot-json\n[1,2,3]\n{"b": 2}\n', encoding="utf-8",
    )
    # blank lines skipped, malformed skipped, non-dict ([1,2,3]) skipped
    assert lfe._load_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_load_json_missing_and_malformed(tmp_path: Path) -> None:
    assert lfe._load_json(tmp_path / "absent.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    assert lfe._load_json(bad) == {}
    # valid non-object -> {}
    arr = tmp_path / "arr.json"
    arr.write_text("[1,2,3]", encoding="utf-8")
    assert lfe._load_json(arr) == {}
    ok = tmp_path / "ok.json"
    ok.write_text('{"k": "v"}', encoding="utf-8")
    assert lfe._load_json(ok) == {"k": "v"}
