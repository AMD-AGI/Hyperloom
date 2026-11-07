# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The sections of the instruction sheet that come from the call's facts and the context's defaults."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from hyperloom.measure.hosted_spec import SpecRefused, build_spec
from hyperloom.measure.socket import SessionFacts
from hyperloom.measure.tests.bundle_factory import STAGING, base_launch
from hyperloom.measure.tests.hosted_fixtures import make_call, point_yaml, spec_context

OTHER_STAGING = "d" * 64


def _tools(tmp_path, **overrides):
    ctx = spec_context(tmp_path)
    call = replace(make_call(tmp_path), **overrides)
    return ctx, build_spec(call, request_id="req-1", ctx=ctx, deadline_unix=time.time() + 60).tools


def test_the_sheet_names_the_staging_of_the_session_facts_and_the_contexts_inferencex_commit(tmp_path):
    ctx, tools = _tools(tmp_path, session_facts=SessionFacts(staging=OTHER_STAGING, base_launch=base_launch()))

    assert (tools.staging, tools.inferencex, tools.magpie) == (OTHER_STAGING, ctx.inferencex_commit, None)


@pytest.mark.parametrize("facts", [None, SessionFacts(), SessionFacts(staging="sha256:" + "d" * 64)])
def test_a_call_whose_facts_name_no_staging_is_refused_not_sent_with_a_guess(tmp_path, facts):
    with pytest.raises(SpecRefused, match="tools_unknown"):
        _tools(tmp_path, session_facts=facts and replace(facts, base_launch=base_launch()))


def test_a_call_whose_facts_carry_no_frozen_base_is_refused_naming_the_sheet_version(tmp_path):
    with pytest.raises(SpecRefused, match="base_launch_missing: a hyperloom.measurement-spec/v2 sheet"):
        _tools(tmp_path, session_facts=SessionFacts(staging=STAGING))


def test_the_sheet_carries_the_frozen_base_whole(tmp_path):
    procedure = _procedure(tmp_path)

    assert procedure.base_launch == base_launch() and procedure.benchmark_backend is None


def test_a_base_launch_over_64_kib_is_refused_when_the_sheet_is_written(tmp_path):
    big = replace(base_launch(), warnings=("w" * 70_000,))

    with pytest.raises(SpecRefused, match=r"sheet_invalid: .*base_launch too large \(\d+ bytes > 65536\)"):
        _tools(tmp_path, session_facts=SessionFacts(staging=STAGING, base_launch=big))


def _procedure(tmp_path, **overrides):
    ctx = spec_context(tmp_path)
    call = replace(make_call(tmp_path, timeout_sec=90), **overrides)
    return build_spec(call, request_id="req-1", ctx=ctx, deadline_unix=1234.0).procedure


def test_a_sheet_without_a_silence_timeout_gives_the_test_timeout_for_it(tmp_path):
    procedure = _procedure(tmp_path)

    assert (procedure.timeout_sec, procedure.silence_timeout_sec) == (90.0, 90.0)


def test_a_sheet_carries_the_silence_timeout_and_the_deadline_it_was_given(tmp_path):
    procedure = _procedure(tmp_path, silence_timeout_sec=30)

    assert (procedure.timeout_sec, procedure.silence_timeout_sec, procedure.deadline_unix) == (90.0, 30.0, 1234.0)


def _build(tmp_path, point):
    call = make_call(tmp_path, point=point)
    return build_spec(call, request_id="req-1", ctx=spec_context(tmp_path), deadline_unix=1234.0)


def test_the_sheet_asks_for_as_many_gpus_as_the_first_point_has_tensor_parallel_ranks(tmp_path):
    sheet = _build(tmp_path, point_yaml(tmp_path, tp=4))

    assert sheet.hardware.gpu_count == 4


@pytest.mark.parametrize("text", ["other: {}\n", "benchmark:\n  envs: [TP]\n", "benchmark:\n  envs:\n    TP: x\n"])
def test_a_first_point_the_sheet_cannot_read_its_gpus_from_is_refused(tmp_path, text):
    point = tmp_path / "point.yaml"
    point.write_text(text, encoding="utf-8")

    with pytest.raises(SpecRefused) as refused:
        _build(tmp_path, point)

    assert refused.value.reason == "point_unreadable"
