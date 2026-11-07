# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The sections of the instruction sheet that come from the call's facts and the context's defaults."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from hyperloom.measure.hosted_spec import SpecRefused, build_spec
from hyperloom.measure.socket import SessionFacts
from hyperloom.measure.tests.hosted_fixtures import make_call, point_yaml, spec_context

OTHER_COMMIT = "b" * 40


def _tools(tmp_path, **overrides):
    ctx = spec_context(tmp_path)
    call = replace(make_call(tmp_path), **overrides)
    return ctx, build_spec(call, request_id="req-1", ctx=ctx, deadline_unix=time.time() + 60).tools


def test_the_sheet_names_the_magpie_commit_of_the_session_facts_over_the_contexts(tmp_path):
    ctx, tools = _tools(tmp_path, session_facts=SessionFacts(magpie_commit=OTHER_COMMIT))

    assert (tools.magpie, tools.inferencex) == (OTHER_COMMIT, ctx.inferencex_commit)
    assert OTHER_COMMIT != ctx.magpie_commit


def test_the_sheet_names_the_contexts_magpie_commit_when_the_session_facts_name_none(tmp_path):
    ctx, without_facts = _tools(tmp_path)
    _, facts_without_commit = _tools(tmp_path, session_facts=SessionFacts())

    assert without_facts.magpie == facts_without_commit.magpie == ctx.magpie_commit


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
