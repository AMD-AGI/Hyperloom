# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The sections of the instruction sheet that come from the call's facts and the context's defaults."""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace

import pytest

from hyperloom.measure.hosted_spec import SpecRefused, build_spec
from hyperloom.measure.socket import SessionFacts
from hyperloom.measure.tests.bundle_factory import STAGING, base_launch
from hyperloom.measure.tests.hosted_fixtures import make_call, point_yaml, spec_context
from inference_suite.contracts import Redacted
from inference_suite.pins import INFERENCEX_TOOL, PINNED_TOOLS

OTHER_STAGING = replace(STAGING, interpreter_version="3.12.16")


def _tools(tmp_path, **overrides):
    ctx = spec_context(tmp_path)
    call = replace(make_call(tmp_path), **overrides)
    return ctx, build_spec(call, request_id="req-1", ctx=ctx, deadline_unix=time.time() + 60).tools


def test_the_sheet_names_the_staging_and_the_inferencex_commit_its_carried_base_is_measured_with(tmp_path):
    base = replace(base_launch(), client=replace(base_launch().client, commit="e" * 40))
    _, tools = _tools(tmp_path, session_facts=SessionFacts(staging=OTHER_STAGING, base_launch=base))

    assert (tools.staging, tools.inferencex, tools.magpie) == (OTHER_STAGING, "e" * 40, None)


def test_a_frozen_base_names_the_inferencex_commit_the_inference_suite_staging_requires(tmp_path):
    _, tools = _tools(tmp_path)

    assert tools.inferencex == PINNED_TOOLS[INFERENCEX_TOOL]


@pytest.mark.parametrize("facts", [None, SessionFacts()])
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


_HF_TOKEN = "hf_" + "q" * 34


@pytest.mark.parametrize(
    ("unredacted", "named"),
    [
        (lambda base: replace(base, env={**base.env, "HF_TOKEN": _HF_TOKEN}), "env HF_TOKEN"),
        (lambda base: replace(base, env={**base.env, "SGLANG_EXTRA": "Bearer " + "q" * 20}), "env SGLANG_EXTRA"),
        (lambda base: replace(base, args=(*base.args, "--api-key=sk-" + "q" * 20)), "credential in arg "),
        (lambda base: replace(base, client=replace(base.client, env={"OPENAI_API_KEY": "x"})), "client env"),
        (lambda base: replace(base, env={**base.env, "LLM_GATEWAY_KEY": "x"}), "env LLM_GATEWAY_KEY"),
        (lambda base: replace(base, env={**base.env, "DB_PASSWD": "x"}), "env DB_PASSWD"),
        (lambda base: replace(base, args=(*base.args, "--api-key", "qqqqqqqq")), "credential in arg "),
        (lambda base: replace(base, args=(*base.args, "--hf-token=" + _HF_TOKEN)), "credential in arg "),
        (lambda base: replace(base, client=replace(base.client, args=("--api-key=qqqqqqqq",))), "client arg "),
        (lambda base: replace(base, entrypoint=(*base.entrypoint, "--api-key=qqqqqqqq")), "entrypoint word"),
    ],
    ids=[
        "secret-named env",
        "credential-shaped env value",
        "credential-shaped arg",
        "secret-named client env",
        "gateway key env",
        "passwd env",
        "api-key next word",
        "hf-token joined",
        "client api-key",
        "entrypoint api-key",
    ],
)
def test_a_base_carrying_an_unredacted_credential_is_refused_naming_where_never_the_value(tmp_path, unredacted, named):
    with pytest.raises(SpecRefused, match="sheet_invalid: .*unredacted credential") as refused:
        _tools(tmp_path, session_facts=SessionFacts(staging=STAGING, base_launch=unredacted(base_launch())))

    assert named in str(refused.value) and "qqqqqq" not in str(refused.value)


def test_a_base_whose_secret_named_env_is_redacted_is_carried(tmp_path):
    base = replace(base_launch(), env={**base_launch().env, "HF_TOKEN": Redacted(hashlib.sha256(b"t").hexdigest())})

    procedure = _procedure(tmp_path, session_facts=SessionFacts(staging=STAGING, base_launch=base))

    assert procedure.base_launch == base


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


def test_the_tool_commits_a_multibranch_dev_caller_still_passes_are_accepted_and_change_no_sheet(tmp_path):
    call = make_call(tmp_path)
    plain = spec_context(tmp_path)
    legacy = spec_context(tmp_path, magpie_commit="a" * 40, inferencex_commit="b" * 40)

    sheets = [build_spec(call, request_id="req-1", ctx=ctx, deadline_unix=1234.0) for ctx in (plain, legacy)]

    assert sheets[0].to_bytes() == sheets[1].to_bytes()
