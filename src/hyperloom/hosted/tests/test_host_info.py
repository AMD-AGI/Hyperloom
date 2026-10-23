# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from hyperloom.common import env_safety
from hyperloom.hosted import host_info
from hyperloom.hosted.host_info import HostGroup, HostInfoError, Var, read_host_info, take_host_info

# Distinct, recognisable values, so a leak anywhere is found by a plain substring search.
_SECRETS = {name: f"sentinel-{index}-{index}{index}{index}" for index, name in enumerate(host_info.SECRET_VARS)}
_ALL_GROUPS = tuple(HostGroup)


@pytest.fixture
def env(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "campaign.json").write_text("{}")
    return {
        Var.INPUT_FILE: str(tmp_path / "campaign.json"),
        Var.RUN_ID: "run-1",
        Var.RUN_CREATED_AT: "2026-10-01T12:30:00Z",
        Var.STATE_DIR: str(tmp_path / "state"),
        Var.MCP_PORT: "8765",
        Var.MCP_PUBLIC_URL: "https://app.example.test:8765",
        Var.SANDBOX_TEMPLATES: json.dumps({"runner": {"1": "r1", "8": "r8"}, "dev": "d"}),
        Var.E2B_API_URL: "https://api.sandbox.example.test",
        Var.E2B_SANDBOX_URL: "https://sbx.example.test",
        Var.RESULT_ARCHIVE: "s3://results-bucket/hl/runs",
        Var.ARCHIVE_ENDPOINT_URL: "https://s3.example.test",
        Var.ARCHIVE_REGION: "us-east-1",
        Var.NODE_NAME: "node-3",
        Var.CLAW_SESSION_ID: "sess-9",
        **_SECRETS,
    }


def test_a_complete_environment_reads_into_typed_fields(env, tmp_path):
    info = read_host_info(env, require=_ALL_GROUPS)

    assert info.input_file == tmp_path / "campaign.json"
    assert info.run_id == "run-1"
    assert info.run_created_at == datetime(2026, 10, 1, 12, 30, tzinfo=timezone.utc)
    assert info.mcp_port == 8765
    assert info.sandbox_templates == {"runner": {"1": "r1", "8": "r8"}, "dev": "d"}
    assert (info.result_archive.bucket, info.result_archive.prefix) == ("results-bucket", "hl/runs")
    assert info.sandbox_keys.subrun.reveal() == _SECRETS[Var.SANDBOX_KEY_SUBRUN]
    assert info.archive_credentials.session_token.reveal() == _SECRETS[Var.AWS_SESSION_TOKEN]
    assert (info.node_name, info.claw_session_id) == ("node-3", "sess-9")


def test_events_dir_defaults_under_the_state_dir_and_an_explicit_one_wins(env, tmp_path):
    assert read_host_info(env).events_dir == tmp_path / "state" / "events"

    env[Var.EVENTS_DIR] = str(tmp_path / "elsewhere")
    assert read_host_info(env).events_dir == tmp_path / "elsewhere"


def test_every_missing_required_input_is_named_in_one_error():
    with pytest.raises(HostInfoError) as raised:
        read_host_info({}, require=_ALL_GROUPS)

    expected = set(host_info._ALWAYS_REQUIRED)
    for group in _ALL_GROUPS:
        expected.update(host_info._GROUP_VARS[group])
    assert set(raised.value.missing) == expected
    assert all(name in str(raised.value) for name in expected)


def test_a_blank_value_counts_as_missing(env):
    env[Var.RUN_ID] = "   "

    with pytest.raises(HostInfoError) as raised:
        read_host_info(env)

    assert raised.value.missing == (Var.RUN_ID,)


def test_the_profile_decides_what_is_required(env):
    for name in (*host_info._GROUP_VARS[HostGroup.SANDBOX], *host_info._GROUP_VARS[HostGroup.ARCHIVE]):
        del env[name]

    read_host_info(env)  # default: run, state, input, MCP

    with pytest.raises(HostInfoError) as raised:
        read_host_info(env, require=(HostGroup.SANDBOX, HostGroup.ARCHIVE))
    assert set(raised.value.missing) == {
        Var.SANDBOX_TEMPLATES,
        Var.E2B_API_URL,
        Var.E2B_SANDBOX_URL,
        Var.RESULT_ARCHIVE,
    } | ({Var.AWS_ACCESS_KEY_ID, Var.AWS_SECRET_ACCESS_KEY} - set(env))


def test_mcp_is_required_by_default_and_droppable(env):
    del env[Var.MCP_PORT], env[Var.MCP_TOKEN]

    with pytest.raises(HostInfoError) as raised:
        read_host_info(env)
    assert set(raised.value.missing) == {Var.MCP_PORT, Var.MCP_TOKEN}

    assert read_host_info(env, require=()).mcp_port is None


def test_a_role_key_is_required_only_for_a_hosted_role(env):
    del env[Var.SANDBOX_KEY_DEV]

    assert read_host_info(env, require=(HostGroup.MEASUREMENT_KEY,)).sandbox_keys.dev is None
    with pytest.raises(HostInfoError) as raised:
        read_host_info(env, require=(HostGroup.DEV_KEY,))
    assert raised.value.missing == (Var.SANDBOX_KEY_DEV,)


def test_missing_and_malformed_inputs_are_reported_together(env):
    del env[Var.RUN_ID]
    env[Var.MCP_PORT] = "http"

    with pytest.raises(HostInfoError) as raised:
        read_host_info(env)

    assert raised.value.missing == (Var.RUN_ID,)
    assert set(raised.value.invalid) == {Var.MCP_PORT}


@pytest.mark.parametrize(
    ("name", "value", "reason"),
    [
        (Var.SANDBOX_TEMPLATES, "{not json", "is not valid JSON"),
        (Var.SANDBOX_TEMPLATES, "[]", "must be a non-empty JSON object"),
        (Var.SANDBOX_TEMPLATES, "{}", "must be a non-empty JSON object"),
        (Var.SANDBOX_TEMPLATES, '{"runner": 8}', "every entry must be a template name"),
        (Var.SANDBOX_TEMPLATES, '{"runner": {"1": ""}}', "every entry must be a template name"),
        (Var.MCP_PORT, "0", "integer from 1 to 65535"),
        (Var.MCP_PORT, "65536", "integer from 1 to 65535"),
        (Var.MCP_PORT, "80.5", "integer from 1 to 65535"),
        (Var.MCP_PORT, "-1", "integer from 1 to 65535"),
        (Var.RESULT_ARCHIVE, "https://b/p", "must be s3://<bucket>/<prefix>"),
        (Var.RESULT_ARCHIVE, "s3://", "bucket is not a valid"),
        (Var.RESULT_ARCHIVE, "s3://UPPER_bucket/p", "bucket is not a valid"),
        (Var.RESULT_ARCHIVE, "s3://results-bucket/a/../b", "prefix must not contain"),
        (Var.RESULT_ARCHIVE, "s3://results-bucket/a//b", "prefix must not contain"),
        (Var.RESULT_ARCHIVE, "s3://results-bucket/p?x=1", "must be s3://<bucket>/<prefix>"),
        (Var.RUN_CREATED_AT, "yesterday", "ISO 8601"),
        (Var.RUN_CREATED_AT, "2026-10-01T12:30:00", "UTC offset"),
        (Var.E2B_API_URL, "ftp://x", "http or https URL"),
        (Var.E2B_API_URL, "https://user:pw@api.example.test", "must not carry credentials"),
        (Var.E2B_API_URL, "https://api.example.test/?token=abc", "must not carry credentials"),
        (Var.E2B_SANDBOX_URL, "https://sbx.example.test/#frag", "must not carry credentials"),
        (Var.MCP_PUBLIC_URL, "https://host:notaport", "not a valid URL"),
        (Var.INPUT_FILE, "relative.json", "absolute path"),
        (Var.INPUT_FILE, "/nonexistent/campaign.json", "not an existing file"),
        (Var.STATE_DIR, "/nonexistent/state", "not an existing directory"),
        (Var.MCP_TOKEN, "two words", "whitespace or control"),
    ],
)
def test_a_malformed_input_is_refused_with_its_reason(env, name, value, reason):
    env[name] = value

    with pytest.raises(HostInfoError) as raised:
        read_host_info(env, require=_ALL_GROUPS)

    assert set(raised.value.invalid) == {name}
    assert reason in raised.value.invalid[name]
    assert name in str(raised.value)


def test_half_a_credential_pair_is_refused_even_when_the_archive_is_not_required(env):
    del env[Var.AWS_SECRET_ACCESS_KEY]

    with pytest.raises(HostInfoError) as raised:
        read_host_info(env)

    assert raised.value.missing == (Var.AWS_SECRET_ACCESS_KEY,)


def test_the_session_token_is_optional(env):
    del env[Var.AWS_SESSION_TOKEN]

    assert read_host_info(env, require=_ALL_GROUPS).archive_credentials.session_token is None


def test_an_unrequired_input_that_is_set_is_still_checked(env):
    env[Var.MCP_PUBLIC_URL] = "not a url"

    with pytest.raises(HostInfoError):
        read_host_info(env)


def test_reading_does_not_modify_the_environment(env):
    before = dict(env)

    read_host_info(env, require=_ALL_GROUPS)

    assert env == before


def test_take_removes_every_host_only_input_and_keeps_the_ones_others_read(env):
    env["USER_DATA_PATH"] = "/data"
    info = take_host_info(env, require=_ALL_GROUPS)

    assert not set(host_info.HOST_ONLY_VARS) & set(env)
    assert {name: env[name] for name in env} == {
        "NODE_NAME": "node-3",
        "CLAW_SESSION_ID": "sess-9",
        "USER_DATA_PATH": "/data",
    }
    assert info.mcp_token.reveal() == _SECRETS[Var.MCP_TOKEN]
    assert info.run_id == "run-1"
    assert set(host_info.SECRET_VARS) <= set(host_info.HOST_ONLY_VARS)


def test_take_leaves_the_environment_alone_when_the_read_fails(env):
    del env[Var.RUN_ID]
    before = dict(env)

    with pytest.raises(HostInfoError):
        take_host_info(env)

    assert env == before


# ---- secrets never print -------------------------------------------------------------------------------------


def _surfaces(info):
    return {
        "repr": repr(info),
        "str": str(info),
        "fstring": f"{info!r} {info} {info.mcp_token} {info.sandbox_keys} {info.archive_credentials}",
        "asdict": repr(dataclasses.asdict(info)),
        "deepcopy": repr(copy.deepcopy(info)),
        "secret_format": format(info.mcp_token, ">40"),
    }


@pytest.mark.parametrize("surface", ["repr", "str", "fstring", "asdict", "deepcopy", "secret_format"])
def test_no_secret_value_appears_in_any_rendering_of_the_host_info(env, surface):
    text = _surfaces(read_host_info(env, require=_ALL_GROUPS))[surface]

    assert not [value for value in _SECRETS.values() if value in text]


def test_no_secret_value_appears_in_an_error_message(env):
    del env[Var.RUN_ID]
    env[Var.RESULT_ARCHIVE] = "bad"
    env[Var.E2B_API_URL] = "https://user:" + _SECRETS[Var.SANDBOX_KEY_DEV] + "@api.example.test"
    env[Var.MCP_PORT] = _SECRETS[Var.MCP_TOKEN]
    env[Var.SANDBOX_TEMPLATES] = json.dumps({_SECRETS[Var.SANDBOX_KEY_SUBRUN]: 8})
    env[Var.MCP_PUBLIC_URL] = "https://app.example.test/?token=" + _SECRETS[Var.AWS_SESSION_TOKEN]

    with pytest.raises(HostInfoError) as raised:
        read_host_info(env, require=_ALL_GROUPS)

    text = f"{raised.value} {raised.value!r} {raised.value.invalid}"
    assert not [value for value in _SECRETS.values() if value in text]


def test_a_secret_cannot_be_pickled(env):
    import pickle

    with pytest.raises(TypeError):
        pickle.dumps(read_host_info(env).mcp_token)


# ---- child processes never inherit them -----------------------------------------------------------------------


@pytest.mark.parametrize("name", host_info.SECRET_VARS)
def test_a_child_environment_built_through_env_safety_lacks_each_credential(name, monkeypatch):
    monkeypatch.setenv(name, "credential")

    assert name not in env_safety.build_benchmark_env()
    assert name not in env_safety.scrub_benchmark_process_env({name: "credential", "PATH": "/bin"})


def test_a_child_started_without_env_sees_none_of_the_removed_inputs_and_still_sees_the_kept_ones(env, monkeypatch):
    for name, value in {**env, "USER_DATA_PATH": "/data"}.items():
        monkeypatch.setenv(name, value)

    take_host_info(os.environ, require=_ALL_GROUPS)
    out = subprocess.run(
        [sys.executable, "-c", "import os, json; print(json.dumps(sorted(os.environ)))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    seen = set(json.loads(out))
    assert not [name for name in host_info.HOST_ONLY_VARS if name in seen]
    assert {"NODE_NAME", "CLAW_SESSION_ID", "USER_DATA_PATH"} <= seen


def test_the_archive_region_is_optional_read_when_set_and_removed_from_the_environment(env):
    unset = {name: value for name, value in env.items() if name != Var.ARCHIVE_REGION}
    assert read_host_info(unset).archive_region is None
    assert Var.ARCHIVE_REGION in host_info.HOST_ONLY_VARS

    assert take_host_info(env).archive_region == "us-east-1"
    assert Var.ARCHIVE_REGION not in env
