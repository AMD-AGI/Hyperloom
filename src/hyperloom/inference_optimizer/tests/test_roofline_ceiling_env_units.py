# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the small env/server-args helpers in ``roofline_ceiling``."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from hyperloom.inference_optimizer import roofline_ceiling as rc


def test_benchmark_envs():
    assert rc._benchmark_envs({}) == {}
    assert rc._benchmark_envs({"envs": "nope"}) == {}
    assert rc._benchmark_envs({"envs": {"A": 1}}) == {"A": 1}


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, 0),
        ("8", 8),
        (8, 8),
        (0, 0),
        (-3, 0),
        ("bad", 0),
    ],
)
def test_env_int(raw, expected):
    assert rc._env_int({"K": raw} if raw is not None else {}, "K") == expected


def test_server_args_from_envs():
    assert rc._server_args_from_envs({}) == ""
    envs = {
        "EXTRA_SGLANG_ARGS": " --tp 8 ",
        "EXTRA_VLLM_ARGS": "",
        "EXTRA_ATOM_ARGS": "--foo",
        "IGNORED": "--bar",
    }
    out = rc._server_args_from_envs(envs)
    assert out == "--tp 8 --foo"


def test_server_args_env_override_and_payload():
    assert rc._server_args_env_override("not-a-dict") == ""
    # extra_envs present but not a dict → "".
    assert rc._server_args_env_override({"extra_envs": "nope"}) == ""
    assert rc._server_args_env_override({"extra_envs": {"EXTRA_VLLM_ARGS": "--x"}}) == "--x"

    assert rc._server_args_payload("nope") == ""
    assert rc._server_args_payload({"extra_args": " --z "}) == "--z"
    assert (
        rc._server_args_payload(
            {"candidate_extra_server_args": "--first", "extra_server_args": "--second"},
        )
        == "--first"
    )

    # env override wins over payload.
    entry = {"extra_envs": {"EXTRA_ATOM_ARGS": "--env"}, "extra_args": "--payload"}
    assert rc._server_args_from(entry) == "--env"
    assert rc._server_args_from({"extra_args": "--payload"}) == "--payload"


def test_read_baseline_yaml_benchmark(tmp_path: Path):
    # Non-dict / missing materialized_config short-circuits to {}.
    assert rc._read_baseline_yaml_benchmark(SimpleNamespace(last_baseline=None)) == {}
    # last_baseline is a non-dict truthy value → {}.
    assert rc._read_baseline_yaml_benchmark(SimpleNamespace(last_baseline=[1])) == {}
    assert (
        rc._read_baseline_yaml_benchmark(
            SimpleNamespace(last_baseline={"extras": {}}),
        )
        == {}
    )

    cfg = tmp_path / "materialized.yaml"
    cfg.write_text("benchmark:\n  envs:\n    EXTRA_SGLANG_ARGS: '--tp 8'\n", encoding="utf-8")
    state = SimpleNamespace(
        last_baseline={"extras": {"materialized_config": str(cfg)}},
    )
    bench = rc._read_baseline_yaml_benchmark(state)
    assert bench.get("envs", {}).get("EXTRA_SGLANG_ARGS") == "--tp 8"

    # Full chain: baseline yaml -> server args string.
    assert rc._read_baseline_yaml_server_args(state) == "--tp 8"


def _baseline_state(cfg: Path) -> SimpleNamespace:
    return SimpleNamespace(last_baseline={"extras": {"materialized_config": str(cfg)}})


@pytest.mark.parametrize(
    "content",
    [None, b"benchmark: [unclosed", b"\xff\xfe", b"- just\n- a list\n", b"plain scalar"],
    ids=["missing", "yaml_syntax", "bad_utf8", "list_root", "scalar_root"],
)
def test_unusable_baseline_yaml_yields_no_benchmark(tmp_path: Path, content):
    cfg = tmp_path / "materialized.yaml"
    if content is not None:
        cfg.write_bytes(content)

    assert rc._read_baseline_yaml_benchmark(_baseline_state(cfg)) == {}


def test_unexpected_baseline_yaml_failure_surfaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "materialized.yaml"
    cfg.write_text("benchmark: {}\n", encoding="utf-8")

    def _boom(*_a, **_k):
        raise RuntimeError("yaml loader broke")

    monkeypatch.setattr("yaml.safe_load", _boom)

    with pytest.raises(RuntimeError, match="yaml loader broke"):
        rc._read_baseline_yaml_benchmark(_baseline_state(cfg))
