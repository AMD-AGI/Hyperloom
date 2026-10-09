# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regressions for bounded SGLang configuration evidence from launch logs."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.common import launch_log_evidence as evidence


@pytest.mark.parametrize("form", ["dictionary", "server_args"])
def test_server_args_records_preserve_typed_identity(tmp_path: Path, form: str) -> None:
    fields = {
        "model_path": "/models/qwen",
        "tokenizer_path": "/models/tokenizer",
        "served_model_name": "qwen",
        "tp_size": 1,
        "dp_size": 1,
        "mem_fraction_static": 0.8,
        "context_length": 6144,
        "chunked_prefill_size": 16384,
        "quantization": None,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "attention_backend": "aiter",
        "prefill_attention_backend": None,
        "decode_attention_backend": None,
        "disable_radix_cache": True,
        "trust_remote_code": False,
    }
    record = (
        repr(fields) if form == "dictionary" else f"ServerArgs({', '.join(f'{k}={v!r}' for k, v in fields.items())})"
    )
    log = tmp_path / "server.log"
    log.write_text(f"[2026-09-17 00:00:00] server_args={record}\n", encoding="utf-8")

    identity = evidence.observed_sglang_server_identity_from_log(str(log))

    assert identity == fields
    assert list(identity) == sorted(fields)
    assert type(identity["tp_size"]) is int
    assert type(identity["mem_fraction_static"]) is float
    assert identity["disable_radix_cache"] is True
    assert identity["trust_remote_code"] is False
    assert identity["quantization"] is None
    assert evidence.launch_argv_from_log(str(log), "sglang") == ""


@pytest.mark.parametrize(
    "record",
    [
        """{
    'model_path': '/models/{qwen}(v1)[test]',
    'served_model_name': "qwen's } ) ] \\\"alias\\\"",
    'tp_size': 8,
    'unknown_config': {'nested': [(1, 2), {'delimiters': '}])'}]},
}""",
        """ServerArgs(
    model_path='/models/{qwen}(v1)[test]',
    served_model_name="qwen's } ) ] \\\"alias\\\"",
    tp_size=8,
    unknown_config={'nested': [(1, 2), {'delimiters': '}])'}]},
)""",
    ],
    ids=["dictionary", "server_args"],
)
def test_multiline_records_ignore_quoted_delimiters(tmp_path: Path, record: str) -> None:
    log = tmp_path / "server.log"
    log.write_text(f"startup\nINFO server_args = {record} trailing log text\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {
        "model_path": "/models/{qwen}(v1)[test]",
        "served_model_name": 'qwen\'s } ) ] "alias"',
        "tp_size": 8,
    }


def test_dictionary_records_exclude_sensitive_and_unknown_fields(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "server_args={'tp_size': 1, 'api_key': 'secret', 'host': 'private-host', "
        "'port': 30000, 'disable_cuda_graph': False, 'max_running_requests': None, "
        "'max_total_tokens': None, 'env': {'ACCESS_TOKEN': 'secret'}, "
        "'future_option': future_runtime_object()}\n",
        encoding="utf-8",
    )

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {"tp_size": 1}


@pytest.mark.parametrize("form", ["dictionary", "server_args"])
def test_executable_values_are_never_evaluated(tmp_path: Path, form: str) -> None:
    sentinel = tmp_path / "executed"
    executable = f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    record = f"{{'model_path': {executable}}}" if form == "dictionary" else f"ServerArgs(model_path={executable})"
    log = tmp_path / "server.log"
    log.write_text(f"server_args={record}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {}
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "record",
    [
        "{'tp_size': 1",
        "{'tp_size': 1]}",
        "{'model_path': 'unterminated}",
        "{'tp_size' 1}",
        "{'tp_size': 1, **other}",
        "{'tp_size': 1, key: 'value'}",
        "{'tp_size': 1, 2: 'value'}",
        "{key: 1 for key in ['tp_size']}",
        "ServerArgs(tp_size=1, **other)",
        "ServerArgs(tp_size=1]",
    ],
    ids=[
        "unclosed",
        "mismatched",
        "unterminated-string",
        "syntax",
        "unpacking",
        "nonliteral-key",
        "nonstring-key",
        "comprehension",
        "legacy-unpacking",
        "legacy-mismatched",
    ],
)
def test_malformed_or_nonliteral_records_are_unavailable(tmp_path: Path, record: str) -> None:
    log = tmp_path / "server.log"
    log.write_text(f"server_args={record}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {}


@pytest.mark.parametrize("form", ["dictionary", "server_args"])
@pytest.mark.parametrize(
    "value",
    [
        "runtime_name",
        "object.attribute",
        "1 + 1",
        "b'bytes'",
        "{1, 2}",
        "[[[[[1]]]]]",
        repr("x" * 4097),
        repr([1] * 65),
        repr({str(index): index for index in range(65)}),
        "{1: 'nonstring nested key'}",
        "[" * 250 + "1" + "]" * 250,
        "+".join(["1"] * 10000),
    ],
    ids=[
        "name",
        "attribute",
        "expression",
        "bytes",
        "set",
        "too-deep",
        "long-string",
        "long-list",
        "large-mapping",
        "nested-nonstring-key",
        "parser-nesting-limit",
        "parser-recursion-limit",
    ],
)
def test_disallowed_or_oversized_values_reject_the_record(tmp_path: Path, form: str, value: str) -> None:
    record = (
        f"{{'tp_size': 1, 'served_model_name': {value}}}"
        if form == "dictionary"
        else f"ServerArgs(tp_size=1, served_model_name={value})"
    )
    log = tmp_path / "server.log"
    log.write_text(f"server_args={record}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {}


@pytest.mark.parametrize("form", ["dictionary", "server_args"])
def test_supported_nested_values_remain_bounded_and_json_safe(tmp_path: Path, form: str) -> None:
    value = "{'z': ('name', None), 'a': [True, {'size': -1}]}"
    record = f"{{'served_model_name': {value}}}" if form == "dictionary" else f"ServerArgs(served_model_name={value})"
    log = tmp_path / "server.log"
    log.write_text(f"server_args={record}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {
        "served_model_name": {"a": [True, {"size": -1}], "z": ["name", None]}
    }


@pytest.mark.parametrize("form", ["dictionary", "server_args"])
def test_top_level_field_count_is_capped(tmp_path: Path, form: str) -> None:
    fields = {"tp_size": 1, **{f"unknown_{index}": None for index in range(2048)}}
    record = (
        repr(fields) if form == "dictionary" else f"ServerArgs({', '.join(f'{k}={v!r}' for k, v in fields.items())})"
    )
    log = tmp_path / "server.log"
    log.write_text(f"server_args={record}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {}


@pytest.mark.parametrize("extra", [0, 1], ids=["at-cap", "past-cap"])
def test_log_character_scan_is_capped(tmp_path: Path, extra: int) -> None:
    record = "server_args={'tp_size': 1}"
    log = tmp_path / "server.log"
    log.write_text(" " * (512 * 1024 - len(record) + extra) + record, encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == ({"tp_size": 1} if extra == 0 else {})


@pytest.mark.parametrize("lines", [2047, 2048], ids=["at-cap", "past-cap"])
def test_log_line_scan_is_capped(tmp_path: Path, lines: int) -> None:
    log = tmp_path / "server.log"
    log.write_text("startup noise\n" * lines + "server_args={'tp_size': 1}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == ({"tp_size": 1} if lines == 2047 else {})


def test_missing_log_is_unavailable(tmp_path: Path) -> None:
    assert evidence.observed_sglang_server_identity_from_log(str(tmp_path / "missing.log")) == {}


def test_the_config_read_keeps_the_settings_identity_discards(tmp_path: Path) -> None:
    """The filter needs the tuned knobs, which the identity allowlist drops.

    Both engines, one record each. A value that is not a literal (an object
    repr, which vLLM prints routinely) costs its own key and not the record.
    """
    cases = {
        "sglang": (
            "[2026-09-17 00:00:00] server_args={record}",
            {"tp_size": 8, "enable_hierarchical_cache": True, "cuda_graph_bs": [1, 2, 4]},
            {"tp_size": 8},
        ),
        "vllm": (
            "INFO 09-17 00:00:00 non-default args: {record}",
            {"tensor_parallel_size": 8, "enable_prefix_caching": True, "moe_backend": "aiter"},
            {"tensor_parallel_size": 8},
        ),
    }
    for framework, (line, record, identity) in cases.items():
        log = tmp_path / f"{framework}.log"
        # ``Config(x=1)`` is spliced in after repr: it is not a literal.
        text = repr(record)[:-1] + ", 'sampling': Config(x=1)}"
        log.write_text(line.format(record=text) + "\n", encoding="utf-8")

        assert evidence.observed_server_config_from_log(str(log), framework) == record, framework
        # The same record through the identity read keeps only its allowlisted key.
        assert evidence.observed_server_identity_from_log(str(log), framework) == identity, framework


def test_flag_spellings_fold_to_the_name_the_engine_reports() -> None:
    """chaojhou 1: parallelism is compared, so ``--dp`` must meet ``dp_size`` and not slip past it.

    One flag, two keys: SGLang reports ``tp_size`` and vLLM ``tensor_parallel_size``,
    so folding to SGLang's name would hide the setting from a vLLM record.
    """
    for flag in ("--dp", "--dp-size", "--data-parallel-size"):
        assert evidence.launch_flag_setting_name(flag, "sglang") == "dp_size", flag
    assert evidence.launch_flag_setting_name("--tensor-parallel-size", "sglang") == "tp_size"
    assert evidence.launch_flag_setting_name("--tensor-parallel-size", "vllm") == "tensor_parallel_size"
